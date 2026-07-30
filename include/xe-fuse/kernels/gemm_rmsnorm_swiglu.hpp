#pragma once

// K2: gemm_rmsnorm_swiglu — D = SwiGLU( acc * R )
//
// The GEMM produces M×N output where N is 2×hidden_dim (gate and up projections
// interleaved). After RMS scaling, adjacent pairs (2k, 2k+1) are fed to SwiGLU:
//   output = silu(gate) * up = gate / (1 + exp(-gate)) * up
//
// For Phase 1 (correctness), we store the full N-width output (both even/odd
// lanes get the same SwiGLU value). Phase 2 will optimize the N→N/2 contraction.
//
// EVT tree:
//   XeEVT<XePairwiseCompute<SwiGLUFn>,
//     XeEVT<XeCompute<multiplies>,   // RMS scaling
//       XeAccFetch,
//       XeColBroadcast<R>
//     >
//   >

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"

#include <cute/tensor.hpp>

#include "xe-fuse/visitors/xe_pairwise_compute.hpp"

namespace xe_fuse {

template <
  typename ElementA_       = cutlass::bfloat16_t,
  typename ElementB_       = cutlass::bfloat16_t,
  typename ElementD_       = cutlass::bfloat16_t,
  typename ElementScale_   = float,
  typename ElementAcc_     = float,
  typename ElementCompute_ = float,
  typename TileShape_      = cute::Shape<cute::_256, cute::_256, cute::_32>
>
struct GemmRmsNormSwiGLU {
  using ElementA       = ElementA_;
  using ElementB       = ElementB_;
  using ElementD       = ElementD_;
  using ElementScale   = ElementScale_;
  using ElementAcc     = ElementAcc_;
  using ElementCompute = ElementCompute_;
  using TileShape      = TileShape_;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;

  using StrideC = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideD = cute::Stride<int64_t, cute::Int<1>, int64_t>;

  // Inner tree: RMS scaling (same as K1)
  using Accum = cutlass::epilogue::fusion::XeAccFetch;

  using ScaleBroadcast = cutlass::epilogue::fusion::XeColBroadcast<
      0, TileShape, ElementScale, ElementCompute,
      cute::Stride<cute::Int<1>, cute::Int<0>, int64_t>,
      128 / cutlass::sizeof_bits_v<ElementScale>
  >;

  using MulCompute = cutlass::epilogue::fusion::XeCompute<
      cutlass::multiplies, ElementCompute, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest
  >;

  using EVTRmsScale = cutlass::epilogue::fusion::XeEVT<MulCompute, Accum, ScaleBroadcast>;

  // Outer: pairwise SwiGLU on the RMS-scaled result
  using SwiGLUVisitor = XePairwiseCompute<SwiGLUFn>;

  // Full tree: SwiGLU( acc * R[row] )
  using EVT = cutlass::epilogue::fusion::XeEVT<SwiGLUVisitor, EVTRmsScale>;

  using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Xe20, cutlass::arch::OpClassTensorOp,
      TileShape,
      cute::Shape<cute::_1, cute::_1, cute::_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      ElementD, StrideC, 8,
      ElementD, StrideD, 8,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      EVT
    >::CollectiveOp;

  using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Xe20, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAcc,
      TileShape,
      cute::Shape<cute::_1, cute::_1, cute::_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static typename EVT::Arguments make_evt_args(ElementScale const* scale_ptr, int M) {
    // Inner tree: RMS scaling args
    typename Accum::Arguments accum_args{};

    typename ScaleBroadcast::Arguments scale_args;
    scale_args.ptr_col = scale_ptr;
    scale_args.null_default = ElementScale(1);
    scale_args.dCol = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(M)};

    typename MulCompute::Arguments mul_args{};
    typename EVTRmsScale::Arguments rms_args{accum_args, scale_args, mul_args};

    // Outer: SwiGLU visitor args
    typename SwiGLUVisitor::Arguments swiglu_args{};

    return {rms_args, swiglu_args};
  }
};

}  // namespace xe_fuse
