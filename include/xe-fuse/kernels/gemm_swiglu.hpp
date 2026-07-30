#pragma once

// K8: gemm_swiglu — D = SwiGLU( acc )
//
// GEMM produces M×N output where N is 2×hidden_dim (gate and up projections
// interleaved). Adjacent pairs (2k, 2k+1) are fed to SwiGLU directly:
//   output = silu(gate) * up = gate / (1 + exp(-gate)) * up
//
// This is kernel_8 from the CODA paper (arxiv.org/abs/2605.19269): same as K2
// (gemm_rmsnorm_swiglu) but without the RMSNorm scaling step.
//
// EVT tree:
//   XeEVT<XePairwiseCompute<SwiGLUFn>,
//     AccFetch
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
  typename ElementAcc_     = float,
  typename ElementCompute_ = float,
  typename TileShape_      = cute::Shape<cute::_256, cute::_256, cute::_32>
>
struct GemmSwiGLU {
  using ElementA       = ElementA_;
  using ElementB       = ElementB_;
  using ElementD       = ElementD_;
  using ElementAcc     = ElementAcc_;
  using ElementCompute = ElementCompute_;
  using TileShape      = TileShape_;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;

  using StrideC = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideD = cute::Stride<int64_t, cute::Int<1>, int64_t>;

  using Accum = cutlass::epilogue::fusion::XeAccFetch;

  using SwiGLUVisitor = XePairwiseCompute<SwiGLUFn>;

  using EVT = cutlass::epilogue::fusion::XeEVT<SwiGLUVisitor, Accum>;

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

  static typename EVT::Arguments make_evt_args() {
    typename Accum::Arguments accum_args{};
    typename SwiGLUVisitor::Arguments swiglu_args{};
    return {accum_args, swiglu_args};
  }
};

}  // namespace xe_fuse
