#pragma once

// K4: gemm_rmsnorm_rope — D = RoPE((A@B) * R, cos_sin)
//
// Composes K1 (GEMM + RMSNorm scaling) with K3 (RoPE rotation):
//   1. Compute the GEMM accumulator:          acc = A @ B
//   2. Apply per-row RMSNorm scaling:          scaled = acc * R[row]
//   3. Apply RoPE rotation using cos_sin:      D = RoPE(scaled, cos_sin)
//
// R is the precomputed reciprocal standard deviation (per-row broadcast).
// cos_sin is interleaved: even columns = cos, odd columns = sin.
//
// EVT tree:
//   XeEVT<XeRoPEComputeTwoChild,                                // node: RoPE on child 0, using child 1
//     XeEVT<Compute<mul>, AccFetch, ColBroadcast<R>>,            // child 0: RMS-scaled acc
//     XeAuxLoad<cos_sin>                                          // child 1: interleaved cos/sin
//   >

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"

#include <cute/tensor.hpp>

#include "xe-fuse/visitors/xe_rope_compute.hpp"

namespace xe_fuse {

template <
  typename ElementA_       = cutlass::bfloat16_t,
  typename ElementB_       = cutlass::bfloat16_t,
  typename ElementD_       = cutlass::bfloat16_t,
  typename ElementScale_   = float,
  typename ElementCosSin_  = float,
  typename ElementAcc_     = float,
  typename ElementCompute_ = float,
  typename TileShape_      = cute::Shape<cute::_256, cute::_256, cute::_32>
>
struct GemmRmsNormRoPE {
  using ElementA       = ElementA_;
  using ElementB       = ElementB_;
  using ElementD       = ElementD_;
  using ElementScale   = ElementScale_;
  using ElementCosSin  = ElementCosSin_;
  using ElementAcc     = ElementAcc_;
  using ElementCompute = ElementCompute_;
  using TileShape      = TileShape_;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;

  using StrideC      = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideD      = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideCosSin = cute::Stride<int64_t, cute::Int<1>, int64_t>;

  // ---------- Child 0: RMS-scaled accumulator (inner tree) ----------
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

  using RmsScale = cutlass::epilogue::fusion::XeEVT<MulCompute, Accum, ScaleBroadcast>;

  // ---------- Child 1: cos_sin via Block2D load ----------
  using CosSinLoad = cutlass::epilogue::fusion::XeAuxLoad<
      ElementCosSin, StrideCosSin, void,
      128 / cutlass::sizeof_bits_v<ElementCosSin>, true, true
  >;

  // ---------- Root node: RoPE rotation (two-child) ----------
  using RoPENode = XeRoPEComputeTwoChild;

  // D = RoPE(acc * R[row], cos_sin)
  using EVT = cutlass::epilogue::fusion::XeEVT<RoPENode, RmsScale, CosSinLoad>;

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

  // Helper to build EVT arguments
  static typename EVT::Arguments make_evt_args(
      ElementScale const* scale_ptr, int M,
      ElementCosSin const* cos_sin_ptr, StrideCosSin stride_cos_sin) {

    // Child 0: RmsScale tree = XeEVT<MulCompute, Accum, ScaleBroadcast>
    typename Accum::Arguments accum_args{};

    typename ScaleBroadcast::Arguments scale_args;
    scale_args.ptr_col = scale_ptr;
    scale_args.null_default = ElementScale(1);
    scale_args.dCol = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(M)};

    typename MulCompute::Arguments mul_args{};

    typename RmsScale::Arguments rms_scale_args{accum_args, scale_args, mul_args};

    // Child 1: CosSinLoad
    typename CosSinLoad::Arguments cos_sin_args;
    cos_sin_args.ptr_aux = cos_sin_ptr;
    cos_sin_args.null_default = ElementCosSin(0);
    cos_sin_args.dAux = stride_cos_sin;

    // Root node: RoPENode
    typename RoPENode::Arguments rope_args{};

    return {rms_scale_args, cos_sin_args, rope_args};
  }
};

}  // namespace xe_fuse
