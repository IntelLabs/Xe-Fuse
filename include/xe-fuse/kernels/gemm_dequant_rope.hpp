#pragma once

// K4_W8A8: gemm_dequant_rope — D = RoPE( dequant(A_i8 @ B_i8) )
//
// INT8×INT8 GEMM with W8A8 dequantization and RoPE fused in the epilogue.
// Used for Q and K projections in the quantized transformer forward pass.
//
// scale_token[m] absorbs both the per-token RMSNorm reciprocal std and the
// per-token quantization range, so no separate RMSNorm multiply is needed:
//   scale_token[m] = quant_scale[m] * rstd[m]
//
// EVT tree:
//   XeEVT<XeRoPEComputeTwoChild,                              // root: RoPE on child 0 using child 1
//     XeEVT<MulCompute,                                        // child 0: outer dequant
//       XeEVT<MulCompute,                                      //   inner dequant: acc * scale_token[m]
//         XeAccFetch,
//         XeColBroadcast<0, scale_token>
//       >,
//       XeRowBroadcast<1, scale_channel>                       //   * scale_channel[n]
//     >,
//     XeAuxLoad<cos_sin>                                       // child 1: interleaved cos/sin
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
  typename ElementA_       = int8_t,
  typename ElementB_       = int8_t,
  typename ElementD_       = cutlass::bfloat16_t,
  typename ElementScale_   = float,     // type for scale_token and scale_channel
  typename ElementCosSin_  = float,
  typename ElementAcc_     = int32_t,
  typename ElementCompute_ = float,
  typename TileShape_      = cute::Shape<cute::_256, cute::_256, cute::_32>
>
struct GemmDequantRoPE {
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

  // INT8 requires 256-bit aligned loads: 32 elements of int8_t = 256 bits
  static constexpr int AlignmentAB = 32;
  static constexpr int AlignmentCD = 8;

  // ── Inner dequant tree: int32_acc * scale_token[m] ──────────────────────
  using Accum = cutlass::epilogue::fusion::XeAccFetch;

  using TokenScaleBroadcast = cutlass::epilogue::fusion::XeColBroadcast<
      0, TileShape, ElementScale, ElementCompute,
      cute::Stride<cute::Int<1>, cute::Int<0>, int64_t>,
      128 / cutlass::sizeof_bits_v<ElementScale>
  >;

  using MulCompute = cutlass::epilogue::fusion::XeCompute<
      cutlass::multiplies, ElementCompute, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest
  >;

  // int32_acc * scale_token[m]
  using InnerDequant = cutlass::epilogue::fusion::XeEVT<MulCompute, Accum, TokenScaleBroadcast>;

  // ── Outer dequant: InnerDequant * scale_channel[n] ──────────────────────
  using ChannelScaleBroadcast = cutlass::epilogue::fusion::XeRowBroadcast<
      0, TileShape, ElementScale, ElementCompute,
      cute::Stride<cute::Int<0>, cute::Int<1>, int64_t>,
      128 / cutlass::sizeof_bits_v<ElementScale>
  >;

  // (acc * scale_token[m]) * scale_channel[n]
  using DequantTree = cutlass::epilogue::fusion::XeEVT<MulCompute, InnerDequant, ChannelScaleBroadcast>;

  // ── cos/sin table load ───────────────────────────────────────────────────
  using CosSinLoad = cutlass::epilogue::fusion::XeAuxLoad<
      ElementCosSin, StrideCosSin, void,
      128 / cutlass::sizeof_bits_v<ElementCosSin>, true, true
  >;

  // ── Root: RoPE( dequant_result, cos_sin ) ───────────────────────────────
  using RoPENode = XeRoPEComputeTwoChild;

  using EVT = cutlass::epilogue::fusion::XeEVT<RoPENode, DequantTree, CosSinLoad>;

  using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Xe20, cutlass::arch::OpClassTensorOp,
      TileShape,
      cute::Shape<cute::_1, cute::_1, cute::_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      ElementD, StrideC, AlignmentCD,
      ElementD, StrideD, AlignmentCD,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      EVT
    >::CollectiveOp;

  using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Xe20, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentAB,
      ElementB, LayoutB, AlignmentAB,
      ElementAcc,
      TileShape,
      cute::Shape<cute::_1, cute::_1, cute::_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // Build EVT arguments.
  // scale_token[m]: per-token scale absorbing both quant range and RMSNorm rstd
  // scale_channel[n]: per-channel weight quantization scale
  static typename EVT::Arguments make_evt_args(
      ElementScale const* scale_token_ptr, int M,
      ElementScale const* scale_channel_ptr, int N,
      ElementCosSin const* cos_sin_ptr,
      StrideCosSin stride_cos_sin) {

    // Inner dequant: XeEVT<MulCompute, Accum, TokenScaleBroadcast>
    typename Accum::Arguments accum_args{};

    typename TokenScaleBroadcast::Arguments token_scale_args;
    token_scale_args.ptr_col      = scale_token_ptr;
    token_scale_args.null_default = ElementScale(1);
    token_scale_args.dCol         = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(M)};

    typename MulCompute::Arguments inner_mul_args{};
    typename InnerDequant::Arguments inner_args{accum_args, token_scale_args, inner_mul_args};

    // Outer dequant: XeEVT<MulCompute, InnerDequant, ChannelScaleBroadcast>
    typename ChannelScaleBroadcast::Arguments channel_scale_args;
    channel_scale_args.ptr_row      = scale_channel_ptr;
    channel_scale_args.null_default = ElementScale(1);
    channel_scale_args.dRow         = {cute::Int<0>{}, cute::Int<1>{}, static_cast<int64_t>(N)};

    typename MulCompute::Arguments outer_mul_args{};
    typename DequantTree::Arguments dequant_args{inner_args, channel_scale_args, outer_mul_args};

    // cos/sin AuxLoad
    typename CosSinLoad::Arguments cos_sin_args;
    cos_sin_args.ptr_aux      = cos_sin_ptr;
    cos_sin_args.null_default = ElementCosSin(0);
    cos_sin_args.dAux         = stride_cos_sin;

    // Root RoPE node
    typename RoPENode::Arguments rope_args{};

    return {dequant_args, cos_sin_args, rope_args};
  }
};

}  // namespace xe_fuse
