#pragma once

// K2_W8A8: gemm_dequant_swiglu — D = SwiGLU( dequant(A_i8 @ B_i8) )
//
// INT8×INT8 GEMM with W8A8 dequantization and SwiGLU fused in the epilogue.
// Used for the FFN (gate+up projection) in the quantized transformer forward
// pass for SwiGLU-based models (LLaMA 3, Mistral, Qwen 2.5, etc.).
//
// The GEMM produces M×N_ffn output where N_ffn = 2*I (interleaved gate and
// up projections). After dequantization, adjacent pairs are fed to SwiGLU:
//   output[m, 2k]   = silu(dequant[m, 2k]) * dequant[m, 2k+1]
//   output[m, 2k+1] = silu(dequant[m, 2k]) * dequant[m, 2k+1]  (same value)
//
// The N→N/2 contraction (writing only I output columns) is handled by the
// caller; both even and odd lanes carry the same SwiGLU value here.
//
// EVT tree:
//   XeEVT<XePairwiseCompute<SwiGLUFn>,                           // root: SwiGLU
//     XeEVT<MulCompute,                                           // outer dequant
//       XeEVT<MulCompute,                                         //   inner dequant: acc * scale_token[m]
//         XeAccFetch,
//         XeColBroadcast<0, scale_token>
//       >,
//       XeRowBroadcast<1, scale_channel>                          //   * scale_channel[n]
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
  typename ElementA_       = int8_t,
  typename ElementB_       = int8_t,
  typename ElementD_       = cutlass::bfloat16_t,
  typename ElementScale_   = float,     // type for scale_token and scale_channel
  typename ElementAcc_     = int32_t,
  typename ElementCompute_ = float,
  typename TileShape_      = cute::Shape<cute::_256, cute::_256, cute::_32>
>
struct GemmDequantSwiGLU {
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

  static constexpr int AlignmentAB = 32;
  static constexpr int AlignmentCD = 8;

  // ── Inner dequant: int32_acc * scale_token[m] ───────────────────────────
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

  using InnerDequant = cutlass::epilogue::fusion::XeEVT<MulCompute, Accum, TokenScaleBroadcast>;

  // ── Outer dequant: InnerDequant * scale_channel[n] ──────────────────────
  using ChannelScaleBroadcast = cutlass::epilogue::fusion::XeRowBroadcast<
      0, TileShape, ElementScale, ElementCompute,
      cute::Stride<cute::Int<0>, cute::Int<1>, int64_t>,
      128 / cutlass::sizeof_bits_v<ElementScale>
  >;

  using DequantTree = cutlass::epilogue::fusion::XeEVT<MulCompute, InnerDequant, ChannelScaleBroadcast>;

  // ── Root: SwiGLU( dequant_result ) ──────────────────────────────────────
  using SwiGLUVisitor = XePairwiseCompute<SwiGLUFn>;

  using EVT = cutlass::epilogue::fusion::XeEVT<SwiGLUVisitor, DequantTree>;

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

  static typename EVT::Arguments make_evt_args(
      ElementScale const* scale_token_ptr, int M,
      ElementScale const* scale_channel_ptr, int N) {

    typename Accum::Arguments accum_args{};

    typename TokenScaleBroadcast::Arguments token_scale_args;
    token_scale_args.ptr_col      = scale_token_ptr;
    token_scale_args.null_default = ElementScale(1);
    token_scale_args.dCol         = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(M)};

    typename MulCompute::Arguments inner_mul_args{};
    typename InnerDequant::Arguments inner_args{accum_args, token_scale_args, inner_mul_args};

    typename ChannelScaleBroadcast::Arguments channel_scale_args;
    channel_scale_args.ptr_row      = scale_channel_ptr;
    channel_scale_args.null_default = ElementScale(1);
    channel_scale_args.dRow         = {cute::Int<0>{}, cute::Int<1>{}, static_cast<int64_t>(N)};

    typename MulCompute::Arguments outer_mul_args{};
    typename DequantTree::Arguments dequant_args{inner_args, channel_scale_args, outer_mul_args};

    typename SwiGLUVisitor::Arguments swiglu_args{};

    return {dequant_args, swiglu_args};
  }
};

// GeGLU variant for Gemma-style models
template <
  typename ElementA_       = int8_t,
  typename ElementB_       = int8_t,
  typename ElementD_       = cutlass::bfloat16_t,
  typename ElementScale_   = float,
  typename ElementAcc_     = int32_t,
  typename ElementCompute_ = float,
  typename TileShape_      = cute::Shape<cute::_256, cute::_256, cute::_32>
>
struct GemmDequantGeGLU {
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

  static constexpr int AlignmentAB = 32;
  static constexpr int AlignmentCD = 8;

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

  using InnerDequant = cutlass::epilogue::fusion::XeEVT<MulCompute, Accum, TokenScaleBroadcast>;

  using ChannelScaleBroadcast = cutlass::epilogue::fusion::XeRowBroadcast<
      0, TileShape, ElementScale, ElementCompute,
      cute::Stride<cute::Int<0>, cute::Int<1>, int64_t>,
      128 / cutlass::sizeof_bits_v<ElementScale>
  >;

  using DequantTree = cutlass::epilogue::fusion::XeEVT<MulCompute, InnerDequant, ChannelScaleBroadcast>;

  using GeGLUVisitor = XePairwiseCompute<GeGLUFn>;

  using EVT = cutlass::epilogue::fusion::XeEVT<GeGLUVisitor, DequantTree>;

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

  static typename EVT::Arguments make_evt_args(
      ElementScale const* scale_token_ptr, int M,
      ElementScale const* scale_channel_ptr, int N) {

    typename Accum::Arguments accum_args{};

    typename TokenScaleBroadcast::Arguments token_scale_args;
    token_scale_args.ptr_col      = scale_token_ptr;
    token_scale_args.null_default = ElementScale(1);
    token_scale_args.dCol         = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(M)};

    typename MulCompute::Arguments inner_mul_args{};
    typename InnerDequant::Arguments inner_args{accum_args, token_scale_args, inner_mul_args};

    typename ChannelScaleBroadcast::Arguments channel_scale_args;
    channel_scale_args.ptr_row      = scale_channel_ptr;
    channel_scale_args.null_default = ElementScale(1);
    channel_scale_args.dRow         = {cute::Int<0>{}, cute::Int<1>{}, static_cast<int64_t>(N)};

    typename MulCompute::Arguments outer_mul_args{};
    typename DequantTree::Arguments dequant_args{inner_args, channel_scale_args, outer_mul_args};

    typename GeGLUVisitor::Arguments geglu_args{};

    return {dequant_args, geglu_args};
  }
};

}  // namespace xe_fuse
