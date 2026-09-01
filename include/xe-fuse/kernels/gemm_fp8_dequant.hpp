#pragma once

// FP8 GEMM epilogue fusion kernels for Intel Xe / BMG-G31.
//
// BMG FP8 implementation note:
//   BMG-G31 has no native FP8 XMX instruction. The mainloop upcasts
//   float_e4m3_t/float_e5m2_t to FP16 before the XMX16 MMA, which
//   accumulates in float. Requires a VNNI layout workaround for 8-bit loads
//   (sycl-tla issue #357).
//
// Scale convention:
//   D[m,n] = epilogue( acc[m,n] * scale_a[m] * scale_b[n] )
//   scale_a[m]: per-token input scale   (M values, ColBroadcast)
//   scale_b[n]: per-channel weight scale (N values, RowBroadcast)
//   For per-tensor quant: pass uniform arrays (all M copies of scale_a, etc.)
//
// Kernels provided:
//   GemmFP8Dequant      — FP8×FP8 → dequant → BF16         (K0_FP8)
//   GemmFP8DequantSwiGLU — FP8×FP8 → dequant → SwiGLU → BF16 (K2_FP8)

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/float8.h"
#include "cutlass/fp8_to_fp16.h"

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>

#include "xe-fuse/visitors/xe_pairwise_compute.hpp"

namespace xe_fuse {

using namespace cute;

// ─────────────────────────────────────────────────────────────────────────────
// GemmFP8Dequant — K0_FP8
// D[m,n] = bf16( acc[m,n] * scale_a[m] * scale_b[n] )
// ─────────────────────────────────────────────────────────────────────────────
template <
  typename ElementA_       = cutlass::float_e4m3_t,
  typename ElementB_       = cutlass::float_e4m3_t,
  typename ElementD_       = cutlass::bfloat16_t,
  typename ElementScale_   = float,
  typename ElementAcc_     = float,
  typename ElementCompute_ = float,
  typename TileShape_      = cute::Shape<cute::_256, cute::_256, cute::_32>
>
struct GemmFP8Dequant {
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

  static constexpr int AlignmentAB = 32;  // FP8 = 1 byte; 32 elements = 32 B
  static constexpr int AlignmentCD = 8;   // BF16 = 2 bytes; 8 elements = 16 B
  static constexpr int PipelineStages = 2;

  // ── MMA atom: FP16 MMA with float accumulator (FP8 upcasted to FP16) ────
  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32F16F16F32_TT>,
                              Layout<TileShape>,
                              Layout<Shape<_8,_4,_1>, Stride<_4,_1,_0>>>::TiledMMA;

  // ── Mainloop: same dispatch as W8A8, FP8 types instead of INT8 ──────────
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelW8A8<PipelineStages>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy, TileShape,
      ElementA, cutlass::gemm::TagToStrideA_t<LayoutA>,
      ElementB, cutlass::gemm::TagToStrideB_t<LayoutB>,
      TiledMma,
      XE_2D_U8x32x32_LD_N, void, void, cute::identity,
      XE_2D_U8x32x32_LD_V, void, void, cute::identity
  >;

  // ── EVT: acc[float] * scale_a[m] * scale_b[n] → bf16 ───────────────────
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

  using EVT = cutlass::epilogue::fusion::XeEVT<MulCompute, InnerDequant, ChannelScaleBroadcast>;

  // ── Epilogue via CollectiveBuilder ───────────────────────────────────────
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

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static typename EVT::Arguments make_evt_args(
      ElementScale const* scale_a_ptr, int M,
      ElementScale const* scale_b_ptr, int N) {

    typename Accum::Arguments accum_args{};

    typename TokenScaleBroadcast::Arguments token_scale_args;
    token_scale_args.ptr_col      = scale_a_ptr;
    token_scale_args.null_default = ElementScale(1);
    token_scale_args.dCol         = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(M)};

    typename MulCompute::Arguments inner_mul_args{};
    typename InnerDequant::Arguments inner_args{accum_args, token_scale_args, inner_mul_args};

    typename ChannelScaleBroadcast::Arguments channel_scale_args;
    channel_scale_args.ptr_row      = scale_b_ptr;
    channel_scale_args.null_default = ElementScale(1);
    channel_scale_args.dRow         = {cute::Int<0>{}, cute::Int<1>{}, static_cast<int64_t>(N)};

    typename MulCompute::Arguments outer_mul_args{};
    return {inner_args, channel_scale_args, outer_mul_args};
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// GemmFP8DequantSwiGLU — K2_FP8
// D[m,n] = SwiGLU( acc[m,n] * scale_a[m] * scale_b[n] )
// For FFN gate+up projections; output is N columns wide — each pair (2i, 2i+1)
// carries the same silu(gate)*up value; caller consumes only N/2 (even columns).
// ─────────────────────────────────────────────────────────────────────────────
template <
  typename ElementA_       = cutlass::float_e4m3_t,
  typename ElementB_       = cutlass::float_e4m3_t,
  typename ElementD_       = cutlass::bfloat16_t,
  typename ElementScale_   = float,
  typename ElementAcc_     = float,
  typename ElementCompute_ = float,
  typename TileShape_      = cute::Shape<cute::_256, cute::_256, cute::_32>
>
struct GemmFP8DequantSwiGLU {
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
  static constexpr int PipelineStages = 2;

  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32F16F16F32_TT>,
                              Layout<TileShape>,
                              Layout<Shape<_8,_4,_1>, Stride<_4,_1,_0>>>::TiledMMA;

  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelW8A8<PipelineStages>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy, TileShape,
      ElementA, cutlass::gemm::TagToStrideA_t<LayoutA>,
      ElementB, cutlass::gemm::TagToStrideB_t<LayoutB>,
      TiledMma,
      XE_2D_U8x32x32_LD_N, void, void, cute::identity,
      XE_2D_U8x32x32_LD_V, void, void, cute::identity
  >;

  // ── EVT: SwiGLU( acc * scale_a[m] * scale_b[n] ) ───────────────────────
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

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static typename EVT::Arguments make_evt_args(
      ElementScale const* scale_a_ptr, int M,
      ElementScale const* scale_b_ptr, int N) {

    typename Accum::Arguments accum_args{};

    typename TokenScaleBroadcast::Arguments token_scale_args;
    token_scale_args.ptr_col      = scale_a_ptr;
    token_scale_args.null_default = ElementScale(1);
    token_scale_args.dCol         = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(M)};

    typename MulCompute::Arguments inner_mul_args{};
    typename InnerDequant::Arguments inner_args{accum_args, token_scale_args, inner_mul_args};

    typename ChannelScaleBroadcast::Arguments channel_scale_args;
    channel_scale_args.ptr_row      = scale_b_ptr;
    channel_scale_args.null_default = ElementScale(1);
    channel_scale_args.dRow         = {cute::Int<0>{}, cute::Int<1>{}, static_cast<int64_t>(N)};

    typename MulCompute::Arguments outer_mul_args{};
    typename DequantTree::Arguments dequant_args{inner_args, channel_scale_args, outer_mul_args};

    typename SwiGLUVisitor::Arguments swiglu_args{};
    return {dequant_args, swiglu_args};
  }
};

}  // namespace xe_fuse
