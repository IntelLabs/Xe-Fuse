#pragma once

// K1_W8A8 / K0_W8A8: plain INT8 GEMM + W8A8 dequantization to BF16.
//
// EVT tree:
//   XeEVT<MulCompute,                           // outer: * scale_channel[n]
//     XeEVT<MulCompute,                          // inner: acc * scale_token[m]
//       XeAccFetch,
//       XeColBroadcast<0, scale_token>
//     >,
//     XeRowBroadcast<1, scale_channel>
//   >
//
// Used for:
//   K1_W8A8 — V projection:  x_i8 @ W_v_i8 → dequant → BF16
//   K0_W8A8 — O projection:  attn_i8 @ W_o_i8 → dequant → BF16
//             (residual add + gamma applied via standalone ops afterwards)

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"

#include <cute/tensor.hpp>

namespace xe_fuse {

template <
  typename ElementA_       = int8_t,
  typename ElementB_       = int8_t,
  typename ElementD_       = cutlass::bfloat16_t,
  typename ElementScale_   = float,
  typename ElementAcc_     = int32_t,
  typename ElementCompute_ = float,
  typename TileShape_      = cute::Shape<cute::_256, cute::_256, cute::_32>
>
struct GemmDequantW8A8 {
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

  using EVT = cutlass::epilogue::fusion::XeEVT<MulCompute, InnerDequant, ChannelScaleBroadcast>;

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

    return {inner_args, channel_scale_args, outer_mul_args};
  }
};

}  // namespace xe_fuse
