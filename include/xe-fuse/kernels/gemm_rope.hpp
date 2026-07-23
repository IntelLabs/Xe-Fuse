#pragma once

// K3: gemm_rope — D = RoPE(acc, cos_sin)
//
// Applies RoPE rotation to the GEMM accumulator using interleaved cos_sin.
// cos_sin[m, 2k] = cos(position * freq_k), cos_sin[m, 2k+1] = sin(position * freq_k)
//
// For each pair of adjacent columns (2k, 2k+1):
//   D[m, 2k]   =  acc[m, 2k] * cos + acc[m, 2k+1] * sin
//   D[m, 2k+1] = -acc[m, 2k] * sin + acc[m, 2k+1] * cos
//
// EVT tree:
//   XeEVT<XeRoPECompute,
//     XeAuxLoad<cos_sin>    // child: interleaved cos/sin values
//   >
// The RoPE visitor uses frg_acc (GEMM output) directly.

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
  typename ElementCosSin_  = float,
  typename ElementAcc_     = float,
  typename ElementCompute_ = float,
  typename TileShape_      = cute::Shape<cute::_256, cute::_256, cute::_32>
>
struct GemmRoPE {
  using ElementA       = ElementA_;
  using ElementB       = ElementB_;
  using ElementD       = ElementD_;
  using ElementCosSin  = ElementCosSin_;
  using ElementAcc     = ElementAcc_;
  using ElementCompute = ElementCompute_;
  using TileShape      = TileShape_;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;

  using StrideC      = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideD      = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideCosSin = cute::Stride<int64_t, cute::Int<1>, int64_t>;

  // Child: cos_sin via Block2D load
  using CosSinLoad = cutlass::epilogue::fusion::XeAuxLoad<
      ElementCosSin, StrideCosSin, void,
      128 / cutlass::sizeof_bits_v<ElementCosSin>, true, true
  >;

  // Node: RoPE rotation (single-child, uses frg_acc directly)
  using RoPENode = XeRoPECompute;

  // D = RoPE(acc, cos_sin)
  using EVT = cutlass::epilogue::fusion::XeEVT<RoPENode, CosSinLoad>;

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

  static typename EVT::Arguments make_evt_args(
      ElementCosSin const* cos_sin_ptr, StrideCosSin stride_cos_sin) {
    typename CosSinLoad::Arguments cos_sin_args;
    cos_sin_args.ptr_aux = cos_sin_ptr;
    cos_sin_args.null_default = ElementCosSin(0);
    cos_sin_args.dAux = stride_cos_sin;

    typename RoPENode::Arguments rope_args{};

    return {cos_sin_args, rope_args};
  }
};

}  // namespace xe_fuse
