#pragma once

// K0a: gemm_residual_gamma — D[m,n] = gamma[n] * (acc[m,n] + residual[m,n])
//
// This is the "residual add + weight multiply" part of K0.
// K0 in the full CODA pipeline also produces a second output O = D * W,
// which requires Sm90SplitTreeVisitor. This header covers the single-output
// variant first; the split-tree version will be added in Step 3.
//
// EVT tree:
//   XeEVT<Compute<multiplies>,
//     XeRowBroadcast<gamma>,
//     XeEVT<Compute<plus>,
//       AccFetch,
//       XeAuxLoad<residual>
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

namespace xe_fuse {

template <
  typename ElementA_        = cutlass::bfloat16_t,
  typename ElementB_        = cutlass::bfloat16_t,
  typename ElementD_        = cutlass::bfloat16_t,
  typename ElementResidual_ = cutlass::bfloat16_t,
  typename ElementGamma_    = float,
  typename ElementAcc_      = float,
  typename ElementCompute_  = float,
  typename TileShape_       = cute::Shape<cute::_256, cute::_256, cute::_32>
>
struct GemmResidualGamma {
  using ElementA        = ElementA_;
  using ElementB        = ElementB_;
  using ElementD        = ElementD_;
  using ElementResidual = ElementResidual_;
  using ElementGamma    = ElementGamma_;
  using ElementAcc      = ElementAcc_;
  using ElementCompute  = ElementCompute_;
  using TileShape       = TileShape_;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;

  using StrideC        = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideD        = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideResidual = cute::Stride<int64_t, cute::Int<1>, int64_t>;

  // EVT leaves
  using Accum = cutlass::epilogue::fusion::XeAccFetch;

  using ResidualLoad = cutlass::epilogue::fusion::XeAuxLoad<
      ElementResidual, StrideResidual, void,
      128 / cutlass::sizeof_bits_v<ElementResidual>, true, true
  >;

  using AddCompute = cutlass::epilogue::fusion::XeCompute<
      cutlass::plus, ElementCompute, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest
  >;

  using EVTAdd = cutlass::epilogue::fusion::XeEVT<AddCompute, Accum, ResidualLoad>;

  using GammaBroadcast = cutlass::epilogue::fusion::XeRowBroadcast<
      0, TileShape, ElementGamma, ElementCompute,
      cute::Stride<cute::Int<0>, cute::Int<1>, int64_t>,
      128 / cutlass::sizeof_bits_v<ElementGamma>
  >;

  using MulCompute = cutlass::epilogue::fusion::XeCompute<
      cutlass::multiplies, ElementD, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest
  >;

  // D = gamma[n] * (acc + residual)
  using EVT = cutlass::epilogue::fusion::XeEVT<MulCompute, GammaBroadcast, EVTAdd>;

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
      ElementResidual const* residual_ptr, StrideResidual stride_residual,
      ElementGamma const* gamma_ptr, int N) {
    typename Accum::Arguments accum_args{};

    typename ResidualLoad::Arguments residual_args;
    residual_args.ptr_aux = residual_ptr;
    residual_args.null_default = ElementResidual(0);
    residual_args.dAux = stride_residual;

    typename AddCompute::Arguments add_args{};
    typename EVTAdd::Arguments evt_add_args{accum_args, residual_args, add_args};

    typename GammaBroadcast::Arguments gamma_args;
    gamma_args.ptr_row = gamma_ptr;
    gamma_args.null_default = ElementGamma(1);
    gamma_args.dRow = {cute::Int<0>{}, cute::Int<1>{}, static_cast<int64_t>(N)};

    typename MulCompute::Arguments mul_args{};

    return {gamma_args, evt_add_args, mul_args};
  }
};

}  // namespace xe_fuse
