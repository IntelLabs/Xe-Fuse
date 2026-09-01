#pragma once

// K0g: gemm_gate_residual_norm -- D[m,n] = gamma[n] * (gate[m] * acc[m,n] + residual[m,n])
//
// Extends K0a (residual add + gamma) with a per-token gate scalar that
// modulates the GEMM accumulator before the residual add.  Used in DiT-style
// transformer blocks (e.g. FLUX.2) where a timestep-dependent gate vector
// controls how much of the GEMM output contributes to the residual stream.
//
// EVT tree:
//   XeEVT<MulCompute,              // outer: x gamma[n]
//     GammaBroadcast,              // gamma[n]: per-column
//     XeEVT<AddCompute,            // inner: gate_acc + residual
//       XeEVT<MulCompute,          //   gate_acc = gate[m] x acc
//         GateBroadcast,           //   gate[m]: per-row ColBroadcast
//         AccFetch>,
//       ResidualLoad>>
//
// gate[m] is a per-token scalar (ColBroadcast over M rows).
// gamma[n] is the RMSNorm weight pre-folded with rstd, same convention as K0a.

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
  typename ElementGate_     = float,
  typename ElementGamma_    = float,
  typename ElementAcc_      = float,
  typename ElementCompute_  = float,
  typename TileShape_       = cute::Shape<cute::_256, cute::_256, cute::_32>
>
struct GemmGateResidualGamma {
  using ElementA        = ElementA_;
  using ElementB        = ElementB_;
  using ElementD        = ElementD_;
  using ElementResidual = ElementResidual_;
  using ElementGate     = ElementGate_;
  using ElementGamma    = ElementGamma_;
  using ElementAcc      = ElementAcc_;
  using ElementCompute  = ElementCompute_;
  using TileShape       = TileShape_;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;

  using StrideC        = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideD        = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideResidual = cute::Stride<int64_t, cute::Int<1>, int64_t>;

  // gate[m] x acc
  using Accum = cutlass::epilogue::fusion::XeAccFetch;

  using GateBroadcast = cutlass::epilogue::fusion::XeColBroadcast<
      0, TileShape, ElementGate, ElementCompute,
      cute::Stride<cute::Int<1>, cute::Int<0>, int64_t>,
      128 / cutlass::sizeof_bits_v<ElementGate>
  >;

  using MulCompute = cutlass::epilogue::fusion::XeCompute<
      cutlass::multiplies, ElementCompute, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest
  >;

  using GatedAcc = cutlass::epilogue::fusion::XeEVT<MulCompute, GateBroadcast, Accum>;

  // gate_acc + residual
  using ResidualLoad = cutlass::epilogue::fusion::XeAuxLoad<
      ElementResidual, StrideResidual, void,
      128 / cutlass::sizeof_bits_v<ElementResidual>, true, true
  >;

  using AddCompute = cutlass::epilogue::fusion::XeCompute<
      cutlass::plus, ElementCompute, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest
  >;

  using GatedResidual = cutlass::epilogue::fusion::XeEVT<AddCompute, GatedAcc, ResidualLoad>;

  // gamma[n] x (gate_acc + residual)
  using GammaBroadcast = cutlass::epilogue::fusion::XeRowBroadcast<
      0, TileShape, ElementGamma, ElementCompute,
      cute::Stride<cute::Int<0>, cute::Int<1>, int64_t>,
      128 / cutlass::sizeof_bits_v<ElementGamma>
  >;

  using EVT = cutlass::epilogue::fusion::XeEVT<MulCompute, GammaBroadcast, GatedResidual>;

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
      ElementGate     const* gate_ptr,     int M,
      ElementResidual const* residual_ptr, StrideResidual stride_residual,
      ElementGamma    const* gamma_ptr,    int N) {

    typename GateBroadcast::Arguments gate_args;
    gate_args.ptr_col      = gate_ptr;
    gate_args.null_default = ElementGate(1);
    gate_args.dCol         = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(M)};

    typename Accum::Arguments accum_args{};
    typename MulCompute::Arguments gate_mul_args{};
    typename GatedAcc::Arguments gated_acc_args{gate_args, accum_args, gate_mul_args};

    typename ResidualLoad::Arguments residual_args;
    residual_args.ptr_aux      = residual_ptr;
    residual_args.null_default = ElementResidual(0);
    residual_args.dAux         = stride_residual;

    typename AddCompute::Arguments add_args{};
    typename GatedResidual::Arguments gated_residual_args{gated_acc_args, residual_args, add_args};

    typename GammaBroadcast::Arguments gamma_args;
    gamma_args.ptr_row      = gamma_ptr;
    gamma_args.null_default = ElementGamma(1);
    gamma_args.dRow         = {cute::Int<0>{}, cute::Int<1>{}, static_cast<int64_t>(N)};

    typename MulCompute::Arguments outer_mul_args{};

    return {gamma_args, gated_residual_args, outer_mul_args};
  }
};

}  // namespace xe_fuse
