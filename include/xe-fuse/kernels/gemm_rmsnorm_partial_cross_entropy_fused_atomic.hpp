#pragma once

// K5_fused: gemm_rmsnorm_partial_cross_entropy — fully-fused epilogue variant
//
// Same semantics as K5 (gemm_rmsnorm_partial_cross_entropy) but replaces post-GEMM
// re-read of M×N logits with in-epilogue atomic logsumexp:
//
//   D[m,n] = rstd[m] * acc[m,n]   (RMSNorm scaling, same as K1/K5)
//
//   Per GEMM tile (tile_m, tile_n):
//     lse[m] = atomic_logsumexp(lse[m], logsumexp_{n in tile} D[m,n])
//
// Uses XeColReduction<IsAtomic=true> so all M rows are correctly written
// via the filter_zeros path (vs FinalReduction=false which had a copy_aligned bug).

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/epilogue/fusion/xe_visitor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"

#include <cute/tensor.hpp>
#include <limits>

#include "xe-fuse/visitors/xe_reduce_fns.hpp"
#include "xe-fuse/standalone/ops.hpp"

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
struct GemmRmsNormPartialCrossEntropyFused {
  using ElementA       = ElementA_;
  using ElementB       = ElementB_;
  using ElementD       = ElementD_;
  using ElementScale   = ElementScale_;
  using ElementAcc     = ElementAcc_;
  using ElementCompute = ElementCompute_;
  using TileShape      = TileShape_;

  static constexpr int CTA_M = cute::size<0>(TileShape{});
  static constexpr int CTA_N = cute::size<1>(TileShape{});

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;

  using StrideC = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideD = cute::Stride<int64_t, cute::Int<1>, int64_t>;

  // ── InputTree: D[m,n] = rstd[m] * acc[m,n]  (identical to K1 / K5) ─────
  using Accum = cutlass::epilogue::fusion::XeAccFetch;

  using ScaleBroadcast = cutlass::epilogue::fusion::XeColBroadcast<
      0, TileShape, ElementScale, ElementCompute,
      cute::Stride<cute::Int<1>, cute::Int<0>, int64_t>,
      128 / cutlass::sizeof_bits_v<ElementScale>
  >;
  using MulCompute = cutlass::epilogue::fusion::XeCompute<
      cutlass::multiplies, ElementD, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest
  >;
  using InputTree = cutlass::epilogue::fusion::XeEVT<MulCompute, Accum, ScaleBroadcast>;

  // ── OutputTree: write scaled logits D ────────────────────────────────────
  using IdentityOp = cutlass::epilogue::fusion::XeCompute<
      cutlass::epilogue::thread::Identity,
      ElementD, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using OutputTree = cutlass::epilogue::fusion::XeEVT<IdentityOp, Accum>;

  // ── LseTree: atomic logsumexp into lse[M*L] ──────────────────────────────
  using LseColRed = cutlass::epilogue::fusion::XeColReduction<
      reduce_logsumexp,          // RegReduceFn
      reduce_logsumexp,          // ShuffleReduceFn
      atomic_reduce_logsumexp,   // GmemReduceFn (atomic → IsAtomic=true)
      0,
      TileShape,
      float,             // ElementOutput = float
      ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<cute::Int<1>, cute::Int<0>, int64_t>,
      128 / cutlass::sizeof_bits_v<float>,
      true,   // EnableNullptr
      true    // FinalReduction=true (required for IsAtomic)
  >;
  using LseTree = cutlass::epilogue::fusion::XeEVT<LseColRed, Accum>;

  // ── Combined SplitTree ────────────────────────────────────────────────────
  using EVT = cutlass::epilogue::fusion::Sm90SplitTreeVisitor<
      InputTree, OutputTree, LseTree>;

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
      ElementScale const* rstd_ptr, int M,
      float* lse, int /*N*/, int L = 1) {

    typename Accum::Arguments          accum_args{};
    typename ScaleBroadcast::Arguments scale_args;
    scale_args.ptr_col    = rstd_ptr;
    scale_args.null_default = ElementScale(1);
    scale_args.dCol       = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(M)};
    typename MulCompute::Arguments     mul_args{};
    typename InputTree::Arguments      input_args{accum_args, scale_args, mul_args};

    typename LseColRed::Arguments lse_args;
    lse_args.ptr_col            = lse;
    lse_args.reduction_identity = -std::numeric_limits<float>::infinity();
    lse_args.dCol               = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(M)};
    typename LseTree::Arguments lse_tree_args{accum_args, lse_args};

    typename Accum::Arguments      out_acc_args{};
    typename IdentityOp::Arguments id_args{};
    typename OutputTree::Arguments output_args{out_acc_args, id_args};

    return {input_args, lse_tree_args, output_args};
  }

  static void launch_select_logits(sycl::queue& q,
                                    ElementD const* logits, int const* targets,
                                    float* logits_tgt, int M, int N, int L = 1) {
    standalone::select_logits(q, logits, targets, logits_tgt, M, N, L);
  }
};

}  // namespace xe_fuse
