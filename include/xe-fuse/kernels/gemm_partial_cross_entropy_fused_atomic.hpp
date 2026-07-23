#pragma once

// K9_fused: gemm_partial_cross_entropy — fully-fused epilogue variant
//
// Same semantics as K9 (gemm_partial_cross_entropy) but eliminates the post-GEMM
// re-read of M×N logits for LSE by computing lse atomically in the GEMM epilogue:
//
//   Per GEMM tile (tile_m, tile_n):
//     lse[m] = atomic_logsumexp(lse[m], logsumexp_{n in tile} D[m,n])
//
// Uses XeColReduction<IsAtomic=true> which takes the filter_zeros path, correctly
// writing all M rows (vs FinalReduction=false which only wrote 8 rows due to
// N-stride=0 aliasing in copy_aligned).
// Pre-initialize lse buffer to -infinity before GEMM (done by initialize_workspace).

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
  typename ElementAcc_     = float,
  typename ElementCompute_ = float,
  typename TileShape_      = cute::Shape<cute::_256, cute::_256, cute::_32>
>
struct GemmPartialCrossEntropyFused {
  using ElementA       = ElementA_;
  using ElementB       = ElementB_;
  using ElementD       = ElementD_;
  using ElementAcc     = ElementAcc_;
  using ElementCompute = ElementCompute_;
  using TileShape      = TileShape_;

  static constexpr int CTA_M = cute::size<0>(TileShape{});
  static constexpr int CTA_N = cute::size<1>(TileShape{});

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;

  using StrideC = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideD = cute::Stride<int64_t, cute::Int<1>, int64_t>;

  // ── InputTree: identity cast (acc → ElementD) ────────────────────────────
  using Accum = cutlass::epilogue::fusion::XeAccFetch;
  using CastOp = cutlass::epilogue::fusion::XeCompute<
      cutlass::epilogue::thread::Identity,
      ElementD, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using InputTree = cutlass::epilogue::fusion::XeEVT<CastOp, Accum>;

  // ── OutputTree: write logits D ────────────────────────────────────────────
  using IdentityOp = cutlass::epilogue::fusion::XeCompute<
      cutlass::epilogue::thread::Identity,
      ElementD, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using OutputTree = cutlass::epilogue::fusion::XeEVT<IdentityOp, Accum>;

  // ── LseTree: atomic logsumexp into lse[M*L] ──────────────────────────────
  // atomic_reduce_logsumexp triggers IsAtomic=true → filter_zeros path in reduce(),
  // which correctly iterates all unique M rows regardless of N-stride=0.
  // FinalReduction=true required when IsAtomic=true.
  // lse buffer must be pre-initialized to -infinity (done by initialize_workspace).
  using LseColRed = cutlass::epilogue::fusion::XeColReduction<
      reduce_logsumexp,          // RegReduceFn (warp-register accumulation)
      reduce_logsumexp,          // ShuffleReduceFn (warp-shuffle reduction)
      atomic_reduce_logsumexp,   // GmemReduceFn (atomic write to final lse)
      0,
      TileShape,
      float,             // ElementOutput = float (final lse)
      ElementCompute,    // ElementCompute = float
      cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<cute::Int<1>, cute::Int<0>, int64_t>,  // M-stride=1, N-stride=0, batch
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

  // Build EVT arguments pointing to final lse output (not a partial buffer).
  // initialize_workspace fills lse with -infinity before the first run.
  // For repeated runs, re-fill lse with -infinity before each gemm_op.run().
  static typename EVT::Arguments make_evt_args(
      float* lse, int M, int /*N*/, int L = 1) {

    typename Accum::Arguments  accum_args{};
    typename CastOp::Arguments cast_args{};
    typename InputTree::Arguments input_args{accum_args, cast_args};

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
