#pragma once

// K9_fused: gemm_partial_cross_entropy — fully-fused epilogue variant
//
// Same semantics as K9 (gemm_partial_cross_entropy) but eliminates the post-GEMM
// re-read of M×N logits for LSE by computing per-tile partial LSE in the epilogue:
//
//   Per GEMM tile (tile_m, tile_n):
//     lse_partial[m, tile_n] = logsumexp_{n in tile} D[m, n]
//
//   Post-GEMM launch_combine_lse():
//     lse[m] = logsumexp_k lse_partial[m, k]
//
// Uses XeColReduction<FinalReduction=false>.  The upstream xe_visitor.hpp bug
// where the "multiple warps in N" smem path used SLM address 0 (nullptr from
// xe_epilogue.hpp) with no-op barriers is fixed by the local_mem + xe_sync patch.
//
// Partial buffer layout (blocked_product of gBuf_layout and outer tile grid):
//   addr(m, tile_n, l) = m + padded_M * tile_n + padded_M * num_tiles_n * l
//   where padded_M = ceil(M, CTA_M)

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

  // ── LseTree: per-tile logsumexp → partial buffer (FinalReduction=false) ──
  // Each CTA writes lse_partial[m, tile_n] = logsumexp over its N tile.
  // launch_combine_lse() reduces across tile_n dimension post-GEMM.
  using LseColRed = cutlass::epilogue::fusion::XeColReduction<
      reduce_logsumexp,   // RegReduceFn
      reduce_logsumexp,   // ShuffleReduceFn
      reduce_logsumexp,   // GmemReduceFn (non-atomic: each CTA owns its partition)
      0,
      TileShape,
      ElementCompute,
      ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<cute::Int<1>, cute::Int<0>, int64_t>,  // M-stride=1, N-stride=0, batch
      128 / cutlass::sizeof_bits_v<ElementCompute>,
      true,    // EnableNullptr
      false    // FinalReduction=false: user combines partial results
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

  // Partial buffer size: padded_M × num_tiles_n × L floats.
  static size_t get_partials_size(int M, int N, int L) {
    int padded_M    = ((M + CTA_M - 1) / CTA_M) * CTA_M;
    int num_tiles_n = (N + CTA_N - 1) / CTA_N;
    return static_cast<size_t>(padded_M) * num_tiles_n * L;
  }

  static typename EVT::Arguments make_evt_args(
      ElementCompute* lse_partial,
      int M, int N, int L = 1) {

    int padded_M    = ((M + CTA_M - 1) / CTA_M) * CTA_M;
    int num_tiles_n = (N + CTA_N - 1) / CTA_N;
    // batch_stride = padded_M * num_tiles_n (L-stride of the blocked_product mBuf)
    auto batch_stride = static_cast<int64_t>(padded_M) * num_tiles_n;

    typename Accum::Arguments  accum_args{};
    typename CastOp::Arguments cast_args{};
    typename InputTree::Arguments input_args{accum_args, cast_args};

    typename LseColRed::Arguments lse_args;
    lse_args.ptr_col            = lse_partial;
    lse_args.reduction_identity = -std::numeric_limits<ElementCompute>::infinity();
    lse_args.dCol               = {cute::Int<1>{}, cute::Int<0>{}, batch_stride};
    typename LseTree::Arguments lse_tree_args{accum_args, lse_args};

    typename Accum::Arguments      out_acc_args{};
    typename IdentityOp::Arguments id_args{};
    typename OutputTree::Arguments output_args{out_acc_args, id_args};

    return {input_args, lse_tree_args, output_args};
  }

  // Post-GEMM: combine per-tile partial lse → lse[M*L]
  //
  // Buffer address (blocked_product layout, derived in header comment):
  //   addr(m, tile_n, l) = m + padded_M * tile_n + padded_M * num_tiles_n * l
  static void launch_combine_lse(
      sycl::queue& q,
      ElementCompute const* lse_partial,
      float* lse, int M, int N, int L = 1) {
    int padded_M    = ((M + CTA_M - 1) / CTA_M) * CTA_M;
    int num_tiles_n = (N + CTA_N - 1) / CTA_N;
    q.parallel_for(sycl::range<1>(static_cast<size_t>(M) * L), [=](sycl::id<1> idx) {
      int row = static_cast<int>(idx[0]);
      int m   = row % M;
      int l   = row / M;

      float result = -INFINITY;
      for (int tn = 0; tn < num_tiles_n; ++tn) {
        int addr = m + padded_M * tn + padded_M * num_tiles_n * l;
        float v  = lse_partial[addr];
        float mx = sycl::fmax(result, v);
        if (mx == -INFINITY) continue;
        result = mx + sycl::log(sycl::exp(result - mx) + sycl::exp(v - mx));
      }
      lse[row] = result;
    });
  }

  static void launch_select_logits(sycl::queue& q,
                                    ElementD const* logits, int const* targets,
                                    float* logits_tgt, int M, int N, int L = 1) {
    standalone::select_logits(q, logits, targets, logits_tgt, M, N, L);
  }
};

}  // namespace xe_fuse
