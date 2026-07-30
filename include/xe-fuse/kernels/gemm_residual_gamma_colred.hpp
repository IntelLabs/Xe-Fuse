#pragma once

// K0a_CR: gemm_residual_gamma + fused ColReduction (CODA-style)
//
// D[m,n] = gamma[n] * (acc[m,n] + residual[m,n])
// sum_sq[tile_m, tile_n, m_local] += sum_col(D[m,n]^2) per CTA tile, non-atomically
//
// Eliminates the standalone compute_rstd kernel. After this GEMM:
//   launch_reduce_partials_rsqrt() sums tile partials and computes rstd[M]
//   reads O(M * num_N_tiles) floats instead of O(M * N) bf16 values
//
// Key design: residual is loaded via the C matrix (XeSrcFetch), NOT
// XeAuxLoad. This avoids the IGC bug where AuxLoad + ColReduction
// in the same kernel causes "Constraints for inline assembly" errors.
//
// EVT tree (SplitTreeVisitor):
//   InputTree:  gamma[n] * (acc + SrcFetch)       → D values
//   OutputTree: Identity(Acc)                      → store D
//   AuxOutTree: ColReduction(Acc * Acc)            → per-tile sum_sq (FinalReduction=false)

#include "xe-fuse/builder/epilogue_builder.hpp"

#include <sycl/sycl.hpp>

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
struct GemmResidualGammaColRed {
  using ElementA        = ElementA_;
  using ElementB        = ElementB_;
  using ElementD        = ElementD_;
  using ElementResidual = ElementResidual_;
  using ElementGamma    = ElementGamma_;
  using ElementAcc      = ElementAcc_;
  using ElementCompute  = ElementCompute_;
  using TileShape       = TileShape_;

  static constexpr int CTA_M = cute::size<0>(TileShape{});
  static constexpr int CTA_N = cute::size<1>(TileShape{});

  // Per-tile partial sum_sq buffer: shape [padded_M * num_tiles_n * L].
  // Each CTA writes its partial sum to a unique slot (no atomics, no races).
  // No pre-zeroing needed; each slot is overwritten exactly once per GEMM.
  static size_t get_partials_size(int M, int N, int L) {
    int padded_M    = ((M + CTA_M - 1) / CTA_M) * CTA_M;
    int num_tiles_n = (N + CTA_N - 1) / CTA_N;
    return static_cast<size_t>(padded_M) * num_tiles_n * L;
  }

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;

  using StrideC        = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideD        = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideResidual = cute::Stride<int64_t, cute::Int<1>, int64_t>;

  // C matrix source fetch — loads residual through the standard C-load path.
  // This avoids AuxLoad, which triggers an IGC bug when combined with ColReduction.
  using SrcFetch = cutlass::epilogue::fusion::Sm90SrcFetch<ElementResidual>;

  // ── InputTree: gamma[n] * (acc + C) ──
  using InputTree = builder::EVT<
      builder::MulOp<ElementCompute, ElementCompute>,
      builder::RowBroadcast<0, TileShape, ElementGamma, ElementCompute>,
      builder::EVT<
          builder::AddOp<ElementCompute, ElementCompute>,
          builder::Acc,
          SrcFetch
      >
  >;

  // ── OutputTree: pass-through to D store ──
  using IdentityOp = cutlass::epilogue::fusion::XeCompute<
      cutlass::epilogue::thread::Identity,
      ElementD, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest
  >;
  using OutputTree = builder::EVT<IdentityOp, builder::Acc>;

  // ── AuxOutTree: non-atomic ColReduction of D² per row ──
  // FinalReduction=false: each CTA writes its full row partial sums to a unique
  // per-tile slot in the partials buffer. Within-CTA races across N sub-groups
  // are resolved by SLM reduction in XeColReduction (xe_visitor.hpp Case 2).
  using SquaredInput = builder::Mul<builder::Acc, builder::Acc,
                                     ElementCompute, ElementCompute>;

  using ColRed = cutlass::epilogue::fusion::XeColReduction<
      cutlass::plus,
      cutlass::plus,
      cutlass::plus,
      0,
      TileShape,
      ElementCompute,
      ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest,
      cute::Stride<cute::Int<1>, cute::Int<0>, int64_t>,
      128 / cutlass::sizeof_bits_v<ElementCompute>,
      true,   // EnableNullptr
      false   // FinalReduction=false: per-tile writes, user reduces afterward
  >;

  using ColRedTree = builder::EVT<ColRed, SquaredInput>;

  // ── Combined: SplitTree(InputTree, OutputTree, ColRedTree) ──
  using EVT = cutlass::epilogue::fusion::Sm90SplitTreeVisitor<
      InputTree, OutputTree, ColRedTree>;

  using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Xe20, cutlass::arch::OpClassTensorOp,
      TileShape,
      cute::Shape<cute::_1, cute::_1, cute::_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      ElementResidual, StrideC, 8,
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
      ElementGamma const* gamma_ptr, int N,
      ElementCompute* sum_sq_ptr, int M, int L = 1) {

    // InputTree: Mul<RowBroadcast<gamma>, Add<Acc, SrcFetch>>
    typename builder::Acc::Arguments accum_args{};
    typename SrcFetch::Arguments src_args{};

    typename builder::AddOp<>::Arguments add_args{};
    using AddTree = builder::EVT<builder::AddOp<ElementCompute, ElementCompute>,
                                  builder::Acc, SrcFetch>;
    typename AddTree::Arguments add_tree_args{accum_args, src_args, add_args};

    typename builder::RowBroadcast<0, TileShape, ElementGamma, ElementCompute>::Arguments gamma_args;
    gamma_args.ptr_row = gamma_ptr;
    gamma_args.null_default = ElementGamma(1);
    gamma_args.dRow = {cute::Int<0>{}, cute::Int<1>{}, static_cast<int64_t>(N)};

    typename builder::MulOp<>::Arguments mul_args{};
    typename InputTree::Arguments input_args{gamma_args, add_tree_args, mul_args};

    // OutputTree: EVT<Identity, Acc>
    typename builder::Acc::Arguments out_acc_args{};
    typename IdentityOp::Arguments identity_args{};
    typename OutputTree::Arguments output_args{out_acc_args, identity_args};

    // ColRedTree: EVT<ColRed, Mul<Acc, Acc>>
    typename builder::Acc::Arguments sq_acc1_args{};
    typename builder::Acc::Arguments sq_acc2_args{};
    typename builder::MulOp<>::Arguments sq_mul_args{};
    typename SquaredInput::Arguments sq_args{sq_acc1_args, sq_acc2_args, sq_mul_args};

    // dCol batch stride = padded_M * num_tiles_n (blocked_product per-tile layout)
    int padded_M    = ((M + CTA_M - 1) / CTA_M) * CTA_M;
    int num_tiles_n = (N + CTA_N - 1) / CTA_N;
    typename ColRed::Arguments col_red_args;
    col_red_args.ptr_col = sum_sq_ptr;
    col_red_args.reduction_identity = ElementCompute(0);
    col_red_args.dCol = {cute::Int<1>{}, cute::Int<0>{},
                         static_cast<int64_t>(padded_M) * num_tiles_n};

    typename ColRedTree::Arguments col_red_tree_args{sq_args, col_red_args};

    // SplitTreeVisitor args order: InputTree, AuxOutTrees..., OutputTree
    return {input_args, col_red_tree_args, output_args};
  }
};

// Post-GEMM reduction: partials[padded_M * num_tiles_n * L] → rstd[M * L]
// Sums per-tile partial sums across all N-tiles for each M-row, then computes rstd.
// No pre-zeroing needed: each tile slot is overwritten once per GEMM run.
template <int CTA_M_, int CTA_N_, typename ElementOutput = float>
void launch_reduce_partials_rsqrt(
    sycl::queue& q,
    float const* partials_ptr,   // per-tile sums, shape [padded_M * num_tiles_n * L]
    ElementOutput* rstd_ptr,     // output, shape [M * L]
    int M, int N, int L,
    float eps = 1e-6f)
{
  int padded_M    = ((M + CTA_M_ - 1) / CTA_M_) * CTA_M_;
  int num_tiles_n = (N + CTA_N_ - 1) / CTA_N_;
  q.parallel_for(
    sycl::range<1>(static_cast<size_t>(M) * L),
    [=](sycl::id<1> idx) {
      int i       = static_cast<int>(idx[0]);
      int m       = i % M;
      int l       = i / M;
      int tile_m  = m / CTA_M_;
      int local_m = m % CTA_M_;
      // mBuf address: local_m + CTA_M_ * (tile_m * num_tiles_n * L + tile_n * L + l)
      int base = local_m + CTA_M_ * (tile_m * num_tiles_n * L + l);
      float sum_sq = 0.0f;
      for (int tn = 0; tn < num_tiles_n; ++tn) {
        sum_sq += partials_ptr[base + CTA_M_ * tn * L];
      }
      rstd_ptr[i] = static_cast<ElementOutput>(
          1.0f / sycl::sqrt(sum_sq / static_cast<float>(N) + eps));
    }
  );
}

}  // namespace xe_fuse
