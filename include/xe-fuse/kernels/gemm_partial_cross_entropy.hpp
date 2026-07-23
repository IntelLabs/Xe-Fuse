#pragma once

// K9: gemm_partial_cross_entropy — D[m,n] = acc[m,n] (bare GEMM, output = logits)
//
// Post-GEMM (standalone kernels):
//   lse[m]         = log(sum_n exp(D[m,n]))
//   logits_tgt[m]  = D[m, targets[m]]
//
// kernel_9 from CODA (arxiv.org/abs/2605.19269): vocab-projection GEMM with
// cross-entropy statistics, no RMSNorm scaling. See K5 for the scaled variant.
//
// The GEMM epilogue is the identity (acc cast to ElementD). LSE and logit
// selection are computed post-GEMM from the written logits tensor.

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"

#include <cute/tensor.hpp>

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
struct GemmPartialCrossEntropy {
  using ElementA       = ElementA_;
  using ElementB       = ElementB_;
  using ElementD       = ElementD_;
  using ElementAcc     = ElementAcc_;
  using ElementCompute = ElementCompute_;
  using TileShape      = TileShape_;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;

  using StrideC = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideD = cute::Stride<int64_t, cute::Int<1>, int64_t>;

  // Identity epilogue: cast acc to ElementD
  using Accum = cutlass::epilogue::fusion::XeAccFetch;

  using CastOp = cutlass::epilogue::fusion::XeCompute<
      cutlass::epilogue::thread::Identity,
      ElementD, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest
  >;

  using EVT = cutlass::epilogue::fusion::XeEVT<CastOp, Accum>;

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

  static typename EVT::Arguments make_evt_args() {
    typename Accum::Arguments accum_args{};
    typename CastOp::Arguments cast_args{};
    return {accum_args, cast_args};
  }

  // Post-GEMM: compute log-sum-exp per row
  static void launch_lse(sycl::queue& q,
                          ElementD const* logits, float* lse,
                          int M, int N, int L) {
    standalone::compute_lse(q, logits, lse, M, N, L);
  }

  // Post-GEMM: extract target logit per row
  static void launch_select_logits(sycl::queue& q,
                                    ElementD const* logits, int const* targets,
                                    float* logits_tgt, int M, int N, int L) {
    standalone::select_logits(q, logits, targets, logits_tgt, M, N, L);
  }
};

}  // namespace xe_fuse
