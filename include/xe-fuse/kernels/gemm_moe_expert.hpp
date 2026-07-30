#pragma once

// MoE Expert Batched GEMM + fused SwiGLU/GeGLU
//
// Batches all routed experts into a single kernel launch using the L dimension:
//   A[l] = expert_input[l]  — [M_exp, K] per expert, L = num_active_experts
//   B[l] = expert_weight[l] — [N, K] per expert (gate+up stacked: N = 2*I)
//   D[l] = SwiGLU(A[l] @ B[l] * scale[l])
//
// One kernel launch replaces num_experts separate GEMM + SwiGLU launches.
// The EVT epilogue (SwiGLU/GeGLU + optional RMSNorm) runs per-batch-slice.
//
// Usage:
//   using Config = MoEExpertBatched<>;
//   // Allocate stacked: A[num_experts * M_exp * K], B[num_experts * K * N]
//   // Launch with L = num_active_experts

#include "xe-fuse/builder/epilogue_builder.hpp"
#include "xe-fuse/visitors/xe_pairwise_compute.hpp"

#include <sycl/sycl.hpp>

namespace xe_fuse {

template <
  typename ElementA_       = cutlass::bfloat16_t,
  typename ElementB_       = cutlass::bfloat16_t,
  typename ElementD_       = cutlass::bfloat16_t,
  typename ElementScale_   = float,
  typename ElementAcc_     = float,
  typename ElementCompute_ = float,
  typename TileShape_      = cute::Shape<cute::_256, cute::_256, cute::_32>,
  typename Activation_     = xe_fuse::SwiGLUFn  // or xe_fuse::GeGLUFn
>
struct MoEExpertBatched {
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

  // EVT: SwiGLU(acc * scale[m])  — same as K2
  using ScaleVec = cutlass::epilogue::fusion::XeColBroadcast<
      0, TileShape, ElementScale, ElementCompute,
      cute::Stride<cute::Int<1>, cute::Int<0>, int64_t>,
      128 / cutlass::sizeof_bits_v<ElementScale>>;

  using AccFetch = cutlass::epilogue::fusion::Sm90AccFetch;

  using MulCompute = cutlass::epilogue::fusion::XeCompute<
      cutlass::multiplies, ElementCompute, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTRmsScale = cutlass::epilogue::fusion::XeEVT<MulCompute, AccFetch, ScaleVec>;

  using PairwiseVisitor = xe_fuse::XePairwiseCompute<Activation_>;

  using EVT = cutlass::epilogue::fusion::XeEVT<PairwiseVisitor, EVTRmsScale>;

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
      ElementScale const* scale_ptr, int M_exp) {
    typename AccFetch::Arguments accum_args{};
    typename ScaleVec::Arguments scale_args;
    scale_args.ptr_col = scale_ptr;
    scale_args.null_default = ElementScale(1);
    scale_args.dCol = {cute::Int<1>{}, cute::Int<0>{}, static_cast<int64_t>(M_exp)};

    typename MulCompute::Arguments mul_args{};
    typename EVTRmsScale::Arguments rms_args{accum_args, scale_args, mul_args};

    typename PairwiseVisitor::Arguments swiglu_args{};
    return {rms_args, swiglu_args};
  }
};

// Convenience: SwiGLU variant
template <
  typename ElementA = cutlass::bfloat16_t,
  typename ElementB = cutlass::bfloat16_t,
  typename ElementD = cutlass::bfloat16_t,
  typename ElementScale = float,
  typename ElementAcc = float,
  typename ElementCompute = float,
  typename TileShape = cute::Shape<cute::_256, cute::_256, cute::_32>
>
using MoEExpertSwiGLU = MoEExpertBatched<
    ElementA, ElementB, ElementD, ElementScale, ElementAcc, ElementCompute,
    TileShape, xe_fuse::SwiGLUFn>;

// Convenience: GeGLU variant
template <
  typename ElementA = cutlass::bfloat16_t,
  typename ElementB = cutlass::bfloat16_t,
  typename ElementD = cutlass::bfloat16_t,
  typename ElementScale = float,
  typename ElementAcc = float,
  typename ElementCompute = float,
  typename TileShape = cute::Shape<cute::_256, cute::_256, cute::_32>
>
using MoEExpertGeGLU = MoEExpertBatched<
    ElementA, ElementB, ElementD, ElementScale, ElementAcc, ElementCompute,
    TileShape, xe_fuse::GeGLUFn>;

}  // namespace xe_fuse
