#pragma once

// xe-fuse builder: Level 1 template aliases for composing GEMM epilogues.
//
// Replaces verbose nested XeEVT<XeCompute<multiplies, ...>, XeAccFetch, ...>
// chains with readable domain-specific aliases.
//
// Example — K1 (GEMM + RmsNorm):
//
//   // Before (raw CUTLASS EVT):
//   using ScaleBroadcast = XeColBroadcast<0, TileShape, float, float,
//       Stride<Int<1>, Int<0>, int64_t>, 4>;
//   using MulCompute = XeCompute<multiplies, bf16, float, round_to_nearest>;
//   using EVT = XeEVT<MulCompute, XeAccFetch, ScaleBroadcast>;
//
//   // After (builder):
//   namespace b = xe_fuse::builder;
//   using EVT = b::ScaleRows<b::Acc, TileShape, float>;
//
// Example — K0a (GEMM + Residual + Gamma):
//
//   using Step1 = b::AddResidual<bf16>;
//   using EVT   = b::ScaleCols<Step1, TileShape, float>;
//
// Example — K4 (GEMM + RmsNorm + RoPE):
//
//   using Step1 = b::ScaleRows<b::Acc, TileShape, float>;
//   using EVT   = b::RoPEComposed<Step1, float>;
//
// Full kernel with MakeGemm:
//
//   using K = b::MakeGemm<EVT, bf16, bf16, bf16, float, float, TileShape>;
//   using Gemm = typename K::Gemm;

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"

#include <cute/tensor.hpp>

#include "xe-fuse/visitors/xe_elementwise_compute.hpp"
#include "xe-fuse/visitors/xe_pairwise_compute.hpp"
#include "xe-fuse/visitors/xe_rope_compute.hpp"
#include "xe-fuse/visitors/xe_scalerows_compute.hpp"

namespace xe_fuse::builder {

// ============================================================
// Leaf Nodes — data sources in the epilogue
// ============================================================

// GEMM accumulator (frg_acc)
using Acc = cutlass::epilogue::fusion::XeAccFetch;

// Full MxN auxiliary tensor load via Block2D
template <typename Element,
          typename Stride = cute::Stride<int64_t, cute::Int<1>, int64_t>,
          int VecWidth = 128 / cutlass::sizeof_bits_v<Element>>
using AuxLoad = cutlass::epilogue::fusion::XeAuxLoad<
    Element, Stride, void, VecWidth, true, true>;

// Per-column vector broadcast: scale[m] replicated across N columns
// Idx is kept for API consistency but always maps to Stages=0 (no smem pipelining on Xe)
template <int Idx, typename TileShape, typename Element,
          typename ElementCompute = float,
          int VecWidth = 128 / cutlass::sizeof_bits_v<Element>>
using ColBroadcast = cutlass::epilogue::fusion::XeColBroadcast<
    0, TileShape, Element, ElementCompute,
    cute::Stride<cute::Int<1>, cute::Int<0>, int64_t>,
    VecWidth>;

// Per-row vector broadcast: scale[n] replicated across M rows
// Idx is kept for API consistency but always maps to Stages=0 (no smem pipelining on Xe)
template <int Idx, typename TileShape, typename Element,
          typename ElementCompute = float,
          int VecWidth = 128 / cutlass::sizeof_bits_v<Element>>
using RowBroadcast = cutlass::epilogue::fusion::XeRowBroadcast<
    0, TileShape, Element, ElementCompute,
    cute::Stride<cute::Int<0>, cute::Int<1>, int64_t>,
    VecWidth>;

// ============================================================
// Compute Nodes — element-wise operations
// ============================================================

template <typename ElementOut = float, typename ElementCompute = float>
using MulOp = cutlass::epilogue::fusion::XeCompute<
    cutlass::multiplies, ElementOut, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

template <typename ElementOut = float, typename ElementCompute = float>
using AddOp = cutlass::epilogue::fusion::XeCompute<
    cutlass::plus, ElementOut, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

// Generic EVT composition: EVT<NodeOp, Children...>
template <typename Op, typename... Children>
using EVT = cutlass::epilogue::fusion::XeEVT<Op, Children...>;

// Shorthand: Mul<A, B> and Add<A, B>
template <typename A, typename B,
          typename ElementOut = float, typename ElementCompute = float>
using Mul = EVT<MulOp<ElementOut, ElementCompute>, A, B>;

template <typename A, typename B,
          typename ElementOut = float, typename ElementCompute = float>
using Add = EVT<AddOp<ElementOut, ElementCompute>, A, B>;

// ============================================================
// Domain-Specific Compound Aliases
// ============================================================

// ScaleRows: input * R[row] — per-row scaling via column broadcast
// Used for RMSNorm: D = acc * R[m]
template <typename Input, typename TileShape,
          typename ElementScale = float, typename ElementCompute = float>
using ScaleRows = Mul<Input,
    ColBroadcast<0, TileShape, ElementScale, ElementCompute>,
    ElementCompute, ElementCompute>;

// ScaleCols: W[col] * input — per-column scaling via row broadcast
// Used for gamma weighting: D = gamma[n] * input
template <typename Input, typename TileShape,
          typename ElementScale = float, typename ElementCompute = float>
using ScaleCols = Mul<
    RowBroadcast<0, TileShape, ElementScale, ElementCompute>,
    Input,
    ElementCompute, ElementCompute>;

// AddResidual: acc + AuxLoad(residual)
template <typename ElementResidual = cutlass::bfloat16_t,
          typename ElementCompute = float>
using AddResidual = Add<Acc, AuxLoad<ElementResidual>, ElementCompute, ElementCompute>;

// ============================================================
// Pairwise Operations — lane-shuffle-based pair computations
// ============================================================

// Generic pairwise compute: applies PairwiseFn across adjacent lane pairs
template <typename PairwiseFn, typename Input>
using Pairwise = EVT<XePairwiseCompute<PairwiseFn>, Input>;

// SwiGLU on pairwise columns of input
template <typename Input>
using SwiGLU = Pairwise<SwiGLUFn, Input>;

// GeGLU on pairwise columns of input (Gemma-style gated FFN)
template <typename Input>
using GeGLU = Pairwise<GeGLUFn, Input>;

// ============================================================
// RoPE — Rotary Position Embedding
// ============================================================

// RoPE on raw GEMM accumulator (single-child visitor, uses frg_acc directly)
// cos_sin loaded via Block2D AuxLoad
template <typename ElementCosSin = float>
using RoPE = EVT<XeRoPECompute, AuxLoad<ElementCosSin>>;

// RoPE on pre-processed input (two-child visitor)
// child 0 = pre-processed data (e.g., RMS-scaled), child 1 = cos_sin AuxLoad
template <typename Input, typename ElementCosSin = float>
using RoPEComposed = EVT<XeRoPEComputeTwoChild, Input, AuxLoad<ElementCosSin>>;

// Merged scale + RoPE (reads frg_acc directly, flattened single-level tree)
// child 0 = per-row scale (ColBroadcast), child 1 = cos_sin (AuxLoad)
// Fewer register spills than RoPEComposed<ScaleRows<Acc, ...>, ...>
template <typename TileShape, typename ElementScale = float, typename ElementCosSin = float>
using RoPEScaled = EVT<XeRoPEScaledCompute,
    ColBroadcast<0, TileShape, ElementScale>,
    AuxLoad<ElementCosSin>>;

// ============================================================
// Merged Visitors — flat trees with fewer register spills
// ============================================================

// ScaleRowsMerged: acc * R[m] in a single visitor (merged K1v2)
// Flat tree: reads frg_acc directly, no AccFetch or MulCompute nodes
template <typename TileShape, typename ElementScale = float>
using ScaleRowsMerged = EVT<XeScaleRowsCompute,
    ColBroadcast<0, TileShape, ElementScale>>;

// SwiGLUScaled: SwiGLU(acc * R[m]) in a single visitor (merged K2v2)
// Flat tree: scale + shuffle + silu(gate)*up in one visit() call
template <typename TileShape, typename ElementScale = float>
using SwiGLUScaled = EVT<XeScaleRowsSwiGLUCompute,
    ColBroadcast<0, TileShape, ElementScale>>;

// GeGLUScaled: GeGLU(acc * R[m]) in a single visitor (merged K2_geglu_v2)
// Same as SwiGLUScaled but with GeLU activation
template <typename TileShape, typename ElementScale = float>
using GeGLUScaled = EVT<XeScaleRowsGeGLUCompute,
    ColBroadcast<0, TileShape, ElementScale>>;

// ============================================================
// Activation Functions — element-wise unary operations
// ============================================================

// GeLU: x * 0.5 * (1 + erf(x / sqrt(2)))  — BERT, GPT-2/3/4, Gemma
template <typename Input>
using GeLU = EVT<XeElementwiseCompute<GeLUFn>, Input>;

// GeLU with tanh approximation — common fast variant
template <typename Input>
using GeLUTanh = EVT<XeElementwiseCompute<GeLUTanhFn>, Input>;

// SiLU (Swish): x * sigmoid(x) — LLaMA, Mistral (scalar, not pairwise)
template <typename Input>
using SiLU = EVT<XeElementwiseCompute<SiLUFn>, Input>;

// ReLU: max(0, x)
template <typename Input>
using ReLU = EVT<XeElementwiseCompute<ReLUFn>, Input>;

// Sigmoid: 1 / (1 + exp(-x))
template <typename Input>
using Sigmoid = EVT<XeElementwiseCompute<SigmoidFn>, Input>;

// ============================================================
// BiasAdd — per-column bias addition
// ============================================================

// BiasAdd: acc + bias[n] — BERT, GPT-2, very common
template <typename TileShape, typename ElementBias = float,
          typename ElementCompute = float>
using BiasAdd = Add<Acc, RowBroadcast<0, TileShape, ElementBias, ElementCompute>,
                    ElementCompute, ElementCompute>;

// ============================================================
// Quantization — dequant/requant epilogues for INT8/FP8
// ============================================================

// DequantW8A8: int32_acc * scale_token[m] * scale_channel[n] → bf16
// Per-token (ColBroadcast, Idx=0) × per-channel (RowBroadcast, Idx=1)
// Standard W8A8 dequantization for INT8×INT8 GEMM in LLM inference
template <typename TileShape,
          typename ElementScale = float, typename ElementCompute = float>
using DequantW8A8 = Mul<
    Mul<Acc,
        ColBroadcast<0, TileShape, ElementScale, ElementCompute>,
        ElementCompute, ElementCompute>,
    RowBroadcast<1, TileShape, ElementScale, ElementCompute>,
    ElementCompute, ElementCompute>;

// DequantW8A8Biased: int32_acc * scale_token[m] * scale_channel[n] + bias[n] → bf16
// Same as DequantW8A8 but adds a per-channel bias after dequantization
template <typename TileShape,
          typename ElementScale = float, typename ElementBias = float,
          typename ElementCompute = float>
using DequantW8A8Biased = Add<
    DequantW8A8<TileShape, ElementScale, ElementCompute>,
    RowBroadcast<2, TileShape, ElementBias, ElementCompute>,
    ElementCompute, ElementCompute>;

// ============================================================
// Auxiliary Store — write intermediate results to a buffer
// ============================================================

template <typename Element,
          typename Stride = cute::Stride<int64_t, cute::Int<1>, int64_t>,
          int VecWidth = 128 / cutlass::sizeof_bits_v<Element>>
using AuxStore = cutlass::epilogue::fusion::XeAuxStore<
    Element, Stride, void, VecWidth, true, false>;

// ============================================================
// Split-Tree — dual output from a shared input
// ============================================================

// DualOutput: evaluate InputTree once, then:
//   - store the result to an auxiliary buffer (AuxElement type)
//   - apply OutputTree to produce the primary D output
//
// Example — K0 (residual add with aux store for rstd):
//   using Input = AddResidual<bf16>;                        // acc + residual
//   using Output = ScaleCols<Acc, TileShape, float>;        // input * gamma
//   using EVT = DualOutput<Input, Output, bf16>;            // store raw sum + gamma-weighted
template <typename InputTree, typename OutputTree,
          typename AuxElement = cutlass::bfloat16_t,
          typename AuxStride = cute::Stride<int64_t, cute::Int<1>, int64_t>>
using DualOutput = cutlass::epilogue::fusion::Sm90SplitTreeVisitor<
    InputTree,
    OutputTree,
    EVT<AuxStore<AuxElement, AuxStride>, Acc>
>;

// ============================================================
// MakeGemm — eliminates CollectiveBuilder boilerplate
// ============================================================

template <
  typename EVT_,
  typename ElementA_       = cutlass::bfloat16_t,
  typename ElementB_       = cutlass::bfloat16_t,
  typename ElementD_       = cutlass::bfloat16_t,
  typename ElementAcc_     = float,
  typename ElementCompute_ = float,
  typename TileShape_      = cute::Shape<cute::_256, cute::_256, cute::_32>,
  typename LayoutA_        = cutlass::layout::RowMajor,
  typename LayoutB_        = cutlass::layout::RowMajor,
  int AlignmentAB_         = 8,
  int AlignmentCD_         = 8
>
struct MakeGemm {
  using ElementA       = ElementA_;
  using ElementB       = ElementB_;
  using ElementD       = ElementD_;
  using ElementAcc     = ElementAcc_;
  using ElementCompute = ElementCompute_;
  using TileShape      = TileShape_;
  using LayoutA        = LayoutA_;
  using LayoutB        = LayoutB_;

  using StrideC = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideD = cute::Stride<int64_t, cute::Int<1>, int64_t>;

  using EVT = EVT_;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Xe20, cutlass::arch::OpClassTensorOp,
      TileShape, cute::Shape<cute::_1, cute::_1, cute::_1>,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAcc, ElementCompute,
      ElementD, StrideC, AlignmentCD_,
      ElementD, StrideD, AlignmentCD_,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      EVT
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Xe20, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentAB_,
      ElementB, LayoutB, AlignmentAB_,
      ElementAcc, TileShape,
      cute::Shape<cute::_1, cute::_1, cute::_1>,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

}  // namespace xe_fuse::builder
