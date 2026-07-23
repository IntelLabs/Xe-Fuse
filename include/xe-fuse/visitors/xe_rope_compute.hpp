#pragma once

// XeRoPECompute — Single-child epilogue visitor for RoPE rotation.
//
// Takes frg_acc (GEMM accumulator) directly, and one child input (cos_sin).
//
// cos_sin is interleaved: even columns = cos, odd columns = sin.
// Data values from frg_acc are shuffled between adjacent lane pairs via
// shfl_xor_sync, then the RoPE rotation is applied:
//
//   lane 2k (even):  output =  x_even * cos + x_odd * sin
//   lane 2k+1 (odd): output = -x_even * sin + x_odd * cos

#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/gpu_generics.h"

namespace xe_fuse {

struct XeRoPECompute : cutlass::epilogue::fusion::Sm90VisitorImpl<> {

  using Sm90VisitorImpl<>::Sm90VisitorImpl;

  struct ConsumerStoreCallbacks : cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks {

    template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
    CUTLASS_DEVICE cutlass::Array<ElementAccumulator, FragmentSize>
    visit(cutlass::Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          cutlass::Array<ElementInput, FragmentSize> const& frg_cos_sin) {

      cutlass::Array<ElementAccumulator, FragmentSize> result;

      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      uint32_t lane_id = sg.get_local_linear_id();
      bool is_even = (lane_id & 1) == 0;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        // Shuffle accumulator data with partner lane
        uint32_t my_bits = reinterpret_cast<const uint32_t&>(frg_acc[i]);
        uint32_t partner_bits = shfl_xor_sync(0xFFFFFFFF, my_bits, 1, 16);
        float my_val = reinterpret_cast<const float&>(my_bits);
        float partner_val = reinterpret_cast<const float&>(partner_bits);

        // cos_sin: convert to float first (handles both float and bf16 inputs),
        // then shuffle the float bits between lanes
        float my_cs = static_cast<float>(frg_cos_sin[i]);
        uint32_t my_cs_bits = reinterpret_cast<const uint32_t&>(my_cs);
        uint32_t partner_cs_bits = shfl_xor_sync(0xFFFFFFFF, my_cs_bits, 1, 16);
        float partner_cs = reinterpret_cast<const float&>(partner_cs_bits);

        float cos_val = is_even ? my_cs : partner_cs;
        float sin_val = is_even ? partner_cs : my_cs;

        float out;
        if (is_even) {
          out = my_val * cos_val + partner_val * sin_val;
        } else {
          out = -partner_val * sin_val + my_val * cos_val;
        }

        result[i] = static_cast<ElementAccumulator>(out);
      }

      return result;
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(cutlass::epilogue::fusion::ConsumerStoreArgs<Args...> const& args) {
    return ConsumerStoreCallbacks{};
  }
};

// XeRoPEComputeTwoChild — Two-child epilogue visitor for RoPE rotation.
//
// Unlike XeRoPECompute (which reads frg_acc directly), this visitor takes
// its data values from child 0 (e.g. an RMS-scaled sub-tree) and cos_sin
// from child 1 (e.g. an AuxLoad).
//
// This allows composing arbitrary pre-processing (RMSNorm, scaling, etc.)
// before the RoPE rotation:
//
//   XeEVT<XeRoPEComputeTwoChild,
//     XeEVT<Compute<mul>, AccFetch, ColBroadcast<R>>,   // child 0: pre-processed data
//     XeAuxLoad<cos_sin>                                  // child 1: interleaved cos/sin
//   >
//
// cos_sin is interleaved: even columns = cos, odd columns = sin.
// Data values from child 0 are shuffled between adjacent lane pairs via
// shfl_xor_sync, then the RoPE rotation is applied:
//
//   lane 2k (even):  output =  x_even * cos + x_odd * sin
//   lane 2k+1 (odd): output = -x_even * sin + x_odd * cos

struct XeRoPEComputeTwoChild : cutlass::epilogue::fusion::Sm90VisitorImpl<> {

  using Sm90VisitorImpl<>::Sm90VisitorImpl;

  struct ConsumerStoreCallbacks : cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks {

    template <typename ElementAccumulator, typename ElementInput0, typename ElementInput1, int FragmentSize>
    CUTLASS_DEVICE cutlass::Array<ElementInput0, FragmentSize>
    visit(cutlass::Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          cutlass::Array<ElementInput0, FragmentSize> const& frg_data,       // child 0: pre-processed values
          cutlass::Array<ElementInput1, FragmentSize> const& frg_cos_sin) {  // child 1: cos_sin

      cutlass::Array<ElementInput0, FragmentSize> result;

      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      uint32_t lane_id = sg.get_local_linear_id();
      bool is_even = (lane_id & 1) == 0;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        // Shuffle pre-processed data with partner lane
        uint32_t my_bits = reinterpret_cast<const uint32_t&>(frg_data[i]);
        uint32_t partner_bits = shfl_xor_sync(0xFFFFFFFF, my_bits, 1, 16);
        float my_val = reinterpret_cast<const float&>(my_bits);
        float partner_val = reinterpret_cast<const float&>(partner_bits);

        // cos_sin: convert to float first (handles both float and bf16 inputs),
        // then shuffle the float bits between lanes
        float my_cs = static_cast<float>(frg_cos_sin[i]);
        uint32_t my_cs_bits = reinterpret_cast<const uint32_t&>(my_cs);
        uint32_t partner_cs_bits = shfl_xor_sync(0xFFFFFFFF, my_cs_bits, 1, 16);
        float partner_cs = reinterpret_cast<const float&>(partner_cs_bits);

        float cos_val = is_even ? my_cs : partner_cs;
        float sin_val = is_even ? partner_cs : my_cs;

        float out;
        if (is_even) {
          out = my_val * cos_val + partner_val * sin_val;
        } else {
          out = -partner_val * sin_val + my_val * cos_val;
        }

        result[i] = static_cast<ElementInput0>(out);
      }

      return result;
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(cutlass::epilogue::fusion::ConsumerStoreArgs<Args...> const& args) {
    return ConsumerStoreCallbacks{};
  }
};

// XeRoPEScaledCompute — Merged scale + RoPE in a single visitor.
//
// Flattens K4's two-level tree (MulCompute sub-tree + RoPE) into one visit().
// Takes frg_acc directly (not through a sub-tree), plus two children:
//   child 0: per-row scale factors (e.g., ColBroadcast<R>)
//   child 1: interleaved cos/sin (e.g., AuxLoad)
//
// In one pass: scale → shuffle → rotate. Eliminates intermediate tree
// evaluation and reduces register pressure vs the composed approach.

struct XeRoPEScaledCompute : cutlass::epilogue::fusion::Sm90VisitorImpl<> {

  using Sm90VisitorImpl<>::Sm90VisitorImpl;

  struct ConsumerStoreCallbacks : cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks {

    template <typename ElementAccumulator, typename ElementScale, typename ElementCosSin, int FragmentSize>
    CUTLASS_DEVICE cutlass::Array<ElementAccumulator, FragmentSize>
    visit(cutlass::Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          cutlass::Array<ElementScale, FragmentSize> const& frg_scale,
          cutlass::Array<ElementCosSin, FragmentSize> const& frg_cos_sin) {

      cutlass::Array<ElementAccumulator, FragmentSize> result;

      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      uint32_t lane_id = sg.get_local_linear_id();
      bool is_even = (lane_id & 1) == 0;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        float scaled = static_cast<float>(frg_acc[i]) * static_cast<float>(frg_scale[i]);

        uint32_t my_bits = reinterpret_cast<const uint32_t&>(scaled);
        uint32_t partner_bits = shfl_xor_sync(0xFFFFFFFF, my_bits, 1, 16);
        float my_val = reinterpret_cast<const float&>(my_bits);
        float partner_val = reinterpret_cast<const float&>(partner_bits);

        // cos_sin: convert to float first (handles both float and bf16 inputs),
        // then shuffle the float bits between lanes
        float my_cs = static_cast<float>(frg_cos_sin[i]);
        uint32_t my_cs_bits = reinterpret_cast<const uint32_t&>(my_cs);
        uint32_t partner_cs_bits = shfl_xor_sync(0xFFFFFFFF, my_cs_bits, 1, 16);
        float partner_cs = reinterpret_cast<const float&>(partner_cs_bits);

        float cos_val = is_even ? my_cs : partner_cs;
        float sin_val = is_even ? partner_cs : my_cs;

        float out;
        if (is_even) {
          out = my_val * cos_val + partner_val * sin_val;
        } else {
          out = -partner_val * sin_val + my_val * cos_val;
        }

        result[i] = static_cast<ElementAccumulator>(out);
      }

      return result;
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(cutlass::epilogue::fusion::ConsumerStoreArgs<Args...> const& args) {
    return ConsumerStoreCallbacks{};
  }
};

}  // namespace xe_fuse
