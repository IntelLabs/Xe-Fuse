#pragma once

// XePairwiseCompute<PairwiseFn> — Generic epilogue visitor for pairwise
// operations on N-adjacent column pairs via sub-group shuffle.
//
// Xe DPAS CLayout: Layout<Shape<_16, _M>, Stride<_M, _1>>
// Each of 16 sub-group lanes owns one N column. shuffle_xor(val, 1)
// exchanges between lanes (2k, 2k+1), which are N-adjacent column pairs.
//
// PairwiseFn must provide:
//   CUTLASS_DEVICE float operator()(float my_val, float partner_val, bool is_even)
//
// The visitor receives one child input and applies the pairwise function.

#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/gpu_generics.h"

namespace xe_fuse {

template <typename PairwiseFn_>
struct XePairwiseCompute : cutlass::epilogue::fusion::Sm90VisitorImpl<> {

  using PairwiseFn = PairwiseFn_;
  using Sm90VisitorImpl<>::Sm90VisitorImpl;

  struct ConsumerStoreCallbacks : cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks {

    template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
    CUTLASS_DEVICE cutlass::Array<ElementInput, FragmentSize>
    visit(cutlass::Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          cutlass::Array<ElementInput, FragmentSize> const& frg_input) {
      cutlass::Array<ElementInput, FragmentSize> result;

      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      uint32_t lane_id = sg.get_local_linear_id();
      bool is_even = (lane_id & 1) == 0;

      PairwiseFn fn{};

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        uint32_t my_bits = reinterpret_cast<const uint32_t&>(frg_input[i]);
        uint32_t partner_bits = shfl_xor_sync(0xFFFFFFFF, my_bits, 1, 16);
        float my_val = reinterpret_cast<const float&>(my_bits);
        float partner_val = reinterpret_cast<const float&>(partner_bits);

        float out = fn(my_val, partner_val, is_even);
        result[i] = static_cast<ElementInput>(out);
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

// Pairwise swap: D[m, 2k] <-> D[m, 2k+1]
struct PairwiseSwapFn {
  CUTLASS_DEVICE float operator()(float my_val, float partner_val, bool) const {
    return partner_val;
  }
};

// SwiGLU: given interleaved (gate, up) pairs, compute silu(gate) * up
// Both even and odd lanes produce the same result; only even lanes should store.
struct SwiGLUFn {
  CUTLASS_DEVICE float operator()(float my_val, float partner_val, bool is_even) const {
    float gate = is_even ? my_val : partner_val;
    float up   = is_even ? partner_val : my_val;
    float silu_gate = gate / (1.0f + sycl::exp(-gate));
    return silu_gate * up;
  }
};

// GeGLU: given interleaved (gate, up) pairs, compute gelu(gate) * up
// Same structure as SwiGLU but with GeLU activation instead of SiLU.
struct GeGLUFn {
  CUTLASS_DEVICE float operator()(float my_val, float partner_val, bool is_even) const {
    float gate = is_even ? my_val : partner_val;
    float up   = is_even ? partner_val : my_val;
    float gelu_gate = gate * 0.5f * (1.0f + sycl::erf(gate * 0.7071067811865475f));
    return gelu_gate * up;
  }
};

// RoPE rotation: given interleaved (x_even, x_odd) and external (cos, sin),
// compute: even = x_even * cos + x_odd * sin
//          odd  = -x_even * sin + x_odd * cos
// Note: cos/sin must be passed in via the functor or a separate visitor.
// This is a partial implementation — the full K3 will compose this with AuxLoad.
struct RoPEFn {
  float cos_val;
  float sin_val;

  CUTLASS_DEVICE float operator()(float my_val, float partner_val, bool is_even) const {
    if (is_even) {
      return my_val * cos_val + partner_val * sin_val;
    } else {
      return -partner_val * sin_val + my_val * cos_val;
    }
  }
};

}  // namespace xe_fuse
