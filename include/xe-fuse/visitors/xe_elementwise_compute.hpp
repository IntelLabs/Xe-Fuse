#pragma once

// XeElementwiseCompute<Fn> — single-child unary visitor for element-wise ops.
//
// Applies Fn to each element of the child input. Unlike XeCompute (which is
// designed for binary functors like multiplies/plus), this handles unary
// operations: activations (GeLU, SiLU, ReLU), normalization, etc.
//
// Usage:
//   XeEVT<XeElementwiseCompute<GeLUFn>, SomeChildTree>

#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"

#include <sycl/sycl.hpp>

namespace xe_fuse {

template <class ElementwiseFn>
struct XeElementwiseCompute : cutlass::epilogue::fusion::Sm90VisitorImpl<> {

  using Sm90VisitorImpl<>::Sm90VisitorImpl;

  struct ConsumerStoreCallbacks : cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks {

    template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
    CUTLASS_DEVICE cutlass::Array<ElementInput, FragmentSize>
    visit(cutlass::Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          cutlass::Array<ElementInput, FragmentSize> const& frg_input) {

      cutlass::Array<ElementInput, FragmentSize> result;
      ElementwiseFn fn{};

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        result[i] = static_cast<ElementInput>(fn(static_cast<float>(frg_input[i])));
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

// ============================================================
// Activation Functors
// ============================================================

struct GeLUFn {
  CUTLASS_DEVICE float operator()(float x) const {
    return x * 0.5f * (1.0f + sycl::erf(x * 0.7071067811865475f));
  }
};

struct GeLUTanhFn {
  CUTLASS_DEVICE float operator()(float x) const {
    constexpr float k = 0.7978845608028654f;  // sqrt(2/pi)
    float x3 = x * x * x;
    return x * 0.5f * (1.0f + sycl::tanh(k * (x + 0.044715f * x3)));
  }
};

struct SiLUFn {
  CUTLASS_DEVICE float operator()(float x) const {
    return x / (1.0f + sycl::exp(-x));
  }
};

struct ReLUFn {
  CUTLASS_DEVICE float operator()(float x) const {
    return x > 0.0f ? x : 0.0f;
  }
};

struct SigmoidFn {
  CUTLASS_DEVICE float operator()(float x) const {
    return 1.0f / (1.0f + sycl::exp(-x));
  }
};

struct ExpFn {
  CUTLASS_DEVICE float operator()(float x) const {
    return sycl::exp(x);
  }
};

}  // namespace xe_fuse
