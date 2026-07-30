#pragma once

// Merged visitors that flatten ScaleRows-based EVT trees.
//
// Instead of XeEVT<MulCompute, AccFetch, ColBroadcast> (3 nodes, intermediate
// Array temporaries), these visitors read frg_acc directly and multiply by the
// scale in-line, eliminating tree-walker overhead and reducing register pressure.
//
// XeScaleRowsCompute       — D = acc * R[m]           (replaces K1 composed tree)
// XeScaleRowsSwiGLUCompute — D = SwiGLU(acc * R[m])   (replaces K2 composed tree)
// XeScaleRowsGeGLUCompute  — D = GeGLU(acc * R[m])    (replaces K2_geglu composed tree)

#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/gpu_generics.h"

namespace xe_fuse {

// D = acc * scale[m]
// Takes frg_acc directly + one child (ColBroadcast for per-row scale).
struct XeScaleRowsCompute : cutlass::epilogue::fusion::Sm90VisitorImpl<> {

  using Sm90VisitorImpl<>::Sm90VisitorImpl;

  struct ConsumerStoreCallbacks : cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks {

    template <typename ElementAccumulator, typename ElementScale, int FragmentSize>
    CUTLASS_DEVICE cutlass::Array<ElementAccumulator, FragmentSize>
    visit(cutlass::Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          cutlass::Array<ElementScale, FragmentSize> const& frg_scale) {

      cutlass::Array<ElementAccumulator, FragmentSize> result;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        result[i] = static_cast<ElementAccumulator>(
            static_cast<float>(frg_acc[i]) * static_cast<float>(frg_scale[i]));
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

// D = SwiGLU(acc * scale[m])
// Merged scale + pairwise SwiGLU in a single visit(). Takes frg_acc directly
// + one child (ColBroadcast for per-row scale). Scales, shuffles with partner
// lane, then applies silu(gate) * up.
struct XeScaleRowsSwiGLUCompute : cutlass::epilogue::fusion::Sm90VisitorImpl<> {

  using Sm90VisitorImpl<>::Sm90VisitorImpl;

  struct ConsumerStoreCallbacks : cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks {

    template <typename ElementAccumulator, typename ElementScale, int FragmentSize>
    CUTLASS_DEVICE cutlass::Array<ElementAccumulator, FragmentSize>
    visit(cutlass::Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          cutlass::Array<ElementScale, FragmentSize> const& frg_scale) {

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

        float gate = is_even ? my_val : partner_val;
        float up   = is_even ? partner_val : my_val;
        float silu_gate = gate / (1.0f + sycl::exp(-gate));

        result[i] = static_cast<ElementAccumulator>(silu_gate * up);
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

// D = GeGLU(acc * scale[m])
// Same as SwiGLU variant but uses GeLU activation: gelu(gate) * up.
struct XeScaleRowsGeGLUCompute : cutlass::epilogue::fusion::Sm90VisitorImpl<> {

  using Sm90VisitorImpl<>::Sm90VisitorImpl;

  struct ConsumerStoreCallbacks : cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks {

    template <typename ElementAccumulator, typename ElementScale, int FragmentSize>
    CUTLASS_DEVICE cutlass::Array<ElementAccumulator, FragmentSize>
    visit(cutlass::Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
          cutlass::Array<ElementScale, FragmentSize> const& frg_scale) {

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

        float gate = is_even ? my_val : partner_val;
        float up   = is_even ? partner_val : my_val;
        float gelu_gate = gate * 0.5f * (1.0f + sycl::erf(gate * 0.7071067811865475f));

        result[i] = static_cast<ElementAccumulator>(gelu_gate * up);
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
