#pragma once

// XeHadamardCompute<GroupSize> — Epilogue visitor for Walsh-Hadamard Transform (WHT).
//
// Applies an in-register WHT of size GroupSize to the output of a child EVT subtree,
// using sub-group shuffles (shfl_xor_sync) across lanes — the same mechanism used by
// XeRoPECompute and XePairwiseCompute.
//
// For GroupSize=16 (= XeGPU sub-group size): 4 butterfly stages, all intra-SG, no SLM.
//   Stage k: shfl_xor(mask = 2^k); bit k of lane_id → a-lane (+) or b-lane (-)
// Normalized by 1/sqrt(GroupSize) so the transform is orthonormal.
//
// Usage in QuaRot W8A8:
//   Wrap the K0 (O-projection) output with HadamardOutput<InnerEVT>  so the BF16
//   activations written to DRAM are already rotated — the next layer's
//   launch_compute_rstd_and_quantize sees rotated values without an extra kernel.
//
// Current support: GroupSize <= 16 (single sub-group). GroupSize > 16 requires
// inter-SG SLM staging and is deferred to a future extension.

#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/gpu_generics.h"

namespace xe_fuse {

template <int GroupSize_ = 16>
struct XeHadamardCompute : cutlass::epilogue::fusion::Sm90VisitorImpl<> {

  static constexpr int GroupSize = GroupSize_;
  static_assert(GroupSize >= 2 && (GroupSize & (GroupSize - 1)) == 0,
                "GroupSize must be a power of 2");
  static_assert(GroupSize <= 16,
                "GroupSize > 16 requires inter-subgroup SLM staging (not yet implemented)");

  // Number of butterfly stages = log2(GroupSize)
  static constexpr int kStages = []() constexpr {
    int s = 0, g = GroupSize;
    while (g > 1) { g >>= 1; ++s; }
    return s;
  }();

  using Sm90VisitorImpl<>::Sm90VisitorImpl;

  struct ConsumerStoreCallbacks : cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks {

    // Takes the output of a child EVT subtree (frg_input) and applies WHT.
    // frg_acc is passed through but not used — same pattern as XePairwiseCompute.
    template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
    CUTLASS_DEVICE cutlass::Array<ElementInput, FragmentSize>
    visit(cutlass::Array<ElementAccumulator, FragmentSize> const& frg_acc,
          int epi_v, int epi_m, int epi_n,
          cutlass::Array<ElementInput, FragmentSize> const& frg_input) {

      cutlass::Array<ElementInput, FragmentSize> result;

      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      uint32_t lane_id = sg.get_local_linear_id();

      // Normalization: 1 / sqrt(GroupSize). Computed as a float constant.
      // For GroupSize=16: 0.25f; GroupSize=8: ~0.354f; GroupSize=4: 0.5f.
      const float kInvSqrtG = sycl::rsqrt(static_cast<float>(GroupSize));

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        float val = static_cast<float>(frg_input[i]);

        // Apply log2(GroupSize) butterfly stages.
        //
        // At stage k (xor_mask = 2^k):
        //   - Each lane exchanges its value with its xor-partner.
        //   - bit k of lane_id == 0 → "a-lane": output = my_val + partner_val
        //   - bit k of lane_id == 1 → "b-lane": output = partner_val - my_val
        //
        // This is the iterative Hadamard butterfly, which matches the Kronecker
        // product definition H_n = H_2 ⊗ H_{n/2}.
        CUTLASS_PRAGMA_UNROLL
        for (int stage = 0; stage < kStages; ++stage) {
          uint32_t mask = 1u << stage;
          uint32_t val_bits = reinterpret_cast<const uint32_t&>(val);
          uint32_t partner_bits = shfl_xor_sync(0xFFFFFFFF, val_bits, mask, GroupSize);
          float partner_val = reinterpret_cast<const float&>(partner_bits);

          bool is_b_lane = (lane_id >> stage) & 1u;
          val = is_b_lane ? (partner_val - val) : (val + partner_val);
        }

        result[i] = static_cast<ElementInput>(val * kInvSqrtG);
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
