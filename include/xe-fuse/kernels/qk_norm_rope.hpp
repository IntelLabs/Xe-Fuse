#pragma once

// Standalone fused QK normalization + RoPE kernel.
//
// Applies per-head RMSNorm followed by RoPE rotation in a single pass,
// eliminating the intermediate write-and-read between separate norm and
// RoPE kernels.  Used for Q and K tensors in attention blocks of models
// that apply per-head normalization (Gemma 3, Chameleon, Qwen2.5-VL,
// FLUX.2 single-block path).
//
// Input layout:  [M, num_heads * head_dim]  (M = batch * seq_len)
// cos_sin layout: [M, head_dim]             (interleaved: even = cos, odd = sin)
// gamma layout:   [head_dim]                (per-dim RMSNorm weight; nullable)
//
// One SYCL workgroup = one (token, head) pair.
// SG_SIZE = 16 lanes; head_dim must be a multiple of 16.
//
// Two passes per workgroup:
//   Pass 1 — reduce sum_sq across head_dim → rstd
//   Pass 2 — normalize (× rstd × gamma), then apply RoPE via shfl_xor

#include <cstdint>
#include <sycl/sycl.hpp>

#include "cutlass/detail/helper_macros.hpp"
#include "cutlass/gpu_generics.h"

namespace xe_fuse {

template <typename ElementInput,
          typename ElementGamma  = float,
          typename ElementCosSin = float>
void launch_qk_norm_rope(
    sycl::queue&         q,
    ElementInput const*  input_ptr,
    ElementInput*        output_ptr,
    ElementGamma const*  gamma_ptr,    // [head_dim]; nullptr = no per-dim scale
    ElementCosSin const* cos_sin_ptr,  // [M, head_dim] interleaved
    int M,
    int num_heads,
    int head_dim,
    float eps = 1e-6f)
{
  constexpr int SG_SIZE = 16;

  int work_groups    = M * num_heads;
  int total_head_dim = num_heads * head_dim;
  int elems_per_lane = head_dim / SG_SIZE;  // e.g. head_dim=128 → 8

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
      sycl::nd_range<1>(
        static_cast<size_t>(work_groups) * SG_SIZE,
        static_cast<size_t>(SG_SIZE)),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
        int wg   = static_cast<int>(item.get_group(0));
        int tok  = wg / num_heads;
        int head = wg % num_heads;
        int lane = static_cast<int>(item.get_local_id(0));

        // Base offset into the full [M, total_head_dim] tensor for this head
        int row_base = tok * total_head_dim + head * head_dim;
        // cos_sin row offset for this token
        int cs_base  = tok * head_dim;

        auto sg = item.get_sub_group();

        // ── Pass 1: reduce sum_sq over head_dim ──────────────────────────
        float sum_sq = 0.f;
        for (int j = 0; j < elems_per_lane; ++j) {
          int d  = lane + j * SG_SIZE;
          float v = static_cast<float>(input_ptr[row_base + d]);
          sum_sq += v * v;
        }
        for (int off = SG_SIZE / 2; off > 0; off /= 2)
          sum_sq += sycl::shift_group_left(sg, sum_sq, off);

        float rstd = sycl::rsqrt(sum_sq / static_cast<float>(head_dim) + eps);
        rstd = sycl::group_broadcast(sg, rstd, 0);

        // ── Pass 2: normalize × gamma, then RoPE via shfl_xor ────────────
        bool is_even = (lane & 1) == 0;

        for (int j = 0; j < elems_per_lane; ++j) {
          int d = lane + j * SG_SIZE;

          float v = static_cast<float>(input_ptr[row_base + d]) * rstd;
          if (gamma_ptr)
            v *= static_cast<float>(gamma_ptr[d]);

          // RoPE: interleave-shuffle with adjacent lane
          uint32_t my_bits      = reinterpret_cast<const uint32_t&>(v);
          uint32_t partner_bits = shfl_xor_sync(0xFFFFFFFF, my_bits, 1, 16);
          float my_val      = reinterpret_cast<const float&>(my_bits);
          float partner_val = reinterpret_cast<const float&>(partner_bits);

          float cs_val = static_cast<float>(cos_sin_ptr[cs_base + d]);
          uint32_t cs_bits         = reinterpret_cast<const uint32_t&>(cs_val);
          uint32_t partner_cs_bits = shfl_xor_sync(0xFFFFFFFF, cs_bits, 1, 16);
          float partner_cs = reinterpret_cast<const float&>(partner_cs_bits);

          float cos_val = is_even ? cs_val     : partner_cs;
          float sin_val = is_even ? partner_cs : cs_val;

          float out;
          if (is_even)
            out =  my_val * cos_val + partner_val * sin_val;
          else
            out = -partner_val * sin_val + my_val * cos_val;

          output_ptr[row_base + d] = static_cast<ElementInput>(out);
        }
      }
    );
  });
}

}  // namespace xe_fuse
