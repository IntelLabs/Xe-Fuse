#pragma once

// Standalone SwiGLU + INT8 requantization kernel.
//
// Reads the BF16 output of a SwiGLU GEMM epilogue (K2/K8), applies SwiGLU
// if the input is the raw gate+up interleaved tensor, and writes:
//   - INT8 quantized output for the downstream W8A8 GEMM (down projection)
//   - per-token scale factors scale_token[m]
//
//
// Two usage modes controlled by ApplySwiGLU:
//   ApplySwiGLU = false (default): input is already post-SwiGLU BF16 [M, I].
//                 Just quantize it to INT8.
//   ApplySwiGLU = true:  input is interleaved gate+up BF16 [M, 2*I].
//                 Apply SwiGLU then quantize; output is [M, I].
//
// Two sub-group passes per row:
//   Pass 1 -- SwiGLU (if ApplySwiGLU) + reduce max_abs -> scale_token
//   Pass 2 -- write INT8 clamped output

#include <cstdint>
#include <sycl/sycl.hpp>

namespace xe_fuse {

template <typename ElementInput, bool ApplySwiGLU = false>
void launch_swiglu_requant(
    sycl::queue&        q,
    ElementInput const* input_ptr,   // [M, N_in]  N_in = I (post-SwiGLU) or 2*I (interleaved)
    int8_t*             quant_out_ptr,
    float*              scale_token_ptr,
    int M,
    int N_in,   // input columns
    int L = 1)
{
  constexpr int SG_SIZE = 16;
  // Output columns: I = N_in when not applying SwiGLU, N_in/2 otherwise
  int N_out     = ApplySwiGLU ? N_in / 2 : N_in;
  int work_groups = M * L;

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
      sycl::nd_range<1>(static_cast<size_t>(work_groups) * SG_SIZE, SG_SIZE),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
        int row  = item.get_group(0);
        int lane = item.get_local_id(0);
        auto sg  = item.get_sub_group();

        // ── Pass 1: reduce max_abs (with optional SwiGLU) ──────────────────
        float max_abs = 0.f;
        for (int col = lane; col < N_out; col += SG_SIZE) {
          float val;
          if constexpr (ApplySwiGLU) {
            float gate = static_cast<float>(input_ptr[row * N_in + col * 2]);
            float up   = static_cast<float>(input_ptr[row * N_in + col * 2 + 1]);
            float silu_gate = gate / (1.0f + sycl::exp(-gate));
            val = silu_gate * up;
          } else {
            val = static_cast<float>(input_ptr[row * N_in + col]);
          }
          max_abs = sycl::fmax(max_abs, sycl::fabs(val));
        }
        for (int off = SG_SIZE / 2; off > 0; off /= 2)
          max_abs = sycl::fmax(max_abs, sycl::shift_group_left(sg, max_abs, off));

        float scale_tok = max_abs / 127.f + 1e-8f;
        scale_tok = sycl::group_broadcast(sg, scale_tok, 0);

        // ── Pass 2: write INT8 output ────────────────────────────────────────
        for (int col = lane; col < N_out; col += SG_SIZE) {
          float val;
          if constexpr (ApplySwiGLU) {
            float gate = static_cast<float>(input_ptr[row * N_in + col * 2]);
            float up   = static_cast<float>(input_ptr[row * N_in + col * 2 + 1]);
            float silu_gate = gate / (1.0f + sycl::exp(-gate));
            val = silu_gate * up;
          } else {
            val = static_cast<float>(input_ptr[row * N_in + col]);
          }
          float qval = sycl::round(val / scale_tok);
          qval = sycl::fmin(sycl::fmax(qval, -128.f), 127.f);
          quant_out_ptr[row * N_out + col] = static_cast<int8_t>(qval);
        }

        if (lane == 0)
          scale_token_ptr[row] = scale_tok;
      }
    );
  });
}

}  // namespace xe_fuse
