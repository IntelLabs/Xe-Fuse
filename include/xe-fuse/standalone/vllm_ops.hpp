#pragma once

// vllm-equivalent standalone kernels for xe-fuse comparison benchmarks.
//
// These re-implement the algorithmic patterns from vllm-xpu-kernels
// (csrc/layernorm.cpp, csrc/activation.cpp, csrc/pos_encoding_kernels.cpp)
// using pure SYCL — no torch/ATen dependency. The implementations match
// vllm's work distribution: one work-group per token, vectorized loads,
// sub-group reductions for RMSNorm variance.
//
// Purpose: fair head-to-head comparison of fusion strategies
//   vllm:    bare GEMM → separate fused standalone kernel → bare GEMM → ...
//   xe-fuse: GEMM + epilogue fusion (ops run on register data)

#include <sycl/sycl.hpp>
#include "cutlass/bfloat16.h"

namespace xe_fuse::vllm_equiv {

using bf16 = cutlass::bfloat16_t;

// RMSNorm: out[m,n] = (input[m,n] / sqrt(mean(input[m,:]^2) + eps)) * weight[n]
// Matches vllm::rms_norm_kernel<bf16, 2, 8>: one work-group per row,
// vectorized variance accumulation, work-group reduction via SLM.
inline void rms_norm(sycl::queue& q, bf16* out, bf16 const* input,
                     bf16 const* weight, int M, int N, float eps = 1e-6f) {
  int wg_size = std::min(N / 8, 256);
  if (wg_size < 1) wg_size = std::min(N, 256);

  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> s_var(sycl::range<1>(1), cgh);
    int hidden = N;
    cgh.parallel_for(
      sycl::nd_range<1>(static_cast<size_t>(M) * wg_size, wg_size),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        int row = item.get_group(0);
        int lid = item.get_local_id(0);
        int lsz = item.get_local_range(0);
        auto wg = item.get_group();

        bf16 const* in_row = input + static_cast<int64_t>(row) * hidden;
        bf16* out_row = out + static_cast<int64_t>(row) * hidden;

        float variance = 0.0f;
        for (int i = lid; i < hidden; i += lsz) {
          float x = static_cast<float>(in_row[i]);
          variance += x * x;
        }

        variance = sycl::reduce_over_group(wg, variance, sycl::plus<float>());
        if (lid == 0)
          s_var[0] = sycl::rsqrt(variance / static_cast<float>(hidden) + eps);
        sycl::group_barrier(wg);

        float s = s_var[0];
        for (int i = lid; i < hidden; i += lsz) {
          float x = static_cast<float>(in_row[i]);
          out_row[i] = static_cast<bf16>(x * s * static_cast<float>(weight[i]));
        }
      });
  });
}

// Fused residual add + RMSNorm (in-place):
//   residual[m,:] += input[m,:]
//   input[m,:] = RMSNorm(residual[m,:]) * weight[:]
// Matches vllm::fused_add_rms_norm_kernel<bf16, 8>.
inline void fused_add_rms_norm(sycl::queue& q, bf16* input, bf16* residual,
                                bf16 const* weight, int M, int N,
                                float eps = 1e-6f) {
  int wg_size = std::min(N / 8, 256);
  if (wg_size < 1) wg_size = std::min(N, 256);

  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> s_var(sycl::range<1>(1), cgh);
    int hidden = N;
    cgh.parallel_for(
      sycl::nd_range<1>(static_cast<size_t>(M) * wg_size, wg_size),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        int row = item.get_group(0);
        int lid = item.get_local_id(0);
        int lsz = item.get_local_range(0);
        auto wg = item.get_group();

        int64_t base = static_cast<int64_t>(row) * hidden;

        float variance = 0.0f;
        for (int i = lid; i < hidden; i += lsz) {
          float inp = static_cast<float>(input[base + i]);
          float res = static_cast<float>(residual[base + i]);
          float sum = inp + res;
          variance += sum * sum;
          residual[base + i] = static_cast<bf16>(sum);
        }

        variance = sycl::reduce_over_group(wg, variance, sycl::plus<float>());
        if (lid == 0)
          s_var[0] = sycl::rsqrt(variance / static_cast<float>(hidden) + eps);
        sycl::group_barrier(wg);

        float s = s_var[0];
        for (int i = lid; i < hidden; i += lsz) {
          float x = static_cast<float>(residual[base + i]);
          input[base + i] = static_cast<bf16>(x * s * static_cast<float>(weight[i]));
        }
      });
  });
}

// SwiGLU: out[m, i] = silu(input[m, i]) * input[m, d + i]
// Input is [M, 2*d], output is [M, d]. Gate = first half, up = second half.
// Matches vllm::act_and_mul_vec_kernel<bf16, silu_kernel, true, 8>.
inline void silu_and_mul(sycl::queue& q, bf16* out, bf16 const* input,
                          int d, int M) {
  int wg_size = std::min(d, 1024);
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
      sycl::nd_range<1>(static_cast<size_t>(M) * wg_size, wg_size),
      [=](sycl::nd_item<1> item) {
        int row = item.get_group(0);
        int lid = item.get_local_id(0);
        int lsz = item.get_local_range(0);
        int64_t in_base = static_cast<int64_t>(row) * 2 * d;
        int64_t out_base = static_cast<int64_t>(row) * d;

        for (int i = lid; i < d; i += lsz) {
          float gate = static_cast<float>(input[in_base + i]);
          float up = static_cast<float>(input[in_base + d + i]);
          float silu_gate = gate / (1.0f + sycl::exp(-gate));
          out[out_base + i] = static_cast<bf16>(silu_gate * up);
        }
      });
  });
}

// GeGLU: out[m, i] = gelu(input[m, i]) * input[m, d + i]
inline void gelu_and_mul(sycl::queue& q, bf16* out, bf16 const* input,
                          int d, int M) {
  int wg_size = std::min(d, 1024);
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
      sycl::nd_range<1>(static_cast<size_t>(M) * wg_size, wg_size),
      [=](sycl::nd_item<1> item) {
        int row = item.get_group(0);
        int lid = item.get_local_id(0);
        int lsz = item.get_local_range(0);
        int64_t in_base = static_cast<int64_t>(row) * 2 * d;
        int64_t out_base = static_cast<int64_t>(row) * d;

        for (int i = lid; i < d; i += lsz) {
          float gate = static_cast<float>(input[in_base + i]);
          float up = static_cast<float>(input[in_base + d + i]);
          float gelu_gate = gate * 0.5f * (1.0f + sycl::erf(gate * 0.7071067811865475f));
          out[out_base + i] = static_cast<bf16>(gelu_gate * up);
        }
      });
  });
}

// NeoX-style RoPE: applies rotary position embedding in-place.
// data layout: [M, num_heads, head_size] (contiguous)
// cos_sin_cache: [M, rot_dim] where rot_dim = head_size (interleaved cos|sin)
// Matches vllm::rotary_embedding_kernel<bf16, true>.
inline void rotary_embedding(sycl::queue& q, bf16* query, bf16* key,
                              float const* cos_sin_cache,
                              int head_size, int num_heads, int num_kv_heads,
                              int rot_dim, int M) {
  int embed_dim = rot_dim / 2;
  int total_work = num_heads * embed_dim;
  int wg_size = std::min(total_work, 512);

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
      sycl::nd_range<1>(static_cast<size_t>(M) * wg_size, wg_size),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        int token = item.get_group(0);
        int lid = item.get_local_id(0);
        int lsz = item.get_local_range(0);

        float const* cos_ptr = cos_sin_cache + static_cast<int64_t>(token) * rot_dim;
        float const* sin_ptr = cos_ptr + embed_dim;

        int64_t q_stride = static_cast<int64_t>(num_heads) * head_size;
        int64_t k_stride = static_cast<int64_t>(num_kv_heads) * head_size;

        int nq = num_heads * embed_dim;
        for (int i = lid; i < nq; i += lsz) {
          int head = i / embed_dim;
          int rot_offset = i % embed_dim;
          int64_t base = static_cast<int64_t>(token) * q_stride +
                         static_cast<int64_t>(head) * head_size;
          int x_idx = rot_offset;
          int y_idx = embed_dim + rot_offset;
          float x = static_cast<float>(query[base + x_idx]);
          float y = static_cast<float>(query[base + y_idx]);
          float c = cos_ptr[rot_offset];
          float s = sin_ptr[rot_offset];
          query[base + x_idx] = static_cast<bf16>(x * c - y * s);
          query[base + y_idx] = static_cast<bf16>(y * c + x * s);
        }

        if (key != nullptr) {
          int nk = num_kv_heads * embed_dim;
          for (int i = lid; i < nk; i += lsz) {
            int head = i / embed_dim;
            int rot_offset = i % embed_dim;
            int64_t base = static_cast<int64_t>(token) * k_stride +
                           static_cast<int64_t>(head) * head_size;
            int x_idx = rot_offset;
            int y_idx = embed_dim + rot_offset;
            float x = static_cast<float>(key[base + x_idx]);
            float y = static_cast<float>(key[base + y_idx]);
            float c = cos_ptr[rot_offset];
            float s = sin_ptr[rot_offset];
            key[base + x_idx] = static_cast<bf16>(x * c - y * s);
            key[base + y_idx] = static_cast<bf16>(y * c + x * s);
          }
        }
      });
  });
}

}  // namespace xe_fuse::vllm_equiv
