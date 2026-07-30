#pragma once

// xe-fuse standalone ops: element-wise SYCL kernels that mirror the
// GEMM epilogue operations. These serve two purposes:
//
// 1. Unfused baselines for comparison against GEMM+epilogue fusion
// 2. Building blocks for non-GEMM kernel composition
//
// Each function operates in-place on an M x N x L tensor (bf16).
// Naming mirrors the builder API:
//
//   Builder (GEMM epilogue)          Standalone (separate kernel)
//   ─────────────────────────────    ──────────────────────────────
//   b::ScaleRows<Acc, TS, float>    standalone::scale_rows(...)
//   b::ScaleCols<Input, TS, float>  standalone::scale_cols(...)
//   b::AddResidual<bf16>            standalone::add_residual(...)
//   b::SwiGLU<Input>                standalone::swiglu(...)
//   b::RoPE<float>                  standalone::rope(...)
//   b::RoPEScaled<TS, float>        standalone::rope_scaled(...)
//
// The LLM agent can compare fused vs unfused by benchmarking both paths
// and computing the memory bandwidth saved by fusion.

#include <sycl/sycl.hpp>
#include "cutlass/bfloat16.h"

namespace xe_fuse::standalone {

using bf16 = cutlass::bfloat16_t;

// D[m,n] *= scale[m]  (per-row scaling, e.g., RMSNorm)
inline void scale_rows(sycl::queue& q, bf16* data, float const* scale,
                        int M, int N, int L) {
  int64_t total = static_cast<int64_t>(M) * N * L;
  int n_val = N, m_val = M;
  q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
    int64_t i = idx[0];
    int m = static_cast<int>((i / n_val) % m_val);
    int batch = static_cast<int>(i / (static_cast<int64_t>(m_val) * n_val));
    data[i] = static_cast<bf16>(static_cast<float>(data[i]) * scale[batch * m_val + m]);
  });
}

// D[m,n] *= gamma[n]  (per-column scaling)
inline void scale_cols(sycl::queue& q, bf16* data, float const* gamma,
                        int M, int N, int L) {
  int64_t total = static_cast<int64_t>(M) * N * L;
  int n_val = N;
  q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
    int64_t i = idx[0];
    int col = static_cast<int>(i % n_val);
    data[i] = static_cast<bf16>(static_cast<float>(data[i]) * gamma[col]);
  });
}

// D[m,n] += residual[m,n]
inline void add_residual(sycl::queue& q, bf16* data, bf16 const* residual,
                          int M, int N, int L) {
  int64_t total = static_cast<int64_t>(M) * N * L;
  q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
    int64_t i = idx[0];
    data[i] = static_cast<bf16>(static_cast<float>(data[i]) + static_cast<float>(residual[i]));
  });
}

// D[m,n] = gamma[n] * (D[m,n] + residual[m,n])
inline void residual_gamma(sycl::queue& q, bf16* data, bf16 const* residual,
                            float const* gamma, int M, int N, int L) {
  int64_t total = static_cast<int64_t>(M) * N * L;
  int n_val = N;
  q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
    int64_t i = idx[0];
    int col = static_cast<int>(i % n_val);
    float val = static_cast<float>(data[i]) + static_cast<float>(residual[i]);
    data[i] = static_cast<bf16>(val * gamma[col]);
  });
}

// SwiGLU on adjacent column pairs: output = silu(gate) * up
inline void swiglu(sycl::queue& q, bf16* data, int M, int N, int L) {
  int64_t total = static_cast<int64_t>(M) * N * L;
  int n_val = N;
  q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
    int64_t i = idx[0];
    int col = static_cast<int>(i % n_val);
    int64_t row_base = i - col;
    int even_col = col & ~1;
    int odd_col = even_col + 1;
    if (odd_col < n_val) {
      float gate = static_cast<float>(data[row_base + even_col]);
      float up = static_cast<float>(data[row_base + odd_col]);
      float silu_gate = gate / (1.0f + sycl::exp(-gate));
      data[i] = static_cast<bf16>(silu_gate * up);
    }
  });
}

// GeGLU on adjacent column pairs: output = gelu(gate) * up
inline void geglu(sycl::queue& q, bf16* data, int M, int N, int L) {
  int64_t total = static_cast<int64_t>(M) * N * L;
  int n_val = N;
  q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
    int64_t i = idx[0];
    int col = static_cast<int>(i % n_val);
    int64_t row_base = i - col;
    int even_col = col & ~1;
    int odd_col = even_col + 1;
    if (odd_col < n_val) {
      float gate = static_cast<float>(data[row_base + even_col]);
      float up = static_cast<float>(data[row_base + odd_col]);
      float gelu_gate = gate * 0.5f * (1.0f + sycl::erf(gate * 0.7071067811865475f));
      data[i] = static_cast<bf16>(gelu_gate * up);
    }
  });
}

// RoPE rotation on adjacent column pairs using cos_sin tensor
// Requires tmp buffer (same size as data) for read-only source
inline void rope(sycl::queue& q, bf16* data, bf16 const* tmp,
                  float const* cos_sin, int M, int N, int L) {
  int64_t total = static_cast<int64_t>(M) * N * L;
  int n_val = N;
  q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
    int64_t i = idx[0];
    int col = static_cast<int>(i % n_val);
    int64_t row_base = i - col;
    int even_col = col & ~1;
    int odd_col = even_col + 1;
    if (odd_col < n_val) {
      float x_even = static_cast<float>(tmp[row_base + even_col]);
      float x_odd = static_cast<float>(tmp[row_base + odd_col]);
      float cos_val = cos_sin[row_base + even_col];
      float sin_val = cos_sin[row_base + odd_col];
      float out;
      if ((col & 1) == 0)
        out = x_even * cos_val + x_odd * sin_val;
      else
        out = -x_even * sin_val + x_odd * cos_val;
      data[i] = static_cast<bf16>(out);
    }
  });
}

// Merged: D[m,n] = RoPE(D[m,n] * scale[m], cos_sin[m,n])
// Requires tmp buffer for read-only source
inline void rope_scaled(sycl::queue& q, bf16* data, bf16 const* tmp,
                         float const* scale, float const* cos_sin,
                         int M, int N, int L) {
  int64_t total = static_cast<int64_t>(M) * N * L;
  int n_val = N, m_val = M;
  q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
    int64_t i = idx[0];
    int col = static_cast<int>(i % n_val);
    int row = static_cast<int>((i / n_val) % m_val);
    int batch = static_cast<int>(i / (static_cast<int64_t>(m_val) * n_val));
    int64_t row_base = i - col;

    float s = scale[batch * m_val + row];
    int even_col = col & ~1;
    int odd_col = even_col + 1;
    if (odd_col < n_val) {
      float x_even = static_cast<float>(tmp[row_base + even_col]) * s;
      float x_odd = static_cast<float>(tmp[row_base + odd_col]) * s;
      float cos_val = cos_sin[row_base + even_col];
      float sin_val = cos_sin[row_base + odd_col];
      float out;
      if ((col & 1) == 0)
        out = x_even * cos_val + x_odd * sin_val;
      else
        out = -x_even * sin_val + x_odd * cos_val;
      data[i] = static_cast<bf16>(out);
    } else {
      data[i] = static_cast<bf16>(static_cast<float>(tmp[i]) * s);
    }
  });
}

// D[m,n] = GeLU(D[m,n])  — exact: x * 0.5 * (1 + erf(x / sqrt(2)))
inline void gelu(sycl::queue& q, bf16* data, int M, int N, int L) {
  int64_t total = static_cast<int64_t>(M) * N * L;
  q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
    int64_t i = idx[0];
    float x = static_cast<float>(data[i]);
    data[i] = static_cast<bf16>(x * 0.5f * (1.0f + sycl::erf(x * 0.7071067811865475f)));
  });
}

// D[m,n] = GeLU_tanh(D[m,n])  — tanh approximation
inline void gelu_tanh(sycl::queue& q, bf16* data, int M, int N, int L) {
  int64_t total = static_cast<int64_t>(M) * N * L;
  constexpr float k = 0.7978845608028654f;
  q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
    int64_t i = idx[0];
    float x = static_cast<float>(data[i]);
    float x3 = x * x * x;
    data[i] = static_cast<bf16>(x * 0.5f * (1.0f + sycl::tanh(k * (x + 0.044715f * x3))));
  });
}

// D[m,n] = SiLU(D[m,n])  — x * sigmoid(x), a.k.a. Swish
inline void silu(sycl::queue& q, bf16* data, int M, int N, int L) {
  int64_t total = static_cast<int64_t>(M) * N * L;
  q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
    int64_t i = idx[0];
    float x = static_cast<float>(data[i]);
    data[i] = static_cast<bf16>(x / (1.0f + sycl::exp(-x)));
  });
}

// D[m,n] = ReLU(D[m,n])
inline void relu(sycl::queue& q, bf16* data, int M, int N, int L) {
  int64_t total = static_cast<int64_t>(M) * N * L;
  q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
    int64_t i = idx[0];
    float x = static_cast<float>(data[i]);
    data[i] = static_cast<bf16>(x > 0.0f ? x : 0.0f);
  });
}

// D[m,n] += bias[n]  — per-column bias addition
inline void bias_add(sycl::queue& q, bf16* data, float const* bias,
                      int M, int N, int L) {
  int64_t total = static_cast<int64_t>(M) * N * L;
  int n_val = N;
  q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
    int64_t i = idx[0];
    int col = static_cast<int>(i % n_val);
    data[i] = static_cast<bf16>(static_cast<float>(data[i]) + bias[col]);
  });
}

// Compute reciprocal standard deviation per row:
//   rstd[m] = rsqrt(mean_n(x[m,n]^2) + eps)
inline void compute_rstd(sycl::queue& q, float* rstd, bf16 const* input,
                          int M, int N, int L, float eps = 1e-6f) {
  int m_val = M, n_val = N;
  q.parallel_for(sycl::range<1>(static_cast<size_t>(M) * L), [=](sycl::id<1> idx) {
    int64_t row_idx = idx[0];
    int batch = static_cast<int>(row_idx / m_val);
    int m = static_cast<int>(row_idx % m_val);
    int64_t base = static_cast<int64_t>(batch) * m_val * n_val + static_cast<int64_t>(m) * n_val;

    float sum_sq = 0.0f;
    for (int n = 0; n < n_val; ++n) {
      float v = static_cast<float>(input[base + n]);
      sum_sq += v * v;
    }
    rstd[row_idx] = sycl::rsqrt(sum_sq / static_cast<float>(n_val) + eps);
  });
}

// lse[m] = log(sum_n exp(logits[m,n]))  — numerically stable (two-pass)
inline void compute_lse(sycl::queue& q, bf16 const* logits, float* lse,
                         int M, int N, int L) {
  int m_val = M, n_val = N;
  q.parallel_for(sycl::range<1>(static_cast<size_t>(M) * L), [=](sycl::id<1> idx) {
    int64_t row = idx[0];
    int batch = static_cast<int>(row / m_val);
    int m     = static_cast<int>(row % m_val);
    int64_t base = static_cast<int64_t>(batch) * m_val * n_val +
                   static_cast<int64_t>(m) * n_val;
    float max_val = -3.4028235e+38f;
    for (int n = 0; n < n_val; ++n)
      max_val = sycl::fmax(max_val, static_cast<float>(logits[base + n]));
    float sum_exp = 0.0f;
    for (int n = 0; n < n_val; ++n)
      sum_exp += sycl::exp(static_cast<float>(logits[base + n]) - max_val);
    lse[row] = max_val + sycl::log(sum_exp);
  });
}

// logits_tgt[m] = logits[m, targets[m]]
inline void select_logits(sycl::queue& q, bf16 const* logits, int const* targets,
                           float* logits_tgt, int M, int N, int L) {
  int m_val = M, n_val = N;
  q.parallel_for(sycl::range<1>(static_cast<size_t>(M) * L), [=](sycl::id<1> idx) {
    int64_t row = idx[0];
    int batch = static_cast<int>(row / m_val);
    int m     = static_cast<int>(row % m_val);
    int64_t base = static_cast<int64_t>(batch) * m_val * n_val +
                   static_cast<int64_t>(m) * n_val;
    logits_tgt[row] = static_cast<float>(logits[base + targets[row]]);
  });
}

}  // namespace xe_fuse::standalone
