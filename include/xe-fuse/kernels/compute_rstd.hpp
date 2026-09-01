#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

#include "cutlass/bfloat16.h"

namespace xe_fuse {

// Standalone rstd reduction kernel.
// Computes R[m] = 1 / sqrt( mean_n(X[m,n]^2) + eps ) for each row.
// Uses sub-group reduction across N.

template <typename ElementInput, typename ElementOutput = float>
void launch_compute_rstd(
    sycl::queue& q,
    ElementInput const* input_ptr,
    ElementOutput* rstd_ptr,
    int M, int N, int L,
    float eps = 1e-6f)
{
  constexpr int SG_SIZE = 16;
  int work_groups = M * L;

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
      sycl::nd_range<1>(work_groups * SG_SIZE, SG_SIZE),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
        int row = item.get_group(0);
        int lane = item.get_local_id(0);

        float sum_sq = 0.0f;
        for (int col = lane; col < N; col += SG_SIZE) {
          float val = static_cast<float>(input_ptr[row * N + col]);
          sum_sq += val * val;
        }

        auto sg = item.get_sub_group();
        for (int offset = SG_SIZE / 2; offset > 0; offset /= 2) {
          sum_sq += sycl::shift_group_left(sg, sum_sq, offset);
        }

        if (lane == 0) {
          float mean_sq = sum_sq / static_cast<float>(N);
          rstd_ptr[row] = static_cast<ElementOutput>(
              1.0f / sycl::sqrt(mean_sq + eps));
        }
      }
    );
  });
}

// Combined RMSNorm + INT8 quantization kernel.
//
// Computes per-row:
//   rstd[m]         = rsqrt( mean_n(X[m,n]^2) + eps )
//   normed[m,n]     = X[m,n] * rstd[m]
//   scale_token[m]  = max_n(|normed[m,n]|) / 127   (per-token quant scale)
//   quant_out[m,n]  = round(normed[m,n] / scale_token[m])  clamped to [-128, 127]
//
// The combined scale_token[m] encodes both the RMSNorm reciprocal std and the
// per-token quantization range.  The W8A8 GEMM epilogue uses this combined scale
// directly via ColBroadcast, so no separate RMSNorm multiply is needed.
//
// Three sub-group passes per row:
//   Pass 1 — reduce sum_sq  → compute rstd
//   Pass 2 — reduce max_abs of (X * rstd) → compute scale_token
//   Pass 3 — write clamped INT8 values and scale_token[m]
template <typename ElementInput>
void launch_compute_rstd_and_quantize(
    sycl::queue& q,
    ElementInput const* input_ptr,
    int8_t*             quant_out_ptr,
    float*              scale_token_ptr,
    int M, int N, int L,
    float eps = 1e-6f)
{
  constexpr int SG_SIZE = 16;
  int work_groups = M * L;

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
      sycl::nd_range<1>(static_cast<size_t>(work_groups) * SG_SIZE, SG_SIZE),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
        int row  = item.get_group(0);
        int lane = item.get_local_id(0);

        // ── Pass 1: reduce sum_sq ──────────────────────────────────────────
        float sum_sq = 0.f;
        for (int col = lane; col < N; col += SG_SIZE) {
          float v = static_cast<float>(input_ptr[row * N + col]);
          sum_sq += v * v;
        }
        auto sg = item.get_sub_group();
        for (int off = SG_SIZE / 2; off > 0; off /= 2)
          sum_sq += sycl::shift_group_left(sg, sum_sq, off);

        float rstd = sycl::rsqrt(sum_sq / static_cast<float>(N) + eps);
        // Broadcast rstd to all lanes via group_broadcast
        rstd = sycl::group_broadcast(sg, rstd, 0);

        // ── Pass 2: reduce max_abs of normalized values ────────────────────
        float max_abs = 0.f;
        for (int col = lane; col < N; col += SG_SIZE) {
          float normed = static_cast<float>(input_ptr[row * N + col]) * rstd;
          max_abs = sycl::fmax(max_abs, sycl::fabs(normed));
        }
        for (int off = SG_SIZE / 2; off > 0; off /= 2)
          max_abs = sycl::fmax(max_abs, sycl::shift_group_left(sg, max_abs, off));

        float scale_tok = max_abs / 127.f + 1e-8f;  // epsilon guards against all-zero rows
        scale_tok = sycl::group_broadcast(sg, scale_tok, 0);

        // ── Pass 3: quantize and write outputs ─────────────────────────────
        for (int col = lane; col < N; col += SG_SIZE) {
          float normed = static_cast<float>(input_ptr[row * N + col]) * rstd;
          float qval   = sycl::round(normed / scale_tok);
          qval = sycl::fmin(sycl::fmax(qval, -128.f), 127.f);
          quant_out_ptr[row * N + col] = static_cast<int8_t>(qval);
        }

        if (lane == 0)
          scale_token_ptr[row] = scale_tok;
      }
    );
  });
}

// Dual-output RMSNorm + INT8 quantization kernel.
//
// Same three-pass algorithm as launch_compute_rstd_and_quantize but also
// writes a BF16 normed output for the residual path.  Use when both a normed
// BF16 value (residual path) and an INT8 quantized value (next GEMM input)
// are needed from the same input.
template <typename ElementInput, typename ElementNormed = cutlass::bfloat16_t>
void launch_norm_quantize_dual(
    sycl::queue&        q,
    ElementInput const* input_ptr,
    int8_t*             quant_out_ptr,
    ElementNormed*      normed_out_ptr,
    float*              scale_token_ptr,
    int M, int N, int L,
    float eps = 1e-6f)
{
  constexpr int SG_SIZE = 16;
  int work_groups = M * L;

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
      sycl::nd_range<1>(static_cast<size_t>(work_groups) * SG_SIZE, SG_SIZE),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
        int row  = item.get_group(0);
        int lane = item.get_local_id(0);

        // ── Pass 1: sum_sq → rstd ──────────────────────────────────────────
        float sum_sq = 0.f;
        for (int col = lane; col < N; col += SG_SIZE) {
          float v = static_cast<float>(input_ptr[row * N + col]);
          sum_sq += v * v;
        }
        auto sg = item.get_sub_group();
        for (int off = SG_SIZE / 2; off > 0; off /= 2)
          sum_sq += sycl::shift_group_left(sg, sum_sq, off);

        float rstd = sycl::rsqrt(sum_sq / static_cast<float>(N) + eps);
        rstd = sycl::group_broadcast(sg, rstd, 0);

        // ── Pass 2: max_abs of normalized values ──────────────────────────
        float max_abs = 0.f;
        for (int col = lane; col < N; col += SG_SIZE) {
          float normed = static_cast<float>(input_ptr[row * N + col]) * rstd;
          max_abs = sycl::fmax(max_abs, sycl::fabs(normed));
        }
        for (int off = SG_SIZE / 2; off > 0; off /= 2)
          max_abs = sycl::fmax(max_abs, sycl::shift_group_left(sg, max_abs, off));

        float scale_tok = max_abs / 127.f + 1e-8f;
        scale_tok = sycl::group_broadcast(sg, scale_tok, 0);

        // ── Pass 3: write INT8 quantized + BF16 normed ────────────────────
        for (int col = lane; col < N; col += SG_SIZE) {
          float normed = static_cast<float>(input_ptr[row * N + col]) * rstd;
          normed_out_ptr[row * N + col] = static_cast<ElementNormed>(normed);

          float qval = sycl::round(normed / scale_tok);
          qval = sycl::fmin(sycl::fmax(qval, -128.f), 127.f);
          quant_out_ptr[row * N + col] = static_cast<int8_t>(qval);
        }

        if (lane == 0)
          scale_token_ptr[row] = scale_tok;
      }
    );
  });
}

}  // namespace xe_fuse
