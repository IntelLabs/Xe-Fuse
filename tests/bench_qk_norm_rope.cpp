// xe-fuse bench: QK norm + RoPE
// Compares three fusion strategies for per-head RMSNorm + RoPE on Q/K tensors.
//
// XE_FUSE:    launch_qk_norm_rope         (1 kernel, fused)
// VLLM_EQUIV: rms_norm_per_head + rope_interleaved   (2 kernels, intermediate DRAM write)
// NAIVE:      compute_rstd_per_head + scale_rows + rope  (3 separate kernels, 2 DRAM writes)
//
// oneDNN has no native QK-norm+RoPE primitive; the VLLM_EQUIV path represents
// what an oneDNN-based implementation would require at minimum.

#include "xe-fuse/kernels/qk_norm_rope.hpp"
#include "xe-fuse/kernels/compute_rstd.hpp"
#include "xe-fuse/standalone/ops.hpp"
#include "xe-fuse/standalone/vllm_ops.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "sycl_common.hpp"
#include "helper.h"
#include <random>
#include <cmath>
#include <cstdio>

using bf16 = cutlass::bfloat16_t;

struct Options {
  int seq_len   = 4096;
  int num_heads = 32;
  int head_dim  = 128;
  int batch     = 1;
  int iterations = 100;

  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);
    cmd.get_cmd_line_argument("seq_len",    seq_len,    4096);
    cmd.get_cmd_line_argument("num_heads",  num_heads,  32);
    cmd.get_cmd_line_argument("head_dim",   head_dim,   128);
    cmd.get_cmd_line_argument("batch",      batch,      1);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
  }
};

int main(int argc, const char** argv) {
  Options opts;
  opts.parse(argc, argv);

  int M         = opts.batch * opts.seq_len;
  int num_heads = opts.num_heads;
  int head_dim  = opts.head_dim;
  int iters     = opts.iterations;

  int64_t total_size   = static_cast<int64_t>(M) * num_heads * head_dim;
  int64_t cos_sin_size = static_cast<int64_t>(M) * head_dim;
  // cos_sin_flat: [M, num_heads * head_dim] — per-token values broadcast per head
  int64_t cos_sin_flat_size = total_size;

  // ── Shared read-only buffers ─────────────────────────────────────────────
  cutlass::DeviceAllocation<bf16>   block_input(total_size);
  cutlass::DeviceAllocation<float>  block_gamma(head_dim);
  cutlass::DeviceAllocation<float>  block_cos_sin(cos_sin_size);

  initialize_block(block_input, 2025);

  // gamma: uniform (0.5, 1.5)
  {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.5f, 1.5f);
    std::vector<float> h(head_dim);
    for (auto& v : h) v = dist(rng);
    compat::get_default_queue().memcpy(block_gamma.get(), h.data(),
                                       head_dim * sizeof(float));
  }

  // cos_sin: realistic RoPE frequencies [M, head_dim] interleaved
  {
    std::vector<float> h(cos_sin_size);
    for (int tok = 0; tok < M; ++tok) {
      for (int k = 0; k < head_dim / 2; ++k) {
        float freq  = 1.0f / std::pow(10000.f, 2.f * k / static_cast<float>(head_dim));
        float angle = static_cast<float>(tok) * freq;
        int base = tok * head_dim;
        h[base + 2 * k]     = std::cos(angle);
        h[base + 2 * k + 1] = std::sin(angle);
      }
    }
    compat::get_default_queue().memcpy(block_cos_sin.get(), h.data(),
                                       cos_sin_size * sizeof(float));
  }

  // cos_sin_flat: [M, num_heads * head_dim] — broadcast per head for NAIVE rope
  cutlass::DeviceAllocation<float>  block_cos_sin_flat(cos_sin_flat_size);
  {
    std::vector<float> h_cs(cos_sin_size);
    compat::get_default_queue().memcpy(h_cs.data(), block_cos_sin.get(),
                                       cos_sin_size * sizeof(float)).wait();
    std::vector<float> h_flat(cos_sin_flat_size);
    for (int tok = 0; tok < M; ++tok) {
      for (int hd = 0; hd < num_heads; ++hd) {
        for (int d = 0; d < head_dim; ++d) {
          h_flat[tok * num_heads * head_dim + hd * head_dim + d] =
              h_cs[tok * head_dim + d];
        }
      }
    }
    compat::get_default_queue().memcpy(block_cos_sin_flat.get(), h_flat.data(),
                                       cos_sin_flat_size * sizeof(float));
  }
  compat::wait();

  // ── Per-approach output / working buffers ────────────────────────────────
  // XE_FUSE
  cutlass::DeviceAllocation<bf16>  out_xe(total_size);

  // VLLM_EQUIV: uses normed intermediate
  cutlass::DeviceAllocation<bf16>  normed_vllm(total_size);
  cutlass::DeviceAllocation<bf16>  out_vllm(total_size);

  // NAIVE: uses normed working copy (modified in-place by scale_rows),
  //        rstd [M * num_heads], separate rope output
  cutlass::DeviceAllocation<bf16>   normed_naive(total_size);
  cutlass::DeviceAllocation<float>  rstd_naive(static_cast<int64_t>(M) * num_heads);
  cutlass::DeviceAllocation<bf16>   out_naive(total_size);

  // Pre-copy input to normed_naive (scale_rows operates in-place)
  compat::get_default_queue().memcpy(normed_naive.get(), block_input.get(),
                                     total_size * sizeof(bf16));
  compat::wait();

  sycl::queue q = compat::get_default_queue();

  // ── Warm up ──────────────────────────────────────────────────────────────
  constexpr int warmup = 5;

  for (int i = 0; i < warmup; ++i)
    xe_fuse::launch_qk_norm_rope<bf16>(q, block_input.get(), out_xe.get(),
        block_gamma.get(), block_cos_sin.get(), M, num_heads, head_dim);

  for (int i = 0; i < warmup; ++i) {
    xe_fuse::vllm_equiv::rms_norm_per_head(q, normed_vllm.get(), block_input.get(),
        block_gamma.get(), M, num_heads, head_dim);
    xe_fuse::vllm_equiv::rope_interleaved(q, out_vllm.get(), normed_vllm.get(),
        block_cos_sin.get(), M, num_heads, head_dim);
  }

  for (int i = 0; i < warmup; ++i) {
    xe_fuse::launch_compute_rstd<bf16, float>(q, block_input.get(), rstd_naive.get(),
        M * num_heads, head_dim, 1);
    xe_fuse::standalone::scale_rows(q, normed_naive.get(), rstd_naive.get(),
        M * num_heads, head_dim, 1);
    xe_fuse::standalone::rope(q, out_naive.get(), normed_naive.get(),
        block_cos_sin_flat.get(), M, num_heads * head_dim, 1);
  }
  compat::wait();

  // ── Benchmark XE_FUSE ────────────────────────────────────────────────────
  GPU_Clock timer;
  timer.start();
  for (int i = 0; i < iters; ++i)
    xe_fuse::launch_qk_norm_rope<bf16>(q, block_input.get(), out_xe.get(),
        block_gamma.get(), block_cos_sin.get(), M, num_heads, head_dim);
  compat::wait();
  float time_xe = timer.seconds() / iters;
  double bytes_xe = 2.0 * total_size * sizeof(bf16);

  // ── Benchmark VLLM_EQUIV ─────────────────────────────────────────────────
  timer.start();
  for (int i = 0; i < iters; ++i) {
    xe_fuse::vllm_equiv::rms_norm_per_head(q, normed_vllm.get(), block_input.get(),
        block_gamma.get(), M, num_heads, head_dim);
    xe_fuse::vllm_equiv::rope_interleaved(q, out_vllm.get(), normed_vllm.get(),
        block_cos_sin.get(), M, num_heads, head_dim);
  }
  compat::wait();
  float time_vllm = timer.seconds() / iters;
  // read + intermediate write + read (final write implicit)
  double bytes_vllm = 3.0 * total_size * sizeof(bf16);

  // ── Benchmark NAIVE ───────────────────────────────────────────────────────
  timer.start();
  for (int i = 0; i < iters; ++i) {
    xe_fuse::launch_compute_rstd<bf16, float>(q, block_input.get(), rstd_naive.get(),
        M * num_heads, head_dim, 1);
    xe_fuse::standalone::scale_rows(q, normed_naive.get(), rstd_naive.get(),
        M * num_heads, head_dim, 1);
    xe_fuse::standalone::rope(q, out_naive.get(), normed_naive.get(),
        block_cos_sin_flat.get(), M, num_heads * head_dim, 1);
  }
  compat::wait();
  float time_naive = timer.seconds() / iters;
  // rstd read + scale read+write + rope read+write
  double bytes_naive = 4.0 * total_size * sizeof(bf16);

  // ── Print results ─────────────────────────────────────────────────────────
  printf("\n=== QK Norm + RoPE: M=%d num_heads=%d head_dim=%d ===\n",
         M, num_heads, head_dim);
  printf("XE_FUSE    (1 kernel, fused):                [%.3f]GB/s  (%.4f)ms\n",
         bytes_xe   * 1e-9 / time_xe,   time_xe   * 1000.f);
  printf("VLLM_EQUIV (2 kernels, norm+rope):           [%.3f]GB/s  (%.4f)ms\n",
         bytes_vllm * 1e-9 / time_vllm, time_vllm * 1000.f);
  printf("NAIVE      (3 kernels, rstd+scale+rope):     [%.3f]GB/s  (%.4f)ms\n",
         bytes_naive * 1e-9 / time_naive, time_naive * 1000.f);
  printf("Speedup XE_FUSE vs VLLM_EQUIV: %.2fx\n", time_vllm / time_xe);
  printf("Speedup XE_FUSE vs NAIVE:      %.2fx\n", time_naive / time_xe);

  return 0;
}
