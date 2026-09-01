// xe-fuse bench: SwiGLU + INT8 requant
// Compares fusion strategies for SwiGLU activation followed by INT8 quantization.
//
// XE_FUSE_FUSED:  launch_swiglu_requant<bf16,true>  (1 kernel, interleaved gate+up -> INT8)
// XE_FUSE_POST:   launch_swiglu_requant<bf16,false> (1 kernel, post-SwiGLU BF16 -> INT8)
// VLLM_EQUIV:     vllm_equiv::silu_and_mul + quantize_bf16_to_int8 (2 kernels)
// NAIVE:          standalone::swiglu + compute_rstd + quantize_activations (3 kernels)
//
// oneDNN: no native SwiGLU+quantize primitive.

#include "xe-fuse/kernels/swiglu_requant.hpp"
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
  int M     = 512;
  int N_ffn = 28672;  // LLaMA 3 8B FFN intermediate size (gate+up concatenated)
  int L     = 1;
  int iterations = 100;

  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);
    cmd.get_cmd_line_argument("m",          M,          512);
    cmd.get_cmd_line_argument("n_ffn",      N_ffn,      28672);
    cmd.get_cmd_line_argument("l",          L,          1);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
  }
};

int main(int argc, const char** argv) {
  Options opts;
  opts.parse(argc, argv);

  int M     = opts.M;
  int N_ffn = opts.N_ffn;
  int L     = opts.L;
  int iters = opts.iterations;
  int d     = N_ffn / 2;  // post-SwiGLU intermediate size

  int64_t total_gate_up = static_cast<int64_t>(M) * N_ffn * L;
  int64_t total_out     = static_cast<int64_t>(M) * d * L;

  sycl::queue q = compat::get_default_queue();

  // ── Shared interleaved gate+up input (for XE_FUSE_FUSED and VLLM_EQUIV) ──
  cutlass::DeviceAllocation<bf16>  block_gate_up(total_gate_up);
  initialize_block(block_gate_up, 2025);

  // Pre-compute post-SwiGLU BF16 for XE_FUSE_POST and NAIVE sub-steps.
  cutlass::DeviceAllocation<bf16>  block_post_swiglu(total_out);
  xe_fuse::vllm_equiv::silu_and_mul(q,
      block_post_swiglu.get(), block_gate_up.get(), d, M * L);
  compat::wait();

  // ── Per-approach output buffers ──────────────────────────────────────────
  // XE_FUSE_FUSED
  cutlass::DeviceAllocation<int8_t> quant_fused(total_out);
  cutlass::DeviceAllocation<float>  scale_fused(static_cast<int64_t>(M) * L);

  // XE_FUSE_POST
  cutlass::DeviceAllocation<int8_t> quant_post(total_out);
  cutlass::DeviceAllocation<float>  scale_post(static_cast<int64_t>(M) * L);

  // VLLM_EQUIV: intermediate post-swiglu bf16
  cutlass::DeviceAllocation<bf16>   swiglu_vllm(total_out);
  cutlass::DeviceAllocation<int8_t> quant_vllm(total_out);
  cutlass::DeviceAllocation<float>  scale_vllm(static_cast<int64_t>(M) * L);

  // NAIVE: swiglu step on gate+up in-place working copy; rstd+quant on post_swiglu
  cutlass::DeviceAllocation<bf16>   swiglu_work_naive(total_gate_up);
  cutlass::DeviceAllocation<float>  rstd_naive(static_cast<int64_t>(M) * L);
  cutlass::DeviceAllocation<int8_t> quant_naive(total_out);
  cutlass::DeviceAllocation<float>  scale_naive(static_cast<int64_t>(M) * L);

  // Initialize NAIVE swiglu working copy from gate+up input
  q.memcpy(swiglu_work_naive.get(), block_gate_up.get(),
           total_gate_up * sizeof(bf16));
  compat::wait();

  // ── Warm up ──────────────────────────────────────────────────────────────
  constexpr int warmup = 5;

  for (int i = 0; i < warmup; ++i)
    xe_fuse::launch_swiglu_requant<bf16, true>(q, block_gate_up.get(),
        quant_fused.get(), scale_fused.get(), M * L, N_ffn);

  for (int i = 0; i < warmup; ++i)
    xe_fuse::launch_swiglu_requant<bf16, false>(q, block_post_swiglu.get(),
        quant_post.get(), scale_post.get(), M * L, d);

  for (int i = 0; i < warmup; ++i) {
    xe_fuse::vllm_equiv::silu_and_mul(q, swiglu_vllm.get(), block_gate_up.get(),
        d, M * L);
    xe_fuse::standalone::quantize_bf16_to_int8(q, swiglu_vllm.get(),
        quant_vllm.get(), scale_vllm.get(), M, d, L);
  }

  for (int i = 0; i < warmup; ++i) {
    xe_fuse::standalone::swiglu(q, swiglu_work_naive.get(), M * L, N_ffn, 1);
    xe_fuse::standalone::compute_rstd(q, rstd_naive.get(),
        block_post_swiglu.get(), M, d, L);
    xe_fuse::standalone::quantize_activations(q, block_post_swiglu.get(),
        rstd_naive.get(), quant_naive.get(), scale_naive.get(), M, d, L);
  }
  compat::wait();

  // ── Benchmark XE_FUSE_FUSED ──────────────────────────────────────────────
  GPU_Clock timer;
  timer.start();
  for (int i = 0; i < iters; ++i)
    xe_fuse::launch_swiglu_requant<bf16, true>(q, block_gate_up.get(),
        quant_fused.get(), scale_fused.get(), M * L, N_ffn);
  compat::wait();
  float time_fused = timer.seconds() / iters;
  // read gate+up [M, N_ffn] bf16 + write [M, d] int8
  double bytes_fused = (double)M * L * (N_ffn * sizeof(bf16) + d * sizeof(int8_t));

  // ── Benchmark XE_FUSE_POST ───────────────────────────────────────────────
  timer.start();
  for (int i = 0; i < iters; ++i)
    xe_fuse::launch_swiglu_requant<bf16, false>(q, block_post_swiglu.get(),
        quant_post.get(), scale_post.get(), M * L, d);
  compat::wait();
  float time_post = timer.seconds() / iters;
  // read post-swiglu [M, d] bf16 (2 passes) + write [M, d] int8
  double bytes_post = (double)M * L * d * (2 * sizeof(bf16) + sizeof(int8_t));

  // ── Benchmark VLLM_EQUIV ─────────────────────────────────────────────────
  timer.start();
  for (int i = 0; i < iters; ++i) {
    xe_fuse::vllm_equiv::silu_and_mul(q, swiglu_vllm.get(), block_gate_up.get(),
        d, M * L);
    xe_fuse::standalone::quantize_bf16_to_int8(q, swiglu_vllm.get(),
        quant_vllm.get(), scale_vllm.get(), M, d, L);
  }
  compat::wait();
  float time_vllm = timer.seconds() / iters;
  // silu_and_mul: read [M, N_ffn] + write [M, d]; quant: read [M, d] (2 passes) + write [M, d] int8
  double bytes_vllm = (double)M * L * (N_ffn * sizeof(bf16) + d * sizeof(bf16)
                                       + 2 * d * sizeof(bf16) + d * sizeof(int8_t));

  // ── Benchmark NAIVE ───────────────────────────────────────────────────────
  timer.start();
  for (int i = 0; i < iters; ++i) {
    // swiglu: reads+writes [M, N_ffn] in-place
    xe_fuse::standalone::swiglu(q, swiglu_work_naive.get(), M * L, N_ffn, 1);
    // rstd: reads [M, d] post-swiglu
    xe_fuse::standalone::compute_rstd(q, rstd_naive.get(),
        block_post_swiglu.get(), M, d, L);
    // quantize: reads [M, d] twice + writes [M, d] int8
    xe_fuse::standalone::quantize_activations(q, block_post_swiglu.get(),
        rstd_naive.get(), quant_naive.get(), scale_naive.get(), M, d, L);
  }
  compat::wait();
  float time_naive = timer.seconds() / iters;
  // swiglu: 2*N_ffn; rstd: d; quant: 2*d read + d int8 write
  double bytes_naive = (double)M * L * (2 * N_ffn * sizeof(bf16) + d * sizeof(bf16)
                                        + 2 * d * sizeof(bf16) + d * sizeof(int8_t));

  // ── Print results ─────────────────────────────────────────────────────────
  printf("\n=== SwiGLU + INT8 Requant: M=%d N_ffn=%d (d=%d) ===\n", M, N_ffn, d);
  printf("XE_FUSE_FUSED (1 kernel, gate+up->INT8): [%.3f]GB/s  (%.4f)ms\n",
         bytes_fused * 1e-9 / time_fused, time_fused * 1000.f);
  printf("XE_FUSE_POST  (1 kernel, swiglu->INT8):  [%.3f]GB/s  (%.4f)ms\n",
         bytes_post  * 1e-9 / time_post,  time_post  * 1000.f);
  printf("VLLM_EQUIV    (2 kernels):              [%.3f]GB/s  (%.4f)ms\n",
         bytes_vllm  * 1e-9 / time_vllm,  time_vllm  * 1000.f);
  printf("NAIVE         (3 kernels):              [%.3f]GB/s  (%.4f)ms\n",
         bytes_naive * 1e-9 / time_naive, time_naive * 1000.f);
  printf("Speedup XE_FUSE_FUSED vs VLLM_EQUIV: %.2fx\n", time_vllm  / time_fused);
  printf("Speedup XE_FUSE_FUSED vs NAIVE:      %.2fx\n", time_naive / time_fused);

  return 0;
}
