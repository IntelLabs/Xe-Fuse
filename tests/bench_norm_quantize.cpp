// xe-fuse bench: norm + INT8 quantize
// Compares fusion strategies for RMSNorm followed by INT8 quantization.
//
// XE_FUSE_DUAL:  launch_norm_quantize_dual      (1 kernel -> BF16 normed + INT8)
// XE_FUSE_ORIG:  launch_compute_rstd_and_quantize (1 kernel -> INT8 only)
// VLLM_EQUIV:    vllm_equiv::rms_norm + quantize_bf16_to_int8   (2 kernels)
// NAIVE:         compute_rstd + quantize_activations              (3 kernels: rstd + 2-pass quant)
//
// oneDNN: no native fused RMSNorm+quantize; would require >=2 primitives.

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
  int M = 512;
  int N = 4096;
  int L = 1;
  int iterations = 100;

  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);
    cmd.get_cmd_line_argument("m",          M,          512);
    cmd.get_cmd_line_argument("n",          N,          4096);
    cmd.get_cmd_line_argument("l",          L,          1);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
  }
};

int main(int argc, const char** argv) {
  Options opts;
  opts.parse(argc, argv);

  int M     = opts.M;
  int N     = opts.N;
  int L     = opts.L;
  int iters = opts.iterations;

  int64_t total = static_cast<int64_t>(M) * N * L;

  // ── Shared input buffer ──────────────────────────────────────────────────
  cutlass::DeviceAllocation<bf16>   block_input(total);
  initialize_block(block_input, 2025);

  // bf16 weight for vllm_equiv::rms_norm (identity = 1.0)
  cutlass::DeviceAllocation<bf16>   block_weight(N);
  {
    std::vector<bf16> hw(N, static_cast<bf16>(1.0f));
    compat::get_default_queue().memcpy(block_weight.get(), hw.data(),
                                       N * sizeof(bf16));
  }
  compat::wait();

  // ── Per-approach output buffers ──────────────────────────────────────────
  // XE_FUSE_DUAL
  cutlass::DeviceAllocation<int8_t> quant_dual(total);
  cutlass::DeviceAllocation<bf16>   normed_dual(total);
  cutlass::DeviceAllocation<float>  scale_dual(static_cast<int64_t>(M) * L);

  // XE_FUSE_ORIG
  cutlass::DeviceAllocation<int8_t> quant_orig(total);
  cutlass::DeviceAllocation<float>  scale_orig(static_cast<int64_t>(M) * L);

  // VLLM_EQUIV
  cutlass::DeviceAllocation<bf16>   normed_vllm(total);
  cutlass::DeviceAllocation<int8_t> quant_vllm(total);
  cutlass::DeviceAllocation<float>  scale_vllm(static_cast<int64_t>(M) * L);

  // NAIVE
  cutlass::DeviceAllocation<float>  rstd_naive(static_cast<int64_t>(M) * L);
  cutlass::DeviceAllocation<int8_t> quant_naive(total);
  cutlass::DeviceAllocation<float>  scale_naive(static_cast<int64_t>(M) * L);

  sycl::queue q = compat::get_default_queue();

  // ── Warm up ──────────────────────────────────────────────────────────────
  constexpr int warmup = 5;

  for (int i = 0; i < warmup; ++i)
    xe_fuse::launch_norm_quantize_dual<bf16>(q, block_input.get(),
        quant_dual.get(), normed_dual.get(), scale_dual.get(), M, N, L);

  for (int i = 0; i < warmup; ++i)
    xe_fuse::launch_compute_rstd_and_quantize<bf16>(q, block_input.get(),
        quant_orig.get(), scale_orig.get(), M, N, L);

  for (int i = 0; i < warmup; ++i) {
    xe_fuse::vllm_equiv::rms_norm(q, normed_vllm.get(), block_input.get(),
        block_weight.get(), M * L, N);
    xe_fuse::standalone::quantize_bf16_to_int8(q, normed_vllm.get(),
        quant_vllm.get(), scale_vllm.get(), M, N, L);
  }

  for (int i = 0; i < warmup; ++i) {
    xe_fuse::standalone::compute_rstd(q, rstd_naive.get(), block_input.get(),
        M, N, L);
    xe_fuse::standalone::quantize_activations(q, block_input.get(),
        rstd_naive.get(), quant_naive.get(), scale_naive.get(), M, N, L);
  }
  compat::wait();

  // ── Benchmark XE_FUSE_DUAL ───────────────────────────────────────────────
  GPU_Clock timer;
  timer.start();
  for (int i = 0; i < iters; ++i)
    xe_fuse::launch_norm_quantize_dual<bf16>(q, block_input.get(),
        quant_dual.get(), normed_dual.get(), scale_dual.get(), M, N, L);
  compat::wait();
  float time_dual = timer.seconds() / iters;
  // read bf16 input + write bf16 normed + write int8
  double bytes_dual = (double)total * (sizeof(bf16) + sizeof(bf16) + sizeof(int8_t));

  // ── Benchmark XE_FUSE_ORIG ───────────────────────────────────────────────
  timer.start();
  for (int i = 0; i < iters; ++i)
    xe_fuse::launch_compute_rstd_and_quantize<bf16>(q, block_input.get(),
        quant_orig.get(), scale_orig.get(), M, N, L);
  compat::wait();
  float time_orig = timer.seconds() / iters;
  // read bf16 input (3 passes) + write int8
  double bytes_orig = (double)total * (3 * sizeof(bf16) + sizeof(int8_t));

  // ── Benchmark VLLM_EQUIV ─────────────────────────────────────────────────
  timer.start();
  for (int i = 0; i < iters; ++i) {
    xe_fuse::vllm_equiv::rms_norm(q, normed_vllm.get(), block_input.get(),
        block_weight.get(), M * L, N);
    xe_fuse::standalone::quantize_bf16_to_int8(q, normed_vllm.get(),
        quant_vllm.get(), scale_vllm.get(), M, N, L);
  }
  compat::wait();
  float time_vllm = timer.seconds() / iters;
  // rms_norm: read input + write normed; quant: read normed (2 passes) + write int8
  double bytes_vllm = (double)total * (sizeof(bf16) + sizeof(bf16) + 2 * sizeof(bf16) + sizeof(int8_t));

  // ── Benchmark NAIVE ───────────────────────────────────────────────────────
  timer.start();
  for (int i = 0; i < iters; ++i) {
    xe_fuse::standalone::compute_rstd(q, rstd_naive.get(), block_input.get(),
        M, N, L);
    xe_fuse::standalone::quantize_activations(q, block_input.get(),
        rstd_naive.get(), quant_naive.get(), scale_naive.get(), M, N, L);
  }
  compat::wait();
  float time_naive = timer.seconds() / iters;
  // compute_rstd: read input; quantize pass1: read input; quantize pass2: read input + write int8
  double bytes_naive = (double)total * (3 * sizeof(bf16) + sizeof(int8_t));

  // ── Print results ─────────────────────────────────────────────────────────
  printf("\n=== Norm + INT8 Quantize: M=%d N=%d ===\n", M, N);
  printf("XE_FUSE_DUAL (1 kernel, BF16+INT8 out): [%.3f]GB/s  (%.4f)ms\n",
         bytes_dual * 1e-9 / time_dual, time_dual * 1000.f);
  printf("XE_FUSE_ORIG (1 kernel, INT8 only):     [%.3f]GB/s  (%.4f)ms\n",
         bytes_orig * 1e-9 / time_orig, time_orig * 1000.f);
  printf("VLLM_EQUIV   (2 kernels):               [%.3f]GB/s  (%.4f)ms\n",
         bytes_vllm * 1e-9 / time_vllm, time_vllm * 1000.f);
  printf("NAIVE        (3 kernels):               [%.3f]GB/s  (%.4f)ms\n",
         bytes_naive * 1e-9 / time_naive, time_naive * 1000.f);
  printf("Speedup XE_FUSE_DUAL vs VLLM_EQUIV: %.2fx\n", time_vllm / time_dual);
  printf("Speedup XE_FUSE_ORIG vs NAIVE:      %.2fx\n", time_naive / time_orig);

  return 0;
}
