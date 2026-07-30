
// xe-fuse test: K0a_CR — gemm_residual_gamma + fused ColReduction
//
// Compares two approaches:
//   Current:  K0a GEMM + standalone compute_rstd (reads M*N from DRAM)
//   Fused:    K0a_CR GEMM (ColReduction in epilogue) + tiny rsqrt (M elements)
//
// Validates D and rstd against FP32 reference.

#include "xe-fuse/kernels/gemm_residual_gamma_colred.hpp"
#include "xe-fuse/kernels/gemm_residual_gamma.hpp"
#include "xe-fuse/kernels/compute_rstd.hpp"

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"

#include "sycl_common.hpp"
#include "helper.h"

#include <random>

using namespace cute;

struct Options {
  int m = 4096, n = 4096, k = 4096, l = 1;
  int iterations = 100;
  int verify = 1;

  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);
    cmd.get_cmd_line_argument("m", m, 4096);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
    cmd.get_cmd_line_argument("verify", verify, 1);
  }
};

using K0a_CR = xe_fuse::GemmResidualGammaColRed<>;
using K0a    = xe_fuse::GemmResidualGamma<>;

using FusedGemm    = K0a_CR::Gemm;
using BaselineGemm = K0a::Gemm;

int main(int argc, const char** argv) {
  Options opts;
  opts.parse(argc, argv);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  int M = opts.m, N = opts.n, K = opts.k, L = opts.l;

  using StrideA = typename FusedGemm::GemmKernel::StrideA;
  using StrideB = typename FusedGemm::GemmKernel::StrideB;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, L));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, L));
  auto stride_C = cutlass::make_cute_packed_stride(K0a_CR::StrideC{}, make_shape(M, N, L));
  auto stride_D = cutlass::make_cute_packed_stride(K0a_CR::StrideD{}, make_shape(M, N, L));
  auto stride_res = cutlass::make_cute_packed_stride(K0a_CR::StrideResidual{}, make_shape(M, N, L));

  // Allocate
  cutlass::DeviceAllocation<K0a_CR::ElementA> block_A(static_cast<size_t>(M) * K * L);
  cutlass::DeviceAllocation<K0a_CR::ElementB> block_B(static_cast<size_t>(K) * N * L);
  cutlass::DeviceAllocation<K0a_CR::ElementResidual> block_residual(static_cast<size_t>(M) * N * L);
  cutlass::DeviceAllocation<K0a_CR::ElementGamma> block_gamma(static_cast<size_t>(N) * L);

  // Fused outputs — per-tile partial sum_sq buffer (non-atomic, FinalReduction=false)
  cutlass::DeviceAllocation<K0a_CR::ElementD> block_D_fused(static_cast<size_t>(M) * N * L);
  size_t sum_sq_count = K0a_CR::get_partials_size(M, N, L);  // = padded_M * num_tiles_n * L
  cutlass::DeviceAllocation<K0a_CR::ElementCompute> block_sum_sq(sum_sq_count);
  cutlass::DeviceAllocation<K0a_CR::ElementCompute> block_rstd_fused(static_cast<size_t>(M) * L);

  // Baseline outputs
  cutlass::DeviceAllocation<K0a::ElementD> block_D_baseline(static_cast<size_t>(M) * N * L);
  cutlass::DeviceAllocation<K0a::ElementCompute> block_rstd_baseline(static_cast<size_t>(M) * L);

  // Reference
  cutlass::DeviceAllocation<K0a_CR::ElementD> block_ref_D(static_cast<size_t>(M) * N * L);
  cutlass::DeviceAllocation<K0a_CR::ElementCompute> block_ref_rstd(static_cast<size_t>(M) * L);

  initialize_block(block_A, 2023);
  initialize_block(block_B, 2022);
  initialize_block(block_residual, 2021);

  std::vector<K0a_CR::ElementGamma> h_gamma(static_cast<size_t>(N) * L);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.8f, 1.2f);
  for (auto& v : h_gamma) v = static_cast<K0a_CR::ElementGamma>(dist(rng));
  compat::get_default_queue().memcpy(block_gamma.get(), h_gamma.data(), h_gamma.size() * sizeof(K0a_CR::ElementGamma));
  compat::wait();

  auto q = compat::get_default_queue();

  // ================================================================
  // Build fused K0a_CR GEMM
  // ================================================================
  auto fused_evt_args = K0a_CR::make_evt_args(
      block_gamma.get(), N,
      block_sum_sq.get(), M);

  // Residual passed as ptr_C (not through EVT AuxLoad) to avoid IGC bug.
  typename FusedGemm::GemmKernel::EpilogueArguments fused_epi_args{
    fused_evt_args,
    block_residual.get(),
    stride_C,
    block_D_fused.get(),
    stride_D
  };

  typename FusedGemm::GemmKernel::Arguments fused_arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, L},
    {block_A.get(), stride_A, block_B.get(), stride_B},
    fused_epi_args,
    hw_info
  };

  FusedGemm fused_gemm;
  size_t fused_ws_size = FusedGemm::get_workspace_size(fused_arguments);
  cutlass::device_memory::allocation<uint8_t> fused_workspace(fused_ws_size);
  printf("Workspace size: %zu bytes\n", fused_ws_size);

  auto status = fused_gemm.can_implement(fused_arguments);
  if (status != cutlass::Status::kSuccess) {
    printf("K0a_CR can_implement failed: %d\n", static_cast<int>(status));
    return 1;
  }
  // initialize() is a no-op for FinalReduction=false (each tile overwrites its unique slot)
  CUTLASS_CHECK(fused_gemm.initialize(fused_arguments, fused_workspace.get()));

  CUTLASS_CHECK(fused_gemm.run());
  compat::wait();

  // Tiny post-GEMM rsqrt pass: reads M floats instead of M*N bf16 values
  xe_fuse::launch_reduce_partials_rsqrt<K0a_CR::CTA_M, K0a_CR::CTA_N>(
      q, block_sum_sq.get(), block_rstd_fused.get(), M, N, L);
  compat::wait();

  // ================================================================
  // Build baseline K0a GEMM + standalone compute_rstd
  // ================================================================
  using BaseStrideRes = K0a::StrideResidual;
  auto base_stride_res = cutlass::make_cute_packed_stride(BaseStrideRes{}, make_shape(M, N, L));

  auto base_evt_args = K0a::make_evt_args(
      block_residual.get(), base_stride_res,
      block_gamma.get(), N);

  typename BaselineGemm::GemmKernel::EpilogueArguments base_epi_args{
    base_evt_args,
    nullptr,
    stride_C,
    block_D_baseline.get(),
    stride_D
  };

  typename BaselineGemm::GemmKernel::Arguments base_arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, L},
    {block_A.get(), stride_A, block_B.get(), stride_B},
    base_epi_args,
    hw_info
  };

  BaselineGemm base_gemm;
  size_t base_ws_size = BaselineGemm::get_workspace_size(base_arguments);
  cutlass::device_memory::allocation<uint8_t> base_workspace(base_ws_size);

  CUTLASS_CHECK(base_gemm.can_implement(base_arguments));
  CUTLASS_CHECK(base_gemm.initialize(base_arguments, base_workspace.get()));

  CUTLASS_CHECK(base_gemm.run());
  compat::wait();

  xe_fuse::launch_compute_rstd(q, block_D_baseline.get(), block_rstd_baseline.get(), M, N, L);
  compat::wait();

  // ================================================================
  // Verify
  // ================================================================
  if (opts.verify) {
    // Reference GEMM
    cutlass::TensorRef ref_A(block_A.get(), cutlass::layout::RowMajor::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(), cutlass::layout::RowMajor::packed({K, N}));
    cutlass::TensorRef ref_C(block_ref_D.get(), cutlass::layout::RowMajor::packed({M, N}));
    cutlass::TensorRef ref_D(block_ref_D.get(), cutlass::layout::RowMajor::packed({M, N}));

    cutlass::reference::device::GemmComplex(
      {M, N, K}, K0a_CR::ElementAcc(1), ref_A, cutlass::ComplexTransform::kNone,
      ref_B, cutlass::ComplexTransform::kNone, K0a_CR::ElementAcc(0),
      ref_C, ref_D, K0a_CR::ElementAcc(0), L, M * K, K * N, M * N, M * N);
    compat::wait();

    // ref_D = gamma * (acc + residual)
    {
      auto* ref_ptr = block_ref_D.get();
      auto* res_ptr = block_residual.get();
      auto* gamma_ptr = block_gamma.get();
      int64_t total = static_cast<int64_t>(M) * N * L;
      int n_val = N, m_val = M;
      q.parallel_for(
        sycl::range<1>(total),
        [=](sycl::id<1> idx) {
          int64_t i = idx[0];
          int col = static_cast<int>(i % n_val);
          int batch = static_cast<int>(i / (m_val * n_val));
          float acc = static_cast<float>(ref_ptr[i]);
          float res = static_cast<float>(res_ptr[i]);
          float g = gamma_ptr[batch * n_val + col];
          ref_ptr[i] = static_cast<K0a_CR::ElementD>(g * (acc + res));
        }
      );
    }
    compat::wait();

    // Reference rstd
    {
      auto* ref_ptr = block_ref_D.get();
      auto* rms_ptr = block_ref_rstd.get();
      int n_val = N;
      float eps = 1e-6f;
      q.parallel_for(
        sycl::range<1>(static_cast<size_t>(M) * L),
        [=](sycl::id<1> idx) {
          int row = static_cast<int>(idx[0]);
          float sum_sq = 0.0f;
          for (int col = 0; col < n_val; ++col) {
            float val = static_cast<float>(ref_ptr[row * n_val + col]);
            sum_sq += val * val;
          }
          rms_ptr[row] = 1.0f / sycl::sqrt(sum_sq / static_cast<float>(n_val) + eps);
        }
      );
    }
    compat::wait();

    // Check fused D
    bool d_fused_passed = cutlass::reference::device::BlockCompareRelativelyEqual(
      block_ref_D.get(), block_D_fused.get(), block_D_fused.size(),
      static_cast<K0a_CR::ElementD>(0.05f), static_cast<K0a_CR::ElementD>(0.05f));

    // Check baseline D
    bool d_base_passed = cutlass::reference::device::BlockCompareRelativelyEqual(
      block_ref_D.get(), block_D_baseline.get(), block_D_baseline.size(),
      static_cast<K0a::ElementD>(0.05f), static_cast<K0a::ElementD>(0.05f));

    // sum_sq diagnostics: compare fused sum_sq[m] vs reference sum(D²)
    {
      std::vector<float> h_sum_sq(sum_sq_count);
      std::vector<K0a_CR::ElementD> h_D_fused(static_cast<size_t>(M) * N * L);
      q.memcpy(h_sum_sq.data(), block_sum_sq.get(), sum_sq_count * sizeof(float));
      q.memcpy(h_D_fused.data(), block_D_fused.get(), h_D_fused.size() * sizeof(K0a_CR::ElementD));
      compat::wait();

      printf("\n=== sum_sq diagnostics ===\n");
      for (int row = 0; row < 3 && row < M; ++row) {
        float ref_sq = 0.0f;
        for (int col = 0; col < N; ++col) {
          float v = static_cast<float>(h_D_fused[row * N + col]);
          ref_sq += v * v;
        }
        float ratio = (ref_sq > 1e-8f) ? h_sum_sq[row] / ref_sq : 0.0f;
        printf("  row[%d]: fused_sum_sq=%.2f ref_sum_sq=%.2f ratio=%.4f\n",
               row, h_sum_sq[row], ref_sq, ratio);
      }
    }

    // Check fused rstd
    std::vector<float> h_rstd_fused(static_cast<size_t>(M) * L);
    std::vector<float> h_rstd_base(static_cast<size_t>(M) * L);
    std::vector<float> h_ref_rstd(static_cast<size_t>(M) * L);
    q.memcpy(h_rstd_fused.data(), block_rstd_fused.get(), h_rstd_fused.size() * sizeof(float));
    q.memcpy(h_rstd_base.data(), block_rstd_baseline.get(), h_rstd_base.size() * sizeof(float));
    q.memcpy(h_ref_rstd.data(), block_ref_rstd.get(), h_ref_rstd.size() * sizeof(float));
    compat::wait();

    auto check_rstd = [&](const char* name, const std::vector<float>& h_rstd) {
      double max_rel_err = 0.0;
      int err_count = 0;
      for (int i = 0; i < M * L; ++i) {
        if (std::abs(h_ref_rstd[i]) > 1e-8f) {
          double rel = std::abs(h_rstd[i] - h_ref_rstd[i]) / std::abs(h_ref_rstd[i]);
          max_rel_err = std::max(max_rel_err, rel);
          if (rel > 0.05) {
            if (err_count < 5)
              printf("  %s rstd[%d]: kernel=%.6f ref=%.6f rel=%.4f\n",
                     name, i, h_rstd[i], h_ref_rstd[i], rel);
            err_count++;
          }
        }
      }
      printf("%s rstd:  %s (max_rel_err=%.6f, errors=%d/%d)\n",
             name, err_count == 0 ? "Passed" : "FAILED",
             max_rel_err, err_count, M * L);
      return err_count == 0;
    };

    printf("\n=== Verification ===\n");
    printf("Fused D:      %s\n", d_fused_passed ? "Passed" : "FAILED");
    printf("Baseline D:   %s\n", d_base_passed ? "Passed" : "FAILED");
    bool rstd_fused_ok = check_rstd("Fused", h_rstd_fused);
    bool rstd_base_ok  = check_rstd("Baseline", h_rstd_base);

    bool passed = d_fused_passed && d_base_passed && rstd_fused_ok && rstd_base_ok;
    std::cout << "\nDisposition: " << (passed ? "Passed" : "Failed") << std::endl;
    if (!passed) return 1;
  } else {
    std::cout << "Disposition is skipped." << std::endl;
  }

  // ================================================================
  // Benchmark
  // ================================================================
  if (opts.iterations > 0) {
    double tflops = (2.0 * M * N * K * L) * 1e-12;

    // ── Baseline: K0a GEMM + standalone compute_rstd ──
    GPU_Clock timer_base;
    timer_base.start();
    for (int i = 0; i < opts.iterations; ++i) {
      base_gemm.run();
      xe_fuse::launch_compute_rstd(q, block_D_baseline.get(), block_rstd_baseline.get(), M, N, L);
    }
    compat::wait();
    float t_baseline = timer_base.seconds() / opts.iterations;

    GPU_Clock timer_base_gemm;
    timer_base_gemm.start();
    for (int i = 0; i < opts.iterations; ++i) {
      base_gemm.run();
    }
    compat::wait();
    float t_base_gemm = timer_base_gemm.seconds() / opts.iterations;

    GPU_Clock timer_rstd;
    timer_rstd.start();
    for (int i = 0; i < opts.iterations; ++i) {
      xe_fuse::launch_compute_rstd(q, block_D_baseline.get(), block_rstd_baseline.get(), M, N, L);
    }
    compat::wait();
    float t_rstd = timer_rstd.seconds() / opts.iterations;

    // ── Fused: K0a_CR GEMM + rsqrt ──
    // initialize() is a no-op for FinalReduction=false; kept for API consistency.
    GPU_Clock timer_fused;
    timer_fused.start();
    for (int i = 0; i < opts.iterations; ++i) {
      fused_gemm.initialize(fused_arguments, fused_workspace.get());
      fused_gemm.run();
      xe_fuse::launch_reduce_partials_rsqrt<K0a_CR::CTA_M, K0a_CR::CTA_N>(
          q, block_sum_sq.get(), block_rstd_fused.get(), M, N, L);
    }
    compat::wait();
    float t_fused = timer_fused.seconds() / opts.iterations;

    // Fused GEMM only (no re-init, to measure GEMM overhead alone)
    GPU_Clock timer_fused_gemm;
    timer_fused_gemm.start();
    for (int i = 0; i < opts.iterations; ++i) {
      fused_gemm.run();
    }
    compat::wait();
    float t_fused_gemm = timer_fused_gemm.seconds() / opts.iterations;

    std::cout << "\n=== Performance ===" << std::endl;
    std::cout << "Problem Size: " << M << 'x' << N << 'x' << K << 'x' << L << std::endl;
    printf("\nBaseline (K0a + standalone compute_rstd):\n");
    printf("  K0a GEMM:           [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / t_base_gemm, t_base_gemm * 1000);
    printf("  compute_rstd:                         (%6.4f)ms\n", t_rstd * 1000);
    printf("  Total:              [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / t_baseline, t_baseline * 1000);

    printf("\nFused (K0a_CR + rsqrt):\n");
    printf("  K0a_CR GEMM only:   [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / t_fused_gemm, t_fused_gemm * 1000);
    printf("  Total (incl init+rsqrt): [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / t_fused, t_fused * 1000);

    float speedup = t_baseline / t_fused;
    float saved_ms = (t_baseline - t_fused) * 1000;
    printf("\nSpeedup: %.2fx (saved %.4f ms per iteration)\n", speedup, saved_ms);
    printf("Standalone rstd cost: %.4f ms (eliminated by fusion)\n", t_rstd * 1000);

    printf("\n=== STRUCTURED OUTPUT ===\n");
    printf("COLRED_FUSED: M=%d N=%d K=%d\n", M, N, K);
    printf("BASELINE_TOTAL: %.4f ms\n", t_baseline * 1000);
    printf("BASELINE_GEMM: %.4f ms\n", t_base_gemm * 1000);
    printf("STANDALONE_RSTD: %.4f ms\n", t_rstd * 1000);
    printf("FUSED_TOTAL: %.4f ms\n", t_fused * 1000);
    printf("FUSED_GEMM: %.4f ms\n", t_fused_gemm * 1000);
    printf("SPEEDUP: %.4f\n", speedup);
  }

  return 0;
}
