
// xe-fuse test: K0a — gemm_residual_gamma
// D[m,n] = gamma[n] * (acc[m,n] + residual[m,n])
// Then compute_rstd: R[m] = 1/sqrt(mean(D[m,:]^2) + eps)
// Validates both D output and R output against reference.

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

using K0a = xe_fuse::GemmResidualGamma<>;
using GemmOp = K0a::Gemm;

int main(int argc, const char** argv) {
  Options opts;
  opts.parse(argc, argv);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  int M = opts.m, N = opts.n, K = opts.k, L = opts.l;

  using StrideA = typename GemmOp::GemmKernel::StrideA;
  using StrideB = typename GemmOp::GemmKernel::StrideB;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, L));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, L));
  auto stride_C = cutlass::make_cute_packed_stride(K0a::StrideC{}, make_shape(M, N, L));
  auto stride_D = cutlass::make_cute_packed_stride(K0a::StrideD{}, make_shape(M, N, L));
  auto stride_res = cutlass::make_cute_packed_stride(K0a::StrideResidual{}, make_shape(M, N, L));

  // Allocate
  cutlass::DeviceAllocation<K0a::ElementA> block_A(static_cast<size_t>(M) * K * L);
  cutlass::DeviceAllocation<K0a::ElementB> block_B(static_cast<size_t>(K) * N * L);
  cutlass::DeviceAllocation<K0a::ElementD> block_D(static_cast<size_t>(M) * N * L);
  cutlass::DeviceAllocation<K0a::ElementD> block_ref_D(static_cast<size_t>(M) * N * L);
  cutlass::DeviceAllocation<K0a::ElementResidual> block_residual(static_cast<size_t>(M) * N * L);
  cutlass::DeviceAllocation<K0a::ElementGamma> block_gamma(static_cast<size_t>(N) * L);
  cutlass::DeviceAllocation<K0a::ElementCompute> block_rstd(static_cast<size_t>(M) * L);
  cutlass::DeviceAllocation<K0a::ElementCompute> block_ref_rstd(static_cast<size_t>(M) * L);

  initialize_block(block_A, 2023);
  initialize_block(block_B, 2022);
  initialize_block(block_residual, 2021);

  // Initialize gamma with values in [0.8, 1.2]
  std::vector<K0a::ElementGamma> h_gamma(static_cast<size_t>(N) * L);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.8f, 1.2f);
  for (auto& v : h_gamma) v = static_cast<K0a::ElementGamma>(dist(rng));
  compat::get_default_queue().memcpy(block_gamma.get(), h_gamma.data(), h_gamma.size() * sizeof(K0a::ElementGamma));
  compat::wait();

  // Build arguments
  auto evt_args = K0a::make_evt_args(block_residual.get(), stride_res, block_gamma.get(), N);

  typename GemmOp::GemmKernel::EpilogueArguments epilogue_args{
    evt_args,
    nullptr,
    stride_C,
    block_D.get(),
    stride_D
  };

  typename GemmOp::GemmKernel::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, L},
    {block_A.get(), stride_A, block_B.get(), stride_B},
    epilogue_args,
    hw_info
  };

  GemmOp gemm_op;
  size_t workspace_size = GemmOp::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  CUTLASS_CHECK(gemm_op.can_implement(arguments));
  CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));

  // Step 1: K0a GEMM
  CUTLASS_CHECK(gemm_op.run());
  compat::wait();

  // Step 2: compute_rstd
  auto q = compat::get_default_queue();
  xe_fuse::launch_compute_rstd(q, block_D.get(), block_rstd.get(), M, N, L);
  compat::wait();

  // Verify
  if (opts.verify) {
    // Reference GEMM
    cutlass::TensorRef ref_A(block_A.get(), cutlass::layout::RowMajor::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(), cutlass::layout::RowMajor::packed({K, N}));
    cutlass::TensorRef ref_C(block_ref_D.get(), cutlass::layout::RowMajor::packed({M, N}));
    cutlass::TensorRef ref_D(block_ref_D.get(), cutlass::layout::RowMajor::packed({M, N}));

    cutlass::reference::device::GemmComplex(
      {M, N, K}, K0a::ElementAcc(1), ref_A, cutlass::ComplexTransform::kNone,
      ref_B, cutlass::ComplexTransform::kNone, K0a::ElementAcc(0),
      ref_C, ref_D, K0a::ElementAcc(0), L, M * K, K * N, M * N, M * N);
    compat::wait();

    // ref_D[m,n] = gamma[n] * (acc[m,n] + residual[m,n])
    {
      auto* ref_ptr = block_ref_D.get();
      auto* res_ptr = block_residual.get();
      auto* gamma_ptr = block_gamma.get();
      int64_t total = static_cast<int64_t>(M) * N * L;
      int n_val = N, m_val = M;
      compat::get_default_queue().parallel_for(
        sycl::range<1>(total),
        [=](sycl::id<1> idx) {
          int64_t i = idx[0];
          int col = static_cast<int>(i % n_val);
          int batch = static_cast<int>(i / (m_val * n_val));
          float acc = static_cast<float>(ref_ptr[i]);
          float res = static_cast<float>(res_ptr[i]);
          float g = gamma_ptr[batch * n_val + col];
          ref_ptr[i] = static_cast<K0a::ElementD>(g * (acc + res));
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
      compat::get_default_queue().parallel_for(
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

    // Check D
    bool d_passed = cutlass::reference::device::BlockCompareRelativelyEqual(
      block_ref_D.get(), block_D.get(), block_D.size(),
      static_cast<K0a::ElementD>(0.05f), static_cast<K0a::ElementD>(0.05f));

    // Check rstd
    std::vector<float> h_rstd(static_cast<size_t>(M) * L);
    std::vector<float> h_ref_rstd(static_cast<size_t>(M) * L);
    q.memcpy(h_rstd.data(), block_rstd.get(), h_rstd.size() * sizeof(float));
    q.memcpy(h_ref_rstd.data(), block_ref_rstd.get(), h_ref_rstd.size() * sizeof(float));
    compat::wait();

    double max_rel_err = 0.0;
    int err_count = 0;
    for (int i = 0; i < M * L; ++i) {
      if (std::abs(h_ref_rstd[i]) > 1e-8f) {
        double rel = std::abs(h_rstd[i] - h_ref_rstd[i]) / std::abs(h_ref_rstd[i]);
        max_rel_err = std::max(max_rel_err, rel);
        if (rel > 0.05) {
          if (err_count < 5)
            printf("  rstd[%d]: kernel=%.6f ref=%.6f rel=%.4f\n", i, h_rstd[i], h_ref_rstd[i], rel);
          err_count++;
        }
      }
    }
    bool rstd_passed = (err_count == 0);

    printf("D output:     %s\n", d_passed ? "Passed" : "FAILED");
    printf("rstd output:  %s (max_rel_err=%.6f, errors=%d/%d)\n",
           rstd_passed ? "Passed" : "FAILED", max_rel_err, err_count, M * L);

    bool passed = d_passed && rstd_passed;
    std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;
    if (!passed) return 1;
  } else {
    std::cout << "Disposition is skipped." << std::endl;
  }

  // Benchmark
  if (opts.iterations > 0) {
    // K0a GEMM only
    GPU_Clock timer_k0a;
    timer_k0a.start();
    for (int i = 0; i < opts.iterations; ++i) {
      gemm_op.run();
    }
    compat::wait();
    float k0a_time = timer_k0a.seconds() / opts.iterations;

    // compute_rstd only
    GPU_Clock timer_rstd;
    timer_rstd.start();
    for (int i = 0; i < opts.iterations; ++i) {
      xe_fuse::launch_compute_rstd(q, block_D.get(), block_rstd.get(), M, N, L);
    }
    compat::wait();
    float rstd_time = timer_rstd.seconds() / opts.iterations;

    // K0a + rstd together
    GPU_Clock timer_total;
    timer_total.start();
    for (int i = 0; i < opts.iterations; ++i) {
      gemm_op.run();
      xe_fuse::launch_compute_rstd(q, block_D.get(), block_rstd.get(), M, N, L);
    }
    compat::wait();
    float total_time = timer_total.seconds() / opts.iterations;

    double tflops = (2.0 * M * N * K * L) * 1e-12;
    std::cout << "Problem Size: " << M << 'x' << N << 'x' << K << 'x' << L << std::endl;
    printf("xe-fuse K0a GEMM only:        [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / k0a_time, k0a_time * 1000);
    printf("xe-fuse compute_rstd:                           (%6.4f)ms\n", rstd_time * 1000);
    printf("xe-fuse K0a + rstd total:     [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / total_time, total_time * 1000);
  }

  return 0;
}
