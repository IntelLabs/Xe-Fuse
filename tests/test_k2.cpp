
// xe-fuse test: K2 — gemm_rmsnorm_swiglu
// D = SwiGLU( acc * R )
// N must be even (interleaved gate/up). Both even/odd lanes produce the
// same SwiGLU value. Verification checks even columns against reference.

#include "xe-fuse/kernels/gemm_rmsnorm_swiglu.hpp"

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"

#include "sycl_common.hpp"
#include "helper.h"

#include <random>
#include <cmath>

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

using K2 = xe_fuse::GemmRmsNormSwiGLU<>;
using GemmOp = K2::Gemm;

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
  auto stride_C = cutlass::make_cute_packed_stride(K2::StrideC{}, make_shape(M, N, L));
  auto stride_D = cutlass::make_cute_packed_stride(K2::StrideD{}, make_shape(M, N, L));

  cutlass::DeviceAllocation<K2::ElementA> block_A(static_cast<size_t>(M) * K * L);
  cutlass::DeviceAllocation<K2::ElementB> block_B(static_cast<size_t>(K) * N * L);
  cutlass::DeviceAllocation<K2::ElementD> block_D(static_cast<size_t>(M) * N * L);
  cutlass::DeviceAllocation<K2::ElementD> block_ref_D(static_cast<size_t>(M) * N * L);
  cutlass::DeviceAllocation<K2::ElementScale> block_scale(static_cast<size_t>(M) * L);

  initialize_block(block_A, 2023);
  initialize_block(block_B, 2022);

  std::vector<K2::ElementScale> h_scale(static_cast<size_t>(M) * L);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.5f, 1.5f);
  for (auto& v : h_scale) v = static_cast<K2::ElementScale>(dist(rng));
  compat::get_default_queue().memcpy(block_scale.get(), h_scale.data(), h_scale.size() * sizeof(K2::ElementScale));
  compat::wait();

  auto evt_args = K2::make_evt_args(block_scale.get(), M);

  typename GemmOp::GemmKernel::EpilogueArguments epilogue_args{
    evt_args, nullptr, stride_C, block_D.get(), stride_D
  };

  typename GemmOp::GemmKernel::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, L},
    {block_A.get(), stride_A, block_B.get(), stride_B},
    epilogue_args, hw_info
  };

  GemmOp gemm_op;
  size_t workspace_size = GemmOp::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  CUTLASS_CHECK(gemm_op.can_implement(arguments));
  CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm_op.run());
  compat::wait();

  if (opts.verify) {
    // Reference GEMM
    cutlass::TensorRef ref_A(block_A.get(), cutlass::layout::RowMajor::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(), cutlass::layout::RowMajor::packed({K, N}));
    cutlass::TensorRef ref_C(block_ref_D.get(), cutlass::layout::RowMajor::packed({M, N}));
    cutlass::TensorRef ref_D(block_ref_D.get(), cutlass::layout::RowMajor::packed({M, N}));

    cutlass::reference::device::GemmComplex(
      {M, N, K}, K2::ElementAcc(1), ref_A, cutlass::ComplexTransform::kNone,
      ref_B, cutlass::ComplexTransform::kNone, K2::ElementAcc(0),
      ref_C, ref_D, K2::ElementAcc(0), L, M * K, K * N, M * N, M * N);
    compat::wait();

    // Reference: scale, then SwiGLU
    // ref_D[m,n] = acc[m,n] * R[m], then SwiGLU on pairs
    {
      auto* ref_ptr = block_ref_D.get();
      auto* scale_ptr = block_scale.get();
      int64_t total = static_cast<int64_t>(M) * N * L;
      int n_val = N, m_val = M;
      // First: apply RMS scaling
      compat::get_default_queue().parallel_for(
        sycl::range<1>(total),
        [=](sycl::id<1> idx) {
          int64_t i = idx[0];
          int m = static_cast<int>((i / n_val) % m_val);
          int batch = static_cast<int>(i / (m_val * n_val));
          float val = static_cast<float>(ref_ptr[i]);
          float s = scale_ptr[batch * m_val + m];
          ref_ptr[i] = static_cast<K2::ElementD>(val * s);
        }
      );
    }
    compat::wait();

    // Copy scaled result for SwiGLU reference (need read-only copy)
    cutlass::DeviceAllocation<K2::ElementD> block_scaled(static_cast<size_t>(M) * N * L);
    compat::get_default_queue().memcpy(block_scaled.get(), block_ref_D.get(),
                                        static_cast<size_t>(M) * N * L * sizeof(K2::ElementD));
    compat::wait();

    // Apply SwiGLU: for each pair (2k, 2k+1), output = silu(gate) * up
    {
      auto* ref_ptr = block_ref_D.get();
      auto* src_ptr = block_scaled.get();
      int64_t total = static_cast<int64_t>(M) * N * L;
      int n_val = N;
      compat::get_default_queue().parallel_for(
        sycl::range<1>(total),
        [=](sycl::id<1> idx) {
          int64_t i = idx[0];
          int col = static_cast<int>(i % n_val);
          int64_t row_base = i - col;
          int even_col = col & ~1;
          int odd_col = even_col + 1;
          if (odd_col < n_val) {
            float gate = static_cast<float>(src_ptr[row_base + even_col]);
            float up = static_cast<float>(src_ptr[row_base + odd_col]);
            float silu_gate = gate / (1.0f + sycl::exp(-gate));
            ref_ptr[i] = static_cast<K2::ElementD>(silu_gate * up);
          }
        }
      );
    }
    compat::wait();

    // Compare (both even and odd columns should match since both lanes compute same value)
    bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(
      block_ref_D.get(), block_D.get(), block_D.size(),
      static_cast<K2::ElementD>(0.05f), static_cast<K2::ElementD>(0.05f));

    std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;
    if (!passed) return 1;
  } else {
    std::cout << "Disposition is skipped." << std::endl;
  }

  if (opts.iterations > 0) {
    GPU_Clock timer;
    timer.start();
    for (int i = 0; i < opts.iterations; ++i) gemm_op.run();
    compat::wait();

    float time_s = timer.seconds() / opts.iterations;
    double tflops = (2.0 * M * N * K * L) * 1e-12;
    std::cout << "Problem Size: " << M << 'x' << N << 'x' << K << 'x' << L << std::endl;
    printf("xe-fuse K2 (GEMM+RmsNorm+SwiGLU): [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / time_s, time_s * 1000);
  }

  return 0;
}
