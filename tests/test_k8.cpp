
// xe-fuse test: K8 — gemm_swiglu
// D = SwiGLU( acc )
// N must be even (interleaved gate/up). Both even/odd lanes produce the
// same SwiGLU value. Verification checks both lanes against reference.

#include "xe-fuse/kernels/gemm_swiglu.hpp"

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"

#include "sycl_common.hpp"
#include "helper.h"

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

using K8 = xe_fuse::GemmSwiGLU<>;
using GemmOp = K8::Gemm;

int main(int argc, const char** argv) {
  Options opts;
  opts.parse(argc, argv);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  int M = opts.m, N = opts.n, K = opts.k, L = opts.l;
  if ((N & 1) != 0) {
    printf("Error: N must be even for SwiGLU (interleaved gate/up). Got N=%d\n", N);
    return 1;
  }
  using StrideA = typename GemmOp::GemmKernel::StrideA;
  using StrideB = typename GemmOp::GemmKernel::StrideB;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, L));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, L));
  auto stride_C = cutlass::make_cute_packed_stride(K8::StrideC{}, make_shape(M, N, L));
  auto stride_D = cutlass::make_cute_packed_stride(K8::StrideD{}, make_shape(M, N, L));

  cutlass::DeviceAllocation<K8::ElementA> block_A(static_cast<size_t>(M) * K * L);
  cutlass::DeviceAllocation<K8::ElementB> block_B(static_cast<size_t>(K) * N * L);
  cutlass::DeviceAllocation<K8::ElementD> block_D(static_cast<size_t>(M) * N * L);
  cutlass::DeviceAllocation<K8::ElementD> block_ref_D(static_cast<size_t>(M) * N * L);

  initialize_block(block_A, 2023);
  initialize_block(block_B, 2022);

  auto evt_args = K8::make_evt_args();

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
      {M, N, K}, K8::ElementAcc(1), ref_A, cutlass::ComplexTransform::kNone,
      ref_B, cutlass::ComplexTransform::kNone, K8::ElementAcc(0),
      ref_C, ref_D, K8::ElementAcc(0), L, M * K, K * N, M * N, M * N);
    compat::wait();

    // Reference: apply SwiGLU directly to GEMM output (no scaling)
    {
      auto* ref_ptr = block_ref_D.get();
      cutlass::DeviceAllocation<K8::ElementD> block_gemm_out(static_cast<size_t>(M) * N * L);
      compat::get_default_queue().memcpy(block_gemm_out.get(), ref_ptr,
                                          static_cast<size_t>(M) * N * L * sizeof(K8::ElementD));
      compat::wait();

      auto* src_ptr = block_gemm_out.get();
      int n_val = N;
      compat::get_default_queue().parallel_for(
        sycl::range<1>(static_cast<size_t>(M) * N * L),
        [=](sycl::id<1> idx) {
          int64_t i = idx[0];
          int col = static_cast<int>(i % n_val);
          int64_t row_base = i - col;
          int even_col = col & ~1;
          int odd_col = even_col + 1;
          if (odd_col < n_val) {
            float gate = static_cast<float>(src_ptr[row_base + even_col]);
            float up   = static_cast<float>(src_ptr[row_base + odd_col]);
            float silu_gate = gate / (1.0f + sycl::exp(-gate));
            ref_ptr[i] = static_cast<K8::ElementD>(silu_gate * up);
          }
        }
      );
    }
    compat::wait();

    bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(
      block_ref_D.get(), block_D.get(), block_D.size(),
      static_cast<K8::ElementD>(0.05f), static_cast<K8::ElementD>(0.05f));

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
    printf("xe-fuse K8 (GEMM+SwiGLU): [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / time_s, time_s * 1000);
  }

  return 0;
}
