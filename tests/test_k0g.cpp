// xe-fuse test: K0g -- gemm_gate_residual_norm
// D[m,n] = gamma[n] * (gate[m] * acc[m,n] + residual[m,n])

#include "xe-fuse/kernels/gemm_gate_residual_norm.hpp"

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"

#include "sycl_common.hpp"
#include "helper.h"

#include <random>
#include <cstdio>

using namespace cute;

struct Options {
  int m = 512, n = 4096, k = 4096, l = 1;
  int iterations = 100;
  int verify = 1;

  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);
    cmd.get_cmd_line_argument("m", m, 512);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
    cmd.get_cmd_line_argument("verify", verify, 1);
  }
};

using K0g = xe_fuse::GemmGateResidualGamma<>;
using GemmOp = K0g::Gemm;

int main(int argc, const char** argv) {
  Options opts;
  opts.parse(argc, argv);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  int M = opts.m, N = opts.n, K = opts.k, L = opts.l;

  using StrideA = typename GemmOp::GemmKernel::StrideA;
  using StrideB = typename GemmOp::GemmKernel::StrideB;

  auto stride_A        = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, L));
  auto stride_B        = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, L));
  auto stride_C        = cutlass::make_cute_packed_stride(K0g::StrideC{}, make_shape(M, N, L));
  auto stride_D        = cutlass::make_cute_packed_stride(K0g::StrideD{}, make_shape(M, N, L));
  auto stride_residual = cutlass::make_cute_packed_stride(K0g::StrideResidual{}, make_shape(M, N, L));

  cutlass::DeviceAllocation<K0g::ElementA>        block_A(static_cast<size_t>(M) * K * L);
  cutlass::DeviceAllocation<K0g::ElementB>        block_B(static_cast<size_t>(K) * N * L);
  cutlass::DeviceAllocation<K0g::ElementD>        block_D(static_cast<size_t>(M) * N * L);
  cutlass::DeviceAllocation<K0g::ElementD>        block_ref_D(static_cast<size_t>(M) * N * L);
  cutlass::DeviceAllocation<K0g::ElementResidual> block_residual(static_cast<size_t>(M) * N * L);
  cutlass::DeviceAllocation<K0g::ElementGate>     block_gate(static_cast<size_t>(M) * L);
  cutlass::DeviceAllocation<K0g::ElementGamma>    block_gamma(static_cast<size_t>(N));

  initialize_block(block_A,        2023);
  initialize_block(block_B,        2022);
  initialize_block(block_residual, 2021);

  // gate: uniform (0.0, 1.0) to mimic timestep-dependent gating
  {
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> h(static_cast<size_t>(M) * L);
    for (auto& v : h) v = dist(rng);
    compat::get_default_queue().memcpy(block_gate.get(), h.data(), h.size() * sizeof(float));
  }

  // gamma: uniform (0.5, 1.5)
  {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.5f, 1.5f);
    std::vector<float> h(static_cast<size_t>(N));
    for (auto& v : h) v = dist(rng);
    compat::get_default_queue().memcpy(block_gamma.get(), h.data(), h.size() * sizeof(float));
  }
  compat::wait();

  auto evt_args = K0g::make_evt_args(
      block_gate.get(), M,
      block_residual.get(), stride_residual,
      block_gamma.get(), N);

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
    cutlass::DeviceAllocation<float> block_gemm_f32(static_cast<size_t>(M) * N * L);
    cutlass::TensorRef ref_A(block_A.get(), cutlass::layout::RowMajor::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(), cutlass::layout::RowMajor::packed({K, N}));
    cutlass::TensorRef ref_C_f32(block_gemm_f32.get(), cutlass::layout::RowMajor::packed({M, N}));
    cutlass::TensorRef ref_D_f32(block_gemm_f32.get(), cutlass::layout::RowMajor::packed({M, N}));

    cutlass::reference::device::GemmComplex(
        {M, N, K}, float(1), ref_A, cutlass::ComplexTransform::kNone,
        ref_B, cutlass::ComplexTransform::kNone, float(0),
        ref_C_f32, ref_D_f32, float(0), L, M * K, K * N, M * N, M * N);
    compat::wait();

    {
      auto* ref_ptr      = block_ref_D.get();
      auto* acc_ptr      = block_gemm_f32.get();
      auto* res_ptr      = block_residual.get();
      auto* gate_ptr     = block_gate.get();
      auto* gamma_ptr    = block_gamma.get();
      int n_val = N, m_val = M;

      compat::get_default_queue().parallel_for(
        sycl::range<1>(static_cast<size_t>(M) * N * L),
        [=](sycl::id<1> idx) {
          int64_t i   = idx[0];
          int col     = static_cast<int>(i % n_val);
          int row     = static_cast<int>((i / n_val) % m_val);
          int batch   = static_cast<int>(i / (static_cast<int64_t>(m_val) * n_val));

          float acc      = acc_ptr[i];
          float residual = static_cast<float>(res_ptr[i]);
          float gate     = gate_ptr[batch * m_val + row];
          float gamma    = gamma_ptr[col];

          ref_ptr[i] = static_cast<K0g::ElementD>(gamma * (gate * acc + residual));
        }
      );
    }
    compat::wait();

    bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(
        block_ref_D.get(), block_D.get(), block_D.size(),
        K0g::ElementD(0.05f), K0g::ElementD(0.05f));

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
    printf("xe-fuse K0g (GEMM+Gate+Residual+Gamma): [%4.3f]TFlop/s  (%6.4f)ms\n",
           tflops / time_s, time_s * 1000);
  }

  return 0;
}
