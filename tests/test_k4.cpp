
// xe-fuse test: K4 — gemm_rmsnorm_rope
// D = RoPE((A@B) * R, cos_sin)
// Composes RMSNorm scaling with RoPE rotation in a single GEMM epilogue.

#include "xe-fuse/kernels/gemm_rmsnorm_rope.hpp"

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

using K4 = xe_fuse::GemmRmsNormRoPE<>;
using GemmOp = K4::Gemm;

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
  auto stride_C = cutlass::make_cute_packed_stride(K4::StrideC{}, make_shape(M, N, L));
  auto stride_D = cutlass::make_cute_packed_stride(K4::StrideD{}, make_shape(M, N, L));
  auto stride_cs = cutlass::make_cute_packed_stride(K4::StrideCosSin{}, make_shape(M, N, L));

  cutlass::DeviceAllocation<K4::ElementA> block_A(static_cast<size_t>(M) * K * L);
  cutlass::DeviceAllocation<K4::ElementB> block_B(static_cast<size_t>(K) * N * L);
  cutlass::DeviceAllocation<K4::ElementD> block_D(static_cast<size_t>(M) * N * L);
  cutlass::DeviceAllocation<K4::ElementD> block_ref_D(static_cast<size_t>(M) * N * L);
  cutlass::DeviceAllocation<K4::ElementScale> block_scale(static_cast<size_t>(M) * L);
  cutlass::DeviceAllocation<K4::ElementCosSin> block_cos_sin(static_cast<size_t>(M) * N * L);

  initialize_block(block_A, 2023);
  initialize_block(block_B, 2022);

  // Initialize scale (per-row RMS normalization factor)
  std::vector<K4::ElementScale> h_scale(static_cast<size_t>(M) * L);
  {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.5f, 1.5f);
    for (auto& v : h_scale) v = dist(rng);
    compat::get_default_queue().memcpy(block_scale.get(), h_scale.data(),
                                        h_scale.size() * sizeof(K4::ElementScale));
  }

  // Initialize cos_sin with realistic RoPE frequencies
  std::vector<K4::ElementCosSin> h_cos_sin(static_cast<size_t>(M) * N * L);
  for (int batch = 0; batch < L; ++batch) {
    for (int m = 0; m < M; ++m) {
      for (int k_pair = 0; k_pair < N / 2; ++k_pair) {
        float freq = 1.0f / std::pow(10000.0f, 2.0f * k_pair / static_cast<float>(N));
        float angle = static_cast<float>(m) * freq;
        size_t base = static_cast<size_t>(batch) * M * N + static_cast<size_t>(m) * N;
        h_cos_sin[base + 2 * k_pair]     = std::cos(angle);
        h_cos_sin[base + 2 * k_pair + 1] = std::sin(angle);
      }
    }
  }
  compat::get_default_queue().memcpy(block_cos_sin.get(), h_cos_sin.data(),
                                      h_cos_sin.size() * sizeof(K4::ElementCosSin));
  compat::wait();

  auto evt_args = K4::make_evt_args(block_scale.get(), M, block_cos_sin.get(), stride_cs);

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
    // Reference GEMM in float
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

    // Reference: scale by R, then RoPE
    {
      auto* ref_ptr = block_ref_D.get();
      auto* src_ptr = block_gemm_f32.get();
      auto* s_ptr = block_scale.get();
      auto* cs_ptr = block_cos_sin.get();
      int n_val = N, m_val = M;
      int64_t total = static_cast<int64_t>(M) * N * L;

      compat::get_default_queue().parallel_for(
        sycl::range<1>(total),
        [=](sycl::id<1> idx) {
          int64_t i = idx[0];
          int col = static_cast<int>(i % n_val);
          int row = static_cast<int>((i / n_val) % m_val);
          int batch = static_cast<int>(i / (m_val * n_val));
          int64_t row_base = i - col;

          // RMS scale
          float scale = s_ptr[batch * m_val + row];
          int even_col = col & ~1;
          int odd_col = even_col + 1;
          if (odd_col >= n_val) {
            ref_ptr[i] = static_cast<K4::ElementD>(src_ptr[i] * scale);
            return;
          }

          float x_even = src_ptr[row_base + even_col] * scale;
          float x_odd  = src_ptr[row_base + odd_col] * scale;
          float cos_val = cs_ptr[row_base + even_col];
          float sin_val = cs_ptr[row_base + odd_col];

          float out;
          if ((col & 1) == 0) {
            out = x_even * cos_val + x_odd * sin_val;
          } else {
            out = -x_even * sin_val + x_odd * cos_val;
          }
          ref_ptr[i] = static_cast<K4::ElementD>(out);
        }
      );
    }
    compat::wait();

    bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(
      block_ref_D.get(), block_D.get(), block_D.size(),
      static_cast<K4::ElementD>(0.05f), static_cast<K4::ElementD>(0.05f));

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
    printf("xe-fuse K4 (GEMM+RmsNorm+RoPE): [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / time_s, time_s * 1000);
  }

  return 0;
}
