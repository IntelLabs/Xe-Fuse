// xe-fuse test: K2_FP8 — gemm_fp8_dequant_swiglu
// D = SwiGLU( dequant(A_f8 @ B_f8) )
//
// FP8×FP8 GEMM on BMG-G31: FP8 inputs are upcasted to FP16 before the XMX16
// MMA, which accumulates in float. Dequantization applies per-token scale_a[m]
// and per-channel scale_b[n] in the epilogue. SwiGLU follows.
//
// Reference:
//   acc[m,n] = sum_k( float(A_f8[m,k]) * float(B_f8[k,n]) )
//   dequant[m,n] = acc[m,n] * scale_a[m] * scale_b[n]
//   D[m, 2i]   = silu(dequant[m, 2i]) * dequant[m, 2i+1]
//   D[m, 2i+1] = silu(dequant[m, 2i]) * dequant[m, 2i+1]
//
// Verification tolerance is looser than W8A8 (FP8 range ≈ ±448 vs INT8 ±127).

#include "xe-fuse/kernels/gemm_fp8_dequant.hpp"

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/tensor_compare.h"

#include "sycl_common.hpp"
#include "helper.h"

#include <random>
#include <cmath>
#include <vector>
#include <cstdio>

using namespace cute;

struct Options {
  int m = 512, n = 28672, k = 4096, l = 1;
  int iterations = 100;
  int verify = 1;

  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);
    cmd.get_cmd_line_argument("m", m, 512);
    cmd.get_cmd_line_argument("n", n, 28672);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
    cmd.get_cmd_line_argument("verify", verify, 1);
  }
};

using K2FP8  = xe_fuse::GemmFP8DequantSwiGLU<>;
using GemmOp = K2FP8::Gemm;
using ElementFP8 = K2FP8::ElementA;  // float_e4m3_t

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
  auto stride_C = cutlass::make_cute_packed_stride(K2FP8::StrideC{}, make_shape(M, N, L));
  auto stride_D = cutlass::make_cute_packed_stride(K2FP8::StrideD{}, make_shape(M, N, L));

  cutlass::DeviceAllocation<ElementFP8>       block_A(static_cast<size_t>(M) * K * L);
  cutlass::DeviceAllocation<ElementFP8>       block_B(static_cast<size_t>(K) * N * L);
  cutlass::DeviceAllocation<K2FP8::ElementD>  block_D(static_cast<size_t>(M) * N * L);
  cutlass::DeviceAllocation<K2FP8::ElementD>  block_ref_D(static_cast<size_t>(M) * N * L);

  cutlass::DeviceAllocation<float> block_scale_a(static_cast<size_t>(M) * L);
  cutlass::DeviceAllocation<float> block_scale_b(static_cast<size_t>(N) * L);

  // Float working buffers for reference GEMM
  cutlass::DeviceAllocation<float> block_A_f32(static_cast<size_t>(M) * K * L);
  cutlass::DeviceAllocation<float> block_B_f32(static_cast<size_t>(K) * N * L);
  cutlass::DeviceAllocation<float> block_acc_f32(static_cast<size_t>(M) * N * L);

  // Initialize FP8 A and B on host with values in [-4, 4] (well within E4M3 range)
  {
    std::mt19937 rng_a(2001), rng_b(2002);
    std::uniform_real_distribution<float> dist(-4.f, 4.f);

    std::vector<ElementFP8> h_A(static_cast<size_t>(M) * K * L);
    std::vector<ElementFP8> h_B(static_cast<size_t>(K) * N * L);
    for (auto& v : h_A) v = ElementFP8(dist(rng_a));
    for (auto& v : h_B) v = ElementFP8(dist(rng_b));

    compat::get_default_queue().memcpy(block_A.get(), h_A.data(), h_A.size() * sizeof(ElementFP8));
    compat::get_default_queue().memcpy(block_B.get(), h_B.data(), h_B.size() * sizeof(ElementFP8));
  }

  // Scales: small positive floats in [0.001, 0.01]
  {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.001f, 0.01f);

    std::vector<float> h_sa(static_cast<size_t>(M) * L);
    std::vector<float> h_sb(static_cast<size_t>(N) * L);
    for (auto& v : h_sa) v = dist(rng);
    for (auto& v : h_sb) v = dist(rng);

    compat::get_default_queue().memcpy(block_scale_a.get(), h_sa.data(), h_sa.size() * sizeof(float));
    compat::get_default_queue().memcpy(block_scale_b.get(), h_sb.data(), h_sb.size() * sizeof(float));
  }
  compat::wait();

  // ── Run kernel ────────────────────────────────────────────────────────────
  auto evt_args = K2FP8::make_evt_args(
      block_scale_a.get(), M,
      block_scale_b.get(), N);

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

  // ── Reference ─────────────────────────────────────────────────────────────
  if (opts.verify) {
    // FP8 → float32 upcast on GPU
    {
      const ElementFP8* a_ptr = block_A.get();
      const ElementFP8* b_ptr = block_B.get();
      float* af = block_A_f32.get();
      float* bf = block_B_f32.get();
      int64_t na = static_cast<int64_t>(M) * K * L;
      int64_t nb = static_cast<int64_t>(K) * N * L;

      compat::get_default_queue().parallel_for(sycl::range<1>(na), [=](sycl::id<1> idx) {
        af[idx[0]] = static_cast<float>(a_ptr[idx[0]]);
      });
      compat::get_default_queue().parallel_for(sycl::range<1>(nb), [=](sycl::id<1> idx) {
        bf[idx[0]] = static_cast<float>(b_ptr[idx[0]]);
      });
      compat::wait();
    }

    // Float GEMM reference
    {
      const float* af  = block_A_f32.get();
      const float* bf  = block_B_f32.get();
      float*       acc = block_acc_f32.get();
      int M_ = M, N_ = N, K_ = K, L_ = L;
      compat::get_default_queue().parallel_for(
        sycl::range<1>(static_cast<size_t>(M) * N * L),
        [=](sycl::id<1> idx) {
          int64_t i   = idx[0];
          int col     = static_cast<int>(i % N_);
          int row     = static_cast<int>((i / N_) % M_);
          int batch   = static_cast<int>(i / ((int64_t)M_ * N_));
          float sum   = 0.f;
          for (int k = 0; k < K_; ++k)
            sum += af[batch * M_ * K_ + row * K_ + k]
                 * bf[batch * K_ * N_ + k * N_ + col];
          acc[i] = sum;
        }
      );
      compat::wait();
    }

    // Dequant + SwiGLU reference
    {
      const float*  acc = block_acc_f32.get();
      const float*  sa  = block_scale_a.get();
      const float*  sb  = block_scale_b.get();
      auto*         ref = block_ref_D.get();
      int M_ = M, N_ = N, L_ = L;
      compat::get_default_queue().parallel_for(
        sycl::range<1>(static_cast<size_t>(M) * N * L),
        [=](sycl::id<1> idx) {
          int64_t i    = idx[0];
          int col      = static_cast<int>(i % N_);
          int row      = static_cast<int>((i / N_) % M_);
          int batch    = static_cast<int>(i / ((int64_t)M_ * N_));
          int64_t base = static_cast<int64_t>(batch) * M_ * N_;

          int even_col = col & ~1;
          int odd_col  = even_col + 1;
          if (odd_col >= N_) {
            float v = acc[i] * sa[batch * M_ + row] * sb[batch * N_ + col];
            ref[i] = static_cast<K2FP8::ElementD>(v);
            return;
          }
          float gate = acc[base + row * N_ + even_col]
                     * sa[batch * M_ + row] * sb[batch * N_ + even_col];
          float up   = acc[base + row * N_ + odd_col]
                     * sa[batch * M_ + row] * sb[batch * N_ + odd_col];
          float silu_gate = gate / (1.f + sycl::exp(-gate));
          ref[i] = static_cast<K2FP8::ElementD>(silu_gate * up);
        }
      );
      compat::wait();
    }

    // FP8 rounding introduces ~0.5 ULP per multiply; tolerate 15% relative error
    bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(
        block_ref_D.get(), block_D.get(), block_D.size(),
        static_cast<K2FP8::ElementD>(0.15f), static_cast<K2FP8::ElementD>(0.05f));

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
    printf("Problem: %dx%dx%dx%d\n", M, N, K, L);
    printf("xe-fuse K2_FP8 (FP8 GEMM+Dequant+SwiGLU): [%4.3f]TFlop/s  (%6.4f)ms\n",
           tflops / time_s, time_s * 1000);
  }

  return 0;
}
