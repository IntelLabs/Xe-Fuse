
// xe-fuse test: K2_W8A8 — gemm_dequant_swiglu
// D = SwiGLU( dequant(A_i8 @ B_i8) )
// INT8×INT8 GEMM with W8A8 dequantization and SwiGLU fused in a single epilogue.
//
// Reference:
//   acc_f32[m,n] = sum_k( float(A_i8[m,k]) * float(B_i8[k,n]) )
//   dequant[m,n] = acc_f32[m,n] * scale_token[m] * scale_channel[n]
//   D[m, 2i]   = silu(dequant[m, 2i]) * dequant[m, 2i+1]
//   D[m, 2i+1] = silu(dequant[m, 2i]) * dequant[m, 2i+1]  (same value)

#include "xe-fuse/kernels/gemm_dequant_swiglu.hpp"

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/tensor_compare.h"

#include "sycl_common.hpp"
#include "helper.h"

#include <random>
#include <cmath>

using namespace cute;

struct Options {
  int m = 512, n = 28672, k = 4096, l = 1;  // LLaMA 3 8B FFN dims
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

using K2W8A8 = xe_fuse::GemmDequantSwiGLU<>;
using GemmOp = K2W8A8::Gemm;

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
  auto stride_C = cutlass::make_cute_packed_stride(K2W8A8::StrideC{}, make_shape(M, N, L));
  auto stride_D = cutlass::make_cute_packed_stride(K2W8A8::StrideD{}, make_shape(M, N, L));

  cutlass::DeviceAllocation<int8_t>  block_A(static_cast<size_t>(M) * K * L);
  cutlass::DeviceAllocation<int8_t>  block_B(static_cast<size_t>(K) * N * L);
  cutlass::DeviceAllocation<K2W8A8::ElementD> block_D(static_cast<size_t>(M) * N * L);
  cutlass::DeviceAllocation<K2W8A8::ElementD> block_ref_D(static_cast<size_t>(M) * N * L);

  cutlass::DeviceAllocation<float>   block_scale_token(static_cast<size_t>(M) * L);
  cutlass::DeviceAllocation<float>   block_scale_channel(static_cast<size_t>(N) * L);

  cutlass::DeviceAllocation<float>   block_A_f32(static_cast<size_t>(M) * K * L);
  cutlass::DeviceAllocation<float>   block_B_f32(static_cast<size_t>(K) * N * L);
  cutlass::DeviceAllocation<float>   block_acc_f32(static_cast<size_t>(M) * N * L);

  {
    std::vector<int8_t> h_A(static_cast<size_t>(M) * K * L);
    std::vector<int8_t> h_B(static_cast<size_t>(K) * N * L);
    std::mt19937 rng_a(2001), rng_b(2002);
    std::uniform_int_distribution<int> dist(-64, 63);
    for (auto& v : h_A) v = static_cast<int8_t>(dist(rng_a));
    for (auto& v : h_B) v = static_cast<int8_t>(dist(rng_b));
    compat::get_default_queue().memcpy(block_A.get(), h_A.data(), h_A.size());
    compat::get_default_queue().memcpy(block_B.get(), h_B.data(), h_B.size());
  }

  {
    std::vector<float> h_st(static_cast<size_t>(M) * L);
    std::vector<float> h_sc(static_cast<size_t>(N) * L);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.001f, 0.01f);
    for (auto& v : h_st) v = dist(rng);
    for (auto& v : h_sc) v = dist(rng);
    compat::get_default_queue().memcpy(block_scale_token.get(), h_st.data(), h_st.size() * sizeof(float));
    compat::get_default_queue().memcpy(block_scale_channel.get(), h_sc.data(), h_sc.size() * sizeof(float));
  }
  compat::wait();

  auto evt_args = K2W8A8::make_evt_args(
      block_scale_token.get(), M,
      block_scale_channel.get(), N);

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
    // Convert INT8 → float32
    {
      const int8_t* a_ptr = block_A.get();
      float* af_ptr = block_A_f32.get();
      int64_t na = static_cast<int64_t>(M) * K * L;
      compat::get_default_queue().parallel_for(sycl::range<1>(na), [=](sycl::id<1> idx) {
        af_ptr[idx[0]] = static_cast<float>(a_ptr[idx[0]]);
      });
      const int8_t* b_ptr = block_B.get();
      float* bf_ptr = block_B_f32.get();
      int64_t nb = static_cast<int64_t>(K) * N * L;
      compat::get_default_queue().parallel_for(sycl::range<1>(nb), [=](sycl::id<1> idx) {
        bf_ptr[idx[0]] = static_cast<float>(b_ptr[idx[0]]);
      });
      compat::wait();
    }

    // Reference GEMM in float
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
          int batch   = static_cast<int>(i / (M_ * N_));
          float sum   = 0.f;
          for (int k = 0; k < K_; ++k)
            sum += af[batch * M_ * K_ + row * K_ + k] * bf[batch * K_ * N_ + k * N_ + col];
          acc[i] = sum;
        }
      );
      compat::wait();
    }

    // Apply dequant + SwiGLU reference
    // SwiGLU: out[2i] = out[2i+1] = silu(gate) * up
    //         gate = dequant[2i], up = dequant[2i+1]
    {
      const float* acc  = block_acc_f32.get();
      const float* st   = block_scale_token.get();
      const float* sc   = block_scale_channel.get();
      auto*        ref  = block_ref_D.get();
      int M_ = M, N_ = N, L_ = L;
      compat::get_default_queue().parallel_for(
        sycl::range<1>(static_cast<size_t>(M) * N * L),
        [=](sycl::id<1> idx) {
          int64_t i   = idx[0];
          int col     = static_cast<int>(i % N_);
          int row     = static_cast<int>((i / N_) % M_);
          int batch   = static_cast<int>(i / (M_ * N_));
          int64_t base = static_cast<int64_t>(batch) * M_ * N_;

          int even_col = col & ~1;
          int odd_col  = even_col + 1;
          if (odd_col >= N_) {
            float v = acc[i] * st[batch * M_ + row] * sc[batch * N_ + col];
            ref[i] = static_cast<K2W8A8::ElementD>(v);
            return;
          }
          float gate = acc[base + row * N_ + even_col]
                     * st[batch * M_ + row] * sc[batch * N_ + even_col];
          float up   = acc[base + row * N_ + odd_col]
                     * st[batch * M_ + row] * sc[batch * N_ + odd_col];
          // silu(gate) * up
          float silu_gate = gate / (1.f + sycl::exp(-gate));
          float out = silu_gate * up;
          ref[base - batch * M_ * N_ + row * N_ + even_col] = static_cast<K2W8A8::ElementD>(out);
          ref[base - batch * M_ * N_ + row * N_ + odd_col]  = static_cast<K2W8A8::ElementD>(out);
          // Note: both lanes get the same value; indexed by the even col for atomicity.
          // The parallel_for will write even and odd separately but with the same value.
          (void)col;  // col is redundant here; both branches write same value
        }
      );
      compat::wait();
    }

    bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(
      block_ref_D.get(), block_D.get(), block_D.size(),
      static_cast<K2W8A8::ElementD>(0.15f), static_cast<K2W8A8::ElementD>(0.05f));

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
    double tops = (2.0 * M * N * K * L) * 1e-12;
    std::cout << "Problem Size: " << M << 'x' << N << 'x' << K << 'x' << L << std::endl;
    printf("xe-fuse K2_W8A8 (INT8 GEMM+Dequant+SwiGLU): [%4.3f]TOp/s  (%6.4f)ms\n",
           tops / time_s, time_s * 1000);
  }

  return 0;
}
