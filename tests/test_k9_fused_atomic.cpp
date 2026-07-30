
// xe-fuse test: K9_fused — gemm_partial_cross_entropy (fully-fused epilogue)
//
// Tests GemmPartialCrossEntropyFused: GEMM epilogue atomically writes lse[M]
// directly during the GEMM, eliminating the post-GEMM re-read of M×N logits.

#include "xe-fuse/kernels/gemm_partial_cross_entropy_fused_atomic.hpp"

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

using K9F = xe_fuse::GemmPartialCrossEntropyFused<>;
using GemmOp = K9F::Gemm;

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
  auto stride_C = cutlass::make_cute_packed_stride(K9F::StrideC{}, make_shape(M, N, L));
  auto stride_D = cutlass::make_cute_packed_stride(K9F::StrideD{}, make_shape(M, N, L));

  cutlass::DeviceAllocation<K9F::ElementA> block_A(static_cast<size_t>(M) * K * L);
  cutlass::DeviceAllocation<K9F::ElementB> block_B(static_cast<size_t>(K) * N * L);
  cutlass::DeviceAllocation<K9F::ElementD> block_D(static_cast<size_t>(M) * N * L);
  cutlass::DeviceAllocation<K9F::ElementD> block_ref_D(static_cast<size_t>(M) * N * L);

  cutlass::DeviceAllocation<float> block_lse(static_cast<size_t>(M) * L);
  cutlass::DeviceAllocation<float> block_ref_lse(static_cast<size_t>(M) * L);
  cutlass::DeviceAllocation<float> block_logits_tgt(static_cast<size_t>(M) * L);
  cutlass::DeviceAllocation<float> block_ref_logits_tgt(static_cast<size_t>(M) * L);
  cutlass::DeviceAllocation<int>   block_targets(static_cast<size_t>(M) * L);

  initialize_block(block_A, 2023);
  initialize_block(block_B, 2022);

  std::mt19937 rng(42);
  std::vector<int> h_targets(static_cast<size_t>(M) * L);
  std::uniform_int_distribution<int> dist(0, N - 1);
  for (auto& t : h_targets) t = dist(rng);
  compat::get_default_queue().memcpy(block_targets.get(), h_targets.data(),
                                      h_targets.size() * sizeof(int));
  compat::wait();

  // lse buffer is passed directly to the fused epilogue (no partial buffer needed).
  // initialize_workspace fills it with -infinity before the first run.
  auto evt_args = K9F::make_evt_args(block_lse.get(), M, N, L);

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
  // xe_epilogue::initialize_workspace is a no-op; fill lse with -inf manually
  // before each run so atomic logsumexp starts from the identity element.
  auto q = compat::get_default_queue();
  q.fill(block_lse.get(), -std::numeric_limits<float>::infinity(),
         static_cast<size_t>(M) * L);
  CUTLASS_CHECK(gemm_op.run());
  compat::wait();

  K9F::launch_select_logits(q, block_D.get(), block_targets.get(),
                              block_logits_tgt.get(), M, N, L);
  compat::wait();

  if (opts.verify) {
    // Reference GEMM
    cutlass::TensorRef ref_A(block_A.get(), cutlass::layout::RowMajor::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(), cutlass::layout::RowMajor::packed({K, N}));
    cutlass::TensorRef ref_C(block_ref_D.get(), cutlass::layout::RowMajor::packed({M, N}));
    cutlass::TensorRef ref_D(block_ref_D.get(), cutlass::layout::RowMajor::packed({M, N}));

    cutlass::reference::device::GemmComplex(
      {M, N, K}, K9F::ElementAcc(1), ref_A, cutlass::ComplexTransform::kNone,
      ref_B, cutlass::ComplexTransform::kNone, K9F::ElementAcc(0),
      ref_C, ref_D, K9F::ElementAcc(0), L, M * K, K * N, M * N, M * N);
    compat::wait();

    xe_fuse::standalone::compute_lse(q, block_ref_D.get(), block_ref_lse.get(), M, N, L);
    xe_fuse::standalone::select_logits(q, block_ref_D.get(), block_targets.get(),
                                        block_ref_logits_tgt.get(), M, N, L);
    compat::wait();

    bool d_passed = cutlass::reference::device::BlockCompareRelativelyEqual(
      block_ref_D.get(), block_D.get(), block_D.size(),
      static_cast<K9F::ElementD>(0.05f), static_cast<K9F::ElementD>(0.05f));

    std::vector<float> h_lse(M * L), h_ref_lse(M * L);
    std::vector<float> h_tgt(M * L), h_ref_tgt(M * L);
    q.memcpy(h_lse.data(), block_lse.get(), M * L * sizeof(float));
    q.memcpy(h_ref_lse.data(), block_ref_lse.get(), M * L * sizeof(float));
    q.memcpy(h_tgt.data(), block_logits_tgt.get(), M * L * sizeof(float));
    q.memcpy(h_ref_tgt.data(), block_ref_logits_tgt.get(), M * L * sizeof(float));
    compat::wait();

    auto check_vec = [&](const char* name, const std::vector<float>& got,
                          const std::vector<float>& ref) {
      int errs = 0;
      double max_rel = 0.0;
      for (int i = 0; i < (int)got.size(); ++i) {
        double rel = std::abs(ref[i]) > 1e-6
                     ? std::abs(got[i] - ref[i]) / std::abs(ref[i]) : 0.0;
        max_rel = std::max(max_rel, rel);
        if (rel > 0.05 && errs < 3)
          printf("  %s[%d]: got=%.5f ref=%.5f rel=%.4f\n", name, i, got[i], ref[i], (float)rel);
        if (rel > 0.05) ++errs;
      }
      printf("%s: %s (max_rel=%.5f, errors=%d/%d)\n",
             name, errs == 0 ? "Passed" : "FAILED", max_rel, errs, (int)got.size());
      return errs == 0;
    };

    bool lse_ok = check_vec("lse", h_lse, h_ref_lse);
    bool tgt_ok = check_vec("logits_tgt", h_tgt, h_ref_tgt);

    bool passed = d_passed && lse_ok && tgt_ok;
    std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;
    if (!passed) return 1;
  } else {
    std::cout << "Disposition is skipped." << std::endl;
  }

  if (opts.iterations > 0) {
    GPU_Clock timer;
    timer.start();
    for (int i = 0; i < opts.iterations; ++i) {
      // Re-initialize lse to -inf before each run (atomic logsumexp accumulates).
      q.fill(block_lse.get(), -std::numeric_limits<float>::infinity(),
             static_cast<size_t>(M) * L);
      gemm_op.run();
      K9F::launch_select_logits(q, block_D.get(), block_targets.get(),
                                  block_logits_tgt.get(), M, N, L);
    }
    compat::wait();

    float time_s = timer.seconds() / opts.iterations;
    double tflops = (2.0 * M * N * K * L) * 1e-12;
    std::cout << "Problem Size: " << M << 'x' << N << 'x' << K << 'x' << L << std::endl;
    printf("xe-fuse K9_fused (GEMM+LSE+SelectLogits fused): [%4.3f]TFlop/s  (%6.4f)ms\n",
           tflops / time_s, time_s * 1000);
  }

  return 0;
}
