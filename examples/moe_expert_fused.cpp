
// xe-fuse example: MoE Expert Batched GEMM + Fused SwiGLU
//
// Demonstrates batching all routed experts into a single kernel launch
// using the CUTLASS L (batch) dimension. Each expert's gate+up projection
// is fused with RMSNorm scaling and SwiGLU activation in the epilogue —
// no separate kernel launches for post-ops.
//
// LLaMA 4 Scout: 16 experts, H=5120, expert_I=8192
//   Per-expert GEMM: [M_exp, 16384, 5120] with SwiGLU epilogue
//   Single launch: L=16 processes all experts simultaneously
//
// Build:
//   icpx -fsycl -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET \
//       -I $SYCL_TLA/include -I $SYCL_TLA/tools/util/include \
//       -I $SYCL_TLA/examples/common -I $SYCL_TLA/applications \
//       -I $SYCL_TLA/applications/xe-fuse/include \
//       -O2 -std=c++17 -fsycl-targets=spir64_gen \
//       -o moe_expert_fused moe_expert_fused.cpp
//
// Run:
//   ./moe_expert_fused --m=32 --experts=16

#include "xe-fuse/kernels/gemm_moe_expert.hpp"

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

#include "sycl_common.hpp"
#include "helper.h"

using namespace cute;

int main(int argc, const char** argv) {
  // --- Parse args ---
  int M_exp = 32;       // tokens per expert
  int H = 5120;         // hidden dim
  int I = 8192;         // expert intermediate size
  int num_experts = 16; // LLaMA 4 Scout
  int iterations = 200;

  cutlass::CommandLine cmd(argc, argv);
  cmd.get_cmd_line_argument("m", M_exp, 32);
  cmd.get_cmd_line_argument("h", H, 5120);
  cmd.get_cmd_line_argument("i", I, 8192);
  cmd.get_cmd_line_argument("experts", num_experts, 16);
  cmd.get_cmd_line_argument("iterations", iterations, 200);

  int N = 2 * I;  // gate + up projection stacked
  int K = H;
  int L = num_experts;

  printf("MoE Expert Fused GEMM + SwiGLU\n");
  printf("  M_per_expert=%d  N=%d (2*I=%d)  K=%d (H=%d)  experts=%d\n",
         M_exp, N, I, K, H, num_experts);
  printf("  Total tokens: %d  Total FLOPs: %.3f TFlop\n",
         M_exp * num_experts, 2.0 * M_exp * N * K * L * 1e-12);

  // --- Configure kernel ---
  // Tile shape: 64x128x32 works well for small M_exp (decode batches)
  using TileShape = Shape<_64, _128, _32>;

  using Config = xe_fuse::MoEExpertSwiGLU<
      cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t,
      float, float, float, TileShape>;
  using GemmOp = Config::Gemm;

  // --- Allocate stacked expert tensors ---
  // A[l]: input tokens for expert l    — [M_exp, K] × L
  // B[l]: expert weights (gate+up)     — [N, K] × L
  // D[l]: output after SwiGLU          — [M_exp, N] × L
  // scale[l]: RMSNorm scale per expert — [M_exp] × L
  cutlass::DeviceAllocation<cutlass::bfloat16_t> block_A(size_t(M_exp) * K * L);
  cutlass::DeviceAllocation<cutlass::bfloat16_t> block_B(size_t(K) * N * L);
  cutlass::DeviceAllocation<cutlass::bfloat16_t> block_D(size_t(M_exp) * N * L);
  cutlass::DeviceAllocation<float> block_scale(size_t(M_exp) * L);

  initialize_block(block_A, 2023);
  initialize_block(block_B, 2022);
  initialize_block(block_scale, 42);

  // --- Set up GEMM arguments ---
  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  using StrideA = typename GemmOp::GemmKernel::StrideA;
  using StrideB = typename GemmOp::GemmKernel::StrideB;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M_exp, K, L));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, L));
  auto stride_C = cutlass::make_cute_packed_stride(typename Config::StrideC{}, make_shape(M_exp, N, L));
  auto stride_D = cutlass::make_cute_packed_stride(typename Config::StrideD{}, make_shape(M_exp, N, L));

  // EVT args: per-row RMSNorm scale + SwiGLU activation
  auto evt_args = Config::make_evt_args(block_scale.get(), M_exp);

  typename GemmOp::GemmKernel::EpilogueArguments epi_args{
    evt_args, nullptr, stride_C, block_D.get(), stride_D
  };
  typename GemmOp::GemmKernel::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M_exp, N, K, L},   // L = num_experts: all experts in one launch
    {block_A.get(), stride_A, block_B.get(), stride_B},
    epi_args, hw_info
  };

  // --- Initialize and run ---
  GemmOp gemm_op;
  size_t ws = GemmOp::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(ws);

  auto status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    printf("ERROR: can_implement failed (%d)\n", int(status));
    return 1;
  }

  CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm_op.run());
  compat::wait();

  printf("  Warmup: OK\n");

  // --- Benchmark ---
  GPU_Clock timer;
  timer.start();
  for (int i = 0; i < iterations; ++i) gemm_op.run();
  compat::wait();
  float t = timer.seconds() / iterations;

  double tflops = (2.0 * M_exp * N * K * L) * 1e-12;
  printf("\n  Batched (%d experts, 1 kernel launch):\n", L);
  printf("    Time:       %.4f ms\n", t * 1000);
  printf("    Throughput: %.1f TFlop/s\n", tflops / t);
  printf("    Per-expert: %.4f ms\n", t * 1000 / L);

  return 0;
}
