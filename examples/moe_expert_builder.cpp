
// xe-fuse example: MoE Expert GEMM using the Builder API
//
// Shows that MoE batched experts need NO special kernel — the standard builder
// API works directly. Set L=num_experts to batch all experts in one launch.
//
// This is the same builder composition used for dense models:
//   using EVT = b::SwiGLU<b::ScaleRows<b::Acc, TileShape, float>>;
//   using Kernel = b::MakeGemm<EVT, bf16, bf16, bf16, float, float, TileShape>;
//
// The only difference from a dense FFN GEMM is the launch shape:
//   Dense:  {M, N=2*I, K=H, L=1}
//   MoE:    {M_per_expert, N=2*I, K=H, L=num_experts}
//
// Build:
//   icpx -fsycl -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET \
//       -I $SYCL_TLA/include -I $SYCL_TLA/tools/util/include \
//       -I $SYCL_TLA/examples/common -I $SYCL_TLA/applications \
//       -I $SYCL_TLA/applications/xe-fuse/include \
//       -O2 -std=c++17 -fsycl-targets=spir64_gen \
//       -o moe_builder moe_expert_builder.cpp
//
// Run:
//   ./moe_builder                          # LLaMA 4 Scout defaults
//   ./moe_builder --m=64 --experts=128     # LLaMA 4 Maverick

#include "xe-fuse/builder/epilogue_builder.hpp"

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

#include "sycl_common.hpp"
#include "helper.h"

using namespace cute;
using bf16 = cutlass::bfloat16_t;
namespace b = xe_fuse::builder;

int main(int argc, const char** argv) {
  int M_exp = 32, H = 5120, I = 8192, num_experts = 16, iterations = 200;

  cutlass::CommandLine cmd(argc, argv);
  cmd.get_cmd_line_argument("m", M_exp, 32);
  cmd.get_cmd_line_argument("experts", num_experts, 16);
  cmd.get_cmd_line_argument("iterations", iterations, 200);

  int N = 2 * I;
  int L = num_experts;

  printf("MoE Expert GEMM + SwiGLU (Builder API)\n");
  printf("  M_exp=%d  N=%d  K=%d  L=%d (experts)\n\n", M_exp, N, H, L);

  // --- The builder composition is identical to dense FFN ---
  using TileShape = Shape<_64, _128, _32>;
  using EVT = b::SwiGLU<b::ScaleRows<b::Acc, TileShape, float>>;
  using Kernel = b::MakeGemm<EVT, bf16, bf16, bf16, float, float, TileShape>;
  using Gemm = typename Kernel::Gemm;

  // --- Allocate: stacked expert tensors along L dimension ---
  cutlass::DeviceAllocation<bf16>  block_A(size_t(M_exp) * H * L);
  cutlass::DeviceAllocation<bf16>  block_B(size_t(H) * N * L);
  cutlass::DeviceAllocation<bf16>  block_D(size_t(M_exp) * N * L);
  cutlass::DeviceAllocation<float> block_scale(size_t(M_exp) * L);

  initialize_block(block_A, 2023);
  initialize_block(block_B, 2022);
  initialize_block(block_scale, 42);

  // --- EVT args: RMSNorm scale + SwiGLU ---
  typename b::Acc::Arguments acc_args{};
  typename b::ColBroadcast<0, TileShape, float>::Arguments scale_args;
  scale_args.ptr_col = block_scale.get();
  scale_args.null_default = 1.0f;
  scale_args.dCol = {Int<1>{}, Int<0>{}, int64_t(M_exp)};

  typename b::MulOp<>::Arguments mul_args{};
  using RmsTree = b::ScaleRows<b::Acc, TileShape, float>;
  typename RmsTree::Arguments rms_args{acc_args, scale_args, mul_args};
  typename xe_fuse::XePairwiseCompute<xe_fuse::SwiGLUFn>::Arguments swiglu_args{};
  typename EVT::Arguments evt_args{rms_args, swiglu_args};

  // --- GEMM arguments: L=num_experts is the only MoE-specific part ---
  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = cute::Stride<int64_t, Int<1>, int64_t>;
  using StrideD = cute::Stride<int64_t, Int<1>, int64_t>;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M_exp, H, L));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, H, L));
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, make_shape(M_exp, N, L));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M_exp, N, L));

  typename Gemm::GemmKernel::EpilogueArguments epi_args{
    evt_args, nullptr, stride_C, block_D.get(), stride_D
  };
  typename Gemm::GemmKernel::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M_exp, N, H, L},    // <-- L=num_experts: batches all experts
    {block_A.get(), stride_A, block_B.get(), stride_B},
    epi_args, hw_info
  };

  // --- Run ---
  Gemm gemm_op;
  size_t ws = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(ws);

  CUTLASS_CHECK(gemm_op.can_implement(arguments));
  CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm_op.run());
  compat::wait();

  GPU_Clock timer;
  timer.start();
  for (int i = 0; i < iterations; ++i) gemm_op.run();
  compat::wait();
  float t = timer.seconds() / iterations;

  double tflops = (2.0 * M_exp * N * H * L) * 1e-12;
  printf("  %d experts in 1 launch: %.4f ms, %.1f TFlop/s\n", L, t*1000, tflops/t);
  printf("  Per-expert: %.4f ms\n", t*1000/L);

  return 0;
}
