# xe-fuse

<img src="assets/xe-fuse-logo.png" alt="xe-fuse banner" width="100%"/>

[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/IntelLabs/Xe-Fuse/badge)](https://scorecard.dev/viewer/?uri=github.com/IntelLabs/Xe-Fuse)

> ⚠️ **Disclaimer**: This project is currently in active development. The code is **not stable** and **not intended for use in production environments**. Interfaces, features, and behaviors are subject to change without notice.

GEMM epilogue fusion framework for Intel Xe GPUs, built on sycl-tla.

Fuses memory-bound Transformer operations (RMSNorm, SwiGLU, RoPE, GeLU, residual-add, etc.) into GEMM epilogues — the ops execute on data still in registers from the accumulator, avoiding separate kernel launches and global memory round-trips.

## Quick Start

### Using a preset kernel

```bash
# Generate a fused GEMM + RMSNorm + SwiGLU kernel
python3 autotune/generate_kernel.py --preset k2 -o /tmp/kernel.cpp

# Compile
icpx -fsycl -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET \
    -I $SYCL_TLA_DIR/include -I $SYCL_TLA_DIR/tools/util/include \
    -I $SYCL_TLA_DIR/examples/common \
    -I $XE_FUSE_DIR/include \
    -O2 -std=c++17 -fsycl-targets=spir64_gen \
    -o /tmp/kernel /tmp/kernel.cpp

# Run
/tmp/kernel --m=4096 --n=4096 --k=4096 --iterations=200
```

### Available kernel presets

```
python3 autotune/generate_kernel.py --list-presets

GEMM epilogue presets:
  k1               D = acc * R[m]                         (RMSNorm)
  k0a              D = gamma[n] * (acc + residual)         (residual + weight)
  k2               D = SwiGLU(acc * R[m])                  (RMSNorm + SwiGLU)
  k2_geglu         D = GeGLU(acc * R[m])                   (RMSNorm + GeGLU)
  k3               D = RoPE(acc, cos_sin)                  (positional encoding)
  k4               D = RoPE(acc * R[m], cos_sin)           (RMSNorm + RoPE, composed tree)
  k4v2             D = RoPE(acc * R[m], cos_sin)           (RMSNorm + RoPE, merged)

Merged visitors (flat tree):
  k1v2             D = acc * R[m]                          (merged ScaleRows)
  k2v2             D = SwiGLU(acc * R[m])                  (merged scale + SwiGLU)
  k2v2_geglu       D = GeGLU(acc * R[m])                   (merged scale + GeGLU)

INT8 quantization epilogues:
  w8a8_dequant         D_bf16 = int32_acc * scale_token[m] * scale_channel[n]
  w8a8_dequant_biased  D_bf16 = ... + bias[n]

Standalone presets (unfused baselines):
  sa_scale_rows     D[m,n] *= scale[m]
  sa_residual_gamma D = gamma[n] * (D + residual)
  sa_swiglu         D = SwiGLU(D)
  sa_rope_scaled    D = RoPE(D * scale[m], cos_sin)
```

### Available model presets

```
python3 autotune/generate_pipeline.py --list-presets

Available model presets:
  Name             Model                    H   H_kv      I      FFN  RoPE
  llama3_8b        LLaMA 3 8B            4096   1024  14336   swiglu  yes
  llama2_7b        LLaMA 2 7B            4096   4096  11008   swiglu  yes
  llama3_70b       LLaMA 3 70B           8192   1024  28672   swiglu  yes
  gemma2_9b        Gemma 2 9B            3584   2048  14336    geglu  yes
  gemma2_27b       Gemma 2 27B           4608   2048  36864    geglu  yes
  mistral_7b       Mistral 7B            4096   1024  14336   swiglu  yes
  qwen25_7b        Qwen 2.5 7B           3584    512  18944   swiglu  yes
  qwen25_72b       Qwen 2.5 72B          8192   1024  29568   swiglu  yes
  phi3_mini        Phi-3 Mini 3.8B       3072   3072   8192   swiglu  yes
  phi3_medium      Phi-3 Medium 14B      5120   5120  17920   swiglu  yes
```

Each preset defines H (hidden dim), H_kv (KV head dim for GQA), I (intermediate/FFN dim),
activation type, and whether RoPE is used. The pipeline generator maps these to the correct
kernel variants automatically.

### Writing a kernel in C++

```cpp
#include "xe-fuse/builder/epilogue_builder.hpp"
namespace b = xe_fuse::builder;

using TileShape = cute::Shape<cute::_256, cute::_256, cute::_32>;

// Pick your epilogue — just compose the ops you need:
using EVT = b::GeLU<b::ScaleRows<b::Acc, TileShape, float>>;   // GEMM + RMSNorm + GeLU

// Build the full GEMM kernel:
using Kernel = b::MakeGemm<EVT, bf16, bf16, bf16, float, float, TileShape>;
using Gemm = typename Kernel::Gemm;
```

## Builder API Reference

### Data Sources

| Alias | Description |
|-------|-------------|
| `Acc` | GEMM accumulator (frg_acc) |
| `AuxLoad<Element>` | Load M×N auxiliary tensor via Block2D |
| `ColBroadcast<Idx, TS, Element>` | Per-row vector: `scale[m]` broadcast across columns |
| `RowBroadcast<Idx, TS, Element>` | Per-column vector: `scale[n]` broadcast across rows |

### Binary Ops

| Alias | Formula |
|-------|---------|
| `Mul<A, B>` | `A * B` element-wise |
| `Add<A, B>` | `A + B` element-wise |
| `ScaleRows<Input, TS, E>` | `Input * scale[m]` — per-row scaling (RMSNorm) |
| `ScaleCols<Input, TS, E>` | `gamma[n] * Input` — per-column scaling |
| `AddResidual<E>` | `acc + AuxLoad(residual)` |
| `BiasAdd<TS, E>` | `acc + bias[n]` — per-column bias |

### Activation Functions

| Alias | Formula | Used By |
|-------|---------|---------|
| `GeLU<Input>` | `x * 0.5 * (1 + erf(x/√2))` | BERT, GPT-2/3/4, Gemma |
| `GeLUTanh<Input>` | tanh approximation of GeLU | Common fast variant |
| `SiLU<Input>` | `x * sigmoid(x)` (Swish) | LLaMA, Mistral |
| `ReLU<Input>` | `max(0, x)` | Older models |
| `Sigmoid<Input>` | `1 / (1 + exp(-x))` | General |

### Pairwise Ops (lane shuffle)

| Alias | Formula | Notes |
|-------|---------|-------|
| `SwiGLU<Input>` | `silu(gate) * up` on adjacent pairs | LLaMA, Mistral, Qwen |
| `GeGLU<Input>` | `gelu(gate) * up` on adjacent pairs | Gemma 2 |
| `RoPE<ECosSin>` | Rotary position embedding on acc | Direct on accumulator |
| `RoPEComposed<Input, ECosSin>` | RoPE on pre-processed input | Two-child visitor |
| `RoPEScaled<TS, EScale, ECosSin>` | Merged scale + RoPE | Flat tree, fewer dispatches |

### Quantization (INT8/W8A8)

| Alias | Formula | Notes |
|-------|---------|-------|
| `DequantW8A8<TS, EScale>` | `int32_acc * scale_token[m] * scale_channel[n]` | INT8 GEMM → bf16 output |
| `DequantW8A8Biased<TS, EScale, EBias>` | `... + bias[n]` | Same with per-channel bias |

INT8 GEMM uses `int8_t` A/B inputs, `int32_t` accumulator, `bf16` output, and `AlignmentAB=32`
for 256-bit INT8 loads via the `XE_8x16x32_S32S8S8S32_TT` MMA atom.

### Merged Visitors

| Alias | Formula | Notes |
|-------|---------|-------|
| `ScaleRowsMerged<TS, E>` | `acc * R[m]` in one visitor | Flat tree: no AccFetch/MulCompute nodes |
| `SwiGLUScaled<TS, E>` | `SwiGLU(acc * R[m])` in one visitor | Scale + shuffle + silu(gate)*up |
| `GeGLUScaled<TS, E>` | `GeGLU(acc * R[m])` in one visitor | Same with GeLU activation |

These read `frg_acc` directly and do all math in one `visit()` call.

### Patterns

| Alias | Description |
|-------|-------------|
| `DualOutput<Input, Output, AuxE>` | Split-tree: evaluate Input once, store to aux buffer, apply Output for primary D |

### Kernel Builder

```cpp
// MakeGemm<EVT, ElementA, ElementB, ElementD, ElementAcc, ElementCompute, TileShape,
//           LayoutA, LayoutB, AlignmentAB, AlignmentCD>
using K = b::MakeGemm<EVT, bf16, bf16, bf16, float, float, TileShape>;
using Gemm = typename K::Gemm;  // ready to instantiate and run

// INT8 GEMM with custom alignment:
using K = b::MakeGemm<EVT, int8_t, int8_t, bf16, int32_t, float, TileShape,
                       cutlass::layout::RowMajor, cutlass::layout::RowMajor, 32, 8>;
```

## Composition Examples

```cpp
// BERT: GEMM + bias + GeLU
using EVT = b::GeLU<b::BiasAdd<TileShape, float>>;

// LLaMA / Mistral / Qwen: GEMM + RMSNorm + SwiGLU (pairwise)
using EVT = b::SwiGLU<b::ScaleRows<b::Acc, TileShape, float>>;

// Gemma 2: GEMM + RMSNorm + GeGLU (pairwise)
using EVT = b::GeGLU<b::ScaleRows<b::Acc, TileShape, float>>;

// GPT-2: GEMM + bias + GeLU_tanh
using EVT = b::GeLUTanh<b::BiasAdd<TileShape, float>>;

// K0 dual output: store raw sum for rstd + gamma-weighted primary output
using Input = b::AddResidual<bf16>;
using Output = b::ScaleCols<b::Acc, TileShape, float>;
using EVT = b::DualOutput<Input, Output, bf16>;

// Custom: GEMM + residual + RMSNorm + RoPE
using Step1 = b::AddResidual<bf16>;
using Step2 = b::ScaleRows<Step1, TileShape, float>;
using EVT = b::RoPEComposed<Step2, float>;

// W8A8 INT8 dequantization: int8×int8 GEMM → bf16 with per-token/per-channel scales
using EVT = b::DequantW8A8<TileShape, float>;
using K = b::MakeGemm<EVT, int8_t, int8_t, bf16, int32_t, float, TileShape,
                       cutlass::layout::RowMajor, cutlass::layout::RowMajor, 32, 8>;

// Merged visitors (flat tree):
using EVT = b::ScaleRowsMerged<TileShape, float>;  // same as ScaleRows<Acc, ...> but flat
using EVT = b::SwiGLUScaled<TileShape, float>;     // same as SwiGLU<ScaleRows<Acc, ...>> but flat
```

## Standalone Ops

For unfused baselines or non-GEMM use:

```cpp
#include "xe-fuse/standalone/ops.hpp"

auto q = compat::get_default_queue();
xe_fuse::standalone::scale_rows(q, data, scale, M, N, L);
xe_fuse::standalone::gelu(q, data, M, N, L);
xe_fuse::standalone::swiglu(q, data, M, N, L);
xe_fuse::standalone::geglu(q, data, M, N, L);
xe_fuse::standalone::rope_scaled(q, data, tmp, scale, cos_sin, M, N, L);
```

## Code Generation

The `autotune/` directory contains Python tooling for generating kernel and pipeline benchmarks:

| Tool | Description |
|------|-------------|
| `generate_kernel.py` | Single-kernel C++ from preset or JSON spec |
| `generate_pipeline.py` | Full model pipeline C++ from architecture preset |
| `model_presets.py` | Model architecture configs (LLaMA, Gemma, Mistral, Qwen, Phi-3) |
| `pipeline_template.cpp.j2` | Jinja2 template for pipeline benchmarks |
| `kernel_template.cpp.j2` | Jinja2 template for single-kernel benchmarks |
| `tile_selector.py` | Tile shape selector |
| `run_kernel.sh` | Compile + benchmark a generated kernel with structured output |

The pipeline generator maps model architectures to kernel variants:
- SwiGLU models (LLaMA, Mistral, Qwen) use K2 with `b::SwiGLU<>`
- GeGLU models (Gemma 2) use K2 with `b::GeGLU<>`
- GQA models get correct H_kv dimensions for V/K projections
- RoPE is conditionally included based on architecture

## MoE Expert Batched GEMM

For Mixture-of-Experts models (LLaMA 4, Mixtral, DeepSeek-V3, DBRX), xe-fuse provides
batched expert kernels that process all routed experts in a single launch using the
CUTLASS L (batch) dimension:

```cpp
#include "xe-fuse/kernels/gemm_moe_expert.hpp"

// SwiGLU variant — one kernel launch for all experts
using Config = xe_fuse::MoEExpertSwiGLU<bf16, bf16, bf16, float, float, float, TileShape>;
using Gemm = Config::Gemm;

// Launch: {M_per_expert, N=2*I, K=H, L=num_experts}
// Each batch slice gets its own fused RMSNorm + SwiGLU epilogue
```

Instead of 16 separate GEMM + SwiGLU kernel launches (one per expert), a single
batched launch processes all experts. The fused epilogue runs per-batch-slice, so
each expert gets its own scale vector and SwiGLU activation without extra launches.

### MoE architectures supported

| Model | H | Expert I | Experts | top_k | Expert GEMM (N×K) |
|-------|---|----------|---------|-------|--------------------|
| LLaMA 4 Scout | 5120 | 8192 | 16 | 1 | 16384×5120 |
| LLaMA 4 Maverick | 5120 | 8192 | 128 | 1 | 16384×5120 |
| Mixtral 8x7B | 4096 | 14336 | 8 | 2 | 28672×4096 |
| Mixtral 8x22B | 6144 | 16384 | 8 | 2 | 32768×6144 |
| DeepSeek-V3 | 7168 | 2048 | 256 | 8 | 4096×7168 |
| DBRX | 6144 | 10752 | 16 | 4 | 21504×6144 |

## Autotune Tile Selection

xe-fuse includes a tile shape auto-selector based on empirical sweep data (792 configurations
across 7 tiles × 5 kernels × 32 GEMM shapes from 10+ model architectures).

```bash
# Generate pipeline with autotuned tiles for target sequence length
python3 autotune/generate_pipeline.py --preset llama3_8b --autotune --seq-len 128 -o pipeline.cpp

# Or override tile for a single kernel
python3 autotune/generate_kernel.py --preset k2 --tile 64x128x32 -o kernel.cpp
python3 autotune/generate_kernel.py --preset k1 --tile auto --m 128 --n 4096 --k 4096 -o kernel.cpp
```

The tile selector (`autotune/tile_selector.py`) uses empirically-derived heuristics:
tile_M tracks the problem M (64 for M≤64, 128 for M≤128, 256 for M≥256),
tile_N adapts to the output dimension, and K is always 32 (XMX constraint).

## File Layout

```
xe-fuse/
├── include/xe-fuse/
│   ├── builder/epilogue_builder.hpp    — Builder API (start here)
│   ├── visitors/
│   │   ├── xe_elementwise_compute.hpp  — Unary ops: GeLU, SiLU, ReLU, Sigmoid
│   │   ├── xe_pairwise_compute.hpp     — Pairwise: SwiGLU, GeGLU, generic pairs
│   │   ├── xe_rope_compute.hpp         — RoPE: 3 visitor variants
│   │   └── xe_scalerows_compute.hpp    — Merged: ScaleRows, SwiGLUScaled, GeGLUScaled
│   ├── kernels/
│   │   ├── gemm_rmsnorm.hpp            — K1: GEMM + RMSNorm
│   │   ├── gemm_rmsnorm_swiglu.hpp     — K2: GEMM + RMSNorm + SwiGLU
│   │   ├── gemm_rmsnorm_rope.hpp       — K4: GEMM + RMSNorm + RoPE
│   │   ├── gemm_residual_gamma.hpp     — K0a: GEMM + residual + gamma
│   │   ├── gemm_dual_output.hpp        — K0: split-tree dual output
│   │   ├── gemm_moe_expert.hpp         — MoE: batched expert GEMM + SwiGLU/GeGLU
│   │   └── ...
│   └── standalone/ops.hpp              — Element-wise SYCL baselines
├── autotune/
│   ├── generate_kernel.py              — Single-kernel code generator (--tile auto)
│   ├── generate_pipeline.py            — Model pipeline code generator (--autotune)
│   ├── tile_selector.py                — Tile shape heuristics for BMG-G31
│   ├── model_presets.py                — Architecture configs (10+ models)
│   ├── pipeline_template.cpp.j2        — Pipeline Jinja2 template
│   ├── kernel_template.cpp.j2          — Kernel Jinja2 template
│   └── run_kernel.sh                   — Compile + benchmark a generated kernel
├── examples/
│   ├── moe_expert_builder.cpp          — MoE via builder API (L=num_experts)
│   └── moe_expert_fused.cpp            — MoE via kernel template
└── tests/
    ├── run_vllm_comparison.sh          — xe-fuse vs vllm-xpu three-way comparison
    ├── run_bench_e2e.sh                — Full pipeline comparison (requires torch + vllm-xpu)
    └── run_bench_vllm_real.sh          — Real vllm-xpu standalone ops benchmark
```

