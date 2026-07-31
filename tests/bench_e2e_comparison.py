#!/usr/bin/env python3
"""
xe-fuse vs real vllm: End-to-end pipeline comparison.

Measures the FULL pipeline cost including GEMMs:
  vllm path:    torch.mm (oneDNN) + real vllm standalone ops (torch.ops._C)
  xe-fuse path: CUTLASS GEMM + epilogue fusion (from C++ benchmark)

This is the true apples-to-apples comparison. The vllm path uses the same
GEMM sequence as xe-fuse's C++ pipeline:
  1. Q projection: x @ W_q -> RMSNorm -> RoPE
  2. V projection: x @ W_v -> RMSNorm
  3. O projection: attn_out @ W_o -> residual add + RMSNorm
  4. FFN:          residual @ W_ffn -> RMSNorm -> SwiGLU/GeGLU

Usage:
    sbatch run_bench_e2e.sh
"""

import argparse
import time

import torch


def load_vllm_ops():
    import vllm_xpu_kernels._C  # noqa: F401

    return torch.ops._C


MODEL_PRESETS = {
    "llama3_8b": {
        "name": "LLaMA 3 8B",
        "H": 4096,
        "H_kv": 1024,
        "I": 14336,
        "head_dim": 128,
        "num_heads": 32,
        "num_kv_heads": 8,
        "act": "swiglu",
    },
    "gemma2_9b": {
        "name": "Gemma 2 9B",
        "H": 3584,
        "H_kv": 2048,
        "I": 14336,
        "head_dim": 256,
        "num_heads": 16,
        "num_kv_heads": 8,
        "act": "geglu",
    },
    "qwen25_7b": {
        "name": "Qwen 2.5 7B",
        "H": 3584,
        "H_kv": 512,
        "I": 18944,
        "head_dim": 128,
        "num_heads": 28,
        "num_kv_heads": 4,
        "act": "swiglu",
    },
    "phi3_mini": {
        "name": "Phi-3 Mini 3.8B",
        "H": 3072,
        "H_kv": 3072,
        "I": 8192,
        "head_dim": 128,
        "num_heads": 24,
        "num_kv_heads": 24,
        "act": "swiglu",
    },
}


def bench_vllm_e2e(ops, cfg, M, eps=1e-6, warmup=20, iters=200):
    """Full vllm-style pipeline: oneDNN GEMMs + real vllm standalone ops."""
    H = cfg["H"]
    H_kv = cfg["H_kv"]
    I = cfg["I"]
    N_ffn = 2 * I
    head_dim = cfg["head_dim"]
    num_heads = cfg["num_heads"]
    num_kv_heads = cfg["num_kv_heads"]
    Q_dim = num_heads * head_dim
    KV_dim = num_kv_heads * head_dim
    act_fn = cfg["act"]

    # Weight matrices (transposed for torch.mm: x @ W = [M,H] @ [H,N])
    W_q = torch.randn(H, H, dtype=torch.bfloat16, device="xpu")
    W_v = torch.randn(H, H_kv, dtype=torch.bfloat16, device="xpu")
    W_o = torch.randn(H, H, dtype=torch.bfloat16, device="xpu")
    W_ffn = torch.randn(H, N_ffn, dtype=torch.bfloat16, device="xpu")

    # RMSNorm weights
    w_rms_q = torch.randn(H, dtype=torch.bfloat16, device="xpu")
    w_rms_v = torch.randn(H_kv, dtype=torch.bfloat16, device="xpu")
    w_rms_o = torch.randn(H, dtype=torch.bfloat16, device="xpu")
    w_rms_ffn = torch.randn(H, dtype=torch.bfloat16, device="xpu")

    # Activation buffers
    x = torch.randn(M, H, dtype=torch.bfloat16, device="xpu")
    attn_out = torch.randn(M, H, dtype=torch.bfloat16, device="xpu")
    residual = torch.randn(M, H, dtype=torch.bfloat16, device="xpu")

    # RoPE buffers
    positions = torch.arange(M, dtype=torch.long, device="xpu")
    cos_sin_cache = torch.randn(M + 1024, head_dim, dtype=torch.bfloat16, device="xpu")

    # Output buffers
    q_proj = torch.empty(M, H, dtype=torch.bfloat16, device="xpu")
    q_normed = torch.empty(M, H, dtype=torch.bfloat16, device="xpu")
    v_proj = torch.empty(M, H_kv, dtype=torch.bfloat16, device="xpu")
    v_normed = torch.empty(M, H_kv, dtype=torch.bfloat16, device="xpu")
    o_proj = torch.empty(M, H, dtype=torch.bfloat16, device="xpu")
    ffn_proj = torch.empty(M, N_ffn, dtype=torch.bfloat16, device="xpu")
    ffn_normed = torch.empty(M, N_ffn, dtype=torch.bfloat16, device="xpu")
    ffn_out = torch.empty(M, I, dtype=torch.bfloat16, device="xpu")

    def run_pipeline():
        # 1. Q projection: x @ W_q -> RMSNorm -> RoPE
        torch.mm(x, W_q, out=q_proj)
        ops.rms_norm(q_normed, q_proj, w_rms_q, eps)
        ops.rotary_embedding(positions, q_normed, None, head_dim, cos_sin_cache, True)

        # 2. V projection: x @ W_v -> RMSNorm
        torch.mm(x, W_v, out=v_proj)
        ops.rms_norm(v_normed, v_proj, w_rms_v, eps)

        # 3. O projection: attn_out @ W_o -> residual add + RMSNorm
        torch.mm(attn_out, W_o, out=o_proj)
        ops.fused_add_rms_norm(o_proj, residual, w_rms_o, eps)

        # 4. FFN: residual @ W_ffn -> RMSNorm -> SwiGLU/GeGLU
        torch.mm(residual, W_ffn, out=ffn_proj)
        if act_fn == "swiglu":
            ops.silu_and_mul(ffn_out, ffn_proj)
        else:
            ops.gelu_and_mul(ffn_out, ffn_proj)

    for _ in range(warmup):
        run_pipeline()

    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run_pipeline()
    torch.xpu.synchronize()
    total_ms = (time.perf_counter() - t0) / iters * 1e3
    return total_ms


def bench_onednn_gemm_only(cfg, M, warmup=20, iters=200):
    """Benchmark just the 4 GEMMs via torch.mm (oneDNN) without any post-ops."""
    H = cfg["H"]
    H_kv = cfg["H_kv"]
    N_ffn = 2 * cfg["I"]

    W_q = torch.randn(H, H, dtype=torch.bfloat16, device="xpu")
    W_v = torch.randn(H, H_kv, dtype=torch.bfloat16, device="xpu")
    W_o = torch.randn(H, H, dtype=torch.bfloat16, device="xpu")
    W_ffn = torch.randn(H, N_ffn, dtype=torch.bfloat16, device="xpu")

    x = torch.randn(M, H, dtype=torch.bfloat16, device="xpu")
    attn_out = torch.randn(M, H, dtype=torch.bfloat16, device="xpu")
    residual = torch.randn(M, H, dtype=torch.bfloat16, device="xpu")

    q_proj = torch.empty(M, H, dtype=torch.bfloat16, device="xpu")
    v_proj = torch.empty(M, H_kv, dtype=torch.bfloat16, device="xpu")
    o_proj = torch.empty(M, H, dtype=torch.bfloat16, device="xpu")
    ffn_proj = torch.empty(M, N_ffn, dtype=torch.bfloat16, device="xpu")

    def run_gemms():
        torch.mm(x, W_q, out=q_proj)
        torch.mm(x, W_v, out=v_proj)
        torch.mm(attn_out, W_o, out=o_proj)
        torch.mm(residual, W_ffn, out=ffn_proj)

    for _ in range(warmup):
        run_gemms()

    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run_gemms()
    torch.xpu.synchronize()
    total_ms = (time.perf_counter() - t0) / iters * 1e3
    return total_ms


def pipeline_flops(M, H, H_kv, N_ffn):
    """Total FLOPs for 4 GEMMs in the pipeline."""
    return 2.0 * M * (H * H + H_kv * H + H * H + N_ffn * H)


def main():
    parser = argparse.ArgumentParser(description="xe-fuse vs real vllm: E2E comparison")
    parser.add_argument(
        "--preset", default="llama3_8b", choices=list(MODEL_PRESETS.keys())
    )
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--all", action="store_true", help="Run all presets")
    args = parser.parse_args()

    ops = load_vllm_ops()
    presets = list(MODEL_PRESETS.keys()) if args.all else [args.preset]

    print("=" * 72)
    print("xe-fuse vs Real vllm: End-to-End Pipeline Comparison")
    print(f"Device: {torch.xpu.get_device_name(0)}")
    print(f"M={args.m}, iterations={args.iters}")
    print("vllm path: torch.mm (oneDNN) + real vllm standalone ops")
    print("=" * 72)

    for preset_name in presets:
        cfg = MODEL_PRESETS[preset_name]
        H, H_kv, I = cfg["H"], cfg["H_kv"], cfg["I"]
        N_flops = pipeline_flops(args.m, H, H_kv, 2 * I)
        M = args.m

        print(f"\n{'=' * 60}")
        print(f"  {cfg['name']} (H={H}, H_kv={H_kv}, I={I})")
        print(
            f"  Pipeline: Q(RMSNorm+RoPE) + V(RMSNorm) + O(ResAdd+RMSNorm) + FFN({cfg['act']})"
        )
        print(f"{'=' * 60}")

        # Benchmark oneDNN GEMMs only (no ops)
        t_gemm_only = bench_onednn_gemm_only(cfg, M, iters=args.iters)
        gemm_tflops = N_flops / (t_gemm_only * 1e-3) / 1e12

        # Benchmark full vllm pipeline (GEMMs + ops)
        t_vllm_e2e = bench_vllm_e2e(ops, cfg, M, iters=args.iters)
        vllm_tflops = N_flops / (t_vllm_e2e * 1e-3) / 1e12

        # Ops overhead
        t_ops_overhead = t_vllm_e2e - t_gemm_only
        ops_pct = t_ops_overhead / t_vllm_e2e * 100

        print("\n  oneDNN GEMMs only (4x torch.mm):")
        print(f"    Time: {t_gemm_only:.4f} ms")
        print(f"    Throughput: {gemm_tflops:.1f} TFlop/s")
        print("\n  vllm full pipeline (oneDNN GEMM + real vllm ops):")
        print(f"    Time: {t_vllm_e2e:.4f} ms")
        print(f"    Throughput: {vllm_tflops:.1f} TFlop/s")
        print(f"    Ops overhead: {t_ops_overhead:.4f} ms ({ops_pct:.1f}% of pipeline)")

        # Structured output for parsing
        print("\n=== STRUCTURED OUTPUT ===")
        print(f"E2E: {preset_name}")
        print(f"MODEL: {cfg['name']}")
        print(f"DIMS: M={M} H={H} H_kv={H_kv} I={I}")
        print(f"TOTAL_FLOPS: {N_flops:.0f}")
        print(f"ONEDNN_GEMM_ONLY: {t_gemm_only:.4f} ms {gemm_tflops:.1f} TFlop/s")
        print(f"VLLM_E2E: {t_vllm_e2e:.4f} ms {vllm_tflops:.1f} TFlop/s")
        print(f"OPS_OVERHEAD: {t_ops_overhead:.4f} ms {ops_pct:.1f}%")


if __name__ == "__main__":
    main()
