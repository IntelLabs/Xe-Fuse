#!/usr/bin/env python3
"""
xe-fuse comparison: Real vllm-xpu-kernels standalone ops benchmark.

Measures the time spent in post-GEMM operations using the real vllm SYCL
kernels (via torch.ops._C). This benchmark does NOT include GEMM time —
it isolates the cost of the standalone ops that xe-fuse eliminates by
fusing into GEMM epilogues.

The companion C++ pipeline benchmark (run_vllm_comparison.sh) measures
end-to-end including GEMM. This script measures just the ops cost.

Usage:
    sbatch run_bench_vllm_real.sh
"""

import argparse
import time

import torch


def load_vllm_ops():
    """Load vllm-xpu-kernels C extension."""
    import vllm_xpu_kernels._C  # noqa: F401

    return torch.ops._C


def bench_rms_norm(ops, M, H, weight, eps, warmup=20, iters=200):
    """Benchmark rms_norm: out = RMSNorm(input, weight)."""
    x = torch.randn(M, H, dtype=torch.bfloat16, device="xpu")
    out = torch.empty_like(x)
    for _ in range(warmup):
        ops.rms_norm(out, x, weight[:H], eps)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        ops.rms_norm(out, x, weight[:H], eps)
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def bench_fused_add_rms_norm(ops, M, H, weight, eps, warmup=20, iters=200):
    """Benchmark fused_add_rms_norm: residual += input; input = RMSNorm(residual) * weight."""
    inp = torch.randn(M, H, dtype=torch.bfloat16, device="xpu")
    res = torch.randn(M, H, dtype=torch.bfloat16, device="xpu")
    for _ in range(warmup):
        ops.fused_add_rms_norm(inp, res, weight[:H], eps)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        ops.fused_add_rms_norm(inp, res, weight[:H], eps)
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def bench_silu_and_mul(ops, M, N, warmup=20, iters=200):
    """Benchmark silu_and_mul: out = silu(input[:,:d]) * input[:,d:]."""
    inp = torch.randn(M, 2 * N, dtype=torch.bfloat16, device="xpu")
    out = torch.empty(M, N, dtype=torch.bfloat16, device="xpu")
    for _ in range(warmup):
        ops.silu_and_mul(out, inp)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        ops.silu_and_mul(out, inp)
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def bench_gelu_and_mul(ops, M, N, warmup=20, iters=200):
    """Benchmark gelu_and_mul: out = gelu(input[:,:d]) * input[:,d:]."""
    inp = torch.randn(M, 2 * N, dtype=torch.bfloat16, device="xpu")
    out = torch.empty(M, N, dtype=torch.bfloat16, device="xpu")
    for _ in range(warmup):
        ops.gelu_and_mul(out, inp)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        ops.gelu_and_mul(out, inp)
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def bench_rotary_embedding(
    ops, M, num_heads, num_kv_heads, head_dim, warmup=20, iters=200
):
    """Benchmark rotary_embedding: in-place NeoX RoPE on Q+K."""
    rot_dim = head_dim

    query = torch.randn(M, num_heads * head_dim, dtype=torch.bfloat16, device="xpu")
    key = torch.randn(M, num_kv_heads * head_dim, dtype=torch.bfloat16, device="xpu")
    positions = torch.arange(M, dtype=torch.long, device="xpu")
    cos_sin_cache = torch.randn(M + 1024, rot_dim, dtype=torch.bfloat16, device="xpu")

    for _ in range(warmup):
        ops.rotary_embedding(positions, query, key, head_dim, cos_sin_cache, True)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        ops.rotary_embedding(positions, query, key, head_dim, cos_sin_cache, True)
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def bench_pipeline_standalone(
    ops,
    M,
    H,
    H_kv,
    I,
    head_dim,
    num_heads,
    num_kv_heads,
    eps,
    act_fn,
    warmup=20,
    iters=200,
):
    """Benchmark a full pipeline of vllm standalone ops (no GEMM).
    Measures the total cost of post-GEMM operations that xe-fuse eliminates."""
    rot_dim = head_dim
    N_ffn = 2 * I
    Q_dim = num_heads * head_dim
    KV_dim = num_kv_heads * head_dim

    weight_h = torch.randn(H, dtype=torch.bfloat16, device="xpu")
    weight_kv = torch.randn(KV_dim, dtype=torch.bfloat16, device="xpu")

    q_out = torch.randn(M, Q_dim, dtype=torch.bfloat16, device="xpu")
    v_out = torch.randn(M, KV_dim, dtype=torch.bfloat16, device="xpu")
    o_proj = torch.randn(M, H, dtype=torch.bfloat16, device="xpu")
    residual = torch.randn(M, H, dtype=torch.bfloat16, device="xpu")
    ffn_in = torch.randn(M, N_ffn, dtype=torch.bfloat16, device="xpu")
    ffn_out = torch.empty(M, I, dtype=torch.bfloat16, device="xpu")

    positions = torch.arange(M, dtype=torch.long, device="xpu")
    cos_sin_cache = torch.randn(M + 1024, rot_dim, dtype=torch.bfloat16, device="xpu")
    q_normed = torch.empty_like(q_out)
    v_normed = torch.empty_like(v_out)

    for _ in range(warmup):
        ops.rms_norm(
            q_normed, q_out, torch.randn(Q_dim, dtype=torch.bfloat16, device="xpu"), eps
        )
        ops.rotary_embedding(positions, q_normed, None, head_dim, cos_sin_cache, True)
        ops.rms_norm(v_normed, v_out, weight_kv, eps)
        ops.fused_add_rms_norm(o_proj, residual, weight_h, eps)
        if act_fn == "swiglu":
            ops.silu_and_mul(ffn_out, ffn_in)
        else:
            ops.gelu_and_mul(ffn_out, ffn_in)

    weight_q = torch.randn(Q_dim, dtype=torch.bfloat16, device="xpu")
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        ops.rms_norm(q_normed, q_out, weight_q, eps)
        ops.rotary_embedding(positions, q_normed, None, head_dim, cos_sin_cache, True)
        ops.rms_norm(v_normed, v_out, weight_kv, eps)
        ops.fused_add_rms_norm(o_proj, residual, weight_h, eps)
        if act_fn == "swiglu":
            ops.silu_and_mul(ffn_out, ffn_in)
        else:
            ops.gelu_and_mul(ffn_out, ffn_in)
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


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


def main():
    parser = argparse.ArgumentParser(description="Real vllm-xpu-kernels benchmark")
    parser.add_argument(
        "--preset", default="llama3_8b", choices=list(MODEL_PRESETS.keys())
    )
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--all", action="store_true", help="Run all presets")
    args = parser.parse_args()

    ops = load_vllm_ops()
    eps = 1e-6

    presets = list(MODEL_PRESETS.keys()) if args.all else [args.preset]

    print("=" * 72)
    print("Real vllm-xpu-kernels Standalone Ops Benchmark")
    print(f"Device: {torch.xpu.get_device_name(0)}")
    print(f"M={args.m}, iterations={args.iters}")
    print("=" * 72)

    for preset_name in presets:
        cfg = MODEL_PRESETS[preset_name]
        H, H_kv, I = cfg["H"], cfg["H_kv"], cfg["I"]
        head_dim = cfg["head_dim"]
        num_heads = cfg["num_heads"]
        num_kv_heads = cfg["num_kv_heads"]
        M = args.m

        print(
            f"\n--- {cfg['name']} (H={H}, H_kv={H_kv}, I={I}, head_dim={head_dim}) ---"
        )

        weight = torch.randn(max(H, H_kv), dtype=torch.bfloat16, device="xpu")

        # Per-op benchmarks
        t_rms = bench_rms_norm(ops, M, H, weight, eps, iters=args.iters)
        t_fused_rms = bench_fused_add_rms_norm(ops, M, H, weight, eps, iters=args.iters)
        if cfg["act"] == "swiglu":
            t_act = bench_silu_and_mul(ops, M, I, iters=args.iters)
        else:
            t_act = bench_gelu_and_mul(ops, M, I, iters=args.iters)
        t_rope = bench_rotary_embedding(
            ops, M, num_heads, num_kv_heads, head_dim, iters=args.iters
        )

        # Pipeline of all ops
        t_pipeline = bench_pipeline_standalone(
            ops,
            M,
            H,
            H_kv,
            I,
            head_dim,
            num_heads,
            num_kv_heads,
            eps,
            cfg["act"],
            iters=args.iters,
        )

        # Memory traffic estimates (bytes)
        bf = 2
        rms_bytes = M * H * bf * 2 + H * bf  # read input + weight, write output
        fused_rms_bytes = M * H * bf * 4 + H * bf  # read inp+res, write both + weight
        act_bytes = M * 2 * I * bf + M * I * bf  # read [M,2I], write [M,I]
        Q_dim = num_heads * head_dim
        rope_bytes = (
            M * Q_dim * bf * 2 + M * head_dim * bf
        )  # read+write Q + read cos_sin
        pipeline_bytes_total = (
            rms_bytes
            + rope_bytes
            + M * H_kv * bf * 2
            + H_kv * bf
            + fused_rms_bytes
            + act_bytes
        )

        print(
            f"  rms_norm:           {t_rms:.4f} ms  ({rms_bytes / (t_rms * 1e-3) / 1e9:.1f} GB/s)"
        )
        print(
            f"  fused_add_rms_norm: {t_fused_rms:.4f} ms  ({fused_rms_bytes / (t_fused_rms * 1e-3) / 1e9:.1f} GB/s)"
        )
        print(
            f"  {cfg['act']}:       {t_act:.4f} ms  ({act_bytes / (t_act * 1e-3) / 1e9:.1f} GB/s)"
        )
        print(
            f"  rotary_embedding:   {t_rope:.4f} ms  ({rope_bytes / (t_rope * 1e-3) / 1e9:.1f} GB/s)"
        )
        print("  ---")
        print(
            f"  Pipeline (all ops): {t_pipeline:.4f} ms  ({pipeline_bytes_total / (t_pipeline * 1e-3) / 1e9:.1f} GB/s)"
        )
        print(f"  Sum of individual:  {t_rms + t_fused_rms + t_act + t_rope:.4f} ms")

        # Structured output
        print("\n=== STRUCTURED OUTPUT ===")
        print(f"VLLM_REAL: {preset_name}")
        print(f"MODEL: {cfg['name']}")
        print(f"DIMS: M={M} H={H} H_kv={H_kv} I={I} head_dim={head_dim}")
        print(f"RMS_NORM: {t_rms:.4f} ms")
        print(f"FUSED_ADD_RMS_NORM: {t_fused_rms:.4f} ms")
        print(f"ACTIVATION: {t_act:.4f} ms {cfg['act']}")
        print(f"ROPE: {t_rope:.4f} ms")
        print(f"OPS_PIPELINE: {t_pipeline:.4f} ms")
        print(
            f"OPS_PIPELINE_BW: {pipeline_bytes_total / (t_pipeline * 1e-3) / 1e9:.1f} GB/s"
        )


if __name__ == "__main__":
    main()
