#!/usr/bin/env python3
"""
Offline weight preprocessing for QuaRot-style W8A8 quantization.

Implements the weight rotation + per-column INT8 quantization described in:
  QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs (Ashkboos et al., 2024)
  https://arxiv.org/abs/2404.00456

For each weight matrix W [K, N]:
  1. Apply random Hadamard rotation block-wise (group_size K dimension, group_size N dimension)
  2. Per-column quantize the rotated matrix to INT8 with symmetric scaling

The rotation redistributes activation outliers uniformly across dimensions,
enabling INT8 quantization with near-BF16 accuracy on large models (70B+).

Usage:
    python3 quantize_weights_quaRot.py --preset llama3_8b --group-size 128
    python3 quantize_weights_quaRot.py --input weights.npy --output weights_q.npz --seed 42

Output (.npz):
    W_i8:           int8  [K, N]   — quantized rotated weight
    scale_channel:  float [N]      — per-column dequant scale (max/127)
    rotation_seed:  uint64 scalar  — random seed for the Hadamard ±1 diagonal
    group_size:     int   scalar   — rotation group size used
"""

import argparse
import sys
import numpy as np
from pathlib import Path

try:
    from model_presets import MODEL_PRESETS
except ImportError:
    MODEL_PRESETS = {}


# ── Walsh-Hadamard Transform (WHT) ───────────────────────────────────────────

def wht(x: np.ndarray) -> np.ndarray:
    """
    Fast Walsh-Hadamard Transform (unnormalized) applied to the last axis of x.
    Last dimension must be a power of 2.
    Uses the butterfly (Cooley-Tukey) algorithm: O(n log n).
    """
    n = x.shape[-1]
    assert n > 0 and (n & (n - 1)) == 0, f"WHT requires power-of-2 size, got {n}"
    h = n
    while h > 1:
        h //= 2
        x = x.copy()
        x[..., :h], x[..., h:2*h] = (x[..., :h] + x[..., h:2*h],
                                       x[..., :h] - x[..., h:2*h])
        # Reshape so the butterfly operates on non-contiguous blocks
        # Full standard WHT butterfly (operates on all pairs):
        x = np.reshape(x, x.shape[:-1] + (n // (h * 2), 2, h))
        lo = x[..., 0, :] + x[..., 1, :]
        hi = x[..., 0, :] - x[..., 1, :]
        x = np.stack([lo, hi], axis=-2).reshape(x.shape[:-3] + (n,))
    return x


def wht_correct(x: np.ndarray) -> np.ndarray:
    """
    Reference WHT using the recursive Hadamard matrix definition.
    Slower but unambiguously correct — used for validation.
    """
    n = x.shape[-1]
    if n == 1:
        return x.copy()
    H = np.array([[1, 1], [1, -1]], dtype=float)
    # Build H_n via Kronecker product
    H_n = np.array([[1.0]])
    size = 1
    while size < n:
        H_n = np.kron(H_n, H)
        size *= 2
    return (x @ H_n.T.astype(x.dtype))


def wht_fast(x: np.ndarray) -> np.ndarray:
    """
    Iterative in-place WHT. x shape: (..., n), n = power of 2.
    Returns a new array with the transform applied along the last axis.
    """
    x = x.copy().astype(np.float32)
    n = x.shape[-1]
    step = 1
    while step < n:
        for i in range(0, n, step * 2):
            lo = x[..., i:i+step].copy()
            hi = x[..., i+step:i+2*step].copy()
            x[..., i:i+step]       = lo + hi
            x[..., i+step:i+2*step] = lo - hi
        step *= 2
    return x


# ── Nearest power of 2 ───────────────────────────────────────────────────────

def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def pad_to_pow2(x: np.ndarray, axis: int) -> tuple:
    """Pad x along axis to the next power of 2. Returns (padded, original_size)."""
    n = x.shape[axis]
    p = next_pow2(n)
    if p == n:
        return x, n
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, p - n)
    return np.pad(x, pad_width), n


# ── Block-wise Hadamard rotation ──────────────────────────────────────────────

def hadamard_rotate_matrix(W: np.ndarray,
                            group_size: int,
                            seed: int) -> np.ndarray:
    """
    Apply random Hadamard rotation to weight matrix W [K, N].

    For each K-group of rows and N-group of columns:
      W_rot[k_blk*g:(k_blk+1)*g, n_blk*g:(n_blk+1)*g]
          = D_k @ H_g @ W_block @ H_g.T @ D_n / g

    where D_k, D_n are random ±1 diagonal matrices derived from `seed`.
    H_g is the unnormalized Hadamard matrix of size g.
    Normalization: / g keeps the Frobenius norm invariant.

    If K or N are not divisible by group_size, the matrix is padded with zeros,
    rotated, and then un-padded.

    Returns W_rot [K, N] as float32.
    """
    K, N = W.shape
    g = group_size

    # Pad to multiple of group_size
    K_pad = ((K + g - 1) // g) * g
    N_pad = ((N + g - 1) // g) * g
    W_padded = np.zeros((K_pad, N_pad), dtype=np.float32)
    W_padded[:K, :N] = W.astype(np.float32)

    rng = np.random.default_rng(seed)
    n_k_blks = K_pad // g
    n_n_blks = N_pad // g

    W_rot = W_padded.copy()

    for k_blk in range(n_k_blks):
        k0, k1 = k_blk * g, (k_blk + 1) * g
        # Random ±1 diagonal for K dimension
        d_k = rng.choice([-1.0, 1.0], size=g).astype(np.float32)
        for n_blk in range(n_n_blks):
            n0, n1 = n_blk * g, (n_blk + 1) * g
            d_n = rng.choice([-1.0, 1.0], size=g).astype(np.float32)
            block = W_padded[k0:k1, n0:n1].copy()
            # Apply: D_k @ H_g @ block @ H_g.T @ D_n / g
            block = (d_k[:, None] * block)          # D_k @ block
            block = wht_fast(block)                  # H_g @ (D_k @ block)
            block = (block * d_n[None, :])           # result @ D_n
            block = wht_fast(block.T).T              # result @ H_g.T (WHT on columns)
            block /= g                               # normalize
            W_rot[k0:k1, n0:n1] = block

    return W_rot[:K, :N]


# ── Per-column quantization ───────────────────────────────────────────────────

def quantize_columns(W_rot: np.ndarray) -> tuple:
    """
    Symmetric per-column INT8 quantization.

    scale_channel[n] = max(|W_rot[:,n]|) / 127
    W_i8[k,n] = round(clamp(W_rot[k,n] / scale_channel[n], -128, 127))

    Returns (W_i8: int8 [K,N], scale_channel: float32 [N]).
    """
    max_abs = np.abs(W_rot).max(axis=0, keepdims=True)  # [1, N]
    scale_channel = (max_abs / 127.0).squeeze(0)         # [N]
    scale_channel = np.where(scale_channel == 0, 1e-8, scale_channel)

    W_scaled = W_rot / scale_channel[None, :]
    W_i8 = np.round(np.clip(W_scaled, -128.0, 127.0)).astype(np.int8)
    return W_i8, scale_channel.astype(np.float32)


# ── Verification helpers ──────────────────────────────────────────────────────

def quantization_error(W_orig: np.ndarray, W_i8: np.ndarray,
                        scale_channel: np.ndarray) -> dict:
    """Compute relative quantization error stats."""
    W_dequant = W_i8.astype(np.float32) * scale_channel[None, :]
    err = np.abs(W_dequant - W_orig.astype(np.float32))
    ref = np.abs(W_orig.astype(np.float32)).mean() + 1e-8
    return {
        "mean_abs_err":     float(err.mean()),
        "max_abs_err":      float(err.max()),
        "relative_err_pct": float(err.mean() / ref * 100),
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_weight_matrix(W: np.ndarray,
                           group_size: int = 128,
                           seed: int = 42,
                           verbose: bool = True) -> dict:
    """
    Rotate and quantize a single weight matrix.

    Returns dict with keys: W_i8, scale_channel, rotation_seed, group_size,
                            quantization_error (stats for logging)
    """
    K, N = W.shape
    if verbose:
        print(f"  Shape: [{K}, {N}], group_size={group_size}, seed={seed}")

    W_rot = hadamard_rotate_matrix(W, group_size=group_size, seed=seed)
    W_i8, scale_channel = quantize_columns(W_rot)
    err = quantization_error(W_rot, W_i8, scale_channel)

    if verbose:
        print(f"  Quant error: mean={err['mean_abs_err']:.4f}, "
              f"max={err['max_abs_err']:.4f}, "
              f"relative={err['relative_err_pct']:.2f}%")

    return {
        "W_i8":          W_i8,
        "scale_channel": scale_channel,
        "rotation_seed": np.uint64(seed),
        "group_size":    np.int32(group_size),
        "quant_error":   err,
    }


def generate_transformer_weights(H: int, H_kv: int, I: int,
                                  group_size: int = 128,
                                  base_seed: int = 42,
                                  dtype: np.dtype = np.float32) -> dict:
    """
    Generate random weight matrices for the 5-GEMM xe-fuse pipeline and
    apply QuaRot quantization to each.

    Returns a dict of {name: {W_i8, scale_channel, ...}} for:
      W_q [H, H], W_k [H, H_kv], W_v [H, H_kv], W_o [H, H], W_ffn [H, 2*I]
    """
    rng = np.random.default_rng(base_seed)

    weight_shapes = {
        "W_q":   (H, H),
        "W_k":   (H, H_kv),
        "W_v":   (H, H_kv),
        "W_o":   (H, H),
        "W_ffn": (H, 2 * I),
    }

    results = {}
    for i, (name, shape) in enumerate(weight_shapes.items()):
        print(f"\n[{name}]")
        W = rng.standard_normal(shape).astype(dtype)
        # Scale to typical LLM weight magnitude
        W *= 0.02
        results[name] = process_weight_matrix(W,
                                               group_size=group_size,
                                               seed=base_seed + i,
                                               verbose=True)
        results[name]["shape"] = shape
    return results


def save_quantized_weights(results: dict, output_path: str) -> None:
    """Save all quantized weight matrices to a .npz archive."""
    arrays = {}
    for name, data in results.items():
        arrays[f"{name}_i8"]          = data["W_i8"]
        arrays[f"{name}_scale"]       = data["scale_channel"]
        arrays[f"{name}_seed"]        = np.array(data["rotation_seed"])
        arrays[f"{name}_group_size"]  = np.array(data["group_size"])
    np.savez_compressed(output_path, **arrays)
    print(f"\nSaved to: {output_path}.npz")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="QuaRot offline weight rotation and INT8 quantization for xe-fuse")
    ap.add_argument("--preset", choices=list(MODEL_PRESETS.keys()) if MODEL_PRESETS else [],
                    help="Model preset (uses model_presets.py dims)")
    ap.add_argument("--input",  type=str,
                    help="Path to .npy weight matrix [K, N] (if not using preset)")
    ap.add_argument("--output", type=str, default="weights_quaRot",
                    help="Output .npz file path (without extension)")
    ap.add_argument("--group-size", type=int, default=128,
                    help="Hadamard rotation group size (must be power of 2, default: 128)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Base random seed for Hadamard diagonal signs")
    ap.add_argument("--verify", action="store_true",
                    help="Run WHT correctness check before processing")
    args = ap.parse_args()

    if args.verify:
        print("Verifying WHT implementation...")
        for n in [4, 8, 16, 32, 128]:
            x = np.random.default_rng(0).standard_normal((4, n)).astype(np.float32)
            ref = wht_correct(x)
            fast = wht_fast(x)
            max_err = np.abs(ref - fast).max()
            status = "OK" if max_err < 1e-4 else "FAIL"
            print(f"  WHT n={n:4d}: max_err={max_err:.2e}  [{status}]")
        print()

    if args.input:
        W = np.load(args.input).astype(np.float32)
        if W.ndim != 2:
            sys.exit(f"Expected 2D weight matrix, got shape {W.shape}")
        print(f"Processing single weight matrix from {args.input}")
        result = process_weight_matrix(W, group_size=args.group_size, seed=args.seed)
        np.savez_compressed(args.output,
                            W_i8=result["W_i8"],
                            scale_channel=result["scale_channel"],
                            rotation_seed=np.array(result["rotation_seed"]),
                            group_size=np.array(result["group_size"]))
        print(f"Saved: {args.output}.npz")

    elif args.preset and MODEL_PRESETS:
        config = MODEL_PRESETS[args.preset]
        H    = config["H"]
        H_kv = config["H_kv"]
        I    = config["I"]
        print(f"Preset: {args.preset}  H={H}, H_kv={H_kv}, I={I}")
        results = generate_transformer_weights(H, H_kv, I,
                                               group_size=args.group_size,
                                               base_seed=args.seed)
        save_quantized_weights(results, args.output)
        print("\nQuantization summary:")
        for name, data in results.items():
            err = data["quant_error"]
            print(f"  {name:8s} {str(data['shape']):18s}  "
                  f"relative_err={err['relative_err_pct']:.2f}%")

    else:
        ap.print_help()
        print("\nExample:")
        print("  python3 quantize_weights_quaRot.py --preset llama3_8b --group-size 128")
        print("  python3 quantize_weights_quaRot.py --input W_q.npy --output W_q_quaRot --verify")


if __name__ == "__main__":
    main()
