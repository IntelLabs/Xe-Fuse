"""
xe-fuse autotune: tile shape selector.

Selects the best tile shape for a GEMM based on (M, N, K, groups) dimensions.
Rules derived from empirical sweeps on Intel Arc Pro B70 (BMG G31) and
CRI SKU4 (Xe3P, 32 XeCores, 2500 MHz):
  - Round 1: 7 tiles × 32 shapes × 6 M values (444 points)
  - Round 3: 25 tiles × 44 shapes including skinny-M DPAS (420 points)
  - Comprehensive sweep: 25 tiles × 79 shapes incl. N=384/3584 (822 points)
  - CRI grouped GEMM: ww27_bd_benchmarks (fp8/mxfp8, G=4/8/16, MoE shapes)
    MFU formula: TFLOPS*1000/(32*8192*2.5*0.4375[*2 for fp8])*100
    CRI peak: ~287 TFLOPS bf16, ~573 TFLOPS fp8
  - G>16 extrapolation: G=24/48/96/192 shapes from bytedance_ops_WW35 (shapes only,
    no perf data). Extrapolated from G>=8 tall-M rule — G>=16 branch covers all G>16.
    One mxfp4 data point (G=32, M=171, N=3584, K=1280) shows 192x512x128 winning
    over baseline at 23.6%; fp8 equivalent unknown. G>=96 small-M shapes flagged.

Usage:
    from tile_selector import select_tile

    tile = select_tile(M=128, N=4096, K=4096)
    # Returns "_128, _128, _32"

    tile = select_tile(M=683, N=3584, K=1280, groups=8)
    # Returns "_352, _256, _64"  (71% MFU on CRI fp8)
"""

import csv
import os
from pathlib import Path

TILES = {
    # --- BMG B70 tiles (empirical sweep on Arc Pro B70 / BMG G31) ---
    "512x128x32": "_512, _128, _32",
    "512x64x32":  "_512, _64, _32",
    "384x128x32": "_384, _128, _32",
    "384x64x32":  "_384, _64, _32",
    "256x512x32": "_256, _512, _32",
    "256x256x32": "_256, _256, _32",
    "256x128x32": "_256, _128, _32",
    "256x64x32":  "_256, _64, _32",
    "256x64x64":  "_256, _64, _64",
    "192x128x32": "_192, _128, _32",
    "128x256x32": "_128, _256, _32",
    "128x128x32": "_128, _128, _32",
    "128x128x64": "_128, _128, _64",
    "128x64x32":  "_128, _64, _32",
    "128x64x64":  "_128, _64, _64",
    "64x256x32":  "_64, _256, _32",
    "64x128x32":  "_64, _128, _32",
    "64x64x32":   "_64, _64, _32",
    "32x256x32":  "_32, _256, _32",
    "32x256x64":  "_32, _256, _64",
    "32x128x32":  "_32, _128, _32",
    "32x128x64":  "_32, _128, _64",
    "16x256x32":  "_16, _256, _32",
    "16x128x32":  "_16, _128, _32",
    "8x256x32":   "_8, _256, _32",
    "8x128x32":   "_8, _128, _32",
    "4x256x32":   "_4, _256, _32",
    "2x256x32":   "_2, _256, _32",
    "1x256x32":   "_1, _256, _32",
    "1x128x32":   "_1, _128, _32",
    # --- CRI SKU4 tiles (best_vs_baseline_ordered.xlsx + ww27_bd_benchmarks) ---
    # Winners: +35–42 pp MFU over 256x256 baseline for MoE shapes
    "352x256x64":  "_352, _256, _64",
    "448x256x64":  "_448, _256, _64",
    "192x640x64":  "_192, _640, _64",
    "448x320x64":  "_448, _320, _64",
    "320x512x128": "_320, _512, _128",
    "192x640x128": "_192, _640, _128",
    "96x896x128":  "_96, _896, _128",
    "352x256x128": "_352, _256, _128",
    "448x256x128": "_448, _256, _128",
    "448x320x128": "_448, _320, _128",
    "128x896x128": "_128, _896, _128",
    "320x512x64":  "_320, _512, _64",
    # ww27_bd_benchmarks new tiles
    "128x896x64":  "_128, _896, _64",
    "64x896x64":   "_64, _896, _64",
    "96x896x64":   "_96, _896, _64",
    "608x128x64":  "_608, _128, _64",
    # G>16 extrapolation (tile_recalc model + G=32 mxfp4 data point)
    "192x512x64":  "_192, _512, _64",
}


def select_tile(M: int, N: int, K: int, kernel: str = "bare", groups: int = 1) -> str:
    """Select optimal tile shape string for CUTLASS cute::Shape<>.

    Args:
        M: number of rows (total across all groups for grouped GEMM)
        N: output columns (hidden dim, FFN dim, etc.)
        K: reduction dim (input hidden dim)
        kernel: "bare", "k1", "k2", or "k4" (affects register pressure)
        groups: number of GEMM groups (>1 enables CRI grouped GEMM rules)

    Returns:
        CUTLASS tile shape string, e.g. "_128, _128, _32"
    """
    tile_k = 32  # default; overridden below for shapes where K=64 validated better

    # ── CRI Grouped GEMM (G > 1) ────────────────────────────────────────────
    # Rules from ww27_bd_benchmarks (fp8/mxfp8, CRI SKU4, 32 XeCores).
    # Key insight: G>=8 → tall-M tiles for wave fill; G=4 → wider-N tiles.
    # Wide-N tiles (64_1792, 64_1280) are WRONG for G>=8 regardless of dtype.
    # G>16 (G=24/48/96/192 from ByteDance WW35): no fp8 perf data — extrapolated
    # from G=16 rule (groups>=16 branch covers all G>16). Exception: G>=32 with
    # M<=200 (e.g. G=96/192 decode) has one mxfp4 data point (G=32, M=171) showing
    # 192x512x128 winning at 23.6% MFU; fp8 tile unknown — needs fulsim validation.
    if groups > 1:
        if N >= 3000 and K <= 1500:
            # N=3584, K=1280 family (short K — GroupScheduler overhead dominant)
            if groups >= 16:
                # tall-M: 69% MFU at M=427 G=16 with 448x256x64
                # 64_1792_64 gives 17-20% MFU (wrong tile class)
                # G=24/48/96/192: extrapolated from G=16, no fp8 data.
                # Small M (≤200): tile_recalc + G=32 mxfp4 data (192x512x128 at 23.6%)
                # prefer 192x512 — tile_m matches group boundary; 448 spans ~2.6 groups
                if M <= 200:
                    tile_m, tile_n, tile_k = 192, 512, 64  # model-extrapolated, needs fulsim
                else:
                    tile_m, tile_n, tile_k = 448, 256, 64
            elif groups >= 8:
                if M <= 800:
                    tile_m, tile_n, tile_k = 352, 256, 64   # 71% at M=683 G=8
                elif M <= 1500:
                    tile_m, tile_n, tile_k = 128, 896, 64   # 64% at M=1366 G=8
                elif M <= 3000:
                    tile_m, tile_n, tile_k = 64, 896, 64    # 37% at M=2731 G=8
                else:
                    tile_m, tile_n, tile_k = 96, 896, 64    # 54% at M=5462 G=8
            else:
                # G=4: per-group M is large enough for wider N tiles
                tile_m, tile_n, tile_k = 320, 512, 64       # 38% at M=2560 G=4
        elif N >= 2000 and K >= 3000:
            # N=2560, K=3584 family (longer K — better arithmetic intensity)
            if groups >= 16:
                # G=16 wrong tile: 64_1280_64 gives 35%; use tall-M
                # G=24/48/96/192: extrapolated from G=16, no fp8 data.
                # Small M (≤200): tile_recalc + ww27 G=8 winner (192x640x64 at 53%)
                if M <= 200:
                    tile_m, tile_n, tile_k = 192, 640, 64  # model-extrapolated, needs fulsim
                else:
                    tile_m, tile_n, tile_k = 448, 256, 64
            elif groups >= 8:
                if M <= 200:
                    tile_m, tile_n, tile_k = 192, 640, 64   # 53% at M=171 G=8
                elif M <= 800:
                    tile_m, tile_n, tile_k = 352, 256, 64   # 83% at M=683 G=8
                else:
                    tile_m, tile_n, tile_k = 608, 128, 64   # 88% at M=5462 G=8
            else:
                # G=4
                if M <= 500:
                    tile_m, tile_n, tile_k = 448, 320, 64   # 77% at M=427 G=4
                elif M <= 1500:
                    tile_m, tile_n, tile_k = 352, 256, 64   # 84% at M=1366 G=4
                else:
                    tile_m, tile_n, tile_k = 320, 512, 64   # 61% at M=2560 G=4
        else:
            # Unknown grouped shape — fall through to bare GEMM heuristic
            tile_m, tile_n = 256, 256

        key = f"{tile_m}x{tile_n}x{tile_k}"
        return TILES.get(key, "_256, _256, _32")

    # ── Skinny-M DPAS tiles for GEMV (M ≤ 64) ──
    # Rescue3 sweep: skinny tiles give 2-7x over 64x256 at small M.
    # Best tile depends on (M, N, K) combination.
    if M <= 1:
        if N >= 8192:
            tile_m, tile_n = 4, 256    # M=1 N=8192 K=1024: 3.8x speedup
        elif N >= 4096:
            tile_m, tile_n = 16, 128   # M=1 N=4096: 1.8x
        else:
            tile_m, tile_n = 8, 128    # M=1 N=1024: 6x
    elif M <= 2:
        if K <= 1024 and N >= 8192:
            tile_m, tile_n = 8, 256
        else:
            tile_m, tile_n = 4, 256    # M=2 N=1024 K=8192: 5.7x
    elif M <= 4:
        if N >= 8192 and K <= 1024:
            tile_m, tile_n = 8, 128    # M=4 N=8192 K=1024: 3.6x
        else:
            tile_m, tile_n = 32, 256
    elif M <= 8:
        if N <= 1024:
            tile_m, tile_n = 2, 256    # M=8 N=1024 K=8192: 6.6x
        elif N >= 8192 and K <= 1024:
            tile_m, tile_n = 8, 256    # M=8 N=8192 K=1024: 2.1x
        else:
            tile_m, tile_n = 16, 128
    elif M <= 16:
        if N <= 1024:
            tile_m, tile_n = 8, 128    # M=16 N=1024: 4x
        elif N >= 8192 and K <= 1024:
            tile_m, tile_n = 16, 256   # M=16 N=8192 K=1024: 2.6x
        else:
            tile_m, tile_n = 16, 128
    elif M <= 32:
        if N <= 1024:
            tile_m, tile_n = 8, 128    # M=32 N=1024: 3.3x
        elif N >= 8192 and K <= 1024:
            tile_m, tile_n = 32, 256   # M=32 N=8192 K=1024: 2.2x
        else:
            tile_m, tile_n = 16, 256
    elif M <= 64:
        if N <= 1024 and K >= 4096:
            tile_m, tile_n, tile_k = 32, 128, 64  # val 307282: 7.3% MFU (K=64 2x over K=32)
        elif N <= 1024:
            tile_m, tile_n = 8, 128    # M=64 N=1024: 2.9x
        elif N >= 4096:
            tile_m, tile_n = 64, 256
        else:
            tile_m, tile_n = 32, 256
    # ── Standard tiles for M ≥ 128 ──
    elif M <= 128:
        if N <= 384 and K > 1024:
            tile_m, tile_n = 128, 128  # N=384 K=3584: 128x128 best
        elif N <= 384:
            tile_m, tile_n = 128, 64
        elif N <= 1024:
            tile_m, tile_n = 128, 64   # sweep 307257: 9.3% MFU (was 5.9% with 128x128)
        elif N <= 4096:
            tile_m, tile_n = 64, 64    # sweep 307257: 28.7% MFU (was 24.3% with 64x256)
        elif K <= 1024:
            tile_m, tile_n = 32, 128   # deep sweep 307267: 24.9% MFU (was 22.5% with 64x128)
        else:
            tile_m, tile_n = 128, 256
    elif M <= 256:
        if N <= 384:
            tile_m, tile_n = 256, 64
        elif N <= 1024:
            tile_m, tile_n, tile_k = 256, 64, 64  # val 307282: 25.1% MFU (K=64 vs 18.1% K=32)
        elif N <= 4096:
            tile_m, tile_n = 256, 128  # M=256 N=4096: 1.8x over 256x256
        else:
            tile_m, tile_n = 256, 128
    elif M <= 384:
        if N <= 384 and K > 1024:
            tile_m, tile_n = 128, 128  # N=384 K=3584: 128x128 3x over 256x64
        elif N <= 384:
            tile_m, tile_n = 256, 64
        elif N <= 1024:
            tile_m, tile_n = 384, 64   # val 307282: 24.0% MFU (was 19.8% with 64x128)
        elif N <= 4096 and K <= 384:
            tile_m, tile_n = 192, 128  # N=3584 K=384: 192x128 2x over 384x128
        else:
            tile_m, tile_n = 384, 128  # M=384: 1.5-1.6x over 256x256
    elif M <= 512:
        if N <= 384 and K > 1024:
            tile_m, tile_n = 128, 128  # N=384 K=3584: 128x128 1.4x over 256x64
        elif N <= 384:
            tile_m, tile_n = 256, 64
        elif N <= 1024:
            tile_m, tile_n, tile_k = 32, 128, 64  # val 307282: 50.1% MFU (K=64 vs 31.6% K=32)
        elif N <= 4096 and K <= 384:
            tile_m, tile_n = 128, 128  # N=3584 K=384: 128x128 1.5x
        elif N <= 4096:
            tile_m, tile_n = 256, 128  # M=512 N=4096: 256x128 ≈ best
        else:
            tile_m, tile_n = 256, 256  # M=512 N=8192: 256x256 best
    elif M <= 640:
        if N <= 384 and K > 1024:
            tile_m, tile_n = 128, 128
        elif N <= 384:
            tile_m, tile_n = 384, 64
        elif N <= 1024:
            tile_m, tile_n = 384, 64   # val 307282: 27.9% MFU (43.9% was run-to-run variance)
        elif N <= 4096 and K <= 384:
            tile_m, tile_n = 128, 128
        elif N >= 8192 and K <= 1024:
            tile_m, tile_n = 128, 256
        elif N >= 4096 and K >= 4096:
            tile_m, tile_n = 256, 512  # CRI SKU4 archstudy: +19% over 256x256 for N=K=4096
        else:
            tile_m, tile_n = 256, 256 if N >= 4096 else 128
    elif M <= 768:
        if N <= 384 and K > 1024:
            tile_m, tile_n = 128, 128  # N=384 K=3584: 128x128 dominant
        elif N <= 384:
            tile_m, tile_n = 384, 64
        elif N <= 1024:
            tile_m, tile_n = 32, 256   # val 307282: 51.9% MFU! (768/32=24 M-tiles, 96% wave fill)
        elif N <= 4096 and K <= 384:
            tile_m, tile_n = 128, 128  # N=3584 K=384: 128x128
        elif N >= 8192 and K <= 1024:
            tile_m, tile_n = 128, 256
        else:
            tile_m, tile_n = 256, 256 if N >= 4096 else 128
    elif M <= 896:
        if N <= 384 and K > 1024:
            tile_m, tile_n = 128, 128
        elif N <= 384:
            tile_m, tile_n = 384, 64
        elif N <= 1024:
            tile_m, tile_n, tile_k = 32, 256, 64  # val 307282: 35.7% MFU (K=64 vs 30.2% K=32)
        elif N <= 4096 and K <= 384:
            tile_m, tile_n = 128, 128
        elif N >= 8192 and K <= 1024:
            tile_m, tile_n = 128, 256
        else:
            tile_m, tile_n = 256, 256 if N >= 4096 else 128
    else:
        # M >= 1024
        if N <= 384 and K > 1024:
            if M <= 1024:
                tile_m, tile_n = 256, 64   # M=1024 N=384 K=3584: 256x64
            elif M <= 2048:
                tile_m, tile_n = 256, 128  # M=2048: 256x128 best
            elif M <= 3328:
                tile_m, tile_n = 384, 128  # M=3328: 384x128 best
            elif M <= 4096:
                tile_m, tile_n = 256, 64   # M=4096: 256x64
            elif M <= 7680:
                tile_m, tile_n = 128, 64   # val 307282: 47.7% MFU (was 42.3% with 384x64)
            else:
                tile_m, tile_n, tile_k = 128, 128, 64  # val 307282: 53.4% MFU (K=64 vs 40.7% K=32)
        elif N <= 384:
            tile_m, tile_n = 512, 64
        elif N <= 1024:
            if M <= 1024:
                tile_m, tile_n, tile_k = 32, 128, 64  # val 307282: 56.6% MFU (K=64+S3, was 56.5% K=32)
            else:
                tile_m, tile_n = 256, 256  # large M: 256x256 still good (65.4% at M=8192)
        elif N <= 4096 and K <= 384:
            if M <= 1024:
                tile_m, tile_n = 256, 64   # N=3584 K=384 M=1024: 256x64
            elif M <= 2048:
                tile_m, tile_n = 128, 256  # M=2048: 128x256
            elif M <= 3328:
                tile_m, tile_n = 256, 64   # M=3328: 256x64
            elif M <= 4096:
                tile_m, tile_n = 128, 256  # M=4096: 128x256
            else:
                tile_m, tile_n = 256, 128  # M=7680: 256x128
        elif N >= 11264:
            tile_m, tile_n = 512, 128  # N=11264: 1.5x over 256x128
        elif N >= 8192 and K <= 1024:
            if M <= 1024:
                tile_m, tile_n = 256, 128  # M=1024 N=8192 K=1024: 256x128
            else:
                tile_m, tile_n = 192, 128  # large M: 192x128
        elif N >= 4096 and K >= 4096:
            tile_m, tile_n = 256, 512  # CRI SKU4 archstudy: compute_bound at 92.6% EU active for M=1024
        else:
            tile_m, tile_n = 256, 256 if M >= 512 else 128

    # K2 (SwiGLU/GeGLU) has more register pressure — prefer smaller tile_n
    if kernel in ("k2", "k2_geglu") and tile_m * tile_n > 256 * 128:
        if tile_n > 128 and tile_m >= 256:
            tile_n = 128

    # K4 (RMSNorm+RoPE) has the highest register pressure.
    # 64x256 does NOT compile for K4.
    if kernel in ("k4", "k4v2"):
        if tile_m <= 64:
            tile_n = min(tile_n, 128)
        elif tile_m <= 128:
            tile_n = min(tile_n, 256)

    key = f"{tile_m}x{tile_n}x{tile_k}"
    return TILES.get(key, "_256, _256, _32")


def select_tile_from_csv(M: int, N: int, K: int, kernel: str = "Bare_GEMM(bf16)", groups: int = 1) -> str:
    """Look up the best tile from the sweep CSV if available, else fall back to heuristic."""
    csv_path = Path(__file__).parent.parent / "tests" / "best_tiles.csv"
    if not csv_path.exists():
        return select_tile(M, N, K, kernel, groups)

    best_tf = 0.0
    best_tile = None

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row["kernel"] == kernel and
                int(row["M"]) == M and
                int(row["N"]) == N and
                int(row["K"]) == K):
                tf = float(row["tflops"])
                if tf > best_tf:
                    best_tf = tf
                    best_tile = row["best_tile"]

    if best_tile and best_tile in TILES:
        return TILES[best_tile]

    return select_tile(M, N, K, kernel, groups)


def tile_shape_str(M: int, N: int, K: int, kernel: str = "bare", groups: int = 1) -> str:
    """Convenience: returns the tile shape for generate_kernel/pipeline use."""
    return select_tile(M, N, K, kernel, groups)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="xe-fuse tile selector")
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--kernel", default="bare")
    parser.add_argument("--groups", type=int, default=1)
    args = parser.parse_args()

    tile = select_tile(args.m, args.n, args.k, args.kernel, args.groups)
    print(f"M={args.m} N={args.n} K={args.k} groups={args.groups} kernel={args.kernel} -> tile={tile}")
