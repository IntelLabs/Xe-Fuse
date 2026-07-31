"""
xe-fuse autotune: tile shape selector.

Selects the best tile shape for a GEMM based on (M, N, K) dimensions.
Rules derived from sweeps on Intel Arc Pro B70 (BMG G31).

Usage:
    from tile_selector import select_tile

    tile = select_tile(M=128, N=4096, K=4096)
    # Returns "_128, _128, _32"
"""

TILES = {
    "512x128x32": "_512, _128, _32",
    "512x64x32": "_512, _64, _32",
    "384x128x32": "_384, _128, _32",
    "384x64x32": "_384, _64, _32",
    "256x512x32": "_256, _512, _32",
    "256x256x32": "_256, _256, _32",
    "256x128x32": "_256, _128, _32",
    "256x64x32": "_256, _64, _32",
    "256x64x64": "_256, _64, _64",
    "192x128x32": "_192, _128, _32",
    "128x256x32": "_128, _256, _32",
    "128x128x32": "_128, _128, _32",
    "128x128x64": "_128, _128, _64",
    "128x64x32": "_128, _64, _32",
    "128x64x64": "_128, _64, _64",
    "64x256x32": "_64, _256, _32",
    "64x128x32": "_64, _128, _32",
    "64x64x32": "_64, _64, _32",
    "32x256x32": "_32, _256, _32",
    "32x256x64": "_32, _256, _64",
    "32x128x32": "_32, _128, _32",
    "32x128x64": "_32, _128, _64",
    "16x256x32": "_16, _256, _32",
    "16x128x32": "_16, _128, _32",
    "8x256x32": "_8, _256, _32",
    "8x128x32": "_8, _128, _32",
    "4x256x32": "_4, _256, _32",
    "2x256x32": "_2, _256, _32",
    "1x256x32": "_1, _256, _32",
    "1x128x32": "_1, _128, _32",
}


def select_tile(M: int, N: int, K: int, kernel: str = "bare", groups: int = 1) -> str:
    """Select optimal tile shape string for CUTLASS cute::Shape<>.

    Args:
        M: number of rows
        N: output columns (hidden dim, FFN dim, etc.)
        K: reduction dim (input hidden dim)
        kernel: "bare", "k1", "k2", or "k4" (affects register pressure)
        groups: unused, kept for API compatibility

    Returns:
        CUTLASS tile shape string, e.g. "_128, _128, _32"
    """
    tile_k = 32  # default; overridden below for shapes where K=64 validated better

    # ── Skinny-M DPAS tiles for GEMV (M ≤ 64) ──
    # Skinny tiles give 2-7x over 64x256 at small M.
    if M <= 1:
        if N >= 8192:
            tile_m, tile_n = 4, 256
        elif N >= 4096:
            tile_m, tile_n = 16, 128
        else:
            tile_m, tile_n = 8, 128
    elif M <= 2:
        if K <= 1024 and N >= 8192:
            tile_m, tile_n = 8, 256
        else:
            tile_m, tile_n = 4, 256
    elif M <= 4:
        if N >= 8192 and K <= 1024:
            tile_m, tile_n = 8, 128
        else:
            tile_m, tile_n = 32, 256
    elif M <= 8:
        if N <= 1024:
            tile_m, tile_n = 2, 256
        elif N >= 8192 and K <= 1024:
            tile_m, tile_n = 8, 256
        else:
            tile_m, tile_n = 16, 128
    elif M <= 16:
        if N <= 1024:
            tile_m, tile_n = 8, 128
        elif N >= 8192 and K <= 1024:
            tile_m, tile_n = 16, 256
        else:
            tile_m, tile_n = 16, 128
    elif M <= 32:
        if N <= 1024:
            tile_m, tile_n = 8, 128
        elif N >= 8192 and K <= 1024:
            tile_m, tile_n = 32, 256
        else:
            tile_m, tile_n = 16, 256
    elif M <= 64:
        if N <= 1024 and K >= 4096:
            tile_m, tile_n, tile_k = 32, 128, 64
        elif N <= 1024:
            tile_m, tile_n = 8, 128
        elif N >= 4096:
            tile_m, tile_n = 64, 256
        else:
            tile_m, tile_n = 32, 256
    # ── Standard tiles for M ≥ 128 ──
    elif M <= 128:
        if N <= 384 and K > 1024:
            tile_m, tile_n = 128, 128
        elif N <= 1024:
            tile_m, tile_n = 128, 64
        elif N <= 4096:
            tile_m, tile_n = 64, 64
        elif K <= 1024:
            tile_m, tile_n = 32, 128
        else:
            tile_m, tile_n = 128, 256
    elif M <= 256:
        if N <= 384:
            tile_m, tile_n = 256, 64
        elif N <= 1024:
            tile_m, tile_n, tile_k = 256, 64, 64
        elif N <= 4096:
            tile_m, tile_n = 256, 128
        else:
            tile_m, tile_n = 256, 128
    elif M <= 384:
        if N <= 384 and K > 1024:
            tile_m, tile_n = 128, 128
        elif N <= 384:
            tile_m, tile_n = 256, 64
        elif N <= 1024:
            tile_m, tile_n = 384, 64
        elif N <= 4096 and K <= 384:
            tile_m, tile_n = 192, 128
        else:
            tile_m, tile_n = 384, 128
    elif M <= 512:
        if N <= 384 and K > 1024:
            tile_m, tile_n = 128, 128
        elif N <= 384:
            tile_m, tile_n = 256, 64
        elif N <= 1024:
            tile_m, tile_n, tile_k = 32, 128, 64
        elif N <= 4096 and K <= 384:
            tile_m, tile_n = 128, 128
        elif N <= 4096:
            tile_m, tile_n = 256, 128
        else:
            tile_m, tile_n = 256, 256
    elif M <= 640:
        if N <= 384 and K > 1024:
            tile_m, tile_n = 128, 128
        elif N <= 1024:
            tile_m, tile_n = 384, 64
        elif N <= 4096 and K <= 384:
            tile_m, tile_n = 128, 128
        elif N >= 8192 and K <= 1024:
            tile_m, tile_n = 128, 256
        elif N >= 4096 and K >= 4096:
            tile_m, tile_n = 256, 512
        else:
            tile_m, tile_n = 256, 256 if N >= 4096 else 128
    elif M <= 768:
        if N <= 384 and K > 1024:
            tile_m, tile_n = 128, 128
        elif N <= 384:
            tile_m, tile_n = 384, 64
        elif N <= 1024:
            tile_m, tile_n = 32, 256
        elif N <= 4096 and K <= 384:
            tile_m, tile_n = 128, 128
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
            tile_m, tile_n, tile_k = 32, 256, 64
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
                tile_m, tile_n = 256, 64
            elif M <= 2048:
                tile_m, tile_n = 256, 128
            elif M <= 3328:
                tile_m, tile_n = 384, 128
            elif M <= 4096:
                tile_m, tile_n = 256, 64
            elif M <= 7680:
                tile_m, tile_n = 128, 64
            else:
                tile_m, tile_n, tile_k = 128, 128, 64
        elif N <= 384:
            tile_m, tile_n = 512, 64
        elif N <= 1024:
            if M <= 1024:
                tile_m, tile_n, tile_k = 32, 128, 64
            else:
                tile_m, tile_n = 256, 256
        elif N <= 4096 and K <= 384:
            if M <= 1024:
                tile_m, tile_n = 256, 64
            elif M <= 2048:
                tile_m, tile_n = 128, 256
            elif M <= 3328:
                tile_m, tile_n = 256, 64
            elif M <= 4096:
                tile_m, tile_n = 128, 256
            else:
                tile_m, tile_n = 256, 128
        elif N >= 11264:
            tile_m, tile_n = 512, 128
        elif N >= 8192 and K <= 1024:
            if M <= 1024:
                tile_m, tile_n = 256, 128
            else:
                tile_m, tile_n = 192, 128
        elif N >= 4096 and K >= 4096:
            tile_m, tile_n = 256, 512
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
    print(
        f"M={args.m} N={args.n} K={args.k} groups={args.groups} kernel={args.kernel} -> tile={tile}"
    )
