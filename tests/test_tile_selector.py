"""Unit tests for autotune/tile_selector.py.

Run with:
    python3 -m unittest discover -s tests -p "test_*.py"
or:
    pytest tests/test_tile_selector.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "autotune"))

from tile_selector import TILES, _snap_tile_m, select_tile


def tile_dims(tile: str) -> tuple[int, int, int]:
    """Parse a "_16, _256, _32" tile string into (tile_m, tile_n, tile_k)."""
    return tuple(int(part.strip().lstrip("_")) for part in tile.split(","))


class TestTileMTracksProblemM(unittest.TestCase):
    """Measured on Wildcat Lake at N=9728 K=896 (k2): auto's tile_m must
    track the problem M for M >= 32 — undersized tile_m lost at M=32
    (389.5 vs 376.6 us), M=64 (458.7 vs 439.8), and M=128 (825 vs 612)."""

    N, K = 9728, 896

    def test_tile_m_tracks_m(self):
        for m in (16, 32, 64, 128):
            tile_m, _, _ = tile_dims(select_tile(m, self.N, self.K, "k2"))
            self.assertEqual(
                tile_m, m, f"M={m}: expected tile_m to track M, got tile_m={tile_m}"
            )

    def test_measured_winners(self):
        # Sweep winners at M=16/32/64 (M=128's winner was 128x256, but the
        # selector keeps the N/K tile dims of its base pick, checked above).
        self.assertEqual(select_tile(16, self.N, self.K, "k2"), "_16, _256, _32")
        self.assertEqual(select_tile(32, self.N, self.K, "k2"), "_32, _256, _32")
        self.assertEqual(select_tile(64, self.N, self.K, "k2"), "_64, _256, _32")

    def test_tile_m_never_exceeds_m_at_or_above_32(self):
        for m in (32, 48, 64, 96, 128, 256, 512):
            for n in (256, 1024, 4096, 9728):
                for k in (384, 896, 4096):
                    tile = select_tile(m, n, k, "k2")
                    tile_m, tile_n, tile_k = tile_dims(tile)
                    smaller_exists = any(
                        int(key.split("x")[0]) <= m
                        for key in TILES
                        if key.split("x")[1:] == [str(tile_n), str(tile_k)]
                    )
                    if smaller_exists:
                        self.assertLessEqual(
                            tile_m, m, f"M={m} N={n} K={k}: tile {tile} exceeds M"
                        )

    def test_selected_tile_is_instantiated(self):
        for m in (16, 32, 64, 128, 512):
            tile = select_tile(m, self.N, self.K, "k2")
            tile_m, tile_n, tile_k = tile_dims(tile)
            self.assertIn(f"{tile_m}x{tile_n}x{tile_k}", TILES)

    def test_skinny_m_rules_unchanged(self):
        # M < 32 keeps the sweep-validated skinny-M picks (tile_m > M can win).
        self.assertEqual(select_tile(1, 8192, 4096), "_4, _256, _32")
        self.assertEqual(select_tile(4, 4096, 4096), "_32, _256, _32")


class TestSnapTileM(unittest.TestCase):
    def test_snaps_down_to_closest(self):
        self.assertEqual(_snap_tile_m(128, 32, 128, 32), 128)
        self.assertEqual(_snap_tile_m(96, 32, 256, 32), 64)

    def test_falls_back_to_smallest_when_none_at_most_m(self):
        # (64, 64) family has no instantiation <= 32; smallest is 128.
        self.assertEqual(_snap_tile_m(32, 256, 64, 64), 128)

    def test_keeps_tile_m_when_family_missing(self):
        self.assertEqual(_snap_tile_m(64, 48, 999, 32), 48)


if __name__ == "__main__":
    unittest.main()
