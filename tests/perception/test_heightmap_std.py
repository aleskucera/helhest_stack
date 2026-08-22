"""Per-cell std layer of HeightMapBuilder vs a numpy reference.

Guards the float64 accumulation path: cell heights are O(10) m but per-cell
sigma is O(0.02) m, so var = E[z^2] - E[z]^2 cancels ~7 significant digits --
float32 accumulation would destroy the estimate. Also guards that adding the
std layer left max/mean/min/count untouched.

Run: python tests/perception/test_heightmap_std.py
"""

from __future__ import annotations

import numpy as np
import warp as wp
from helhest.perception.heightmap import HeightMapBuilder

RES = 0.1
BOUNDS = (0.0, 2.0, 0.0, 2.0)  # (xmin, xmax, ymin, ymax) -> 20x20 grid


def _build_reference(points: np.ndarray) -> dict[str, np.ndarray]:
    """numpy per-cell max/mean/min/count/std reference, same convention as the kernel."""
    xmin, _, ymin, _ = BOUNDS
    width = int(round((BOUNDS[1] - BOUNDS[0]) / RES))
    height = int(round((BOUNDS[3] - BOUNDS[2]) / RES))
    j = ((points[:, 0] - xmin) / RES).astype(int)
    i = ((points[:, 1] - ymin) / RES).astype(int)
    keep = (i >= 0) & (i < height) & (j >= 0) & (j < width)
    i, j, z = i[keep], j[keep], points[keep, 2]

    ref = {
        name: np.full((height, width), np.nan, np.float64) for name in ("max", "mean", "min", "std")
    }
    count = np.zeros((height, width), np.int64)
    for r, c, v in zip(i, j, z):
        count[r, c] += 1
    for r in range(height):
        for c in range(width):
            cell_z = z[(i == r) & (j == c)]
            if len(cell_z) == 0:
                continue
            ref["max"][r, c] = cell_z.max()
            ref["min"][r, c] = cell_z.min()
            ref["mean"][r, c] = cell_z.mean()
            ref["std"][r, c] = cell_z.std(ddof=0) if len(cell_z) >= 2 else np.nan
    ref["count"] = count
    return ref


def test_std_matches_numpy_and_other_layers_unchanged() -> None:
    rng = np.random.default_rng(0)
    base_z = -6.0
    sigma = 0.02  # the hard case: sigma is ~1e-3 of |z|, needs float64 accumulation

    points = []
    # dense clusters (>=2 points) at known cells, each with its own Gaussian sigma.
    cluster_cells = [(2, 3), (5, 5), (10, 12), (15, 15), (18, 1)]
    for r, c in cluster_cells:
        x = BOUNDS[0] + (c + 0.5) * RES
        y = BOUNDS[2] + (r + 0.5) * RES
        n = rng.integers(5, 50)
        z = rng.normal(base_z, sigma, n)
        points.append(np.stack([np.full(n, x), np.full(n, y), z], axis=1))

    # single-point cell -> std must be NaN (count == 1).
    single_r, single_c = 7, 8
    sx = BOUNDS[0] + (single_c + 0.5) * RES
    sy = BOUNDS[2] + (single_r + 0.5) * RES
    points.append(np.array([[sx, sy, base_z]]))

    points = np.concatenate(points, axis=0).astype(np.float32)
    # empty cells (e.g. (0, 0)) get no points at all -> std/max/mean/min NaN, count 0.

    builder = HeightMapBuilder(RES, BOUNDS, device=wp.get_device("cpu"))
    layers = builder.build(points)
    got = layers.to_numpy()

    ref = _build_reference(points.astype(np.float64))

    assert np.array_equal(got["count"], ref["count"])
    for name in ("max", "mean", "min"):
        assert np.allclose(got[name], ref[name], atol=1e-5, equal_nan=True), name

    assert np.isnan(got["std"][single_r, single_c]), "single-point cell must be NaN"
    assert np.isnan(got["std"][0, 0]), "empty cell must be NaN"
    assert np.allclose(got["std"], ref["std"], atol=1e-5, equal_nan=True), "std"


def main() -> None:
    wp.init()
    test_std_matches_numpy_and_other_layers_unchanged()
    print("PASS: heightmap std layer matches numpy; max/mean/min/count unchanged")


if __name__ == "__main__":
    main()
