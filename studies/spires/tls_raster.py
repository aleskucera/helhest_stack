"""Rasterize a site's TLS merged cloud to a cached 0.10 m max-grid (.npz).

The TLS PCD is the survey ground truth; the max-per-cell convention matches the
engine's drive-on-max contact model and the prereg's FROZEN mu-layer choice.
Streamed in chunks -- the merged clouds are 65-114M points and must not be
loaded whole.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import BinaryIO

import numpy as np

CELL = 0.10
CHUNK = 4_000_000


def _parse_header(f: BinaryIO) -> tuple[list[str], list[int], list[int], int]:
    fields: list[str] = []
    sizes: list[int] = []
    counts: list[int] = []
    npts = 0
    while True:
        line = f.readline().decode("ascii", "replace").strip()
        if line.startswith("FIELDS"):
            fields = line.split()[1:]
        elif line.startswith("SIZE"):
            sizes = [int(x) for x in line.split()[1:]]
        elif line.startswith("COUNT"):
            counts = [int(x) for x in line.split()[1:]]
        elif line.startswith("POINTS"):
            npts = int(line.split()[1])
        elif line.startswith("DATA"):
            if "binary" not in line:
                raise ValueError("only binary PCD supported")
            return fields, sizes, counts, npts


def _offsets(fields: list[str], sizes: list[int], counts: list[int]) -> tuple[dict[str, int], int]:
    off = 0
    out: dict[str, int] = {}
    for name, sz, ct in zip(fields, sizes, counts):
        out[name] = off
        off += sz * ct
    return out, off


def rasterize_tls(pcd_path: Path) -> tuple[np.ndarray, float, float]:
    """Two streamed passes: bounds, then per-cell max. Returns (H, x0, y0)."""
    with open(pcd_path, "rb") as f:
        fields, sizes, counts, npts = _parse_header(f)
        offs, stride = _offsets(fields, sizes, counts)
        data_start = f.tell()

        def col(buf: np.ndarray, name: str) -> np.ndarray:
            o = offs[name]
            return buf[:, o : o + 4].copy().view(np.float32)[:, 0]

        lo = np.array([np.inf, np.inf])
        hi = -lo.copy()
        for s in range(0, npts, CHUNK):
            n = min(CHUNK, npts - s)
            buf = np.frombuffer(f.read(n * stride), np.uint8).reshape(n, stride)
            x, y = col(buf, "x"), col(buf, "y")
            lo = np.minimum(lo, [np.nanmin(x), np.nanmin(y)])
            hi = np.maximum(hi, [np.nanmax(x), np.nanmax(y)])
        nx = int(np.ceil((hi[0] - lo[0]) / CELL)) + 1
        ny = int(np.ceil((hi[1] - lo[1]) / CELL)) + 1
        grid = np.full(nx * ny, -np.inf, np.float32)
        f.seek(data_start)
        for s in range(0, npts, CHUNK):
            n = min(CHUNK, npts - s)
            buf = np.frombuffer(f.read(n * stride), np.uint8).reshape(n, stride)
            x, y, z = col(buf, "x"), col(buf, "y"), col(buf, "z")
            ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            idx = ((y[ok] - lo[1]) / CELL).astype(np.int64) * nx + (
                (x[ok] - lo[0]) / CELL
            ).astype(np.int64)
            np.maximum.at(grid, idx, z[ok])
    H = grid.reshape(ny, nx)
    H[~np.isfinite(H)] = np.nan
    return H, float(lo[0]), float(lo[1])


def cache_path(site: str, root: Path) -> Path:
    return root / "ground_truth_map" / site / "tls_max_raster.npz"


def load_or_build(site: str, root: Path = Path("/home/kuceral4/data/oxford_spires")) -> dict:
    """Load the cached raster, building it on first use."""
    cp = cache_path(site, root)
    if not cp.exists():
        H, x0, y0 = rasterize_tls(root / "ground_truth_map" / site / "merged-cloud-1cm.pcd")
        np.savez_compressed(cp, H=H, x0=x0, y0=y0, cell=CELL)
    d = np.load(cp)
    return {"H": d["H"], "x0": float(d["x0"]), "y0": float(d["y0"]), "cell": float(d["cell"])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sites", nargs="+")
    args = ap.parse_args()
    for site in args.sites:
        r = load_or_build(site)
        H = r["H"]
        print(
            f"{site}: {H.shape[1]}x{H.shape[0]} cells, "
            f"{np.isfinite(H).sum()/1e3:.0f}k filled, origin ({r['x0']:.1f}, {r['y0']:.1f})"
        )
