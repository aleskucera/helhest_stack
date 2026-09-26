"""Scratch scene for the REPL pane: a field, a producer and a world already wired up.

Run with `python -i studies/terrain_value_field/repl.py`. Everything is left in module scope so
the prompt lands with something to poke at rather than fifteen lines of setup to retype.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import warp as wp

from helhest.grid import build_grid
from helhest.planning.terrain_value_field import omni_control_set
from helhest.planning.terrain_value_field import TerrainValueField

# the geometric producer is a test fixture now; the repo root makes `tests` importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tests.planning.terrain_value_field.geometric_producer import GeometricProducer  # noqa: E402

N, CELL = 81, 0.1
XS = -N * CELL / 2 + np.arange(N) * CELL

grid = build_grid(N, N, CELL, -N * CELL / 2, -N * CELL / 2)
height = (0.35 * np.sin(XS[None, :] * 1.1) + 0.25 * np.cos(XS[:, None] * 0.9)).astype(np.float32)
height_sd = np.full((N, N), 0.02, np.float32)

field = TerrainValueField(N, N, CELL, n_theta=1, z_veto=2.0, control_set=omni_control_set(CELL))
producer = GeometricProducer(N, N, 1)


def solve(sd: float = 0.02, k: float = 2.0, seed=(N // 2, N - 3)):
    """Re-run the whole chain at a given map uncertainty and z_veto. Returns the `at` dict."""
    field.set_z_veto(k)
    cons = producer(
        wp.array(height, dtype=wp.float32),
        wp.array(np.full((N, N), sd, np.float32), dtype=wp.float32),
        grid,
    )
    field.seed_cell(*seed)
    field.solve_pair(cons)
    return field.at(N // 2, 2)


def show(sd: float = 0.02, k: float = 2.0):
    """One line of the numbers that usually matter."""
    st = solve(sd, k)
    print(
        f"sd={sd:.3f} k={k:.1f} | blocked {(field.pose_cost.numpy() < 0).mean() * 100:5.1f}% "
        f"doubt>0 {(field.doubt.numpy() > 0).mean() * 100:5.1f}% "
        f"z p50 {np.percentile(field.z.numpy(), 50):5.1f} | "
        f"reachable {st['reachable']} (certain {st['reachable_if_certain']})"
    )
    return st


if __name__ == "__main__":
    print(f"scene: {N}x{N} @ {CELL} m. try: show(), show(sd=0.05), show(k=3.0)")
    show()
