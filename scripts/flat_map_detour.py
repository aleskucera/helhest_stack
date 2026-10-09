"""Does the lattice planner drive straight across flat ground? Regression check for arc pricing.

    python scripts/flat_map_detour.py
    python scripts/flat_map_detour.py --plot /tmp/flat_detour.png

Perfectly flat terrain, start and goal on the same line, nothing blocked and nothing to avoid, so
the optimal route is exactly the segment between them and any deviation is the planner's own doing.
That makes this the sharpest possible test of the arc COST, with terrain contributing nothing.

It exists because it used to fail. _build_primitives charged every primitive the same nominal arc
length (LatticeValueSolver step, 0.3 m) whatever its endpoint, while ROUNDING that endpoint to
whole cells: a gently turning step lands 3 cells along and 2 across, covering 0.36 m of ground for
a 0.30 m fee. Weaving was therefore cheaper per metre than driving straight, and this map came back
with a route that bulged 3.85 m sideways and covered 18.09 m instead of 16.00 m. Primitives are now
charged the displacement they realize, so the same map returns 16.28 m and 0.35 m of deviation --
the remainder being the heading quantization (24 bins, so bin 0 points at 7.5 deg, not 0).

Flat ground is the worst case precisely because it is featureless: on sloped terrain the graded
tilt cost broke the tie by accident, which is why the gentle 5 deg ramp in the benchmark series
(scripts/bench_ramp_series.py) used to come out straighter than flat ground did.

The assertion at the end is the actual check -- every primitive's fee must equal its reach.

CUDA-only (graph capture); skips cleanly without a GPU.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import warp as wp
from helhest import dynamics
from helhest.engine import GridParams
from helhest.planning.costtogo import CostToGo
from helhest.planning.lattice_solver import trace_optimal

EXTENT_X = 20.0  # [m]
EXTENT_Y = 8.0
CELL = 0.1
N_THETA = 24
MARGIN = 2.0  # [m] start/goal inset from the grid edge


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--plot", type=pathlib.Path, default=None, help="write a PNG of the route")
    args = ap.parse_args()

    wp.init()
    if not wp.is_cuda_available():
        print("CUDA not available -- the cost-to-go solve is GPU-only (graph capture). Skipping.")
        return

    nx, ny = int(EXTENT_X / CELL), int(EXTENT_Y / CELL)
    x0, y0 = -0.5 * EXTENT_X, -0.5 * EXTENT_Y
    ctg = CostToGo(
        GridParams(nx, ny, CELL, x0, y0),
        dynamics.robot_params(),
        dynamics.planning_solver(),
        n_theta=N_THETA,
        device="cuda",
    )

    elevation = wp.zeros((ny, nx), dtype=wp.float32, device="cuda")  # z = 0 everywhere
    start = (x0 + MARGIN, 0.0, 0.0)  # facing +x, straight at the goal
    goal = (x0 + EXTENT_X - MARGIN, 0.0)
    ctg.compute(elevation, goal)
    assert float(ctg.blocked.numpy().max()) == 0.0, "flat ground must block nothing"

    pts = trace_optimal(ctg, start, N_THETA, nx, ny, x0, y0, CELL)
    seg = np.hypot(*np.diff(pts, axis=0).T)  # the length of each step actually taken
    straight = float(np.hypot(goal[0] - start[0], goal[1] - start[1]))
    start_rc = (int((start[1] - y0) / CELL), int((start[0] - x0) / CELL), 0)

    print(
        f"\nflat map: {ny}x{nx} cells @ {CELL} m, z = 0 everywhere, 0 poses blocked\n"
        f"start ({start[0]:.1f}, {start[1]:.1f}) facing +x  ->  goal ({goal[0]:.1f}, {goal[1]:.1f})\n"
        f"\n  straight-line distance     {straight:6.2f} m   <- the optimal route\n"
        f"  distance actually covered  {seg.sum():6.2f} m   ({len(seg)} steps)\n"
        f"  furthest off the line      {float(np.abs(pts[:, 1] - start[1]).max()):6.2f} m\n"
        f"  the planner's own estimate {float(ctg.V.numpy()[start_rc]):6.2f} m   (V at the start)\n"
    )
    lengths, counts = np.unique(np.round(seg[1:], 2), return_counts=True)  # [0] is a partial cell
    print(
        "  step lengths really taken: "
        + ", ".join(f"{v:.2f} m x{c}" for v, c in zip(lengths, counts))
    )

    # The regression check. Each primitive must be charged the displacement it realizes on the
    # grid; when it was charged a flat `step` instead, the (2, 3) primitive covered 0.36 m for a
    # 0.30 m fee, and weaving came out cheaper per metre than driving straight.
    s = ctg.solver
    reach = CELL * np.hypot(s._prim_dr.numpy().astype(float), s._prim_dc.numpy().astype(float))
    mismatch = float(np.abs(reach - s._prim_cost.numpy()).max())
    print("\n  primitive table -- ground covered vs fee charged, over all headings:")
    for v in np.unique(np.round(reach, 2)):
        fees = s._prim_cost.numpy()[np.round(reach, 2) == v]
        print(f"    covers {v:.2f} m  ->  charged {fees.min():.2f}-{fees.max():.2f} m")
    print(f"  worst mismatch anywhere in the table: {mismatch:.4f} m\n")
    assert mismatch < 1e-6, "primitives are not charged what they cover -- the detour is back"

    if args.plot is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(
            [start[0], goal[0]],
            [start[1], goal[1]],
            "--",
            color="0.5",
            lw=2,
            label=f"optimal: straight, {straight:.2f} m",
        )
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            "-",
            color="tab:orange",
            lw=2.5,
            label=f"planned: {seg.sum():.2f} m, {float(np.abs(pts[:, 1]).max()):.2f} m off the line",
        )
        ax.plot(*start[:2], "o", color="lime", ms=10, mec="black")
        ax.plot(*goal, "X", color="red", ms=12, mec="black")
        ax.set_xlim(x0, x0 + EXTENT_X)
        ax.set_ylim(y0, y0 + EXTENT_Y)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
        ax.set_title(
            "flat ground, nothing blocked -- the route should be the straight line, and now is"
        )
        fig.savefig(args.plot, dpi=120, bbox_inches="tight")
        print(f"saved {args.plot}")


if __name__ == "__main__":
    main()
