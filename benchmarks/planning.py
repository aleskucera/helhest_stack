"""Timing benchmark for the planning/ stack: the cost-to-go routing solve, as the robot runs it.

`CostToGo.compute` (planning/costtogo.py) builds the orientation-aware routing field
V(x, y, theta): the settle at every pose, read as margins in sigmas, then value iteration. The node
runs it once per frame on its routing window (`route_m` max-pooled by `plan_lat_coarsen`), with the
robot's cost-to-go settings from `ros/config/odin.params.yaml` (see `_common`). The first row is
that point; the sweep varies the heading bin count around it, which the solve scales with.

Compare with the node's `plan:ctg` stage (`profile_stages`, over a bag replay), which also carries
the belief crops feeding it. CUDA-only (graph capture); skips cleanly without a GPU.

Run from the repo root:  python -m benchmarks.planning [--world slalom]
"""

from __future__ import annotations

import argparse

import warp as wp
from helhest import worlds as W

from ._common import build_costtogo
from ._common import robot_scene
from ._common import route_inputs
from ._common import time_fn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", default="slalom", choices=list(W.WORLDS))
    args = ap.parse_args()

    wp.init()
    if not wp.is_cuda_available():
        print("CUDA not available -- the cost-to-go solve is GPU-only (graph capture). Skipping.")
        return
    device = "cuda"
    rs = robot_scene(args.world)
    inputs = route_inputs(rs, device)
    reps = 15
    grid = rs.route_grid
    print(
        f"\n=== cost-to-go  world={args.world}  routing window {grid.cells_y}x{grid.cells_x} "
        f"at {grid.cell_size:.2f} m (the robot's) ==="
    )
    print(f"    {'n_theta':>7} {'states':>9} {'solve_ms':>9}")
    robot_n_theta = rs.cfg.n_theta
    for n_theta in sorted({robot_n_theta, 16, 32}):
        ctg = build_costtogo(rs, device, n_theta=n_theta)
        t = time_fn(lambda: ctg.compute(goal_xy=rs.goal_route, **inputs), reps, device)
        mark = "  <- robot" if n_theta == robot_n_theta else ""
        states = grid.cells_x * grid.cells_y * n_theta
        print(f"    {n_theta:>7} {states:>9} {t * 1e3:>9.2f}{mark}")


if __name__ == "__main__":
    main()
