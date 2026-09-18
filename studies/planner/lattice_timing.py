"""Plan section 6.7: how long does the cost-to-go solve take, and can we afford two of them?

This became a safety number the moment the plan proposed replacing MPPI with a carrot: the
lattice is then the sole authority, so replan latency bounds how far the robot commits blind.
It also decides whether the optimistic/pessimistic two-solve gap of section 4.3 -- the trigger
for exploration and the automatic diagnosis of an ignorance-blocked goal -- is affordable at
all, since that scheme pays for a second solve every cycle.

Deployed settings, from ros/odin/odin_elevation.params.yaml and the node's own defaults:
resolution 0.08 m, plan_lat_coarsen 3 (the params file overrides the node's 4), so a 0.24 m
routing cell; plan_n_theta 24; a 16 m window. The sensor frame is 69 ms at 14.5 Hz.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import warp as wp

from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams
from helhest.planning.costtogo import CostToGo

FRAME_MS = 1000.0 / 14.5  # one Odin dTOF sweep


def time_solve(window_m: float, routing_cell: float, n_theta: int, reps: int = 12) -> dict:
    n = int(round(window_m / routing_cell))
    grid = GridParams(
        cells_x=n, cells_y=n, cell_size=routing_cell, origin_x=-window_m / 2, origin_y=-window_m / 2
    )
    ctg = CostToGo(grid, RobotParams(), SolverParams(), n_theta=n_theta, step=3 * routing_cell)
    rng = np.random.default_rng(0)
    xs = np.linspace(-window_m / 2, window_m / 2, n)
    terrain = 0.25 * np.sin(xs[None, :] * 0.8) + 0.2 * np.cos(xs[:, None] * 0.6)
    terrain += rng.normal(0, 0.01, terrain.shape)
    elev = wp.array(terrain.astype(np.float32), dtype=wp.float32)

    ctg.compute(elev, (window_m / 2 - routing_cell, 0.0))  # warm up: builds the CUDA graph
    wp.synchronize()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        ctg.compute(elev, (window_m / 2 - routing_cell, 0.0))
        wp.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.array(times)
    return {
        "window_m": window_m,
        "routing_cell": routing_cell,
        "n_theta": n_theta,
        "cells": n * n,
        "poses": n * n * n_theta,
        "median_ms": float(np.median(t)),
        "p90_ms": float(np.percentile(t, 90)),
        "frame_frac": float(np.median(t) / FRAME_MS),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="studies/out/planner")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    configs = [
        ("deployed (coarsen 3)", 16.0, 0.24, 24),
        ("coarsen 4", 16.0, 0.32, 24),
        ("coarsen 2", 16.0, 0.16, 24),
        ("12 m window", 12.0, 0.24, 24),
        ("optimistic solve, 12 headings", 16.0, 0.24, 12),
        ("optimistic solve, coarse+12", 16.0, 0.32, 12),
    ]
    rows = []
    print(f"frame budget {FRAME_MS:.1f} ms (14.5 Hz)\n")
    print(f"{'config':30s} {'poses':>9s} {'median_ms':>10s} {'p90_ms':>8s} {'% frame':>8s}")
    for name, w, c, nt in configs:
        r = time_solve(w, c, nt)
        r["name"] = name
        rows.append(r)
        print(
            f"{name:30s} {r['poses']:9d} {r['median_ms']:10.2f} {r['p90_ms']:8.2f} "
            f"{r['frame_frac']*100:7.1f}%"
        )

    dep = rows[0]["median_ms"]
    print(f"\ntwo-solve schemes against a {FRAME_MS:.0f} ms frame:")
    for name, extra in (
        ("pessimistic x2", dep),
        ("pessimistic + 12-heading", rows[4]["median_ms"]),
        ("pessimistic + coarse 12-heading", rows[5]["median_ms"]),
    ):
        tot = dep + extra
        print(f"   {name:32s} {tot:6.2f} ms  = {tot/FRAME_MS*100:5.1f}% of a frame")

    dst = os.path.join(args.out, "lattice_timing.json")
    with open(dst, "w") as fh:
        json.dump({"frame_ms": FRAME_MS, "configs": rows}, fh, indent=1)
    print("\n->", dst)


if __name__ == "__main__":
    main()
