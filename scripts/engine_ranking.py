"""Score the SAME candidate population in our engine, and compare the ordering with Chrono's.

    python scripts/engine_ranking.py /tmp/chrono_rank.npz

IMPROVEMENTS.md section 10's Level-1 vs Level-3 ablation. The controls are read out of Chrono's
own output file rather than regenerated, so the two populations are identical by construction and
no seed has to agree across two environments.

`k_turn` is swept, because our alpha is a FITTED constant (alpha = 1 + k_turn * mu) while Chrono's
emerges from geometry and friction. Comparing at the shipped k_turn = 0.6 would mostly measure that
one known gap; the matched value isolates everything else, which is the interesting part.

Ranking metric is distance from the final pose to a goal, which is the dominant MPPI term on flat
ground (the terrain-dependent penalties are all zero there). Reported as Kendall tau over the whole
population plus overlap of the elite set CEM would actually average.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import warp as wp

from helhest import dynamics
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams
from helhest.heightmap import _grid
from helhest.heightmap import Heightmap

CELL, EXTENT, MU, DT = 0.1, 30.0, 0.8, 0.1
GOALS = ((6.0, 0.0), (5.0, 2.0), (4.0, -2.5))


def flat() -> Heightmap:
    XX, _ = _grid((-EXTENT, EXTENT), (-EXTENT, EXTENT), CELL)
    return Heightmap(np.zeros_like(XX).astype(np.float32), (-EXTENT, -EXTENT), CELL)


def run_engine(U: np.ndarray, k_turn: float, device: str) -> np.ndarray:
    """U is [T, N, 2]; returns [N, T+1, 3] poses (x, y, yaw)."""
    horizon, n, _ = U.shape
    scene = flat()
    sp = SolverParams(dt=DT, k_turn=k_turn, newton_iters=6, atol=1e-4,
                      tau_motor=dynamics.MOTOR_TAU)
    sim = ForwardSimulator(
        RobotParams(), sp,
        GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0), n, horizon, device,
    )
    sim.set_terrain(
        wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device)
    )
    sim.set_uniform_friction(MU)
    om = np.zeros((horizon, n, 3), np.float32)
    om[:, :, 0] = U[:, :, 0]
    om[:, :, 1] = U[:, :, 1]
    om[:, :, 2] = 0.5 * (U[:, :, 0] + U[:, :, 1])
    controlled, *_ = sim.rollout(np.ascontiguousarray(om), (0.0, 0.0, 0.0))
    del sim
    return np.transpose(controlled, (1, 0, 2)).astype(np.float64)


def tau_b(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    da = np.sign(a[:, None] - a[None, :])
    db = np.sign(b[:, None] - b[None, :])
    up = np.triu(np.ones((n, n), bool), 1)
    p = (da * db)[up]
    conc, disc = int((p > 0).sum()), int((p < 0).sum())
    ta = int(((da == 0) & (db != 0))[up].sum())
    tb = int(((db == 0) & (da != 0))[up].sum())
    den = np.sqrt((conc + disc + ta) * (conc + disc + tb))
    return float((conc - disc) / den) if den else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("chrono")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    wp.init()
    d = np.load(args.chrono)
    U, ref = d["U"].astype(np.float64), d["trajs"].astype(np.float64)
    n = U.shape[1]
    print(f"{n} candidates, horizon {U.shape[0]} at dt {DT}\n")

    print(f"{'k_turn':>8}{'alpha':>8}{'endpt mean':>12}{'endpt p95':>11}"
          + "".join(f"{f'tau/elite g{i}':>16}" for i in range(len(GOALS))))
    for k_turn in (dynamics.K_TURN, 0.75, 1.0, 1.5, 2.71):
        eng = run_engine(U, k_turn, args.device)
        derr = np.linalg.norm(eng[:, -1, :2] - ref[:, -1, :2], axis=1)
        cells = ""
        for gx, gy in GOALS:
            ce = np.linalg.norm(eng[:, -1, :2] - np.array([gx, gy]), axis=1)
            cr = np.linalg.norm(ref[:, -1, :2] - np.array([gx, gy]), axis=1)
            k = max(int(0.1 * n), 1)
            ov = len(set(np.argsort(ce)[:k]) & set(np.argsort(cr)[:k])) / k
            cells += f"{tau_b(ce, cr):>9.3f}/{ov:>5.0%}"
        head = f"{k_turn:>8.2f}{1 + k_turn * MU:>8.2f}"
        print(head + f"{derr.mean():>11.3f}m{np.percentile(derr, 95):>10.3f}m" + cells)

    # where do they disagree most? that is the input to any fix
    eng = run_engine(U, 0.75, args.device)
    derr = np.linalg.norm(eng[:, -1, :2] - ref[:, -1, :2], axis=1)
    dyaw = np.degrees(np.abs(np.arctan2(np.sin(eng[:, -1, 2] - ref[:, -1, 2]),
                                        np.cos(eng[:, -1, 2] - ref[:, -1, 2]))))
    diff = np.abs(U[:, :, 1] - U[:, :, 0]).mean(axis=0)
    speed = 0.5 * (U[:, :, 0] + U[:, :, 1]).mean(axis=0)
    print(f"\nat k_turn 0.75: endpoint error {derr.mean():.3f} m mean, "
          f"{derr.max():.3f} m worst; final-yaw error {dyaw.mean():.1f} deg mean, "
          f"{dyaw.max():.1f} deg worst")
    q = np.quantile(diff, [0.25, 0.5, 0.75])
    for lo, hi, lbl in ((0, q[0], "near-straight"), (q[0], q[1], "gentle"),
                        (q[1], q[2], "turning"), (q[2], 1e9, "hard turn")):
        m = (diff >= lo) & (diff < hi)
        if m.any():
            print(f"    {lbl:>14} (|wR-wL| {diff[m].mean():4.2f}, speed {speed[m].mean():4.2f}): "
                  f"endpoint {derr[m].mean():6.3f} m, yaw {dyaw[m].mean():5.1f} deg")


if __name__ == "__main__":
    main()
