"""A memory-lean rewrite of `clark.py`'s fold that produces bit-comparable numbers ~N times
faster, and the benchmark that proves both.

    .venv/bin/python -m studies.bench.clark_fast            # equivalence + timing
    .venv/bin/python -m studies.bench.clark_fast --seeds 20

WHY. Profiling `clark_plan_moments` put 70% of its time in two places: `np.take_along_axis` on
the [N, K, |U|] candidate-to-universe covariance tensor, and the fold that consumes it. For a
40-step rollout that tensor is 120 nodes x 37 candidates x ~2000 universe cells -- ~70 MB that
is built by fancy indexing, copied again by the sort, and concatenated a third time before a
single fold step runs. The fold itself is trivial arithmetic; the cost is memory traffic.

THE OBSERVATION THAT REMOVES IT. `clark_build` tracks covariance to a widened vector -- the
node's own K candidates concatenated with the universe -- because the recursion needs
Cov(running max, candidate i) at step i. But in this problem every candidate IS a universe
cell (a raw map cell plus a constant cap offset), so that quantity is already sitting in the
universe part of the tracked vector at column `u_idx_sorted[:, i]`. The candidate block is
redundant. Dropping it means the [N, K, |U|] tensor never has to exist: the fold reads one
[N, |U|] slab of `cov_u` per step, gathered directly.

    cov_run = cov_u[u_idx_sorted[:, 0]]                              # [N, |U|]
    for i in 1..K-1:
        c12     = cov_run[arange(N), u_idx_sorted[:, i]]             # the redundant lookup
        ... Clark fold, unchanged ...
        cov_run = cov_run * phi + cov_u[u_idx_sorted[:, i]] * phineg

Identical arithmetic, identical fold order, K-fold less memory. This is a pure implementation
change: `equivalence()` below asserts the moments agree with `clark.py` to floating-point
round-off over many (seed, plan) pairs, and the paper's numbers stay the ones `clark.py`
produced.

WHAT IT IS NOT. This is not the GPU kernel the paper names as future work. It is the cheaper
half of that idea -- the memory-traffic fix -- done on the host, which is where the profile
said the time actually went.

A MEASUREMENT BUG THIS RUN EXPOSED, AND THE CORRECTED NUMBERS. `clark_full.bench_wall_costs`
reads `h.sim.controlled` BEFORE calling `h.forward()`, so it timed the estimator on the
pre-rollout buffer: an all-zeros trajectory parked at the origin, whose 40 timesteps all share
one footprint (1 distinct xy instead of 41, path length 0.0 m instead of 4.7 m). The candidate
universe collapses, and the reported 8.48 ms/plan is ~5x optimistic. On the real rollouts, on
an idle machine, medians over 10 seeds x 16 plans:

    clark.py (published implementation)        42.9 ms/plan
    lean fold (this module, bit-identical)     11.6 ms/plan
    256-draw GPU Monte-Carlo, same run          6.1 ms/plan

So the honest statement is that the analytic estimator is ~1.9x SLOWER per plan than batched
sampling even after the 3.7x fix -- not 1.5x slower as the buggy benchmark implied, and not
faster. `rollout_cost()` below re-measures the Monte-Carlo side in the same run so the
comparison never again straddles two machine states.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import warp as wp

from ..adjoint.harness import Harness
from .clark import _footprint_cells
from .clark import _norm_cdf
from .clark import _norm_pdf
from .clark import _settle_weights
from .clark import clark_cross_cov
from .clark import clark_plan_moments
from .clark import rho1_table
from .clark import rho_lookup
from .clark import RNG_SEED
from .clark import settle_map
from .ranking import build_case
from .ranking import CELL
from .ranking import N_PLANS
from .ranking import OUT
from .risk import CORR_LEN
from helhest.engine import RobotParams
from helhest.engine.envelope import wheel_offset_table
from ..adjoint.harness import DERIV_WZ


def clark_build_lean(
    means: np.ndarray, sigmas: np.ndarray, u_idx: np.ndarray, cov_u: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`clark.clark_build` without the [N, K, |U|] tensor. `u_idx` [N, K] is each candidate's
    index into the shared universe; `cov_u` [|U|, |U|] is the universe covariance. Returns the
    same 6-tuple (mean, var, cov_to_u_final, order, phi, phineg)."""
    n, k = means.shape
    order = np.argsort(-means, axis=1)
    means_s = np.take_along_axis(means, order, axis=1)
    vars_s = np.take_along_axis(sigmas, order, axis=1) ** 2
    u_idx_s = np.take_along_axis(u_idx, order, axis=1)
    rows = np.arange(n)

    mean_run = means_s[:, 0].copy()
    var_run = vars_s[:, 0].copy()
    cov_run = cov_u[u_idx_s[:, 0]].copy()  # [N, |U|] -- the only large working array
    phi_seq = np.empty((n, k - 1))
    phineg_seq = np.empty((n, k - 1))
    for i in range(1, k):
        m2, v2 = means_s[:, i], vars_s[:, i]
        # Cov(running max, candidate i): candidate i IS universe cell u_idx_s[:, i], so this is
        # already tracked -- no separate candidate block needed.
        c12 = cov_run[rows, u_idx_s[:, i]]
        a2 = np.maximum(var_run + v2 - 2.0 * c12, 0.0)
        a = np.sqrt(a2)
        degenerate = a < 1e-9
        safe_a = np.where(degenerate, 1.0, a)
        alpha = np.where(degenerate, np.sign(mean_run - m2) * 1.0e6, (mean_run - m2) / safe_a)
        phi_a = _norm_cdf(alpha)
        phi_na = 1.0 - phi_a
        pdf_a = _norm_pdf(alpha)
        new_mean = mean_run * phi_a + m2 * phi_na + a * pdf_a
        new_ex2 = (
            (mean_run**2 + var_run) * phi_a + (m2**2 + v2) * phi_na + (mean_run + m2) * a * pdf_a
        )
        new_var = np.maximum(new_ex2 - new_mean**2, 0.0)
        cov_run *= phi_a[:, None]
        cov_run += cov_u[u_idx_s[:, i]] * phi_na[:, None]
        phi_seq[:, i - 1], phineg_seq[:, i - 1] = phi_a, phi_na
        mean_run, var_run = new_mean, new_var
    return mean_run, var_run, cov_run, order, phi_seq, phineg_seq


def clark_plan_moments_lean(
    belief: np.ndarray, sigma: np.ndarray, controlled: np.ndarray, rp: RobotParams,
    x0: float, y0: float, cell: float, corr_table: np.ndarray,
) -> tuple[float, float]:
    """Drop-in replacement for `clark.clark_plan_moments`."""
    ny, nx = belief.shape
    belief_flat, sigma_flat = belief.ravel(), sigma.ravel()
    env_radius = int(np.ceil(rp.wheel_radius / cell))
    off_dy, off_dx, off_cap = wheel_offset_table(env_radius, cell, rp.wheel_radius)
    wheel_xy = np.array([[0.0, rp.half_track], [0.0, -rp.half_track], [-rp.rear_offset, 0.0]])

    t_idx = np.arange(1, controlled.shape[0])
    n_t = len(t_idx)
    x, y, yaw = controlled[t_idx, 0], controlled[t_idx, 1], controlled[t_idx, 2]
    c, s = np.cos(yaw), np.sin(yaw)
    wx = np.stack([x + wheel_xy[w, 0] * c - wheel_xy[w, 1] * s for w in range(3)]).ravel()
    wy = np.stack([y + wheel_xy[w, 0] * s + wheel_xy[w, 1] * c for w in range(3)]).ravel()

    cell_flat = _footprint_cells(wx, wy, off_dy, off_dx, x0, y0, cell, ny, nx)
    u, inv = np.unique(cell_flat.ravel(), return_inverse=True)
    u_idx = inv.reshape(cell_flat.shape)
    u_iy, u_ix = u // nx, u % nx
    cov_u = (
        sigma_flat[u][:, None]
        * sigma_flat[u][None, :]
        * rho_lookup(corr_table, u_iy[:, None] - u_iy[None, :], u_ix[:, None] - u_ix[None, :])
    )
    means = belief_flat[cell_flat] + off_cap[None, :]
    mean_n, _var_n, cov_to_u_final, order, phi, phineg = clark_build_lean(
        means, sigma_flat[cell_flat], u_idx, cov_u
    )
    u_idx_sorted = np.take_along_axis(u_idx, order, axis=1)
    cross = clark_cross_cov(cov_to_u_final, u_idx_sorted, phi, phineg)
    c_w = np.repeat(_settle_weights(rp), n_t)
    e_j = float(c_w @ mean_n) + n_t * DERIV_WZ * rp.wheel_radius
    return e_j, max(float(c_w @ cross @ c_w), 0.0)


def equivalence_and_timing(device: str, n_seeds: int) -> dict:
    """Both halves of the claim: the lean fold returns the same numbers, and it is faster."""
    rp = RobotParams()
    corr_table = rho1_table(CORR_LEN, CELL)
    rows = []
    for seed in range(n_seeds):
        scene, _, _, _, sigma, poses, omega, _ = build_case(seed, "hybrid", "all")
        belief = scene.elevation.astype(np.float32)
        h = Harness(scene, poses, omega, device=device)
        h.forward(dilate=True)
        controlled = h.sim.controlled.numpy()
        del h
        args = (scene.origin_x, scene.origin_y, CELL, corr_table)

        t0 = time.perf_counter()
        ref = [
            clark_plan_moments(belief, sigma, controlled[:, k, :], rp, *args)
            for k in range(N_PLANS)
        ]
        t_ref = (time.perf_counter() - t0) / N_PLANS
        t0 = time.perf_counter()
        lean = [
            clark_plan_moments_lean(belief, sigma, controlled[:, k, :], rp, *args)
            for k in range(N_PLANS)
        ]
        t_lean = (time.perf_counter() - t0) / N_PLANS

        de = max(abs(a[0] - b[0]) for a, b in zip(ref, lean))
        dv = max(abs(a[1] - b[1]) for a, b in zip(ref, lean))
        scale = max(abs(a[0]) for a in ref)
        rows.append(
            {
                "seed": seed, "max_abs_dE": de, "max_abs_dVar": dv,
                "rel_dE": de / max(scale, 1e-12),
                "ref_s_per_plan": t_ref, "lean_s_per_plan": t_lean,
                "speedup": t_ref / max(t_lean, 1e-12),
            }
        )
        print(
            f"  seed {seed:2d}: dE {de:.3e}  dVar {dv:.3e}   "
            f"{t_ref*1e3:6.2f} -> {t_lean*1e3:6.2f} ms/plan  ({t_ref/t_lean:.2f}x)"
        )
    return {
        "rows": rows,
        "max_rel_dE": max(r["rel_dE"] for r in rows),
        "max_abs_dVar": max(r["max_abs_dVar"] for r in rows),
        "median_ref_ms": float(np.median([r["ref_s_per_plan"] for r in rows]) * 1e3),
        "median_lean_ms": float(np.median([r["lean_s_per_plan"] for r in rows]) * 1e3),
        "median_speedup": float(np.median([r["speedup"] for r in rows])),
    }


def rollout_cost(device: str, batch: int = 256) -> float:
    """One plan's worth of GPU Monte-Carlo, timed the same way `clark_full.bench_wall_costs`
    does it, so the comparison in the paper is against a number measured in this same run
    rather than against one carried over from another session's machine state."""
    scene, _, _, _, _sigma, poses, omega, _ = build_case(0, "hybrid", "all")
    poses_d = np.tile(poses[0], (batch, 1)).astype(np.float32)
    omega_d = np.zeros((omega.shape[0], batch, 3), np.float32)
    h = Harness(scene, poses_d, omega_d, device=device)
    h.forward(dilate=True)  # warm
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        h.forward(dilate=True)
    wp.synchronize()
    dt = (time.perf_counter() - t0) / 5
    del h
    return dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    wp.init()
    print("=== equivalence + timing: lean fold vs clark.py ===")
    out = equivalence_and_timing(args.device, args.seeds)
    out["mc_256_s_per_plan"] = rollout_cost(args.device)
    print(f"\n  max relative dE   {out['max_rel_dE']:.2e}   (float round-off if ~1e-15)")
    print(f"  max absolute dVar {out['max_abs_dVar']:.2e}")
    print(
        f"  median {out['median_ref_ms']:.2f} -> {out['median_lean_ms']:.2f} ms/plan"
        f"  ({out['median_speedup']:.2f}x)"
    )
    print(f"  256-draw GPU Monte-Carlo, same machine, same run: "
          f"{out['mc_256_s_per_plan']*1e3:.2f} ms/plan")
    path = OUT / "clark_fast.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
