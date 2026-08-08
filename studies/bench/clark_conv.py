"""The Clark estimator without any covariance matrix: O(N K^2 + P) instead of O(N K |U| + |U|^2).

    .venv/bin/python -m studies.bench.clark_conv                  # equivalence + timing
    .venv/bin/python -m studies.bench.clark_conv --element cylinder

TWO STRUCTURAL FACTS THE ORIGINAL IMPLEMENTATION DOES NOT USE.

(1) A MOMENT-MATCHED MAX NODE IS A FIXED LINEAR FUNCTIONAL OF ITS OWN CANDIDATES. Clark's
    covariance recursion is `cov_run <- phi * cov_run + (1 - phi) * cov(X_i, .)`, so unrolled it
    is just a weighted sum of the candidates' covariance rows:

        Cov(node, .) = sum_i w_i Cov(X_i, .),   w_0 = prod_{i>=1} phi_i,
                                                w_i = (1 - phi_i) prod_{j>i} phi_j

    with sum_i w_i = 1. Verified against `clark.clark_build` to 8e-16 relative. So the fold never
    needs the [N, K, |U|] candidate-to-universe tensor: it needs only the node's own K x K
    covariance, and it can EMIT K weights per node instead of a |U|-vector.

(2) THE CORRELATION IS SEPARABLE WITH HARD-ZERO SUPPORT. `clark.rho_lookup` is
    rho(dy, dx) = rho1(|dy|) rho1(|dx|), exactly zero past 11 cells. A plan's cost is a linear
    functional of cells, J = const + sum_p g_p h_p, so

        Var[J] = sum_{p,q} g_p g_q sigma_p sigma_q rho1(dy) rho1(dx)
               = <G, (G * rho1) * rho1>,   G_p = g_p sigma_p on the grid

    -- two 1-D convolutions with an 11-tap kernel over the corridor patch, instead of a dense
    |U| x |U| quadratic form. The covariance matrix is never built.

WHAT THIS COSTS, PER PLAN. The old path builds cov_u (|U|^2), a [N, K, |U|] tensor, folds it,
replays a second pass for the [N, N] node covariance, and contracts. This path builds one K x K
correlation matrix ONCE for the whole run (the footprint offsets are the same for every node, so
only the sigma outer product differs), folds K x K, scatters N*K weights onto a patch, and runs
two convolutions. |U| and N drop out of the dominant term entirely.

WHY THE CYLINDER WHEEL MAKES IT BETTER STILL. Both terms that remain scale with K, and K is set
by the structuring element: the spherical wheel's disk has K = 37 candidates at 0.10 m cells,
while the measured 0.10 m-wide cylinder (`envelope.cylinder_offset_table`, added on
engine/exact-arc-integration) is a 0.70 x 0.10 m rotated rectangle -- K = 5-7 depending on
heading. The fold's K^2 term therefore drops by ~28x and the scattered field gets thinner.
`--element cylinder` runs the estimator on that element; note it also changes the physics, so
those numbers are a cost measurement, not a like-for-like accuracy comparison with the sphere.

Everything is checked against `clark.clark_plan_moments` on every case before any timing is
reported.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import warp as wp

from ..adjoint.harness import DERIV_WZ
from ..adjoint.harness import Harness
from .clark import _footprint_cells
from .clark import _settle_weights
from .clark import _norm_cdf
from .clark import _norm_pdf
from .clark import clark_plan_moments
from .clark import rho1_table
from .clark import rho_lookup
from .ranking import build_case
from .ranking import CELL
from .ranking import N_PLANS
from .ranking import OUT
from .risk import CORR_LEN
from helhest.engine import RobotParams
from helhest.engine.envelope import wheel_offset_table

WHEEL_HALF_WIDTH = 0.05  # [m] half of the ruler-measured 0.10 m tread (engine/robot.py)


def cylinder_offsets(
    cell: float, radius: float, half_width: float, yaw: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The cylinder wheel's structuring element at heading `yaw`, mirroring
    `helhest.engine.envelope.cylinder_offset_table`: the rotated rectangle
    2*radius (along travel) x 2*half_width (across), capped by the along-travel offset alone."""
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    env_radius = int(math.ceil(math.hypot(radius, half_width) / cell))
    dy_l, dx_l, cap_l = [], [], []
    for dy in range(-env_radius, env_radius + 1):
        for dx in range(-env_radius, env_radius + 1):
            wx, wy = dx * cell, dy * cell
            along = wx * cos_y + wy * sin_y
            across = -wx * sin_y + wy * cos_y
            if abs(along) <= radius and abs(across) <= half_width:
                dy_l.append(dy)
                dx_l.append(dx)
                cap_l.append(math.sqrt(radius**2 - along**2) - radius)
    return np.array(dy_l, np.int64), np.array(dx_l, np.int64), np.array(cap_l, np.float64)


def fold_weights(
    means: np.ndarray, sigmas: np.ndarray, rho_kk: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clark's fold on the node's own K candidates only, without materializing any [N, K, K]
    tensor. `rho_kk` is [K, K] (one element shared by all nodes) or [N, K, K] (per-node element,
    e.g. the yaw-dependent cylinder). Returns (node mean [N], candidate weights [N, K] in SORTED
    order summing to 1, sort order [N, K]).

    The covariance row needed at step i, Cov(X_i, X_j) for all j, is gathered from `rho_kk` on
    the fly rather than read out of a pre-sorted 3-D copy: that is the same arithmetic with one
    [N, K] gather per step instead of two sorts and a concatenate of [N, K, 2K]."""
    n, k = means.shape
    order = np.argsort(-means, axis=1)
    means_s = np.take_along_axis(means, order, axis=1)
    sig_s = np.take_along_axis(sigmas, order, axis=1)
    vars_s = sig_s**2
    rows = np.arange(n)
    per_node = rho_kk.ndim == 3

    def cov_row(i: int) -> np.ndarray:
        """Cov(X_i, X_j) for every j, all nodes: [N, K]."""
        if per_node:
            rho = rho_kk[rows[:, None], order[:, i][:, None], order]
        else:
            rho = rho_kk[order[:, i][:, None], order]
        return rho * sig_s[:, i][:, None] * sig_s

    mean_run = means_s[:, 0].copy()
    var_run = vars_s[:, 0].copy()
    cov_run = cov_row(0)
    phi_seq = np.empty((n, k - 1))
    phineg_seq = np.empty((n, k - 1))
    for i in range(1, k):
        m2, v2 = means_s[:, i], vars_s[:, i]
        c12 = cov_run[:, i]
        a = np.sqrt(np.maximum(var_run + v2 - 2.0 * c12, 0.0))
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
        var_run = np.maximum(new_ex2 - new_mean**2, 0.0)
        cov_run *= phi_a[:, None]
        cov_run += cov_row(i) * phi_na[:, None]
        phi_seq[:, i - 1], phineg_seq[:, i - 1] = phi_a, phi_na
        mean_run = new_mean

    w = np.empty((n, k))
    tail = np.ones(n)
    for i in range(k - 1, 0, -1):
        w[:, i] = phineg_seq[:, i - 1] * tail
        tail = tail * phi_seq[:, i - 1]
    w[:, 0] = tail
    return mean_run, w, order


def _separable_quadratic(
    field: np.ndarray, rho1: np.ndarray
) -> float:
    """<G, (G * rho1) * rho1> for a 2-D field, by two 1-D convolutions with the symmetric
    (2L-1)-tap kernel built from rho1. This is the plan's variance."""
    kernel = np.concatenate([rho1[:0:-1], rho1])
    tmp = np.apply_along_axis(np.convolve, 1, field, kernel, mode="same")
    tmp = np.apply_along_axis(np.convolve, 0, tmp, kernel, mode="same")
    return float((field * tmp).sum())


def plan_moments_conv(
    belief: np.ndarray, sigma: np.ndarray, controlled: np.ndarray, rp: RobotParams,
    x0: float, y0: float, cell: float, rho1: np.ndarray,
    element: str = "sphere", rho_kk_cache: dict | None = None,
) -> tuple[float, float]:
    """E[J_settle], Var[J_settle] for one plan, with no covariance matrix anywhere."""
    ny, nx = belief.shape
    belief_flat, sigma_flat = belief.ravel(), sigma.ravel()
    # A real map has holes. Replaying a bag through this estimator showed every plan returning
    # a non-finite (E, Var) with no complaint, because NaN flows through the fold silently and
    # comes out the far end. Unobserved cells are the CALLER's decision -- fill policy changes
    # the ranking sharply -- so refuse the input rather than invent one.
    if not np.isfinite(belief).all():
        raise ValueError(
            "belief contains non-finite cells: choose a fill policy before calling "
            "(ground-referenced fill is the one that does not create phantom plateaus)"
        )
    if not np.isfinite(sigma).all():
        raise ValueError("sigma contains non-finite cells")
    wheel_xy = np.array([[0.0, rp.half_track], [0.0, -rp.half_track], [-rp.rear_offset, 0.0]])
    t_idx = np.arange(1, controlled.shape[0])
    n_t = len(t_idx)
    x, y, yaw = controlled[t_idx, 0], controlled[t_idx, 1], controlled[t_idx, 2]
    c, s = np.cos(yaw), np.sin(yaw)
    wx = np.stack([x + wheel_xy[w, 0] * c - wheel_xy[w, 1] * s for w in range(3)]).ravel()
    wy = np.stack([y + wheel_xy[w, 0] * s + wheel_xy[w, 1] * c for w in range(3)]).ravel()

    if element == "sphere":
        env_radius = int(np.ceil(rp.wheel_radius / cell))
        off_dy, off_dx, off_cap = wheel_offset_table(env_radius, cell, rp.wheel_radius)
        off_dy, off_dx = np.asarray(off_dy, np.int64), np.asarray(off_dx, np.int64)
        cell_flat = _footprint_cells(wx, wy, off_dy, off_dx, x0, y0, cell, ny, nx)
    else:
        # The cylinder is not yaw-invariant: one element per pose, quantized to the same yaw
        # bins the simulator uses so the tables are shared rather than rebuilt per node.
        n_bins = 32
        bins = np.round(np.tile(yaw, 3) / (2 * np.pi) * n_bins).astype(np.int64) % n_bins
        tables = {
            b: cylinder_offsets(cell, rp.wheel_radius, WHEEL_HALF_WIDTH, 2 * np.pi * b / n_bins)
            for b in np.unique(bins)
        }
        k_min = min(len(t[0]) for t in tables.values())
        off_dy = np.stack([tables[b][0][:k_min] for b in bins])  # [N, K]
        off_dx = np.stack([tables[b][1][:k_min] for b in bins])
        off_cap = np.stack([tables[b][2][:k_min] for b in bins])
        iy0 = np.round((wy - y0) / cell).astype(np.int64)
        ix0 = np.round((wx - x0) / cell).astype(np.int64)
        iy = np.clip(iy0[:, None] + off_dy, 0, ny - 1)
        ix = np.clip(ix0[:, None] + off_dx, 0, nx - 1)
        cell_flat = iy * nx + ix

    # --- the K x K correlation of the element, built once and reused for every node ----------
    key = (element, off_dy.shape[-1])
    if rho_kk_cache is not None and key in rho_kk_cache and element == "sphere":
        rho_kk = rho_kk_cache[key]
    else:
        d_y = off_dy[..., :, None] - off_dy[..., None, :]
        d_x = off_dx[..., :, None] - off_dx[..., None, :]
        rho_kk = rho_lookup(rho1, d_y, d_x)
        if rho_kk_cache is not None and element == "sphere":
            rho_kk_cache[key] = rho_kk

    means = belief_flat[cell_flat] + (off_cap[None, :] if element == "sphere" else off_cap)
    sigmas = sigma_flat[cell_flat]
    mean_n, w, order = fold_weights(means, sigmas, rho_kk)

    # --- E[J]: unchanged --------------------------------------------------------------------
    c_w = np.repeat(_settle_weights(rp), n_t)
    e_j = float(c_w @ mean_n) + n_t * DERIV_WZ * rp.wheel_radius

    # --- Var[J]: scatter c_a * w_{a,i} * sigma onto a patch, then two convolutions ------------
    cells_sorted = np.take_along_axis(cell_flat, order, axis=1)
    iy_all, ix_all = cells_sorted // nx, cells_sorted % nx
    pad = rho1.shape[0]
    y_lo, y_hi = int(iy_all.min()) - pad, int(iy_all.max()) + pad + 1
    x_lo, x_hi = int(ix_all.min()) - pad, int(ix_all.max()) + pad + 1
    patch = np.zeros((y_hi - y_lo, x_hi - x_lo))
    contrib = (c_w[:, None] * w * sigma_flat[cells_sorted]).ravel()
    np.add.at(patch, ((iy_all - y_lo).ravel(), (ix_all - x_lo).ravel()), contrib)
    return e_j, max(_separable_quadratic(patch, rho1), 0.0)


def run(device: str, n_seeds: int, element: str) -> dict:
    rp = RobotParams()
    rho1 = rho1_table(CORR_LEN, CELL)
    cache: dict = {}
    rows = []
    for seed in range(n_seeds):
        scene, _, _, _, sigma, poses, omega, _ = build_case(seed, "hybrid", "all")
        belief = scene.elevation.astype(np.float32)
        h = Harness(scene, poses, omega, device=device)
        h.forward(dilate=True)
        controlled = h.sim.controlled.numpy()
        del h
        geo = (scene.origin_x, scene.origin_y, CELL)

        t0 = time.perf_counter()
        ref = [
            clark_plan_moments(belief, sigma, controlled[:, k, :], rp, *geo,
                               rho1_table(CORR_LEN, CELL))
            for k in range(N_PLANS)
        ]
        t_ref = (time.perf_counter() - t0) / N_PLANS
        t0 = time.perf_counter()
        fast = [
            plan_moments_conv(belief, sigma, controlled[:, k, :], rp, *geo, rho1,
                              element=element, rho_kk_cache=cache)
            for k in range(N_PLANS)
        ]
        t_fast = (time.perf_counter() - t0) / N_PLANS

        # For the sphere this is a NUMERICAL equivalence check against clark.py. For the
        # cylinder it is not: the element changes the contact model, so the delta is physics
        # (the cylinder does not climb rocks the sphere straddles) and only the timing is
        # comparable. Reported under a different key so the two can never be confused.
        scale_e = max(abs(a[0]) for a in ref)
        scale_v = max(abs(a[1]) for a in ref)
        de = max(abs(a[0] - b[0]) for a, b in zip(ref, fast)) / max(scale_e, 1e-12)
        dv = max(abs(a[1] - b[1]) for a, b in zip(ref, fast)) / max(scale_v, 1e-12)
        row = {"seed": seed, "rel_dE": de, "rel_dVar": dv, "ref_s": t_ref, "fast_s": t_fast,
               "speedup": t_ref / max(t_fast, 1e-12)}
        if element == "sphere":
            # A planner ranks a candidate SET, so the batched fold is the deployment case.
            t0 = time.perf_counter()
            bat = plan_moments_conv_batch(belief, sigma, controlled, rp, *geo, rho1)
            row["batch_s"] = (time.perf_counter() - t0) / N_PLANS
            row["batch_rel_dE"] = max(
                abs(a[0] - b[0]) for a, b in zip(ref, bat)) / max(scale_e, 1e-12)
            row["batch_rel_dVar"] = max(
                abs(a[1] - b[1]) for a, b in zip(ref, bat)) / max(scale_v, 1e-12)
        rows.append(row)
        extra = (f"  batched {row['batch_s']*1e3:.2f}" if "batch_s" in row else "")
        print(f"  seed {seed:2d}: rel dE {de:.2e}  rel dVar {dv:.2e}   "
              f"{t_ref*1e3:6.2f} -> {t_fast*1e3:5.2f} ms/plan  ({t_ref/t_fast:.1f}x){extra}")
    kind = "numerical_error" if element == "sphere" else "physics_difference_vs_sphere"
    return {
        "element": element, "delta_meaning": kind, "rows": rows,
        "max_rel_dE": max(r["rel_dE"] for r in rows),
        "max_rel_dVar": max(r["rel_dVar"] for r in rows),
        "median_ref_ms": float(np.median([r["ref_s"] for r in rows]) * 1e3),
        "median_fast_ms": float(np.median([r["fast_s"] for r in rows]) * 1e3),
        "median_speedup": float(np.median([r["speedup"] for r in rows])),
        "median_batch_ms": (
            float(np.median([r["batch_s"] for r in rows]) * 1e3) if "batch_s" in rows[0] else None
        ),
        "max_batch_rel_dVar": (
            max(r["batch_rel_dVar"] for r in rows) if "batch_s" in rows[0] else None
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--element", default="sphere", choices=("sphere", "cylinder"))
    args = ap.parse_args()
    wp.init()
    print(f"=== convolution form vs clark.py ({args.element} element) ===")
    out = run(args.device, args.seeds, args.element)
    label = ("numerical error vs clark.py" if args.element == "sphere"
             else "PHYSICS difference vs the sphere element, not an error")
    print(f"\n  max relative dE   {out['max_rel_dE']:.2e}   ({label})")
    print(f"  max relative dVar {out['max_rel_dVar']:.2e}")
    print(f"  median {out['median_ref_ms']:.2f} -> {out['median_fast_ms']:.2f} ms/plan"
          f"  ({out['median_speedup']:.1f}x)")
    if out.get("median_batch_ms"):
        print(f"  batched over the candidate set: {out['median_batch_ms']:.2f} ms/plan"
              f"  ({out['median_ref_ms']/out['median_batch_ms']:.1f}x overall,"
              f" max rel dVar {out['max_batch_rel_dVar']:.1e})")
    path = OUT / f"clark_conv_{args.element}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")



# --- batching every plan through one fold ------------------------------------------------------
def plan_moments_conv_batch(
    belief: np.ndarray, sigma: np.ndarray, controlled_all: np.ndarray, rp: RobotParams,
    x0: float, y0: float, cell: float, rho1: np.ndarray,
) -> list[tuple[float, float]]:
    """`plan_moments_conv` for ALL plans of a scene at once (sphere element).

    The fold is K sequential steps of about a dozen numpy calls on [3T, K] arrays -- 4.4k
    elements, small enough that per-call interpreter and dispatch overhead is a third of the
    runtime. The steps are sequential in K but INDEPENDENT across plans, so stacking the plans
    into the row axis runs the same 444 numpy calls on 16x the data and amortizes that overhead.
    Only the fold is batched; the scatter and the convolution stay per plan, because plans sweep
    different corridors and a shared patch would be mostly zeros.
    """
    ny, nx = belief.shape
    belief_flat, sigma_flat = belief.ravel(), sigma.ravel()
    n_plans = controlled_all.shape[1]
    env_radius = int(np.ceil(rp.wheel_radius / cell))
    off_dy, off_dx, off_cap = wheel_offset_table(env_radius, cell, rp.wheel_radius)
    off_dy, off_dx = np.asarray(off_dy, np.int64), np.asarray(off_dx, np.int64)
    rho_kk = rho_lookup(rho1, off_dy[:, None] - off_dy[None, :], off_dx[:, None] - off_dx[None, :])
    wheel_xy = np.array([[0.0, rp.half_track], [0.0, -rp.half_track], [-rp.rear_offset, 0.0]])
    t_idx = np.arange(1, controlled_all.shape[0])
    n_t = len(t_idx)

    cells = []
    for k in range(n_plans):
        ctl = controlled_all[:, k, :]
        x, y, yaw = ctl[t_idx, 0], ctl[t_idx, 1], ctl[t_idx, 2]
        c, s = np.cos(yaw), np.sin(yaw)
        wx = np.stack([x + wheel_xy[w, 0] * c - wheel_xy[w, 1] * s for w in range(3)]).ravel()
        wy = np.stack([y + wheel_xy[w, 0] * s + wheel_xy[w, 1] * c for w in range(3)]).ravel()
        cells.append(_footprint_cells(wx, wy, off_dy, off_dx, x0, y0, cell, ny, nx))
    cell_flat = np.concatenate(cells, axis=0)  # [n_plans * 3T, K]

    means = belief_flat[cell_flat] + off_cap[None, :]
    mean_n, w, order = fold_weights(means, sigma_flat[cell_flat], rho_kk)

    c_w = np.repeat(_settle_weights(rp), n_t)
    cells_sorted = np.take_along_axis(cell_flat, order, axis=1)
    wsig = w * sigma_flat[cells_sorted]
    pad = rho1.shape[0]
    n_nodes = 3 * n_t
    out = []
    for k in range(n_plans):
        sl = slice(k * n_nodes, (k + 1) * n_nodes)
        e_j = float(c_w @ mean_n[sl]) + n_t * DERIV_WZ * rp.wheel_radius
        iy, ix = cells_sorted[sl] // nx, cells_sorted[sl] % nx
        y_lo, x_lo = int(iy.min()) - pad, int(ix.min()) - pad
        patch = np.zeros((int(iy.max()) + pad + 1 - y_lo, int(ix.max()) + pad + 1 - x_lo))
        np.add.at(patch, (iy - y_lo, ix - x_lo), c_w[:, None] * wsig[sl])
        out.append((e_j, max(_separable_quadratic(patch, rho1), 0.0)))
    return out

if __name__ == "__main__":
    main()
