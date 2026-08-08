"""The full cost (settle + clear_soft) in the convolution form: no covariance matrix, no
per-timestep quadratic forms.

    .venv/bin/python -m studies.bench.clark_hinge_fast
    .venv/bin/python -m studies.bench.clark_hinge_fast --element cylinder --seeds 5

`clark_conv.py` collapsed the SETTLE cost from O(3TK|U| + |U|^2) to O(3TK^2 + P) using two
facts: a moment-matched max node is an explicit linear functional of its own candidates, and
the map's correlation is separable with finite support. The clearance term resisted only
because a hinge is non-linear AFTER the max. It does not actually resist: every quantity the
hinge needs is still a linear functional of raw cells, so the same collapse applies once the
pieces are written in the right order.

THE ONE IDENTITY THAT DOES THE WORK. Write each object as a weight vector over cells:

    env node (w,t)      W_{w,t} = sum_k weight_k * cell_k            (clark_conv fact 1)
    belly ground (t,i)  g_{t,i} = the 4 bilinear stencil weights
    chassis height      w_z_{t,i} = sum_w a_i[w] W_{w,t}             (the tripod map)
    hinge argument      F_{t,i} = g_{t,i} - sum_w a_i[w] W_{w,t}
    settle              G_s     = sum_node c_node W_node

Then, with Phi_{t,i} the hinge's activation probability and Y = sum Phi F the Phi-weighted
combination that approximation (f) needs,

    Var[J_full] = Var[settle] + Var[clear] + 2 Cov(settle, clear)
                = quad(G_s + G_h) + sum_{t,i} ( Var[hinge_{t,i}] - Phi_{t,i}^2 Var[X_{t,i}] )

with G_h = sum_{t,i} Phi_{t,i} F_{t,i}. Every cross term between settle and clearance, and
between hinges at different timesteps, is inside that ONE quadratic form -- which is two 1-D
convolutions of a single scattered field. `clark_hinge.py` spends T dense |U| x |U| quadratic
forms plus an [N, K, |U|] tensor to compute the same number.

What stays local: each hinge still needs its own Var[X_{t,i}] before its Phi is known, which is
Var(ground) - 2 sum_w a_w Cov(ground, node_w) + a^T C_nn a. All three come from small blocks
evaluated directly on integer cell offsets through the separable kernel -- 4 x K per
(belly point, wheel) and K x K per wheel pair -- so no shared universe is ever built.

Verified against `clark_hinge.coupled_cost_plan_moments` on every case before any timing is
reported; the arithmetic is algebraically identical but regrouped, so agreement is to
floating-point accumulation order (~1e-12 relative), not bit-identity.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import warp as wp

from ..adjoint.harness import DERIV_WZ
from ..adjoint.harness import Harness
from .clark import _footprint_cells
from .clark import _settle_weights
from .clark import rho1_table
from .clark_conv import cylinder_offsets
from .clark_conv import fold_weights
from .clark_conv import WHEEL_HALF_WIDTH
from .clark_full import _bilinear_stencil
from .clark_full import _hinge_inputs
from .clark_full import _hinge_moments
from .clark_full import CLEAR_MARGIN
from .clark_hinge import belly_pose_weights
from .clark_hinge import coupled_cost_plan_moments
from .ranking import build_case
from .ranking import CELL
from .ranking import N_PLANS
from .ranking import OUT
from .risk import CORR_LEN
from helhest.engine import RobotParams
from helhest.engine.envelope import wheel_offset_table


_RHO_PAD: dict[int, tuple[np.ndarray, int]] = {}


def _rho_table(rho1: np.ndarray, max_lag: int = 1024) -> tuple[np.ndarray, int]:
    """A zero-padded, SIGNED-lag lookup: `tab[lag + off]` is rho1(|lag|), zero outside support.
    Folding the abs, the clip and the out-of-support `where` into the table turns each
    correlation evaluation into one gather -- the four passes over an 800k-element offset stack
    that `clark.rho_lookup` needs were 77% of this function's runtime."""
    key = id(rho1)
    if key not in _RHO_PAD:
        tab = np.zeros(2 * max_lag + 1)
        lim = rho1.shape[0]
        tab[max_lag : max_lag + lim] = rho1
        tab[max_lag - lim + 1 : max_lag + 1] = rho1[::-1]
        _RHO_PAD[key] = (tab, max_lag)
    return _RHO_PAD[key]


def _rho(rho1: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """rho1(|dy|) * rho1(|dx|), zero outside the kernel's support, as two gathers."""
    tab, off = _rho_table(rho1)
    return tab[dy + off] * tab[dx + off]


def _quad(field: np.ndarray, rho1: np.ndarray) -> float:
    kernel = np.concatenate([rho1[:0:-1], rho1])
    tmp = np.apply_along_axis(np.convolve, 1, field, kernel, mode="same")
    tmp = np.apply_along_axis(np.convolve, 0, tmp, kernel, mode="same")
    return float((field * tmp).sum())


def full_cost_moments_conv(
    belief: np.ndarray, sigma: np.ndarray, controlled: np.ndarray, derived: np.ndarray,
    rp: RobotParams, chassis_pts: np.ndarray, clear_margin: float, x0: float, y0: float,
    cell: float, rho1: np.ndarray, element: str = "sphere",
) -> dict:
    """E[J], Var[J] for J = settle + clear_soft, with no covariance matrix anywhere. Same
    return keys as `clark_hinge.coupled_cost_plan_moments` minus the frozen-model diagnostics."""
    ny, nx = belief.shape
    belief_flat, sigma_flat = belief.ravel(), sigma.ravel()
    wheel_xy = np.array([[0.0, rp.half_track], [0.0, -rp.half_track], [-rp.rear_offset, 0.0]])
    t_idx = np.arange(1, controlled.shape[0])
    n_t = len(t_idx)
    x, y, yaw = controlled[t_idx, 0], controlled[t_idx, 1], controlled[t_idx, 2]
    c, s = np.cos(yaw), np.sin(yaw)
    wx = np.stack([x + wheel_xy[w, 0] * c - wheel_xy[w, 1] * s for w in range(3)]).ravel()
    wy = np.stack([y + wheel_xy[w, 0] * s + wheel_xy[w, 1] * c for w in range(3)]).ravel()

    # --- the envelope element ------------------------------------------------------------
    if element == "sphere":
        env_radius = int(np.ceil(rp.wheel_radius / cell))
        off_dy, off_dx, off_cap = wheel_offset_table(env_radius, cell, rp.wheel_radius)
        off_dy, off_dx = np.asarray(off_dy, np.int64), np.asarray(off_dx, np.int64)
        env_cells = _footprint_cells(wx, wy, off_dy, off_dx, x0, y0, cell, ny, nx)
        means_env = belief_flat[env_cells] + off_cap[None, :]
        d_y = off_dy[:, None] - off_dy[None, :]
        d_x = off_dx[:, None] - off_dx[None, :]
        rho_kk = _rho(rho1, d_y, d_x)
    else:
        n_bins = 32
        bins = np.round(np.tile(yaw, 3) / (2 * np.pi) * n_bins).astype(np.int64) % n_bins
        tables = {
            b: cylinder_offsets(cell, rp.wheel_radius, WHEEL_HALF_WIDTH, 2 * np.pi * b / n_bins)
            for b in np.unique(bins)
        }
        k_min = min(len(t[0]) for t in tables.values())
        off_dy = np.stack([tables[b][0][:k_min] for b in bins])
        off_dx = np.stack([tables[b][1][:k_min] for b in bins])
        off_cap = np.stack([tables[b][2][:k_min] for b in bins])
        iy0 = np.round((wy - y0) / cell).astype(np.int64)
        ix0 = np.round((wx - x0) / cell).astype(np.int64)
        env_cells = np.clip(iy0[:, None] + off_dy, 0, ny - 1) * nx + np.clip(
            ix0[:, None] + off_dx, 0, nx - 1
        )
        means_env = belief_flat[env_cells] + off_cap
        rho_kk = _rho(rho1, off_dy[:, :, None] - off_dy[:, None, :],
                      off_dx[:, :, None] - off_dx[:, None, :])

    sig_env = sigma_flat[env_cells]
    mean_env, w_env, order_env = fold_weights(means_env, sig_env, rho_kk)

    # node (w, t) as a weight vector over its own K cells, in sorted order
    node_cells = np.take_along_axis(env_cells, order_env, axis=1)         # [3T, K]
    node_w = w_env * np.take_along_axis(sig_env, order_env, axis=1)       # weight * sigma
    node_iy, node_ix = node_cells // nx, node_cells % nx
    n_nodes, k_env = node_cells.shape
    # node index layout is wheel-major then time, matching _footprint_cells' input order
    nodes_t = np.arange(3)[:, None] * n_t + np.arange(n_t)[None, :]       # [3, T]

    # --- settle -----------------------------------------------------------------------------
    c_w = np.repeat(_settle_weights(rp), n_t)
    e_settle = float(c_w @ mean_env) + n_t * DERIV_WZ * rp.wheel_radius

    # --- belly points -------------------------------------------------------------------------
    wxb, wyb, mu0 = _hinge_inputs(controlled, derived, chassis_pts, clear_margin)  # [T, Np]
    n_p = chassis_pts.shape[0]
    hinge_cells, hinge_w = _bilinear_stencil(wxb.ravel(), wyb.ravel(), x0, y0, cell, ny, nx)
    hinge_cells = hinge_cells.reshape(n_t, n_p, 4)
    hinge_w = hinge_w.reshape(n_t, n_p, 4)
    h_iy, h_ix = hinge_cells // nx, hinge_cells % nx
    h_wsig = hinge_w * sigma_flat[hinge_cells]                            # weight * sigma
    a = belly_pose_weights(rp, chassis_pts)                               # [Np, 3]

    # Var(ground): 4x4 block per (t, i)
    r_gg = _rho(rho1, h_iy[..., :, None] - h_iy[..., None, :],
                h_ix[..., :, None] - h_ix[..., None, :])
    var_ground = np.einsum("tpc,tpcd,tpd->tp", h_wsig, r_gg, h_wsig)

    # Cov(ground_{t,i}, node_{w,t}): 4 x K block per (t, i, w)
    nw_iy = node_iy[nodes_t]                                              # [3, T, K]
    nw_ix = node_ix[nodes_t]
    nw_w = node_w[nodes_t]
    r_gn = _rho(
        rho1,
        h_iy[:, :, :, None, None] - np.moveaxis(nw_iy, 1, 0)[:, None, None, :, :],
        h_ix[:, :, :, None, None] - np.moveaxis(nw_ix, 1, 0)[:, None, None, :, :],
    )                                                                     # [T, Np, 4, 3, K]
    cov_gn = np.einsum("tpc,tpcwk,twk->tpw", h_wsig, r_gn, np.moveaxis(nw_w, 1, 0))

    # Cov(node_{w,t}, node_{v,t}): K x K block per (t, w, v)
    r_nn = _rho(
        rho1,
        np.moveaxis(nw_iy, 1, 0)[:, :, :, None, None] - np.moveaxis(nw_iy, 1, 0)[:, None, None, :, :],
        np.moveaxis(nw_ix, 1, 0)[:, :, :, None, None] - np.moveaxis(nw_ix, 1, 0)[:, None, None, :, :],
    )                                                                     # [T, 3, K, 3, K]
    c_nn = np.einsum("twk,twkvl,tvl->twv", np.moveaxis(nw_w, 1, 0), r_nn,
                     np.moveaxis(nw_w, 1, 0))

    # --- hinge arguments ----------------------------------------------------------------------
    # mean: anchored at the real belief pose, corrected by the envelope max's Jensen uplift
    delta = (mean_env - means_env.max(axis=1)).reshape(3, n_t)            # [3, T]
    mu_frozen = mu0 + (hinge_w * belief_flat[hinge_cells]).sum(axis=2)    # [T, Np]
    mu_x = mu_frozen - (a @ delta).T                                      # [T, Np]
    var_x = np.maximum(
        var_ground
        - 2.0 * np.einsum("pw,tpw->tp", a, cov_gn)
        + np.einsum("pw,twv,pv->tp", a, c_nn, a),
        0.0,
    )
    e_hinge, var_hinge, phi_pos = _hinge_moments(mu_x.ravel(), var_x.ravel())
    e_hinge = e_hinge.reshape(n_t, n_p)
    var_hinge = var_hinge.reshape(n_t, n_p)
    phi = phi_pos.reshape(n_t, n_p)

    # --- ONE scattered field, ONE quadratic form ----------------------------------------------
    pad = rho1.shape[0]
    all_iy = np.concatenate([node_iy.ravel(), h_iy.ravel()])
    all_ix = np.concatenate([node_ix.ravel(), h_ix.ravel()])
    y_lo, x_lo = int(all_iy.min()) - pad, int(all_ix.min()) - pad
    shape = (int(all_iy.max()) + pad + 1 - y_lo, int(all_ix.max()) + pad + 1 - x_lo)

    g_s = np.zeros(shape)
    np.add.at(g_s, (node_iy - y_lo, node_ix - x_lo), c_w[:, None] * node_w)

    g_h = np.zeros(shape)
    np.add.at(g_h, (h_iy - y_lo, h_ix - x_lo), phi[:, :, None] * h_wsig)   # + ground part
    # - chassis part: each belly point pulls on its timestep's three nodes through a_i
    node_pull = np.einsum("tp,pw->twp", phi, a).sum(axis=2)                # [T, 3] total weight
    np.add.at(
        g_h,
        (np.moveaxis(nw_iy, 1, 0) - y_lo, np.moveaxis(nw_ix, 1, 0) - x_lo),
        -node_pull[:, :, None] * np.moveaxis(nw_w, 1, 0),
    )

    var_settle = _quad(g_s, rho1)
    var_j = _quad(g_s + g_h, rho1) + float((var_hinge - phi**2 * var_x).sum())
    e_clear = float(e_hinge.sum())
    return {
        "e_j": e_settle + e_clear, "var_j": max(var_j, 0.0),
        "e_settle": e_settle, "var_settle": max(var_settle, 0.0),
        "e_clear": e_clear,
    }


def run(device: str, n_seeds: int, element: str) -> dict:
    rp = RobotParams()
    chassis_pts = rp._chassis_pts()
    rho1 = rho1_table(CORR_LEN, CELL)
    rows = []
    for seed in range(n_seeds):
        scene, _, _, _, sigma, poses, omega, _ = build_case(seed, "hybrid", "all")
        belief = scene.elevation.astype(np.float32)
        h = Harness(scene, poses, omega, device=device)
        h.forward(dilate=True)
        controlled, derived = h.sim.controlled.numpy(), h.sim.derived.numpy()
        del h
        geo = (scene.origin_x, scene.origin_y, CELL)

        t0 = time.perf_counter()
        ref = [
            coupled_cost_plan_moments(belief, sigma, controlled[:, k, :], derived[:, k, :], rp,
                                      chassis_pts, CLEAR_MARGIN, *geo,
                                      rho1_table(CORR_LEN, CELL))
            for k in range(N_PLANS)
        ]
        t_ref = (time.perf_counter() - t0) / N_PLANS
        t0 = time.perf_counter()
        fast = [
            full_cost_moments_conv(belief, sigma, controlled[:, k, :], derived[:, k, :], rp,
                                   chassis_pts, CLEAR_MARGIN, *geo, rho1, element=element)
            for k in range(N_PLANS)
        ]
        t_fast = (time.perf_counter() - t0) / N_PLANS

        se = max(abs(r["e_j"]) for r in ref)
        sv = max(abs(r["var_j"]) for r in ref)
        de = max(abs(r["e_j"] - f["e_j"]) for r, f in zip(ref, fast)) / max(se, 1e-12)
        dv = max(abs(r["var_j"] - f["var_j"]) for r, f in zip(ref, fast)) / max(sv, 1e-12)
        rows.append({"seed": seed, "rel_dE": de, "rel_dVar": dv, "ref_s": t_ref,
                     "fast_s": t_fast, "speedup": t_ref / max(t_fast, 1e-12)})
        print(f"  seed {seed:2d}: rel dE {de:.2e}  rel dVar {dv:.2e}   "
              f"{t_ref*1e3:6.2f} -> {t_fast*1e3:5.2f} ms/plan  ({t_ref/t_fast:.1f}x)")
    kind = "numerical_error" if element == "sphere" else "physics_difference_vs_sphere"
    return {
        "element": element, "delta_meaning": kind, "rows": rows,
        "max_rel_dE": max(r["rel_dE"] for r in rows),
        "max_rel_dVar": max(r["rel_dVar"] for r in rows),
        "median_ref_ms": float(np.median([r["ref_s"] for r in rows]) * 1e3),
        "median_fast_ms": float(np.median([r["fast_s"] for r in rows]) * 1e3),
        "median_speedup": float(np.median([r["speedup"] for r in rows])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--element", default="sphere", choices=("sphere", "cylinder"))
    args = ap.parse_args()
    wp.init()
    print(f"=== full cost (settle + clear_soft), convolution form, {args.element} ===")
    out = run(args.device, args.seeds, args.element)
    label = ("numerical error vs clark_hinge.py" if args.element == "sphere"
             else "PHYSICS difference vs the sphere element, not an error")
    print(f"\n  max relative dE   {out['max_rel_dE']:.2e}   ({label})")
    print(f"  max relative dVar {out['max_rel_dVar']:.2e}")
    print(f"  median {out['median_ref_ms']:.2f} -> {out['median_fast_ms']:.2f} ms/plan"
          f"  ({out['median_speedup']:.1f}x)")
    path = OUT / f"clark_hinge_fast_{args.element}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
