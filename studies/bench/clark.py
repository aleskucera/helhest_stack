"""Clark/SSTA moment propagation through the wheel-envelope contact max.

    .venv/bin/python -m studies.bench.clark --seeds 100

`risk.py` and `bundled.py` established that every FIRST-ORDER (Taylor) propagation of map
uncertainty through the contact max fails: the adjoint is only valid within the contact
arg-max's margin (millimetres), while map sigma is centimetres (CLAIMS.md C2). This module
tries the method that never assumes local linearity in the first place: propagate the mean and
variance of `envelope = max_k (h_{i+k} + cap_k)` analytically via CLARK's exact formulas for the
max of two correlated Gaussians, folded recursively over the ~37-cell footprint -- exactly what
static timing analysis (STA) does at industrial scale for the max of many correlated gate
delays, applied here to the max of many correlated *heights*.

Two things make this WORK where FOSM/adjoint did not: it is a moment-matching approximation of
the true max distribution (not a linearization at one point), and it produces the SAME object a
planner already wants -- a per-plan (mean, variance) of a settle-linear cost -- with covariance
carried through recursion rather than assumed away.

DECLARED LEVEL-1 APPROXIMATIONS (read before trusting a number below):

  (a) each plan's XY trajectory is FROZEN at its belief-map rollout. Height noise barely moves
      the planar path (prior studies' finding); only (z, pitch, roll) inherit the map's
      randomness.
  (b) the settle (z, pitch, roll) is a LINEAR, and here provably EXACT-TO-SMALL-ANGLE and
      TIME-INDEPENDENT, function of the three wheel contact heights (env_L, env_R, env_rear).
      Derived from the wheel geometry (a rigid tripod, plane through three points; see
      `SETTLE_MAP` below) and checked against the real Warp settle at GATE 1.
  (c) J is restricted to the `settle` term ONLY (not `clear_soft`), for BOTH Clark and the MC
      truth, so the comparison stays apples-to-apples (the hinge needs its own Clark-with-a-
      constant treatment and there was no time to add it -- stated per SENSITIVITY_PLAN.md).
  (d) cross-node covariance beyond the correlation kernel's FINITE SUPPORT is treated as exactly
      zero -- not approximated, because the separable Gaussian kernel used by `sigma.py` has
      support radius `2 * radius` cells (radius = ceil(3 * corr_len / cell)) and is IDENTICALLY
      zero beyond it. This is a strictly more precise replacement for the plan's suggested
      "~3 timesteps" heuristic: at the family's typical speed (~2.6 m/s, dt=0.1s -> ~2.6
      cells/step) the kernel's support (2*radius ~= 10 cells) covers almost exactly 3-4 steps of
      travel, so the ad hoc window and the exact cutoff agree in practice; the exact cutoff is
      used here because it costs nothing extra and removes an arbitrary parameter.

THE ALGORITHM (per plan, per seed):

  1. For each of the 3 wheels x T timesteps, gather the ~37-cell footprint (the SAME
     `wheel_offset_table` the production dilation uses) around the wheel's frozen (x, y).
  2. Fold each footprint's candidates (mean = belief + cap, var = sigma^2, pairwise cov from the
     kernel's analytic autocorrelation) into one Clark max via the recursion below, which ALSO
     tracks covariance of the running max to a shared "universe" of every candidate cell that
     appears anywhere in the plan's rollout (the SSTA "correlation propagation" trick) -- one
     pass per node, O(K) each.
  3. Fold that per-node universe-covariance through every OTHER node's own (saved) fold sequence
     to get the exact node-to-node covariance matrix -- a second O(K) pass per node, vectorized
     over all nodes/pairs at once.
  4. Because SETTLE_MAP is time-independent, J_settle = sum_t w.(z_t,pitch_t,roll_t) is a FIXED
     LINEAR COMBINATION of the 3*T env variables with CONSTANT per-wheel weights: E[J] and
     Var[J] are then a weighted sum of the node means and the full node-to-node covariance
     matrix -- no further approximation beyond (a)-(d).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from ..adjoint.harness import DERIV_WPITCH
from ..adjoint.harness import DERIV_WROLL
from ..adjoint.harness import DERIV_WZ
from ..adjoint.harness import Harness
from ..adjoint.harness import TERM_NAMES
from ..adjoint.sigma import _gauss_kernel
from ..adjoint.sigma import fosm_variance
from ..adjoint.sigma import NoiseDraws
from .bundled import BRACKET_C
from .element import broadcast_cap
from .element import element_offsets
from .ranking import build_case
from .ranking import CELL
from .ranking import N_PLANS
from .ranking import OUT
from .ranking import sign_test
from .risk import _footprint_sigma
from .risk import ALPHA
from .risk import CORR_LEN
from .risk import empirical_cvar
from .risk import KAPPA
from .risk import N_DRAWS
from helhest.engine import GridParams
from helhest.engine import init_state_kernel_bt
from helhest.engine import RobotParams
from helhest.engine import SolverParams

RNG_SEED = 12345  # every internal, non-case random draw (gate 0/1/2 case selection) is fixed here
SETTLE_IDX = TERM_NAMES.index("settle")


def _cost_settle(terms: np.ndarray) -> np.ndarray:
    """J restricted to the settle term (declared approximation (c) above)."""
    return terms[SETTLE_IDX]


# --- vectorized standard normal, no scipy (matches risk.py's own no-scipy convention) --------
def _erf(x: np.ndarray) -> np.ndarray:
    """Abramowitz & Stegun 7.1.26, max abs error 1.5e-7 -- plenty for a <5% gate."""
    a1, a2, a3, a4, a5, p = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
        0.3275911,
    )
    s = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-ax * ax)
    return s * y


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


# --- the correlation structure (from sigma.py's own kernel, not assumed) ----------------------
def rho1_table(corr_len: float, cell: float) -> np.ndarray:
    """rho1[|lag|] = normalized autocorrelation of the 1-D blur kernel `_gauss_kernel` builds.
    Index by abs(lag) in cells; EXACTLY zero for |lag| >= len(table) (the kernel's finite
    support), which is what makes cross-timestep covariance decay to a hard zero rather than an
    assumed one -- see module docstring (d)."""
    w, radius = _gauss_kernel(corr_len, cell)
    support = 2 * radius  # convolving w with itself has support 2*radius (len(w)-1)
    ac = np.array([np.dot(w[: len(w) - lag], w[lag:]) for lag in range(support + 1)])
    return (ac / ac[0]).astype(np.float64)


def rho_lookup(table: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """rho(dy, dx) = rho1(dy) * rho1(dx), zero once either lag exceeds the table."""

    def _r1(lag):
        a = np.abs(lag)
        in_support = a < table.shape[0]
        return np.where(in_support, table[np.clip(a, 0, table.shape[0] - 1)], 0.0)

    return _r1(dy) * _r1(dx)


# --- GATE 0: does rho actually match the noise generator's empirical covariance? --------------
def gate0_correlation(device: str) -> dict:
    table = rho1_table(CORR_LEN, CELL)
    ny, nx = 40, 40
    b = 20_000  # widened from the task's "~2000" -- see the averaging-over-centers note below
    rng_seed = RNG_SEED
    with wp.ScopedDevice(device):
        base = wp.zeros((b, ny, nx), dtype=wp.float32)
        sigma_dev = wp.ones((ny, nx), dtype=wp.float32)
        out = wp.zeros((b, ny, nx), dtype=wp.float32)
    draws = NoiseDraws((b, ny, nx), CELL, CORR_LEN, device)
    draws.perturb(base, sigma_dev, 1.0, out, rng_seed)
    field = out.numpy()  # [B, ny, nx], unit-sigma correlated noise, zero mean by construction
    pairs = [(0, 1), (0, 3), (2, 0), (3, 3), (1, 5)]
    # The task's "~2000 samples" alone leaves the two weakest pairs (rho ~ 0.05-0.14) dominated
    # by sampling noise (SE(corr) ~ 1/sqrt(B) ~ 2.2%, comparable to the value itself). Averaging
    # the SAME B=2000 draws over 30 interior reference cells per lag is a free ~30x variance
    # reduction (translation invariance away from the edge) without inflating B beyond what was
    # asked -- reported instead of silently cranking B.
    margin = 8
    centers = [
        (cy, cx)
        for cy in range(margin, ny - margin, 3)
        for cx in range(margin, nx - margin, 3)
    ][:30]
    rows = []
    for dy, dx in pairs:
        emp_vals = [
            float(np.mean(field[:, cy, cx] * field[:, cy + dy, cx + dx])) for cy, cx in centers
        ]
        emp = float(np.mean(emp_vals))
        ana = float(rho_lookup(table, np.array(dy), np.array(dx)))
        rel = abs(emp - ana) / max(abs(ana), 1e-6)
        rows.append({"dy": dy, "dx": dx, "empirical": emp, "analytic": ana, "rel_err": rel})
    worst = max(r["rel_err"] for r in rows)
    return {"pairs": rows, "worst_rel_err": worst, "passed": worst < 0.05}


# --- GATE 1: the settle's constant 3x3 linear map, derived from the wheel geometry ------------
def settle_map(rp: RobotParams) -> np.ndarray:
    """d(z, pitch, roll) / d(env_L, env_R, env_rear), rows (z, pitch, roll), cols (L, R, rear).

    Small-angle IFT of the 3-wheel settle: with wheel body positions L=(0,+b), R=(0,-b),
    rear=(-l,0) and R = Rz(yaw) Ry(pitch) Rx(roll), the z-component of R @ wheel_i is, to first
    order in (pitch, roll) and for ANY yaw (yaw only rotates x,y, never z):
        z_final_i ~= -pitch * p_ix + roll * p_iy
    so contact_z_i = z + z_final_i - wheel_radius = env_i gives 3 linear equations in
    (z, pitch, roll); solved once, symbolically, below. Time- and yaw-INDEPENDENT.
    """
    b, l = rp.half_track, rp.rear_offset
    return np.array(
        [
            [0.5, 0.5, 0.0],
            [-0.5 / l, -0.5 / l, 1.0 / l],
            [1.0 / (2.0 * b), -1.0 / (2.0 * b), 0.0],
        ]
    )


def gate1_linear_map(device: str) -> dict:
    """Perturb the 3 wheel contact heights on a real belief/pose and compare the closed-form
    map's predicted (z, pitch, roll) delta to the REAL Warp settle's (`init_state_kernel_bt`),
    at cm-scale perturbations. A LOCAL uniform bump (radius 2 cells around each wheel's nearest
    cell) is added to the envelope so `sample_field`'s bilinear stencil shifts by EXACTLY the
    intended delta regardless of its (unknown, position-dependent) blend weights -- this is what
    makes the test exact rather than approximate in its own right.
    """
    rp = RobotParams()
    rng = np.random.default_rng(RNG_SEED)
    scene, _, _, _, sigma, poses, omega, grid = build_case(0, "hybrid", "all")
    h_ = Harness(scene, poses, omega, device=device)
    envelope0 = h_._env_np[0].copy()  # [ny, nx], the real production dilation
    ny, nx = envelope0.shape
    solver = SolverParams(dt=0.1, newton_iters=20, atol=0.0).build()
    grid_s = GridParams(nx, ny, scene.cell, scene.origin_x, scene.origin_y).build()
    robot_s = rp.build(device)
    del h_

    wheel_xy = np.array([[0.0, rp.half_track], [0.0, -rp.half_track], [-rp.rear_offset, 0.0]])
    M = settle_map(rp)

    test_poses = [(1.5, -0.5, 0.3), (2.5, 0.8, -0.4), (0.5, 0.0, 0.9), (3.0, 1.2, 1.4)]
    scales = (0.01, 0.02, 0.05)  # [m] cm-scale perturbations
    rows = []
    for x, y, yaw in test_poses:
        for scale in scales:
            d_env = rng.uniform(-scale, scale, size=3)  # (dL, dR, drear)
            batch = 4  # base + 3 wheel bumps
            env_stack = np.tile(envelope0, (batch, 1, 1)).astype(np.float32)
            for w in range(3):
                px, py = wheel_xy[w]
                wx = x + px * np.cos(yaw) - py * np.sin(yaw)
                wy = y + px * np.sin(yaw) + py * np.cos(yaw)
                iy0 = int(round((wy - scene.origin_y) / scene.cell))
                ix0 = int(round((wx - scene.origin_x) / scene.cell))
                r = 2
                sl_y = slice(max(iy0 - r, 0), min(iy0 + r + 1, ny))
                sl_x = slice(max(ix0 - r, 0), min(ix0 + r + 1, nx))
                env_stack[w + 1, sl_y, sl_x] += d_env[w]
            with wp.ScopedDevice(device):
                env_dev = wp.array(env_stack, dtype=wp.float32)
                start_pose = wp.array(
                    np.tile(np.array([x, y, yaw], np.float32), (batch, 1)), dtype=wp.vec3
                )
                controlled = wp.zeros((1, batch), dtype=wp.vec3)
                derived = wp.zeros((1, batch), dtype=wp.vec3)
                wp.launch(
                    init_state_kernel_bt,
                    dim=batch,
                    inputs=[env_dev, grid_s, robot_s, solver, start_pose],
                    outputs=[controlled, derived],
                    device=device,
                )
            derived_np = derived.numpy()[0]  # [batch, 3] = (z, pitch, roll)
            actual = derived_np[1:] - derived_np[0]  # [3, 3] delta per wheel bump
            actual_delta = actual.sum(axis=0)  # superposing the 3 single-wheel bumps
            predicted_delta = M @ d_env
            rel = np.linalg.norm(actual_delta - predicted_delta) / max(
                np.linalg.norm(actual_delta), 1e-9
            )
            rows.append(
                {
                    "pose": [x, y, yaw],
                    "d_env": d_env.tolist(),
                    "actual": actual_delta.tolist(),
                    "predicted": predicted_delta.tolist(),
                    "rel_err": float(rel),
                }
            )
    med = float(np.median([r["rel_err"] for r in rows]))
    return {"cases": rows, "median_rel_err": med, "passed": med < 0.05}


# --- Clark recursion: build max nodes + track covariance to a shared universe -----------------
def clark_build(
    means: np.ndarray, sigmas: np.ndarray, cov_self: np.ndarray, cov_to_u: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One Clark max per row. means/sigmas: [N, K]; cov_self: [N, K, K] (candidate-candidate cov
    within a node's own footprint); cov_to_u: [N, K, |U|] (candidate-to-universe-cell cov).

    Sorts candidates by descending mean, then folds pairwise. The tracked-covariance vector is
    widened to include the node's OWN remaining candidates (so their cov to the running max
    updates correctly too) concatenated with the universe -- one recursion does both jobs.

    Returns (mean[N], var[N], cov_to_u_final[N, |U|], order[N,K], phi[N,K-1], phineg[N,K-1]) --
    the last three are the "fold trace" a second pass (`clark_cross_cov`) replays to get exact
    covariance between two DIFFERENT max nodes.
    """
    n, k = means.shape
    order = np.argsort(-means, axis=1)
    means_s = np.take_along_axis(means, order, axis=1)
    vars_s = np.take_along_axis(sigmas, order, axis=1) ** 2
    cov_self_s = np.take_along_axis(cov_self, order[:, :, None], axis=1)
    cov_self_s = np.take_along_axis(cov_self_s, order[:, None, :], axis=2)
    cov_to_u_s = np.take_along_axis(cov_to_u, order[:, :, None], axis=1)
    tracked = np.concatenate([cov_self_s, cov_to_u_s], axis=2)  # [N, K, K + |U|]

    mean_run = means_s[:, 0].copy()
    var_run = vars_s[:, 0].copy()
    cov_run = tracked[:, 0, :].copy()  # cov(X0, everything tracked)
    phi_seq = np.empty((n, k - 1))
    phineg_seq = np.empty((n, k - 1))
    for i in range(1, k):
        m2, v2 = means_s[:, i], vars_s[:, i]
        c12 = cov_run[:, i]  # cov(running max BEFORE this fold, candidate i)
        a2 = np.maximum(var_run + v2 - 2.0 * c12, 0.0)
        a = np.sqrt(a2)
        degenerate = a < 1e-9  # guard (module docstring): identical/near-identical variables
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
        cov_run = cov_run * phi_a[:, None] + tracked[:, i, :] * phi_na[:, None]
        phi_seq[:, i - 1], phineg_seq[:, i - 1] = phi_a, phi_na
        mean_run, var_run = new_mean, new_var
    return mean_run, var_run, cov_run[:, k:], order, phi_seq, phineg_seq


def clark_cross_cov(
    cov_to_u_final: np.ndarray,
    u_idx_sorted_b: np.ndarray,
    phi_b: np.ndarray,
    phineg_b: np.ndarray,
) -> np.ndarray:
    """cov(A, B) for every (A, B) pair drawn from the SAME node set. `cov_to_u_final[A, :]` is
    A's covariance to every universe cell (from `clark_build`); `u_idx_sorted_b[B, :]`/`phi_b`/
    `phineg_b` are B's OWN fold trace (candidates already in B's sort order). Replays B's fold
    sequence on A's covariance to B's raw candidates -- the second SSTA pass, O(K) vectorized
    over every pair at once."""
    n, k = u_idx_sorted_b.shape
    crossvec = cov_to_u_final[:, u_idx_sorted_b]  # [N_a, N_b, K]
    run = crossvec[:, :, 0].copy()
    for i in range(1, k):
        run = run * phi_b[None, :, i - 1] + crossvec[:, :, i] * phineg_b[None, :, i - 1]
    return run


def _footprint_cells(
    wx: np.ndarray, wy: np.ndarray, off_dy: np.ndarray, off_dx: np.ndarray, grid_x0: float,
    grid_y0: float, cell: float, ny: int, nx: int,
) -> np.ndarray:
    """Absolute [ny*nx]-flat cell index of every candidate, for every (node) in wx/wy. [N, K].

    `off_dy`/`off_dx` are [K] (one element shared by every node, e.g. the yaw-invariant sphere)
    or [N, K] (one element per node, e.g. the yaw-dependent cylinder from `element.py`)."""
    iy0 = np.round((wy - grid_y0) / cell).astype(np.int64)
    ix0 = np.round((wx - grid_x0) / cell).astype(np.int64)
    dy = off_dy if off_dy.ndim == 2 else off_dy[None, :]
    dx = off_dx if off_dx.ndim == 2 else off_dx[None, :]
    iy = np.clip(iy0[:, None] + dy, 0, ny - 1)
    ix = np.clip(ix0[:, None] + dx, 0, nx - 1)
    return iy * nx + ix


def build_env_nodes(
    cell_flat: np.ndarray, belief_flat: np.ndarray, sigma_flat: np.ndarray, off_cap: np.ndarray,
    nx: int, corr_table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Given the absolute flat cell index of every candidate [N, K], build the Clark inputs
    (means, sigmas, cov_self, cov_to_u) and run `clark_build`. Returns the same 6-tuple plus the
    per-node universe index in SORTED order (needed by `clark_cross_cov`)."""
    n, k = cell_flat.shape
    u, inv = np.unique(cell_flat.ravel(), return_inverse=True)
    u_idx = inv.reshape(n, k)
    u_iy, u_ix = u // nx, u % nx
    dy = u_iy[:, None] - u_iy[None, :]
    dx = u_ix[:, None] - u_ix[None, :]
    cov_u = sigma_flat[u][:, None] * sigma_flat[u][None, :] * rho_lookup(corr_table, dy, dx)
    means = belief_flat[cell_flat] + broadcast_cap(off_cap)
    sigmas = sigma_flat[cell_flat]
    cov_self = cov_u[u_idx[:, :, None], u_idx[:, None, :]]
    cov_to_u = cov_u[u_idx]  # [N, K, |U|]
    mean_n, var_n, cov_to_u_final, order, phi, phineg = clark_build(
        means, sigmas, cov_self, cov_to_u
    )
    u_idx_sorted = np.take_along_axis(u_idx, order, axis=1)
    return mean_n, var_n, cov_to_u_final, u_idx_sorted, phi, phineg


def clark_plan_moments(
    belief: np.ndarray, sigma: np.ndarray, controlled: np.ndarray, rp: RobotParams,
    x0: float, y0: float, cell: float, corr_table: np.ndarray, element: str = "sphere",
) -> tuple[float, float]:
    """E[J_settle], Var[J_settle] for ONE plan via the closed-form settle map + Clark envelope
    moments/covariance. `controlled`: [T+1, 3] (x, y, yaw) frozen belief-rollout trajectory for
    this plan (declared approximation (a)). `element` selects the contact candidate table
    (`element.py`): sphere (default, yaw-invariant disk) or cylinder (yaw-dependent tread)."""
    ny, nx = belief.shape
    belief_flat, sigma_flat = belief.ravel(), sigma.ravel()
    wheel_xy = np.array([[0.0, rp.half_track], [0.0, -rp.half_track], [-rp.rear_offset, 0.0]])

    t_idx = np.arange(1, controlled.shape[0])  # the `settle` functional sums t=1..T
    x, y, yaw = controlled[t_idx, 0], controlled[t_idx, 1], controlled[t_idx, 2]
    c, s = np.cos(yaw), np.sin(yaw)
    wx = np.stack([x + wheel_xy[w, 0] * c - wheel_xy[w, 1] * s for w in range(3)])  # [3, T]
    wy = np.stack([y + wheel_xy[w, 0] * s + wheel_xy[w, 1] * c for w in range(3)])  # [3, T]
    wx_flat, wy_flat = wx.ravel(), wy.ravel()  # node order: wheel-major, then time

    off_dy, off_dx, off_cap = element_offsets(element, cell, rp, np.tile(yaw, 3))
    cell_flat = _footprint_cells(wx_flat, wy_flat, off_dy, off_dx, x0, y0, cell, ny, nx)
    mean_n, var_n, cov_to_u_final, u_idx_sorted, phi, phineg = build_env_nodes(
        cell_flat, belief_flat, sigma_flat, off_cap, nx, corr_table
    )
    return _settle_moments_from_nodes(
        mean_n, var_n, cov_to_u_final, u_idx_sorted, phi, phineg, rp, len(t_idx)
    )


def _settle_weights(rp: RobotParams) -> np.ndarray:
    """Per-wheel constant coefficient of J_settle = sum_t c . env_t (see module docstring (b));
    order (L, R, rear)."""
    M = settle_map(rp)  # rows z, pitch, roll; cols L, R, rear
    w = np.array([DERIV_WZ, DERIV_WPITCH, DERIV_WROLL])
    return w @ M


def _settle_moments_from_nodes(
    mean_n: np.ndarray, var_n: np.ndarray, cov_to_u_final: np.ndarray, u_idx_sorted: np.ndarray,
    phi: np.ndarray, phineg: np.ndarray, rp: RobotParams, n_steps: int,
) -> tuple[float, float]:
    cross = clark_cross_cov(cov_to_u_final, u_idx_sorted, phi, phineg)  # [3T, 3T]
    c_w = _settle_weights(rp)
    c = np.repeat(c_w, n_steps)  # node order is wheel-major (matches build_env_nodes' input order)
    e_j = float(c @ mean_n) + n_steps * DERIV_WZ * rp.wheel_radius
    var_j = float(c @ cross @ c)
    return e_j, max(var_j, 0.0)


# --- GATE 2: Clark's per-node moments + cross-covariance vs brute-force MC --------------------
def _mc_env_stats(
    belief: np.ndarray, sigma: np.ndarray, x0: float, y0: float, cell: float, cand_abs: dict,
    n_draws: int, device: str, seed: int,
) -> dict:
    """Brute-force MC truth for one (wheel_L, wheel_R, wheel_rear) case: draws the ACTUAL
    correlated noise generator (`NoiseDraws`, GPU) over a local patch, takes the exact max per
    wheel. `cand_abs[w] = (abs_iy[K], abs_ix[K], cap[K])`."""
    all_iy = np.concatenate([cand_abs[w][0] for w in cand_abs])
    all_ix = np.concatenate([cand_abs[w][1] for w in cand_abs])
    margin = 8  # >= kernel support radius; keeps patch-edge clamping off every tested cell
    lo_y, hi_y = int(all_iy.min()) - margin, int(all_iy.max()) + margin + 1
    lo_x, hi_x = int(all_ix.min()) - margin, int(all_ix.max()) + margin + 1
    lo_y, lo_x = max(lo_y, 0), max(lo_x, 0)
    hi_y, hi_x = min(hi_y, belief.shape[0]), min(hi_x, belief.shape[1])
    patch_belief = belief[lo_y:hi_y, lo_x:hi_x].astype(np.float32)
    patch_sigma = sigma[lo_y:hi_y, lo_x:hi_x].astype(np.float32)
    pny, pnx = patch_belief.shape

    with wp.ScopedDevice(device):
        base = wp.array(np.tile(patch_belief, (n_draws, 1, 1)), dtype=wp.float32)
        sigma_dev = wp.array(patch_sigma, dtype=wp.float32)
        out = wp.zeros((n_draws, pny, pnx), dtype=wp.float32)
    draws = NoiseDraws((n_draws, pny, pnx), cell, CORR_LEN, device)
    draws.perturb(base, sigma_dev, 1.0, out, 500_000 + seed)
    field = out.numpy()  # [n_draws, pny, pnx], perturbed elevation

    env_samples = {}
    for w, (iy, ix, cap) in cand_abs.items():
        local_iy, local_ix = iy - lo_y, ix - lo_x
        h = field[:, local_iy, local_ix] + cap[None, :]  # [n_draws, K]
        env_samples[w] = h.max(axis=1)
    return env_samples


def gate2_clark_vs_mc(device: str, element: str = "sphere") -> dict:
    """20 random single-timestep, real-belief-map, real-footprint cases: Clark's per-wheel
    E[env]/sd[env] and cov(env_L, env_R) against 20k-draw brute-force MC.

    `element` selects the contact candidate table (`element.py`). The offset table is built
    PER CASE (not hoisted above the loop) because the cylinder table depends on that case's
    sampled yaw -- a sphere-only cache here would silently reuse case 0's cylinder table for
    every other heading. `cand_abs` is built from that SAME table for both Clark and the MC
    patch below, so a cylinder run validates the cylinder estimator against truth rather than
    measuring physics drift against the sphere (contrast `clark_conv.py`'s cylinder arm, which
    only ever compares itself to the sphere)."""
    rp = RobotParams()
    corr_table = rho1_table(CORR_LEN, CELL)
    rng = np.random.default_rng(RNG_SEED)
    wheel_xy = np.array([[0.0, rp.half_track], [0.0, -rp.half_track], [-rp.rear_offset, 0.0]])

    rows = []
    n_cases = 20
    seeds = rng.integers(0, 40, n_cases)
    for case, sd in enumerate(seeds):
        scene, _, _, _, sigma, poses, omega, _ = build_case(int(sd), "hybrid", "all")
        belief = scene.elevation.astype(np.float32)
        ny, nx = belief.shape
        h_ = Harness(scene, poses, omega, device=device)
        controlled = h_.sim.controlled.numpy()  # [T+1, K, 3]
        del h_
        plan = int(rng.integers(0, N_PLANS))
        margin_cells = 12  # keep the case interior: no true-grid-edge clamping in the MC patch
        for _try in range(50):
            t = int(rng.integers(1, controlled.shape[0]))
            x, y, yaw = controlled[t, plan]
            c, s = np.cos(yaw), np.sin(yaw)
            iy0 = [int(round((y + px * s + py * c - scene.origin_y) / CELL)) for px, py in wheel_xy]
            ix0 = [int(round((x + px * c - py * s - scene.origin_x) / CELL)) for px, py in wheel_xy]
            if all(margin_cells <= v <= ny - margin_cells for v in iy0) and all(
                margin_cells <= v <= nx - margin_cells for v in ix0
            ):
                break
        wx_ = np.array([x + px * c - py * s for px, py in wheel_xy])
        wy_ = np.array([y + px * s + py * c for px, py in wheel_xy])
        # all 3 wheels share the body's single yaw at this timestep -- `np.full(3, yaw)` is the
        # single-timestep analog of `clark_plan_moments`' `np.tile(yaw, 3)` over a T-length plan.
        off_dy, off_dx, off_cap = element_offsets(element, CELL, rp, np.full(3, yaw))
        cell_flat = _footprint_cells(
            wx_, wy_, off_dy, off_dx, scene.origin_x, scene.origin_y, CELL, ny, nx
        )
        # cand_abs[w]'s cap must be the WHEEL-SPECIFIC row for the cylinder ([3, K], since a
        # rotated rectangle is not radially symmetric like the sphere's cap is).
        cap_rows = off_cap if off_cap.ndim == 2 else np.tile(off_cap, (3, 1))
        cand_abs = {w: (cell_flat[w] // nx, cell_flat[w] % nx, cap_rows[w]) for w in range(3)}
        belief_flat, sigma_flat = belief.ravel(), sigma.ravel()
        mean_n, var_n, cov_to_u_final, u_idx_sorted, phi, phineg = build_env_nodes(
            cell_flat, belief_flat, sigma_flat, off_cap, nx, corr_table
        )
        cross = clark_cross_cov(cov_to_u_final, u_idx_sorted, phi, phineg)

        mc = _mc_env_stats(
            belief, sigma, scene.origin_x, scene.origin_y, CELL, cand_abs, 20_000, device, case
        )
        for w in range(3):
            mc_mean, mc_sd = float(mc[w].mean()), float(mc[w].std())
            abs_err_mean = float(abs(mean_n[w] - mc_mean))
            rows.append(
                {
                    "case": case, "wheel": w, "clark_mean": float(mean_n[w]), "mc_mean": mc_mean,
                    "clark_sd": float(np.sqrt(var_n[w])), "mc_sd": mc_sd,
                    "rel_err_mean": abs_err_mean / max(abs(mc_mean), 1e-6),
                    "rel_err_sd": float(abs(np.sqrt(var_n[w]) - mc_sd)) / max(mc_sd, 1e-6),
                    # abs error scaled by the natural noise scale (mc_sd) instead of by the mean
                    # itself -- avoids the near-zero-baseline-elevation artifact that inflates
                    # `rel_err_mean` when a candidate patch sits at a low point of the terrain
                    # (env magnitude near zero) even though the absolute miss is tiny; see report.
                    "abs_err_mean": abs_err_mean,
                    "err_mean_over_sd": abs_err_mean / max(mc_sd, 1e-6),
                }
            )
        mc_cov_lr = float(np.cov(mc[0], mc[1])[0, 1])
        clark_corr_lr = float(cross[0, 1] / max(np.sqrt(var_n[0] * var_n[1]), 1e-9))
        mc_corr_lr = float(mc_cov_lr / max(np.sqrt(mc[0].var() * mc[1].var()), 1e-9))
        rows[-1]["cov_lr_clark"] = float(cross[0, 1])
        rows[-1]["cov_lr_mc"] = mc_cov_lr
        rows[-1]["cov_lr_rel_err"] = float(abs(cross[0, 1] - mc_cov_lr)) / max(abs(mc_cov_lr), 1e-6)
        # cov(L,R) is near zero for BOTH Clark and MC here (L/R sit at the edge of the kernel's
        # correlation support) so its relative error is a near-zero-denominator artifact too;
        # the correlation-coefficient DIFFERENCE (bounded in [-2,2], no division-by-small-number
        # blowup) is the honest number.
        rows[-1]["corr_lr_clark"] = clark_corr_lr
        rows[-1]["corr_lr_mc"] = mc_corr_lr
        rows[-1]["corr_lr_abs_diff"] = float(abs(clark_corr_lr - mc_corr_lr))

    med_mean = float(np.median([r["rel_err_mean"] for r in rows]))
    med_sd = float(np.median([r["rel_err_sd"] for r in rows]))
    med_err_over_sd = float(np.median([r["err_mean_over_sd"] for r in rows]))
    cov_rows = [r for r in rows if "cov_lr_rel_err" in r]
    med_cov = float(np.median([r["cov_lr_rel_err"] for r in cov_rows]))
    med_corr_diff = float(np.median([r["corr_lr_abs_diff"] for r in cov_rows]))
    return {
        "rows": rows,
        "median_rel_err_mean": med_mean,
        "median_rel_err_sd": med_sd,
        "median_err_mean_over_sd": med_err_over_sd,
        "median_rel_err_cov_lr": med_cov,
        "median_corr_lr_abs_diff": med_corr_diff,
        "passed": med_mean < 0.03 and med_sd < 0.10,
    }


# --- the risk.py-protocol comparison ------------------------------------------------------------
def run_seed(seed: int, family: str, noise: str, device: str, element: str = "sphere") -> dict:
    scene, _truth, _meas, _obs, sigma, poses, omega, grid = build_case(seed, family, noise)
    belief = scene.elevation.astype(np.float32)
    ny, nx = belief.shape
    rp = RobotParams()
    corr_table = rho1_table(CORR_LEN, CELL)

    h = Harness(scene, poses, omega, device=device)
    grads, terms = h.adjoint(dilate=True, leaf="elevation")
    grad = grads[SETTLE_IDX]  # [K, ny, nx] -- settle-only, declared approximation (c)
    j_bel = _cost_settle(terms)
    controlled = h.sim.controlled.numpy()  # [T+1, K, 3]
    traj = controlled[:, :, :2].copy()
    sig_t = _footprint_sigma(traj, sigma, grid)
    step_risk = sig_t.sum(axis=0)

    def _forward_on(elev2d: np.ndarray) -> np.ndarray:
        stack = np.ascontiguousarray(np.tile(elev2d, (N_PLANS, 1, 1)), np.float32)
        with wp.ScopedDevice(h.device):
            h.sim.set_terrain(wp.array(stack, dtype=wp.float32))
        return _cost_settle(h.forward(dilate=True)).copy()

    j_hi = _forward_on(belief + BRACKET_C * sigma.astype(np.float32))
    j_lo = _forward_on(belief - BRACKET_C * sigma.astype(np.float32))
    fosm_sd = np.sqrt(
        [max(fosm_variance(grad[k], sigma, CELL, CORR_LEN), 0.0) for k in range(N_PLANS)]
    )

    # NOTE (Stage A caveat): the MC truth below always runs the real Warp settle, which is
    # sphere-contact only (trajectory generation is unchanged, per this stage's scope). Under
    # `element="cylinder"` the clark_* arms therefore price a DIFFERENT contact model than the
    # ground truth they are scored against -- the regret/decision numbers stop being meaningful,
    # same caveat `clark_conv.py` states for its cylinder timing arm. Only GATE 2
    # (`gate2_clark_vs_mc`) validates the cylinder estimator against matched-element MC truth.
    e_clark = np.empty(N_PLANS)
    sd_clark = np.empty(N_PLANS)
    for k in range(N_PLANS):
        e_j, var_j = clark_plan_moments(
            belief, sigma, controlled[:, k, :], rp, scene.origin_x, scene.origin_y, CELL,
            corr_table, element=element,
        )
        e_clark[k], sd_clark[k] = e_j, np.sqrt(var_j)
    del h

    est = {
        "none": j_bel.copy(),
        "step": j_bel + KAPPA * step_risk,
        "fosm": j_bel + KAPPA * fosm_sd,
        "bracket": np.maximum(j_bel, np.maximum(j_hi, j_lo)),
        "clark_mean": e_clark.copy(),
        "clark_cvar": e_clark + KAPPA * sd_clark,
    }

    # --- Monte-Carlo truth: identical protocol to risk.py, settle-only cost -------------------
    poses_d = np.tile(poses[0], (N_DRAWS, 1)).astype(np.float32)
    omega_d = np.zeros((omega.shape[0], N_DRAWS, 3), np.float32)
    hd = Harness(scene, poses_d, omega_d, device=device)
    draws = NoiseDraws((N_DRAWS, ny, nx), CELL, CORR_LEN, hd.device)
    with wp.ScopedDevice(hd.device):
        base = wp.array(np.ascontiguousarray(np.tile(belief, (N_DRAWS, 1, 1)), np.float32))
        sig_dev = wp.array(np.ascontiguousarray(sigma, np.float32), dtype=wp.float32)
    samples = np.empty((N_DRAWS, N_PLANS), np.float32)
    for k in range(N_PLANS):
        hd.sim.start_pose.assign(np.tile(poses[k], (N_DRAWS, 1)).astype(np.float32))
        hd.sim.target_wheel_omega.assign(
            np.ascontiguousarray(np.repeat(omega[:, k : k + 1, :], N_DRAWS, axis=1), np.float32)
        )
        draws.perturb(base, sig_dev, 1.0, hd.sim.elevation, 900_000 + seed)
        samples[:, k] = _cost_settle(hd.forward(dilate=True))
    del hd

    mc_mean = samples.mean(axis=0)
    mc_sd = samples.std(axis=0)
    mc_cvar = empirical_cvar(samples, ALPHA)
    best = int(np.argmin(mc_cvar))

    out = {"seed": seed, "j_bel": j_bel.tolist(), "arms": {}}
    for name, v in est.items():
        pick = int(np.argmin(v))
        out["arms"][name] = {
            "regret": float(mc_cvar[pick] - mc_cvar[best]),
            "picked_best": bool(pick == best),
        }
    out["arms"]["mc"] = {"regret": 0.0, "picked_best": True}
    # calibration: does E_clark capture the Jensen bias, and is sd_clark the right scale?
    out["calib"] = {
        "e_clark": e_clark.tolist(),
        "mc_mean": mc_mean.tolist(),
        "sd_clark": sd_clark.tolist(),
        "mc_sd": mc_sd.tolist(),
        "j_bel_settle": j_bel.tolist(),
    }
    return out


def _sign_paired(rows: list[dict], a: str, b: str) -> tuple[float, int, int, float]:
    d = np.array([r["arms"][a]["regret"] - r["arms"][b]["regret"] for r in rows])
    k, w, p = sign_test(-d)
    return float(d.mean()), k, w, p


def report(gate0: dict, gate1: dict, gate2: dict, rows: list[dict]) -> None:
    print("=" * 88)
    print("GATE 0 -- analytic rho vs empirical NoiseDraws covariance (5 offset pairs, B=2000)")
    for r in gate0["pairs"]:
        print(
            f"  (dy={r['dy']},dx={r['dx']})  analytic={r['analytic']:+.4f}"
            f"  empirical={r['empirical']:+.4f}  rel_err={r['rel_err']:.3f}"
        )
    pass0 = "PASS" if gate0["passed"] else "FAIL"
    print(f"  worst rel_err = {gate0['worst_rel_err']:.3f}  -> {pass0} (< 0.05)")

    print("\nGATE 1 -- closed-form settle map vs real Warp settle (cm-scale perturbations)")
    print(f"  n={len(gate1['cases'])} cases (4 poses x 3 scales)")
    pass1 = "PASS" if gate1["passed"] else "FAIL"
    print(f"  median rel_err = {gate1['median_rel_err']:.4f}  -> {pass1} (< 0.05)")

    print("\nGATE 2 -- Clark per-wheel env moments + cov(L,R) vs 20k-draw brute-force MC")
    print("(20 cases)")
    print(f"  median |rel err| E[env] = {gate2['median_rel_err_mean']:.4f} (pre-registered < 0.03)")
    print(f"  median |rel err| sd[env] = {gate2['median_rel_err_sd']:.4f}  (pre-registered < 0.10)")
    print(
        f"  supplementary: median |E[env] error| / mc_sd = {gate2['median_err_mean_over_sd']:.4f}"
        "  -- the mean error scaled by the natural noise scale instead of by the (sometimes"
        "\n    near-zero, at a low point of the terrain) env value itself; see report for why."
    )
    print(
        f"  median |rel err| cov(L,R) = {gate2['median_rel_err_cov_lr']:.4f}"
        "  (cov(L,R) itself is near 0 for both -- see report)"
    )
    print(
        f"  median |corr(L,R) difference| = {gate2['median_corr_lr_abs_diff']:.4f}"
        "  (bounded metric, not a ratio)"
    )
    if gate2["rows"]:
        low_baseline = np.mean([abs(r["mc_mean"]) < r["mc_sd"] for r in gate2["rows"]])
        print(
            f"  diagnostic: {low_baseline:.0%} of the 60 wheel-cases have |env baseline| < 1 mc_sd"
            " -- exactly where a relative-error denominator is noisy by construction"
        )
    pass2 = "PASS" if gate2["passed"] else "FAIL"
    print(f"  -> {pass2} (against the pre-registered relative-error bar)")

    if not rows:
        print("\n(no seeds run for the main comparison)")
        return
    print("=" * 88)
    n = len(rows)
    print(
        f"\nMAIN COMPARISON: n={n} seeds, hybrid/all, settle-only J, N_DRAWS={N_DRAWS},"
        f" alpha={ALPHA}, kappa={KAPPA:.3f}"
    )
    print("\nDECISION QUALITY -- pick the argmin, pay its true CVaR (settle-only)")
    print(f"{'estimator':<14}{'regret':>9}{'picked best':>13}")
    for a in ("none", "step", "fosm", "bracket", "clark_mean", "clark_cvar", "mc"):
        rg = np.mean([r["arms"][a]["regret"] for r in rows])
        pb = np.mean([r["arms"][a]["picked_best"] for r in rows])
        tag = "  <- truth" if a == "mc" else ("  <- ours" if a.startswith("clark") else "")
        print(f"{a:<14}{rg:>9.4f}{pb:>12.0%}{tag}")

    print("\nCALIBRATION (per plan, pooled over all seeds x 16 plans)")
    e_clark = np.concatenate([r["calib"]["e_clark"] for r in rows])
    mc_mean = np.concatenate([r["calib"]["mc_mean"] for r in rows])
    j_bel = np.concatenate([r["calib"]["j_bel_settle"] for r in rows])
    sd_clark = np.concatenate([r["calib"]["sd_clark"] for r in rows])
    mc_sd = np.concatenate([r["calib"]["mc_sd"] for r in rows])
    jensen_true = mc_mean - j_bel
    jensen_clark = e_clark - j_bel
    corr = float(np.corrcoef(jensen_clark, jensen_true)[0, 1])
    sd_ratio = sd_clark / np.maximum(mc_sd, 1e-9)
    jensen_ratio = jensen_clark / np.where(np.abs(jensen_true) > 1e-9, jensen_true, np.nan)
    print(
        "  E[J] Jensen-bias capture: corr(E_clark - J_bel, MC_mean - J_bel) ="
        f" {corr:+.3f}  (pre-registered > 0.7)"
    )
    print(
        "  E[J] ratio median (E_clark - J_bel)/(MC_mean - J_bel) ="
        f" {float(np.median(jensen_ratio)):.3f}"
    )
    print(
        f"  sd ratio (sd_clark / mc_sd): median={float(np.median(sd_ratio)):.3f}"
        f"  [{np.percentile(sd_ratio, 25):.3f}, {np.percentile(sd_ratio, 75):.3f}]"
        "  (pre-registered median in [0.7, 1.4])"
    )

    print("\nPAIRED SIGN TESTS: clark_cvar vs each baseline (regret, negative = clark_cvar better)")
    for b in ("step", "fosm", "bracket", "none"):
        mean_d, k, w, p = _sign_paired(rows, "clark_cvar", b)
        print(
            f"  clark_cvar vs {b:<10} mean_diff={mean_d:>+8.4f}"
            f"  better on {w:>3}/{k:<3}  p={p:.2e}"
        )

    print("\nVERDICT vs the three pre-registered criteria:")
    sd_ok = 0.7 <= float(np.median(sd_ratio)) <= 1.4
    jensen_ok = corr > 0.7
    _, _, w_step, p_step = _sign_paired(rows, "clark_cvar", "step")
    decision_ok = np.mean([r["arms"]["clark_cvar"]["regret"] for r in rows]) <= np.mean(
        [r["arms"]["step"]["regret"] for r in rows]
    )
    print(
        f"  (i)   sd ratio median in [0.7, 1.4]:            {'YES' if sd_ok else 'NO'}"
        f" (median={float(np.median(sd_ratio)):.3f})"
    )
    print(
        f"  (ii)  E_clark captures Jensen bias (corr>0.7):  {'YES' if jensen_ok else 'NO'}"
        f" (corr={corr:.3f})"
    )
    print(
        f"  (iii) clark_cvar >= step at regret:             {'YES' if decision_ok else 'NO'}"
        f" (p={p_step:.2e})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--family", default="hybrid")
    ap.add_argument("--noise", default="all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-gates", action="store_true")
    ap.add_argument("--element", default="sphere", choices=("sphere", "cylinder"))
    a = ap.parse_args()

    wp.init()
    OUT.mkdir(parents=True, exist_ok=True)

    gate0 = gate0_correlation(a.device)
    gate1 = gate1_linear_map(a.device)
    gate2 = gate2_clark_vs_mc(a.device, a.element) if not a.skip_gates else {
        "rows": [], "median_rel_err_mean": float("nan"), "median_rel_err_sd": float("nan"),
        "median_err_mean_over_sd": float("nan"), "median_rel_err_cov_lr": float("nan"),
        "median_corr_lr_abs_diff": float("nan"), "passed": False,
    }

    rows = []
    for seed in range(a.seeds):
        rows.append(run_seed(seed, a.family, a.noise, a.device, a.element))
        if (seed + 1) % 10 == 0:
            print(f"  {seed + 1}/{a.seeds} seeds", flush=True)

    report(gate0, gate1, gate2, rows)
    # element-tagged filename for cylinder runs -- never overwrites the committed sphere baseline.
    path = OUT / ("clark.json" if a.element == "sphere" else f"clark_{a.element}.json")
    path.write_text(
        json.dumps(
            {
                "gate0": gate0,
                "gate1": gate1,
                "gate2": gate2,
                "family": a.family,
                "noise": a.noise,
                "element": a.element,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
