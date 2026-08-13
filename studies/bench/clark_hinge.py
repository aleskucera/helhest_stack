"""Fixes `clark_full.py`'s approximation (e) -- the frozen body pose in the belly-clearance
hinge -- by making the chassis height a Clark node correlated with the ground it is measured
against. This is the open problem that scoped Clark to the settle-only cost.

    .venv/bin/python -m studies.bench.clark_hinge            # gate only
    .venv/bin/python -m studies.bench.clark_hinge --stage2   # + full-cost head-to-head

WHAT WAS WRONG (clark_full.py, approximation (e), measured by its Gate 1b):
`clear_soft = Sum_i max(clear_margin - c_i, 0)` with `c_i = w_z_i - raw_ground_i`. Gate 1b froze
the ENTIRE body pose at its belief rollout and let only `raw_ground_i` be random. That sees
"the ground drops under the belly -> clearance shrinks" but NOT the correlated "the wheels see
the same rough patch -> the chassis rides up -> clearance partially recovers". Two consequences,
both pushing E[clear_soft] the same way:

  1. MEAN: the wheel envelope's max has its own Jensen uplift (E[max] > max of the belief
     values), which raises z. Frozen at the belief pose, that uplift is missing, so
     `mu_X = clear_margin - w_z + ground` is too high.
  2. VARIANCE: Var(X) was Var(ground) alone instead of
     Var(ground) + Var(w_z) - 2 Cov(ground, w_z). The two are POSITIVELY correlated (same
     terrain), so the true variance is smaller. E[max(X,0)] grows with sd at these near-boundary
     nodes (measured 80% of them sit at Phi ~ 0.3-0.7), so an inflated sd inflates E too.

Measured: median E-overshoot 2.20x over a full rollout (clark_full.json gate1b, 12 cases,
e_ratio 2.01-2.51 -- strikingly consistent, which is itself the signature of a structural
missing term rather than noise), while the sd ratio was already fine at 1.02.

THE FIX -- w_z AS A LINEAR FUNCTIONAL OF THE SAME CLARK NODES THE SETTLE USES.
`clark.py::settle_map` already gives the exact small-angle tripod map
(z, pitch, roll) = M @ (env_L, env_R, env_rear) (+ wheel_radius on z), gate-verified to 3.4%
median. A belly point at body-frame p_i = (px, py, pz) has world height
w_z_i = z + (R p_i)_z, and to the SAME first order in (pitch, roll) that M itself is derived
under, (R p_i)_z = -pitch*px + roll*py + pz -- yaw-free, because yaw only rotates x and y.
Composing:

    w_z_{t,i} = a_i . env_t + wheel_radius + pz_i,    a_i = M[z] - px_i * M[pitch] + py_i * M[roll]

so a_i is a CONSTANT 3-vector per belly point (time- and yaw-independent), and the hinge
argument becomes a linear functional of quantities Clark already tracks:

    X_{t,i} = clear_margin - w_z_{t,i} + ground_{t,i} = const + g_{t,i} . U - a_i . N_t

with U the shared universe of raw cells (bilinear stencil weights g) and N_t the three
envelope max-nodes at t. Hence, all in closed form and all from existing machinery:

    mu_X   = (frozen-pose mu_X) - a_i . delta_t,  delta_t = E[max] - max(belief)  <- the Jensen
                                                                                     uplift (1)
    Var(X) = g C_uu g - 2 g C_uN a + a C_NN a                                     <- the
                                                                                     cancellation (2)

`C_uN` is `cov_to_u_final` from `clark_build` (the node's fold-propagated covariance to every
universe cell) and `C_NN` is `clark_cross_cov`'s node-node matrix -- both are already computed
for Var[settle] and are simply reused, so the fix costs no new fold and no new MC.

ANCHORING CHOICE (deliberate): the mean is anchored at the harness's REAL belief-rollout pose
and only the DEVIATION is taken from the linear model (`delta_t`, not `a_i . E[env]`). The
linear tripod map is accurate for cm-scale deviations (that is what Gate 1 measured) but has an
absolute offset against the true nonlinear settle; anchoring keeps the offset out of E[] and
uses the linearization only where it was validated.

WHAT IS STILL APPROXIMATED (declared, not hidden):
  - belly-point world (x, y) stays frozen (pitch/roll perturb it too). Same class as clark.py's
    approximation (a), which froze XY for the settle term; only the Z coupling is repaired here,
    because Z is the one that enters the clearance directly.
  - clark_full.py's (f) survives unchanged: hinge-hinge cross covariance uses the Phi-weighted
    linearization, restricted to same-timestep pairs. The Phi-weighted vector now carries a
    node part as well as a universe part -- Y_t = v_t . U - b_t . N_t -- so its variance picks
    up the same three-term structure as (2).
  - (g), Cov(settle, clear_soft), likewise gains its node term: settle is itself c . N, so
    Cov(settle, Y_t) = c C_NU v_t - c C_NN b_t. The first half is what clark_full.py had; the
    second half is new and is EXACT within the moment-matched model (no Phi linearization on
    settle's side -- see clark_full.py (g) for why one-sided is exact).

PRE-REGISTERED SUCCESS CRITERIA -- written and committed BEFORE the first run of this module.
Gate H is a like-for-like rematch of clark_full.py's Gate 1b: the SAME 12 (seed, plan) cases
(same RNG stream, RNG_SEED + 1), the same true end-to-end MC (full re-settled `Harness.forward`
on N_DRAWS perturbed maps), scoring the frozen and coupled models side by side against it.

    (i)   median E-ratio (coupled / MC) in [0.75, 1.30]        [frozen was 2.20]
    (ii)  median |E_coupled - E_MC| / sd_MC <= 1.5             [frozen was 5.47]
    (iii) median sd-ratio (coupled / MC) in [0.7, 1.4]         [frozen was 1.02 -- the fix must
                                                                not buy E with variance]
    (iv)  corr(E_coupled, E_MC) across cases >= 0.93           [frozen was 0.931 -- the fix must
                                                                not trade bias for ranking]
    PASS = all four. Absolute and correlation metrics are used rather than raw relative error
    per the project's near-zero-baseline rule; (ii) is stated in units of the MC spread, which
    is the scale that matters for ranking.

    STAGE 2, run only if Gate H passes -- the full-cost head-to-head, criterion as specified in
    the paper work order: on the FULL cost (settle + clear_soft), n=100 seeds, hybrid/all,
    clark_cvar beats BOTH step AND bracket on mean regret at p < 0.05 (paired sign test, the
    same protocol as risk.py/clark.py). Stage 2 reuses `clark_full.run_seed_full` unmodified
    (see `_coupled_moments` injection below) so the comparison protocol cannot drift from the
    published settle-only one.

If Gate H fails, the paper scopes Clark to the settle cost and reports this module as the
measured reason -- the same honesty rule clark_full.py already followed.

===============================================================================================
RUN LOG AND VERDICT (2026-08-07, in the order it happened -- nothing here is retrofitted)
===============================================================================================

RUN 1, design cases (rng_offset=1, the same 12 as gate1b), cross_scope="same_t" as inherited
from clark_full.py -- `clark_hinge.json`:
    (i)  E-ratio 0.785 [PASS]   ... the 2.20x overshoot is gone
    (ii) |err|/mc_sd 0.900 [PASS] ... down from 5.47
    (iii) sd-ratio 0.426 [FAIL] ... now UNDER-dispersed by ~2.3x
    (iv) corr 0.916 [FAIL]
    -> GATE FAILED.

DIAGNOSIS AND THE ONE REVISION. clark_full.py's (f) restricted the hinge-hinge cross covariance
to same-timestep pairs because the unrestricted version inflated variance ~9x. That restriction
was calibrated against the FROZEN model, whose per-node variances were themselves inflated (no
pose cancellation): the restriction was compensating for approximation (e), not for a defect in
(f). With (e) repaired, the compensation becomes a deficit -- a rollout's clear_soft is a sum of
hundreds of positively-correlated hinges and dropping every cross-timestep pair removes most of
Var(sum). Revision: `cross_scope` becomes explicit, and "all" (one global Phi-weighted quadratic
form over the whole rollout, which is also CHEAPER -- one quadratic form instead of T) is the
default. This is a structural re-decision of a declared approximation, not a tuned constant.

RUN 2, SAME design cases, cross_scope="all" -- `clark_hinge_design_all.json`:
    sd-ratio 0.426 -> 0.791 [PASS]; E-ratio and corr unchanged (the cross term enters Var of the
    SUM, not the per-node moments that set E). (iv) still 0.916 [FAIL].

RUN 3, VIRGIN cases (rng_offset=7, 12 fresh (seed, plan) pairs never looked at), cross_scope
="all", criteria untouched -- `clark_hinge_virgin.json`. THE REPORTABLE RESULT:
    (i)   E-ratio        0.775  [PASS]   (frozen 2.225)
    (ii)  |err|/mc_sd    1.077  [PASS]   (frozen 5.913)
    (iii) sd-ratio       0.825  [PASS]
    (iv)  corr           0.915  [FAIL vs the 0.93 bar]
    -> GATE H FAILS ON CRITERION (iv), LITERALLY AND AS PRE-REGISTERED.

WHAT (iv)'s FAILURE DOES AND DOES NOT MEAN -- stated plainly because the bar was mine and it was
badly specified. (iv) existed to test "the fix must not trade bias for ranking", operationalized
as beating the frozen model's design-set correlation of 0.931. On the VIRGIN cases the frozen
model's own correlation is 0.905, i.e. BELOW the coupled model's 0.915: the comparator the bar
was pegged to does not replicate, because a Pearson correlation over n=12 has a 95% CI of about
[0.72, 0.98] -- the bar was set inside its own noise. The intent behind (iv) is therefore MET
(no ranking degradation; the coupled model is if anything better on virgin data, and its E-ratio
scatter is 3x tighter: sd 0.061 vs 0.166). The letter of (iv) is not. Both statements belong in
the paper; the bar is NOT retroactively relaxed.

RESIDUAL, DECLARED: a stable ~22% E undershoot with a ~0.83 sd-ratio. One mechanism explains
both, and it is the next declared approximation in line rather than a defect in this fix: the
rollout TRAJECTORY (and with it each belly point's world xy) is frozen at the belief path, so
the true MC's path-level spread -- different terrain visited under different draws -- is absent
from the model's per-node variance, and E[max(X,0)] is increasing in that variance. Repairing it
means making the trajectory itself a random variable, which is a different problem from
propagating a map through a contact max.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import warp as wp
from helhest.engine import RobotParams

from . import clark_full
from . import matched_truth as mt
from ..adjoint.harness import Harness
from ..adjoint.sigma import NoiseDraws
from .clark import _footprint_cells
from .clark import _norm_cdf  # noqa: F401  (re-exported for symmetry with clark_full's helpers)
from .clark import _settle_moments_from_nodes
from .clark import _settle_weights
from .clark import clark_build
from .clark import clark_cross_cov
from .clark import rho1_table
from .clark import rho_lookup
from .clark import RNG_SEED
from .clark import settle_map
from .clark_full import _bilinear_stencil
from .clark_full import _hinge_inputs
from .clark_full import _hinge_moments
from .clark_full import CLEAR_IDX
from .clark_full import CLEAR_MARGIN
from .element import broadcast_cap
from .element import element_offsets
from .ranking import build_case
from .ranking import CELL
from .ranking import N_PLANS
from .ranking import OUT
from .risk import CORR_LEN
from .risk import N_DRAWS

# Which hinge-hinge pairs approximation (f)'s Phi-weighted cross covariance covers. clark_full.py
# fixed this at same-timestep-only, a choice calibrated against the FROZEN-pose model; see the
# revision log in the module docstring for why it has to be re-decided once (e) is fixed.
CROSS_SCOPE = "all"


def belly_pose_weights(rp: RobotParams, chassis_pts: np.ndarray) -> np.ndarray:
    """d(w_z_i) / d(env_L, env_R, env_rear) for every belly point: [n_p, 3].

    Row i is `M[z] - px_i * M[pitch] + py_i * M[roll]`, the composition of `settle_map`'s tripod
    map with the small-angle body-to-world lift of a body-frame point (see module docstring).
    Constant in time and yaw.
    """
    m = settle_map(rp)  # rows (z, pitch, roll), cols (L, R, rear)
    px, py = chassis_pts[:, 0].astype(np.float64), chassis_pts[:, 1].astype(np.float64)
    return m[0][None, :] - px[:, None] * m[1][None, :] + py[:, None] * m[2][None, :]


def coupled_cost_plan_moments(
    belief: np.ndarray,
    sigma: np.ndarray,
    controlled: np.ndarray,
    derived: np.ndarray,
    rp: RobotParams,
    chassis_pts: np.ndarray,
    clear_margin: float,
    x0: float,
    y0: float,
    cell: float,
    corr_table: np.ndarray,
    cross_scope: str = CROSS_SCOPE,
    element: str = "sphere",
) -> dict:
    """E[J], Var[J] for J = settle + clear_soft with the chassis height treated as a Clark node
    correlated with the ground under the belly. Signature is deliberately identical to
    `clark_full.full_cost_plan_moments` (Stage 2 injects this in its place, `element` included so
    the injection keeps working under `run_seed_full`'s `element=` keyword).

    Returns the same keys plus `e_clear_frozen` / `var_clear_frozen`, the same quantities under
    the OLD frozen-pose model, computed from the same intermediates so the A/B is exact.
    """
    ny, nx = belief.shape
    belief_flat, sigma_flat = belief.ravel(), sigma.ravel()

    # --- envelope (settle) candidates: identical construction to clark_full ---------------------
    wheel_xy = np.array([[0.0, rp.half_track], [0.0, -rp.half_track], [-rp.rear_offset, 0.0]])
    t_idx = np.arange(1, controlled.shape[0])
    n_t = len(t_idx)
    x, y, yaw = controlled[t_idx, 0], controlled[t_idx, 1], controlled[t_idx, 2]
    c, s = np.cos(yaw), np.sin(yaw)
    wx_env = np.stack([x + wheel_xy[k, 0] * c - wheel_xy[k, 1] * s for k in range(3)])  # [3, T]
    wy_env = np.stack([y + wheel_xy[k, 0] * s + wheel_xy[k, 1] * c for k in range(3)])
    off_dy, off_dx, off_cap = element_offsets(element, cell, rp, np.tile(yaw, 3))
    env_cells = _footprint_cells(
        wx_env.ravel(), wy_env.ravel(), off_dy, off_dx, x0, y0, cell, ny, nx
    )  # [3T, Kenv], node order wheel-major then time: node(w, t) = w * n_t + t

    # --- belly candidates + the ONE shared universe (what makes the coupling expressible) -------
    wxb, wyb, mu0 = _hinge_inputs(controlled, derived, chassis_pts, clear_margin)  # [T, Np]
    hinge_cells, hinge_w = _bilinear_stencil(wxb.ravel(), wyb.ravel(), x0, y0, cell, ny, nx)
    u = np.unique(np.concatenate([env_cells.ravel(), hinge_cells.ravel()]))
    u_iy, u_ix = u // nx, u % nx
    cov_u = (
        sigma_flat[u][:, None]
        * sigma_flat[u][None, :]
        * rho_lookup(corr_table, u_iy[:, None] - u_iy[None, :], u_ix[:, None] - u_ix[None, :])
    )
    env_u_idx = np.searchsorted(u, env_cells)
    hinge_u_idx = np.searchsorted(u, hinge_cells)

    # --- settle: Clark's max-fold, unchanged ----------------------------------------------------
    means_env = belief_flat[env_cells] + broadcast_cap(off_cap)
    mean_env, var_env, cov_to_u_final_env, order_env, phi_env, phineg_env = clark_build(
        means_env,
        sigma_flat[env_cells],
        cov_u[env_u_idx[:, :, None], env_u_idx[:, None, :]],
        cov_u[env_u_idx],
    )
    u_idx_sorted_env = np.take_along_axis(env_u_idx, order_env, axis=1)
    cross_env = clark_cross_cov(cov_to_u_final_env, u_idx_sorted_env, phi_env, phineg_env)
    c_w = np.repeat(_settle_weights(rp), n_t)
    e_settle, var_settle = _settle_moments_from_nodes(
        mean_env, var_env, cov_to_u_final_env, u_idx_sorted_env, phi_env, phineg_env, rp, n_t
    )

    # --- the coupling terms ---------------------------------------------------------------------
    a = belly_pose_weights(rp, chassis_pts)  # [Np, 3], d(w_z) / d(env_L, env_R, env_rear)
    n_p = chassis_pts.shape[0]
    # delta = the envelope max's Jensen uplift over the belief map's own max, per (wheel, t).
    delta = (mean_env - means_env.max(axis=1)).reshape(3, n_t)
    nodes_t = np.arange(3)[:, None] * n_t + np.arange(n_t)[None, :]  # [3, T] node indices

    hinge_w_r = hinge_w.reshape(n_t, n_p, 4)
    hinge_u_idx_r = hinge_u_idx.reshape(n_t, n_p, 4)
    mu_frozen = (mu0.ravel() + (hinge_w * belief_flat[hinge_cells]).sum(axis=1)).reshape(n_t, n_p)
    var_ground = np.einsum(
        "nc,ncd,nd->n", hinge_w, cov_u[hinge_u_idx[:, :, None], hinge_u_idx[:, None, :]], hinge_w
    ).reshape(n_t, n_p)

    mu_x = np.empty((n_t, n_p))
    var_x = np.empty((n_t, n_p))
    cov_gn = np.empty((n_t, n_p, 3))  # Cov(ground_i, env node w at t)
    for t in range(n_t):
        # Cov(ground_{t,i}, N_{w,t}): the node's fold-propagated covariance to the universe,
        # contracted with the belly point's own bilinear stencil weights.
        c_un_t = cov_to_u_final_env[nodes_t[:, t]]  # [3, |U|]
        cov_gn[t] = np.einsum("wpc,pc->pw", c_un_t[:, hinge_u_idx_r[t]], hinge_w_r[t])
        c_nn_t = cross_env[np.ix_(nodes_t[:, t], nodes_t[:, t])]  # [3, 3]
        mu_x[t] = mu_frozen[t] - a @ delta[:, t]
        var_x[t] = (
            var_ground[t]
            - 2.0 * np.einsum("pw,pw->p", a, cov_gn[t])
            + np.einsum("pw,wv,pv->p", a, c_nn_t, a)
        )
    var_x = np.maximum(var_x, 0.0)

    e_hinge, var_hinge, phi_pos = _hinge_moments(mu_x.ravel(), var_x.ravel())
    phi_r = phi_pos.reshape(n_t, n_p)
    var_hinge_r = var_hinge.reshape(n_t, n_p)

    # (f) with the node part carried through: Y_t = v_t . U - b_t . N_t.
    v_sum = np.zeros(u.shape[0])
    b_all = np.zeros(3 * n_t)
    per_t_var = 0.0
    for t in range(n_t):
        v_t = np.zeros(u.shape[0])
        np.add.at(v_t, hinge_u_idx_r[t].ravel(), (phi_r[t][:, None] * hinge_w_r[t]).ravel())
        b_t = phi_r[t] @ a  # [3]
        c_un_t = cov_to_u_final_env[nodes_t[:, t]]  # [3, |U|]
        c_nn_t = cross_env[np.ix_(nodes_t[:, t], nodes_t[:, t])]
        per_t_var += (
            float(v_t @ cov_u @ v_t) - 2.0 * float(b_t @ (c_un_t @ v_t)) + float(b_t @ c_nn_t @ b_t)
        )
        v_sum += v_t
        b_all[nodes_t[:, t]] = b_t
    if cross_scope == "same_t":
        var_y_total = per_t_var
    else:  # "all": one global quadratic form -- Y = v_sum . U - b_all . N, all pairs included
        var_y_total = (
            float(v_sum @ cov_u @ v_sum)
            - 2.0 * float(b_all @ (cov_to_u_final_env @ v_sum))
            + float(b_all @ cross_env @ b_all)
        )
    cross_hinge = var_y_total - float((phi_r**2 * var_x).sum())

    e_clear = float(e_hinge.sum())
    var_clear = max(float(var_hinge_r.sum() + cross_hinge), 0.0)

    # (g) Cov(settle, clear_soft) = c . C_NU v_sum - c . C_NN b_all (second term new here).
    cov_settle_clear = float(c_w @ cov_to_u_final_env @ v_sum) - float(c_w @ cross_env @ b_all)

    # --- the frozen-pose model, from the same intermediates, for the exact A/B -------------------
    e_hinge_f, var_hinge_f, phi_f = _hinge_moments(mu_frozen.ravel(), var_ground.ravel())
    phi_f_r = phi_f.reshape(n_t, n_p)
    var_ground_f = var_ground
    cross_f = 0.0
    for t in range(n_t):
        v_t = np.zeros(u.shape[0])
        np.add.at(v_t, hinge_u_idx_r[t].ravel(), (phi_f_r[t][:, None] * hinge_w_r[t]).ravel())
        cross_f += float(v_t @ cov_u @ v_t) - float((phi_f_r[t] ** 2 * var_ground_f[t]).sum())

    e_j = e_settle + e_clear
    var_j = max(var_settle + var_clear + 2.0 * cov_settle_clear, 0.0)
    return {
        "e_j": e_j,
        "var_j": var_j,
        "e_settle": e_settle,
        "var_settle": var_settle,
        "e_clear": e_clear,
        "var_clear": var_clear,
        "cov_settle_clear": cov_settle_clear,
        "e_clear_frozen": float(e_hinge_f.sum()),
        "var_clear_frozen": max(float(var_hinge_f.sum() + cross_f), 0.0),
    }


# --- GATE H: the like-for-like rematch of clark_full.py's Gate 1b -------------------------------
def gateH_trajectory_vs_mc(
    device: str,
    n_cases: int = 12,
    cross_scope: str = CROSS_SCOPE,
    rng_offset: int = 1,
    element: str = "sphere",
) -> dict:
    """The SAME (seed, plan) cases as `clark_full.gate1b_trajectory_vs_mc` (same RNG stream), the
    same true end-to-end MC, scoring frozen and coupled side by side. Criteria pre-registered in
    the module docstring.

    `element` (default sphere) selects the settle contact table. Under `element="cylinder"` the
    trajectory AND the true MC below both settle through the real cylinder envelope too (Stage B,
    `matched_truth.py`)."""
    rp = RobotParams()
    chassis_pts = rp._chassis_pts()
    corr_table = rho1_table(CORR_LEN, CELL)
    rng = np.random.default_rng(RNG_SEED + rng_offset)  # offset 1 == gate1b's own cases
    seeds = rng.integers(0, 40, n_cases)
    plans = rng.integers(0, N_PLANS, n_cases)
    rows = []
    for case, (sd, plan) in enumerate(zip(seeds, plans)):
        scene, _, _, _, sigma, poses, omega, _ = build_case(int(sd), "hybrid", "all")
        belief = scene.elevation.astype(np.float32)
        ny, nx = belief.shape
        # Matched-element trajectory (see clark.py's run_seed for the same fix): `h.sim` is
        # DifferentiableSimulator, sphere-only regardless of `element` (matched_truth.py).
        if element == "cylinder":
            controlled, derived = mt.cylinder_controlled_trajectory(
                scene, poses, omega, device=device
            )
        else:
            h = Harness(scene, poses, omega, device=device)
            h.forward(dilate=True)
            controlled = h.sim.controlled.numpy()
            derived = h.sim.derived.numpy()
            del h

        mo = coupled_cost_plan_moments(
            belief,
            sigma,
            controlled[:, plan, :],
            derived[:, plan, :],
            rp,
            chassis_pts,
            CLEAR_MARGIN,
            scene.origin_x,
            scene.origin_y,
            CELL,
            corr_table,
            cross_scope,
            element=element,
        )

        # true end-to-end MC, byte-identical protocol to gate1b (same seed offset 800_000 + case).
        # Matched-element: cylinder truth settles through ForwardSimulator + the real cylinder
        # envelope (matched_truth.py), not the sphere-locked DifferentiableSimulator below.
        if element == "cylinder":
            terms_mc = mt.cylinder_mc_truth_terms(
                scene,
                belief,
                sigma,
                poses[plan : plan + 1],
                omega[:, plan : plan + 1, :],
                device=device,
                seed=800_000 + case,
                n_draws=N_DRAWS,
                corr_len=CORR_LEN,
                cell=CELL,
            )
            clear_mc = terms_mc[CLEAR_IDX, :, 0]  # [N_DRAWS], the single plan
        else:
            poses_d = np.tile(poses[plan], (N_DRAWS, 1)).astype(np.float32)
            omega_d = np.tile(omega[:, plan : plan + 1, :], (1, N_DRAWS, 1)).astype(np.float32)
            hd = Harness(scene, poses_d, omega_d, device=device)
            draws = NoiseDraws((N_DRAWS, ny, nx), CELL, CORR_LEN, hd.device)
            with wp.ScopedDevice(hd.device):
                base = wp.array(np.ascontiguousarray(np.tile(belief, (N_DRAWS, 1, 1)), np.float32))
                sig_dev = wp.array(np.ascontiguousarray(sigma, np.float32), dtype=wp.float32)
            draws.perturb(base, sig_dev, 1.0, hd.sim.elevation, 800_000 + case)
            clear_mc = hd.forward(dilate=True)[CLEAR_IDX]
            del hd

        mc_mean, mc_sd = float(clear_mc.mean()), float(clear_mc.std())
        sd_coupled = float(np.sqrt(mo["var_clear"]))
        rows.append(
            {
                "case": case,
                "seed": int(sd),
                "plan": int(plan),
                "mc_mean": mc_mean,
                "mc_sd": mc_sd,
                "coupled_mean": mo["e_clear"],
                "coupled_sd": sd_coupled,
                "frozen_mean": mo["e_clear_frozen"],
                "frozen_sd": float(np.sqrt(mo["var_clear_frozen"])),
                "e_ratio": mo["e_clear"] / max(mc_mean, 1e-6),
                "sd_ratio": sd_coupled / max(mc_sd, 1e-6),
                "frozen_e_ratio": mo["e_clear_frozen"] / max(mc_mean, 1e-6),
                "err_over_mcsd": abs(mo["e_clear"] - mc_mean) / max(mc_sd, 1e-6),
                "frozen_err_over_mcsd": abs(mo["e_clear_frozen"] - mc_mean) / max(mc_sd, 1e-6),
            }
        )

    med_e = float(np.median([r["e_ratio"] for r in rows]))
    med_sd = float(np.median([r["sd_ratio"] for r in rows]))
    med_err = float(np.median([r["err_over_mcsd"] for r in rows]))
    corr = float(np.corrcoef([r["coupled_mean"] for r in rows], [r["mc_mean"] for r in rows])[0, 1])
    crit = {
        "i_median_e_ratio_in_0.75_1.30": bool(0.75 <= med_e <= 1.30),
        "ii_median_err_over_mcsd_le_1.5": bool(med_err <= 1.5),
        "iii_median_sd_ratio_in_0.7_1.4": bool(0.7 <= med_sd <= 1.4),
        "iv_corr_ge_0.93": bool(corr >= 0.93),
    }
    return {
        "cross_scope": cross_scope,
        "rng_offset": rng_offset,
        "rows": rows,
        "median_e_ratio": med_e,
        "median_sd_ratio": med_sd,
        "median_err_over_mcsd": med_err,
        "corr_mean": corr,
        "frozen_median_e_ratio": float(np.median([r["frozen_e_ratio"] for r in rows])),
        "frozen_median_err_over_mcsd": float(np.median([r["frozen_err_over_mcsd"] for r in rows])),
        "criteria": crit,
        "passed": all(crit.values()),
    }


# --- STAGE 2: the full-cost head-to-head, protocol reused verbatim from clark_full -------------
def run_stage2(
    device: str,
    n_seeds: int,
    family: str,
    noise: str,
    seed_offset: int = 0,
    element: str = "sphere",
) -> dict:
    """`clark_full.run_seed_full` with the coupled moments swapped in for the frozen ones.

    The swap is a module-attribute injection rather than a copy of the 90-line comparison
    protocol: duplicating that protocol is how two arms silently drift apart, and the whole
    point of this stage is that the ONLY difference from the published settle-only comparison is
    which plan-moment function is called.
    """
    original = clark_full.full_cost_plan_moments
    clark_full.full_cost_plan_moments = coupled_cost_plan_moments
    try:
        rows = [
            clark_full.run_seed_full(seed, family, noise, device, use_full=True, element=element)
            for seed in range(seed_offset, seed_offset + n_seeds)
        ]
    finally:
        clark_full.full_cost_plan_moments = original

    arms = list(rows[0]["arms"].keys())
    regret = {a: float(np.mean([r["arms"][a]["regret"] for r in rows])) for a in arms}
    tests = {}
    for other in ("step", "bracket"):
        # clark_full's own paired sign test, so the statistics match the published comparison.
        # _sign_paired returns (mean of regret[a] - regret[b], n non-tied, k where a is
        # BETTER, p) -- it tests -d, so the third element counts clark_cvar's WINS.
        mean_diff, n_nonzero, k_clark_better, p = clark_full._sign_paired(rows, "clark_cvar", other)
        tests[f"clark_cvar_vs_{other}"] = {
            "p": p,
            "n_nonzero": n_nonzero,
            "k_clark_better": k_clark_better,
            "mean_regret_diff": mean_diff,  # clark_cvar - other; NEGATIVE means Clark wins
        }
    passed = all(t["p"] < 0.05 and t["mean_regret_diff"] < 0.0 for t in tests.values())
    return {
        "family": family,
        "noise": noise,
        "n_seeds": n_seeds,
        "seed_offset": seed_offset,
        "rows": rows,
        "mean_regret": regret,
        "tests": tests,
        "passed": passed,  # PRE-REG: clark_cvar beats step AND bracket at p < 0.05, full cost
    }


def _print_gate(g: dict, args: argparse.Namespace) -> None:
    print(
        f"=== GATE H: pose-coupled hinge vs true end-to-end MC "
        f"(cross_scope={args.cross_scope}, rng_offset={args.rng_offset}) ==="
    )
    print(
        f"  E-ratio      coupled {g['median_e_ratio']:.3f}   frozen {g['frozen_median_e_ratio']:.3f}"
    )
    print(
        f"  |err|/mc_sd  coupled {g['median_err_over_mcsd']:.3f}   frozen {g['frozen_median_err_over_mcsd']:.3f}"
    )
    print(f"  sd-ratio     coupled {g['median_sd_ratio']:.3f}")
    print(f"  corr(E, MC)  coupled {g['corr_mean']:.3f}")
    for k, v in g["criteria"].items():
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  GATE H: {'PASSED' if g['passed'] else 'FAILED'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cases", type=int, default=12)
    ap.add_argument("--cross-scope", default=CROSS_SCOPE, choices=("same_t", "all"))
    ap.add_argument("--rng-offset", type=int, default=1, help="1 = gate1b's cases; 7 = virgin")
    ap.add_argument("--tag", default="", help="suffix for the output json")
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--skip-gate", action="store_true", help="stage 2 only; gate already run")
    ap.add_argument(
        "--force-stage2",
        action="store_true",
        help="run stage 2 even though Gate H failed -- EXPLORATORY, must be user-authorized",
    )
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="first stage-2 seed; virgin confirmation uses 5000 (PREREG_stage2_virgin.md)",
    )
    ap.add_argument("--element", default="sphere", choices=("sphere", "cylinder"))
    ap.add_argument("--family", default="hybrid", help="stage-2 plan family")
    ap.add_argument("--noise", default="all", help="stage-2 noise arm")
    args = ap.parse_args()
    wp.init()

    out: dict = {
        "element": args.element,
        # matched-element bookkeeping (Stage B, matched_truth.py): gateH's/stage2's trajectory +
        # MC truth settle under `element` (stage2 inherits it via run_seed_full); the fosm/bracket
        # arms inside stage2 read DifferentiableSimulator directly and stay sphere-only regardless.
        "element_trajectory": args.element,
        "element_mc_truth": args.element,
        "element_gradient_arms": "sphere",
    }
    if not args.skip_gate:
        out["gateH"] = gateH_trajectory_vs_mc(
            args.device, args.cases, args.cross_scope, args.rng_offset, args.element
        )
    g = out.get("gateH")
    if g is not None:
        _print_gate(g, args)

    if args.stage2:
        if g is not None and not g["passed"] and not args.force_stage2:
            print("Gate H failed -- stage 2 not run (pre-registered rule).")
        else:
            # EXPLORATORY when --force-stage2: Gate H failed criterion (iv) (cross-case Pearson
            # correlation 0.915 vs a 0.93 bar whose own comparator, the frozen model, scores
            # 0.905 on the same virgin cases). The user authorized this run in full knowledge of
            # that; it is reported as exploratory, and the pre-registered bar is not relaxed.
            out["stage2_label"] = (
                "exploratory: Gate H failed criterion (iv); user-authorized"
                if args.force_stage2
                else "confirmatory: Gate H passed"
            )
            out["stage2"] = run_stage2(
                args.device, args.seeds, args.family, args.noise, args.seed_offset, args.element
            )
            s = out["stage2"]
            print(f"\n=== STAGE 2: full cost (settle + clear_soft), {args.family}/{args.noise} ===")
            for a, v in sorted(s["mean_regret"].items(), key=lambda kv: kv[1]):
                print(f"  {a:12s} {v:.4f}")
            for k, v in s["tests"].items():
                print(f"  {k}: p={v['p']:.2e}, mean regret diff {v['mean_regret_diff']:+.4f}")
            print(f"  STAGE 2: {'PASSED' if s['passed'] else 'FAILED'}")

    path = OUT / f"clark_hinge{args.tag}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
