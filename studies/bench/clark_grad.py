"""Is Clark's E[J|sigma] a valid per-cell DERIVATIVE at map-noise (cm) scale?

    .venv/bin/python -m studies.bench.clark_grad

CLAIMS.md C2 / RESULTS.md: the HARD adjoint dJ/dh, finite-differenced against ITSELF on the
belief, is exactly correct at millimetre scale (2.3% error) and useless at the scale sigma
actually lives at (32% error at 1 cm, sigma 2-30 cm) -- the contact arg-max's validity radius is
millimetres, the map's uncertainty is centimetres.

`clark.py` built the one object that never linearizes the contact max: E[J|sigma], via Clark's
closed-form moments of the wheel-envelope's max-of-correlated-Gaussians, folded over the ~37-cell
footprint. E[J|sigma] is a SMOOTH function of the belief (each candidate's mean enters through a
Gaussian CDF/PDF, not an arg-max), so unlike the hard adjoint, its own derivative is not gated by
a millimetre-scale switching event -- IF that hypothesis holds, this module is where it is tested.

THREE CANDIDATE PREDICTORS OF A CM-SCALE PERTURBATION dE_mc = E[J](h+delta*e_i) - E[J](h),
against COMMON-RANDOM-NUMBER Monte-Carlo truth:

  dE_fn   = ClarkE[J](h + delta*e_i) - ClarkE[J](h)      -- Clark's own FUNCTION difference,
            still cm-scale accurate if Clark tracks the true E[J] surface at all (this is the
            weaker, "smooth-but-curved" claim if the next one fails).
  dE_lin  = g_clark_i * delta,  g_clark_i = d(ClarkE[J])/dh_i (central FD at eps=1mm, i.e.
            Clark's OWN gradient, evaluated where the hard adjoint is known-valid) -- the actual
            per-cell derivative program's reopening claim.
  dJ_hard = g_hard_i * delta,  g_hard_i = the real DifferentiableSimulator adjoint (Harness.
            adjoint, SETTLE_IDX) -- the historical comparison, reproduced under this protocol so
            the two live or die on the identical MC truth and the identical test cells.

Everything else (the moment-propagation machinery, the correlation kernel, RobotParams, the
settle-only cost scoping) is `clark.py`'s, reused wholesale and unedited.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from ..adjoint.harness import _perturb_cell
from ..adjoint.harness import Harness
from ..adjoint.sigma import NoiseDraws
from .clark import _cost_settle
from .clark import clark_plan_moments
from .clark import rho1_table
from .clark import RNG_SEED
from .clark import SETTLE_IDX
from .ranking import build_case
from .ranking import CELL
from .ranking import kendall_tau
from .ranking import N_PLANS
from .ranking import OUT
from .risk import CORR_LEN
from helhest.engine import RobotParams
from helhest.engine.envelope import wheel_offset_table

N_SEEDS = 10
N_PER_STRATUM = 4  # -> up to 5 * 4 = 20 test cells per seed, per the task's design
STRATA = ("on_support", "near_support", "off_support_swath", "high_sigma_unobserved", "random_far")
NEAR_RADIUS = 2  # [cells] "near-support" box radius
SWATH_DIST = 0.5  # [m] "under the swath" gate on distance to the plan's own trajectory
FAR_DIST = 2.0  # [m] "random far" gate

DELTAS_FIXED = (0.01, 0.03)  # [m]
SIGMA_CAP = 0.10  # [m] cap for the delta = sigma_i arm
FD_EPS = 1.0e-3  # [m] central-FD step for g_clark (millimetre scale, where Clark is smooth)
FD_EPS_LO, FD_EPS_HI = 0.5e-3, 2.0e-3  # smoothness bracket around FD_EPS
SMOOTH_TOL = 0.02  # pre-registered: g(0.5mm) vs g(2mm) must agree to < 2%

N_DRAWS_MC = 4096  # CRN Monte-Carlo truth draws, start value
N_DRAWS_MC_ESCALATED = 6144  # ceiling measured on the 4 GiB shared card (studies/bench/clark_grad.py
# smoke test): 8192 OOMs (Harness unconditionally allocates SubcellDilation's [B,ny,nx] buffers
# even when unused -- helhest_stack/studies/adjoint/harness.py -- so batch memory is several x
# the elevation stack alone). This caps how far the floor-ratio escalation below can go.
FLOOR_GATE = 3.0  # a cell/delta only enters the error tables if |dE_mc| > FLOOR_GATE * SEM
FLOOR_RATIO_TARGET = 0.10  # escalate N_DRAWS_MC if median(SEM)/median(|dE_mc|) exceeds this

RP = RobotParams()


# --- reuse: the same footprint the wheel envelope's Clark nodes are built from ------------------
def _plan_universe(controlled_k: np.ndarray, cell: float, x0: float, y0: float, ny: int, nx: int) -> np.ndarray:
    """Absolute flat cell indices that ANY wheel-envelope Clark node of this plan can see.

    Perturbing a belief cell outside this set changes E[J_settle] by EXACTLY zero (it never
    enters any node's candidate mean) -- so this is also the hard adjoint's own potential-support
    superset, before the arg-max prunes it down to the single winner per node.
    """
    from .clark import _footprint_cells  # local import: private helper, used nowhere else here

    env_radius = int(np.ceil(RP.wheel_radius / cell))
    off_dy, off_dx, _off_cap = wheel_offset_table(env_radius, cell, RP.wheel_radius)
    wheel_xy = np.array([[0.0, RP.half_track], [0.0, -RP.half_track], [-RP.rear_offset, 0.0]])
    t_idx = np.arange(1, controlled_k.shape[0])
    x, y, yaw = controlled_k[t_idx, 0], controlled_k[t_idx, 1], controlled_k[t_idx, 2]
    c, s = np.cos(yaw), np.sin(yaw)
    wx = np.stack([x + wheel_xy[w, 0] * c - wheel_xy[w, 1] * s for w in range(3)])
    wy = np.stack([y + wheel_xy[w, 0] * s + wheel_xy[w, 1] * c for w in range(3)])
    cell_flat = _footprint_cells(wx.ravel(), wy.ravel(), off_dy, off_dx, x0, y0, cell, ny, nx)
    return np.unique(cell_flat)


def _plan_distance(controlled_k: np.ndarray, XX: np.ndarray, YY: np.ndarray) -> np.ndarray:
    """[ny, nx] distance from every cell to the nearest point of THIS plan's own trajectory."""
    p = controlled_k[:, :2]
    return np.sqrt(((XX[None] - p[:, 0, None, None]) ** 2 + (YY[None] - p[:, 1, None, None]) ** 2).min(0))


# --- stratified test-cell selection ---------------------------------------------------------
def select_test_cells(
    grad_k: np.ndarray, sigma: np.ndarray, observed: np.ndarray, dist_k: np.ndarray,
    rng: np.random.Generator,
) -> list[tuple[str, int, int]]:
    """Up to N_PER_STRATUM cells from each of the 5 strata (module docstring / task design)."""
    ny, nx = grad_k.shape
    support = grad_k != 0.0

    near = np.zeros_like(support)
    iys, ixs = np.nonzero(support)
    for dy in range(-NEAR_RADIUS, NEAR_RADIUS + 1):
        for dx in range(-NEAR_RADIUS, NEAR_RADIUS + 1):
            if dy == 0 and dx == 0:
                continue
            near[np.clip(iys + dy, 0, ny - 1), np.clip(ixs + dx, 0, nx - 1)] = True
    near &= ~support

    swath = (dist_k < SWATH_DIST) & ~support & ~near

    unobserved = ~observed
    hi_thr = np.quantile(sigma[unobserved], 0.75) if unobserved.any() else np.inf
    high_sigma = unobserved & (sigma >= hi_thr) & ~support & ~near & ~swath

    far = (dist_k > FAR_DIST) & ~support & ~near & ~swath & ~high_sigma

    masks = {
        "on_support": support, "near_support": near, "off_support_swath": swath,
        "high_sigma_unobserved": high_sigma, "random_far": far,
    }
    cells: list[tuple[str, int, int]] = []
    for label in STRATA:
        idx = np.argwhere(masks[label])
        take = min(N_PER_STRATUM, len(idx))
        for p in rng.choice(len(idx), size=take, replace=False) if take else []:
            iy, ix = idx[p]
            cells.append((label, int(iy), int(ix)))
    return cells


# --- Clark-side: function difference, and its own gradient (central FD at mm scale) -----------
def clark_e(belief: np.ndarray, sigma: np.ndarray, controlled_k, cell, x0, y0, corr_table) -> float:
    e_j, _var_j = clark_plan_moments(belief, sigma, controlled_k, RP, x0, y0, cell, corr_table)
    return e_j


def clark_grad_at(
    belief: np.ndarray, sigma: np.ndarray, controlled_k, cell, x0, y0, corr_table, iy: int, ix: int,
    eps: float,
) -> float:
    """Central FD of ClarkE[J] at cell (iy, ix), step `eps` -- Clark is smooth, so this IS its
    analytic derivative (verified per-cell by the eps=0.5/2mm bracket, see `smoothness_check`)."""
    b_plus, b_minus = belief.copy(), belief.copy()
    b_plus[iy, ix] += eps
    b_minus[iy, ix] -= eps
    e_plus = clark_e(b_plus, sigma, controlled_k, cell, x0, y0, corr_table)
    e_minus = clark_e(b_minus, sigma, controlled_k, cell, x0, y0, corr_table)
    return (e_plus - e_minus) / (2.0 * eps)


def smoothness_check(
    belief: np.ndarray, sigma: np.ndarray, controlled_k, cell, x0, y0, corr_table, iy: int, ix: int,
) -> tuple[float, float, float]:
    """g at eps=0.5mm / 1mm / 2mm and their relative spread -- the pre-registered smoothness
    verification (module docstring): Clark's FD gradient is only "the" derivative if it does not
    depend on the FD step size, unlike the hard adjoint's arg-max, which does."""
    g_lo = clark_grad_at(belief, sigma, controlled_k, cell, x0, y0, corr_table, iy, ix, FD_EPS_LO)
    g_mid = clark_grad_at(belief, sigma, controlled_k, cell, x0, y0, corr_table, iy, ix, FD_EPS)
    g_hi = clark_grad_at(belief, sigma, controlled_k, cell, x0, y0, corr_table, iy, ix, FD_EPS_HI)
    rel = abs(g_lo - g_hi) / max(abs(g_mid), 1e-9)
    return g_lo, g_hi, rel


# --- Monte-Carlo truth: common random numbers, one noise field reused for every test -----------
def mc_paired_diffs(
    scene, belief: np.ndarray, sigma: np.ndarray, pose_k: np.ndarray, omega_k: np.ndarray,
    tests: list[tuple[int, int, float]], n_draws: int, device: str, mc_seed: int,
) -> dict[tuple[int, int, float], tuple[float, float]]:
    """dE_mc, SEM for every (iy, ix, delta) in `tests`, all against the SAME `n_draws` correlated-
    noise field (CRN): the noise draws are generated ONCE, and each perturbation only shifts the
    baseline by a deterministic `delta` at one cell, so `field_pert - field_base` is exactly
    `delta` at that cell and identically zero elsewhere -- the paired difference's variance is
    therefore whatever nonlinearity the contact max itself contributes, not resampling noise."""
    ny, nx = belief.shape
    poses_mc = np.tile(pose_k, (n_draws, 1)).astype(np.float32)
    omega_mc = np.tile(omega_k[:, None, :], (1, n_draws, 1)).astype(np.float32)
    hd = Harness(scene, poses_mc, omega_mc, device=device)
    draws = NoiseDraws((n_draws, ny, nx), CELL, CORR_LEN, hd.device)
    with wp.ScopedDevice(hd.device):
        base_dev = wp.array(np.ascontiguousarray(np.tile(belief, (n_draws, 1, 1)), np.float32))
        sigma_dev = wp.array(np.ascontiguousarray(sigma, np.float32))
    draws.perturb(base_dev, sigma_dev, 1.0, hd.sim.elevation, mc_seed)
    field_base = hd.sim.elevation.numpy().copy()  # host backup: restored between perturbations
    j_base = _cost_settle(hd.forward(dilate=True))

    out: dict[tuple[int, int, float], tuple[float, float]] = {}
    for iy, ix, delta in tests:
        wp.launch(
            _perturb_cell, hd.batch_size, inputs=[hd.sim.elevation, iy, ix, float(delta)],
            device=hd.device,
        )
        j_pert = _cost_settle(hd.forward(dilate=True))
        diff = j_pert - j_base
        out[(iy, ix, delta)] = (float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(n_draws)))
        hd.sim.elevation.assign(field_base)
    del hd
    return out


# --- per-seed pipeline --------------------------------------------------------------------------
def run_seed(seed: int, device: str, n_draws: int) -> dict:
    scene, _truth, _measured, observed, sigma, poses, omega, (XX, YY) = build_case(
        seed, "hybrid", "all"
    )
    belief = scene.elevation.astype(np.float32)
    ny, nx = belief.shape
    corr_table = rho1_table(CORR_LEN, CELL)
    x0, y0 = scene.origin_x, scene.origin_y

    h = Harness(scene, poses, omega, device=device)
    grads, terms = h.adjoint(dilate=True, leaf="elevation")
    grad_all = grads[SETTLE_IDX]  # [N_PLANS, ny, nx]
    j_bel = _cost_settle(terms)
    k = int(np.argmin(j_bel))  # belief-best plan
    grad_k = grad_all[k]
    controlled_k = h.sim.controlled.numpy()[:, k, :]
    pose_k, omega_k = poses[k].copy(), omega[:, k, :].copy()
    del h

    dist_k = _plan_distance(controlled_k, XX, YY)
    rng = np.random.default_rng(RNG_SEED + seed)
    cells = select_test_cells(grad_k, sigma, observed, dist_k, rng)

    e_base = clark_e(belief, sigma, controlled_k, CELL, x0, y0, corr_table)
    per_cell = []
    tests: list[tuple[int, int, float]] = []
    for stratum, iy, ix in cells:
        delta_sigma = float(min(sigma[iy, ix], SIGMA_CAP))
        deltas = (*DELTAS_FIXED, delta_sigma)
        g_clark = clark_grad_at(belief, sigma, controlled_k, CELL, x0, y0, corr_table, iy, ix, FD_EPS)
        g_lo, g_hi, smooth_rel = smoothness_check(
            belief, sigma, controlled_k, CELL, x0, y0, corr_table, iy, ix
        )
        g_hard = float(grad_k[iy, ix])
        row = {
            "stratum": stratum, "iy": iy, "ix": ix, "sigma": float(sigma[iy, ix]),
            "g_clark": g_clark, "g_hard": g_hard, "smoothness_rel_err": smooth_rel,
            "deltas": {},
        }
        for delta in deltas:
            b_pert = belief.copy()
            b_pert[iy, ix] += delta
            e_pert = clark_e(b_pert, sigma, controlled_k, CELL, x0, y0, corr_table)
            row["deltas"][f"{delta:.5f}"] = {
                "delta": delta, "dE_fn": e_pert - e_base, "dE_lin": g_clark * delta,
                "dJ_hard": g_hard * delta,
            }
            tests.append((iy, ix, delta))
        per_cell.append(row)

    mc = mc_paired_diffs(scene, belief, sigma, pose_k, omega_k, tests, n_draws, device, 700_000 + seed)
    for row in per_cell:
        for key, d in row["deltas"].items():
            dE_mc, sem = mc[(row["iy"], row["ix"], d["delta"])]
            d["dE_mc"], d["sem"] = dE_mc, sem

    # --- catch-22: where does |g_clark|'s mass sit, vs the hard adjoint's, over the FULL
    # plan-universe (not just the 20 test cells) --------------------------------------------
    universe = _plan_universe(controlled_k, CELL, x0, y0, ny, nx)
    u_iy, u_ix = universe // nx, universe % nx
    g_clark_u = np.array(
        [clark_grad_at(belief, sigma, controlled_k, CELL, x0, y0, corr_table, iy, ix, FD_EPS)
         for iy, ix in zip(u_iy, u_ix)]
    )
    g_hard_u = grad_k[u_iy, u_ix]
    unobs_u = ~observed[u_iy, u_ix]
    mass_clark = float(np.abs(g_clark_u).sum())
    mass_hard = float(np.abs(g_hard_u).sum())
    catch22 = {
        "universe_size": int(universe.size),
        "frac_clark_mass_unobserved": float(np.abs(g_clark_u)[unobs_u].sum() / max(mass_clark, 1e-12)),
        "frac_hard_mass_unobserved": float(np.abs(g_hard_u)[unobs_u].sum() / max(mass_hard, 1e-12)),
    }

    return {
        "seed": seed, "plan": k, "n_draws_mc": n_draws, "per_cell": per_cell, "catch22": catch22,
        "stratum_counts": {s: sum(1 for c in cells if c[0] == s) for s in STRATA},
    }


# --- aggregation ---------------------------------------------------------------------------
def _gated_rows(rows: list[dict]) -> list[dict]:
    """Flatten (seed, cell, delta) triples, keep only |dE_mc| > FLOOR_GATE * sem (pre-registered)."""
    flat = []
    for r in rows:
        for c in r["per_cell"]:
            for key, d in c["deltas"].items():
                flat.append({**d, "stratum": c["stratum"], "seed": r["seed"], "delta_key": key})
    return [f for f in flat if abs(f["dE_mc"]) > FLOOR_GATE * max(f["sem"], 1e-12)]


def _rel_err_table(flat_gated: list[dict]) -> dict:
    """Median / p90 |rel err| of dE_fn / dE_lin / dJ_hard vs dE_mc, by delta-key and stratum."""
    out: dict = {}
    delta_keys = sorted({f["delta_key"] for f in flat_gated})
    for dk in delta_keys:
        sub = [f for f in flat_gated if f["delta_key"] == dk]
        out[dk] = {"n": len(sub), "by_stratum": {}}
        for metric in ("dE_fn", "dE_lin", "dJ_hard"):
            rel = np.array([abs(f[metric] - f["dE_mc"]) / abs(f["dE_mc"]) for f in sub])
            out[dk][metric] = {"median": float(np.median(rel)), "p90": float(np.percentile(rel, 90))}
        for s in STRATA:
            ssub = [f for f in sub if f["stratum"] == s]
            if not ssub:
                continue
            out[dk]["by_stratum"][s] = {"n": len(ssub)}
            for metric in ("dE_fn", "dE_lin", "dJ_hard"):
                rel = np.array([abs(f[metric] - f["dE_mc"]) / abs(f["dE_mc"]) for f in ssub])
                out[dk]["by_stratum"][s][metric] = {
                    "median": float(np.median(rel)), "p90": float(np.percentile(rel, 90)),
                }
    return out


def _attribution_tau(rows: list[dict]) -> dict:
    """Kendall tau of |g_clark| vs |dE_mc(delta=sigma)|, and |g_hard| vs the same, PER SEED
    (over that seed's ~20 test cells) then averaged -- the sigma-scale delta is the one the task
    designates for this check."""
    taus_clark, taus_hard = [], []
    for r in rows:
        g_clark, g_hard, mc_sigma = [], [], []
        for c in r["per_cell"]:
            sigma_key = f"{min(c['sigma'], SIGMA_CAP):.5f}"
            if sigma_key not in c["deltas"]:
                continue
            g_clark.append(abs(c["g_clark"]))
            g_hard.append(abs(c["g_hard"]))
            mc_sigma.append(abs(c["deltas"][sigma_key]["dE_mc"]))
        if len(g_clark) > 1:
            taus_clark.append(kendall_tau(np.array(g_clark), np.array(mc_sigma)))
            taus_hard.append(kendall_tau(np.array(g_hard), np.array(mc_sigma)))
    return {
        "tau_clark_mean": float(np.mean(taus_clark)) if taus_clark else float("nan"),
        "tau_hard_mean": float(np.mean(taus_hard)) if taus_hard else float("nan"),
        "n_seeds": len(taus_clark),
    }


SIGNAL_STRATA = ("on_support", "near_support", "off_support_swath")  # excludes the strata that
# are EXACT nulls by construction (high_sigma_unobserved / random_far sit outside the plan's
# wheel-envelope universe, so dE_mc and sem are both driven to float32 cancellation noise near
# machine epsilon there -- their ratio is meaningless and swamps a global median with outliers)


def _floor_accounting(rows: list[dict]) -> dict:
    flat, flat_signal = [], []
    for r in rows:
        for c in r["per_cell"]:
            for d in c["deltas"].values():
                flat.append(d)
                if c["stratum"] in SIGNAL_STRATA:
                    flat_signal.append(d)

    def _stats(fl: list[dict]) -> tuple[float, float, float]:
        sem = np.array([f["sem"] for f in fl])
        abs_de = np.abs(np.array([f["dE_mc"] for f in fl]))
        return float(np.median(sem)), float(np.median(abs_de)), float(
            np.median(sem) / max(np.median(abs_de), 1e-12)
        )

    sem_all = np.array([f["sem"] for f in flat])
    abs_de_all = np.abs(np.array([f["dE_mc"] for f in flat]))
    n_gated_out = int((abs_de_all <= FLOOR_GATE * sem_all).sum())
    med_sem, med_de, ratio = _stats(flat_signal)
    return {
        "n_total": len(flat), "n_excluded_by_gate": n_gated_out,
        "median_sem": med_sem, "median_abs_dE_mc": med_de,
        "median_sem_over_median_abs_dE_mc": ratio, "escalation_needed": ratio > FLOOR_RATIO_TARGET,
        "note": "sem/dE_mc ratio computed over SIGNAL_STRATA only (on/near-support, off-swath);"
                " the two null strata are exact zeros structurally and excluded from this ratio",
    }


def _smoothness_summary(rows: list[dict]) -> dict:
    rel = np.array([c["smoothness_rel_err"] for r in rows for c in r["per_cell"]])
    return {"median": float(np.median(rel)), "p90": float(np.percentile(rel, 90)),
            "passed": bool(np.median(rel) < SMOOTH_TOL)}


def report(rows: list[dict]) -> dict:
    smooth = _smoothness_summary(rows)
    print("=" * 92)
    print("SMOOTHNESS CHECK: g_clark(eps=0.5mm) vs g_clark(eps=2mm), relative spread")
    print(f"  median = {smooth['median']:.4f}  p90 = {smooth['p90']:.4f}  (bar: median < {SMOOTH_TOL})")
    if smooth["passed"]:
        print("  -> PASS -- Clark's FD gradient is well-defined independent of step size")
    else:
        print("  -> FAIL -- Clark's own gradient is NOT step-size-stable; treat g_clark results"
              " with caution")

    floor = _floor_accounting(rows)
    print("\nNOISE-FLOOR ACCOUNTING (CRN paired-difference SEM vs |dE_mc|)")
    print(f"  n={floor['n_total']} (seed, cell, delta) triples; excluded by the 3x-SEM gate:"
          f" {floor['n_excluded_by_gate']}")
    print(f"  median SEM = {floor['median_sem']:.5f}   median |dE_mc| = {floor['median_abs_dE_mc']:.5f}"
          f"   ratio = {floor['median_sem_over_median_abs_dE_mc']:.3f}  (target < {FLOOR_RATIO_TARGET})")

    gated = _gated_rows(rows)
    table = _rel_err_table(gated)
    print(f"\nERROR TABLE ({len(gated)}/{sum(len(c['deltas']) for r in rows for c in r['per_cell'])}"
          " triples pass the |dE_mc| > 3x-floor gate)")
    for dk, d in sorted(table.items(), key=lambda kv: float(kv[0])):
        print(f"\n  delta = {dk} m  (n={d['n']})")
        for metric, label in (("dE_fn", "Clark-fn"), ("dE_lin", "Clark-lin"), ("dJ_hard", "hard-adj")):
            print(f"    {label:<10} median={d[metric]['median']:.4f}  p90={d[metric]['p90']:.4f}")
        for s, sd in d["by_stratum"].items():
            print(f"    [{s:<22} n={sd['n']:<3}]  Clark-lin med={sd['dE_lin']['median']:.4f}"
                  f"  hard med={sd['dJ_hard']['median']:.4f}")

    tau = _attribution_tau(rows)
    print(f"\nATTRIBUTION (Kendall tau of |gradient| vs |dE_mc(delta=sigma)|, mean over"
          f" {tau['n_seeds']} seeds)")
    print(f"  |g_clark| tau = {tau['tau_clark_mean']:+.3f}")
    print(f"  |g_hard|  tau = {tau['tau_hard_mean']:+.3f}")

    c22_clark = np.mean([r["catch22"]["frac_clark_mass_unobserved"] for r in rows])
    c22_hard = np.mean([r["catch22"]["frac_hard_mass_unobserved"] for r in rows])
    print("\nCATCH-22: fraction of |gradient| mass sitting on UNOBSERVED cells"
          " (over the full plan universe, not just the 20 test cells)")
    print(f"  Clark: {c22_clark:.3f}   hard adjoint: {c22_hard:.3f}"
          f"  (historically hard put ~5x LESS mass where sigma is largest)")

    med_lin_1cm = table.get("0.01000", {}).get("dE_lin", {}).get("median", float("nan"))
    # delta=sigma keys vary per cell (each cell's own min(sigma_i, cap)); pool everything that
    # is NOT one of the two fixed deltas to get the "delta=sigma" arm's pooled error.
    sigma_keys = [dk for dk in table if dk not in ("0.01000", "0.03000")]
    sigma_rel = [
        abs(f["dE_lin"] - f["dE_mc"]) / abs(f["dE_mc"]) for f in gated if f["delta_key"] in sigma_keys
    ]
    med_lin_sigma = float(np.median(sigma_rel)) if sigma_rel else float("nan")

    crit_i = med_lin_1cm < 0.10
    crit_ii = med_lin_sigma < 0.25
    crit_iii = tau["tau_clark_mean"] > tau["tau_hard_mean"]
    print("\n" + "=" * 92)
    print("VERDICT vs the three pre-registered criteria")
    print(f"  (i)   Clark-lin median |rel err| @1cm  < 0.10 : {'YES' if crit_i else 'NO'}"
          f"  (measured {med_lin_1cm:.3f}; hard adjoint historically 0.32)")
    print(f"  (ii)  Clark-lin median |rel err| @sigma < 0.25: {'YES' if crit_ii else 'NO'}"
          f"  (measured {med_lin_sigma:.3f})")
    print(f"  (iii) tau(|g_clark|) > tau(|g_hard|)          : {'YES' if crit_iii else 'NO'}"
          f"  ({tau['tau_clark_mean']:.3f} vs {tau['tau_hard_mean']:.3f})")

    return {
        "smoothness": smooth, "floor": floor, "error_table": table, "attribution": tau,
        "catch22": {"clark_frac_unobserved": float(c22_clark), "hard_frac_unobserved": float(c22_hard)},
        "verdict": {
            "i_clark_lin_1cm_lt_10pct": crit_i, "ii_clark_lin_sigma_lt_25pct": crit_ii,
            "iii_tau_clark_gt_tau_hard": crit_iii,
            "median_rel_err_lin_1cm": med_lin_1cm, "median_rel_err_lin_sigma": med_lin_sigma,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-draws", type=int, default=N_DRAWS_MC)
    a = ap.parse_args()

    wp.init()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = [run_seed(seed, a.device, a.n_draws) for seed in range(a.seeds)]
    print(f"  ran {a.seeds} seeds at n_draws={a.n_draws}", flush=True)

    floor = _floor_accounting(rows)
    if floor["escalation_needed"] and a.n_draws < N_DRAWS_MC_ESCALATED:
        print(
            f"\n!! signal-strata floor ratio {floor['median_sem_over_median_abs_dE_mc']:.3f} >"
            f" {FLOOR_RATIO_TARGET} -- escalating to n_draws={N_DRAWS_MC_ESCALATED} and rerunning !!",
            flush=True,
        )
        try:
            rows = [run_seed(seed, a.device, N_DRAWS_MC_ESCALATED) for seed in range(a.seeds)]
        except RuntimeError as exc:  # the 4 GiB shared card has a hard ceiling near 6144-8192
            print(f"\n!! escalation OOM'd ({exc}) -- keeping the n_draws={a.n_draws} results !!")

    summary = report(rows)
    path = OUT / "clark_grad.json"
    path.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
