"""How well can you estimate a plan's RISK, and what does it cost?

    .venv/bin/python -m studies.bench.risk --seeds 40

This reframes the study around the question a risk-aware planner actually asks. Not "which
cells should I look at" -- section 7 settled that the adjoint beats entropy but ties a distance
transform -- but "given a belief about the map, how risky is this plan?", which is what every
uncertainty-aware off-road planner in the literature needs and computes.

The useful property of that question is that MONTE-CARLO IS BOTH THE STRONGEST BASELINE AND THE
GROUND TRUTH. Draw terrain realisations from the belief, roll every plan out on each, and the
empirical distribution of the cost IS the answer -- no independence assumption, no linearisation,
the wheel-envelope max handled exactly because it is simply evaluated. Everything else is an
approximation of it, and can be scored against it directly.

The ladder, cheapest first:

  none      J(belief). Ignores uncertainty. The "does any of this help" control.
  sum_sigma J(belief) + lambda * sum of sigma under the path. The obvious heuristic, and the
            one this project's own Study B predicts should fail: an unobserved patch holding
            67.3% of the map's total sigma^2 drew 0.000% of the cost variance, because no wheel
            could reach it.
  step      J(belief) + kappa * sum_t sigma_t, the Gaussian CVaR of STEP (Fan et al., RSS 2021):
            per-timestep risk assumed normal, kappa = phi(Phi^-1(alpha)) / (1 - alpha) fixed by
            the risk level rather than tuned. This is the field's default and the baseline a
            reviewer will name.
  fosm      J(belief) + kappa * sqrt(Var_FOSM), OURS: the variance obtained analytically from
            the taped adjoint under the same correlated noise model, one backward pass.
  mc        the empirical CVaR over N_DRAWS terrain realisations. Truth, and expensive.

WHAT IS HELD FIXED so the comparison is about the estimator and nothing else: the same belief,
the same sigma field, the same correlation length, the same plans, the same seeds. The MC draws
use COMMON RANDOM NUMBERS across plans -- every plan sees the identical set of terrains -- so
plan-to-plan differences are paired and the truth is far less noisy than independent draws.

Two scores are reported, and they answer different questions:

  ACCURACY   how close is the estimated CVaR to the MC CVaR, in cost units. Only meaningful for
             `step` and `fosm`, which produce a genuine CVaR; `sum_sigma` carries an arbitrary
             lambda and can only be scored on the ordering it induces.
  DECISION   pick the argmin plan under each estimator, then look up that plan's TRUE (MC) CVaR.
             The regret against the best achievable is what a planner actually pays, and it is
             comparable across every arm including the ones with arbitrary scaling.
"""

from __future__ import annotations

import argparse
import json
from math import erf
from math import exp
from math import pi
from math import sqrt

import numpy as np
import warp as wp

from ..adjoint.harness import Harness
from ..adjoint.harness import TERM_NAMES
from ..adjoint.sigma import fosm_variance
from ..adjoint.sigma import NoiseDraws
from .ranking import _cost
from .ranking import build_case
from .ranking import CELL
from .ranking import COST_TERMS
from .ranking import kendall_tau
from .ranking import N_PLANS
from .ranking import OUT
from .ranking import sign_test

CORR_LEN = 0.15  # [m] map-error correlation length; MC and FOSM MUST share it
N_DRAWS = 256  # terrain realisations per seed -- the MC truth
ALPHA = 0.90  # CVaR risk level
# `sum_sigma` accumulates sigma over the AREA swept by the path -- each cell within the
# footprint of any pose, counted once. That is the shape of the naive heuristic and it is
# genuinely different from STEP's per-timestep sum: it grows with swept area rather than with
# time, which is the overcounting failure mode. Its absolute scale is arbitrary, so per seed it
# is rescaled to the same MEAN risk magnitude as `step`; only its variation ACROSS plans -- the
# part that decides a ranking -- is then under test.
FOOTPRINT_M = 0.35  # [m] radius of the contact patch aggregated per timestep
# How sigma under the footprint becomes one sigma_t. "max" is the physically tempting choice --
# the height that decides the settle is itself a max over the patch -- but it DEGENERATES here:
# with ~80% of the map unobserved at the sigma cap, the max is the cap under every footprint of
# every plan, so `sum_t sigma_t` is identical for all 16 plans (measured spread: 0.000) and the
# STEP baseline cannot rank anything. That is a real property of saturating sigma fields and is
# reported as a finding, but running the baseline in a form that cannot discriminate would be a
# straw man. "rms" partially credits observed cells under the patch and does discriminate.
FOOTPRINT_AGG = "rms"


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF by bisection on `math.erf`. scipy is not a dependency."""
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + erf(mid / sqrt(2.0))) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cvar_kappa(alpha: float) -> float:
    """kappa such that CVaR_alpha(N(mu, sigma^2)) = mu + kappa * sigma  (STEP eq. 15)."""
    z = _norm_ppf(alpha)
    return exp(-0.5 * z * z) / sqrt(2.0 * pi) / (1.0 - alpha)


KAPPA = cvar_kappa(ALPHA)


def empirical_cvar(samples: np.ndarray, alpha: float) -> np.ndarray:
    """Mean of the worst (1 - alpha) tail, per column. `samples` is [n_draws, n_plans]."""
    n_tail = max(1, round((1.0 - alpha) * samples.shape[0]))
    return np.sort(samples, axis=0)[-n_tail:].mean(axis=0)


def _footprint_sigma(
    traj: np.ndarray, sigma: np.ndarray, grid, how: str = FOOTPRINT_AGG
) -> np.ndarray:
    """sigma_t: sigma under the contact patch at each step, aggregated per plan. [T+1, K]

    See FOOTPRINT_AGG for why this is not a max.
    """
    XX, YY = grid
    out = np.empty(traj.shape[:2], np.float32)
    r2 = FOOTPRINT_M**2
    for t in range(traj.shape[0]):
        for k in range(traj.shape[1]):
            m = (XX - traj[t, k, 0]) ** 2 + (YY - traj[t, k, 1]) ** 2 <= r2
            if not m.any():
                out[t, k] = 0.0
            elif how == "max":
                out[t, k] = sigma[m].max()
            elif how == "mean":
                out[t, k] = sigma[m].mean()
            else:
                out[t, k] = np.sqrt((sigma[m] ** 2).mean())
    return out


def _swept_area_sigma(traj: np.ndarray, sigma: np.ndarray, grid) -> np.ndarray:
    """Total sigma over the AREA the path sweeps: each cell within the footprint, counted once.

    The naive per-cell heuristic. Unlike a per-timestep sum it accumulates with swept area, so
    two plans over equally uncertain ground are separated by how much of it they cover.
    """
    XX, YY = grid
    out = np.empty(traj.shape[1], np.float64)
    r2 = FOOTPRINT_M**2
    for k in range(traj.shape[1]):
        covered = np.zeros(sigma.shape, bool)
        for t in range(traj.shape[0]):
            covered |= (XX - traj[t, k, 0]) ** 2 + (YY - traj[t, k, 1]) ** 2 <= r2
        out[k] = sigma[covered].sum()
    return out


def run_seed(seed: int, family: str, noise: str) -> dict:
    scene, _truth, _meas, _obs, sigma, poses, omega, grid = build_case(seed, family, noise)
    belief = scene.elevation.astype(np.float32)
    ny, nx = belief.shape

    # --- the estimators that only need the belief -------------------------------------
    h = Harness(scene, poses, omega, device="cuda")
    grads, terms = h.adjoint(dilate=True, leaf="elevation")
    grad = sum(w * grads[TERM_NAMES.index(k)] for k, w in COST_TERMS.items())  # [K, ny, nx]
    j_bel = _cost(terms)
    traj = h.sim.controlled.numpy()[:, :, :2].copy()

    sig_t = _footprint_sigma(traj, sigma, grid)  # [T+1, K]
    sig_t_max = _footprint_sigma(traj, sigma, grid, "max")
    area_sig = _swept_area_sigma(traj, sigma, grid)  # [K]
    step_risk = sig_t.sum(axis=0)
    # match the mean magnitude so the arms differ in SHAPE, not in an arbitrary lambda
    area_risk = area_sig * (step_risk.mean() / max(area_sig.mean(), 1e-9))
    est = {
        "none": j_bel.copy(),
        "sum_sigma": j_bel + KAPPA * area_risk,
        "step": j_bel + KAPPA * step_risk,
        "fosm": j_bel
        + KAPPA
        * np.sqrt(
            [max(fosm_variance(grad[k], sigma, CELL, CORR_LEN), 0.0) for k in range(N_PLANS)]
        ),
    }
    del h

    # --- Monte-Carlo truth: batch = DRAWS, one plan at a time -------------------------
    # NoiseDraws fills a [B, ny, nx] stack with an independent correlated field per slice, so
    # putting the draws on the batch axis gives N_DRAWS realisations from ONE forward pass.
    poses_d = np.tile(poses[0], (N_DRAWS, 1)).astype(np.float32)
    omega_d = np.zeros((omega.shape[0], N_DRAWS, 3), np.float32)
    hd = Harness(scene, poses_d, omega_d, device="cuda")
    draws = NoiseDraws((N_DRAWS, ny, nx), CELL, CORR_LEN, hd.device)
    with wp.ScopedDevice(hd.device):
        base = wp.array(np.ascontiguousarray(np.tile(belief, (N_DRAWS, 1, 1)), np.float32))
        # sigma is shared across slices (2-D by design): every draw uses the same field.
        sig_dev = wp.array(np.ascontiguousarray(sigma, np.float32), dtype=wp.float32)
    samples = np.empty((N_DRAWS, N_PLANS), np.float32)
    for k in range(N_PLANS):
        hd.sim.start_pose.assign(np.tile(poses[k], (N_DRAWS, 1)).astype(np.float32))
        hd.sim.target_wheel_omega.assign(
            np.ascontiguousarray(np.repeat(omega[:, k : k + 1, :], N_DRAWS, axis=1), np.float32)
        )
        # COMMON RANDOM NUMBERS: the same seed for every plan, so all plans are evaluated on the
        # identical set of terrains and their differences are paired.
        draws.perturb(base, sig_dev, 1.0, hd.sim.elevation, 900_000 + seed)
        samples[:, k] = _cost(hd.forward(dilate=True))
    del hd

    # --- the 2x2 that unconfounds WEIGHTING from NORM --------------------------------
    # The path-sigma surrogate and FOSM differ in two ways at once: whether sigma is weighted
    # by the cost's sensitivity, and whether the aggregation is an L1 sum or an L2 norm. Varying
    # them independently says which one is doing the work. Scored on the ORDERING they induce
    # over plans (Kendall tau against the MC truth), so their arbitrary units cancel.
    sig_flat = sigma.ravel()
    g_abs = np.abs(grad).reshape(N_PLANS, -1)
    surro = {
        "unweighted_L1": sig_t.sum(axis=0),
        "unweighted_L2": np.sqrt((sig_t**2).sum(axis=0)),
        "weighted_L1": (g_abs * sig_flat).sum(axis=1),
        "weighted_L2": np.sqrt(((g_abs * sig_flat) ** 2).sum(axis=1)),
        "fosm_corr": np.sqrt(
            [max(fosm_variance(grad[k], sigma, CELL, CORR_LEN), 0.0) for k in range(N_PLANS)]
        ),
    }

    mc_mean = samples.mean(axis=0)
    mc_cvar = empirical_cvar(samples, ALPHA)
    best = int(np.argmin(mc_cvar))

    out = {
        "seed": seed,
        "mc_cvar_spread": float(mc_cvar.max() - mc_cvar.min()),
        "mc_bias": float(np.mean(mc_mean - j_bel)),  # E[J] - J(belief): the max's own bias
        # how much a max-aggregated STEP cost could discriminate at all (0 = not at all)
        "sigma_sum_spread_max": float(np.ptp(sig_t_max.sum(axis=0))),
        "sigma_sum_spread_rms": float(np.ptp(sig_t.sum(axis=0))),
        "arms": {},
    }
    for name, v in est.items():
        pick = int(np.argmin(v))
        rec = {
            "regret": float(mc_cvar[pick] - mc_cvar[best]),
            "tau": kendall_tau(v, mc_cvar),
            "picked_best": bool(pick == best),
        }
        if name in ("step", "fosm"):  # only these produce a CVaR in cost units
            rec["cvar_err"] = float(np.mean(np.abs(v - mc_cvar)))
            rec["cvar_ratio"] = float(np.mean((v - j_bel) / np.maximum(mc_cvar - j_bel, 1e-6)))
        out["arms"][name] = rec
    out["arms"]["mc"] = {"regret": 0.0, "tau": 1.0, "picked_best": True}
    # Rank each surrogate against the TRUE risk (MC CVaR minus the belief cost), i.e. against
    # the quantity a risk term is supposed to be proportional to.
    true_risk = mc_cvar - j_bel
    out["surrogates"] = {
        k: {"tau": kendall_tau(v, true_risk), "spread": float(np.ptp(v))} for k, v in surro.items()
    }
    return out


def report(rows: list[dict]) -> None:
    n = len(rows)
    print(f"\nn={n} seeds, {N_DRAWS} terrain draws each, CVaR at alpha={ALPHA} (kappa={KAPPA:.3f})")
    print(
        f"true CVaR spread across the {N_PLANS} plans: {np.mean([r['mc_cvar_spread'] for r in rows]):.2f}"
    )
    print(
        f"E[J] - J(belief) = {np.mean([r['mc_bias'] for r in rows]):+.3f}  "
        "-- the wheel envelope's max biases the cost UP under uncertainty; a mean-map\n"
        "   planner is optimistic by this much before any risk term is added\n"
    )

    sm = np.mean([r["sigma_sum_spread_max"] for r in rows])
    sr = np.mean([r["sigma_sum_spread_rms"] for r in rows])
    print(
        f"sum_t sigma_t spread across plans: {sr:.3f} with an RMS footprint, {sm:.3f} with a MAX\n"
        "   -- a max-aggregated per-cell risk cost cannot rank these plans AT ALL, because a\n"
        "      saturating sigma field pins it to the cap under every footprint\n"
    )
    print("DECISION QUALITY -- pick the argmin, pay its true CVaR")
    print(f"{'estimator':<12}{'regret':>9}{'picked best':>13}{'tau vs truth':>14}")
    for a in ("none", "sum_sigma", "step", "fosm", "mc"):
        rg = np.mean([r["arms"][a]["regret"] for r in rows])
        pb = np.mean([r["arms"][a]["picked_best"] for r in rows])
        tt = np.mean([r["arms"][a]["tau"] for r in rows])
        tag = "  <- truth" if a == "mc" else ("  <- ours" if a == "fosm" else "")
        print(f"{a:<12}{rg:>9.3f}{pb:>12.0%}{tt:>+14.3f}{tag}")

    print("\nACCURACY of the risk term itself (only arms that produce a CVaR in cost units)")
    print(f"{'estimator':<12}{'|err|':>9}{'est/true risk':>15}")
    for a in ("step", "fosm"):
        e = np.mean([r["arms"][a]["cvar_err"] for r in rows])
        rt = np.mean([r["arms"][a]["cvar_ratio"] for r in rows])
        print(f"{a:<12}{e:>9.3f}{rt:>15.2f}")
    print("  (est/true risk = 1.0 is perfect; >1 over-conservative, <1 over-confident)")

    print("\nDOES GRADIENT WEIGHTING HELP?  2x2 over {weighted, unweighted} x {L1, L2}")
    print("Kendall tau of each surrogate against the TRUE risk (MC CVaR - J(belief)),")
    print("scored on ORDERING so the arbitrary units of each surrogate cancel.\n")
    keys = ("unweighted_L1", "unweighted_L2", "weighted_L1", "weighted_L2", "fosm_corr")
    print(f"{'surrogate':<16}{'tau vs true risk':>18}")
    for k in keys:
        print(f"{k:<16}{np.mean([r['surrogates'][k]['tau'] for r in rows]):>+18.3f}")
    print("\npaired: does weighting help at a FIXED norm?")
    for w, u in (("weighted_L1", "unweighted_L1"), ("weighted_L2", "unweighted_L2")):
        d = np.array([r["surrogates"][w]["tau"] - r["surrogates"][u]["tau"] for r in rows])
        k, wn, p = sign_test(d)
        print(f"  {w:<14} - {u:<16} {d.mean():>+7.3f}   better on {wn:>3}/{k:<3}   p={p:.2e}")
    print("paired: does the norm matter at FIXED weighting?")
    for a, b in (("unweighted_L2", "unweighted_L1"), ("weighted_L2", "weighted_L1")):
        d = np.array([r["surrogates"][a]["tau"] - r["surrogates"][b]["tau"] for r in rows])
        k, wn, p = sign_test(d)
        print(f"  {a:<14} - {b:<16} {d.mean():>+7.3f}   better on {wn:>3}/{k:<3}   p={p:.2e}")

    print("\npaired per seed, regret difference (negative = first arm better)")
    for a, b in (("fosm", "step"), ("fosm", "sum_sigma"), ("fosm", "none"), ("step", "none")):
        d = np.array([r["arms"][a]["regret"] - r["arms"][b]["regret"] for r in rows])
        k, w, p = sign_test(-d)
        print(f"  {a:<10} vs {b:<10} mean {d.mean():>+7.3f}   better on {w:>3}/{k:<3}   p={p:.2e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--family", default="hybrid")
    ap.add_argument("--noise", default="all")
    a = ap.parse_args()

    wp.init()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(a.seeds):
        rows.append(run_seed(seed, a.family, a.noise))
        if (seed + 1) % 10 == 0:
            print(f"  {seed + 1}/{a.seeds} seeds", flush=True)
    report(rows)
    path = OUT / f"risk_{a.family}_{a.noise}.json"
    path.write_text(json.dumps({"family": a.family, "noise": a.noise, "rows": rows}, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
