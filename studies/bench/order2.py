"""Second-order FOSM as a risk estimator, and the paired domain test §7h left owed.

    .venv/bin/python -m studies.bench.order2 --seeds 40

Two open questions, answered in ONE run so every comparison is paired within-seed:

  1. Does the second-order correction of §4 rescue the risk estimate? §4 measured it cutting
     badly-wrong variance estimates from 34.1% to 3.2%, and §7f measured first-order FOSM
     over-predicting the risk by 2.04x. If that miscalibration is curvature, this fixes it.

  2. Is the inversion really the AGGREGATION DOMAIN? §7h concluded it is -- per-timestep
     tau = +0.101 against per-cell -0.128 -- but that comparison was made ACROSS runs
     (n = 100 vs n = 80), which is below the standard the rest of the study holds to. Both
     scores are computed here on the same seeds, so the domain test is finally paired.

Every score is ranked against the same truth: the Monte-Carlo CVaR minus the belief cost, i.e.
the risk a plan actually carries. Scores are compared on ORDERING, so their differing units
cancel and no arbitrary lambda enters.

TRUNCATION, stated because it is the one thing that could bias question 1. Curvature costs two
forwards per cell, so it is computed only on a shortlist: cells inside the plans' contact
support, taken by largest sigma. First and second order are then BOTH evaluated on that same
shortlist (`fosm1_short` vs `fosm2_short`), so the curvature term is the only difference between
them. `fosm1_full` is carried alongside to show what the truncation itself costs.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import warp as wp

from ..adjoint.harness import Harness
from ..adjoint.sigma import NoiseDraws
from .element import element_offsets_single
from .ranking import _cost
from .ranking import build_case
from .ranking import CELL
from .ranking import kendall_tau
from .ranking import N_PLANS
from .ranking import OUT
from .ranking import sign_test
from .risk import _footprint_sigma
from .risk import ALPHA
from .risk import CORR_LEN
from .risk import empirical_cvar
from .risk import KAPPA
from .risk import N_DRAWS
from .second_order import _curvature
from .softgrad import _split_gradients
from .softgrad import soft_gradient

SHORTLIST = 400  # cells the curvature probe is spent on


def run_seed(seed: int, family: str, noise: str, element: str = "sphere") -> dict:
    scene, _t, _m, _o, sigma, poses, omega, grid = build_case(seed, family, noise)
    belief = scene.elevation.astype(np.float32)
    ny, nx = belief.shape

    h = Harness(scene, poses, omega, device="cuda")
    g_env, g_direct, g_hard, terms = _split_gradients(h)
    j_bel = _cost(terms)
    traj = h.sim.controlled.numpy()[:, :, :2].copy()
    # soft_gradient applies ONE table uniformly to the whole grid (no per-cell pose), so the
    # cylinder table is taken at a single reference heading -- see `element_offsets_single`.
    dy, dx, cap = element_offsets_single(element, CELL, h.robot_params)
    g_soft = soft_gradient(g_env, belief, dy, dx, cap, sigma) + g_direct

    # Shortlist: inside the contact support of ANY plan, largest sigma first. Chosen by support
    # (which §7g/§7h established is the reliable part of the adjoint) and by sigma -- never by
    # gradient magnitude, which is the quantity under suspicion.
    support = (np.abs(g_hard) > 0).any(axis=0)
    cand = np.flatnonzero(support.ravel())
    cand = cand[np.argsort(-sigma.ravel()[cand])][:SHORTLIST]
    cells = np.stack(np.unravel_index(cand, sigma.shape), axis=1)
    deltas = sigma.ravel()[cand]
    curv = _curvature(h, cells, deltas, j_bel)  # [N, K]
    del h

    gs = g_hard.reshape(N_PLANS, -1)[:, cand]  # [K, N]
    sg = (gs * deltas) ** 2  # first-order variance contribution, on the shortlist
    cv = 0.5 * (curv.T * deltas**2) ** 2  # second-order variance contribution
    bias = 0.5 * (curv.T * deltas**2).sum(axis=1)  # E[J] - J0, truncated to the shortlist

    sig_t = _footprint_sigma(traj, sigma, grid)
    scores = {
        "time_sigma": sig_t.sum(axis=0),
        "cell_sigma": np.array([float(sigma[np.abs(g_hard[k]) > 0].sum()) for k in range(N_PLANS)]),
        "fosm1_full": np.sqrt(((g_hard * sigma) ** 2).sum(axis=(1, 2))),
        "fosm1_short": np.sqrt(sg.sum(axis=1)),
        "fosm2_short": np.sqrt(sg.sum(axis=1) + cv.sum(axis=1)),
        "fosm2_bias": bias + KAPPA * np.sqrt(sg.sum(axis=1) + cv.sum(axis=1)),
        "soft": np.sqrt(((g_soft * sigma) ** 2).sum(axis=(1, 2))),
    }
    # --- Monte-Carlo truth -------------------------------------------------------------
    hd = Harness(
        scene,
        np.tile(poses[0], (N_DRAWS, 1)).astype(np.float32),
        np.zeros((omega.shape[0], N_DRAWS, 3), np.float32),
        device="cuda",
    )
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
        samples[:, k] = _cost(hd.forward(dilate=True))
    del hd

    mc_cvar = empirical_cvar(samples, ALPHA)
    true_risk = mc_cvar - j_bel
    return {
        "seed": seed,
        "taus": {k: kendall_tau(v, true_risk) for k, v in scores.items()},
        # calibration of the two arms that produce a risk in cost units
        "ratio1": float(np.mean(KAPPA * scores["fosm1_short"] / np.maximum(true_risk, 1e-6))),
        "ratio2": float(np.mean(KAPPA * scores["fosm2_short"] / np.maximum(true_risk, 1e-6))),
        "bias_pred": float(bias.mean()),
        "bias_true": float(np.mean(samples.mean(axis=0) - j_bel)),
    }


def report(rows: list[dict]) -> None:
    keys = list(rows[0]["taus"])
    print(f"\nn={len(rows)}  Kendall tau against the TRUE (MC) risk\n")
    print(f"{'score':<14}{'tau':>9}   what it is")
    what = {
        "time_sigma": "sum over TIMESTEPS of footprint sigma",
        "cell_sigma": "sum over CELLS of sigma, on the support (unweighted)",
        "fosm1_full": "first-order FOSM, full support",
        "fosm1_short": "first-order FOSM, shortlist only",
        "fosm2_short": "SECOND-order FOSM, same shortlist",
        "fosm2_bias": "second order incl. the E[J] bias term",
        "soft": "soft contact gradient at tau = sigma",
    }
    for k in keys:
        print(f"{k:<14}{np.mean([r['taus'][k] for r in rows]):>+9.3f}   {what[k]}")

    print("\nQ2 -- THE PAIRED DOMAIN TEST (per-cell minus per-timestep):")
    d = np.array([r["taus"]["cell_sigma"] - r["taus"]["time_sigma"] for r in rows])
    n, w, p = sign_test(d)
    print(f"  cell_sigma - time_sigma   {d.mean():>+7.3f}   better on {w:>3}/{n:<3}   p={p:.2e}")
    print("  (negative confirms section 7h: summing over CELLS is what inverts the ordering)")

    print("\n  weighting at a FIXED per-cell domain:")
    d = np.array([r["taus"]["fosm1_full"] - r["taus"]["cell_sigma"] for r in rows])
    n, w, p = sign_test(d)
    print(f"  fosm1_full - cell_sigma   {d.mean():>+7.3f}   better on {w:>3}/{n:<3}   p={p:.2e}")

    print("\nQ1 -- DOES SECOND ORDER RESCUE IT?")
    for a, b in (
        ("fosm2_short", "fosm1_short"),
        ("fosm2_bias", "fosm1_short"),
        ("soft", "fosm1_full"),
        ("fosm1_short", "fosm1_full"),
    ):
        d = np.array([r["taus"][a] - r["taus"][b] for r in rows])
        n, w, p = sign_test(d)
        print(f"  {a:<12} - {b:<12} {d.mean():>+7.3f}   better on {w:>3}/{n:<3}   p={p:.2e}")

    r1 = np.mean([r["ratio1"] for r in rows])
    r2 = np.mean([r["ratio2"] for r in rows])
    print(f"\n  calibration (est/true risk, 1.0 is perfect): first {r1:.2f}  second {r2:.2f}")
    bp = np.mean([r["bias_pred"] for r in rows])
    bt = np.mean([r["bias_true"] for r in rows])
    print(f"  E[J]-J(belief): predicted by curvature {bp:+.3f}  actual {bt:+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--family", default="hybrid")
    ap.add_argument("--noise", default="all")
    ap.add_argument("--element", default="sphere", choices=("sphere", "cylinder"))
    a = ap.parse_args()
    wp.init()
    rows = []
    for seed in range(a.seeds):
        rows.append(run_seed(seed, a.family, a.noise, a.element))
        if (seed + 1) % 10 == 0:
            print(f"  {seed + 1}/{a.seeds} seeds", flush=True)
    report(rows)
    tag = "" if a.element == "sphere" else f"_{a.element}"
    path = OUT / f"order2_{a.family}_{a.noise}{tag}.json"
    path.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
