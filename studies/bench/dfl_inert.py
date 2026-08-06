"""Decisive check: does the decision-trained model's LOW fill (measured bias -0.22, CLAIMS.md
"Follow-up direction validated") make unobserved cells INERT in the wheel-envelope contact
argmax -- so plans get ranked on observed ground -- or does the low fill change costs through
some other route while unobserved cells remain load-bearing?

    .venv/bin/python -m studies.bench.dfl_inert

No retraining: reuses the mse/decision weights `dfl.py`'s full run persisted in
studies/out/bench/dfl_full.json (same pattern as dfl_mechanism.py). For every fill model
(zero_fill, mse, decision, and a control "decision_shift" with the mean bias undone) on the 80
virgin test seeds (FAMILY/NOISE from dfl.py), assembles the filled map exactly as dfl.py's
evaluation does, then reads the per-cell adjoint SUPPORT (|grad| summed over the K plans, via
`bundled._weighted_adjoint`) as the cheap, exact proxy for "cells the costs actually rest on"
(contact/argmax cells + belly-sample cells). Reports the unobserved share of that support, in
mass and in cell count, overall and restricted to cells within 0.5m of the plan trajectories
(GEOM_RADIUS_M, same convention as dfl_mechanism.py), plus a per-seed correlation against that
seed's regret improvement (mse - decision) already recorded in dfl_full.json.
"""

from __future__ import annotations

import json
import time

import numpy as np
import warp as wp

from .bundled import _weighted_adjoint
from .dfl import _assemble
from .dfl import build_seed
from .dfl import FAMILY
from .dfl import model_fill
from .dfl import NOISE
from .dfl import SeedData
from .dfl import TEST_SEEDS
from .dfl import zero_fill as zero_fill_fn
from .ranking import build_case
from .ranking import OUT
from .ranking import _plan_distance

GEOM_RADIUS_M = 0.5  # same "decision-relevant" convention as dfl_mechanism.py
ADJ_SUPPORT_THRESHOLD = 1e-6  # same support-cliff threshold as dfl_mechanism.py

CONDITIONS = ("zero_fill", "mse", "decision", "decision_shift")


def _load_weights() -> dict[str, np.ndarray]:
    full = json.loads((OUT / "dfl_full.json").read_text())
    return {name: np.asarray(full["training"][name]["weights"]) for name in ("mse", "decision")}


def _geom_mask(s: SeedData) -> np.ndarray:
    """[ny, nx] bool: within GEOM_RADIUS_M of any of the K plans' trajectories, read off the
    truth-forward `build_seed` already left on `s.harness` (its last step is `_evaluate` on
    truth) -- exactly the pattern dfl_mechanism.py uses, so the plan geometry compared across
    fill models is the SAME (truth-based) reference, not re-derived per fill."""
    _scene, truth, _measured, observed, _sigma, _poses, _omega, grid = build_case(s.seed, FAMILY, NOISE)
    assert np.allclose(truth, s.truth) and np.array_equal(observed, s.observed), (
        "build_case(seed, FAMILY, NOISE) must reproduce build_seed's own truth/observed exactly"
    )
    dist = _plan_distance(s.harness, grid)  # [K, ny, nx]
    return dist.min(axis=0) <= GEOM_RADIUS_M


def _support(s: SeedData, fill: np.ndarray) -> np.ndarray:
    """[ny, nx]: |grad(cost)/d(elevation)| summed over the K plans, on the map assembled from
    `fill` -- the cheap, exact proxy for "cells the costs actually rest on" (contact/argmax +
    belly-sample cells), per dfl_mechanism.py's own convention."""
    elev = _assemble(s, fill)
    g_map, _costs = _weighted_adjoint(s.harness, elev)  # [K, ny, nx]
    return np.abs(g_map).sum(axis=0)


def _shares(support: np.ndarray, unobs: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """Given one seed's support field, return (mass fraction on unobserved, count fraction on
    unobserved), optionally restricted to `mask` (e.g. the geometric relevance band)."""
    m = mask if mask is not None else np.ones_like(unobs)
    sup_m, unobs_m = support[m], unobs[m]
    total_mass = float(sup_m.sum())
    unobs_mass = float(sup_m[unobs_m].sum())
    above = sup_m > ADJ_SUPPORT_THRESHOLD
    total_count = int(above.sum())
    unobs_count = int((above & unobs_m).sum())
    return {
        "total_mass": total_mass,
        "unobs_mass": unobs_mass,
        "total_count": total_count,
        "unobs_count": unobs_count,
    }


def _pool(per_seed: list[dict[str, float]]) -> dict[str, float]:
    total_mass = sum(d["total_mass"] for d in per_seed)
    unobs_mass = sum(d["unobs_mass"] for d in per_seed)
    total_count = sum(d["total_count"] for d in per_seed)
    unobs_count = sum(d["unobs_count"] for d in per_seed)
    return {
        "unobs_mass_frac": unobs_mass / total_mass if total_mass > 0 else float("nan"),
        "unobs_count_frac": unobs_count / total_count if total_count > 0 else float("nan"),
        "total_mass": total_mass,
        "total_count": total_count,
    }


def main() -> None:
    t0 = time.time()
    wp.init()
    OUT.mkdir(parents=True, exist_ok=True)

    weights = _load_weights()
    full = json.loads((OUT / "dfl_full.json").read_text())
    regret_improve = np.asarray(full["paired"]["decision"]["per_seed_diff"])  # mse - decision, TEST_SEEDS order

    print(f"building {len(TEST_SEEDS)} held-out seeds (family={FAMILY}, noise={NOISE}) ...")
    test_data = [build_seed(s) for s in TEST_SEEDS]
    print(f"  done in {time.time() - t0:.1f}s")

    # --- pass 1: fills, and the auto-computed bias shift for the control ------------------------
    fills = {
        "zero_fill": [zero_fill_fn(s) for s in test_data],
        "mse": [model_fill(s, weights["mse"]) for s in test_data],
        "decision": [model_fill(s, weights["decision"]) for s in test_data],
    }
    dec_resid = np.concatenate(
        [(fills["decision"][i] - s.truth)[s.unobs] for i, s in enumerate(test_data)]
    )
    bias_decision = float(dec_resid.mean())  # measured pooled bias of the decision fill, unobserved cells
    shift = -bias_decision  # amount that zeroes the pooled bias (undoes the "low fill" globally)
    fills["decision_shift"] = [f + shift for f in fills["decision"]]
    print(f"\nmeasured pooled decision-fill bias on unobserved cells: {bias_decision:+.4f} m")
    print(f"control shift applied to decision_shift: {shift:+.4f} m")

    # --- pass 2: geometric mask + support, per seed, per condition -----------------------------
    print("\ncomputing adjoint support per seed/condition ...")
    geom_masks = [_geom_mask(s) for s in test_data]  # uses each seed's truth-forward harness state

    per_seed_shares: dict[str, list[dict[str, float]]] = {c: [] for c in CONDITIONS}
    per_seed_geom_shares: dict[str, list[dict[str, float]]] = {c: [] for c in CONDITIONS}
    for i, s in enumerate(test_data):
        for cond in CONDITIONS:
            support = _support(s, fills[cond][i])
            per_seed_shares[cond].append(_shares(support, s.unobs))
            per_seed_geom_shares[cond].append(_shares(support, s.unobs, mask=geom_masks[i]))

    # --- report: (a)/(b) overall, (c) geometric-band-restricted, per condition -----------------
    table = {}
    for cond in CONDITIONS:
        pooled = _pool(per_seed_shares[cond])
        pooled_geom = _pool(per_seed_geom_shares[cond])
        table[cond] = {
            "unobs_mass_frac": pooled["unobs_mass_frac"],
            "unobs_count_frac": pooled["unobs_count_frac"],
            "geom_unobs_mass_frac": pooled_geom["unobs_mass_frac"],
            "geom_unobs_count_frac": pooled_geom["unobs_count_frac"],
        }

    print(f"\n{'model':<16}{'(a) mass unobs':>16}{'(b) count unobs':>18}{'(c) mass unobs <=0.5m':>24}")
    for cond in CONDITIONS:
        r = table[cond]
        print(
            f"{cond:<16}{r['unobs_mass_frac']:>16.3%}{r['unobs_count_frac']:>18.3%}"
            f"{r['geom_unobs_mass_frac']:>24.3%}"
        )

    # --- per-seed correlation: drop in unobserved-mass share (decision vs mse) vs regret gain --
    share_mse = np.asarray([d["unobs_mass"] / d["total_mass"] if d["total_mass"] > 0 else np.nan
                             for d in per_seed_shares["mse"]])
    share_decision = np.asarray([d["unobs_mass"] / d["total_mass"] if d["total_mass"] > 0 else np.nan
                                  for d in per_seed_shares["decision"]])
    share_drop = share_mse - share_decision  # positive: decision has a LOWER unobserved share than mse
    valid = np.isfinite(share_drop) & np.isfinite(regret_improve)
    corr = float(np.corrcoef(share_drop[valid], regret_improve[valid])[0, 1])
    print(
        f"\nper-seed corr(unobs-mass-share drop [mse-decision], regret improvement [mse-decision]): "
        f"r={corr:+.4f}  (n={int(valid.sum())})"
    )

    # --- shift control verdict -------------------------------------------------------------------
    a_decision = table["decision"]["unobs_mass_frac"]
    a_shift = table["decision_shift"]["unobs_mass_frac"]
    a_mse = table["mse"]["unobs_mass_frac"]
    a_zero = table["zero_fill"]["unobs_mass_frac"]
    print(
        f"\nshift control: decision unobs-mass share {a_decision:.3%} -> shifted +{shift:.3f}m "
        f"gives {a_shift:.3%}  (mse {a_mse:.3%}, zero_fill {a_zero:.3%})"
    )

    if a_decision < 0.6 * min(a_mse, a_zero) and a_shift > 0.8 * a_mse and corr > 0.15:
        verdict = "CONFIRMED"
    elif a_decision < min(a_mse, a_zero) and (a_shift > a_decision or corr > 0):
        verdict = "PARTIAL"
    else:
        verdict = "REFUTED"

    print(f"\nverdict: inert-fill hypothesis {verdict}")

    report = {
        "family": FAMILY,
        "noise": NOISE,
        "n_test": len(test_data),
        "geom_radius_m": GEOM_RADIUS_M,
        "adj_support_threshold": ADJ_SUPPORT_THRESHOLD,
        "bias_decision_pooled": bias_decision,
        "shift_applied_m": shift,
        "table": table,
        "per_seed": {
            cond: {
                "unobs_mass_frac": [
                    d["unobs_mass"] / d["total_mass"] if d["total_mass"] > 0 else None
                    for d in per_seed_shares[cond]
                ],
            }
            for cond in CONDITIONS
        },
        "regret_improvement_mse_minus_decision": regret_improve.tolist(),
        "correlation_share_drop_vs_regret_improvement": {"r": corr, "n": int(valid.sum())},
        "verdict": verdict,
        "seconds_total": time.time() - t0,
    }
    (OUT / "dfl_inert.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT / 'dfl_inert.json'}  ({report['seconds_total']:.1f}s total)")


if __name__ == "__main__":
    main()
