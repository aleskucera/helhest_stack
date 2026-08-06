"""Follow-up A+B on the decision-focused-training result (dfl.py / CLAIMS.md, 2026-08-06 evening):
does decision-trained beat MSE-trained by learning a shaped ("planning") bias, or is its 3x-worse
height RMSE just unshaped noise that happens to help on this plan set?

    .venv/bin/python -m studies.bench.dfl_mechanism

No retraining: this reuses the mse/cost_space/decision weights `dfl.py`'s full run persisted in
studies/out/bench/dfl_full.json, exactly as `dfl_ablation.py` does.

A. MECHANISM. Stratify the 80 virgin test seeds' unobserved cells into decision-relevant vs
   irrelevant two ways -- geometric (within 0.5m of one of the K hybrid-family plans) and
   adjoint-support (nonzero |grad| of `bundled._weighted_adjoint` on the belief, summed over the
   K plans) -- and report signed bias + RMSE of the fill error, per model, per stratum. The
   "planning map" hypothesis: decision-trained is specifically BIASED (upward/pessimistic) in
   relevant cells and no more accurate than MSE elsewhere. The null: its extra error is unshaped
   noise everywhere, equally.

B. TRANSFER. Evaluate the SAME already-trained weights on the SAME 80 seeds but under the `fan`
   plan family's K plans (ranking.PLAN_FAMILIES["fan"]) instead of the `hybrid` family used for
   training -- a family the models never saw a decision loss computed against. Since the features
   and the fill do not depend on plans at all (verified below: `build_features` takes no plan
   argument), only the LOSS saw `hybrid`'s plans during training; transfer failure would mean the
   loss carved plan-specific structure into terrain space rather than learning terrain-shaped
   caution.
"""

from __future__ import annotations

import json
import time

import numpy as np
import warp as wp

from ..adjoint.harness import Harness
from .bundled import _weighted_adjoint
from .dfl import build_seed
from .dfl import evaluate_method
from .dfl import FAMILY
from .dfl import model_fill
from .dfl import NOISE
from .dfl import SeedData
from .dfl import TEST_SEEDS
from .dfl import zero_fill
from .ranking import _evaluate
from .ranking import _plan_distance
from .ranking import build_case
from .ranking import OUT
from .ranking import sign_test

GEOM_RADIUS_M = 0.5  # "decision-relevant" = within this of any of the K plans (task spec)
# The point adjoint has a documented mm-scale validity cliff (CLAIMS.md): support is either
# exactly 0.0 (cell never reached by any plan's contact envelope) or O(1)-O(1e2). Measured on
# seed 100: 90% of unobserved cells are EXACTLY 0.0, the rest jump straight to >3e-5. Any
# threshold in that gap is equivalent; 1e-6 sits in it.
ADJ_SUPPORT_THRESHOLD = 1e-6
N_BOOTSTRAP = 1000
FAN_FAMILY = "fan"


def _load_weights() -> tuple[dict[str, np.ndarray], dict]:
    full = json.loads((OUT / "dfl_full.json").read_text())
    weights = {name: np.asarray(full["training"][name]["weights"]) for name in ("mse", "cost_space", "decision")}
    return weights, full


# --- part A: mechanism -----------------------------------------------------------------------


def _seed_strata_and_errors(
    s: SeedData, w_mse: np.ndarray, w_decision: np.ndarray
) -> dict[str, np.ndarray]:
    """One seed's pooled-cell arrays, restricted to unobserved cells: the two relevance masks,
    the adjoint-support mass, and both models' signed fill error."""
    _scene, truth, _measured, observed, _sigma, _poses, _omega, grid = build_case(s.seed, FAMILY, NOISE)
    assert np.allclose(truth, s.truth) and np.array_equal(observed, s.observed), (
        "build_case(seed, FAMILY, NOISE) must reproduce build_seed's own truth/observed exactly "
        "-- both derive them from `seed` directly, before the plan family consumes the rng."
    )

    # Geometric relevance: read off the trajectories cached from build_seed's own last forward
    # (on ground truth), matching `ranking._plan_distance`'s own "read the last forward" contract.
    dist = _plan_distance(s.harness, grid)  # [K, ny, nx]
    geom_relevant = dist.min(axis=0) <= GEOM_RADIUS_M

    # Adjoint support on the BELIEF (task spec): one weighted backward pass, |grad| over K plans.
    g_map, _costs = _weighted_adjoint(s.harness, s.belief.astype(np.float32))
    support = np.abs(g_map).sum(axis=0)  # [ny, nx]

    fill_mse = model_fill(s, w_mse)
    fill_decision = model_fill(s, w_decision)
    e_mse = fill_mse - s.truth
    e_decision = fill_decision - s.truth

    u = s.unobs
    return {
        "geom_relevant": geom_relevant[u],
        "adj_relevant": support[u] > ADJ_SUPPORT_THRESHOLD,
        "support": support[u],
        "e_mse": e_mse[u],
        "e_decision": e_decision[u],
    }


def _bias_rmse(e: np.ndarray) -> dict[str, float]:
    return {"bias": float(np.mean(e)), "rmse": float(np.sqrt(np.mean(e**2))), "n": int(e.size)}


def _stratum_table(mask: np.ndarray, e_mse: np.ndarray, e_decision: np.ndarray) -> dict:
    return {
        "relevant": {"mse": _bias_rmse(e_mse[mask]), "decision": _bias_rmse(e_decision[mask])},
        "irrelevant": {"mse": _bias_rmse(e_mse[~mask]), "decision": _bias_rmse(e_decision[~mask])},
    }


def part_a(test_data: list[SeedData], w_mse: np.ndarray, w_decision: np.ndarray) -> dict:
    print("\n=== A: mechanism -- per-seed strata + fill error ===")
    per_seed = [_seed_strata_and_errors(s, w_mse, w_decision) for s in test_data]

    geom_relevant = np.concatenate([r["geom_relevant"] for r in per_seed])
    adj_relevant = np.concatenate([r["adj_relevant"] for r in per_seed])
    support = np.concatenate([r["support"] for r in per_seed])
    e_mse = np.concatenate([r["e_mse"] for r in per_seed])
    e_decision = np.concatenate([r["e_decision"] for r in per_seed])
    n_cells = e_mse.size

    geom_table = _stratum_table(geom_relevant, e_mse, e_decision)
    adj_table = _stratum_table(adj_relevant, e_mse, e_decision)

    extra_err = np.abs(e_decision) - np.abs(e_mse)
    pooled_r = float(np.corrcoef(extra_err, support)[0, 1])

    # Bootstrap over SEEDS (cells within a seed are spatially correlated, so resampling cells
    # directly would understate the CI): resample the 80 seeds with replacement, recompute the
    # pooled correlation each time.
    rng = np.random.default_rng(0)
    n_seeds = len(per_seed)
    boot_r = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_seeds, n_seeds)
        ee = np.concatenate([np.abs(per_seed[i]["e_decision"]) - np.abs(per_seed[i]["e_mse"]) for i in idx])
        ss = np.concatenate([per_seed[i]["support"] for i in idx])
        boot_r[b] = np.corrcoef(ee, ss)[0, 1]
    ci_lo, ci_hi = float(np.percentile(boot_r, 2.5)), float(np.percentile(boot_r, 97.5))

    out = {
        "n_test": len(test_data),
        "n_unobs_cells_pooled": int(n_cells),
        "geom_radius_m": GEOM_RADIUS_M,
        "adj_support_threshold": ADJ_SUPPORT_THRESHOLD,
        "geom_relevant_frac": float(geom_relevant.mean()),
        "adj_relevant_frac": float(adj_relevant.mean()),
        "geom_stratified": geom_table,
        "adjoint_stratified": adj_table,
        "extra_error_vs_support_correlation": {
            "pearson_r": pooled_r,
            "bootstrap_ci_95": [ci_lo, ci_hi],
            "n_bootstrap": N_BOOTSTRAP,
        },
    }

    def _print_table(name: str, table: dict) -> None:
        print(f"\n  -- stratified by {name} --")
        print(f"  {'stratum':<12}{'model':<10}{'n':>8}{'bias':>10}{'rmse':>10}")
        for stratum in ("relevant", "irrelevant"):
            for model in ("mse", "decision"):
                r = table[stratum][model]
                print(f"  {stratum:<12}{model:<10}{r['n']:>8}{r['bias']:>+10.4f}{r['rmse']:>10.4f}")

    _print_table("GEOMETRIC (<=0.5m of a plan)", geom_table)
    print(f"  relevant fraction of unobserved cells: {out['geom_relevant_frac']:.3%}")
    _print_table("ADJOINT SUPPORT (|grad| sum > 1e-6)", adj_table)
    print(f"  relevant fraction of unobserved cells: {out['adj_relevant_frac']:.3%}")
    print(
        f"\n  corr(|extra error| of decision vs mse, adjoint-support mass), pooled over "
        f"{n_cells} cells: r={pooled_r:+.4f}  (95% seed-bootstrap CI [{ci_lo:+.4f}, {ci_hi:+.4f}])"
    )
    return out


# --- part B: transfer to the fan plan family --------------------------------------------------


def _build_fan_variant(s: SeedData) -> SeedData:
    """Same seed, same features/fill (plan-independent -- verified: `build_features` takes no
    plan argument), but a fresh Harness/true-cost built from the FAN family's K plans instead of
    the hybrid family `dfl.py` trained/evaluated against."""
    scene, truth, _measured, observed, _sigma, poses, omega, _grid = build_case(s.seed, FAN_FAMILY, NOISE)
    assert np.allclose(truth, s.truth) and np.array_equal(observed, s.observed), (
        "fan build_case must reproduce the same truth/observed as the hybrid one for this seed"
    )
    harness = Harness(scene, poses, omega, device="cuda")
    j_true = _evaluate(harness, truth.astype(np.float32))
    return SeedData(s.seed, harness, s.truth, s.belief, s.observed, s.unobs, s.phi, j_true)


def part_b(test_data: list[SeedData], weights: dict[str, np.ndarray]) -> dict:
    print("\n=== B: transfer to the FAN plan family (no retraining) ===")
    t0 = time.time()
    fan_data = [_build_fan_variant(s) for s in test_data]
    print(f"  built {len(fan_data)} fan-family harnesses in {time.time() - t0:.1f}s")

    methods = {
        "zero_fill": zero_fill,
        "mse": lambda s: model_fill(s, weights["mse"]),
        "cost_space": lambda s: model_fill(s, weights["cost_space"]),
        "decision": lambda s: model_fill(s, weights["decision"]),
    }
    fan_eval = {name: evaluate_method(fan_data, fn) for name, fn in methods.items()}

    print(f"\n  {'method':<12}{'regret':>10}{'tau':>10}   (fan-family plans)")
    for name in methods:
        r = fan_eval[name]
        print(f"  {name:<12}{np.mean(r['regret']):>10.4f}{np.mean(r['tau']):>+10.3f}")

    per_seed_diff = np.asarray(fan_eval["mse"]["regret"]) - np.asarray(fan_eval["decision"]["regret"])
    n, k, p = sign_test(per_seed_diff)
    mean_improve = float(per_seed_diff.mean())
    print(
        f"\n  paired sign test, decision vs mse regret on fan plans (positive = decision wins): "
        f"mean {mean_improve:+.4f}  median {np.median(per_seed_diff):+.4f}  wins {k}/{n} (ties dropped)  p={p:.3g}"
    )

    if p < 0.05 and mean_improve > 0:
        verdict = (
            "decision-trained still beats MSE-trained on fan-family plan choice (p<0.05): the "
            "learned fill generalises across plan families -- it learned terrain-shaped caution, "
            "not a hybrid-plan-specific trick."
        )
    else:
        verdict = (
            "decision-trained's edge over MSE-trained vanished or reversed on fan-family plan "
            "choice: the advantage is plan-family-specific and must be scoped to the hybrid "
            "family the loss was trained against."
        )
    print(f"\n  verdict: {verdict}")

    return {
        "family": FAN_FAMILY,
        "noise": NOISE,
        "n_test": len(fan_data),
        "eval": fan_eval,
        "paired_decision_vs_mse": {
            "mean_improvement": mean_improve,
            "median_improvement": float(np.median(per_seed_diff)),
            "n_nonzero": n,
            "wins": k,
            "p": p,
        },
        "verdict": verdict,
    }


def main() -> None:
    t0 = time.time()
    wp.init()
    OUT.mkdir(parents=True, exist_ok=True)

    weights, full = _load_weights()
    print(f"loaded trained weights from dfl_full.json (mse={weights['mse']}, decision={weights['decision']})")
    print(f"reference (dfl_full.json, hybrid family): mse regret {np.mean(full['eval']['mse']['regret']):.4f}, "
          f"decision regret {np.mean(full['eval']['decision']['regret']):.4f}")

    print(f"\nbuilding {len(TEST_SEEDS)} held-out seeds (family={FAMILY}, noise={NOISE}) ...")
    test_data = [build_seed(s) for s in TEST_SEEDS]
    print(f"  done in {time.time() - t0:.1f}s")

    report: dict = {
        "family": FAMILY,
        "noise": NOISE,
        "n_test": len(test_data),
        "weights": {k: v.tolist() for k, v in weights.items()},
        "hybrid_reference": {
            "mse_regret_mean": float(np.mean(full["eval"]["mse"]["regret"])),
            "decision_regret_mean": float(np.mean(full["eval"]["decision"]["regret"])),
        },
    }
    report["part_a_mechanism"] = part_a(test_data, weights["mse"], weights["decision"])
    report["part_b_transfer"] = part_b(test_data, weights)
    report["seconds_total"] = time.time() - t0

    (OUT / "dfl_mechanism.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT / 'dfl_mechanism.json'}  ({report['seconds_total']:.1f}s total)")


if __name__ == "__main__":
    main()
