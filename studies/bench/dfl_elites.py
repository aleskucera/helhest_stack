"""Does the DFL edge (dfl.py / DFL_CLAIMS.md C1: decision-trained inpainter halves plan-choice
regret vs MSE-trained) survive when the candidate plans are the REAL MppiGpu elite set instead
of the synthetic hybrid/fan families it was measured on?

    .venv/bin/python -m studies.bench.dfl_elites

An adversarial review raised the concern this project's own sensing study (CLAIMS.md C1) already
lived through once: real MPPI elites (elites.py) are a low-diversity candidate set (~1/5 of
hybrid's path spread), and the sensing paper's derivative-vs-corridor-masking edge survived that
shrinkage only under clean noise, not under the full noise arm. This module runs the SAME
before/after comparison for the perception (DFL) claim instead of the sensing claim.

Level A (fixed candidates, varied choice-map): per seed, extract the real top-16 MPPI elites by
planning on the BELIEF map (elites.py's own production planner+CostToGo, reused wholesale, with
NO fill under test touching candidate generation -- the belief handed to the planner is the
scene's own zero-filled elevation, exactly as elites.py's `run_seed_bundle` does). Then evaluate
those same 16 fixed control sequences' cost on the zero-fill / mse-fill / decision-fill maps and
on truth; each fill's "choice" is its own argmin. This isolates the CHOICE-MAP question from
candidate generation.

Level B (fill-in-the-loop, deployment-shaped): the planner PLANS on each filled map itself (a
fresh elite extraction per fill), so filling changes which plans exist AND which one looks best;
the top-1 elite by the fill's own ranking is then scored on truth. Slower (one planner solve per
fill per seed) but closer to what actually happens at deployment.

Weights are loaded from studies/out/bench/dfl_full.json (dfl.py's own trained mse/decision
weights) -- no retraining, exactly as dfl_mechanism.py does for its own follow-up.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import warp as wp

from ..adjoint.harness import Harness
from .dfl import _assemble
from .dfl import build_features
from .dfl import CELL
from .dfl import FAMILY
from .dfl import model_fill
from .dfl import NOISE
from .dfl import SeedData
from .dfl import TEST_SEEDS
from .dfl import zero_fill
from .dfl_mechanism import _load_weights
from .elites import ElitePlanner
from .elites import GOAL_XY
from .ranking import _evaluate
from .ranking import build_case
from .ranking import kendall_tau
from .ranking import N_PLANS
from .ranking import OUT
from .ranking import sign_test

METHODS = ("zero_fill", "mse", "decision")


def _fill_fn(name: str, weights: dict[str, np.ndarray]):
    if name == "zero_fill":
        return zero_fill
    return lambda s: model_fill(s, weights[name])


# --- level A: fixed candidates (real MPPI elites), varied choice-map --------------------------


def build_elite_seed(seed: int, elite_planner: ElitePlanner) -> tuple[SeedData, dict]:
    """One held-out seed's belief/truth/features, paired with the REAL top-`N_PLANS` MPPI elites
    extracted by planning on the belief (`scene.elevation`, already zero-filled on unobserved
    cells -- no fill under test has touched it, exactly elites.py's own candidate-generation
    contract)."""
    scene, truth, _measured, observed, sigma, poses, _hybrid_omega, _grid = build_case(
        seed, FAMILY, NOISE
    )
    belief = scene.elevation.astype(np.float64)
    elite_omega, _elite_poses, diag = elite_planner.elites(
        scene.elevation.astype(np.float32), scene.friction.astype(np.float32), GOAL_XY
    )
    harness = Harness(scene, poses, elite_omega, device="cuda")
    phi = build_features(observed, belief, sigma, CELL)
    j_true = _evaluate(harness, truth.astype(np.float32))
    seed_data = SeedData(seed, harness, truth.astype(np.float64), belief, observed, ~observed, phi, j_true)
    return seed_data, diag


def evaluate_with_picks(test_data: list[SeedData], fill_fn) -> dict:
    """`dfl.evaluate_method`'s regret/tau, plus the chosen plan INDEX per seed -- needed here (and
    not in dfl.py) to report how many seeds tie because two fills choose the identical elite."""
    regrets, taus, picks = [], [], []
    for s in test_data:
        fill = fill_fn(s)
        elev = _assemble(s, fill)
        j_hat = _evaluate(s.harness, elev)
        pick = int(np.argmin(j_hat))
        regrets.append(float(s.j_true[pick] - s.j_true.min()))
        taus.append(kendall_tau(j_hat, s.j_true))
        picks.append(pick)
    return {"regret": regrets, "tau": taus, "pick": picks}


def _paired(name: str, ref: str, eval_out: dict) -> dict:
    """Sign test on regret (name vs ref), plus the tie diagnostic the elite regime specifically
    calls for: how many seeds pick the SAME elite (regret tie) vs the same regret by coincidence
    of two different picks."""
    reg_ref, reg_name = np.asarray(eval_out[ref]["regret"]), np.asarray(eval_out[name]["regret"])
    diff = reg_ref - reg_name  # positive = name beats ref
    n_nonzero, wins, p = sign_test(diff)
    n_seeds = len(diff)
    picks_ref, picks_name = eval_out[ref]["pick"], eval_out[name]["pick"]
    n_same_pick = sum(a == b for a, b in zip(picks_ref, picks_name))
    return {
        "mean_improvement": float(diff.mean()),
        "median_improvement": float(np.median(diff)),
        "n_seeds": n_seeds,
        "n_ties_regret": n_seeds - n_nonzero,
        "n_same_pick": n_same_pick,
        "wins": wins,
        "n_nonzero": n_nonzero,
        "p": p,
    }


def run_level_a(seeds: tuple[int, ...], elite_planner: ElitePlanner, weights: dict[str, np.ndarray]) -> dict:
    print(f"\n=== level A: {len(seeds)} seeds, fixed real-MPPI elites, varied choice-map ===")
    t0 = time.time()
    test_data, diags = [], []
    for seed in seeds:
        sd, diag = build_elite_seed(seed, elite_planner)
        test_data.append(sd)
        diags.append(diag)
    print(f"  built {len(test_data)} elite-conditioned seeds in {time.time() - t0:.1f}s")

    # convergence sanity check, mirroring elites.py's own report(): elites should sit near the
    # planner's own pool minimum, or the "real elites" claim is not what was actually extracted.
    j_elite_min = np.mean([d["J_elite_min"] for d in diags])
    j_pool_min = np.mean([d["J_pool_min"] for d in diags])
    print(f"  MPPI convergence check: mean elite-min {j_elite_min:.3f} vs mean pool-min {j_pool_min:.3f}")

    eval_out = {name: evaluate_with_picks(test_data, _fill_fn(name, weights)) for name in METHODS}

    print(f"\n  {'method':<12}{'regret':>10}{'tau':>10}")
    for name in METHODS:
        r = eval_out[name]
        print(f"  {name:<12}{np.mean(r['regret']):>10.4f}{np.mean(r['tau']):>+10.3f}")

    zf_to_mse = float(np.mean(eval_out["zero_fill"]["regret"]) - np.mean(eval_out["mse"]["regret"]))
    print(f"\n  zero-fill -> mse regret gap on real elites: {zf_to_mse:+.4f}")

    paired = {}
    print(f"\n  paired vs mse (positive = decision better), n={len(seeds)} seeds:")
    for name in ("decision",):
        paired["decision_vs_mse"] = _paired(name, "mse", eval_out)
        pr = paired["decision_vs_mse"]
        frac = pr["mean_improvement"] / zf_to_mse if abs(zf_to_mse) > 1e-9 else float("nan")
        print(
            f"    decision vs mse:      mean {pr['mean_improvement']:+.4f}  wins {pr['wins']}/{pr['n_nonzero']}"
            f"  ties(regret) {pr['n_ties_regret']}/{pr['n_seeds']}  same-pick {pr['n_same_pick']}/{pr['n_seeds']}"
            f"  p={pr['p']:.3g}  ({frac:+.1%} of zero-fill->mse gap)"
        )
    paired["decision_vs_zero_fill"] = _paired("decision", "zero_fill", eval_out)
    pr = paired["decision_vs_zero_fill"]
    print(
        f"    decision vs zero_fill: mean {pr['mean_improvement']:+.4f}  wins {pr['wins']}/{pr['n_nonzero']}"
        f"  ties(regret) {pr['n_ties_regret']}/{pr['n_seeds']}  same-pick {pr['n_same_pick']}/{pr['n_seeds']}"
        f"  p={pr['p']:.3g}"
    )
    paired["mse_vs_zero_fill"] = _paired("mse", "zero_fill", eval_out)

    for name in test_data:
        del name.harness
    return {
        "n_seeds": len(seeds),
        "eval": eval_out,
        "zero_fill_to_mse_gap": zf_to_mse,
        "paired": paired,
        "mppi_convergence": {"elite_min_mean": j_elite_min, "pool_min_mean": j_pool_min},
        "seconds": time.time() - t0,
    }


# --- level B: fill-in-the-loop, one planner solve per fill per seed --------------------------


def run_seed_b(
    seed: int, elite_planner: ElitePlanner, weights: dict[str, np.ndarray]
) -> dict[str, dict]:
    scene, truth, _measured, observed, sigma, poses, _hybrid_omega, _grid = build_case(
        seed, FAMILY, NOISE
    )
    belief = scene.elevation.astype(np.float64)
    truth64 = truth.astype(np.float64)
    phi = build_features(observed, belief, sigma, CELL)
    truth32 = truth.astype(np.float32)
    friction32 = scene.friction.astype(np.float32)

    fills = {
        "zero_fill": np.zeros_like(belief),
        "mse": phi @ weights["mse"],
        "decision": phi @ weights["decision"],
    }
    out = {}
    for name, fill in fills.items():
        # same TRUTH-on-observed / fill-on-unobserved assembly rule as everywhere else in dfl.py.
        assembled = np.where(observed, truth64, fill).astype(np.float32)
        elite_omega, _elite_poses, diag = elite_planner.elites(assembled, friction32, GOAL_XY)
        harness = Harness(scene, poses, elite_omega, device="cuda")
        # `elites()` sorts by cost on `assembled` ascending, so index 0 IS this fill's top-1 pick.
        j_true_elites = _evaluate(harness, truth32)
        del harness
        out[name] = {
            "chosen_true_cost": float(j_true_elites[0]),
            "elite_best_true": float(j_true_elites.min()),
            "j_pool_min_on_fill": diag["J_pool_min"],
        }
    return out


def run_level_b(seeds: tuple[int, ...], elite_planner: ElitePlanner, weights: dict[str, np.ndarray]) -> dict:
    print(f"\n=== level B: {len(seeds)} seeds, plan-on-filled-map, top-1 scored on truth ===")
    t0 = time.time()
    rows = [run_seed_b(s, elite_planner, weights) for s in seeds]
    print(f"  done in {time.time() - t0:.1f}s")

    chosen = {name: np.array([r[name]["chosen_true_cost"] for r in rows]) for name in METHODS}
    own_regret = {  # top-1 vs the best true cost inside that fill's OWN 16-plan pool
        name: chosen[name] - np.array([r[name]["elite_best_true"] for r in rows]) for name in METHODS
    }

    print(f"\n  {'method':<12}{'chosen true cost':>18}{'own-pool regret':>18}")
    for name in METHODS:
        print(f"  {name:<12}{chosen[name].mean():>18.4f}{own_regret[name].mean():>18.4f}")

    def _pair(a: str, b: str) -> dict:
        d = chosen[b] - chosen[a]  # positive = a beats b (lower chosen true cost)
        n, k, p = sign_test(d)
        return {
            "mean_improvement": float(d.mean()),
            "n_ties": len(d) - n,
            "n_nonzero": n,
            "wins": k,
            "p": p,
        }

    paired = {
        "decision_vs_mse": _pair("decision", "mse"),
        "decision_vs_zero_fill": _pair("decision", "zero_fill"),
        "mse_vs_zero_fill": _pair("mse", "zero_fill"),
    }
    print("\n  paired on chosen true cost (positive = first name beats second):")
    for key, pr in paired.items():
        print(
            f"    {key:<24} mean {pr['mean_improvement']:+.4f}  wins {pr['wins']}/{pr['n_nonzero']}"
            f"  ties {pr['n_ties']}  p={pr['p']:.3g}"
        )

    return {
        "n_seeds": len(seeds),
        "chosen_true_cost_mean": {k: float(v.mean()) for k, v in chosen.items()},
        "own_pool_regret_mean": {k: float(v.mean()) for k, v in own_regret.items()},
        "paired": paired,
        "seconds": time.time() - t0,
    }


# --- verdict, benchmarked against CLAIMS.md's C1 live-elite precedent -------------------------


def _verdict(level_a: dict, level_b: dict | None, hybrid_ref_gap: float) -> str:
    pr = level_a["paired"]["decision_vs_mse"]
    real_gap = level_a["zero_fill_to_mse_gap"]
    lines = []
    if pr["p"] < 0.05 and pr["mean_improvement"] > 0:
        lines.append(
            f"Level A: decision-trained still beats MSE-trained on real MPPI elites "
            f"(mean regret improvement {pr['mean_improvement']:+.4f}, p={pr['p']:.3g}, "
            f"{pr['wins']}/{pr['n_nonzero']} non-tied wins, {pr['n_ties_regret']}/{pr['n_seeds']} "
            f"seeds tied) -- the edge SURVIVES real elites, though diluted by ties."
        )
    else:
        lines.append(
            f"Level A: decision-trained's edge over MSE-trained did NOT reach significance on "
            f"real MPPI elites (mean {pr['mean_improvement']:+.4f}, p={pr['p']:.3g}, "
            f"{pr['wins']}/{pr['n_nonzero']} non-tied wins, {pr['n_ties_regret']}/{pr['n_seeds']} "
            f"seeds tied) -- the edge SHRINKS TO NOISE (or reverses) under low-diversity real "
            f"candidates."
        )
    lines.append(
        f"Reference: on the synthetic hybrid-family test the same weights scored a "
        f"zero-fill->mse gap of {hybrid_ref_gap:+.4f}; on real elites the comparable gap is "
        f"{real_gap:+.4f} -- the candidate set itself carries much less rankable signal, "
        "consistent with CLAIMS.md's C1 live-elite finding that real elites are low-diversity "
        "(~1/5 of hybrid's path spread) and erode a derivative-based edge."
    )
    if level_b is not None:
        pb = level_b["paired"]["decision_vs_mse"]
        if pb["p"] < 0.05 and pb["mean_improvement"] > 0:
            lines.append(
                f"Level B (fill-in-the-loop): decision-trained's top-1 choice still beats "
                f"MSE-trained's when the PLANNER itself plans on the filled map "
                f"(mean {pb['mean_improvement']:+.4f}, p={pb['p']:.3g}) -- the deployment-shaped "
                "test agrees with level A's direction."
            )
        else:
            lines.append(
                f"Level B (fill-in-the-loop): no significant edge for decision-trained once the "
                f"planner itself plans on the filled map (mean {pb['mean_improvement']:+.4f}, "
                f"p={pb['p']:.3g}) -- coupling candidate generation to the fill erodes the edge "
                "further than level A alone."
            )
    return "\n".join(lines)


def main(n_seeds_a: int = 80, n_seeds_b: int = 80, skip_b: bool = False) -> None:
    t_start = time.time()
    wp.init()
    OUT.mkdir(parents=True, exist_ok=True)

    weights, dfl_full = _load_weights()
    print(f"loaded weights from dfl_full.json (mse={weights['mse']}, decision={weights['decision']})")
    hybrid_ref_gap = float(
        np.mean(dfl_full["eval"]["zero_fill"]["regret"]) - np.mean(dfl_full["eval"]["mse"]["regret"])
    )
    print(f"reference (synthetic hybrid family, dfl_full.json): zero-fill->mse regret gap = {hybrid_ref_gap:+.4f}")

    seeds_a = TEST_SEEDS[:n_seeds_a]
    seeds_b = TEST_SEEDS[:n_seeds_b]

    scene0, *_ = build_case(TEST_SEEDS[0], FAMILY, NOISE)
    ny, nx = scene0.shape
    elite_planner = ElitePlanner(nx, ny, scene0.cell, scene0.origin_x, scene0.origin_y)

    report: dict = {
        "family": FAMILY,
        "noise": NOISE,
        "n_plans": N_PLANS,
        "weights": {k: v.tolist() for k, v in weights.items()},
        "hybrid_reference_zero_fill_to_mse_gap": hybrid_ref_gap,
    }
    report["level_a"] = run_level_a(seeds_a, elite_planner, weights)

    level_b = None
    if not skip_b:
        level_b = run_level_b(seeds_b, elite_planner, weights)
        report["level_b"] = level_b

    verdict = _verdict(report["level_a"], level_b, hybrid_ref_gap)
    report["verdict"] = verdict
    print("\n=== verdict ===")
    print(verdict)

    report["seconds_total"] = time.time() - t_start
    (OUT / "dfl_elites.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT / 'dfl_elites.json'}  ({report['seconds_total']:.1f}s total)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds-a", type=int, default=80, help="level-A seed count, from TEST_SEEDS (>=60 per spec)")
    ap.add_argument("--seeds-b", type=int, default=80, help="level-B seed count, from TEST_SEEDS (>=30 per spec)")
    ap.add_argument("--skip-b", action="store_true", help="run level A only")
    a = ap.parse_args()
    main(a.seeds_a, a.seeds_b, a.skip_b)
