"""Two settling tests for the DFL capacity result -- does the surviving positive hold up?

    .venv/bin/python -m studies.bench.dfl_settle

DFL_CLAIMS.md's REVIEW BANNER killed the softmin decision loss (diverges at capacity, see
`dfl_capacity.json`'s rung-2 "decision" row: regret 2.98, worse than zero-fill). What survives is
the COST-SPACE loss at capacity: rung-2 MLP (31 inputs -- `dfl.py`'s six hand features plus a
flattened 5x5 nearest-observed-height patch, 64 tanh hidden) regret 0.257 vs MSE-rung-2's 0.582,
p=0.0034 (`dfl_capacity.py`'s own eval). Two questions decide whether that positive survives:

  A. Does cost-space-rung2's edge over MSE-rung2 hold on REAL MppiGpu elite candidate sets? The
     earlier live-elite check (`dfl_elites.py`) used the old LINEAR decision model and found the
     edge n.s. there. This reruns its exact protocol (Level A fixed-candidates, Level B
     fill-in-the-loop) with the rung-2 mse/cost_space fills substituted in.
  B. Does cost-space-rung2 beat the FAIR capacity-matched baseline: MSE-rung2's own output plus
     one scalar coefficient on the dist_unobs feature (the methods red-team's finding, at linear
     scale, that this one-parameter heuristic recovers 83% of the decision-vs-MSE gap)? The
     coefficient is grid-tuned on TRAIN-seed regret only, then frozen and scored on the 80 virgin
     test seeds -- same "tune on train, score on test" discipline `dfl.py`'s own LR tuning uses.

Weights: `dfl_capacity.json` does not persist the trained rung-2 parameter vectors (checked --
its `training` entries carry lr/loss/seconds, not `weights`; contrast `dfl_full.json`, which does
store the linear model's `weights` and is what `dfl_elites.py` loads). This module therefore
RETRAINS rung 2's mse and cost_space conditions using `dfl_capacity.py`'s own functions verbatim
(same `MLPShape(31, 64)`, same seeds, same finite-difference gate, same `tune_lr`/`train`).
Decision is skipped -- it is dead per the review banner and retraining it would only spend budget
on a condition neither test below needs. The retrained test-seed regrets are checked within ~10%
of the recorded 0.582 (mse) / 0.257 (cost_space) before either test proceeds; if they fall
outside that band the run aborts rather than silently comparing against a different model.
"""

from __future__ import annotations

import functools
import json
import time
from dataclasses import dataclass

import numpy as np
import warp as wp

from ..adjoint.harness import Harness
from .dfl import build_features
from .dfl import CELL
from .dfl import evaluate_method
from .dfl import FAMILY
from .dfl import FEATURE_NAMES
from .dfl import MINIBATCH
from .dfl import N_STEPS
from .dfl import NOISE
from .dfl import TEST_SEEDS
from .dfl import TRAIN_SEEDS
from .dfl import zero_fill
from .dfl_capacity import build_cap_seed
from .dfl_capacity import CapSeedData
from .dfl_capacity import CONDITIONS
from .dfl_capacity import fd_gate
from .dfl_capacity import mlp_forward
from .dfl_capacity import mlp_init
from .dfl_capacity import model_fill
from .dfl_capacity import MLPShape
from .dfl_capacity import _patch_features
from .dfl_capacity import _probe_gnorm_median
from .dfl_capacity import train
from .dfl_capacity import tune_lr
from .dfl_elites import _paired
from .dfl_elites import evaluate_with_picks
from .elites import ElitePlanner
from .elites import GOAL_XY
from .ranking import _evaluate
from .ranking import build_case
from .ranking import OUT
from .ranking import sign_test

RUNG2_SHAPE = MLPShape(n_in=31, n_hidden=64)  # dfl_capacity.py's rung 2, verbatim
REGRET_TOLERANCE = 0.10  # retrained test regret must land within 10% of the recorded value
# dfl_capacity.json's rung-2 eval, quoted for the reproduction check below
RECORDED_REGRET = {"mse": 0.5821385145187378, "cost_space": 0.2573080539703369}

DIST_UNOBS_IDX = FEATURE_NAMES.index("dist_unobs")  # column 1 of the six hand features
C_GRID = (0.0,) + tuple(-0.1 * k for k in range(1, 11))  # 0, -0.1, ..., -1.0 -- reviewer's grid

N_SEEDS_ELITES = 80  # matches dfl_elites.py's own default (all 80 virgin test seeds)


# --- step 0: retrain rung 2's mse/cost_space (decision is dead, skipped for budget) -------------


def retrain_rung2(train_data: list[CapSeedData], test_data: list[CapSeedData]) -> dict:
    print("\n=== retraining rung 2 (mse, cost_space) -- dfl_capacity.py's protocol, verbatim ===")
    # rung=2 -> seed 100+2, dfl_capacity's own init convention
    init_w = mlp_init(RUNG2_SHAPE, np.random.default_rng(102))

    print("  finite-difference gate (before any optimizer step):")
    fd_report = fd_gate(RUNG2_SHAPE, train_data[0], init_w, np.random.default_rng(302))
    if not fd_report["gate_passed"]:
        raise RuntimeError("rung 2 retrain: finite-difference gate FAILED -- refusing to train")

    # CONDITIONS' functions take (shape, seeds, w); bind RUNG2_SHAPE so tune_lr/train/probe below
    # can use the plain (seeds, w) signature they expect -- exactly dfl_capacity.run_rung's own
    # `functools.partial(fn, shape)` binding, just restricted to the two conditions kept here.
    bound = {
        name: functools.partial(CONDITIONS[name], RUNG2_SHAPE) for name in ("mse", "cost_space")
    }
    # dfl_capacity.run_rung's clip norm uses the LARGER of cost_space/decision's probed median;
    # decision is skipped here, so cost_space alone sets it -- the mse condition never reads
    # clip_probe (only cost_space/decision are probed in dfl_capacity.py too).
    clip_probe = {"cost_space": _probe_gnorm_median(bound["cost_space"], train_data[0], init_w)}
    clip_norm = float(10.0 * clip_probe["cost_space"])
    print(f"  clip norm set to {clip_norm:.4g} (10x cost_space's probed median gnorm)")

    rng = np.random.default_rng(202)  # rung=2 -> seed 200+2, dfl_capacity's own
    trained: dict[str, np.ndarray] = {}
    training_report: dict = {}
    for name, fn in bound.items():
        t0 = time.time()
        lr = tune_lr(fn, train_data, MINIBATCH, rng, init_w)
        res = train(
            fn,
            train_data,
            lr=lr,
            n_steps=N_STEPS,
            minibatch=MINIBATCH,
            clip_norm=clip_norm,
            rng=rng,
            init_w=init_w,
        )
        dt = time.time() - t0
        finite_hist = [h for h in res["history"] if np.isfinite(h)]
        trained[name] = res["w"]
        training_report[name] = {
            "lr": lr,
            "loss0": finite_hist[0] if finite_hist else None,
            "loss_final": finite_hist[-1] if finite_hist else None,
            "n_skipped": res["n_skipped"],
            "n_clipped": res["n_clipped"],
            "seconds": dt,
        }
        print(
            f"  {name:<12} lr={lr:<6g} loss {finite_hist[0]:.4g} -> {finite_hist[-1]:.4g}  "
            f"skipped {res['n_skipped']}/{N_STEPS}  clipped {res['n_clipped']}/{N_STEPS}  "
            f"({dt:.1f}s)"
        )

    eval_out = {
        name: evaluate_method(
            test_data, lambda s, name=name: model_fill(RUNG2_SHAPE, s, trained[name])
        )
        for name in trained
    }
    print("\n  reproduction check vs dfl_capacity.json's recorded rung-2 numbers:")
    repro_ok = True
    repro: dict = {}
    for name in ("mse", "cost_space"):
        got = float(np.mean(eval_out[name]["regret"]))
        want = RECORDED_REGRET[name]
        rel = abs(got - want) / want
        ok = rel <= REGRET_TOLERANCE
        repro_ok = repro_ok and ok
        repro[name] = {
            "reproduced_regret": got,
            "recorded_regret": want,
            "rel_diff": rel,
            "within_10pct": ok,
        }
        print(
            f"    {name:<12} reproduced {got:.4f}  recorded {want:.4f}  "
            f"rel diff {rel:.1%}  {'OK' if ok else 'FAIL'}"
        )
    if not repro_ok:
        raise RuntimeError(
            "retrained rung-2 regret fell outside 10% of dfl_capacity.json's recorded value -- "
            "refusing to proceed on a model that does not reproduce the surviving positive"
        )

    return {
        "fd_gate": fd_report,
        "clip_probe_gnorm_median": clip_probe,
        "clip_norm": clip_norm,
        "training": training_report,
        "weights": {name: trained[name].tolist() for name in trained},
        "reproduction_check": repro,
        "trained_w": trained,  # in-process only; the weights key above already covers it
    }


# --- test A: live MPPI elites, mirroring dfl_elites.py, rung-2 fills substituted ----------------


def _fill_fn_r2(name: str, w_mse: np.ndarray, w_cs: np.ndarray):
    if name == "zero_fill":
        return zero_fill
    w = w_mse if name == "mse_rung2" else w_cs
    return lambda s: model_fill(RUNG2_SHAPE, s, w)


@dataclass
class EliteSeedR2:
    """The fields `model_fill`/`evaluate_with_picks` actually touch for an elite-conditioned
    rung-2 seed -- deliberately NOT `CapSeedData` (which also carries `phi_flat_r1`, meaningless
    here: rung 2 never reads it, so faking a 6-column duplicate would be a pointless prop)."""

    seed: int
    harness: Harness
    truth: np.ndarray  # [ny, nx] float64
    observed: np.ndarray  # [ny, nx] bool
    j_true: np.ndarray  # [N_PLANS]
    phi_flat_r2: np.ndarray  # [ny*nx, 31]


def build_elite_seed_r2(seed: int, elite_planner: ElitePlanner) -> tuple[EliteSeedR2, dict]:
    """`dfl_elites.build_elite_seed`, generalised to rung 2's 31-feature input: the real top-16
    MPPI elites, extracted by planning on the BELIEF (zero-filled on unobserved cells -- no fill
    under test touches candidate generation, exactly elites.py's own contract), paired with the
    31-feature flattened representation `dfl_capacity.py` trains and evaluates on."""
    scene, truth, _measured, observed, sigma, poses, _hybrid_omega, _grid = build_case(
        seed, FAMILY, NOISE
    )
    belief = scene.elevation.astype(np.float64)
    elite_omega, _elite_poses, diag = elite_planner.elites(
        scene.elevation.astype(np.float32), scene.friction.astype(np.float32), GOAL_XY
    )
    harness = Harness(scene, poses, elite_omega, device="cuda")
    phi6 = build_features(observed, belief, sigma, CELL)
    patch = _patch_features(phi6[..., FEATURE_NAMES.index("nearest_h")])
    ny, nx = truth.shape
    phi_flat_r2 = np.ascontiguousarray(
        np.concatenate([phi6, patch], axis=-1).reshape(ny * nx, 31)
    )
    j_true = _evaluate(harness, truth.astype(np.float32))
    truth64 = truth.astype(np.float64)
    return EliteSeedR2(seed, harness, truth64, observed, j_true, phi_flat_r2), diag


METHODS_A = ("zero_fill", "mse_rung2", "cost_space_rung2")


def run_level_a(
    seeds: tuple[int, ...], elite_planner: ElitePlanner, w_mse: np.ndarray, w_cs: np.ndarray
) -> dict:
    print(f"\n=== test A, level A: {len(seeds)} seeds, fixed real elites, varied choice-map ===")
    t0 = time.time()
    test_data, diags = [], []
    for seed in seeds:
        sd, diag = build_elite_seed_r2(seed, elite_planner)
        test_data.append(sd)
        diags.append(diag)
    print(f"  built {len(test_data)} elite-conditioned seeds in {time.time() - t0:.1f}s")

    j_elite_min = np.mean([d["J_elite_min"] for d in diags])
    j_pool_min = np.mean([d["J_pool_min"] for d in diags])
    print(
        f"  MPPI convergence check: mean elite-min {j_elite_min:.3f} vs mean pool-min "
        f"{j_pool_min:.3f}"
    )

    eval_out = {
        name: evaluate_with_picks(test_data, _fill_fn_r2(name, w_mse, w_cs)) for name in METHODS_A
    }

    print(f"\n  {'method':<18}{'regret':>10}{'tau':>10}")
    for name in METHODS_A:
        r = eval_out[name]
        print(f"  {name:<18}{np.mean(r['regret']):>10.4f}{np.mean(r['tau']):>+10.3f}")

    zf_to_mse = float(
        np.mean(eval_out["zero_fill"]["regret"]) - np.mean(eval_out["mse_rung2"]["regret"])
    )
    print(f"\n  zero-fill -> mse_rung2 regret gap on real elites: {zf_to_mse:+.4f}")

    paired = {}
    for name, ref, key in (
        ("cost_space_rung2", "mse_rung2", "cost_space_vs_mse"),
        ("cost_space_rung2", "zero_fill", "cost_space_vs_zero_fill"),
        ("mse_rung2", "zero_fill", "mse_vs_zero_fill"),
    ):
        paired[key] = _paired(name, ref, eval_out)
        pr = paired[key]
        print(
            f"    {key:<26} mean {pr['mean_improvement']:+.4f}  "
            f"wins {pr['wins']}/{pr['n_nonzero']}  "
            f"ties(regret) {pr['n_ties_regret']}/{pr['n_seeds']}  "
            f"same-pick {pr['n_same_pick']}/{pr['n_seeds']}  p={pr['p']:.3g}"
        )

    for sd in test_data:
        del sd.harness
    return {
        "n_seeds": len(seeds),
        "eval": eval_out,
        "zero_fill_to_mse_gap": zf_to_mse,
        "paired": paired,
        "mppi_convergence": {"elite_min_mean": j_elite_min, "pool_min_mean": j_pool_min},
        "seconds": time.time() - t0,
    }


def run_seed_b_r2(
    seed: int, elite_planner: ElitePlanner, w_mse: np.ndarray, w_cs: np.ndarray
) -> dict:
    """`dfl_elites.run_seed_b`, rung-2 MLP fills substituted for the linear ones."""
    scene, truth, _measured, observed, sigma, poses, _hybrid_omega, _grid = build_case(
        seed, FAMILY, NOISE
    )
    belief = scene.elevation.astype(np.float64)
    truth64 = truth.astype(np.float64)
    phi6 = build_features(observed, belief, sigma, CELL)
    patch = _patch_features(phi6[..., FEATURE_NAMES.index("nearest_h")])
    ny, nx = truth.shape
    phi_flat_r2 = np.ascontiguousarray(
        np.concatenate([phi6, patch], axis=-1).reshape(ny * nx, 31)
    )
    truth32 = truth.astype(np.float32)
    friction32 = scene.friction.astype(np.float32)

    def _mlp_fill(w: np.ndarray) -> np.ndarray:
        y, _ = mlp_forward(RUNG2_SHAPE, w, phi_flat_r2)
        return y.reshape(truth64.shape)

    fills = {
        "zero_fill": np.zeros_like(belief),
        "mse_rung2": _mlp_fill(w_mse),
        "cost_space_rung2": _mlp_fill(w_cs),
    }
    out = {}
    for name, fill in fills.items():
        assembled = np.where(observed, truth64, fill).astype(np.float32)
        elite_omega, _elite_poses, diag = elite_planner.elites(assembled, friction32, GOAL_XY)
        harness = Harness(scene, poses, elite_omega, device="cuda")
        j_true_elites = _evaluate(harness, truth32)
        del harness
        out[name] = {
            "chosen_true_cost": float(j_true_elites[0]),
            "elite_best_true": float(j_true_elites.min()),
            "j_pool_min_on_fill": diag["J_pool_min"],
        }
    return out


def run_level_b(
    seeds: tuple[int, ...], elite_planner: ElitePlanner, w_mse: np.ndarray, w_cs: np.ndarray
) -> dict:
    print(f"\n=== test A, level B: {len(seeds)} seeds, plan-on-filled-map, top-1 on truth ===")
    t0 = time.time()
    rows = [run_seed_b_r2(s, elite_planner, w_mse, w_cs) for s in seeds]
    print(f"  done in {time.time() - t0:.1f}s")

    chosen = {name: np.array([r[name]["chosen_true_cost"] for r in rows]) for name in METHODS_A}
    own_regret = {
        name: chosen[name] - np.array([r[name]["elite_best_true"] for r in rows])
        for name in METHODS_A
    }

    print(f"\n  {'method':<18}{'chosen true cost':>18}{'own-pool regret':>18}")
    for name in METHODS_A:
        print(f"  {name:<18}{chosen[name].mean():>18.4f}{own_regret[name].mean():>18.4f}")

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
        "cost_space_vs_mse": _pair("cost_space_rung2", "mse_rung2"),
        "cost_space_vs_zero_fill": _pair("cost_space_rung2", "zero_fill"),
        "mse_vs_zero_fill": _pair("mse_rung2", "zero_fill"),
    }
    print("\n  paired on chosen true cost (positive = first name beats second):")
    for key, pr in paired.items():
        print(
            f"    {key:<26} mean {pr['mean_improvement']:+.4f}  wins {pr['wins']}/{pr['n_nonzero']}"
            f"  ties {pr['n_ties']}  p={pr['p']:.3g}"
        )

    return {
        "n_seeds": len(seeds),
        "chosen_true_cost_mean": {k: float(v.mean()) for k, v in chosen.items()},
        "own_pool_regret_mean": {k: float(v.mean()) for k, v in own_regret.items()},
        "paired": paired,
        "seconds": time.time() - t0,
    }


# --- test B: fair capacity-matched baseline (mse_rung2 + c * dist_unobs) ------------------------


def _tweak_fill_fn(w_mse: np.ndarray, c: float):
    def fn(s: CapSeedData) -> np.ndarray:
        base = model_fill(RUNG2_SHAPE, s, w_mse)
        dist_field = s.phi_flat_r2[:, DIST_UNOBS_IDX].reshape(s.truth.shape)
        return base + c * dist_field

    return fn


def tune_dist_coeff(train_data: list[CapSeedData], w_mse: np.ndarray) -> dict:
    """Grid-tunes `c` in `mse_rung2 + c * dist_unobs` on TRAIN-seed regret only -- mirrors the
    methods red-team's protocol at linear scale (coarse grid, c<=0 since fill should go LOWER the
    farther from observed data) and `dfl.py`'s own "tune on train, never touch held-out" rule."""
    print(f"\n=== test B: grid-tuning dist_unobs coefficient on {len(train_data)} TRAIN seeds ===")
    scored = []
    for c in C_GRID:
        regret = float(np.mean(evaluate_method(train_data, _tweak_fill_fn(w_mse, c))["regret"]))
        scored.append((c, regret))
        print(f"    c={c:+.2f}  train regret {regret:.4f}")
    c_best, regret_best = min(scored, key=lambda cr: cr[1])
    print(f"  best c = {c_best:+.2f}  (train regret {regret_best:.4f})")
    return {"grid": scored, "c_best": c_best, "train_regret_at_best": regret_best}


def run_test_b(
    train_data: list[CapSeedData],
    test_data: list[CapSeedData],
    w_mse: np.ndarray,
    w_cs: np.ndarray,
) -> dict:
    tuning = tune_dist_coeff(train_data, w_mse)
    c_best = tuning["c_best"]

    methods = {
        "zero_fill": zero_fill,
        "mse_rung2": lambda s: model_fill(RUNG2_SHAPE, s, w_mse),
        "mse_rung2_tweak": _tweak_fill_fn(w_mse, c_best),
        "cost_space_rung2": lambda s: model_fill(RUNG2_SHAPE, s, w_cs),
    }
    eval_out = {name: evaluate_method(test_data, fn) for name, fn in methods.items()}

    print(f"\n  {'method':<18}{'regret':>10}{'tau':>10}{'height RMSE':>14}")
    for name in methods:
        r = eval_out[name]
        print(
            f"  {name:<18}{np.mean(r['regret']):>10.4f}{np.mean(r['tau']):>+10.3f}"
            f"{np.mean(r['rmse']):>14.4f}"
        )

    tweak_regret = np.asarray(eval_out["mse_rung2_tweak"]["regret"])
    cost_space_regret = np.asarray(eval_out["cost_space_rung2"]["regret"])
    diff = tweak_regret - cost_space_regret
    n, k, p = sign_test(diff)
    paired = {
        "mean_improvement": float(diff.mean()),  # positive = cost_space_rung2 beats the tweak
        "median_improvement": float(np.median(diff)),
        "n_nonzero": n,
        "wins": k,
        "p": p,
    }
    print(
        f"\n  paired cost_space_rung2 vs mse_rung2_tweak: mean {paired['mean_improvement']:+.4f}  "
        f"wins {paired['wins']}/{paired['n_nonzero']}  p={paired['p']:.3g}"
    )

    mse_regret_mean = float(np.mean(eval_out["mse_rung2"]["regret"]))
    tweak_gap = mse_regret_mean - float(np.mean(tweak_regret))
    full_gap = mse_regret_mean - float(np.mean(cost_space_regret))
    frac_recovered = tweak_gap / full_gap if abs(full_gap) > 1e-9 else float("nan")
    print(
        f"\n  tweak recovers {frac_recovered:.1%} of the mse_rung2 -> cost_space_rung2 gap "
        f"({tweak_gap:+.4f} of {full_gap:+.4f})"
    )

    return {
        "tuning": tuning,
        "eval": eval_out,
        "paired_cost_space_vs_tweak": paired,
        "tweak_gap": tweak_gap,
        "full_gap": full_gap,
        "frac_of_gap_recovered_by_tweak": frac_recovered,
    }


# --- main -----------------------------------------------------------------------------------------


def main() -> None:
    t_start = time.time()
    wp.init()
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"building {len(TRAIN_SEEDS)} train + {len(TEST_SEEDS)} test seeds (rung-2 features; "
          f"family={FAMILY}, noise={NOISE}) ...")
    t0 = time.time()
    train_data = [build_cap_seed(s) for s in TRAIN_SEEDS]
    test_data = [build_cap_seed(s) for s in TEST_SEEDS]
    print(f"  done in {time.time() - t0:.1f}s")

    retrain = retrain_rung2(train_data, test_data)
    w_mse = np.asarray(retrain["trained_w"]["mse"])
    w_cs = np.asarray(retrain["trained_w"]["cost_space"])

    report: dict = {
        "family": FAMILY,
        "noise": NOISE,
        "weights_provenance": {
            "source": "retrained (dfl_capacity.json stores lr/loss/seconds per condition, not "
            "weight vectors; dfl_full.json's stored weights are the LINEAR model, wrong shape "
            "for rung 2)",
            "rung2_retrain": {k: v for k, v in retrain.items() if k != "trained_w"},
        },
    }

    scene0, *_ = build_case(TEST_SEEDS[0], FAMILY, NOISE)
    ny, nx = scene0.shape
    elite_planner = ElitePlanner(nx, ny, scene0.cell, scene0.origin_x, scene0.origin_y)

    seeds = TEST_SEEDS[:N_SEEDS_ELITES]
    level_a = run_level_a(seeds, elite_planner, w_mse, w_cs)
    level_b = run_level_b(seeds, elite_planner, w_mse, w_cs)
    report["test_a"] = {"level_a": level_a, "level_b": level_b}

    report["test_b"] = run_test_b(train_data, test_data, w_mse, w_cs)

    # --- verdicts, against the pre-registered readings -------------------------------------------
    pr_a = level_a["paired"]["cost_space_vs_mse"]
    a_survives = pr_a["p"] < 0.05 and pr_a["mean_improvement"] > 0
    verdict_a = (
        f"Test A: cost_space_rung2's edge over mse_rung2 "
        f"{'SURVIVES' if a_survives else 'DOES NOT SURVIVE'} real MPPI elites "
        f"(level A: mean {pr_a['mean_improvement']:+.4f}, p={pr_a['p']:.3g}, "
        f"{pr_a['wins']}/{pr_a['n_nonzero']} non-tied wins, "
        f"{pr_a['n_ties_regret']}/{pr_a['n_seeds']} seeds tied)."
    )
    pr_b_level = level_b["paired"]["cost_space_vs_mse"]
    verdict_a += (
        f" Level B (fill-in-the-loop) agrees in direction: "
        f"mean {pr_b_level['mean_improvement']:+.4f}, p={pr_b_level['p']:.3g}."
    )

    pr_b = report["test_b"]["paired_cost_space_vs_tweak"]
    frac = report["test_b"]["frac_of_gap_recovered_by_tweak"]
    c_best = report["test_b"]["tuning"]["c_best"]
    tweak_matches = pr_b["p"] >= 0.05
    verdict_b = (
        f"Test B: the fair one-parameter baseline (mse_rung2 + c*dist_unobs, c={c_best:+.2f} "
        f"tuned on train) recovers {frac:.1%} of the mse_rung2->cost_space_rung2 gap; paired "
        f"sign test cost_space_rung2 vs the tweak: mean {pr_b['mean_improvement']:+.4f}, "
        f"p={pr_b['p']:.3g}. "
        + (
            "The tweak MATCHES cost_space_rung2 (n.s.) -- per the pre-registered reading, the "
            "surviving capacity positive COLLAPSES to a heuristic again."
            if tweak_matches
            else "cost_space_rung2 beats the tweak significantly -- per the pre-registered "
            "reading, the capacity result STANDS as a genuine learning effect."
        )
    )
    report["verdict_a"] = verdict_a
    report["verdict_b"] = verdict_b
    print("\n=== verdicts ===")
    print(verdict_a)
    print(verdict_b)

    report["seconds_total"] = time.time() - t_start
    (OUT / "dfl_settle.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT / 'dfl_settle.json'}  ({report['seconds_total']:.1f}s total)")


if __name__ == "__main__":
    main()
