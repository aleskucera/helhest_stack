"""Does the second-order correction improve the DECISION, or only the variance estimate?

    .venv/bin/python -m studies.bench.second_order --seeds 50 --family hybrid

Section 4 established that second-order FOSM cuts the fraction of badly-wrong variance
estimates from 34.1% to 3.2% at realistic sigma, and section 5 that it fits in a control tick.
Section 7 established that first-order attribution improves plan ranking. Nothing connects
them: being more accurate about Var(J) is not the same as making a better decision, and a
planner has no use for the former except through the latter.

Curvature can enter the decision by two entirely different routes, and they are tested apart
because they cost different things and could easily come out opposite ways:

  1. BIAS CORRECTION, which needs no sensing at all. E[J_k] - J_k(h_belief) = 1/2 sum_i c_ki
     sigma_i^2. A cost offset common to every plan cannot change a ranking -- but this one is
     NOT common: a plan that dwells on rough ground accumulates more curvature than one that
     crosses it, so the correction is per-plan and moves the ordering. If it helps, curvature
     buys decision quality for zero observations, which is a strictly stronger claim than
     anything sensing-based.

  2. CELL SELECTION. Replace Var_k(g_ki) sigma_i^2 with the second-order variance
     Var_k(g_ki) sigma_i^2 + 1/2 Var_k(c_ki) sigma_i^4 and re-rank the candidate cells.

The selection comparison is MATCHED: both scores choose from the same shortlist of cells (the
top `SHORTLIST` by first order), and every budget is smaller than the shortlist, so the
first-order arm's picks are exactly what it would have chosen unrestricted. The only difference
between the arms is the curvature term -- not the candidate set, not the budget, not the seed.

Curvature is evaluated by central differences at each cell's OWN sigma, following section 4's
convention: that is deliberately the ceiling, the best the correction could do, not an estimate
of what a cheap online approximation would achieve.
"""

from __future__ import annotations

import json

import numpy as np
import warp as wp

from ..adjoint.harness import _perturb_cell
from ..adjoint.harness import Harness
from ..adjoint.harness import TERM_NAMES
from .ranking import _cost
from .ranking import _evaluate
from .ranking import BUDGETS
from .ranking import build_case
from .ranking import COST_TERMS
from .ranking import kendall_tau
from .ranking import N_PLANS
from .ranking import OUT
from .ranking import PLAN_GROUPS
from .ranking import sign_test
from .ranking import tau_within

SHORTLIST = 600  # cells the curvature probe is spent on; must exceed max(BUDGETS)


def _curvature(harness: Harness, cells: np.ndarray, deltas: np.ndarray, j0: np.ndarray):
    """c[n, k] = d2 J_k / dh_n^2 at the belief, central difference at each cell's own sigma.

    Two forwards per cell over the whole plan batch: one launch perturbs cell n in every slice,
    so all K plans are differenced together. `_reset_terrain` restores from the pristine copy
    rather than by subtracting, so no drift accumulates over 600 cells.
    """
    curv = np.zeros((len(cells), N_PLANS), np.float32)
    for n, ((iy, ix), d) in enumerate(zip(cells, deltas)):
        js = []
        for sign in (1.0, -1.0):
            harness._reset_terrain(True)
            wp.launch(
                _perturb_cell,
                harness.batch_size,
                inputs=[harness.sim.elevation, int(iy), int(ix), sign * float(d)],
                device=harness.device,
            )
            js.append(_cost(harness.forward(True)).copy())
        curv[n] = (js[0] + js[1] - 2.0 * j0) / (d * d)
    harness._reset_terrain(True)
    return curv


def run_seed(seed: int, family: str) -> dict:
    groups = PLAN_GROUPS[family]
    scene, truth, observed, sigma, poses, omega, _ = build_case(seed, family)
    harness = Harness(scene, poses, omega, device="cuda")
    belief = scene.elevation.astype(np.float32)

    grads, _ = harness.adjoint(dilate=True, leaf="elevation")
    grad = sum(w * grads[TERM_NAMES.index(k)] for k, w in COST_TERMS.items())  # [K, ny, nx]

    j_bel = _evaluate(harness, belief)
    j_true = _evaluate(harness, truth)

    first = grad.var(axis=0) * sigma**2  # the section-7 `disagreement` score
    first_masked = np.where(observed, -np.inf, first).ravel()
    shortlist = np.argsort(-first_masked)[:SHORTLIST]
    cells = np.stack(np.unravel_index(shortlist, sigma.shape), axis=1)
    deltas = sigma.ravel()[shortlist]

    curv = _curvature(harness, cells, deltas, j_bel)  # [SHORTLIST, K]

    # --- route 1: bias correction, no sensing -------------------------------------------
    # E[J_k] - J_k(belief) = 1/2 sum_i c_ki sigma_i^2, summed over the shortlist. Truncating to
    # the shortlist understates it, so this is a LOWER bound on what full curvature would do.
    bias = 0.5 * (curv * (deltas**2)[:, None]).sum(axis=0)  # [K]
    j_debiased = j_bel + bias

    # --- route 2: cell selection --------------------------------------------------------
    s1 = first.ravel()[shortlist]
    s2 = s1 + 0.5 * curv.var(axis=1) * deltas**4
    rng = np.random.default_rng(20_000 + seed)

    out = {
        "seed": seed,
        "tau_before": kendall_tau(j_bel, j_true),
        "tau_debiased": kendall_tau(j_debiased, j_true),
        "tau_within_before": tau_within(j_bel, j_true, groups),
        "tau_within_debiased": tau_within(j_debiased, j_true, groups),
        "top1_before": bool(np.argmin(j_bel) == np.argmin(j_true)),
        "top1_debiased": bool(np.argmin(j_debiased) == np.argmin(j_true)),
        "bias_spread": float(bias.max() - bias.min()),
        "cost_spread": float(j_true.max() - j_true.min()),
        "arms": {},
    }
    for name, score in (("first", s1), ("second", s2)):
        perm = rng.permutation(len(score))
        order = shortlist[perm[np.argsort(-score[perm], kind="stable")]]
        rec = {}
        for m in BUDGETS:
            revealed = np.zeros(sigma.size, bool)
            revealed[order[:m]] = True
            updated = np.where(observed | revealed.reshape(sigma.shape), truth, 0.0)
            j_new = _evaluate(harness, updated.astype(np.float32))
            rec[str(m)] = {
                "tau": kendall_tau(j_new, j_true),
                "top1": bool(np.argmin(j_new) == np.argmin(j_true)),
            }
        out["arms"][name] = rec
    del harness
    return out


def report(rows: list[dict], family: str) -> None:
    n = len(rows)
    print(f"\n=== family {family}, n={n}, curvature on the top {SHORTLIST} cells ===\n")

    print("ROUTE 1 -- bias correction, NO sensing at all")
    tb = np.array([r["tau_before"] for r in rows])
    td = np.array([r["tau_debiased"] for r in rows])
    k, w, p = sign_test(td - tb)
    print(f"  tau            {tb.mean():+.3f} -> {td.mean():+.3f}   better on {w}/{k}   p={p:.2e}")
    wb = np.array([r["tau_within_before"] for r in rows])
    wd = np.array([r["tau_within_debiased"] for r in rows])
    k, w, p = sign_test(wd - wb)
    print(f"  tau within     {wb.mean():+.3f} -> {wd.mean():+.3f}   better on {w}/{k}   p={p:.2e}")
    print(
        f"  top-1          {np.mean([r['top1_before'] for r in rows]):.0%} -> "
        f"{np.mean([r['top1_debiased'] for r in rows]):.0%}"
    )
    bs = np.mean([r["bias_spread"] for r in rows])
    cs = np.mean([r["cost_spread"] for r in rows])
    print(f"  the correction spreads the plans by {bs:.3f} against a true cost spread of {cs:.2f}")

    print("\nROUTE 2 -- cell selection, first vs second order, same shortlist")
    for m in BUDGETS:
        t1 = np.array([r["arms"]["first"][str(m)]["tau"] for r in rows])
        t2 = np.array([r["arms"]["second"][str(m)]["tau"] for r in rows])
        k, w, p = sign_test(t2 - t1)
        o1 = np.mean([r["arms"]["first"][str(m)]["top1"] for r in rows])
        o2 = np.mean([r["arms"]["second"][str(m)]["top1"] for r in rows])
        print(
            f"  @{m:<4} tau {t1.mean():+.3f} -> {t2.mean():+.3f}  (mean {(t2 - t1).mean():+.4f}, "
            f"better on {w}/{k}, p={p:.2e})   top-1 {o1:.0%} -> {o2:.0%}"
        )


def validate_bias(n_seeds: int, offset: int, k_cells: int, family: str) -> None:
    """Held-out check of the bias correction at a PRE-COMMITTED truncation.

    The exploratory sweep on seeds 0-19 showed the correction helping up to ~100-300 cells and
    then degrading, because summing per-cell second-order terms independently through a max
    overcounts -- the correction's spread grows linearly in the number of cells summed, which is
    Study B's "i.i.d. per-cell map noise is not a valid model here" appearing again.

    An interior optimum found by sweeping on the same data is a tuned hyperparameter, not a
    result. So `k_cells` is fixed from that sweep and evaluated here on DISJOINT seeds, once.
    """
    groups = PLAN_GROUPS[family]
    before, after = [], []
    for seed in range(offset, offset + n_seeds):
        scene, truth, observed, sigma, poses, omega, _ = build_case(seed, family)
        harness = Harness(scene, poses, omega, device="cuda")
        belief = scene.elevation.astype(np.float32)
        grads, _ = harness.adjoint(dilate=True, leaf="elevation")
        grad = sum(w * grads[TERM_NAMES.index(k)] for k, w in COST_TERMS.items())
        j_bel = _evaluate(harness, belief)
        j_true = _evaluate(harness, truth)
        first = np.where(observed, -np.inf, grad.var(axis=0) * sigma**2).ravel()
        sl = np.argsort(-first)[:k_cells]
        cells = np.stack(np.unravel_index(sl, sigma.shape), axis=1)
        deltas = sigma.ravel()[sl]
        curv = _curvature(harness, cells, deltas, j_bel)
        bias = 0.5 * (curv * (deltas**2)[:, None]).sum(axis=0)
        before.append((kendall_tau(j_bel, j_true), tau_within(j_bel, j_true, groups)))
        after.append((kendall_tau(j_bel + bias, j_true), tau_within(j_bel + bias, j_true, groups)))
        del harness
        if (seed - offset + 1) % 20 == 0:
            print(f"  {seed - offset + 1}/{n_seeds}", flush=True)

    b, a = np.array(before), np.array(after)
    print(f"\nHELD-OUT bias correction, seeds {offset}-{offset + n_seeds - 1}, top {k_cells} cells")
    print("(truncation fixed in advance from the seeds 0-19 sweep; evaluated here once)")
    for j, label in ((0, "tau"), (1, "tau within")):
        k, w, p = sign_test(a[:, j] - b[:, j])
        print(
            f"  {label:<12} {b[:, j].mean():+.3f} -> {a[:, j].mean():+.3f}"
            f"   better on {w}/{k}   p={p:.2e}"
        )


def main(n_seeds: int = 50, family: str = "hybrid") -> None:
    wp.init()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(n_seeds):
        rows.append(run_seed(seed, family))
        if (seed + 1) % 10 == 0:
            print(f"  {seed + 1}/{n_seeds} seeds", flush=True)
    report(rows, family)
    path = OUT / f"second_order_{family}.json"
    path.write_text(json.dumps({"family": family, "rows": rows}, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--family", default="hybrid", choices=tuple(PLAN_GROUPS))
    ap.add_argument("--validate-bias", type=int, default=0, help="held-out run: cells to sum")
    ap.add_argument("--seed-offset", type=int, default=0)
    a = ap.parse_args()
    if a.validate_bias:
        wp.init()
        validate_bias(a.seeds, a.seed_offset, a.validate_bias, a.family)
    else:
        main(a.seeds, a.family)
