"""Does attribution pick the cells that decide WHICH PLAN WINS?  (open loop)

    .venv/bin/python -m studies.bench.ranking

This is C4's actual claim, tested directly and without a closed loop. The closed-loop benchmark
could not settle it: its outcome was entangled with the planner's route-selection behaviour,
the look-budget arithmetic, and a confirmatory-attribution bug -- and it posed a TOPOLOGY
question the method is structurally unsuited to (RESULTS.md section 6e).

Here the question is posed as ranking, which is all a sampling planner ever does with a cost:

  1. a ground-truth terrain, and a belief that has observed only a disc around the start
  2. K candidate plans, ranked by their TRUE cost (evaluated on ground truth)
  3. the same plans ranked by their BELIEVED cost (evaluated on the inpainted belief)
  4. a sensing policy picks M cells to reveal; those cells take their true values
  5. re-rank on the updated belief and score against the TRUE ranking

A policy is good exactly insofar as revealing its cells recovers the true ranking. Kendall tau
measures the whole ordering; top-1 accuracy measures the decision a planner actually makes.

WHY THIS IS HARDER TO FOOL THAN THE CLOSED LOOP:
  - the gradients are the REAL adjoint from DifferentiableSimulator, not the geometric proxy
    the closed-loop policies used, so the engine's own derivative is what is on trial
  - ranking K plans is inherently multi-plan, so the confirmatory single-plan failure mode
    cannot occur by construction
  - the terrain is random fractal, the observation is a plain disc, and the plans are a fixed
    fan: there are no scenario knobs to tune toward a result
  - it is fast enough (no driving) that n is large enough for the statistics to mean something

THE THREE WAYS THIS COMPARISON COULD STILL BE RIGGED, AND WHAT IS DONE ABOUT THEM:

  1. a DEGENERATE sigma. With sigma binary (observed / not), entropy has no preference among
     unobserved cells and its "choice" is really the argsort's tie order -- a straw man that
     loses to random. sigma is therefore given real spatial structure: residual uncertainty
     rises over locally rough ground, which is both realistic and, crucially, decided by the
     TERRAIN rather than by where the plans go. Entropy then has a genuine, sensible target
     and the question becomes the real one: is uncertainty the same thing as decision
     relevance?

  2. TIE ORDER. Every policy's scores are sorted after a random permutation, so no policy can
     profit from raster order on a plateau of equal scores.

  3. the adjoint might be doing NOTHING a straightedge could not. If "reveal the cells under
     where you are about to drive" scores as well, the derivative is decoration. So two purely
     geometric controls are included -- proximity to the plan swath, and the variance over
     plans of that proximity, which is the geometric analogue of the disagreement score. The
     adjoint has to beat its own shadow.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np
import warp as wp

from ..adjoint.generalise import fractal_terrain
from ..adjoint.harness import Harness
from ..adjoint.harness import TERM_NAMES
from ..adjoint.scene import Scene
from . import noise as noise_mod

OUT = Path(__file__).resolve().parents[2] / "studies" / "out" / "bench"

CELL = 0.10  # the real perception resolution (docs/performance.md)
EXTENT = (9.0, 9.0)  # [m] terrain patch
N_PLANS = 16
N_GROUPS = 4  # `hybrid` only: distinct paths
N_PER_GROUP = N_PLANS // N_GROUPS  # speed profiles riding each path
T_STEPS = 40
BUDGETS = (25, 100, 400)  # cells a policy may reveal (of ~7500 unobserved)
COST_TERMS = {"settle": 1.0, "clear_soft": 1.0}


def sign_test(d: np.ndarray) -> tuple[int, int, float]:
    """Two-sided sign test on paired differences. Ties dropped, as the test requires."""
    nz = d[d != 0]
    n, k = len(nz), int((nz > 0).sum())
    if n == 0:
        return 0, 0, 1.0
    tail = sum(comb(n, i) for i in range(min(k, n - k) + 1))
    return n, k, min(1.0, 2.0 * tail / 2**n)


def tau_within(a: np.ndarray, b: np.ndarray, groups: np.ndarray) -> float:
    """Mean Kendall tau computed separately inside each group of path-identical plans.

    Plans in one group cross exactly the same cells, so every geometric policy makes the same
    reveals for all of them and can only reorder them by accident. This is the component of the
    decision that requires a derivative, isolated from the part geometry can do.
    """
    taus = [kendall_tau(a[groups == g], b[groups == g]) for g in np.unique(groups)]
    taus = [t for t, g in zip(taus, np.unique(groups)) if (groups == g).sum() > 1]
    return float(np.mean(taus)) if taus else float("nan")


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Tau-b between two score vectors, via concordant/discordant pairs. K is small.

    Ties enter the denominator (tau_b = (C-D) / sqrt((C+D+T_a)(C+D+T_b))), unlike the naive
    (C-D)/(C+D) -- that is Goodman-Kruskal gamma, which drops ties entirely and inflates the
    reported tau whenever believed costs tie. A pair tied in BOTH a and b contributes to neither
    C, D, T_a nor T_b, per the standard tau_b definition.
    """
    n = len(a)
    conc = disc = tie_a = tie_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                continue
            elif da == 0:
                tie_a += 1
            elif db == 0:
                tie_b += 1
            else:
                s = np.sign(da) * np.sign(db)
                if s > 0:
                    conc += 1
                else:
                    disc += 1
    denom = np.sqrt((conc + disc + tie_a) * (conc + disc + tie_b))
    return (conc - disc) / denom if denom else 0.0


def _plans_fan(rng: np.random.Generator) -> np.ndarray:
    """SPATIALLY SEPARATED: a fan of constant-curvature drives. Each plan has its own corridor.

    Here "which plan wins" and "which cells does it cross" are nearly the same question, so a
    distance transform can answer it without any derivative. This is the easy case for geometry
    and, as it turns out, the hard case for justifying an adjoint.
    """
    turns = np.linspace(-0.55, 0.55, N_PLANS)
    speeds = 2.6 + 0.5 * rng.standard_normal(N_PLANS)
    omega = np.zeros((T_STEPS, N_PLANS, 3), np.float32)
    for k, (t, v) in enumerate(zip(turns, speeds)):
        omega[:, k, 0] = v - t
        omega[:, k, 1] = v + t
        omega[:, k, 2] = v
    return omega


def _plans_speed(rng: np.random.Generator) -> np.ndarray:
    """SPATIALLY CONFOUNDED: one path, sixteen speed profiles. Same cells, different loading.

    For a differential drive the PATH is set by the ratio of the wheel speeds, not their
    magnitude -- scaling both wheels by a common factor s(t) retraces the identical curve and
    only changes the timing along it. So a fixed turn ratio plus profiles s_k(t) that share a
    mean gives sixteen plans with the same path, the same endpoint and the same cells crossed,
    differing only in where along the path the robot is slow and where it is fast.

    That is the case the adjoint exists for. `swath` sees sixteen identical corridors and cannot
    rank them even in principle; `swath_var` sees ~zero variance everywhere and degenerates to
    its random tie-break. Only a derivative knows that the same cell carries a different WEIGHT
    for a plan that dwells on it than for one that crosses it quickly.

    (The wheel-lag model means the realised paths are not bit-identical; `coverage_spread` in
    the output measures the residual, so "same cells" is reported rather than assumed.)
    """
    curve = 0.12  # constant wL:wR ratio -> one circular arc, shared by every plan
    v0 = 2.6
    amp = 0.55
    phase = 2.0 * np.pi * np.arange(N_PLANS) / N_PLANS
    tt = 2.0 * np.pi * np.arange(T_STEPS) / T_STEPS
    # A whole number of periods, so every profile sums to exactly v0*T: equal arc length, and
    # therefore the same endpoint, not merely the same shape.
    s = v0 * (1.0 + amp * np.cos(tt[:, None] + phase[None, :]))  # [T, K]
    omega = np.empty((T_STEPS, N_PLANS, 3), np.float32)
    omega[:, :, 0] = s * (1.0 - curve)
    omega[:, :, 1] = s * (1.0 + curve)
    omega[:, :, 2] = s
    return omega


def _plans_hybrid(rng: np.random.Generator) -> np.ndarray:
    """PARTIALLY CONFOUNDED: 4 paths x 4 speed profiles. The case that actually discriminates.

    `speed` turned out to be the WRONG test: when every plan shares one path, the decision-
    relevant terrain is a single narrow corridor, so "reveal the corridor" is trivially optimal
    and geometry reaches tau = 1.0. Confounding made the problem easier for geometry, not harder.

    What is needed is confounding *within* a broad area, so the budget still bites. Four paths
    fan out as in `fan`; four speed profiles ride each path. Geometry can rank ACROSS the four
    groups -- they occupy different corridors -- but is structurally incapable of ranking WITHIN
    a group, because those four plans cross exactly the same cells and any purely geometric
    score assigns them identical selections. The adjoint has no such blind spot.

    `tau_within` isolates precisely that part of the ordering.
    """
    curves = np.linspace(-0.2, 0.2, N_GROUPS)  # multiplicative, so the path is fixed per group
    phase = 2.0 * np.pi * np.arange(N_PER_GROUP) / N_PER_GROUP
    tt = 2.0 * np.pi * np.arange(T_STEPS) / T_STEPS
    omega = np.empty((T_STEPS, N_PLANS, 3), np.float32)
    for g, c in enumerate(curves):
        for j in range(N_PER_GROUP):
            s = 2.6 * (1.0 + 0.55 * np.cos(tt + phase[j]))
            k = g * N_PER_GROUP + j
            omega[:, k, 0] = s * (1.0 - c)
            omega[:, k, 1] = s * (1.0 + c)
            omega[:, k, 2] = s
    return omega


PLAN_FAMILIES = {"fan": _plans_fan, "speed": _plans_speed, "hybrid": _plans_hybrid}
# Which plans share a path exactly, and are therefore indistinguishable to any geometric score.
PLAN_GROUPS = {
    "fan": np.arange(N_PLANS),  # every plan its own corridor
    "speed": np.zeros(N_PLANS, int),  # one shared corridor
    "hybrid": np.repeat(np.arange(N_GROUPS), N_PER_GROUP),
}


def build_case(seed: int, family: str = "fan", noise: str = "clean", flat_sigma: bool = False):
    """Ground truth, the belief, sigma, and K candidate plans reaching into the unknown.

    The Scene is built on the BELIEF, not on the truth. `Harness.adjoint` resets the terrain to
    the scene's own elevation before recording, so a scene built on ground truth would hand
    every gradient-based policy a derivative evaluated at the answer -- an oracle leak worth
    roughly a third of the effect when it was there.
    """
    rng = np.random.default_rng(seed)
    ny, nx = int(EXTENT[1] / CELL), int(EXTENT[0] / CELL)
    truth = fractal_terrain(ny, nx, CELL, seed=seed)
    x0, y0 = -1.0, -EXTENT[1] / 2

    xs = x0 + (np.arange(nx) + 0.5) * CELL
    ys = y0 + (np.arange(ny) + 0.5) * CELL
    XX, YY = np.meshgrid(xs, ys)
    # `belief` is what the robot thinks; `measured` is what a reveal actually hands over, which
    # is the TRUTH only under `noise="clean"`. sigma is consistent with what was injected.
    belief, measured, observed, sigma, _ = noise_mod.build_belief(
        truth, XX, YY, CELL, seed, noise, flat_sigma
    )
    mu = np.clip(0.6 + 0.1 * np.sin(XX) * np.cos(YY), 0.3, 0.9)

    # The plan family is fixed by construction, not drawn per seed, so no plan set can be
    # tuned to favour a policy.
    omega = PLAN_FAMILIES[family](rng)
    poses = np.tile(np.array([0.0, 0.0, 0.0], np.float32), (N_PLANS, 1))

    scene = Scene(belief, mu, np.zeros(truth.shape, np.int8), CELL, x0, y0)
    return scene, truth.astype(np.float32), measured, observed, sigma, poses, omega, (XX, YY)


def _cost(terms: np.ndarray) -> np.ndarray:
    return sum(w * terms[TERM_NAMES.index(k)] for k, w in COST_TERMS.items())


def _evaluate(harness: Harness, elev2d: np.ndarray) -> np.ndarray:
    """Cost of every plan on one terrain."""
    stack = np.ascontiguousarray(np.tile(elev2d, (N_PLANS, 1, 1)), np.float32)
    with wp.ScopedDevice(harness.device):
        harness.sim.set_terrain(wp.array(stack, dtype=wp.float32))
    return _cost(harness.forward(dilate=True)).copy()


def _plan_distance(harness: Harness, grid: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """[K, ny, nx] distance from every cell to the nearest point of each plan's path.

    Read off the trajectories the LAST forward produced, so it is the believed path -- the
    same information a purely geometric policy would have.
    """
    XX, YY = grid
    traj = harness.sim.controlled.numpy()[:, :, :2]  # [T+1, K, 2]
    d = np.empty((N_PLANS, *XX.shape), np.float32)
    for k in range(N_PLANS):
        p = traj[:, k, :]
        d[k] = np.sqrt(
            ((XX[None] - p[:, 0, None, None]) ** 2 + (YY[None] - p[:, 1, None, None]) ** 2).min(0)
        )
    return d


# --- the policies -----------------------------------------------------------------------
# Each returns a per-cell score; the top M unobserved cells are revealed. `ctx` carries
# everything a policy could legitimately know: gradients, sigma, believed costs, plan
# distances -- and, for the oracle only, the truth.
def p_random(ctx):
    return ctx["rng"].random(ctx["sigma"].shape)


def p_entropy(ctx):
    """Information-theoretic: reveal the most uncertain cells. Knows nothing about the task."""
    return ctx["sigma"] ** 2


def _proximity(dist: np.ndarray) -> np.ndarray:
    """A plan's geometric 'influence' on a cell: 1 under the wheels, decaying over ~a track."""
    return np.exp(-0.5 * (dist / 0.35) ** 2)


def p_swath(ctx):
    """Geometric control: reveal what you are about to drive over. No derivative at all."""
    return _proximity(ctx["dist"]).max(axis=0)


def p_swath_best(ctx):
    """Geometric control matched to `attribution`: the incumbent plan's corridor only.

    `attribution` sees one plan and therefore one narrow strip of nonzero gradient, so part of
    its deficit against the plan-set scores could be coverage rather than information. This
    isolates that: it is the same strip, chosen with no derivative.
    """
    return _proximity(ctx["dist"][int(np.argmin(ctx["believed"]))])


def p_swath_var(ctx):
    """Geometric control, discriminative form: cells SOME plans pass and others do not.

    The geometric analogue of `disagreement`, and the control that matters -- if it ties, the
    adjoint is decoration on a distance transform. Variance of the proximity kernel, not of raw
    distance: raw distance disagrees most in the far field, where no plan goes at all.
    """
    return _proximity(ctx["dist"]).var(axis=0)


def p_corridor_sigma(ctx):
    """Corridor mask x uncertainty, no derivative: the strongest fair baseline (P2).

    max_k proximity(dist_k) * sigma^2 -- task-region-masked entropy. In the motion-coupled
    study this TIED or BEAT the adjoint on holdout, so the free-look claim has to answer to it
    too, not just to the derivative-blind `entropy` and `swath`.
    """
    return _proximity(ctx["dist"]).max(axis=0) * ctx["sigma"] ** 2


def p_corridor_mi(ctx):
    """Corridor mask x mutual-information-shaped uncertainty, no derivative: same mask as
    `p_corridor_sigma`, but log-saturating in sigma rather than quadratic."""
    # 0.05 m is a round mid-field noise floor, NOT the sensor's measured per-cell std -- that
    # (bench/noise.py SENSOR_BASE/SENSOR_PER_M) ranges 0.010-0.046 m, range-dependent.
    return _proximity(ctx["dist"]).max(axis=0) * 0.5 * np.log1p(ctx["sigma"] ** 2 / 0.05**2)


def p_attribution(ctx):
    """Single-plan: sensitivity of the currently-BEST believed plan. sum_i (dJ/dh_i * sigma_i)^2.

    This is the score the closed-loop benchmark used, and the one whose confirmatory bias was
    diagnosed there. Included so the comparison is against the method as previously stated.
    """
    g = ctx["grad"][int(np.argmin(ctx["believed"]))]
    return (np.abs(g) * ctx["sigma"]) ** 2


def p_magnitude(ctx):
    """Total sensitivity over the plan set -- sensitive, but not necessarily discriminative.

    Separates "this cell moves the cost" from "this cell moves the RANKING": a cell every plan
    crosses identically scores high here and contributes nothing to the decision.
    """
    return (np.abs(ctx["grad"]).sum(axis=0) * ctx["sigma"]) ** 2


def p_disagreement(ctx):
    """Where the candidate plans' sensitivities DISAGREE: Var_k(dJ_k/dh_i) * sigma_i^2.

    To first order this is the variance across the plan set of each plan's cost response to
    cell i, which is exactly the part of the uncertainty that can reorder them. A cell all
    plans respond to identically shifts every cost together and cannot change the argmin.
    """
    return ctx["grad"].var(axis=0) * ctx["sigma"] ** 2


def p_margin(ctx):
    """Decision margin: d(J_a - J_b)/dh for the top TWO believed plans.

    The sharpest form of "what would change my mind" -- the derivative of the quantity whose
    sign IS the decision. Narrower than `disagreement`: it only defends the current top pair.
    """
    order = np.argsort(ctx["believed"])
    return ((ctx["grad"][order[0]] - ctx["grad"][order[1]]) * ctx["sigma"]) ** 2


def _disc_pool(score: np.ndarray, radius_cells: float) -> np.ndarray:
    """Sum a per-cell score over the disc that one wheel's contact search covers.

    A cell reaches the cost only through `envelope[p] = max_q (h[q] + cap)`, a max over a
    cap-sized neighbourhood. So the unit that actually moves the cost is a cap-sized PATCH, not
    a cell: revealing one gradient-hot cell while its neighbours keep the inpaint value often
    leaves the max sitting on an unrevealed neighbour and changes nothing. Pooling scores the
    patch instead, which is what the max makes decision-relevant.
    """
    r = int(np.ceil(radius_cells))
    out = np.zeros_like(score)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx <= radius_cells**2:
                out += np.roll(np.roll(score, dy, 0), dx, 1)
    return out


def p_disagreement_pooled(ctx):
    """`disagreement`, pooled over the contact cap -- the patch form of the same score."""
    return _disc_pool(ctx["grad"].var(axis=0) * ctx["sigma"] ** 2, ctx["env_cells"])


def p_magnitude_pooled(ctx):
    return _disc_pool((np.abs(ctx["grad"]).sum(axis=0) * ctx["sigma"]) ** 2, ctx["env_cells"])


def p_oracle(ctx):
    """Ceiling, not a policy: knows the ACTUAL belief error, so it needs no sigma at all."""
    err = np.abs(ctx["measured"] - ctx["belief"])
    return ctx["grad"].var(axis=0) * err**2


POLICIES = {
    "random": p_random,
    "entropy": p_entropy,
    "swath": p_swath,
    "swath_best": p_swath_best,
    "swath_var": p_swath_var,
    "corridor_sigma": p_corridor_sigma,
    "corridor_mi": p_corridor_mi,
    "attribution": p_attribution,
    "magnitude": p_magnitude,
    "margin": p_margin,
    "disagreement": p_disagreement,
    "magn_pooled": p_magnitude_pooled,
    "disag_pooled": p_disagreement_pooled,
    "oracle": p_oracle,
}
NOT_A_POLICY = ("oracle",)


def run_seed(
    seed: int, family: str = "fan", noise: str = "clean", flat_sigma: bool = False
) -> dict:
    groups = PLAN_GROUPS[family]
    scene, truth, measured, observed, sigma, poses, omega, grid = build_case(
        seed, family, noise, flat_sigma
    )
    harness = Harness(scene, poses, omega, device="cuda")
    belief = scene.elevation.astype(np.float32)

    # Gradients FIRST: `adjoint` resets the terrain to the scene's elevation, i.e. the belief,
    # which is the only terrain a policy is entitled to differentiate. The rollouts it leaves
    # in `controlled` are the believed paths, so the geometric controls read them from here.
    grads, _ = harness.adjoint(dilate=True, leaf="elevation")
    grad = sum(w * grads[TERM_NAMES.index(k)] for k, w in COST_TERMS.items())  # [K, ny, nx]
    dist = _plan_distance(harness, grid)
    traj_end = harness.sim.controlled.numpy()[-1, :, :2]

    j_bel = _evaluate(harness, belief)
    # `_rollout` re-derives the contact, so the envelope buffer is now the BELIEF's envelope.
    # Kept to measure how much of each policy's reveal actually reaches the cost: a cell only
    # matters through this max, so reveals that leave the envelope untouched are inert.
    env_bel = harness.sim.envelope.numpy()[0].copy()
    j_true = _evaluate(harness, truth)

    rng_field = np.hypot(*grid)  # [ny, nx] range from the robot's start
    rng = np.random.default_rng(10_000 + seed)
    ctx = {
        "grad": grad,
        "sigma": sigma,
        "believed": j_bel,
        "dist": dist,
        "truth": truth,
        "measured": measured,
        "belief": belief,
        # env_radius is already in CELLS (ceil(wheel_radius / cell_size)), not metres.
        "env_cells": float(harness.sim.env_radius),
        "rng": rng,
    }
    # How spatially SEPARATE the plans actually are, so "same cells" is measured not assumed:
    # the mean across cells of the spread, over plans, of each plan's geometric influence. This
    # is exactly the quantity `swath_var` scores on, so a near-zero value means the geometric
    # discriminator has nothing to work with -- by construction, not by tuning.
    prox = _proximity(dist)
    out = {
        "seed": seed,
        "tau_before": kendall_tau(j_bel, j_true),
        "top1_before": bool(np.argmin(j_bel) == np.argmin(j_true)),
        "spread_true": float(j_true.max() - j_true.min()),
        "coverage_spread": float(prox.std(axis=0).mean()),
        "tau_within_before": tau_within(j_bel, j_true, groups),
        "path_spread": float(np.linalg.norm(traj_end - traj_end.mean(0), axis=1).mean()),
        "policies": {},
    }
    n_cells = sigma.size
    for name, fn in POLICIES.items():
        score = np.asarray(fn(ctx), float).ravel()
        score[observed.ravel()] = -np.inf  # revealing a known cell is worthless
        # Random tie-break: permute, then a STABLE descending sort. Without this, a policy
        # whose score plateaus (entropy on a binary sigma, swath outside the swath) is really
        # being ranked by raster order, which is a straw man rather than a baseline.
        perm = rng.permutation(n_cells)
        order = perm[np.argsort(-score[perm], kind="stable")]
        rec = {}
        for m in BUDGETS:
            revealed = np.zeros(n_cells, bool)
            revealed[order[:m]] = True
            # a reveal hands over the MEASUREMENT. Under a sensor or pose error that is not
            # the truth, so sensing no longer converges on perfect knowledge -- which is the
            # single biggest way section 7's clean setup flattered every policy at once.
            updated = np.where(observed | revealed.reshape(sigma.shape), measured, 0.0)
            j_new = _evaluate(harness, updated.astype(np.float32))
            denv = np.abs(harness.sim.envelope.numpy()[0] - env_bel)
            rec[str(m)] = {
                "tau": kendall_tau(j_new, j_true),
                "tau_within": tau_within(j_new, j_true, groups),
                "top1": bool(np.argmin(j_new) == np.argmin(j_true)),
                "regret": float(j_true[int(np.argmin(j_new))] - j_true.min()),
                # envelope cells moved per cell revealed: the reveal's actual reach into the cost
                "reach": float((denv > 1e-4).sum() / m),
                # WHERE the policy looked: range from the robot, and how far off the plans it
                # strayed. These separate "picked the wrong place" from "picked too few places".
                "range": float(rng_field.ravel()[order[:m]].mean()),
                "off_plan": float(dist.min(axis=0).ravel()[order[:m]].mean()),
            }
        out["policies"][name] = rec
    del harness
    return out


def report(rows: list[dict]) -> dict:
    n = len(rows)
    tb = np.mean([r["tau_before"] for r in rows])
    t1b = np.mean([r["top1_before"] for r in rows])
    rb = np.mean([r["spread_true"] for r in rows])
    cs = np.mean([r["coverage_spread"] for r in rows])
    ps = np.mean([r["path_spread"] for r in rows])
    print(f"\nbefore sensing:  Kendall tau {tb:+.3f}   top-1 correct {t1b:.0%}   (n={n})")
    print(f"true cost spread across the {N_PLANS} plans: {rb:.2f} mean")
    print(
        f"plan separation: endpoints {ps:.2f} m apart, coverage spread {cs:.4f}"
        "  (low = geometry cannot discriminate)\n"
    )
    print("tau  = agreement of the post-sensing ranking with the truth (1 = perfect)")
    print("top1 = the believed-best plan really is the best")
    print("reg  = true excess cost of the plan a planner would pick (0 = optimal)")
    print("rch  = envelope cells actually moved per cell revealed (low = inert reveals)\n")
    hdr = "".join(f"{'tau@' + str(m):>9}{'top1':>6}{'reg':>7}{'rch':>7}" for m in BUDGETS)
    print(f"{'policy':<14}{hdr}")
    summary = {}
    for name in POLICIES:
        line = f"{name:<14}"
        summary[name] = {}
        for m in BUDGETS:
            tau = np.mean([r["policies"][name][str(m)]["tau"] for r in rows])
            t1 = np.mean([r["policies"][name][str(m)]["top1"] for r in rows])
            reg = np.mean([r["policies"][name][str(m)]["regret"] for r in rows])
            rch = np.mean([r["policies"][name][str(m)]["reach"] for r in rows])
            line += f"{tau:>+9.3f}{t1:>5.0%}{reg:>7.2f}{rch:>7.1f}"
            summary[name][str(m)] = {
                "tau": float(tau),
                "top1": float(t1),
                "regret": float(reg),
                "reach": float(rch),
            }
        print(line + ("   <- ceiling, not a policy" if name in NOT_A_POLICY else ""))

    print("\nwhere each policy looked: mean range of the revealed cells [m], and their mean")
    print(f"distance off the nearest plan [m], at {BUDGETS[1]} cells")
    for name in POLICIES:
        rr = np.mean([r["policies"][name][str(BUDGETS[1])]["range"] for r in rows])
        op = np.mean([r["policies"][name][str(BUDGETS[1])]["off_plan"] for r in rows])
        print(f"  {name:<14} range {rr:>5.2f}   off-plan {op:>5.2f}")
        summary[name][str(BUDGETS[1])].update(range=float(rr), off_plan=float(op))

    def paired(ref: str) -> None:
        print(f"\npaired per seed, X minus {ref} in tau (positive = X better):")
        print(f"{'':<16}" + "".join(f"{'@' + str(m):>25}" for m in BUDGETS))
        for name in POLICIES:
            if name == ref:
                continue
            line = f"  {name:<14}"
            for m in BUDGETS:
                d = np.array(
                    [
                        r["policies"][name][str(m)]["tau"] - r["policies"][ref][str(m)]["tau"]
                        for r in rows
                    ]
                )
                k, w, p = sign_test(d)
                line += f"{d.mean():>+8.3f} {w:>3}/{k:<3} p={p:>7.1e}"
            print(line)

    paired("entropy")
    paired("swath")
    paired("swath_var")
    paired("corridor_sigma")
    paired("corridor_mi")
    return summary


FIG_POLICIES = ("entropy", "random", "swath_best", "attribution", "swath", "disagreement", "oracle")
FIG_COLOR = {
    "entropy": "#2f7ec4",
    "random": "#9aa5b1",
    "swath_best": "#c9a227",
    "attribution": "#d1495b",
    "swath": "#4c9a6a",
    "disagreement": "#7a3ea8",
    "oracle": "#333333",
}


def figure(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))
    xs = np.array(BUDGETS, float)

    for key, ax, label in (
        ("tau", axes[0], "Kendall tau vs the true ranking"),
        ("top1", axes[1], "top-1 plan correct"),
    ):
        for name in FIG_POLICIES:
            y = [np.mean([r["policies"][name][str(m)][key] for r in rows]) for m in BUDGETS]
            ax.plot(
                xs,
                y,
                marker="o",
                color=FIG_COLOR[name],
                lw=2.2 if name in ("disagreement", "swath") else 1.4,
                ls="--" if name == "oracle" else "-",
                label=name,
                ms=5,
            )
        ax.set_xscale("log")
        ax.minorticks_off()  # log minor ticks otherwise collide with the budget labels
        ax.set_xticks(xs)
        ax.set_xticklabels([str(m) for m in BUDGETS])
        ax.set_xlabel("cells revealed")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0].axhline(
        np.mean([r["tau_before"] for r in rows]), color="k", lw=0.9, ls=":", label="no sensing"
    )
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].set_title(
        "(a) the adjoint beats entropy at every budget,\nbut only beats geometry at 400"
    )
    axes[1].set_title("(b) the decision a planner actually makes")

    ax = axes[2]
    nudge = {  # hand-placed: several policies land almost on top of each other
        "entropy": (8, -12),
        "random": (8, 4),
        "swath": (-6, 12),
        "oracle": (8, 6),
        "disagreement": (8, -4),
        "attribution": (8, 2),
        "swath_best": (8, -4),
    }
    for name in FIG_POLICIES:
        rr = np.mean([r["policies"][name][str(BUDGETS[1])]["range"] for r in rows])
        op = np.mean([r["policies"][name][str(BUDGETS[1])]["off_plan"] for r in rows])
        ax.scatter(op, rr, s=90, color=FIG_COLOR[name], edgecolor="k", lw=0.5, zorder=3)
        ax.annotate(
            name, (op, rr), textcoords="offset points", xytext=nudge[name], fontsize=8, zorder=4
        )
    ax.set_xlabel("mean distance off the nearest plan [m]")
    ax.set_ylabel("mean range from the robot [m]")
    ax.set_title("(c) WHERE they looked, at 100 cells:\nthe whole effect is task-aware vs not")
    ax.grid(alpha=0.25)

    fig.suptitle(
        "Open-loop ranking: does map-cell attribution pick the cells that decide which plan wins?"
        f"  (n={len(rows)})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(
    n_seeds: int = 200, family: str = "fan", noise: str = "clean", flat_sigma: bool = False
) -> None:
    wp.init()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"plan family: {family} -- {PLAN_FAMILIES[family].__doc__.splitlines()[0]}")
    print(f"noise: {noise}")
    rows = []
    for seed in range(n_seeds):
        rows.append(run_seed(seed, family, noise, flat_sigma))
        if (seed + 1) % 50 == 0:
            print(f"  {seed + 1}/{n_seeds} seeds", flush=True)
    summary = report(rows)
    tag = family if noise == "clean" else f"{family}_{noise}"
    tag += "_flatsigma" if flat_sigma else ""
    figure(rows, OUT / f"ranking_{tag}.png")
    (OUT / f"ranking_{tag}.json").write_text(
        json.dumps({"family": family, "noise": noise, "rows": rows, "summary": summary}, indent=2)
    )
    print(f"\nwrote {OUT / f'ranking_{tag}.json'} and ranking_{tag}.png")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--family", choices=tuple(PLAN_FAMILIES), default="fan")
    ap.add_argument("--noise", choices=noise_mod.SOURCES, default="clean")
    ap.add_argument(
        "--flat-sigma",
        action="store_true",
        help="uniform sigma over unobserved cells: removes the truth-derived roughness leak",
    )
    a = ap.parse_args()
    main(a.seeds, a.family, a.noise, a.flat_sigma)
