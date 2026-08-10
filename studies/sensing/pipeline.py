"""Sensing you have to DRIVE to: one look, chosen from where the body can point the lidar.

    .venv/bin/python -m studies.sensing.pipeline --seeds 120

`studies/bench/ranking.py` settled the question of WHICH CELLS are worth knowing: the per-cell
score Var_k(dJ_k/dh_i) * sigma_i^2 beats an entropy-directed reveal decisively (FINDINGS section
2.4), and it beats a pure distance transform exactly in the regime where the candidate plans
partly share ground (section 2.5). But it settled it with a cell BUDGET -- a policy named M
cells anywhere on the map and they were handed over. No robot can do that.

This module poses the same question under the constraint the hardware actually imposes. The
lidar is bolted to the front of the chassis; the only way to point it is to move the body. So a
sensing action is a VIEWPOINT -- turn in place by some bearing, or creep out and turn -- and what
it reveals is a ray-cast cone, not a set of cells. The per-cell score becomes an objective to be
INTEGRATED over the cone rather than a list to be sorted, and three things change with it:

  * the action set is tiny (36 viewpoints, not 7500 cells) and every action reveals hundreds of
    cells at once, so the differences between policies have to survive being smeared over a cone
  * cells are no longer independently selectable -- the score's SHAPE decides the answer only
    insofar as it varies at the scale of the cone
  * looking costs TIME (turn + creep), which is the currency a real planner trades against

The policies are the same as ranking.py's, restricted to the ones that survived it: entropy
(the information-theoretic baseline), swath_var (the geometric control -- if it ties, the
adjoint is decoration on a distance transform), disagreement (ours), random, and none.

Three stronger baselines are added here, because a cone integral is a much coarser instrument
than a cell budget and the weak forms of the baselines could be losing for the wrong reason:

  * `entropy_mi` -- the proper Gaussian mutual information 0.5*log(1 + sigma^2/sigma_n^2)
    rather than raw sigma^2. Concave in sigma^2, so it saturates where the raw form keeps
    growing, and over a cone that changes which bearing wins.
  * `swath_sigma` -- corridor proximity times sigma^2: geometry x uncertainty, the simplest
    task-region-masked uncertainty.
  * `corridor_mi` -- the entropy_mi field masked by the same corridor weight. THIS IS THE
    DECISIVE ABLATION. It is task-region masking WITHOUT any derivative, which is what the
    nearest prior art does (SPOT 2510.16308 masks uncertainty by trajectory proximity; the
    risk-averse NBV of 2510.06481 masks entropy the same way). The paper's claim is that
    weighting by the COST GRADIENT's disagreement buys something over masking by where the
    plans go. If corridor_mi matches disagreement, that claim is dead and the honest report
    is that the contribution is the mask, not the adjoint.

A second question this module can answer that the cell-budget study cannot: whether the
motion-coupled collapse of the effect (+0.633 free-look -> +0.045 here) is a property of the
METHOD or of the SENSOR. A 120 deg / 8 m look on a 9 m map sees nearly everything reachable,
so every policy's cone overlaps every other's. `--fov` / `--range` / `--looks` parameterise
exactly that ratio, and `--mode sweep` walks it.

`oracle` is the ceiling, not a policy: it EXECUTES every candidate viewpoint against ground
truth and keeps the one whose re-ranking scores best. It peeks at the answer. It bounds what any
viewpoint chooser could achieve with this action set, which is the number that says whether a
policy gap is small because the policy is weak or because the action set is coarse.

A reveal hands over the MEASUREMENT, not the truth (build_case's `measured`), so sensing does
not converge on perfect knowledge -- the same honesty ranking.py adopted after its clean setup
was found to flatter every policy at once.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

from ..adjoint.harness import Harness
from ..bench.bundled import _weighted_adjoint
from ..bench.noise import _bilinear
from ..bench.noise import RAY_STEPS
from ..bench.noise import SENSOR_HEIGHT
from ..bench.ranking import _evaluate
from ..bench.ranking import _plan_distance
from ..bench.ranking import _proximity
from ..bench.ranking import build_case
from ..bench.ranking import CELL
from ..bench.ranking import kendall_tau
from ..bench.ranking import N_PLANS
from ..bench.ranking import OUT as BENCH_OUT
from ..bench.ranking import sign_test

OUT = BENCH_OUT.parent / "sensing"

# --- the sensor: fixed to the chassis, so heading IS the look direction -------------------
FOV_DEG = 120.0  # full width, i.e. +-60 deg about the heading
LOOK_RANGE = 8.0  # [m] a reveal past this is not credited
# Measurement noise floor entering the Gaussian MI form. The sensor arm's actual per-cell std in
# bench/noise.py is range-dependent (SENSOR_BASE=0.010 m at zero range, growing to ~0.046 m at
# MAX_RANGE=6 m); 0.05 m is a fixed single-number approximation of that, so a cell already at the
# noise floor is worth ~zero bits, as it should be.
SIGMA_MEASURE = 0.05  # [m]

# --- the action set --------------------------------------------------------------------
N_BEARINGS = 12
BEARING_SPAN = np.radians(270.0)  # +-135 deg; stops short of straight behind
RADII = (0.0, 0.7, 1.4)  # [m] 0 = turn in place, then two creep distances
TURN_RATE = 1.0  # [rad/s] time charged for the turn
CREEP_SPEED = 0.7  # [m/s] time charged for the translation

POLICY_NAMES = (
    "none",
    "random",
    "entropy",
    "entropy_mi",
    "swath_var",
    "swath_sigma",
    "corridor_mi",
    "disagreement",
    "oracle",
)
SCORE_NAMES = (
    "random",
    "entropy",
    "entropy_mi",
    "swath_var",
    "swath_sigma",
    "corridor_mi",
    "disagreement",
)
NOT_A_POLICY = ("oracle",)


def wrap(a: np.ndarray | float) -> np.ndarray | float:
    """Angle(s) into (-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))


def visible_cells(
    elev2d: np.ndarray,
    grid: tuple[np.ndarray, np.ndarray],
    pose_xy: tuple[float, float],
    heading: float,
    fov_deg: float = FOV_DEG,
    max_range: float = LOOK_RANGE,
    cell: float = CELL,
) -> np.ndarray:
    """Cells a look from `pose_xy` along `heading` resolves, ray-cast against `elev2d`.

    The same 2.5-D line-of-sight test as `bench/noise.visibility` -- a cell is seen iff no
    sample along the ray to it subtends a larger elevation angle -- generalised off the origin
    and given a finite field of view, because the sensor here is body-fixed and the viewpoint
    moves. Cast against whichever map is passed: the BELIEF when a policy is scoring a candidate
    (that is all it knows), the TRUTH when the chosen look is executed (that is what happens).

    Two deliberate departures from `visibility`, both noted rather than silently absorbed:
      * the sensor rides SENSOR_HEIGHT above the local ground rather than above z = 0, since a
        viewpoint 1.4 m away can sit on relief that the origin does not
      * cell-centre index coordinates are computed without its half-cell offset

    Only cells inside the cone are marched, which is where the runtime goes.
    """
    XX, YY = grid
    px, py = float(pose_xy[0]), float(pose_xy[1])
    dx, dy = XX - px, YY - py
    rng_field = np.hypot(dx, dy)
    delta = np.abs(wrap(np.arctan2(dy, dx) - heading))
    cone = (rng_field <= max_range) & (delta <= np.radians(0.5 * fov_deg))

    # Index coordinates: XX.min() / YY.min() are the CENTRES of cell (0, 0).
    gx_all = (XX - XX.min()) / cell
    gy_all = (YY - YY.min()) / cell
    gx_s = (px - XX.min()) / cell
    gy_s = (py - YY.min()) / cell
    sensor_z = float(_bilinear(elev2d, np.array(gy_s), np.array(gx_s))) + SENSOR_HEIGHT

    gx_c, gy_c = gx_all[cone], gy_all[cone]  # [M] candidate cells, flattened
    rng_c, h_c = rng_field[cone], elev2d[cone]
    frac = np.linspace(0.0, 1.0, RAY_STEPS + 1)[1:-1].reshape(-1, 1)  # [S, 1]
    h_along = _bilinear(elev2d, gy_s + frac * (gy_c - gy_s), gx_s + frac * (gx_c - gx_s))
    ang_along = (h_along - sensor_z) / np.maximum(frac * rng_c, 1e-6)
    ang_target = (h_c - sensor_z) / np.maximum(rng_c, 1e-6)
    seen = ~(ang_along > ang_target[None] + 1e-4).any(axis=0)

    out = np.zeros(XX.shape, bool)
    out[cone] = seen
    return out


def candidate_viewpoints(
    pose: np.ndarray, grid: tuple[np.ndarray, np.ndarray], cell: float = CELL
) -> tuple[np.ndarray, np.ndarray]:
    """The (x, y, heading) the body can reach with one manoeuvre, and the time each costs.

    Turning in place is r = 0: twelve headings from one position. The two creep radii place the
    sensor somewhere else as well, which is the only way to see round a ridge. Heading equals
    the bearing driven, because a differential drive that goes there ends up facing that way and
    the lidar cannot look anywhere else.
    """
    x0, y0, yaw0 = float(pose[0]), float(pose[1]), float(pose[2])
    XX, YY = grid
    lo_x, hi_x = XX.min() - 0.5 * cell, XX.max() + 0.5 * cell
    lo_y, hi_y = YY.min() - 0.5 * cell, YY.max() + 0.5 * cell
    vps, costs = [], []
    for b in yaw0 + np.linspace(-BEARING_SPAN / 2, BEARING_SPAN / 2, N_BEARINGS):
        for r in RADII:
            x, y = x0 + r * np.cos(b), y0 + r * np.sin(b)
            if not (lo_x <= x <= hi_x and lo_y <= y <= hi_y):
                continue  # off the map: the rollouts there are not defined
            vps.append((x, y, float(b)))
            costs.append(abs(float(wrap(b - yaw0))) / TURN_RATE + r / CREEP_SPEED)
    return np.array(vps, np.float32), np.array(costs, float)


def _pick(gain: np.ndarray, time_cost: np.ndarray, tie_rank: np.ndarray) -> int:
    """Argmax gain, ties broken by cheaper look and then by a per-seed random order.

    ranking.py's rationale, transplanted: a policy whose objective plateaus over the action set
    (entropy's cone integral barely varies with bearing on open ground) would otherwise be
    ranked by the order the viewpoints happen to be enumerated in, which is a straw man rather
    than a baseline. np.lexsort's LAST key is the primary one.
    """
    return int(np.lexsort((tie_rank, time_cost, -gain))[0])


def _reveal(
    belief: np.ndarray, meas: np.ndarray, obs: np.ndarray, vis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Execute a look: the newly seen cells take their MEASURED value, not the truth."""
    reveal = vis & ~obs
    belief2 = np.where(reveal, meas, belief).astype(np.float32)
    assert not (reveal & obs).any()
    assert np.array_equal(belief2[~reveal], belief[~reveal])
    assert np.array_equal(belief2[reveal], meas[reveal])
    return belief2, reveal


def run_seed(
    seed: int,
    family: str,
    noise: str,
    want_dump: bool,
    fov_deg: float = FOV_DEG,
    look_range: float = LOOK_RANGE,
    n_looks: int = 1,
) -> tuple[dict, dict | None]:
    scene, truth, meas, obs, sigma, poses, omega, grid = build_case(seed, family, noise)
    # Every plan in this family starts where the robot is; the viewpoints are offsets from it.
    assert np.allclose(poses, poses[0]), "plans must share one start pose"
    robot_pose = poses[0][:3].astype(float)
    belief = scene.elevation.astype(np.float32)
    XX, YY = grid

    h = Harness(scene, poses, omega, device="cuda")
    # One taped backward for the full weighted cost. The trajectories it leaves in `controlled`
    # are the BELIEVED paths, which is what a geometric policy is entitled to read.
    grad, j_belief = _weighted_adjoint(h, belief)
    traj = h.sim.controlled.numpy()[:, :, :2].copy()  # [T+1, K, 2]
    dist = _plan_distance(h, grid)  # [K, ny, nx]

    # GATE 1: the adjoint's support must lie under the plans. If it does not, the tape is not
    # measuring what it claims to and every score built on it is meaningless.
    assert np.isfinite(grad).all(), "non-finite weighted adjoint"
    mass = np.abs(grad).sum(axis=0)
    near = dist.min(axis=0) <= 1.0  # [m] of the nearest plan
    frac_near = float(mass[near].sum() / max(mass.sum(), 1e-30))
    assert frac_near > 0.90, f"adjoint support is off the plans: {frac_near:.3f} within 1 m"

    j_true = _evaluate(h, truth)
    tau_before = kendall_tau(j_belief, j_true)

    rng = np.random.default_rng(20_000 + seed)
    unobs = ~obs
    # Fields that do NOT depend on the belief -- sigma and plan geometry are fixed by the case,
    # so a sequential look only shrinks `unobs`, never these. `entropy_mi` is the proper
    # Gaussian mutual information (concave in sigma^2, unlike raw `entropy`); `corridor_mi` and
    # `swath_sigma` mask an uncertainty field by the same plan-proximity kernel `swath_var`
    # already uses, with no derivative -- corridor_mi is the decisive ablation (see docstring).
    prox = _proximity(dist)
    prox_max = prox.max(axis=0)
    entropy_mi_field = 0.5 * np.log1p(sigma**2 / SIGMA_MEASURE**2)
    static_scores = {
        "entropy": sigma**2,
        "entropy_mi": entropy_mi_field,
        "swath_var": prox.var(axis=0),
        "swath_sigma": prox_max * sigma**2,
        "corridor_mi": prox_max * entropy_mi_field,
    }
    # `disagreement` is the only score coupled to the belief, through the adjoint. Only its
    # look-0 value is needed here for the GATE/dump snapshot below; extra looks recompute it.
    static_scores_disagreement0 = grad.var(axis=0) * sigma**2
    # Revealing an already-observed cell is a no-op, so it must not earn a policy any gain.
    # Kept only for the frozen npz dump keys (score_entropy / score_swath_var /
    # score_disagreement), which snapshot the look-0 field exactly as before this change.
    look0_masked = {
        "entropy": np.where(unobs, static_scores["entropy"], 0.0),
        "swath_var": np.where(unobs, static_scores["swath_var"], 0.0),
        "disagreement": np.where(unobs, static_scores_disagreement0, 0.0),
    }

    vps, time_cost = candidate_viewpoints(robot_pose, grid)
    tie_rank = np.argsort(rng.permutation(len(vps)))
    vis_belief = [visible_cells(belief, grid, v[:2], float(v[2]), fov_deg, look_range) for v in vps]
    vis_truth = [visible_cells(truth, grid, v[:2], float(v[2]), fov_deg, look_range) for v in vps]

    row = {"seed": seed, "tau_before": tau_before, "policies": {}}
    dump: dict = {}

    def _commit(j_after: np.ndarray) -> tuple[float, float, bool]:
        pick = int(np.argmin(j_after))  # the commit ignores sigma: a planner picks the argmin
        return (
            kendall_tau(j_after, j_true),
            float(j_true[pick] - j_true.min()),
            bool(pick == int(np.argmin(j_true))),
        )

    for name in SCORE_NAMES:
        belief_i, obs_i, grad_i = belief, obs, grad
        total_time, total_revealed, total_gain, first_vp = 0.0, 0, 0.0, -1
        reveal = None
        # Sequential greedy looks: score, pick, reveal, re-score on what remains unobserved.
        # `vis_belief` / `vis_truth` stay fixed (same 36 viewpoints, same start pose); only the
        # score field and the shrinking `unobs` mask change look to look.
        for look in range(n_looks):
            unobs_i = ~obs_i
            if name == "random":
                raw = rng.random(sigma.shape)
            elif name == "disagreement":
                if look > 0:  # look 0 reuses the adjoint already taped above
                    grad_i, _ = _weighted_adjoint(h, belief_i)
                raw = grad_i.var(axis=0) * sigma**2
            else:
                raw = static_scores[name]
            s = np.where(unobs_i, raw, 0.0)
            gain = np.array([float(s[vb & unobs_i].sum()) for vb in vis_belief])
            v = _pick(gain, time_cost, tie_rank)
            belief_i, reveal = _reveal(belief_i, meas, obs_i, vis_truth[v])
            obs_i = obs_i | reveal
            total_time += float(time_cost[v])
            total_revealed += int(reveal.sum())
            total_gain += float(gain[v])
            if look == 0:
                first_vp = v
        j_after = _evaluate(h, belief_i)
        tau, regret, top1 = _commit(j_after)
        row["policies"][name] = {
            "tau": tau,
            "regret": regret,
            "top1": top1,
            "time_cost": total_time,
            "n_revealed": total_revealed,
            "gain": total_gain,
            "vp": int(first_vp),  # first look's viewpoint, so "did they even differ" is measured
        }
        if want_dump:
            dump[f"vp_{name}"] = vps[first_vp]
            dump[f"jafter_{name}"] = j_after.astype(np.float32)
            dump[f"tau_{name}"] = tau
            if name != "random":
                dump[f"reveal_{name}"] = reveal
            if name == "disagreement":
                dump["belief_after_disagreement"] = belief_i

    # `none` never looks: the belief is unchanged, so this is tau_before by construction.
    tau, regret, top1 = _commit(j_belief)
    row["policies"]["none"] = {
        "tau": tau,
        "regret": regret,
        "top1": top1,
        "time_cost": 0.0,
        "n_revealed": 0,
        "gain": 0.0,
        "vp": -1,  # no look taken
    }

    # The ceiling: execute EVERY viewpoint against truth and keep the best re-ranking, `n_looks`
    # times greedily. This reads the answer, so it is not a policy -- it is the bound on this
    # action set (per look, since it must re-peek at every candidate each time it is extended).
    belief_o, obs_o = belief, obs
    total_time_o, total_revealed_o, first_vp_o = 0.0, 0, -1
    j_cur, gain_o = j_belief, 0.0
    for look in range(n_looks):
        unobs_o = ~obs_o
        taus_v = np.empty(len(vps))
        j_v = []
        for i, vt in enumerate(vis_truth):
            belief2, _ = _reveal(belief_o, meas, obs_o, vt)
            j_v.append(_evaluate(h, belief2))
            taus_v[i] = kendall_tau(j_v[-1], j_true)
        v = _pick(taus_v, time_cost, tie_rank)
        belief_o, reveal_o = _reveal(belief_o, meas, obs_o, vis_truth[v])
        obs_o = obs_o | reveal_o
        total_time_o += float(time_cost[v])
        total_revealed_o += int(reveal_o.sum())
        j_cur, gain_o = j_v[v], float(taus_v[v])
        if look == 0:
            first_vp_o = v
    tau, regret, top1 = _commit(j_cur)
    row["policies"]["oracle"] = {
        "tau": tau,
        "regret": regret,
        "top1": top1,
        "time_cost": total_time_o,
        "n_revealed": total_revealed_o,
        "gain": gain_o,
        "vp": int(first_vp_o),
    }

    if want_dump:
        dump.update(
            truth=truth.astype(np.float32),
            belief=belief,
            sigma=sigma.astype(np.float32),
            meas=meas.astype(np.float32),
            obs=obs,
            cell=float(CELL),
            origin=np.array([scene.origin_x, scene.origin_y], float),
            robot_pose=robot_pose,
            traj=traj.astype(np.float32),
            j_true=j_true.astype(np.float32),
            j_belief=j_belief.astype(np.float32),
            score_entropy=look0_masked["entropy"].astype(np.float32),
            score_swath_var=look0_masked["swath_var"].astype(np.float32),
            score_disagreement=look0_masked["disagreement"].astype(np.float32),
            vp_oracle=vps[first_vp_o],
            jafter_none=j_belief.astype(np.float32),
            jafter_oracle=j_cur.astype(np.float32),
            tau_none=row["policies"]["none"]["tau"],
            tau_oracle=tau,
            tau_before=tau_before,
            viewpoints=vps,
        )
    del h
    return row, (dump if want_dump else None)


def report(rows: list[dict]) -> None:
    n = len(rows)
    print(f"\nn={n} seeds   one look each, chosen from {N_BEARINGS} bearings x {len(RADII)} radii")
    tb = np.mean([r["tau_before"] for r in rows])
    print(f"belief ranking before any look: Kendall tau {tb:+.3f}")
    print(f"\ntau  = agreement of the post-look ranking of the {N_PLANS} plans with the truth")
    print("reg  = true excess cost of the plan a planner would then commit to (0 = optimal)")
    print("time = seconds of turning and creeping the look cost\n")
    print(f"{'policy':<14}{'tau':>9}{'regret':>9}{'top1':>7}{'time':>8}{'cells':>8}")

    def mean_of(name: str, key: str) -> float:
        return float(np.mean([r["policies"][name][key] for r in rows]))

    for name in POLICY_NAMES:
        line = (
            f"{name:<14}{mean_of(name, 'tau'):>+9.3f}{mean_of(name, 'regret'):>9.3f}"
            f"{mean_of(name, 'top1'):>6.0%}{mean_of(name, 'time_cost'):>8.2f}"
            f"{mean_of(name, 'n_revealed'):>8.0f}"
        )
        print(line + ("   <- ceiling, not a policy" if name in NOT_A_POLICY else ""))

    # A tau gap means nothing if the policies picked the same look; this says whether they did.
    print("\nhow often each policy chose the SAME viewpoint as another (of 36 candidates):")
    for a in SCORE_NAMES:
        same = {
            b: np.mean([r["policies"][a]["vp"] == r["policies"][b]["vp"] for r in rows])
            for b in (*SCORE_NAMES, "oracle")
            if b != a
        }
        print("  " + f"{a:<14}" + "  ".join(f"{b} {v:.0%}" for b, v in same.items()))

    print("\npaired per seed, X minus Y in tau (positive = X better):")
    for a, b in (
        ("disagreement", "entropy"),
        ("disagreement", "entropy_mi"),
        ("disagreement", "swath_var"),
        ("disagreement", "swath_sigma"),
        ("disagreement", "corridor_mi"),
        ("disagreement", "random"),
        ("disagreement", "none"),
        ("oracle", "disagreement"),
    ):
        d = np.array([r["policies"][a]["tau"] - r["policies"][b]["tau"] for r in rows])
        k, w, p = sign_test(d)
        print(f"  {a:<14} - {b:<14}{d.mean():>+8.3f}   better on {w:>3}/{k:<3}   p={p:.2e}")


def _paired(rows: list[dict], a: str, b: str) -> tuple[float, int, int, float]:
    """Mean(a - b) in tau plus the sign test, reused by the sweep and holdout reports."""
    d = np.array([r["policies"][a]["tau"] - r["policies"][b]["tau"] for r in rows])
    k, w, p = sign_test(d)
    return float(d.mean()), k, w, p


# --- Task 2: does the motion-coupled collapse depend on the sensor/world ratio? ------------
# fov x range at looks=1, plus the default sensor at looks in {2, 3} -- 11 configs total.
SWEEP_FOV_DEG = (30.0, 60.0, 120.0)
SWEEP_RANGE_M = (2.0, 4.0, 8.0)
SWEEP_EXTRA_LOOKS = (2, 3)


def run_sweep(family: str, noise: str, seeds: range) -> dict:
    configs = [(fov, rng, 1) for fov in SWEEP_FOV_DEG for rng in SWEEP_RANGE_M]
    configs += [(FOV_DEG, LOOK_RANGE, looks) for looks in SWEEP_EXTRA_LOOKS]
    sweep: dict[str, dict] = {}
    for fov, look_range, looks in configs:
        key = f"fov{fov:.0f}_range{look_range:.0f}_looks{looks}"
        rows = [run_seed(s, family, noise, False, fov, look_range, looks)[0] for s in seeds]
        d_mean, k, w, p = _paired(rows, "disagreement", "corridor_mi")
        sweep[key] = {
            "fov": fov,
            "range": look_range,
            "looks": looks,
            "rows": rows,
            "disagreement_minus_corridor_mi": {"mean": d_mean, "n": k, "wins": w, "p": p},
        }
        means = {n: float(np.mean([r["policies"][n]["tau"] for r in rows])) for n in POLICY_NAMES}
        tau_str = "  ".join(f"{n} {means[n]:+.3f}" for n in POLICY_NAMES)
        print(f"{key}:  {tau_str}")
        print(f"    disagreement - corridor_mi = {d_mean:+.3f}   better on {w}/{k}   p={p:.2e}")
    return sweep


# --- Task 3: pre-registered holdout confirmation --------------------------------------------
def run_holdout(
    family: str, noise: str, config_b: tuple[float, float, int], config_b_name: str
) -> dict:
    """Design-set (already-seen seeds) vs holdout (virgin seeds 200-319), both frozen configs.

    Config A is always the default (fov=120, range=8, looks=1); config B is whichever sweep
    config showed the largest design-set disagreement-minus-corridor_mi gap. Both are re-run
    fresh on both seed ranges rather than read from cached json, so this function is a complete
    record of exactly what was compared.
    """
    config_a: tuple[float, float, int] = (FOV_DEG, LOOK_RANGE, 1)
    design_a_seeds = range(120)  # the pre-registered design set for the default config
    design_b_seeds = range(100)  # the sweep's design set
    holdout_seeds = range(200, 320)  # virgin: never touched by Task 1 or Task 2

    def run_all(cfg: tuple[float, float, int], seeds: range) -> list[dict]:
        return [run_seed(s, family, noise, False, *cfg)[0] for s in seeds]

    design_a = run_all(config_a, design_a_seeds)
    design_b = run_all(config_b, design_b_seeds)
    holdout_a = run_all(config_a, holdout_seeds)
    holdout_b = run_all(config_b, holdout_seeds)

    pairs = (("disagreement", "corridor_mi"), ("disagreement", "entropy_mi"), ("disagreement", "entropy"))
    result = {
        "config_a": {"fov": config_a[0], "range": config_a[1], "looks": config_a[2], "name": "default"},
        "config_b": {
            "fov": config_b[0],
            "range": config_b[1],
            "looks": config_b[2],
            "name": config_b_name,
        },
        "design_a": design_a,
        "design_b": design_b,
        "holdout_a": holdout_a,
        "holdout_b": holdout_b,
    }
    print(f"\nholdout confirmation: config A = default, config B = {config_b_name}")
    for label, rows in (
        ("design A", design_a),
        ("holdout A", holdout_a),
        ("design B", design_b),
        ("holdout B", holdout_b),
    ):
        print(f"\n{label}  (n={len(rows)}):")
        for a, b in pairs:
            d_mean, k, w, p = _paired(rows, a, b)
            print(f"  {a} - {b:<12}{d_mean:>+8.3f}   better on {w:>3}/{k:<3}   p={p:.2e}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("single", "sweep", "holdout"), default="single")
    ap.add_argument("--seeds", type=int, default=120)
    ap.add_argument("--family", default="hybrid")
    ap.add_argument("--noise", default="all")
    ap.add_argument("--fov", type=float, default=FOV_DEG, help="full look FOV in degrees")
    ap.add_argument("--range", dest="look_range", type=float, default=LOOK_RANGE, help="look range, m")
    ap.add_argument("--looks", type=int, default=1, help="sequential greedy looks per case")
    ap.add_argument(
        "--holdout-name",
        default="config_b",
        help="label for --fov/--range/--looks as config B in --mode holdout",
    )
    ap.add_argument(
        "--dump-seed",
        type=int,
        default=None,
        help="seed to write a showcase npz for; default = the most discriminating of the first 20",
    )
    a = ap.parse_args()

    wp.init()
    OUT.mkdir(parents=True, exist_ok=True)

    if a.mode == "sweep":
        sweep = run_sweep(a.family, a.noise, range(100))
        path = OUT / "sweep.json"
        path.write_text(json.dumps(sweep, indent=2))
        print(f"\nwrote {path}")
        return

    if a.mode == "holdout":
        result = run_holdout(a.family, a.noise, (a.fov, a.look_range, a.looks), a.holdout_name)
        path = OUT / "holdout.json"
        path.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {path}")
        return

    rows = []
    for seed in range(a.seeds):
        rows.append(run_seed(seed, a.family, a.noise, False, a.fov, a.look_range, a.looks)[0])
        if (seed + 1) % 10 == 0:
            print(f"  {seed + 1}/{a.seeds} seeds", flush=True)
    report(rows)
    path = OUT / "results.json"
    path.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\nwrote {path}")

    dump_seed = a.dump_seed
    if dump_seed is None:
        # The seed where the two hypotheses disagree most is the one worth LOOKING at; the mean
        # says whether the effect exists, a picture has to show what it looks like.
        head = rows[:20]
        gaps = [
            r["policies"]["disagreement"]["tau"] - r["policies"]["entropy"]["tau"] for r in head
        ]
        dump_seed = head[int(np.argmax(gaps))]["seed"]
    _, dump = run_seed(dump_seed, a.family, a.noise, True, a.fov, a.look_range, a.looks)
    np.savez_compressed(OUT / f"case_seed{dump_seed}.npz", **dump)
    print(f"wrote {OUT / f'case_seed{dump_seed}.npz'}  (seed {dump_seed})")


if __name__ == "__main__":
    main()
