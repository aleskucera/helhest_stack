"""Does DRIVING constrain what the NEXT DECISION needs? (identifiability vs decision-relevance)

    .venv/bin/python -m studies.bench.observability --seeds 40

Self-supervised terrain learning (MonoForce-style) fits terrain from what the robot already
drove: the training signal is only informative along directions of h that move the driven
trajectory's outputs. A planner's next decision instead needs terrain along the directions
given by the plan-cost adjoint over the CANDIDATE set it is choosing among. These two sets of
directions need not coincide -- a cell three metres off the driven line is unconstrained by
that drive, but may sit under a candidate plan the planner is about to consider.

Per seed (`build_case(seed, "hybrid", "all")`, the same fixture `ranking.py` uses):

  1. DRIVEN trajectory: the true-best of the K=16 candidate plans (argmin cost on the TRUTH
     map -- "what the robot would have driven, in hindsight"), rolled out on the TRUTH map.
  2. obs_mass_i: trajectory-observability mass per cell. A Hutchinson probe of
     diag(Gramian) of the driven rollout's outputs (controlled x,y,yaw AND derived
     z,pitch,roll, all T+1 steps) w.r.t. elevation: R random UNIT cotangent vectors v drawn
     isotropically over the flattened output space, backprop each via the harness's own tape
     (the same `tape.backward(grads={...})` primitive `backward_from_cotangents` wraps),
     read `elevation.grad`, accumulate grad^2. For an isotropic unit v of total dimension D,
     E[(v^T J)_i^2] = G_i / D -- proportional to the true Gramian diagonal, so mean_r grad_i^2
     is used directly (only the spatial PATTERN of obs_mass is used below, never its absolute
     scale, so the 1/D factor is immaterial).
  3. rel_mass_i: decision-relevance mass per cell. Var_k(dJ_k/dh_i) over the K=16 candidate
     plans' cost adjoints, evaluated on the BELIEF map in one weighted backward pass
     (`bundled._weighted_adjoint`) -- the disagreement field the next sensing/planning
     decision cares about.
  4. Overlap: how much of the decision-relevant mass sits on cells the drive left
     unconstrained (obs_mass ~ 0), overall and restricted to cells still unobserved by any
     sensor -- and the spatial picture (how far the top decision-relevant cells sit from the
     driven track).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import warp as wp

from ..adjoint.harness import Harness
from ..adjoint.scene import Scene
from .bundled import _weighted_adjoint
from .ranking import _evaluate
from .ranking import build_case
from .ranking import OUT

R_PROBES = 6  # Hutchinson probes for the trajectory-observability Gramian diagonal
# On-track radius: half_track (0.365) + wheel_radius (0.35) -- the reach of a single wheel
# contact off the trajectory centreline (see studies/adjoint/scene.py). Cells farther than
# this are never touched by any wheel of the driven plan, so obs_mass there should be a
# structural, not merely numerical, zero -- the sanity gate below checks exactly that.
TRACK_RADIUS_M = 0.72


def _track_distance(traj_xy: np.ndarray, XX: np.ndarray, YY: np.ndarray) -> np.ndarray:
    """[ny, nx] distance from every cell to the nearest waypoint of one [T+1, 2] trajectory."""
    d2 = (XX[None] - traj_xy[:, 0, None, None]) ** 2 + (YY[None] - traj_xy[:, 1, None, None]) ** 2
    return np.sqrt(d2.min(axis=0))


def _obs_mass(
    scene: Scene, pose: np.ndarray, omega: np.ndarray, device: str, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Hutchinson-probed diag(Gramian) [ny, nx] of one driven rollout's outputs w.r.t.
    elevation, plus the rolled-out trajectory [T+1, 2] (x, y) for the sanity gate."""
    h = Harness(scene, pose[None, :], omega[:, None, :], device=device)
    tape = wp.Tape()
    h._rollout(dilate=True, tape=tape)
    traj_xy = h.sim.controlled.numpy()[:, 0, :2].copy()

    t1 = h.n_steps + 1
    accum = np.zeros(scene.shape, np.float64)
    adj_controlled = wp.zeros_like(h.sim.controlled)
    adj_derived = wp.zeros_like(h.sim.derived)
    for _ in range(R_PROBES):
        v = rng.standard_normal((t1, 2, 3))  # [T+1, {controlled, derived}, 3]
        v /= np.linalg.norm(v)
        adj_controlled.assign(v[:, 0, :][:, None, :].astype(np.float32))
        adj_derived.assign(v[:, 1, :][:, None, :].astype(np.float32))
        tape.zero()
        tape.backward(grads={h.sim.controlled: adj_controlled, h.sim.derived: adj_derived})
        accum += h.sim.elevation.grad.numpy()[0].astype(np.float64) ** 2
    tape.zero()
    del h
    return accum / R_PROBES, traj_xy


def run_seed(seed: int, family: str, noise: str) -> dict:
    scene, truth, _measured, observed, _sigma, poses, omega, (XX, YY) = build_case(
        seed, family, noise
    )
    belief = scene.elevation.astype(np.float32)

    # --- the K=16 candidate set on the BELIEF: who is the true-best plan, and its disagreement
    h = Harness(scene, poses, omega, device="cuda")
    j_true = _evaluate(h, truth)
    best_k = int(np.argmin(j_true))

    grad_bel, _ = _weighted_adjoint(h, belief)  # [K, ny, nx], dJ_k/dh on the BELIEF
    rel_mass = grad_bel.var(axis=0)
    device = h.device
    del h

    # --- the DRIVEN trajectory: the true-best plan, rolled out on the TRUTH map ---------
    truth_scene = Scene(truth, scene.friction, scene.region, scene.cell, scene.origin_x, scene.origin_y)
    rng = np.random.default_rng(500_000 + seed)
    obs_mass, traj_xy = _obs_mass(truth_scene, poses[best_k], omega[:, best_k, :], device, rng)

    track_dist = _track_distance(traj_xy, XX, YY)
    on_track = track_dist <= TRACK_RADIUS_M
    off_track = ~on_track

    # --- sanity gate: is obs_mass actually concentrated near the driven track? ----------
    obs_total = float(obs_mass.sum())
    obs_frac_on_track = float(obs_mass[on_track].sum() / max(obs_total, 1e-300))
    top100_obs = np.argsort(obs_mass.ravel())[-100:]
    obs_top100_dist = float(track_dist.ravel()[top100_obs].mean())

    # --- thresholds for "unconstrained by driving" -------------------------------------
    thr_fixed = 1e-6 * float(obs_mass.max())
    thr_adaptive = float(np.percentile(obs_mass[off_track], 99)) if off_track.any() else thr_fixed
    # `<=` matters here: off-track obs_mass is a STRUCTURAL zero (no wheel of the driven plan
    # ever reaches those cells), so `thr_adaptive` is exactly 0.0 in practice and a strict `<`
    # would exclude every such cell instead of flagging it as unconstrained.
    unconstrained_fixed = obs_mass <= thr_fixed
    unconstrained_adaptive = obs_mass <= thr_adaptive

    # --- overlap metrics ------------------------------------------------------------------
    rel_total = float(rel_mass.sum())
    frac_unconstrained_fixed = float(rel_mass[unconstrained_fixed].sum() / max(rel_total, 1e-300))
    frac_unconstrained_adaptive = float(
        rel_mass[unconstrained_adaptive].sum() / max(rel_total, 1e-300)
    )

    unobserved = ~observed
    rel_unobs_total = float(rel_mass[unobserved].sum())
    frac_unobs_fixed = float(
        rel_mass[unobserved & unconstrained_fixed].sum() / max(rel_unobs_total, 1e-300)
    )
    frac_unobs_adaptive = float(
        rel_mass[unobserved & unconstrained_adaptive].sum() / max(rel_unobs_total, 1e-300)
    )

    top100_rel = np.argsort(rel_mass.ravel())[-100:]
    rel_top100_dist = float(track_dist.ravel()[top100_rel].mean())

    return {
        "seed": seed,
        "best_k": best_k,
        "obs_frac_on_track": obs_frac_on_track,
        "obs_top100_dist_to_track_m": obs_top100_dist,
        "thr_fixed": thr_fixed,
        "thr_adaptive": thr_adaptive,
        "frac_rel_mass_unconstrained_fixed": frac_unconstrained_fixed,
        "frac_rel_mass_unconstrained_adaptive": frac_unconstrained_adaptive,
        "frac_rel_mass_unconstrained_fixed_unobserved": frac_unobs_fixed,
        "frac_rel_mass_unconstrained_adaptive_unobserved": frac_unobs_adaptive,
        "rel_top100_dist_to_track_m": rel_top100_dist,
        "n_unobserved_cells": int(unobserved.sum()),
        "n_on_track_cells": int(on_track.sum()),
    }


def _dist(rows: list[dict], key: str) -> tuple[float, float, float]:
    v = np.array([r[key] for r in rows])
    return float(np.median(v)), float(np.percentile(v, 10)), float(np.percentile(v, 90))


def report(rows: list[dict]) -> None:
    n = len(rows)
    print(f"\nn={n} seeds")

    print("\nSANITY GATE -- is obs_mass concentrated on/near the driven track?")
    m, p10, p90 = _dist(rows, "obs_frac_on_track")
    print(f"  fraction of obs_mass within {TRACK_RADIUS_M:.2f} m of the track:  "
          f"median {m:.3f}  [p10 {p10:.3f}, p90 {p90:.3f}]")
    m, p10, p90 = _dist(rows, "obs_top100_dist_to_track_m")
    print(f"  mean distance of the top-100 obs_mass cells to the track [m]:  "
          f"median {m:.3f}  [p10 {p10:.3f}, p90 {p90:.3f}]")
    seed0 = rows[0]
    print(f"  seed 0 detail: on-track frac={seed0['obs_frac_on_track']:.3f}, "
          f"top-100 dist={seed0['obs_top100_dist_to_track_m']:.3f} m")

    print("\nDECISION-RELEVANT MASS UNCONSTRAINED BY DRIVING (fraction of rel_mass on cells "
          "where obs_mass ~ 0)")
    for label, key in (
        ("overall, fixed thr (1e-6 x max)", "frac_rel_mass_unconstrained_fixed"),
        ("overall, adaptive thr (p99 off-track)", "frac_rel_mass_unconstrained_adaptive"),
        ("unobserved-only, fixed thr", "frac_rel_mass_unconstrained_fixed_unobserved"),
        ("unobserved-only, adaptive thr", "frac_rel_mass_unconstrained_adaptive_unobserved"),
    ):
        m, p10, p90 = _dist(rows, key)
        print(f"  {label:<40}median {m:.3f}  [p10 {p10:.3f}, p90 {p90:.3f}]")

    print("\nSPATIAL PICTURE")
    m, p10, p90 = _dist(rows, "rel_top100_dist_to_track_m")
    print(f"  mean distance of the top-100 rel_mass cells to the driven track [m]:  "
          f"median {m:.3f}  [p10 {p10:.3f}, p90 {p90:.3f}]")


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
    path = OUT / "observability.json"
    path.write_text(json.dumps({"family": a.family, "noise": a.noise, "rows": rows}, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
