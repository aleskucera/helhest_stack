"""C5: does the REAL MPPI elite set carry enough candidate-diversity for the disagreement signal
to survive, or is it closer to the signal-starved `speed` family than to `hybrid`/`fan`?

    .venv/bin/python -m studies.bench.elites --seeds 60

`ranking.py`'s families are all HAND-BUILT: `fan` (16 separate corridors), `hybrid` (4 paths x 4
speed profiles), `speed` (1 path, 16 timings). CLAIMS.md's C1/C3 sensing edge was measured on
those. The unmeasured compounding risk (significance referee): this stack's actual MPPI perturbs
ONE lattice-chosen nominal with Gaussian knot jitter (+ a WIDE global-search prior) and keeps the
CEM top-k -- real elites might sit much closer together than any synthetic family, shrinking the
disagreement score toward `p_disagreement`'s failure mode (a flat field with nothing to rank).

MECHANISM: for each seed, build the study's belief terrain (`ranking.build_case`, family-
independent -- see below), run the real `MppiGpu` on it toward a fixed goal for one full replan
cycle, and take the top-16 lowest-cost final-refine candidates as the elite set -- this stack's
actual candidate-selection output, not a re-derived approximation of it.

CONVENTION NOTE (read before trusting the numbers): `MppiGpu`'s sampler always commands the REAR
wheel at 0 (`_sample_target_wheel_omega_kernel`, "rear -> 0") -- that is the physics MPPI itself
costs candidates on. The closed-loop demos instead send the EXECUTED command's rear wheel at
mean(wL, wR) (see demos/eval.py's `cmd = [..., 0.5*(u[0,0]+u[0,1])]`) -- but that convention is
applied only to the one nominal actually driven, never to the candidate pool. Since the ask is
"the real planner's elite set", elites here keep MPPI's own internal rear=0 convention: it is
what the planner itself rolled out and scored. `ranking.py`'s synthetic families use rear=mean
instead (arbitrarily -- there was no real planner to match), so this is a genuine, disclosed
physics difference between "family" candidate sets, not a bug.

REUSE, NOT REIMPLEMENTATION: `ranking.build_case`'s only family-dependent line is
`omega = PLAN_FAMILIES[family](rng)` -- everything upstream of it (the terrain, the belief, sigma)
is a pure function of (seed, noise, flat_sigma) and does not depend on `family` at all (verified by
reading the function body: `rng` is untouched before that line). So this module registers a new
family "mppi_elites" into `ranking.PLAN_FAMILIES`/`PLAN_GROUPS` at runtime (mutating the imported
module's dict, not editing its file) whose factory returns a pre-computed elite set cached by seed.
Every downstream consumer -- `run_seed`, `report`, the whole POLICIES sweep and its paired sign
tests -- then runs completely unmodified over the real elites, exactly as it does for `fan`/
`hybrid`/`speed`. This is what lets step 5 (the sensing comparison) reuse ranking.py's machinery
verbatim rather than re-deriving it.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import warp as wp
from helhest import dynamics
from helhest.control.mppi import CostParams
from helhest.control.mppi import MppiGpu
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.planning.costtogo import CostToGo

from . import ranking as rk
from .bundled import _weighted_adjoint

GOAL_XY = (4.5, 0.0)  # ~4.5 m ahead of the shared (0,0,0) start, inside the 9x9 belief patch
N_ROLLOUTS = 4096  # MPPI candidate pool: production order of magnitude (demos/eval.py's default B)
N_REFINE = 20  # refine iterations for ONE replan cycle; checked to have converged well before this
N_THETA = 24
FAMILY_TAG = "mppi_elites"
TOP400 = 400  # matches ranking.BUDGETS[-1]: how concentrated the disagreement field is at the
# biggest look budget any policy is scored at


class ElitePlanner:
    """Owns one persistent MppiGpu + CostToGo pair so their CUDA graphs are captured ONCE and
    replayed per seed (recapturing per seed -- the naive way -- costs seconds each; replay costs
    milliseconds). Only `set_terrain`/`compute`'s stable input buffers change between seeds; the
    grid/robot/cost weights are fixed, which is exactly what graph replay requires."""

    def __init__(self, nx: int, ny: int, cell: float, x0: float, y0: float, device: str = "cuda"):
        grid = GridParams(nx, ny, cell, x0, y0)
        self.plan_sim = ForwardSimulator(
            dynamics.robot_params(), dynamics.planning_solver(), grid, N_ROLLOUTS, rk.T_STEPS, device
        )
        self.device = self.plan_sim.device
        self.planner = MppiGpu(self.plan_sim, CostParams(), n_theta=N_THETA)
        self.ctg = CostToGo(
            grid, dynamics.robot_params(), dynamics.planning_solver(), n_theta=N_THETA, device=device
        )
        # must be set before the FIRST replan captures the graph: lattice_cap is baked into the
        # cost kernel's struct arg at capture time, same as demos/eval.py's ordering.
        self.planner.cw.lattice_cap = self.ctg._vcap
        self._grid_struct = grid.build()

    def elites(
        self, elevation: np.ndarray, friction: np.ndarray, goal_xy: tuple[float, float]
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Run one converged replan on `elevation`/`friction` and return the top-`rk.N_PLANS`
        lowest-cost final-refine candidates: (omega [T_STEPS, N_PLANS, 3], poses [N_PLANS, 3],
        a convergence-diagnostic dict)."""
        elev_dev = wp.array(np.ascontiguousarray(elevation, np.float32), device=self.device)
        fric_dev = wp.array(np.ascontiguousarray(friction, np.float32), device=self.device)
        self.plan_sim.set_terrain(elev_dev)
        wp.copy(self.plan_sim.friction, fric_dev)  # bypasses set_friction's Heightmap wrapper --
        # same direct-copy convention studies/adjoint/harness.py's Harness uses for its own terrain.

        V = self.ctg.compute(elev_dev, goal_xy)
        self.planner.set_lattice(V, self._grid_struct)
        self.planner.reset_nominal(1.5)  # cold start per seed: terrains are iid, no warm-start bias
        state = np.array([0.0, 0.0, 0.0], np.float32)  # matches build_case's fixed plan start pose
        self.planner.replan(state, goal_xy, N_REFINE)

        J = self.planner.J.numpy()  # [N_ROLLOUTS] final-refine candidate costs
        omega_pool = self.planner.sim.target_wheel_omega.numpy()  # [T_STEPS, N_ROLLOUTS, 3]
        elite_idx = np.argsort(J)[: rk.N_PLANS]
        elite_omega = np.ascontiguousarray(omega_pool[:, elite_idx, :], np.float32)
        poses = np.tile(state, (rk.N_PLANS, 1)).astype(np.float32)
        diag = {
            "J_elite_min": float(J[elite_idx].min()),
            "J_elite_max": float(J[elite_idx].max()),
            "J_pool_min": float(J.min()),
            "J_pool_median": float(np.median(J)),
        }
        return elite_omega, poses, diag


# --- monkeypatch: register "mppi_elites" as a real ranking.py plan family --------------------
# `build_case` calls `PLAN_FAMILIES[family](rng)` with no seed visible to the factory, so the
# driver loop stashes "which seed is this" in `_CURRENT` right before calling `ranking.run_seed`.
_ELITE_CACHE: dict[int, np.ndarray] = {}
_CURRENT = [-1]


def _elite_family(rng: np.random.Generator) -> np.ndarray:
    return _ELITE_CACHE[_CURRENT[0]]


_elite_family.__doc__ = "REAL MPPI: the top-16 lowest-cost candidates of one converged CEM replan."
rk.PLAN_FAMILIES[FAMILY_TAG] = _elite_family
rk.PLAN_GROUPS[FAMILY_TAG] = np.arange(rk.N_PLANS)  # no known path-sharing structure among real
# elites (unlike hybrid's 4x4 design) -- tau_within degenerates to nan, which is the honest answer.


def _disagreement_field(seed: int, noise: str, scene, sigma: np.ndarray, poses, omega) -> dict:
    """Same score as `ranking.p_disagreement` (Var_k(dJ_k/dh) * sigma^2), but kept as a 2D FIELD
    (run_seed only returns scalar summaries) so mass/concentration can be measured directly."""
    h = rk.Harness(scene, poses, omega, device="cuda")
    grad, _ = _weighted_adjoint(h, scene.elevation.astype(np.float32))
    del h
    field = grad.var(axis=0) * sigma**2
    total = float(field.sum())
    flat = np.sort(field.ravel())[::-1]
    return {
        "mass": total,
        "cv": float(field.std() / max(field.mean(), 1e-12)),  # coefficient of variation: a FLAT
        # field (nothing to rank) has cv -> 0 regardless of its total mass
        "top400_frac": float(flat[:TOP400].sum() / max(total, 1e-12)),
    }


def run_seed_bundle(seed: int, elite_planner: ElitePlanner, noise: str) -> dict:
    """One seed, everything: build the belief once (family-independent), get the real MPPI
    elites, then run `ranking.run_seed` for fan/hybrid/speed/mppi_elites (paired diversity + the
    full POLICIES sensing sweep) and the disagreement-field signal stats for elites vs hybrid."""
    scene, _truth, _meas, _obs, sigma, poses, hybrid_omega, _grid = rk.build_case(
        seed, "hybrid", noise
    )
    elevation = scene.elevation.astype(np.float32)
    friction = scene.friction.astype(np.float32)
    elite_omega, elite_poses, mppi_diag = elite_planner.elites(elevation, friction, GOAL_XY)

    rows = {fam: rk.run_seed(seed, fam, noise) for fam in ("fan", "hybrid", "speed")}
    _ELITE_CACHE[seed] = elite_omega
    _CURRENT[0] = seed
    rows[FAMILY_TAG] = rk.run_seed(seed, FAMILY_TAG, noise)
    del _ELITE_CACHE[seed]
    rows[FAMILY_TAG]["mppi_diag"] = mppi_diag
    rows[FAMILY_TAG]["elite_start_pose"] = elite_poses[0].tolist()

    signal = {
        "hybrid": _disagreement_field(seed, noise, scene, sigma, poses, hybrid_omega),
        FAMILY_TAG: _disagreement_field(seed, noise, scene, sigma, poses, elite_omega),
    }
    return {"seed": seed, "rows": rows, "signal": signal}


FAMILIES = ("fan", "hybrid", "speed", FAMILY_TAG)


def report(bundles: list[dict], noise: str) -> None:
    n = len(bundles)
    print(f"\n=== C5: real MPPI elites vs synthetic plan families (n={n} seeds, noise={noise}) ===")

    jmin = np.mean([b["rows"][FAMILY_TAG]["mppi_diag"]["J_elite_min"] for b in bundles])
    jmax = np.mean([b["rows"][FAMILY_TAG]["mppi_diag"]["J_elite_max"] for b in bundles])
    jmed = np.mean([b["rows"][FAMILY_TAG]["mppi_diag"]["J_pool_median"] for b in bundles])
    jpmin = np.mean([b["rows"][FAMILY_TAG]["mppi_diag"]["J_pool_min"] for b in bundles])
    print(
        f"MPPI convergence check: elite cost range [{jmin:.3f}, {jmax:.3f}], "
        f"pool min {jpmin:.3f}, pool median {jmed:.3f}  (elites should sit near the pool min)"
    )

    print(f"\n{'family':<14}{'path_spread':>16}{'coverage_spread':>18}")
    print(f"{'':<14}{'mean (median)':>16}{'mean (median)':>18}")
    stats = {}
    for fam in FAMILIES:
        ps = np.array([b["rows"][fam]["path_spread"] for b in bundles])
        cs = np.array([b["rows"][fam]["coverage_spread"] for b in bundles])
        stats[fam] = (ps, cs)
        print(
            f"{fam:<14}{ps.mean():>9.3f} ({np.median(ps):>5.3f}){cs.mean():>11.4f} ({np.median(cs):>6.4f})"
        )

    print("\npaired vs hybrid (elites - hybrid), sign test:")
    for key, idx in (("path_spread", 0), ("coverage_spread", 1)):
        d = stats[FAMILY_TAG][idx] - stats["hybrid"][idx]
        k, w, p = rk.sign_test(d)
        print(f"  {key:<18} mean {d.mean():>+9.4f}  better on {w:>3}/{k:<3}  p={p:.2e}")

    print("\n--- disagreement FIELD (Var_k(dJ/dh) * sigma^2): elites vs hybrid ---")
    for name in ("hybrid", FAMILY_TAG):
        mass = np.array([b["signal"][name]["mass"] for b in bundles])
        cv = np.array([b["signal"][name]["cv"] for b in bundles])
        top = np.array([b["signal"][name]["top400_frac"] for b in bundles])
        print(
            f"{name:<14} total mass {mass.mean():>10.4g}   spread(cv) {cv.mean():>7.3f}   "
            f"top-{TOP400}-cell mass frac {top.mean():>6.3f}"
        )
    for key in ("mass", "cv", "top400_frac"):
        d = np.array([b["signal"][FAMILY_TAG][key] - b["signal"]["hybrid"][key] for b in bundles])
        k, w, p = rk.sign_test(d)
        print(f"  paired {key:<12} (elites - hybrid) mean {d.mean():>+10.4g}  better on {w:>3}/{k:<3}  p={p:.2e}")

    print(f"\n--- full sensing comparison, REAL MPPI elites as the candidate set (n={n}) ---")
    rk.report([b["rows"][FAMILY_TAG] for b in bundles])


def main(n_seeds: int = 60, noise: str = "all") -> None:
    wp.init()
    rk.OUT.mkdir(parents=True, exist_ok=True)
    # grid geometry is a fixed constant of ranking.build_case, independent of seed -- read it off
    # a throwaway build_case call instead of duplicating EXTENT/CELL/origin arithmetic here.
    scene0, *_ = rk.build_case(0, "hybrid", noise)
    ny, nx = scene0.shape
    elite_planner = ElitePlanner(nx, ny, scene0.cell, scene0.origin_x, scene0.origin_y)

    bundles = []
    for seed in range(n_seeds):
        bundles.append(run_seed_bundle(seed, elite_planner, noise))
        if (seed + 1) % 10 == 0:
            print(f"  {seed + 1}/{n_seeds} seeds", flush=True)

    report(bundles, noise)
    path = rk.OUT / f"elites_{noise}.json"
    path.write_text(json.dumps({"noise": noise, "n_seeds": n_seeds, "bundles": bundles}, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--noise", default="all", choices=("clean", "sensor", "localisation", "occlusion", "all"))
    a = ap.parse_args()
    main(a.seeds, a.noise)
