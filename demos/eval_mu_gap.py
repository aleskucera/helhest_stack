"""Closed-loop friction sim-to-real gap study: the planner BELIEVES mu = 0.8 while the driven
robot lives on different ground. Compares three planner configs per (world, real-mu):

  naive   -- pre-certificate behavior (n_mu = 1, saturation/tip off)
  cert    -- friction-saturation certificate + tip margin on (n_mu = 1)
  robust  -- cert + 4 mu replicas ranked by worst case + the ONLINE TurnGainEstimator closing
             the loop on realized yaw (the shipped odin config)

All configs run the production-tuned base (elevation_node defaults): turn penalty, straight
prior, peaky elite, plan-consistency EMA, routing tube 0.15 m, tall-step gate 0.2 m. Without
the gate the settle STRADDLES tall thin obstacles (1 m pillars read as drivable terrain) and
every config drives onto them -- margins can't fix a contact that happens on TOP.

  python demos/eval_mu_gap.py                # full sweep
  python demos/eval_mu_gap.py --world gap --mu-real 0.3 --config robust
"""

from __future__ import annotations

import argparse

import numpy as np
import warp as wp

from helhest import dynamics
from helhest import worlds as W
from helhest.control.mppi import CostParams
from helhest.control.mppi import MppiGpu
from helhest.control.mppi import SamplingConfig
from helhest.control.terminal import dock_control
from helhest.control.turn_adapt import TurnGainEstimator
from helhest.driver import WarpDriver
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.planning.costtogo import CostToGo

MU_PLAN = 0.8  # what the planner believes
CONSISTENCY = 0.3  # node plan_consistency
_TUNED = dict(goal_running=0.3, effort=1e-3, turn=0.03)
CONFIGS = {
    "naive": dict(
        cost=CostParams(saturation=0.0, tip=0.0, **_TUNED), n_mu=1, span=0.0, adapt=False
    ),
    "cert": dict(cost=CostParams(**_TUNED), n_mu=1, span=0.0, adapt=False),
    "robust": dict(cost=CostParams(**_TUNED), n_mu=4, span=0.4, adapt=True),
}


def run(
    world: str,
    mu_real: float,
    config: str,
    device: str = "cuda",
    max_frames: int = 800,
    B: int = 4096,
    T: int = 25,
    n_theta: int = 24,
    dock_radius: float = 1.5,
    seed: int = 0,
) -> dict:
    builder, start, goal = W.WORLDS[world]
    scene = builder()
    goal = np.asarray(goal, np.float64)
    mu_field = W.matching_friction(scene)
    mu_field.H[:] = mu_real  # REALITY
    cfg = CONFIGS[config]
    grid = GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0)
    rp = dynamics.robot_params()
    solver = dynamics.planning_solver()
    plan_sim = ForwardSimulator(rp, solver, grid, B, T, device)
    plan_sim.set_terrain(
        wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device)
    )
    plan_sim.set_uniform_friction(MU_PLAN)  # BELIEF
    planner = MppiGpu(
        plan_sim,
        cfg["cost"],
        sampling=SamplingConfig(n_mu=cfg["n_mu"], straight_frac=0.2, elite_frac=0.01),
        n_theta=n_theta,
        seed=seed,
    )
    planner.set_mu_band(1.0, cfg["span"])
    planner.reset_nominal(1.5)
    ctg = CostToGo(
        grid, rp, solver, n_theta=n_theta, robust_margin_m=0.15, obstacle_step_m=0.2, device=device
    )
    V = ctg.compute(
        wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device), goal
    )
    planner.set_lattice(V, grid.build())
    planner.cw.lattice_cap = ctg._vcap
    est = None
    if cfg["adapt"]:
        est = TurnGainEstimator(
            k_turn=solver.k_turn,
            mu_nominal=MU_PLAN,
            wheel_radius=rp.wheel_radius,
            half_track=rp.half_track,
            dt=dynamics.DT,
            tau_s=3.0,
        )
    drv = WarpDriver(scene, mu_field, init_pose=tuple(start), device=device)

    contacts, closest, reached, f = 0, 99.0, False, 0
    prev_U = None
    prev_yaw, prev_diff = float(start[2]), None
    for f in range(max_frames):
        st = drv.render_state()
        state = np.array([st.x, st.y, st.yaw], np.float32)
        if est is not None and prev_diff is not None:
            dyaw = (st.yaw - prev_yaw + np.pi) % (2 * np.pi) - np.pi
            center, span = est.update(prev_diff, dyaw / dynamics.DT)
            planner.set_mu_band(center, span)
        prev_yaw = st.yaw
        d = float(np.hypot(st.x - goal[0], st.y - goal[1]))
        closest = min(closest, d)
        if d < 0.3:
            reached = True
            break
        if dock_radius > 0.0 and d < dock_radius:
            cmd = dock_control(state, goal)
        else:
            planner.replan(state, goal, 3)
            U = planner.nominal()
            if prev_U is not None:  # node plan_consistency EMA (receding-horizon shift)
                shifted = np.roll(prev_U, -1, axis=0)
                shifted[-1] = prev_U[-1]
                U = (1.0 - CONSISTENCY) * U + CONSISTENCY * shifted
                planner.set_nominal(U)
            prev_U = U.copy()
            cmd = np.array([U[0, 0], U[0, 1], 0.5 * (U[0, 0] + U[0, 1])], np.float32)
        prev_diff = float(cmd[1] - cmd[0])
        drv.step(cmd)
        if drv.clear < 0.05:
            contacts += 1
    return dict(reached=reached, frames=f + 1, closest=closest, contacts=contacts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", default=None, choices=list(W.WORLDS))
    ap.add_argument("--mu-real", type=float, default=None)
    ap.add_argument("--config", default=None, choices=list(CONFIGS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    wp.init()
    worlds = [args.world] if args.world else ["gap", "pillars", "bumpy"]
    mus = [args.mu_real] if args.mu_real is not None else [0.3, 0.8, 1.5]
    configs = [args.config] if args.config else list(CONFIGS)
    print(f"{'world':9}{'mu_real':9}{'config':9}{'reach':7}{'frames':8}{'closest':9}{'contacts':9}")
    for world in worlds:
        for mu in mus:
            for config in configs:
                r = run(world, mu, config, device=args.device, seed=args.seed)
                print(
                    f"{world:9}{mu:<9.2f}{config:9}{str(r['reached']):7}{r['frames']:<8}"
                    f"{r['closest']:<9.2f}{r['contacts']:<9}"
                )


if __name__ == "__main__":
    main()
