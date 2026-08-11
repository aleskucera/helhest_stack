"""Closed-loop friction sim-to-real gap study: the planner BELIEVES mu = 0.8 while the driven
robot lives on different ground. Compares three planner configs per (world, real-mu):

  naive   -- pre-certificate behavior (n_mu = 1, saturation/tip off)
  cert    -- friction-saturation certificate + tip margin on (n_mu = 1)
  robust  -- cert + 3 mu replicas ranked by worst case (the shipped odin config, minus the
             online estimator, which needs a real gyro)

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
from helhest.driver import WarpDriver
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.planning.costtogo import CostToGo

MU_PLAN = 0.8  # what the planner believes
CONFIGS = {
    "naive": dict(cost=CostParams(saturation=0.0, tip=0.0), n_mu=1, span=0.0),
    "cert": dict(cost=CostParams(), n_mu=1, span=0.0),
    "robust": dict(cost=CostParams(), n_mu=4, span=0.4),
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
) -> dict:
    builder, start, goal = W.WORLDS[world]
    scene = builder()
    goal = np.asarray(goal, np.float64)
    mu_field = W.matching_friction(scene)
    mu_field.H[:] = mu_real  # REALITY
    cfg = CONFIGS[config]
    grid = GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0)
    plan_sim = ForwardSimulator(
        dynamics.robot_params(), dynamics.planning_solver(), grid, B, T, device
    )
    plan_sim.set_terrain(
        wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device)
    )
    plan_sim.set_uniform_friction(MU_PLAN)  # BELIEF
    planner = MppiGpu(
        plan_sim, cfg["cost"], sampling=SamplingConfig(n_mu=cfg["n_mu"]), n_theta=n_theta
    )
    planner.set_mu_band(1.0, cfg["span"])
    planner.reset_nominal(1.5)
    ctg = CostToGo(
        grid, dynamics.robot_params(), dynamics.planning_solver(), n_theta=n_theta, device=device
    )
    V = ctg.compute(
        wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device), goal
    )
    planner.set_lattice(V, grid.build())
    planner.cw.lattice_cap = ctg._vcap
    drv = WarpDriver(scene, mu_field, init_pose=tuple(start), device=device)

    contacts, closest, reached, f = 0, 99.0, False, 0
    for f in range(max_frames):
        st = drv.render_state()
        state = np.array([st.x, st.y, st.yaw], np.float32)
        d = float(np.hypot(st.x - goal[0], st.y - goal[1]))
        closest = min(closest, d)
        if d < 0.3:
            reached = True
            break
        if dock_radius > 0.0 and d < dock_radius:
            cmd = dock_control(state, goal)
        else:
            planner.replan(state, goal, 3)
            u = planner.nominal()
            cmd = np.array([u[0, 0], u[0, 1], 0.5 * (u[0, 0] + u[0, 1])], np.float32)
        drv.step(cmd)
        if drv.clear < 0.05:
            contacts += 1
    return dict(reached=reached, frames=f + 1, closest=closest, contacts=contacts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", default=None, choices=list(W.WORLDS))
    ap.add_argument("--mu-real", type=float, default=None)
    ap.add_argument("--config", default=None, choices=list(CONFIGS))
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
                r = run(world, mu, config, device=args.device)
                print(
                    f"{world:9}{mu:<9.2f}{config:9}{str(r['reached']):7}{r['frames']:<8}"
                    f"{r['closest']:<9.2f}{r['contacts']:<9}"
                )


if __name__ == "__main__":
    main()
