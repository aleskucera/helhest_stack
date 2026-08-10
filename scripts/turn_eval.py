"""Closed-loop turning behaviour: how willing is the planner to turn, and how fast?

    python scripts/turn_eval.py --sweep
    python scripts/turn_eval.py --bearing 180        # a goal directly behind

Same loop as demos/eval.py (WarpDriver + MPPI + cost-to-go) but on FLAT OPEN ground, so nothing
about the result is about obstacle avoidance -- only about whether the planner chooses to turn and
how quickly it builds the turn. The goal is placed at a bearing from the robot's initial heading,
so a bearing of 180 deg is the "goal behind me" case.

Unlike demos/eval.py this runs the command through `condition_command`, because plan_max_slew lives
there rather than in the planner, and it is one of the knobs under test.

Reported per run: whether the goal was reached, how long it took, the DETOUR ratio (path length
over straight-line distance -- a robot that refuses to turn drives a long arc) and the peak yaw
rate it actually achieved.
"""

from __future__ import annotations

import argparse
import dataclasses
import math

import numpy as np
import warp as wp

from helhest import dynamics
from helhest.control.command import condition_command
from helhest.control.mppi import CostParams
from helhest.control.mppi import MppiGpu
from helhest.control.terminal import dock_control
from helhest.driver import WarpDriver
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.heightmap import _grid
from helhest.heightmap import Heightmap
from helhest.planning.costtogo import CostToGo

EXTENT, CELL, MU = 14.0, 0.1, 0.8


def flat() -> Heightmap:
    XX, _ = _grid((-EXTENT, EXTENT), (-EXTENT, EXTENT), CELL)
    return Heightmap(np.zeros_like(XX).astype(np.float32), (-EXTENT, -EXTENT), CELL)


def run(bearing_deg: float, turn_w: float, max_slew: float, device: str = "cuda",
        dist: float = 5.0, max_frames: int = 700, n_theta: int = 24,
        spin_frac: float = 0.0, in_place_cost: float = 0.0) -> dict:
    scene = flat()
    mu_hm = Heightmap(np.full_like(scene.H, MU), (scene.x0, scene.y0), scene.cell)
    th = math.radians(bearing_deg)
    goal = np.array([dist * math.cos(th), dist * math.sin(th)], np.float64)
    grid = GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0)

    sim = ForwardSimulator(dynamics.robot_params(), dynamics.planning_solver(), grid,
                           4096, 25, device)
    sim.set_terrain(
        wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device)
    )
    sim.set_friction(mu_hm)
    cost = dataclasses.replace(CostParams(), turn=turn_w)  # CostParams is frozen
    kw = {}
    if spin_frac > 0.0:  # only pass it once the sampler supports it
        from helhest.control.mppi import SamplingConfig

        kw["sampling"] = SamplingConfig(spin_frac=spin_frac)
    planner = MppiGpu(sim, cost, n_theta=n_theta, **kw)
    planner.reset_nominal(1.5)

    ctg_kw = {"in_place_cost": in_place_cost} if in_place_cost > 0.0 else {}
    ctg = CostToGo(grid, dynamics.robot_params(), dynamics.planning_solver(),
                   n_theta=n_theta, device=device, **ctg_kw)
    V = ctg.compute(
        wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device),
        (float(goal[0]), float(goal[1])),
    )
    planner.set_lattice(V, grid.build())

    drv = WarpDriver(scene, mu_hm, init_pose=(0.0, 0.0, 0.0), device=device)
    prev = np.zeros(3, np.float32)
    path, yaws, reached, f = 0.0, [], False, 0
    px, py, pyaw = 0.0, 0.0, 0.0
    for f in range(max_frames):
        st = drv.render_state()
        state = np.array([st.x, st.y, st.yaw], np.float32)
        path += math.hypot(st.x - px, st.y - py)
        yaws.append(abs(math.atan2(math.sin(st.yaw - pyaw), math.cos(st.yaw - pyaw))) / dynamics.DT)
        px, py, pyaw = st.x, st.y, st.yaw
        d = float(np.hypot(st.x - goal[0], st.y - goal[1]))
        if d < 0.3:
            reached = True
            break
        if d < 1.5:
            cmd = dock_control(state, goal)
        else:
            planner.replan(state, goal, 3)
            u = planner.nominal()
            cmd = condition_command(float(u[0, 0]), float(u[0, 1]), prev,
                                    max_omega=7.5, max_slew=max_slew, dt=dynamics.DT)
        prev = np.asarray(cmd, np.float32)
        drv.step(np.asarray(cmd, np.float32))
    del planner, sim, ctg, drv
    return {
        "reached": reached,
        "s": (f + 1) * dynamics.DT,
        "detour": path / max(dist, 1e-6),
        "peak_yaw": float(np.percentile(yaws[1:], 98)) if len(yaws) > 2 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--bearing", type=float, default=90.0)
    ap.add_argument("--turn", type=float, default=0.8)
    ap.add_argument("--slew", type=float, default=2.0)
    ap.add_argument("--spin-frac", type=float, default=0.0)
    ap.add_argument("--in-place-cost", type=float, default=0.0)
    args = ap.parse_args()
    wp.init()

    if not args.sweep:
        r = run(args.bearing, args.turn, args.slew, spin_frac=args.spin_frac,
                in_place_cost=args.in_place_cost)
        print(f"  bearing {args.bearing:.0f} turn {args.turn} slew {args.slew}: {r}")
        return

    print(f"{'bearing':>8}{'plan_turn':>11}{'max_slew':>10}{'reached':>9}{'time s':>8}"
          f"{'detour':>8}{'peak yaw':>10}")
    for bearing in (90.0, 180.0):
        for turn_w in (0.8, 0.4, 0.2, 0.1):
            for slew in (2.0, 6.0):
                r = run(bearing, turn_w, slew)
                print(f"{bearing:>8.0f}{turn_w:>11.2f}{slew:>10.1f}"
                      f"{('yes' if r['reached'] else 'NO'):>9}{r['s']:>8.1f}"
                      f"{r['detour']:>8.2f}{r['peak_yaw']:>9.2f}")


if __name__ == "__main__":
    main()
