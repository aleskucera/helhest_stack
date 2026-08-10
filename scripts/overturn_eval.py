"""Reproduce the late, violent evasive turn near an obstacle, and test what damps it.

    python scripts/overturn_eval.py --world pocket          # the repro: sweep sensor range
    python scripts/overturn_eval.py --world pocket --fix    # sweep the candidate damping knobs
    python scripts/overturn_eval.py --friction              # the friction-mismatch hypothesis
    python scripts/overturn_eval.py --world pocket --plot /tmp/overturn.png

THE SYMPTOM, from the field: the robot ends up closer to an obstacle than it meant to, then snaps
into a very fast turn to get away.

TWO CANDIDATE CAUSES, and this harness separates them.

  (1) FRICTION MISMATCH. The planner rolls out at a constant `plan_friction` (0.8 as deployed) no
      matter what it is driving on. alpha = 1 + k_turn * mu DIVIDES the yaw rate, so ground softer
      than 0.8 yields more yaw per unit of commanded differential than planned -- the planner is
      systematically surprised in the oversteer direction. Run with --friction.

  (2) LATE DISCOVERY. On the robot the map is BUILT from lidar and unknown cells are inpainted
      optimistically as flat, so an obstacle does not exist for the planner until it comes into
      sensor range. The robot commits to a straight line across ground it cannot see, the wall
      materialises a few metres ahead, and the evasion is late by construction. This is the default
      mode; --sensor-range 0 turns it off and gives the planner the whole world (the control).

Reported per run:
  clear      the closest the robot's CENTRE came to an obstacle cell [m]
  peak yaw   98th-percentile |yaw rate| [rad/s] -- how violent the run was overall
  yaw@clear  peak |yaw rate| within +-0.5 s of closest approach -- the evasion itself
  slew       peak commanded |d(differential)/dt| [rad/s^2] -- the jerk the drivetrain sees
  hits       steps with the settle clearance under 5 cm (actually touching)
"""

from __future__ import annotations

import argparse
import dataclasses
import math

import numpy as np
import warp as wp

from helhest import dynamics
from helhest import worlds
from helhest.control.command import condition_command
from helhest.control.mppi import CostParams
from helhest.control.mppi import MppiGpu
from helhest.control.mppi import SamplingConfig
from helhest.control.terminal import dock_control
from helhest.driver import WarpDriver
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.heightmap import Heightmap
from helhest.perception.lidar import lidar_scan
from helhest.perception.lidar import MultiScanMap
from helhest.planning.costtogo import CostToGo

# The deployed odin configuration, so a number measured here means something on the robot.
# Mirrors ros/odin/odin_elevation.params.yaml; keep the two in step. `lat_coarsen` is chosen to
# reproduce the deployed routing CELL (0.24 m), not the deployed integer -- the stress worlds are
# built at 0.06 m where the node's map is 0.08 m.
DEPLOYED = dict(
    horizon=25,
    n_theta=24,
    friction=0.8,
    turn=0.2,
    smooth=0.04,
    effort=0.001,
    goal_running=0.3,
    wmax=6.0,
    max_omega=7.5,
    max_slew=6.0,
    max_decel=8.0,
    goal_brake_dist=2.0,
    turn_brake_a_max=1.1,
    robust_margin_m=0.20,
    robust_margin_deg=15.0,
    route_cell=0.24,
    wheel_width=0.10,
    k_turn=0.6,
)
SENSOR = dict(fov_deg=180.0, max_range=7.0, mount_height=0.4)


def obstacle_points(scene: Heightmap, thresh: float = 0.3) -> np.ndarray:
    """Centres of the cells tall enough to be an obstacle, as [M, 2] world xy."""
    yi, xi = np.nonzero(np.asarray(scene.H) > thresh)
    return np.stack(
        [scene.x0 + (xi + 0.5) * scene.cell, scene.y0 + (yi + 0.5) * scene.cell], axis=1
    )


def run(
    world: str = "pocket",
    sensor_range: float = SENSOR["max_range"],
    mu_real: float = DEPLOYED["friction"],
    mu_plan: float = DEPLOYED["friction"],
    turn_w: float = DEPLOYED["turn"],
    smooth_w: float = DEPLOYED["smooth"],
    max_slew: float = DEPLOYED["max_slew"],
    robust_m: float = DEPLOYED["robust_margin_m"],
    wmax: float = DEPLOYED["wmax"],
    a_max: float = DEPLOYED["turn_brake_a_max"],
    device: str = "cuda",
    max_frames: int = 1200,
) -> dict:
    builder, start, goal_xy = worlds.WORLDS[world]
    scene = builder()
    goal = np.asarray(goal_xy, np.float64)
    obs_xy = obstacle_points(scene)
    grid = GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0)
    robot = dynamics.robot_params(DEPLOYED["wheel_width"])
    solver = dynamics.planning_solver()
    blind = sensor_range > 0.0

    # PLANNER: rolls out at mu_plan, the constant the node ships.
    sim = ForwardSimulator(robot, solver, grid, 4096, DEPLOYED["horizon"], device)
    sim.set_uniform_friction(mu_plan)
    cost = dataclasses.replace(
        CostParams(),
        turn=turn_w,
        smoothness=smooth_w,
        effort=DEPLOYED["effort"],
        goal_running=DEPLOYED["goal_running"],
    )
    planner = MppiGpu(sim, cost, n_theta=DEPLOYED["n_theta"],
                      sampling=SamplingConfig(wmax=wmax))
    planner.reset_nominal(1.5)

    kr = max(1, int(round(DEPLOYED["route_cell"] / scene.cell)))
    rny, rnx = scene.ny // kr, scene.nx // kr
    route_grid = GridParams(rnx, rny, scene.cell * kr, scene.x0, scene.y0)
    ctg = CostToGo(
        route_grid,
        robot,
        solver,
        n_theta=DEPLOYED["n_theta"],
        device=device,
        robust_margin_m=robust_m,
        robust_margin_deg=DEPLOYED["robust_margin_deg"],
    )

    # WORLD: the same terrain, but the friction the robot actually has.
    mu_hm = Heightmap(
        np.full((scene.ny, scene.nx), mu_real, np.float32), (scene.x0, scene.y0), scene.cell
    )
    drv = WarpDriver(scene, mu_hm, init_pose=tuple(start), device=device)
    mm = MultiScanMap(scene.ny, scene.nx) if blind else None

    lat_gain = robot.wheel_radius**2 / (
        2.0 * robot.half_track * (1.0 + DEPLOYED["k_turn"] * mu_plan)
    )
    prev = np.zeros(3, np.float32)
    pyaw = start[2]
    clears, yaws, diffs, xs, ys = [], [], [], [], []
    hits, reached, f = 0, False, 0
    for f in range(max_frames):
        st = drv.render_state()
        state = np.array([st.x, st.y, st.yaw], np.float32)
        yaws.append(
            abs(math.atan2(math.sin(st.yaw - pyaw), math.cos(st.yaw - pyaw))) / dynamics.DT
        )
        clears.append(float(np.min(np.hypot(obs_xy[:, 0] - st.x, obs_xy[:, 1] - st.y))))
        xs.append(st.x)
        ys.append(st.y)
        pyaw = st.yaw
        d = float(np.hypot(st.x - goal[0], st.y - goal[1]))
        if d < 0.3:
            reached = True
            break

        # PERCEPTION: what the planner is allowed to know. Unknown -> flat, which is the optimism
        # that lets the robot commit to a line across ground it has never seen.
        if blind:
            scan, known = lidar_scan(
                scene.H, scene.x0, scene.y0, scene.cell, (st.x, st.y, st.yaw),
                fov_deg=SENSOR["fov_deg"], max_range=sensor_range,
                mount_height=SENSOR["mount_height"],
            )
            mm.integrate(scan, known)
            H = np.where(mm.known, mm.elev, 0.0).astype(np.float32)
        else:
            H = np.ascontiguousarray(scene.H, np.float32)
        sim.set_terrain(wp.array(H, dtype=wp.float32, device=device))
        Hc = H[: rny * kr, : rnx * kr].reshape(rny, kr, rnx, kr).max(axis=(1, 3))
        V = ctg.compute(
            wp.array(np.ascontiguousarray(Hc), dtype=wp.float32, device=device),
            (float(goal[0]), float(goal[1])),
        )
        planner.set_lattice(V, route_grid.build())

        if d < 1.5:
            cmd = dock_control(state, goal)
        else:
            planner.replan(state, goal, 3)
            u = planner.nominal()
            cmd = condition_command(
                float(u[0, 0]), float(u[0, 1]), prev,
                max_omega=DEPLOYED["max_omega"], max_slew=max_slew, dt=dynamics.DT,
                max_decel=DEPLOYED["max_decel"], goal_dist=d,
                brake_dist=DEPLOYED["goal_brake_dist"],
                turn_brake_a_max=a_max, lat_gain=lat_gain,
            )
        c = np.asarray(cmd, np.float32)
        diffs.append(float(c[2] - c[0]))
        prev = c
        drv.step(c)
        if drv.clear < 0.05:
            hits += 1
    del planner, sim, ctg, drv

    yaw_arr = np.asarray(yaws[1:]) if len(yaws) > 2 else np.zeros(1)
    clear_arr = np.asarray(clears) if clears else np.zeros(1)
    d_arr = np.asarray(diffs) if diffs else np.zeros(1)
    slew = np.abs(np.diff(d_arr)) / dynamics.DT if len(d_arr) > 1 else np.zeros(1)
    # The evasion itself: the yaw rate within +-0.5 s of closest approach, which is what the
    # operator sees as "he snapped". A whole-run peak also catches ordinary cornering.
    k = int(np.argmin(clear_arr))
    half = max(1, int(0.5 / dynamics.DT))
    win = yaw_arr[max(0, k - half) : k + half]
    return {
        "reached": reached,
        "s": (f + 1) * dynamics.DT,
        "clear": float(clear_arr.min()),
        "peak_yaw": float(np.percentile(yaw_arr, 98)),
        "yaw_at_clear": float(np.max(win)) if len(win) else 0.0,
        "slew": float(np.max(slew)),
        "hits": hits,
        "path": (np.asarray(xs), np.asarray(ys)),
    }


HEAD = (
    f"{'':>22}{'reach':>7}{'time s':>8}{'clear m':>9}{'peak yaw':>10}"
    f"{'yaw@clear':>11}{'slew':>8}{'hits':>6}"
)


def show(tag: str, r: dict) -> None:
    print(
        f"{tag:>22}{('yes' if r['reached'] else 'NO'):>7}{r['s']:>8.1f}{r['clear']:>9.2f}"
        f"{r['peak_yaw']:>10.2f}{r['yaw_at_clear']:>11.2f}{r['slew']:>8.1f}{r['hits']:>6d}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", default="pocket", choices=sorted(worlds.WORLDS))
    ap.add_argument("--friction", action="store_true", help="sweep the friction-mismatch cause")
    ap.add_argument("--fix", action="store_true", help="sweep the candidate damping knobs")
    ap.add_argument("--sensor-range", type=float, default=SENSOR["max_range"])
    ap.add_argument("--plot", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    wp.init()
    paths = {}

    if args.friction:
        print(f"world '{args.world}', sensor {args.sensor_range} m, "
              f"planner always rolls out at mu = {DEPLOYED['friction']}\n{HEAD}")
        for mu_real in (0.8, 0.6, 0.4, 0.3):
            r = run(args.world, args.sensor_range, mu_real, device=args.device)
            show(f"real mu {mu_real:.2f}", r)
            paths[f"mu {mu_real:.2f}"] = r
    elif args.fix:
        print(f"world '{args.world}', sensor {args.sensor_range} m\n{HEAD}")
        base = run(args.world, args.sensor_range, device=args.device)
        show("deployed", base)
        paths["deployed"] = base
        for tag, kw in (
            ("slew 3.0", dict(max_slew=3.0)),
            ("slew 1.5", dict(max_slew=1.5)),
            ("smooth 0.15", dict(smooth_w=0.15)),
            ("smooth 0.40", dict(smooth_w=0.40)),
            ("turn 0.5", dict(turn_w=0.5)),
            ("tube 0.48", dict(robust_m=0.48)),
            ("wmax 4.0", dict(wmax=4.0)),
            ("wmax 3.0", dict(wmax=3.0)),
            ("a_max off", dict(a_max=0.0)),
            ("a_max 0.6", dict(a_max=0.6)),
            ("a_max 0.4", dict(a_max=0.4)),
            ("a_max 0.4 +wmax 4", dict(a_max=0.4, wmax=4.0)),
        ):
            r = run(args.world, args.sensor_range, device=args.device, **kw)
            show(tag, r)
            paths[tag] = r
    else:
        print(f"world '{args.world}', deployed config, sensor range swept\n{HEAD}")
        for rng in (0.0, 10.0, 7.0, 5.0, 4.0):
            r = run(args.world, rng, device=args.device)
            tag = "omniscient" if rng == 0.0 else f"sensor {rng:.0f} m"
            show(tag, r)
            paths[tag] = r

    if args.plot:
        _plot(args.world, paths, args.plot)


def _plot(world: str, runs: dict, out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scene = worlds.WORLDS[world][0]()
    ext = [scene.x0, scene.x0 + scene.nx * scene.cell,
           scene.y0, scene.y0 + scene.ny * scene.cell]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.imshow(scene.H, origin="lower", extent=ext, cmap="Greys", vmin=0.0, vmax=1.5)
    for tag, r in runs.items():
        x, y = r["path"]
        ax.plot(x, y, lw=1.8,
                label=f"{tag}  clear {r['clear']:.2f} m, yaw@clear {r['yaw_at_clear']:.2f}")
    ax.set_aspect("equal")
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"{world}: the planner only knows what it has seen")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
