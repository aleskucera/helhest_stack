"""Drive to a goal on a map the robot builds as a BELIEF, in a simulator that is real physics.

This is the first loop in which everything the probabilistic stack has been given actually
changes what the robot does. Until now the belief's measurement sd, its pose drift and the
lattice's two readings of the map were computed, tested and consumed by nothing: the thing that
drives was handed a height array and a measured mask, and `k_sigma` was never set. A quantity
that no decision depends on cannot be judged, so this closes that.

  reality      ostrich `odin_sim` -- the measured 256x192 dToF ray table, contact physics, and
               the wheel dynamics, rather than a synthetic range scan against a heightmap
  perception   `elevation_belief`: Kalman fusion per cell, the visibility carve, and the pose
               drift accrued since each cell was last seen
  routing      `terrain_value_field` through `CostToGo`, given the SD and the drift, so a pose
               is vetoed on the margin it has in SIGMAS rather than in metres
  control      the MPPI rollouts on a fine window cropped from the same belief

The two windows share the belief's grid and recenter together, so the offset between them is
constant and safe to bake into the captured replan graph.

The scan is self-filtered before anything else sees it. `odin_sim` casts against the robot's own
wheels and chassis deliberately -- "exactly as on the real sensor" -- so 22% of returns land
inside 0.5 m and the belief paints the robot itself as a 0.6 m obstacle that it then carries
around. Without the filter the planner is walled in by its own body: measured here, 73% of seen
cells blocked, and the robot drives 3 m and stops. The gate runs in `ScanPreprocessor`, in the
robot's base frame where the box is exact, and the survivors are rotated to world on device --
the cloud is uploaded once and never comes back.

Unmeasured cells are filled from the measured ground, never with 0.0. A zero fill reads as flat
ground wherever the robot has not looked, and where the ground itself sits below zero that is a
phantom plateau: it walls the routing window off in a ring and the goal goes unreachable. That
failure has been diagnosed twice on real bags and it is not worth a third time.

Runs on dasenka, which is where the simulator lives -- see this directory's README.
"""

from __future__ import annotations

import argparse

import numpy as np
import warp as wp
from elevation_belief import DriftRates
from elevation_belief import ElevationBelief
from elevation_belief import NoiseModel
from examples.helhest_junior.odin_sim.sensor import OdinSensor
from examples.helhest_junior.odin_sim.sim import build_sim
from examples.helhest_junior.odin_sim.sim import ODIN_MOUNT_XYZ

from helhest import dynamics
from helhest.control.mppi import CostParams
from helhest.control.mppi import MppiGpu
from helhest.control.terminal import dock_control
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.perception import multigrid_inpaint
from helhest.perception import ScanPreprocessor
from helhest.perception import transform_points
from helhest.planning.costtogo import CostToGo

# The robot in its own base frame: origin at the front axle, rear wheel 0.75 m behind, wheels
# 0.35 m in radius and half_track 0.365 wide. Widened a little, because a self-filter that is
# slightly too generous costs some ground the robot is standing on and one that is too tight
# costs the whole map.
_MARGIN = 0.10


def self_box(r) -> tuple[float, float, float, float]:
    """(x_min, x_max, y_min, y_max) enclosing the robot, in the base frame."""
    back = r.rear_offset + r.wheel_radius + _MARGIN
    front = r.wheel_radius + _MARGIN
    side = r.half_track + 0.5 * r.wheel_width + _MARGIN
    return (-back, front, -side, side)


def quat_mat(q: np.ndarray) -> np.ndarray:
    """xyzw quaternion -> 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def pose_of(body_q: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """(x, y, yaw, R) of the robot body. Ground truth: this demo tests planning, not ICP."""
    R = quat_mat(body_q[3:7])
    return float(body_q[0]), float(body_q[1]), float(np.arctan2(R[1, 0], R[0, 0])), R


@wp.kernel
def measured_kernel(
    valid: wp.array2d(dtype=wp.int32),
    out: wp.array2d(dtype=wp.float32),
):
    """The belief's validity flag as the float mask the planner takes."""
    i, j = wp.tid()
    out[i, j] = wp.where(valid[i, j] != 0, 1.0, 0.0)


@wp.kernel
def sd_kernel(
    var: wp.array2d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
):
    """Measurement variance -> sd. Clamped at 0: a fused variance can land a hair below it."""
    i, j = wp.tid()
    out[i, j] = wp.sqrt(wp.max(var[i, j], 0.0))


@wp.kernel
def crop_kernel(
    src: wp.array2d(dtype=wp.float32),
    off: wp.int32,
    out: wp.array2d(dtype=wp.float32),
):
    """The fine window out of the routing window. Both are square and share a centre cell."""
    i, j = wp.tid()
    out[i, j] = src[i + off, j + off]


def drive(a: argparse.Namespace) -> dict:
    dt = 1.0 / a.rate
    sim = build_sim(world=a.world, dt=dt, viewer=False)
    sensor = OdinSensor(sim.model, 0, ODIN_MOUNT_XYZ, seed=0)
    for _ in range(a.settle):  # let the wheels find the terrain before anything is measured
        sim.step()

    rx, ry, yaw, _ = pose_of(sim.current_state.body_q.numpy()[0])
    goal = np.asarray(a.goal if a.goal else sim.goal, np.float64)[:2]
    span = a.window
    n = int(round(span / a.cell))
    belief = ElevationBelief(
        (rx - span / 2, rx + span / 2, ry - span / 2, ry + span / 2),
        a.cell,
        # a = 0.012 + 0.004/m: studies/calib/fit_drift.py put Odin's measured near-field
        # var_meas at 1.75 cm, and the range term is the dToF's own growth
        noise=NoiseModel("linear", a=0.012, b=0.004),
        rates=DriftRates.odin_slam(),  # on-device SLAM, 100x below the dead-reckoning default
    )

    # The MPPI window is a centred crop of the belief's, so the offset between the two is a
    # constant and the captured replan graph can hold it.
    nw = int(round(a.fine / a.cell))
    # n // 2 - nw // 2, not (n - nw) // 2: the two differ by a cell when the difference is
    # odd, and only this one puts the robot's centre CELL at the centre of both windows.
    off = n // 2 - nw // 2
    win_grid = GridParams(nw, nw, a.cell, 0.0, 0.0)
    route_grid = GridParams(n, n, a.cell, 0.0, 0.0)
    plan_sim = ForwardSimulator(
        dynamics.robot_params(a.wheel_width),
        # command_delay 0: nothing here feeds sim.command_history, and rolling out against an
        # all-zero history is worse than not modelling the dead time at all
        dynamics.planning_solver(dt=dt, command_delay=0.0),
        win_grid,
        a.batch,
        a.horizon,
        a.device,
    )
    plan_sim.set_uniform_friction(0.8)
    planner = MppiGpu(plan_sim, CostParams(), n_theta=a.n_theta)
    planner.reset_nominal(1.0)
    ctg = CostToGo(
        route_grid,
        dynamics.robot_params(a.wheel_width),
        dynamics.planning_solver(dt=dt, command_delay=0.0),
        n_theta=a.n_theta,
        k_sigma=a.k_sigma,
        device=a.device,
    )
    planner.cw.lattice_cap = ctg._vcap
    sgrid = GridParams(n, n, a.cell, -off * a.cell, -off * a.cell).build()

    # Preallocated so the per-frame path allocates nothing and touches no host memory. `scratch`
    # exists because `multigrid_inpaint` fills IN PLACE, and the array it would fill is the
    # belief's own height layer.
    zeros2d = lambda k: wp.zeros((k, k), dtype=wp.float32, device=a.device)  # noqa: E731
    scratch, measured_d, sd_d, fine_d = zeros2d(n), zeros2d(n), zeros2d(n), zeros2d(nw)
    height_d = scratch  # so a run that arrives before its first frame can still dump

    pre = ScanPreprocessor(sensor.n_rays, device=a.device)
    box = self_box(dynamics.robot_params(a.wheel_width)) if a.self_filter else None
    base_T_sensor = np.eye(4)
    base_T_sensor[:3, 3] = ODIN_MOUNT_XYZ  # the mount is a translation; sensor axes are body axes

    # `body_z` is the integrity check: the robot must stay on the terrain. The simulator's
    # contact buffers are deliberately NOT read -- touching one faults the CUDA context once
    # the step is replayed from a graph, which is why the viewer draws no contacts either.
    trail, closest, reached, body_z = [], 1.0e9, False, []
    for f in range(a.frames):
        body = sim.current_state.body_q.numpy()[0]
        rx, ry, yaw, R = pose_of(body)
        trail.append((rx, ry))
        body_z.append(float(body[2]))
        d = float(np.hypot(rx - goal[0], ry - goal[1]))
        closest = min(closest, d)
        if d < a.reach:
            reached = True
            break

        # PERCEPTION -- the sim's own ray table, self-filtered and rotated to world on device
        origin = body[0:3] + R @ ODIN_MOUNT_XYZ
        raw = sensor.scan(sim.current_state)
        base, count, _, _, _ = pre.run(
            raw,
            None,
            base_T_sensor,
            z_range=None,
            self_box=box,
            max_range=a.window / 2,  # base-frame radius: the belief window cannot hold more
        )
        world_T_base = np.eye(4)
        world_T_base[:3, :3] = R
        world_T_base[:3, 3] = body[0:3]
        pts = transform_points(base, count, world_T_base)
        belief.recenter((rx, ry))
        if f:
            belief.motion_update(dt, (rx, ry))
        if a.carve > 0.0:
            belief.carve(pts, origin, max_range=a.carve)
        belief.measure_scan(pts, origin)

        # BELIEF -> PLANNER, entirely on device. `raw_h` carries NaN where nothing was ever
        # measured, which is exactly the inpaint's unknown set, so the fill is referenced to the
        # surrounding GROUND rather than to zero -- a zero fill reads as flat terrain wherever
        # the robot has not looked, and where the ground sits below zero that phantom plateau
        # closes a ring around the routing window and the goal goes unreachable.
        lay = belief.layers()
        wp.copy(scratch, lay["raw_h"])
        height_d = multigrid_inpaint(scratch)
        wp.launch(
            measured_kernel,
            dim=(n, n),
            inputs=[lay["valid"]],
            outputs=[measured_d],
            device=a.device,
        )
        wp.launch(sd_kernel, dim=(n, n), inputs=[lay["meas_var"]], outputs=[sd_d], device=a.device)

        # ROUTING -- the belief's uncertainty reaches the planner here, and nowhere before
        V = ctg.compute(
            height_d,
            (goal[0] - belief.xmin, goal[1] - belief.ymin),
            measured=measured_d,
            sigma=sd_d,
            drift=belief.drift(),
        )
        planner.set_lattice(V, sgrid)

        # CONTROL -- fine window, cropped from the same belief
        wx0 = belief.xmin + off * a.cell
        wy0 = belief.ymin + off * a.cell
        wp.launch(
            crop_kernel, dim=(nw, nw), inputs=[height_d, off], outputs=[fine_d], device=a.device
        )
        plan_sim.set_terrain(fine_d)
        state_l = np.array([rx - wx0, ry - wy0, yaw], np.float32)
        goal_l = (goal[0] - wx0, goal[1] - wy0)
        if d < a.dock:
            cmd = dock_control(state_l, goal_l)
        else:
            planner.replan(state_l, goal_l, a.refine)
            u = planner.nominal()
            cmd = np.array([u[0, 0], u[0, 1], 0.5 * (u[0, 0] + u[0, 1])], np.float32)

        sim.set_wheel_command(cmd)
        sim.step()
        if f % a.report == 0:
            # the only host reads in the loop, and they happen on report frames alone
            blk = ctg.blocked.numpy()
            seen = measured_d.numpy() != 0.0
            print(
                f"  f{f:>4d}  at ({rx:6.2f},{ry:6.2f}) yaw {np.degrees(yaw):7.1f}d  "
                f"goal {d:5.2f} m  cmd [{cmd[0]:5.2f} {cmd[1]:5.2f}]  "
                f"pts {count:>5d}  seen {100*seen.mean():4.1f}%  "
                # two readings of the same array: what fraction of seen cells is vetoed for
                # ANY of the headings, and what fraction of (cell, heading) pairs. The first
                # is what a "blocked" map shows and is pessimistic by construction -- the
                # settle straddles a thin obstacle, so a cell beside one is blocked for a
                # minority of headings and still counts.
                f"blocked {100*blk.max(2)[seen].mean():4.1f}% of cells / "
                f"{100*blk[seen].mean():4.1f}% of poses"
            )
    if a.out:
        seen = measured_d.numpy() != 0.0
        np.savez_compressed(
            a.out,
            trail=np.array(trail, np.float64),
            body_z=np.array(body_z, np.float64),
            goal=goal,
            height=height_d.numpy(),
            seen=seen,
            blocked=ctg.blocked.numpy(),
            V=ctg.V.numpy(),
            bounds=np.array([belief.xmin, belief.ymin], np.float64),
            cell=a.cell,
            reached=reached,
        )
        print(f"wrote {a.out}")

    return dict(
        reached=reached,
        frames=f + 1,
        closest=closest,
        body_z=body_z,
        trail=trail,
        goal=goal,
        belief=belief,
        ctg=ctg,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world", default="pillars")
    p.add_argument("--goal", type=float, nargs=2, default=None, help="[m] default: the world's")
    p.add_argument("--frames", type=int, default=900)
    p.add_argument("--rate", type=float, default=14.5)  # the real sensor's rate
    p.add_argument("--settle", type=int, default=40)
    p.add_argument("--window", type=float, default=14.0, help="[m] belief + routing window")
    p.add_argument("--fine", type=float, default=9.0, help="[m] MPPI window, a centred crop")
    p.add_argument("--cell", type=float, default=0.2)
    p.add_argument("--carve", type=float, default=6.0, help="[m] 0 disables the visibility carve")
    p.add_argument("--k-sigma", type=float, default=2.0, help="veto below this many sigmas")
    p.add_argument("--n-theta", type=int, default=16)
    p.add_argument("--horizon", type=int, default=25)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--refine", type=int, default=3)
    p.add_argument(
        "--dock",
        type=float,
        default=3.0,
        # At 1.5 the forward-only MPPI orbits the goal instead of arriving: measured, it circles
        # from 2.56 m out and never closes. 3.0 reaches, and reaches with the sigma veto ON --
        # k_sigma = 0 does NOT rescue 1.5, so the veto was never what was holding it off.
        help="[m] hand over to the dock controller",
    )
    p.add_argument("--reach", type=float, default=0.4, help="[m] counts as arrived")
    p.add_argument("--wheel-width", type=float, default=0.10)
    p.add_argument(
        "--no-self-filter",
        dest="self_filter",
        action="store_false",
        help="let the robot's own wheels into the map -- it walls itself in; see the docstring",
    )
    p.add_argument("--report", type=int, default=25)
    p.add_argument("--out", default=None, help="npz of the final map + trail, for the figure")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    r = drive(a)
    z = np.array(r["body_z"])
    t = np.array(r["trail"])
    path = float(np.hypot(*np.diff(t, axis=0).T).sum()) if len(t) > 1 else 0.0
    print(
        f"\n{'REACHED' if r['reached'] else 'did not reach'} in {r['frames']} frames "
        f"({r['frames']/a.rate:.1f} s sim)   closest {r['closest']:.2f} m   goal {r['goal']}\n"
        f"  drove {path:.1f} m of path   body z {z.min():.2f} to {z.max():.2f} m "
        f"(it should stay on the terrain)"
    )


if __name__ == "__main__":
    main()
