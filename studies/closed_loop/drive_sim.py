"""Drive to a goal on a map the robot builds as a BELIEF, in a simulator that is real physics.

This is the first loop in which everything the probabilistic stack has been given actually
changes what the robot does. Until now the belief's measurement sd, its pose drift and the
lattice's two readings of the map were computed, tested and consumed by nothing: the thing that
drives was handed a height array and a measured mask, and `z_veto` was never set. A quantity
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
import json
import pathlib

import numpy as np
import warp as wp
from elevation_belief import DriftRates
from elevation_belief import ElevationBelief
from elevation_belief import NoiseModel
from examples.helhest_junior.odin_sim.sensor import OdinSensor
from examples.helhest_junior.odin_sim.sim import build_sim
from examples.helhest_junior.odin_sim.sim import ODIN_MOUNT_XYZ

from helhest import dynamics
from helhest.control.mppi import MppiGpu
from helhest.control.terminal import dock_control
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.perception import multigrid_inpaint
from helhest.perception import ScanPreprocessor
from helhest.perception import transform_points
from helhest.planner_config import planner_config
from helhest.planner_config import resolve
from helhest.control.command import condition_command
from helhest.control.command import to_engine_order
from helhest.control.command import turn_first
from helhest.control.governor import ClearanceGovernor
from helhest.planning.coarse import CoarseRouter
from helhest.planning.costtogo import CostToGo
from helhest.planning.lattice_solver import trace_optimal
from helhest.worlds import footprint
from helhest.worlds import obstacle_clearance

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


_ROBOT_PARAMS = pathlib.Path(__file__).resolve().parents[2] / "ros/odin/odin_elevation.params.yaml"


def _plan_params(a: argparse.Namespace) -> dict:
    """The robot's plan_* values from its params file, then any explicit flag on top."""
    params: dict = {}
    if a.params != "none":
        import yaml  # present in the container via ROS; deliberately not a helhest dependency

        doc = yaml.safe_load(pathlib.Path(a.params).read_text())
        params = dict(next(v["ros__parameters"] for v in doc.values() if "ros__parameters" in v))
    for flag, key in (
        ("n_theta", "plan_n_theta"),
        ("horizon", "plan_horizon"),
        ("batch", "plan_batch"),
        ("wmax", "plan_wmax"),
        ("wheel_width", "plan_wheel_width"),
        ("veto", "plan_wall_veto"),
    ):
        if getattr(a, flag) is not None:
            params[key] = getattr(a, flag)
    return params


def roll_pitch(R: np.ndarray) -> tuple[float, float]:
    """ZYX roll and pitch in the settle's own convention: climb = nose-up = NEGATIVE pitch, which
    is what `_feasibility_kernel` and the MPPI's envelope terms both compare against."""
    return float(np.arctan2(R[2, 1], R[2, 2])), float(np.arcsin(-np.clip(R[2, 0], -1.0, 1.0)))


def _v_here(v: np.ndarray, x: float, y: float, yaw: float, cell: float, n_theta: int) -> float:
    """V at one pose's own cell and own heading bin, or nan outside the routing window.

    The heading bin is `round(yaw / dth)`, NEAREST -- the same rule `lattice_solver.trace_optimal`
    uses to enter the table. Anything else reads a neighbouring bin and the number stops meaning
    what the policy saw.
    """
    nr = v.shape[0]
    c, r = int(x / cell), int(y / cell)
    if not (0 <= r < nr and 0 <= c < v.shape[1]):
        return float("nan")
    t = int(round((yaw % (2.0 * np.pi)) / (2.0 * np.pi / n_theta))) % n_theta
    return float(v[r, c, t])


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
    """A centred square crop. Every window here shares the belief's centre cell, so one offset
    describes the crop completely."""
    i, j = wp.tid()
    out[i, j] = src[i + off, j + off]


def carrot_command(
    path: np.ndarray,
    state: np.ndarray,
    robot,
    look: float,
    speed: float,
    spin_deg: float,
    spin_rate: float,
    recover: float,
) -> np.ndarray:
    """Pure pursuit along the lattice's OWN policy path -> (wL, wR, w_rear) rad/s.

    The point of this controller is what it CANNOT do. `trace_optimal` walks the policy, so the
    path only ever crosses poses the lattice found feasible -- a carrot follower therefore cannot
    drive through a vetoed pose, which the MPPI demonstrably can (measured on `bumpy`: 20% of
    frames on a vetoed pose, and one 1.7 s stretch at up to 9.5 degrees past the nose-down limit
    with the wheels at full commanded speed). That makes it the arm in which the router's
    feasibility actually binds, and so the arm that can tell us whether the veto set is
    survivable or merely strict.

    Two things the first version got wrong, each of which froze a world:

    Turning is DECOUPLED from driving. Scaling the pure-pursuit curvature by a forward speed that
    itself falls with misalignment is a deadlock -- the robot slows down and loses the turn rate
    that would fix its heading, so it stays misaligned. On `ridge` the commands decayed
    1.88/2.18 -> 0.12/0.43 and it crept to a halt. A skid-steer turns on the spot for nothing, so
    past `spin_deg` it does exactly that and drives only once it is pointing roughly the right
    way.

    And a plan that fails is not a reason to stop. When the policy walk returns nothing the first
    version commanded zero, and since a stopped robot measures nothing new, the plan stayed
    failed: `pillars` sat at one pose from frame 200 to 700. Now it turns on the spot, which
    changes what it can see and costs nothing that the veto forbids.

    A diagnostic, not a replacement. The lattice is kinematic -- no momentum, no motor lag, no
    friction -- so this follows a plan at low speed and knows nothing about braking distance at
    the 1.5-2.5 m/s the stack is meant for.
    """
    x, y, yaw = float(state[0]), float(state[1]), float(state[2])
    if len(path) >= 2:
        d = np.hypot(path[:, 0] - x, path[:, 1] - y)
        far = np.flatnonzero(d >= look)
        tx, ty = path[far[0]] if len(far) else path[-1]
        reach = max(float(np.hypot(tx - x, ty - y)), 1e-3)
        ang = np.arctan2(ty - y, tx - x) - yaw
        ang = float((ang + np.pi) % (2.0 * np.pi) - np.pi)
    else:
        ang, reach = recover, look  # no plan from here: turn and look

    if abs(ang) > np.radians(spin_deg):
        v, omega = 0.0, spin_rate * np.sign(ang)
    else:
        v = speed
        omega = 2.0 * np.sin(ang) / reach * v  # pure-pursuit curvature, times speed
    wl = (v - omega * robot.half_track) / robot.wheel_radius
    wr = (v + omega * robot.half_track) / robot.wheel_radius
    return np.array([wl, wr, 0.5 * (wl + wr)], np.float32)


def drive(a: argparse.Namespace) -> dict:
    dt = 1.0 / a.rate
    sim = build_sim(world=a.world, dt=dt, viewer=False)
    sensor = OdinSensor(sim.model, 0, ODIN_MOUNT_XYZ, seed=0)
    for _ in range(a.settle):  # let the wheels find the terrain before anything is measured
        sim.step_many(1)

    rx, ry, yaw, _ = pose_of(sim.current_state.body_q.numpy()[0])
    goal = np.asarray(a.goal if a.goal else sim.goal, np.float64)[:2]
    # the coarse layer and the turn-first brake come from the robot's own table unless a flag
    # says otherwise, so a run here tests what the node would do (helhest.planner_config)
    if a.coarsen is None:
        a.coarsen = (
            max(1, int(round(a.cfg.coarse["block_m"] / a.cell)))
            if a.cfg.coarse["block_m"] > 0
            else 0
        )
    if a.memory is None:
        a.memory = a.cfg.coarse["memory_m"]
    if a.bridge is None:
        a.bridge = a.cfg.coarse["bridge_m"]
    if a.turn_first is None:
        a.turn_first = a.cfg.turn_first["start_deg"]
    if a.turn_first_reach is None:
        a.turn_first_reach = a.cfg.turn_first["reach_m"]
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

    # Three windows, each a centred crop of the belief's, so every offset between them is a
    # constant the captured replan graph can hold. n // 2 - k // 2 rather than (n - k) // 2: the
    # two differ by a cell when the difference is odd, and only this one puts the robot's centre
    # CELL at the centre of all of them.
    #
    #   belief / coarse   the whole window, pooled to `coarsen` cells -- WHICH WAY round
    #   routing           a crop, at full resolution, settle-based       -- HOW to get there
    #   MPPI              a smaller crop, where the rollouts live
    #
    # The routing window is deliberately much smaller than the belief. A fine window larger than
    # the sensor's reliable coverage is not planning over terrain, it is planning over whatever
    # filled the unobserved cells -- and a 14 m one was measured to produce no usable plan at all,
    # because its boundary ring sat outside the 6 m horizon.
    nw = int(round(a.fine / a.cell))
    nr = int(round(a.route / a.cell))
    if not nw <= nr <= n:
        raise SystemExit(f"need fine <= route <= window, got {a.fine} <= {a.route} <= {a.window}")
    off_w = n // 2 - nw // 2
    off_r = n // 2 - nr // 2
    win_grid = GridParams(nw, nw, a.cell, 0.0, 0.0)
    route_grid = GridParams(nr, nr, a.cell, 0.0, 0.0)
    belief_grid = GridParams(n, n, a.cell, 0.0, 0.0)
    mppi = a.controller == "mppi"
    robot = dynamics.robot_params(a.wheel_width)
    plan_sim = (
        None
        if not mppi
        else ForwardSimulator(
            robot,
            # command_delay 0: nothing here feeds sim.command_history, and rolling out against an
            # all-zero history is worse than not modelling the dead time at all
            dynamics.planning_solver(dt=dt, command_delay=0.0),
            win_grid,
            a.batch,
            a.horizon,
            a.device,
        )
    )
    planner = None
    if mppi:
        plan_sim.set_uniform_friction(0.8)
        # The robot's controller, not library defaults: same cost weights, sampler priors, friction
        # replicas and cost-to-go settings, through the one function the node uses too.
        planner = MppiGpu(
            plan_sim,
            a.cfg.cost,
            sampling=a.cfg.sampling,
            n_theta=a.n_theta,
        )
        planner.reset_nominal(a.cfg.nominal_reset)
        planner.set_mu_band(1.0, a.cfg.mu_span)
    # the clearance speed governor: drive only as fast as the room along the plan allows
    governor = None
    if mppi and a.cfg.governor is not None:
        governor = ClearanceGovernor(
            robot,
            plan_dt=float(planner.cw.dt),
            t_react=a.cfg.governor["t_react"],
            v_min=a.cfg.governor["v_min"],
            lookahead_s=a.cfg.governor["lookahead_s"],
            decel=a.cfg.governor["decel"],
            c0=a.cfg.governor["c0"],
            t_turn=a.cfg.governor["t_turn"],
            v_blind=a.cfg.governor["v_blind"],
            device=a.device,
        )
    ctg = CostToGo(
        route_grid,
        robot,
        dynamics.planning_solver(dt=dt, command_delay=0.0),
        **a.cfg.costtogo,
        # the z-margin is SIM-ONLY: the node passes no sigma and no z_veto, so on the robot this
        # whole feasibility test is off. Kept here, flagged rather than silently matched.
        z_veto=a.z_veto,
        charge_per_sigma=a.charge_per_sigma,
        device=a.device,
    )
    if mppi:
        planner.cw.lattice_cap = ctg._vcap
    # the routing field expressed in the MPPI window's frame: a constant cell offset apart
    sgrid = GridParams(nr, nr, a.cell, (off_r - off_w) * a.cell, (off_r - off_w) * a.cell).build()

    coarse = None
    if a.coarsen > 0:
        memory = None
        if a.memory > 0.0:
            # A coarse map anchored to the WORLD, which the belief window slides across, so a
            # wall the robot drove away from is still there when it matters. On the belief's own
            # lattice: it recenters in whole cells, so the window's offset in the map is always
            # whole cells and its blocks pool into the map's blocks exactly. Centred on the start.
            k = int(round((a.memory - span) / 2.0 / a.cell))
            memory = GridParams(
                n + 2 * k, n + 2 * k, a.cell, belief.xmin - k * a.cell, belief.ymin - k * a.cell
            )
        coarse = CoarseRouter(
            belief_grid,
            factor=a.coarsen,
            max_step_m=a.coarse_step,
            min_pass_fraction=a.coarse_pass,
            bridge_m=a.bridge,
            frontier_m=a.frontier,
            void_penalty=a.void_penalty,
            memory_grid=memory,
            device=a.device,
        )
        # the coarse grid's shape is fixed; where it sits in the ROUTING window's frame, which is
        # where the fine solve reads it, is passed per frame (it moves when the map is anchored)
        ctg.set_coarse(
            GridParams(coarse.grid.cells_x, coarse.grid.cells_y, coarse.grid.cell_size, 0.0, 0.0)
        )

    # Preallocated so the per-frame path allocates nothing and touches no host memory. `scratch`
    # exists because `multigrid_inpaint` fills IN PLACE, and the array it would fill is the
    # belief's own height layer.
    zeros2d = lambda k: wp.zeros((k, k), dtype=wp.float32, device=a.device)  # noqa: E731
    scratch, measured_d, sd_d = zeros2d(n), zeros2d(n), zeros2d(n)
    h_r, m_r, sd_r, drift_r = zeros2d(nr), zeros2d(nr), zeros2d(nr), zeros2d(nr)
    fine_d = zeros2d(nw)
    fine_m = zeros2d(nw)  # the belief's measured mask on the MPPI crop, for the governor
    height_d = scratch  # so a run that arrives before its first frame can still dump

    def crop(src: wp.array, off: int, out: wp.array) -> wp.array:
        wp.launch(crop_kernel, dim=out.shape, inputs=[src, off], outputs=[out], device=a.device)
        return out

    pre = ScanPreprocessor(sensor.n_rays, device=a.device)
    box = self_box(robot) if a.self_filter else None
    base_T_sensor = np.eye(4)
    base_T_sensor[:3, 3] = ODIN_MOUNT_XYZ  # the mount is a translation; sensor axes are body axes

    # `body_z` is the integrity check: the robot must stay on the terrain. The simulator's
    # contact buffers are deliberately NOT read -- touching one faults the CUDA context once
    # the step is replayed from a graph, which is why the viewer draws no contacts either.
    trail, closest, reached, body_z = [], 1.0e9, False, []
    # Opt-in, strided history for the scrub page. Host reads, so it is off by default and never
    # on the measured path -- with --history 0 the loop below is byte-identical to before.
    hist: dict[str, list] = {k: [] for k in ("h", "seen", "blk", "v", "route", "cv", "meta")}
    # per frame with the governor: [frame, clearance m, cap m/s, scale, about to sweep unseen ground]
    gov_log: list = []
    esc: list = []  # per recorded frame: [best rollout's worst violation, median, clean fraction]
    # WALL CLEARANCE, every frame, against the world's exact solids. "Reached" cannot see a robot
    # scraping a wall on its way through, and the robust margin exists precisely so that never
    # happens -- so any change to it is judged by this, not by reaching alone.
    fp = footprint(robot)
    clearance: list[float] = []
    prev_diff = None  # last frame's differential, for the turn-first brake's commitment
    # THE ROBOT'S COMMAND CHAIN (control/command.py), as the node runs it: rear follower, the
    # accel/decel slew limits and the magnitude clamp from the params file. Without it the sim
    # reversed the wheels from +5.5 to -2.2 rad/s in one frame and the tricycle stood on its
    # nose for 180 frames (a reverse study, since closed) -- a transition the robot cannot make.
    # `prev_lrr` is the last conditioned command in /cmd_joints order, which is also what the
    # rollouts are seeded from (the node seeds from the encoders when fresh, else from this).
    chain = not a.no_chain
    prev_lrr = np.zeros(3, np.float32)
    max_omega = float(a.plan_params.get("plan_max_omega", 7.5))
    max_slew = float(a.plan_params.get("plan_max_slew", 6.0))
    max_decel = float(a.plan_params.get("plan_max_decel", 8.0))
    turn_boost = float(a.plan_params.get("plan_turn_boost", 1.0))
    prev_pose = None
    prev_xy = None  # last frame's position: the brake stops before it spins while still moving
    for f in range(a.frames):
        body = sim.current_state.body_q.numpy()[0]
        rx, ry, yaw, R = pose_of(body)
        trail.append((rx, ry))
        speed = None if prev_xy is None else float(np.hypot(rx - prev_xy[0], ry - prev_xy[1]) / dt)
        prev_xy = (rx, ry)
        body_z.append(float(body[2]))
        clearance.append(obstacle_clearance(a.world, rx, ry, yaw, fp))
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

        # WHICH WAY -- the coarse layer over the whole belief window, which is the only layer
        # that can see far enough to choose a side. Its value prices the routing window's border.
        vc = None
        r0 = belief.xmin + off_r * a.cell
        s0 = belief.ymin + off_r * a.cell
        if coarse is not None:
            # the coarse grid's origin in the WORLD: fixed when anchored, the window's otherwise.
            # An anchored grid works in world coordinates, a window-bound one in the window's.
            if coarse.persistent:
                cx0, cy0 = coarse.grid.origin_x, coarse.grid.origin_y
                vc = coarse.solve(height_d, measured_d, goal, (belief.xmin, belief.ymin))
            else:
                cx0, cy0 = belief.xmin, belief.ymin
                vc = coarse.solve(height_d, measured_d, (goal[0] - cx0, goal[1] - cy0))

        # HOW -- the settle-based routing window, a crop, where the belief's uncertainty reaches
        # the planner and nowhere before
        V = ctg.compute(
            crop(height_d, off_r, h_r),
            (goal[0] - r0, goal[1] - s0),
            measured=crop(measured_d, off_r, m_r),
            sigma=crop(sd_d, off_r, sd_r),
            drift=crop(belief.drift(), off_r, drift_r),
            coarse_value=vc,
            coarse_origin=None if coarse is None else (cx0 - r0, cy0 - s0),
        )

        # CONTROL -- fine window, cropped from the same belief
        wx0 = belief.xmin + off_w * a.cell
        wy0 = belief.ymin + off_w * a.cell
        state_l = np.array([rx - wx0, ry - wy0, yaw], np.float32)
        goal_l = (goal[0] - wx0, goal[1] - wy0)
        if d < a.dock:
            cmd = dock_control(state_l, goal_l)
        elif mppi:
            plan_sim.set_terrain(crop(height_d, off_w, fine_d))
            if governor is not None:  # what is measured, so it can slow down over what is not
                planner.set_measured(crop(measured_d, off_w, fine_m))
            if chain:
                # the rollouts start from the REALIZED state: the last conditioned command as the
                # wheel seed, the true body twist from the pose delta (the node's odometry twist)
                plan_sim.set_initial_wheel_omega(to_engine_order(prev_lrr))
                if prev_pose is not None:
                    dyaw = (yaw - prev_pose[2] + np.pi) % (2.0 * np.pi) - np.pi
                    vx = (
                        (rx - prev_pose[0]) * np.cos(yaw) + (ry - prev_pose[1]) * np.sin(yaw)
                    ) / dt
                    vy = (
                        -(rx - prev_pose[0]) * np.sin(yaw) + (ry - prev_pose[1]) * np.cos(yaw)
                    ) / dt
                    plan_sim.set_initial_twist(np.array([vx, vy, dyaw / dt], np.float32))
            # V with a way out of every no-route pose, not V itself: where V is capped MPPI used to
            # follow a straight line to the goal, which pressed the robot into false_door's back
            # wall and turned it back into corridor's dead end mid-turn (CostToGo._escape_kernel)
            planner.set_lattice(ctg.V_escape, sgrid)
            if a.cfg.cost.veto > 0.0:
                # actual CONTACT, priced independently of V and hard. Where V is capped -- which
                # is exactly where a pose is vetoed -- the goal term cannot carry a veto, so it has
                # to be its own term. Walls only, and without the router's margin (CostToGo.hazard)
                planner.set_veto(ctg.hazard, sgrid)
            if a.cfg.cost.clear_time > 0.0:  # the wall-distance map the clearance-time cost reads
                planner.update_clearance()
            planner.replan(state_l, goal_l, a.refine)
            u = planner.nominal()
            wl, wr = float(u[0, 0]), float(u[0, 1])
            if a.turn_first > 0.0:
                # spin first when the route lies well behind: an arc that turns while advancing
                # ends inside the robot's own turning clearance of a wall (control/command.py)
                bearing = ctg.descent_bearing(rx - r0, ry - s0, a.turn_first_reach)
                if np.isfinite(bearing):
                    err = (bearing - yaw + np.pi) % (2.0 * np.pi) - np.pi
                    wl, wr = turn_first(
                        wl,
                        wr,
                        err,
                        start_deg=a.turn_first,
                        min_scale=a.turn_first_min,
                        prev_diff=prev_diff,
                        speed=speed,
                    )
            if governor is not None:
                wl_in = wl
                wl, wr = governor.cap(
                    wl, wr, plan_sim.controlled, plan_sim.elevation, planner.measured, plan_sim.grid
                )
                gov_log.append(
                    [
                        f,
                        governor.clearance,
                        governor.v_cap,
                        wl / wl_in if wl_in != 0.0 else 1.0,
                        float(governor.blind),
                    ]
                )
            cmd = np.array([wl, wr, 0.5 * (wl + wr)], np.float32)
        else:
            # the lattice's own policy, walked in the ROUTING window's frame, then followed
            local = (rx - r0, ry - s0, yaw)
            path = trace_optimal(ctg, local, a.n_theta, nr, nr, 0.0, 0.0, a.cell)
            # which way to turn when there is no plan: toward the goal, which is a direction the
            # robot can always name even when the lattice cannot reach it
            bearing = float(np.arctan2(goal[1] - ry, goal[0] - rx) - yaw)
            bearing = (bearing + np.pi) % (2.0 * np.pi) - np.pi
            cmd = carrot_command(
                path, local, robot, a.look, a.carrot_speed, a.spin_deg, a.spin_rate, bearing
            )
        if chain:
            lrr = condition_command(
                float(cmd[0]),
                float(cmd[1]),
                prev_lrr,
                max_omega=max_omega,
                max_slew=max_slew,
                max_decel=max_decel,
                dt=dt,
                turn_boost=turn_boost,
            )
            prev_lrr = lrr
            cmd = to_engine_order(lrr).astype(np.float32)
        cmd = np.clip(cmd, -a.wmax, a.wmax)
        prev_pose = (rx, ry, yaw)

        sim.set_wheel_command(cmd)
        prev_diff = float(cmd[1] - cmd[0])  # what the brake commits a spin's direction to
        if a.history and f % a.history == 0 and mppi and a.escape:
            # Was there a way out? Per sampled rollout, the worst envelope violation over the
            # horizon; then the BEST rollout's worst. Near zero means an escape existed and the
            # weighting did not take it. Large for every rollout means the robot was already
            # committed when it got here, which is a planner problem no cost weight can undo.
            der = plan_sim.derived.numpy()  # [T+1, B] (z, pitch, roll)
            pitch, roll = der[1:, :, 1], der[1:, :, 2]
            viol = np.maximum(np.abs(roll) - robot.max_roll, 0.0)
            viol += np.maximum(-pitch - robot.max_pitch_up, 0.0)
            viol += np.maximum(pitch - robot.max_pitch_down, 0.0)
            per_rollout = viol.max(axis=0)
            esc.append(
                [
                    float(per_rollout.min()),
                    float(np.median(per_rollout)),
                    float((per_rollout <= 1e-6).mean()),
                ]
            )
        if a.history and f % a.history == 0:
            hist["h"].append(height_d.numpy().copy())
            hist["seen"].append((measured_d.numpy() > 0.5).astype(np.uint8))
            vh = V.numpy()
            hist["blk"].append(ctg.blocked.numpy().mean(2))
            hist["v"].append(vh.min(2))
            # Fraction of HEADINGS with a route, which is the one thing `v` above cannot show.
            # `min` over headings calls a cell reachable when any single heading is, and `blk`
            # is about feasibility, not reachability -- so a lattice that had lost half its
            # heading ring read as perfectly healthy in both. It did, for months
            # (incident_2026-09-22_lattice-heading-connectivity.md). On open ground this should
            # be ~1; the split ring made it exactly 0.5.
            hist["route"].append((vh < 0.9 * ctg._vcap).mean(2).astype(np.float32))
            hist["cv"].append(
                np.zeros((1, 1), np.float32) if coarse is None else coarse.V.numpy()[:, :, 0]
            )
            # the belief window recenters in whole cells as the robot moves, so every frame
            # carries the origin its own maps are expressed in
            hist["meta"].append(
                [
                    f,
                    rx,
                    ry,
                    yaw,
                    float(cmd[0]),
                    float(cmd[1]),
                    d,
                    belief.xmin,
                    belief.ymin,
                    *roll_pitch(R),
                    # V at the robot's OWN cell and OWN heading -- what the controller is
                    # actually offered, as against `v`'s best-over-all-headings. The two
                    # disagreeing is the signature worth seeing: on `pocket` the minimum read
                    # 5.2 m while the heading the robot held read the cap.
                    _v_here(vh, rx - r0, ry - s0, yaw, a.cell, a.n_theta),
                ]
            )
        # Replayed as a captured CUDA graph, not launched from Python: the solver is launch-bound
        # (16 Newton x 26 PCR iterations of small kernels), and the eager step was 80% of the
        # frame -- 81 ms of a 101 ms frame, against a 69 ms sensor period. Same physics: the step
        # copies next_state back into current_state on device, and eager-vs-graph trajectories
        # differ no more than two eager runs do (the contact solve is not bitwise repeatable).
        sim.step_many(1)
        if f % a.report == 0:
            # the only host reads in the loop, and they happen on report frames alone.
            # Two masks, because the windows are different sizes: coverage is a property of the
            # BELIEF, and blocked is a property of the routing crop.
            blk = ctg.blocked.numpy()
            seen = measured_d.numpy() != 0.0
            rseen = m_r.numpy() != 0.0
            print(
                f"  f{f:>4d}  at ({rx:6.2f},{ry:6.2f}) yaw {np.degrees(yaw):7.1f}d  "
                f"goal {d:5.2f} m  cmd [{cmd[0]:5.2f} {cmd[1]:5.2f}]  "
                f"pts {count:>5d}  seen {100*seen.mean():4.1f}%  "
                # two readings of the same array: what fraction of seen cells is vetoed for
                # ANY of the headings, and what fraction of (cell, heading) pairs. The first
                # is what a "blocked" map shows and is pessimistic by construction -- the
                # settle straddles a thin obstacle, so a cell beside one is blocked for a
                # minority of headings and still counts.
                f"blocked {100*blk.max(2)[rseen].mean():4.1f}% of cells / "
                f"{100*blk[rseen].mean():4.1f}% of poses"
            )
    if a.out:
        # The belief and the routing window are different sizes now, so each array is dumped with
        # the origin it is expressed in. Anything that reads this and assumes one grid is wrong.
        np.savez_compressed(
            a.out,
            # which controller produced this: the resolved plan_* values, as JSON
            plan_config=np.array(json.dumps(resolve(a.plan_params))),
            dt=dt,  # [s] simulated time per frame, so a replay can run in real time
            trail=np.array(trail, np.float64),
            body_z=np.array(body_z, np.float64),
            clearance=np.array(clearance, np.float64),  # [m] per frame, < 0 = touching a wall
            goal=goal,
            cell=a.cell,
            reached=reached,
            height=height_d.numpy(),
            seen=measured_d.numpy() != 0.0,
            bounds=np.array([belief.xmin, belief.ymin], np.float64),
            blocked=ctg.blocked.numpy(),
            V=ctg.V.numpy(),
            route_bounds=np.array(
                [belief.xmin + off_r * a.cell, belief.ymin + off_r * a.cell], np.float64
            ),
            coarse_V=(np.zeros((0, 0)) if coarse is None else coarse.V.numpy()[:, :, 0]),
            coarse_cell=(0.0 if coarse is None else coarse.grid.cell_size),
            # an anchored coarse map has one origin for the whole run; a window-bound one sits at
            # each frame's belief origin (hist_meta[:, 7:9])
            coarse_memory=(coarse is not None and coarse.persistent),
            coarse_bounds=np.array([cx0, cy0] if coarse is not None else [0.0, 0.0], np.float64),
            # the windows are robot-centred, so each recorded frame carries its own origin
            **{f"hist_{k}": np.asarray(v) for k, v in hist.items() if v},
            **({"escape": np.asarray(esc)} if esc else {}),
            **({"governor": np.asarray(gov_log)} if gov_log else {}),
        )
        print(f"wrote {a.out}")

    return dict(
        reached=reached,
        frames=f + 1,
        closest=closest,
        clearance=clearance,
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
    # 20 m, not 30: the sensor's useful GROUND coverage falls off well before that (100% at 4 m,
    # 74-85% at 6, 41-49% at 8, 12-14% at 12 on out_odin0), and on these worlds a 30 m window was
    # also 5.6x the area of the world itself -- 83% of it void the coarse layer then routed
    # through. 20 m is 2.25x cheaper and loses nothing that was ever measured.
    p.add_argument("--window", type=float, default=20.0, help="[m] belief + coarse window")
    p.add_argument("--route", type=float, default=10.0, help="[m] settle-based routing window")
    p.add_argument("--fine", type=float, default=9.0, help="[m] MPPI window, a centred crop")
    p.add_argument(
        "--coarsen",
        type=int,
        default=None,
        help="fine cells per coarse cell; 0 = OFF; default from plan_coarse_block_m",
    )
    p.add_argument(
        "--memory",
        type=float,
        default=None,
        help="[m] side of the world-anchored coarse map, centred on the start; 0 = bound to the "
        "belief window, which forgets what scrolls out of it; default plan_coarse_memory_m",
    )
    p.add_argument("--coarse-step", type=float, default=0.25, help="[m] climbable step, coarse")
    p.add_argument(
        "--coarse-pass",
        type=float,
        default=0.5,
        help="climbable fraction to cross (measured 0.1 vs 0.9: <2% on the loop, non-monotonic -- "
        "this knob moves the coarse field hugely and the wheels not at all; see coarse.py)",
    )
    p.add_argument("--frontier", type=float, default=3.0, help="[m] unseen ground that stays free")
    p.add_argument(
        "--bridge",
        type=float,
        default=None,
        help="[m] an unseen run this short between two sealed wall blocks is the wall; 0 = off; "
        "default plan_bridge_m",
    )
    p.add_argument("--void-penalty", type=float, default=1.0, help="[m] per cell of unseen beyond")
    p.add_argument("--cell", type=float, default=0.2)
    p.add_argument(
        "--turn-first",
        type=float,
        default=None,
        help="[deg] heading error past which the forward speed brakes for a turn in place; "
        "0 = off; default plan_turn_first_deg",
    )
    p.add_argument(
        "--turn-first-min",
        type=float,
        default=0.1,
        help="forward speed scale left at a full brake; 0 = a pure spin",
    )
    p.add_argument(
        "--turn-first-reach",
        type=float,
        default=None,
        help="[m] how far to look for the way on; default plan_turn_first_reach_m",
    )
    p.add_argument("--carve", type=float, default=6.0, help="[m] 0 disables the visibility carve")
    # Both sigma terms default to the ROBOT's values, which is off: the node passes no sigma, so
    # neither shapes its field. Left on here they painted 15-20 m of charge over the half of a
    # room the robot had not driven through (false_door f240) -- a planner nobody deploys.
    p.add_argument("--z-veto", type=float, default=0.0, help="veto below this many sigmas; 0 = off")
    p.add_argument(
        "--charge-per-sigma",
        type=float,
        default=0.0,
        help="routing charge per sigma of pose/drift uncertainty under the footprint; 0 = off",
    )
    p.add_argument("--n-theta", type=int, default=None)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--refine", type=int, default=3)
    p.add_argument(
        "--dock",
        type=float,
        default=3.0,
        # At 1.5 the forward-only MPPI orbits the goal instead of arriving: measured, it circles
        # from 2.56 m out and never closes. 3.0 reaches, and reaches with the sigma veto ON --
        # z_veto = 0 does NOT rescue 1.5, so the veto was never what was holding it off.
        help="[m] hand over to the dock controller",
    )
    p.add_argument("--reach", type=float, default=0.4, help="[m] counts as arrived")
    p.add_argument("--wheel-width", type=float, default=None)
    p.add_argument(
        "--no-self-filter",
        dest="self_filter",
        action="store_false",
        help="let the robot's own wheels into the map -- it walls itself in; see the docstring",
    )
    p.add_argument("--report", type=int, default=25)
    p.add_argument("--out", default=None, help="npz of the final map + trail, for the figure")
    p.add_argument(
        "--history",
        type=int,
        default=0,
        help="also record every Nth frame into --out, for the scrub page; 0 = OFF",
    )
    p.add_argument("--controller", choices=("mppi", "carrot"), default="mppi")
    p.add_argument("--look", type=float, default=1.2, help="[m] carrot lookahead along the plan")
    p.add_argument("--carrot-speed", type=float, default=0.8, help="[m/s] carrot cruise")
    p.add_argument(
        "--veto",
        type=float,
        default=None,
        help="override plan_wall_veto, MPPI's hard veto on the cost-to-go's wall field; 0 = OFF",
    )
    p.add_argument(
        "--escape",
        action="store_true",
        help="record whether any sampled rollout avoided the envelope (MPPI only)",
    )
    p.add_argument("--spin-deg", type=float, default=35.0, help="turn on the spot past this error")
    p.add_argument(
        "--spin-rate", type=float, default=0.9, help="[rad/s] body yaw rate when spinning"
    )
    p.add_argument("--wmax", type=float, default=None, help="[rad/s] wheel-speed clamp")
    p.add_argument(
        "--no-chain",
        action="store_true",
        help="skip the robot's command chain (slew/decel limits, rear follower) and the rollout "
        "state seeding -- the old behaviour, which lets the wheels change speed in one frame",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--params",
        default=str(_ROBOT_PARAMS),
        help="ROS params file whose plan_* values build the planner -- by default the ROBOT'S own, "
        "so a result here is a result about the deployed controller. 'none' = table defaults. "
        "--n-theta/--horizon/--batch/--wmax/--wheel-width override it.",
    )
    a = p.parse_args()
    a.plan_params = _plan_params(a)
    a.cfg = planner_config(a.plan_params)
    a.n_theta, a.horizon, a.batch = a.cfg.n_theta, a.cfg.horizon, a.cfg.batch
    a.wmax, a.wheel_width = a.cfg.sampling.wmax, a.cfg.wheel_width

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
    cl = np.array(r["clearance"])
    if np.isfinite(cl).any():
        # the line a margin change is judged by: it must never go negative
        print(
            f"  WALL CLEARANCE min {cl.min():+.3f} m   frames touching a wall {int((cl < 0).sum())}"
        )
    else:
        print("  WALL CLEARANCE n/a (no solid obstacles in this world)")


if __name__ == "__main__":
    main()
