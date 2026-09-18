"""Settle-based orientation-aware cost-to-go V(x, y, theta).

Feasibility comes from the robot's OWN settle, not a thresholded traversability map: for every pose
(x, y, theta) the robot is placed on the terrain and the engine's residual / clearance / tilt are
read. A pose is blocked iff residual > resid_tol OR clearance < clear_margin OR the body exceeds the
robot's stability ENVELOPE -- |roll| > max_roll, or pitch beyond the asymmetric climb/descend limits
(climbing is nose-up = negative pitch). So feasibility is direction-aware: a side-slope is fine to
CLIMB head-on (pitch, tolerated) but blocked to traverse sideways (roll, dangerous) -- exactly what
the orientation-aware lattice can exploit. The envelope + turn radius come from RobotParams, the
SAME robot the rollouts drive. The static (zero-control) settle is friction-independent, so compute()
needs only the elevation and the goal.

Among FEASIBLE poses the arc cost prefers flatter ground via a graded penalty that splits into two
non-redundant pieces: the per-axis SHAPE (roll_cost_weight : pitch_cost_weight, the robot's relative
roll-vs-pitch susceptibility, from RobotParams) and a single STRENGTH gain (flatness_weight, a planner
knob: how much detour to trade for flatness). The lattice arc cost is
    arc_len * (1 + flatness_weight * mean(roll_cost_weight*|roll| + pitch_cost_weight*|pitch|)).
flatness_weight is the only global gain; the per-axis weights only set the shape (keep them a ratio,
e.g. 1.0 : 0.5, not a second gain).

This is the settle-based feasibility PRODUCER: it settles the robot at every pose to make the per-pose
blocked / graded-tilt fields, then hands them to the LatticeValueSolver (lattice_solver.py) that does
the forward-arc value iteration. (The solver was vendored from helhest.perception.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..engine import ForwardSimulator
from ..engine.robot import Robot  # the built struct, passed straight into the feasibility kernel
from ..engine.terrain import Grid
from ..engine.terrain import sample_field
from ..heightmap import Heightmap
from ..profiling import StageProfiler
from .lattice_solver import LatticeValueSolver

if TYPE_CHECKING:
    from ..engine import GridParams
    from ..engine import RobotParams
    from ..engine import SolverParams


@wp.kernel
def _clamp3d_kernel(
    v_in: wp.array3d(dtype=wp.float32),
    vcap: wp.float32,
    v_out: wp.array3d(dtype=wp.float32),
):
    """Copy V, replacing the solver's +inf (unreachable) with a large finite cap so the cost
    kernel's trilinear sampling never blends to inf."""
    r, c, t = wp.tid()
    val = v_in[r, c, t]
    if val > vcap:
        v_out[r, c, t] = vcap
    else:
        v_out[r, c, t] = val


@wp.kernel
def _feasibility_kernel(
    derived: wp.array2d(dtype=wp.vec3f),  # (z, pitch, roll) per pose; row 0 = the static settle
    residual: wp.array2d(dtype=wp.float32),
    clearance: wp.array2d(dtype=wp.float32),
    robot: Robot,
    blocked: wp.array3d(dtype=wp.float32),
    tilt: wp.array3d(dtype=wp.float32),
):
    """The per-pose feasibility OR + graded tilt cost, one thread per (y, x, theta), no readback.
    Direction-aware: climb = nose-up = NEGATIVE pitch, so the climb limit is on -pitch, descend on
    +pitch. Pose b = (r*nx + c)*n_theta + t is the C-order flatten matching start_pose in __init__.
    """
    r, c, t = wp.tid()
    nx = blocked.shape[1]
    n_theta = blocked.shape[2]
    b = (r * nx + c) * n_theta + t
    der = derived[0, b]
    pitch = der[1]
    roll = der[2]
    over_envelope = (
        wp.abs(roll) > robot.max_roll or pitch < -robot.max_pitch_up or pitch > robot.max_pitch_down
    )
    if over_envelope or residual[0, b] > robot.resid_tol or clearance[0, b] < robot.clear_margin:
        blocked[r, c, t] = 1.0
    else:
        blocked[r, c, t] = 0.0
    tilt[r, c, t] = robot.roll_cost_weight * wp.abs(roll) + robot.pitch_cost_weight * wp.abs(pitch)


@wp.kernel
def _margin_kernel(
    derived: wp.array2d(dtype=wp.vec3f),  # (z, pitch, roll) per pose; row 0 = the static settle
    clearance: wp.array2d(dtype=wp.float32),
    sigma: wp.array2d(dtype=wp.float32),  # per-cell MEASUREMENT sd of the elevation belief [m]
    grid: Grid,
    robot: Robot,
    n_theta: wp.int32,
    sigma_floor_m: wp.float32,
    k_sigma: wp.float32,
    z_ref: wp.float32,
    margin_weight: wp.float32,
    blocked: wp.array3d(dtype=wp.float32),
    tilt: wp.array3d(dtype=wp.float32),
    zmargin: wp.array3d(dtype=wp.float32),
):
    """Safety margin in SIGMAS, and the veto and graded penalty that come off it.

    Each feasibility test is asked how much room is left in units of its OWN uncertainty:

        z_roll  = (max_roll - |roll|)           / sigma_roll
        z_climb = (max_pitch_up + pitch)        / sigma_pitch     (climb = NEGATIVE pitch)
        z_desc  = (max_pitch_down - pitch)      / sigma_pitch
        z_clear = (clearance - clear_margin)    / sigma_clear
        z       = min over tests                                  -- the binding constraint

    Dividing each by its own sigma is what makes the `min` meaningful: roll is in radians and
    clearance in metres, and a raw `min` over those compares nothing. `blocked = z < k_sigma`
    then has one interpretable knob -- how many sigmas of room the robot insists on -- and the
    same number drives the graded penalty, so pessimism and "how close is this to bad" are not
    two independently-tuned things.

    Attitude sigmas come from the settle's closed-form rows. With wheels at (0, +b), (0, -b),
    (-l, 0), `roll = (e1 - e2)/2b` and `pitch = (e3 - (e1+e2)/2)/l`, and both are DIFFERENCES of
    supports -- so the pose drift shared by every cell cancels exactly, and what enters is the
    per-cell MEASUREMENT sd, not the total. That is why `sigma` here must be the belief's
    `meas_sd` and not its `sigma`.

    `sigma_floor_m` is not optional. Without it a perfectly known map makes a pose at 14.9 deg
    of roll against a 15 deg limit read as infinitely safe; the floor is the irreducible error
    -- localisation, controller tracking, model -- that never reaches zero.

    Two approximations, both marked for upgrade:
      - sigma is sampled at each WHEEL CENTRE rather than at the cell that won the envelope
        dilation. Elevation sigma varies smoothly with observation range (~3 cm/m measured), so
        over the <=0.35 m to the contact cell this is worth ~1 cm; the terrain max it stands in
        for is not smooth at all, but sigma is.
      - the footprint maximum is not folded. Reading sigma off one cell is the linearized,
        one-hot estimate, which overstates the sd at contested contacts; the Clark fold at the
        dilation stage is the fix, and it needs the envelope's own contact indices.
    """
    r, c, t = wp.tid()
    nx = blocked.shape[1]
    b = (r * nx + c) * n_theta + t
    der = derived[0, b]
    pitch = der[1]
    roll = der[2]

    x = grid.origin_x + float(c) * grid.cell_size
    y = grid.origin_y + float(r) * grid.cell_size
    yaw = (float(t) + 0.5) * 2.0 * 3.14159265 / float(n_theta)
    ca = wp.cos(yaw)
    sa = wp.sin(yaw)
    # Read the layout off the robot itself rather than reconstructing it: `wheel_pos` is the
    # same array the settle uses, so the sigma is sampled where the supports actually are.
    w0 = robot.wheel_pos[0]
    w1 = robot.wheel_pos[1]
    w2 = robot.wheel_pos[2]
    hb = w0[1]  # half track
    rl = -w2[0]  # rear offset

    # Measurement sd under each wheel, and midway back for the belly.
    s1 = _sigma_at(
        sigma, grid, x + ca * w0[0] - sa * w0[1], y + sa * w0[0] + ca * w0[1], sigma_floor_m
    )
    s2 = _sigma_at(
        sigma, grid, x + ca * w1[0] - sa * w1[1], y + sa * w1[0] + ca * w1[1], sigma_floor_m
    )
    s3 = _sigma_at(
        sigma, grid, x + ca * w2[0] - sa * w2[1], y + sa * w2[0] + ca * w2[1], sigma_floor_m
    )
    sb = _sigma_at(sigma, grid, x - ca * rl * 0.5, y - sa * rl * 0.5, sigma_floor_m)

    two_b = 2.0 * hb
    var_roll = (s1 * s1 + s2 * s2) / (two_b * two_b)
    var_pitch = (s3 * s3 + 0.25 * (s1 * s1 + s2 * s2)) / (rl * rl)
    # The belly sits on a weighted mean of the three supports (weights summing to one), so its
    # own height carries about a third of their variance. The cross term against the ground
    # beneath it is dropped, which OVERSTATES sigma_clear -- the conservative direction.
    var_clear = sb * sb + (s1 * s1 + s2 * s2 + s3 * s3) / 9.0

    sigma_roll = wp.sqrt(wp.max(var_roll, 1.0e-12))
    sigma_pitch = wp.sqrt(wp.max(var_pitch, 1.0e-12))
    sigma_clear = wp.sqrt(wp.max(var_clear, 1.0e-12))

    z_roll = (robot.max_roll - wp.abs(roll)) / sigma_roll
    z_climb = (robot.max_pitch_up + pitch) / sigma_pitch
    z_desc = (robot.max_pitch_down - pitch) / sigma_pitch
    z_clear = (clearance[0, b] - robot.clear_margin) / sigma_clear

    z = wp.min(wp.min(z_roll, z_climb), wp.min(z_desc, z_clear))
    zmargin[r, c, t] = z
    if z < k_sigma:
        blocked[r, c, t] = 1.0
    if z < z_ref:  # graded: pay for being near a boundary, not only for crossing it
        tilt[r, c, t] = tilt[r, c, t] + margin_weight * (z_ref - z)


@wp.func
def _sigma_at(
    sigma: wp.array2d(dtype=wp.float32),
    grid: Grid,
    x: wp.float32,
    y: wp.float32,
    floor_m: wp.float32,
) -> wp.float32:
    """Elevation sd at a world point, floored. Outside the grid the map knows nothing."""
    return wp.max(sample_field(sigma, grid, x, y), floor_m)


@wp.kernel
def _local_step_kernel(
    elev: wp.array2d(dtype=wp.float32),
    measured: wp.array2d(dtype=wp.float32),  # 1 = cell has real data, 0 = never observed
    step: wp.array2d(dtype=wp.float32),
):
    """Per-cell prominence: how much a cell rises above its immediate (3x3) neighbourhood -- a STEP.
    A thin pole rises ~its full height above the adjacent ground (large step); a drivable slope rises
    only cell_size*tan(theta) per cell (small step). This lets the gate below catch vertical obstacles
    the settle STRADDLES (a stick that fits between the wheel/belly contacts) without blocking slopes.

    UNOBSERVED cells carry no elevation evidence, so they neither get a prominence of their own nor
    lower a neighbour's minimum. Without that, the caller's blind-cell fill (a constant, e.g. 0.0)
    reads as a real step wherever the ground sits away from that constant, and the map frontier
    gates off as a closed ring. Prominence at the frontier is still taken over MEASURED
    neighbours, so a pole standing at the edge of the mapped area is still caught."""
    r, c = wp.tid()
    ny = elev.shape[0]
    nx = elev.shape[1]
    if measured[r, c] < 0.5:
        step[r, c] = 0.0
        return
    lo = elev[r, c]
    for i in range(-1, 2):
        rr = wp.clamp(r + i, 0, ny - 1)
        for j in range(-1, 2):
            cc = wp.clamp(c + j, 0, nx - 1)
            if measured[rr, cc] > 0.5:
                lo = wp.min(lo, elev[rr, cc])
    step[r, c] = elev[r, c] - lo


@wp.kernel
def _step_gate_kernel(
    step: wp.array2d(dtype=wp.float32),
    foot_r: int,
    step_gate: wp.float32,
    blocked: wp.array3d(dtype=wp.float32),
):
    """OR a hard block (ALL headings) onto any pose whose footprint (radius foot_r cells) contains a
    STEP taller than step_gate -- a vertical obstacle the body would hit but the settle straddles.
    Heading-independent: the robot cannot be centred within foot_r cells of a tall pole in ANY
    orientation. Only ever SETS blocked=1 (never clears), so it composes with the settle feasibility.
    """
    r, c, t = wp.tid()
    ny = step.shape[0]
    nx = step.shape[1]
    hit = float(0.0)
    for i in range(-foot_r, foot_r + 1):
        rr = wp.clamp(r + i, 0, ny - 1)
        for j in range(-foot_r, foot_r + 1):
            cc = wp.clamp(c + j, 0, nx - 1)
            hit = wp.max(hit, step[rr, cc])
    if hit > step_gate:
        blocked[r, c, t] = 1.0


@wp.kernel
def _erode_feasible_kernel(
    blocked: wp.array3d(dtype=wp.float32),
    dr: int,
    dc: int,
    dt: int,
    robust: wp.array3d(dtype=wp.float32),
):
    """Robust feasibility: a pose is blocked if ANY pose within the (y, x, theta) tube is blocked --
    i.e. erode the feasible set by the disturbance the closed loop can't correct before the next
    replan. Orientation-aware: the theta window (which WRAPS) blocks a cell where a small slip-heading
    error would tip the robot, so the margin is heading-dependent, not a fixed radial inflation. This
    is a max-pool (dilation of `blocked`); dr=dc=dt=0 copies `blocked` (a no-op)."""
    r, c, t = wp.tid()
    ny = blocked.shape[0]
    nx = blocked.shape[1]
    nth = blocked.shape[2]
    hit = float(0.0)
    for i in range(-dr, dr + 1):
        rr = wp.clamp(r + i, 0, ny - 1)
        for j in range(-dc, dc + 1):
            cc = wp.clamp(c + j, 0, nx - 1)
            for k in range(-dt, dt + 1):
                tt = (t + k + nth) % nth  # heading wraps
                hit = wp.max(hit, blocked[rr, cc, tt])
    robust[r, c, t] = hit


@wp.kernel
def _goal_cell_kernel(
    goal_xy: wp.array(dtype=wp.float32),  # [2] world (x, y)
    xmin: wp.float32,
    ymin: wp.float32,
    resolution: wp.float32,
    height: wp.int32,
    width: wp.int32,
    goal_rc: wp.array(dtype=wp.int32),  # [2] out (row, col)
):
    goal_rc[0] = wp.clamp(int((goal_xy[1] - ymin) / resolution), 0, height - 1)  # row from y
    goal_rc[1] = wp.clamp(int((goal_xy[0] - xmin) / resolution), 0, width - 1)  # col from x


class CostToGo:
    def __init__(
        self,
        grid_params: GridParams,
        robot_params: RobotParams,
        solver_params: SolverParams,
        n_theta: int = 24,
        step: float = 0.3,
        flatness_weight: float = 2.0,  # planner strength: how much detour to trade for flat ground
        robust_margin_m: float = 0.0,  # lateral disturbance tube -> erode the feasible set by this
        robust_margin_deg: float = 0.0,  # heading disturbance tube (orientation-aware erosion)
        obstacle_step_m: float = 0.0,  # hard-block cells with a local step taller than this [m];
        # 0 = OFF. Catches thin vertical obstacles (sticks/poles) the settle straddles.
        pivot_cost: float = 0.0,  # [m-equiv] per heading bin; > 0 adds point-turn primitives so
        # a goal behind the robot routes as pivot-then-drive instead of a wide loop. 0 = OFF.
        # --- probabilistic feasibility (z-margin). k_sigma = 0 is EXACTLY the old behaviour:
        # the hard thresholds still veto, nothing is added, and `compute` need not be passed a
        # sigma. Above 0 a pose must additionally hold k_sigma standard deviations of room on
        # every test, measured against the elevation belief's own per-cell MEASUREMENT sd.
        k_sigma: float = 0.0,
        sigma_floor_m: float = 0.02,  # irreducible map error: localisation, tracking, model
        z_ref: float = 4.0,  # start charging for proximity to a boundary below this many sigmas
        margin_weight: float = 0.0,  # [m-equiv] per sigma of shortfall; 0 = veto only
        profile: bool = False,  # opt-in per-stage CUDA-event timing (tiny event nodes + per-call sync)
        device: wp.Device | str | None = None,
    ) -> None:

        self.device = wp.get_device(device)
        self.flatness_weight = flatness_weight
        self.k_sigma = float(k_sigma)
        self.sigma_floor_m = float(sigma_floor_m)
        self.z_ref = float(z_ref)
        self.margin_weight = float(margin_weight)
        self.n_theta = int(n_theta)
        self.robot = robot_params.build(self.device)
        self.grid = grid_params.build()
        self.bounds = grid_params.bounds  # (xmin, xmax, ymin, ymax) the solver takes
        # robust-feasibility tube in lattice cells / theta-bins (0 -> no erosion = plain feasibility)
        self._mr = int(round(robust_margin_m / self.grid.cell_size))
        self._mc = self._mr
        self._mt = int(round(robust_margin_deg / (360.0 / n_theta)))
        self._eroded = self._mr > 0 or self._mt > 0
        # tall-step obstacle gate: block cells within a robot footprint of a step > obstacle_step_m.
        self._step_gate = float(obstacle_step_m)
        self._foot_r = max(1, int(round(robot_params.half_track / self.grid.cell_size)))

        self._vcap = (
            1.5
            * (self.grid.cells_x + self.grid.cells_y)
            * self.grid.cell_size
            * (1.0 + self.flatness_weight)
        )

        ny, nx = self.grid.cells_y, self.grid.cells_x
        self.settle_sim = ForwardSimulator(
            robot_params=robot_params,
            solver_params=solver_params,
            grid_params=grid_params,
            batch_size=nx * ny * n_theta,
            n_steps=1,
            device=self.device,
        )
        rr, cc, tt = np.meshgrid(np.arange(ny), np.arange(nx), np.arange(n_theta), indexing="ij")
        px = (self.grid.origin_x + cc * self.grid.cell_size).ravel().astype(np.float32)
        py = (self.grid.origin_y + rr * self.grid.cell_size).ravel().astype(np.float32)
        ph = ((tt + 0.5) * 2.0 * np.pi / n_theta).ravel().astype(np.float32)  # bin-center heading
        self.settle_sim.start_pose.assign(np.stack([px, py, ph], 1))
        self.settle_sim.target_wheel_omega.zero_()
        self._mu = Heightmap(
            np.full((self.grid.cells_y, self.grid.cells_x), 0.8, np.float32),
            (self.grid.origin_x, self.grid.origin_y),
            self.grid.cell_size,
        )
        self.settle_sim.set_friction(self._mu)

        self.solver = LatticeValueSolver(
            self.grid.cell_size,
            self.grid.cells_y,
            self.grid.cells_x,
            n_theta=n_theta,
            turn_radius=self.robot.min_turn_radius,
            step=step,
            pivot_cost=pivot_cost,
            device=self.device,
        )

        self.V = wp.zeros(
            (self.grid.cells_y, self.grid.cells_x, n_theta),
            dtype=wp.float32,
            device=self.device,
        )
        self.blocked = wp.zeros_like(self.V)
        self.zmargin = wp.zeros_like(self.V)  # safety margin in sigmas, per pose
        self.robust_blocked = wp.zeros_like(self.V)  # blocked after the disturbance-tube erosion
        self.graded_tilt = wp.zeros_like(self.V)
        self._step = wp.zeros((ny, nx), dtype=wp.float32, device=self.device)  # per-cell prominence

        self._elev_in = wp.zeros((ny, nx), dtype=wp.float32, device=self.device)
        # Per-cell measurement sd. Defaults to zero, which the floor then lifts to
        # `sigma_floor_m` everywhere -- so an unsupplied sigma is a uniform-uncertainty map, not
        # a claim of perfect knowledge.
        self._sigma_in = wp.zeros((ny, nx), dtype=wp.float32, device=self.device)
        # Stable mask buffer the captured graph reads. Defaults to all-measured, so a caller that
        # passes no mask gets exactly the pre-mask behaviour.
        self._measured_in = wp.full((ny, nx), 1.0, dtype=wp.float32, device=self.device)
        self._goal_xy = wp.zeros(2, dtype=wp.float32, device=self.device)
        self._goal_rc = wp.zeros(2, dtype=wp.int32, device=self.device)
        self._graph = None

        self._prof = StageProfiler(
            self.device, ("settle", "feasibility", "route", "clamp"), profile
        )
        self._n_compute = 0

    def reset_timing(self) -> None:
        """Clear the accumulated per-stage timing stats (e.g. after a warmup run)."""
        self._prof.reset()

    def timing_stats(self) -> dict:
        """Per-stage timing over profiled compute() calls (CUDA + profile=True), the build/warmup call
        excluded: {stage: {"mean_ms", "std_ms", "n"}}. Use the means (the profiling run is serialized
        by the event reads, so its wall-clock runs slower than real throughput)."""
        return self._prof.stats()

    def _record_compute(self, capture: bool) -> None:
        """Record the whole pipeline on stable owned buffers: terrain -> settle -> per-pose
        feasibility -> goal cell (on device) -> value iteration -> clamp into V. Used both to build
        the captured graph (capture=True) and for the eager CPU fallback (capture=False)."""
        sim = self.settle_sim
        self._prof.mark(0)
        sim.set_terrain(self._elev_in)  # D2D copy + envelope rebuild from the stable terrain buffer
        sim.rollout_launch()
        self._prof.mark(1)  # settle done
        wp.launch(
            _feasibility_kernel,
            dim=self.V.shape,
            inputs=[sim.derived, sim.residual, sim.clearance, self.robot],
            outputs=[self.blocked, self.graded_tilt],
            device=self.device,
        )
        if self.k_sigma > 0.0 or self.margin_weight > 0.0:
            wp.launch(
                _margin_kernel,
                dim=self.V.shape,
                inputs=[
                    sim.derived,
                    sim.clearance,
                    self._sigma_in,
                    self.grid,
                    self.robot,
                    self.n_theta,
                    self.sigma_floor_m,
                    self.k_sigma,
                    self.z_ref,
                    self.margin_weight,
                ],
                outputs=[self.blocked, self.graded_tilt, self.zmargin],
                device=self.device,
            )
        if self._step_gate > 0.0:  # hard-block tall steps the settle straddles (thin poles/sticks)
            wp.launch(
                _local_step_kernel,
                dim=(self.grid.cells_y, self.grid.cells_x),
                inputs=[self._elev_in, self._measured_in],
                outputs=[self._step],
                device=self.device,
            )
            wp.launch(
                _step_gate_kernel,
                dim=self.V.shape,
                inputs=[self._step, self._foot_r, self._step_gate],
                outputs=[self.blocked],
                device=self.device,
            )
        self._prof.mark(2)  # feasibility done
        if self._eroded:  # erode the feasible set by the disturbance tube (robust feasibility)
            wp.launch(
                _erode_feasible_kernel,
                dim=self.V.shape,
                inputs=[self.blocked, self._mr, self._mc, self._mt],
                outputs=[self.robust_blocked],
                device=self.device,
            )
        feas = self.robust_blocked if self._eroded else self.blocked
        wp.launch(
            _goal_cell_kernel,
            dim=1,
            inputs=[
                self._goal_xy,
                self.bounds[0],
                self.bounds[2],
                self.grid.cell_size,
                self.grid.cells_y,
                self.grid.cells_x,
            ],
            outputs=[self._goal_rc],
            device=self.device,
        )
        result = self.solver._record_solve(
            feas, self.graded_tilt, self._goal_rc, self.flatness_weight, capture
        )
        self._prof.mark(3)  # value iteration done (goal cell + solve)
        wp.launch(
            _clamp3d_kernel,
            dim=self.V.shape,
            inputs=[result, self._vcap],
            outputs=[self.V],
            device=self.device,
        )
        self._prof.mark(4)  # clamp done

    def compute(
        self,
        elevation: wp.array,
        goal_xy: tuple[float, float],
        measured: wp.array | None = None,
        sigma: wp.array | None = None,
    ) -> wp.array:
        """elevation [ny, nx] device wp.array + goal -> clamped V[ny, nx, n_theta]. The entire solve
        (settle + value iteration) is captured ONCE as a CUDA graph and replayed each call with the
        new terrain/goal (copied into stable device buffers first) -- no host syncs in the loop.

        `measured` [ny, nx] (1 = observed, 0 = blind) is read ONLY by the obstacle_step_m gate, to
        keep the caller's blind-cell fill from reading as a real step. Omit it (or pass
        obstacle_step_m=0) and every cell counts as observed.

        `sigma` [ny, nx] is the elevation belief's per-cell MEASUREMENT sd (its `meas_sd`, not
        its `sigma`: pose drift is common-mode and cancels in the attitude differences). It is
        read only when `k_sigma` or `margin_weight` is non-zero; omitted, every cell falls back
        to `sigma_floor_m`."""
        assert (
            elevation.device == self.device
        ), f"elevation must be a wp.array on {self.device}, got {elevation.device}"

        wp.copy(self._elev_in, elevation)
        if sigma is None:
            self._sigma_in.zero_()  # the floor then applies uniformly
        else:
            assert (
                sigma.device == self.device
            ), f"sigma must be a wp.array on {self.device}, got {sigma.device}"
            wp.copy(self._sigma_in, sigma)
        if self._step_gate > 0.0:  # only the gate reads the mask
            if measured is None:
                self._measured_in.fill_(1.0)
            else:
                assert (
                    measured.device == self.device
                ), f"measured must be a wp.array on {self.device}, got {measured.device}"
                wp.copy(self._measured_in, measured)
        self._goal_xy.assign(np.asarray(goal_xy[:2], np.float32))

        if self.device.is_cuda:
            if self._graph is None:
                with wp.ScopedCapture(device=self.device) as cap:
                    self._record_compute(capture=True)
                self._graph = cap.graph
            wp.capture_launch(self._graph)
        else:
            self._record_compute(capture=False)

        self._n_compute += 1
        if self._prof.enabled and self._n_compute > 1:  # skip the graph-build sample
            self._prof.accumulate()
        return self.V
