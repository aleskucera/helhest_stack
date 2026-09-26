"""Odin's feasibility as terrain_value_field constraints: the settle, read as margins.

For every pose (x, y, theta) the robot is placed on the terrain -- the engine's static settle --
and every test it can fail becomes one constraint of `terrain_value_field`, which then decides
what is blocked, what it costs and what is doubted:

    ROLL      max_roll - |roll|                 [rad]   soft
    CLIMB     max_pitch_up + pitch              [rad]   soft  (climb = nose-up = NEGATIVE pitch)
    DESCEND   max_pitch_down - pitch            [rad]   soft
    BELLY     clearance - clear_margin          [m]     soft
    RESIDUAL  resid_tol - residual              [-]     hard  (the settle did not resolve)
    STEP      obstacle_step_m - tallest step    [m]     hard  (only with the step gate on)

Each soft margin carries its own sigma, propagated from the elevation belief's per-cell
MEASUREMENT sd through the settle's closed-form rows. With wheels at (0, +b), (0, -b), (-l, 0),
`roll = (e1 - e2)/2b` and `pitch = (e3 - (e1+e2)/2)/l` are DIFFERENCES of supports, so the pose
drift shared by every cell cancels exactly and only the measurement sd enters -- which is why the
input is the belief's `meas_sd` and not its `sigma`. Drift that accrued between two contacts does
not cancel; the footprint's drift SPREAD (`terrain_value_field.drift`) is charged to each variance.

`sigma_floor_m` is the irreducible map error. It is applied per CELL, before the propagation, and
each constraint's floor is that floor propagated the same way (`_support_sigmas` computes both), so
`terrain_value_field`'s `max(sigma, floor)` never changes a sigma this produces. Without the floor a
perfectly known map makes a pose at 14.9 deg of roll against a 15 deg limit read as infinitely safe.

The hard thresholds the robot has always run are the `z_veto = 0` case of this: at k = 0 the veto
`margin / sigma < 0` is exactly `margin < 0`, whatever the sigma.

Beside the constraints, three per-pose fields for the robust tube, which treats causes apart:
`hazard` (the settle did not resolve, or the step gate fired -- eroded HARD), `violation` [rad]
(how far a soft test fails; the tube CHARGES it) and the flatness `tilt` (the graded cost of a
feasible pose: roll_cost_weight*|roll| + pitch_cost_weight*|pitch|).

Two approximations, both marked for upgrade:
  - sigma is sampled at each WHEEL CENTRE rather than at the cell that won the envelope
    dilation. Elevation sigma varies smoothly with observation range (~3 cm/m measured), so over
    the <=0.35 m to the contact cell this is worth ~1 cm; the terrain max it stands in for is not
    smooth at all, but sigma is.
  - the footprint maximum is not folded. Reading sigma off one cell is the linearized, one-hot
    estimate, which overstates the sd at contested contacts; the Clark fold at the dilation stage
    is the fix, and it needs the envelope's own contact indices.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..engine import ForwardSimulator
from ..engine import GridParams
from ..engine import RobotParams
from ..engine import SolverParams
from ..engine.robot import Robot  # the built struct, passed straight into the kernels
from ..grid import Grid
from ..grid import sample_field
from ..heightmap import Heightmap
from .terrain_value_field.drift import footprint_drift_spread
from .terrain_value_field.field import Constraints

ROLL, CLIMB, DESCEND, BELLY, RESIDUAL, STEP = 0, 1, 2, 3, 4, 5


@wp.func
def _support_sigmas(
    s1: wp.float32,
    s2: wp.float32,
    s3: wp.float32,
    sb: wp.float32,
    sp: wp.float32,
    two_b: wp.float32,
    rl: wp.float32,
) -> wp.vec3:
    """(sigma_roll, sigma_pitch, sigma_clear) from the sd under each wheel and midway back.

    One function for the per-pose sigmas AND the floors, so that at the floor they are the same
    number to the bit and `max(sigma, floor)` leaves every sigma alone.
    """
    var_roll = (s1 * s1 + s2 * s2 + sp) / (two_b * two_b)
    var_pitch = (s3 * s3 + 0.25 * (s1 * s1 + s2 * s2) + sp) / (rl * rl)
    # The belly sits on a weighted mean of the three supports (weights summing to one), so its
    # own height carries about a third of their variance. The cross term against the ground
    # beneath it is dropped, which OVERSTATES sigma_clear -- the conservative direction.
    var_clear = sb * sb + (s1 * s1 + s2 * s2 + s3 * s3) / 9.0 + sp
    return wp.vec3(
        wp.sqrt(wp.max(var_roll, 1.0e-12)),
        wp.sqrt(wp.max(var_pitch, 1.0e-12)),
        wp.sqrt(wp.max(var_clear, 1.0e-12)),
    )


@wp.func
def _sigma_at(
    sigma: wp.array2d(dtype=wp.float32),
    grid: Grid,
    x: wp.float32,
    y: wp.float32,
    floor_m: wp.float32,
    scale: wp.float32,
) -> wp.float32:
    """Elevation sd at a world point, scaled then floored.

    `scale = 0` collapses every cell to the floor, which is the optimistic reading: what the
    robot would believe if the map carried no uncertainty beyond the irreducible.
    """
    return wp.max(scale * sample_field(sigma, grid, x, y), floor_m)


@wp.kernel
def _floor_kernel(
    robot: Robot,
    floor_m: wp.float32,
    floor: wp.array(dtype=wp.float32),  # [constraint]
):
    """Each soft constraint's floor: the per-cell floor propagated like any sigma. Hard ones: 0."""
    two_b = 2.0 * robot.wheel_pos[0][1]
    rl = -robot.wheel_pos[2][0]
    f = _support_sigmas(floor_m, floor_m, floor_m, floor_m, 0.0, two_b, rl)
    floor[ROLL] = f[0]
    floor[CLIMB] = f[1]
    floor[DESCEND] = f[1]
    floor[BELLY] = f[2]
    for i in range(RESIDUAL, floor.shape[0]):
        floor[i] = 0.0


@wp.kernel
def _local_step_kernel(
    elev: wp.array2d(dtype=wp.float32),
    measured: wp.array2d(dtype=wp.float32),  # 1 = cell has real data, 0 = never observed
    step: wp.array2d(dtype=wp.float32),
):
    """Per-cell prominence: how much a cell rises above its immediate (3x3) neighbourhood -- a STEP.
    A thin pole rises ~its full height above the adjacent ground (large step); a drivable slope rises
    only cell_size*tan(theta) per cell (small step). This lets the gate catch vertical obstacles the
    settle STRADDLES (a stick that fits between the wheel/belly contacts) without blocking slopes.

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
def _settle_constraints_kernel(
    derived: wp.array2d(dtype=wp.vec3f),  # (z, pitch, roll) per pose; row 0 = the static settle
    residual: wp.array2d(dtype=wp.float32),
    clearance: wp.array2d(dtype=wp.float32),
    sigma: wp.array2d(dtype=wp.float32),  # per-cell MEASUREMENT sd of the elevation belief [m]
    spread: wp.array2d(dtype=wp.float32),  # per-cell footprint DRIFT spread [m^2]
    step: wp.array2d(dtype=wp.float32),  # per-cell prominence [m], read only with the gate on
    grid: Grid,
    robot: Robot,
    sigma_floor_m: wp.float32,
    sigma_scale: wp.array(dtype=wp.float32),  # 1 = believe the map, 0 = optimistic (floor only)
    foot_r: wp.int32,  # step-gate footprint radius [cells]
    step_gate: wp.float32,  # [m]
    margin: wp.array4d(dtype=wp.float32),  # [constraint, row, col, heading]
    sigma_out: wp.array4d(dtype=wp.float32),  # [constraint, row, col, heading]
    hazard: wp.array3d(dtype=wp.float32),  # [row, col, heading]
    violation: wp.array3d(dtype=wp.float32),  # [row, col, heading] [rad]
    tilt: wp.array3d(dtype=wp.float32),  # [row, col, heading]
):
    """One thread per pose (row, col, heading); pose b = (r*nx + c)*n_theta + t is the C-order
    flatten of the settle's start poses.

    `violation` [rad] is how badly a SOFT test fails, 0 when none does: the attitude's excess past
    the envelope, or the belly's shortfall below clear_margin divided by the rear offset -- roughly
    the pitch that would lift it clear, so both are angles.
    """
    r, c, t = wp.tid()
    nx = hazard.shape[1]
    n_theta = hazard.shape[2]
    b = (r * nx + c) * n_theta + t
    der = derived[0, b]
    pitch = der[1]
    roll = der[2]
    clear = clearance[0, b]

    margin[ROLL, r, c, t] = robot.max_roll - wp.abs(roll)
    margin[CLIMB, r, c, t] = robot.max_pitch_up + pitch
    margin[DESCEND, r, c, t] = robot.max_pitch_down - pitch
    margin[BELLY, r, c, t] = clear - robot.clear_margin
    margin[RESIDUAL, r, c, t] = robot.resid_tol - residual[0, b]
    sigma_out[RESIDUAL, r, c, t] = 0.0

    unresolved = residual[0, b] > robot.resid_tol
    hazard[r, c, t] = wp.where(unresolved, 1.0, 0.0)
    belly_short = robot.clear_margin - clear
    tilt_over = wp.max(
        wp.max(0.0, wp.abs(roll) - robot.max_roll),
        wp.max(-pitch - robot.max_pitch_up, pitch - robot.max_pitch_down),
    )
    violation[r, c, t] = wp.max(tilt_over, belly_short / (-robot.wheel_pos[2][0]))
    tilt[r, c, t] = robot.roll_cost_weight * wp.abs(roll) + robot.pitch_cost_weight * wp.abs(pitch)

    # --- sigmas: the measurement sd under each support, propagated through the settle's rows
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
    scale = sigma_scale[0]
    s1 = _sigma_at(
        sigma, grid, x + ca * w0[0] - sa * w0[1], y + sa * w0[0] + ca * w0[1], sigma_floor_m, scale
    )
    s2 = _sigma_at(
        sigma, grid, x + ca * w1[0] - sa * w1[1], y + sa * w1[0] + ca * w1[1], sigma_floor_m, scale
    )
    s3 = _sigma_at(
        sigma, grid, x + ca * w2[0] - sa * w2[1], y + sa * w2[0] + ca * w2[1], sigma_floor_m, scale
    )
    sb = _sigma_at(sigma, grid, x - ca * rl * 0.5, y - sa * rl * 0.5, sigma_floor_m, scale)
    # Pose drift is ONE shared random walk, so it cancels between two contacts and only the part
    # accrued since the older of them was last seen survives: Var(h_A - h_B) picks up
    # |drift_A - drift_B|, which the footprint's max-min spread bounds. Charged to each variance
    # directly rather than folded into s1..s3, because each difference here has its own lever arm
    # and inflating the sds would land the wrong coefficient on pitch. `scale` gates it with the
    # measurement term: the optimistic reading assumes a map with no drift either.
    sp = scale * sample_field(spread, grid, x, y)
    s = _support_sigmas(s1, s2, s3, sb, sp, 2.0 * hb, rl)
    sigma_out[ROLL, r, c, t] = s[0]
    sigma_out[CLIMB, r, c, t] = s[1]
    sigma_out[DESCEND, r, c, t] = s[1]
    sigma_out[BELLY, r, c, t] = s[2]

    # --- the step gate: a tall step anywhere within foot_r cells, at every heading. The robot
    # cannot be centred that close to a pole in ANY orientation, though the settle straddles it.
    if margin.shape[0] > STEP:
        ny = step.shape[0]
        hit = float(0.0)
        for i in range(-foot_r, foot_r + 1):
            rr = wp.clamp(r + i, 0, ny - 1)
            for j in range(-foot_r, foot_r + 1):
                cc = wp.clamp(c + j, 0, nx - 1)
                hit = wp.max(hit, step[rr, cc])
        margin[STEP, r, c, t] = step_gate - hit
        sigma_out[STEP, r, c, t] = 0.0
        if hit > step_gate:
            hazard[r, c, t] = 1.0


class SettleProducer:
    """The settle at every pose of a window, as constraints. Owns its buffers; records cleanly
    into a caller's CUDA graph (no host syncs, no host-to-device copies in `run`)."""

    def __init__(
        self,
        grid_params: GridParams,
        robot_params: RobotParams,
        solver_params: SolverParams,
        n_theta: int,
        sigma_floor_m: float,
        obstacle_step_m: float = 0.0,  # 0 = no step gate
        device: wp.Device | str | None = None,
    ) -> None:
        self.device = wp.get_device(device)
        self.robot = robot_params.build(self.device)
        self.grid = grid_params.build()
        self.n_theta = int(n_theta)
        self.sigma_floor_m = float(sigma_floor_m)
        self.step_gate = float(obstacle_step_m)
        self.foot_r = max(1, int(round(robot_params.half_track / self.grid.cell_size)))
        ny, nx = self.grid.cells_y, self.grid.cells_x
        shape = (ny, nx, self.n_theta)

        self.settle_sim = ForwardSimulator(
            robot_params=robot_params,
            solver_params=solver_params,
            grid_params=grid_params,
            batch_size=nx * ny * self.n_theta,
            n_steps=1,
            device=self.device,
        )
        rr, cc, tt = np.meshgrid(
            np.arange(ny), np.arange(nx), np.arange(self.n_theta), indexing="ij"
        )
        px = (self.grid.origin_x + cc * self.grid.cell_size).ravel().astype(np.float32)
        py = (self.grid.origin_y + rr * self.grid.cell_size).ravel().astype(np.float32)
        # Heading bin `it` means exactly it*dth -- the solver's convention. It used to be the bin
        # MIDPOINT, which tilts every primitive half a bin off the grid and costs the left/right
        # symmetry of the fan; the settle poses have to move with it or feasibility would be
        # produced at one set of headings and consumed at another.
        ph = (tt * 2.0 * np.pi / self.n_theta).ravel().astype(np.float32)
        self.settle_sim.start_pose.assign(np.stack([px, py, ph], 1))
        self.settle_sim.target_wheel_omega.zero_()
        # the static (zero-control) settle is friction-independent; any value does
        self.settle_sim.set_friction(
            Heightmap(
                np.full((ny, nx), 0.8, np.float32),
                (self.grid.origin_x, self.grid.origin_y),
                self.grid.cell_size,
            )
        )

        n_constraints = STEP + 1 if self.step_gate > 0.0 else RESIDUAL + 1
        with wp.ScopedDevice(self.device):
            self.margin = wp.zeros((n_constraints, *shape), dtype=wp.float32)
            self.sigma = wp.zeros((n_constraints, *shape), dtype=wp.float32)
            self.floor = wp.zeros(n_constraints, dtype=wp.float32)
            self.hazard = wp.zeros(shape, dtype=wp.float32)
            self.violation = wp.zeros(shape, dtype=wp.float32)
            self.tilt = wp.zeros(shape, dtype=wp.float32)
            self.spread = wp.zeros((ny, nx), dtype=wp.float32)
            self.step = wp.zeros((ny, nx), dtype=wp.float32)  # per-cell prominence
        wp.launch(
            _floor_kernel,
            dim=1,
            inputs=[self.robot, self.sigma_floor_m],
            outputs=[self.floor],
            device=self.device,
        )
        self.constraints = Constraints(margin=self.margin, sigma=self.sigma, floor=self.floor)
        # The settle differences heights under the three wheels and midway back, so the spread has
        # to cover every contact: the furthest of them from the pose centre.
        self._drift_r = max(
            1,
            int(
                round(max(robot_params.rear_offset, robot_params.half_track) / self.grid.cell_size)
            ),
        )

    def settle(self, elevation: wp.array) -> None:
        """Place the robot at every pose. `elevation` must be a stable buffer under a graph."""
        self.settle_sim.set_terrain(elevation)  # D2D copy + envelope rebuild
        self.settle_sim.rollout_launch()

    def run(
        self,
        elevation: wp.array,
        measured: wp.array,
        sigma: wp.array,
        drift: wp.array,
        sigma_scale: wp.array,
    ) -> Constraints:
        """The settled poses -> `constraints`, `hazard`, `violation`, `tilt`. Call `settle` first.

        `sigma` is the per-cell measurement sd (zeros = the floor everywhere), `drift` the belief's
        pose-drift variance (zeros = one age everywhere, no spread), `measured` the 0/1 mask the
        step gate reads, `sigma_scale` a device scalar (1 believe the map, 0 the optimistic read).
        """
        footprint_drift_spread(drift, self._drift_r, out=self.spread)
        if self.step_gate > 0.0:
            wp.launch(
                _local_step_kernel,
                dim=(self.grid.cells_y, self.grid.cells_x),
                inputs=[elevation, measured],
                outputs=[self.step],
                device=self.device,
            )
        sim = self.settle_sim
        wp.launch(
            _settle_constraints_kernel,
            dim=self.hazard.shape,
            inputs=[
                sim.derived,
                sim.residual,
                sim.clearance,
                sigma,
                self.spread,
                self.step,
                self.grid,
                self.robot,
                self.sigma_floor_m,
                sigma_scale,
                self.foot_r,
                self.step_gate,
            ],
            outputs=[self.margin, self.sigma, self.hazard, self.violation, self.tilt],
            device=self.device,
        )
        return self.constraints
