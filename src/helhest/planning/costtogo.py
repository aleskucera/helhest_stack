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

This is the settle-based feasibility PRODUCER, and that is now ALL it is. It settles the robot at
every pose to make the per-pose blocked / graded-tilt fields, packs them into the one signed field
`terrain_value_field` reads, and hands the value iteration to it.

The split is deliberate. What is here is Odin's physics -- the settle, the tall-step gate, the
robust-tube erosion -- and it is worth nothing to another robot. What moved out is the cost-to-go
machinery, which is worth the same to every robot and was previously a copy that had drifted: it
charged a flat step for arcs that covered different ground, took heading bins at their midpoints
so no move ran along a grid axis, and checked every swept cell at the heading the arc STARTED in
even where the arc had turned 45 degrees by the time it got there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import math

import numpy as np
import warp as wp

from ..engine import ForwardSimulator
from ..engine.robot import Robot  # the built struct, passed straight into the feasibility kernel
from ..engine.terrain import Grid
from ..engine.terrain import sample_field
from ..heightmap import Heightmap
from ..profiling import StageProfiler
from terrain_value_field import closing_step
from terrain_value_field.drift import footprint_drift_spread
from terrain_value_field.solver import ValueSolver

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
    hazard: wp.array3d(dtype=wp.float32),
    violation: wp.array3d(dtype=wp.float32),
    tilt: wp.array3d(dtype=wp.float32),
):
    """The per-pose feasibility OR + graded tilt cost, one thread per (y, x, theta), no readback.
    Direction-aware: climb = nose-up = NEGATIVE pitch, so the climb limit is on -pitch, descend on
    +pitch. Pose b = (r*nx + c)*n_theta + t is the C-order flatten matching start_pose in __init__.

    `blocked` is the union; the causes are also kept apart because the robust tube treats them
    differently. `hazard` here is only the settle failing to resolve (residual); walls join it in
    `_step_hazard_kernel`. `violation` [rad] is how badly a SOFT test fails, 0 when none does:
    the attitude's excess past the envelope, or the belly's shortfall below clear_margin divided
    by the rear offset -- roughly the pitch that would lift it clear, so both are angles.
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
    unresolved = residual[0, b] > robot.resid_tol
    belly_short = robot.clear_margin - clearance[0, b]
    if over_envelope or unresolved or belly_short > 0.0:
        blocked[r, c, t] = 1.0
    else:
        blocked[r, c, t] = 0.0
    hazard[r, c, t] = wp.where(unresolved, 1.0, 0.0)
    tilt_over = wp.max(
        wp.max(0.0, wp.abs(roll) - robot.max_roll),
        wp.max(-pitch - robot.max_pitch_up, pitch - robot.max_pitch_down),
    )
    violation[r, c, t] = wp.max(tilt_over, belly_short / (-robot.wheel_pos[2][0]))
    tilt[r, c, t] = robot.roll_cost_weight * wp.abs(roll) + robot.pitch_cost_weight * wp.abs(pitch)


@wp.kernel
def _margin_kernel(
    derived: wp.array2d(dtype=wp.vec3f),  # (z, pitch, roll) per pose; row 0 = the static settle
    clearance: wp.array2d(dtype=wp.float32),
    sigma: wp.array2d(dtype=wp.float32),  # per-cell MEASUREMENT sd of the elevation belief [m]
    spread: wp.array2d(dtype=wp.float32),  # per-cell footprint DRIFT spread [m^2]
    grid: Grid,
    robot: Robot,
    n_theta: wp.int32,
    sigma_floor_m: wp.float32,
    z_veto: wp.array(dtype=wp.float32),  # device scalar: settable between CUDA-graph replays
    sigma_scale: wp.array(dtype=wp.float32),  # 1 = believe the map, 0 = optimistic (floor only)
    z_charge: wp.float32,
    charge_per_sigma: wp.float32,
    blocked: wp.array3d(dtype=wp.float32),
    tilt: wp.array3d(dtype=wp.float32),
    zmargin: wp.array3d(dtype=wp.float32),
    doubt: wp.array3d(dtype=wp.float32),
):
    """Safety margin in SIGMAS, and the veto and graded penalty that come off it.

    Each feasibility test is asked how much room is left in units of its OWN uncertainty:

        z_roll  = (max_roll - |roll|)           / sigma_roll
        z_climb = (max_pitch_up + pitch)        / sigma_pitch     (climb = NEGATIVE pitch)
        z_desc  = (max_pitch_down - pitch)      / sigma_pitch
        z_clear = (clearance - clear_margin)    / sigma_clear
        z       = min over tests                                  -- the binding constraint

    Dividing each by its own sigma is what makes the `min` meaningful: roll is in radians and
    clearance in metres, and a raw `min` over those compares nothing. `blocked = z < z_veto`
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

    two_b = 2.0 * hb
    var_roll = (s1 * s1 + s2 * s2 + sp) / (two_b * two_b)
    var_pitch = (s3 * s3 + 0.25 * (s1 * s1 + s2 * s2) + sp) / (rl * rl)
    # The belly sits on a weighted mean of the three supports (weights summing to one), so its
    # own height carries about a third of their variance. The cross term against the ground
    # beneath it is dropped, which OVERSTATES sigma_clear -- the conservative direction.
    var_clear = sb * sb + (s1 * s1 + s2 * s2 + s3 * s3) / 9.0 + sp

    sigma_roll = wp.sqrt(wp.max(var_roll, 1.0e-12))
    sigma_pitch = wp.sqrt(wp.max(var_pitch, 1.0e-12))
    sigma_clear = wp.sqrt(wp.max(var_clear, 1.0e-12))

    z_roll = (robot.max_roll - wp.abs(roll)) / sigma_roll
    z_climb = (robot.max_pitch_up + pitch) / sigma_pitch
    z_desc = (robot.max_pitch_down - pitch) / sigma_pitch
    z_clear = (clearance[0, b] - robot.clear_margin) / sigma_clear

    z = wp.min(wp.min(z_roll, z_climb), wp.min(z_desc, z_clear))
    zmargin[r, c, t] = z

    # The same pose scored as if the map were CERTAIN -- every cell at the irreducible floor.
    # The attitude sigmas are linear in the per-cell sd, so the optimistic ones follow from the
    # same rows with s1 = s2 = s3 = sb = floor. They are NOT simply the floor: a pose's roll
    # uncertainty is the floor propagated through the track width, not the floor itself.
    k = z_veto[0]
    f = sigma_floor_m
    o_roll = wp.sqrt(2.0 * f * f) / two_b
    o_pitch = wp.sqrt(f * f + 0.5 * f * f) / rl
    o_clear = wp.sqrt(f * f + f * f / 3.0)
    z_opt = wp.min(
        wp.min(
            (robot.max_roll - wp.abs(roll)) / o_roll,
            (robot.max_pitch_up + pitch) / o_pitch,
        ),
        wp.min(
            (robot.max_pitch_down - pitch) / o_pitch,
            (clearance[0, b] - robot.clear_margin) / o_clear,
        ),
    )

    # Doubt: blocked by IGNORANCE, not by terrain. A pose the robot would accept on a certain
    # map and refuses on this one is worth going to look at; a pose that fails either way is
    # simply bad ground and looking at it will not help. This is the distinction that separates
    # purposeful exploration from wandering toward whatever is least observed.
    if z < k and z_opt >= k:
        doubt[r, c, t] = z_opt - z
    else:
        doubt[r, c, t] = 0.0

    if z < k:
        blocked[r, c, t] = 1.0
    if z < z_charge:  # graded: pay for being near a boundary, not only for crossing it
        tilt[r, c, t] = tilt[r, c, t] + charge_per_sigma * (z_charge - z)


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
def _face_kernel(
    elev: wp.array2d(dtype=wp.float32),
    measured: wp.array2d(dtype=wp.float32),  # 1 = cell has real data, 0 = never observed
    face: wp.array2d(dtype=wp.float32),
):
    """Per-cell FACE height: the largest rise to a 4-neighbour, one cell away.

    Not the 3x3 prominence above: that is taken across the diagonal too, so a steep but smooth
    flank reads as a step -- `bumpy`'s mounds reach 0.44 m on it against 0.50 m at a wall. One
    cell apart, a slope can only rise cell * tan(slope) while a wall rises its full height:
    measured 0.29 m at most on the mounds, 0.50 m at least at the walls (0.2 m cells). Unobserved
    cells are skipped for the same reason as in the prominence."""
    r, c = wp.tid()
    ny = elev.shape[0]
    nx = elev.shape[1]
    top = float(0.0)
    if measured[r, c] > 0.5:
        for k in range(4):
            rr = r
            cc = c
            if k == 0:
                rr = r - 1
            elif k == 1:
                rr = r + 1
            elif k == 2:
                cc = c - 1
            else:
                cc = c + 1
            rr = wp.clamp(rr, 0, ny - 1)
            cc = wp.clamp(cc, 0, nx - 1)
            if measured[rr, cc] > 0.5:
                top = wp.max(top, elev[r, c] - elev[rr, cc])
    face[r, c] = top


@wp.kernel
def _step_gate_kernel(
    step: wp.array2d(dtype=wp.float32),
    foot_r: int,
    step_gate: wp.float32,
    blocked: wp.array3d(dtype=wp.float32),
    hazard: wp.array3d(dtype=wp.float32),
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
        hazard[r, c, t] = 1.0


@wp.kernel
def _step_hazard_kernel(
    face: wp.array2d(dtype=wp.float32),  # per-cell face height, from _face_kernel
    grid: Grid,
    robot: Robot,
    blocked: wp.array3d(dtype=wp.float32),
    hazard: wp.array3d(dtype=wp.float32),
):
    """Re-file a SOFT block (tilt, belly) as a hazard when there is a face taller than the wheel
    radius under the footprint.

    The settle has no notion of a vertical face: a wheel against a 1 m wall is lifted onto its
    edge and reads as a tilt, and a wall under the body reads as a belly short of clearance.
    Measured on the stress worlds, 11-15% of the poses whose body overlaps a wall are blocked by
    tilt ALONE, so charging tilt instead of eroding it would have let the tube run within
    millimetres of every wall. A step the wheel cannot mount is a collision whatever the settle
    made of it. A smooth flank or a mound under the belly stays soft: that is what the face
    measure is for. Only already-blocked poses are re-filed, so this never blocks a pose that was
    free.
    """
    r, c, t = wp.tid()
    if blocked[r, c, t] < 0.5 or hazard[r, c, t] > 0.5:
        return
    n_theta = blocked.shape[2]
    ny = face.shape[0]
    nx = face.shape[1]
    x = grid.origin_x + float(c) * grid.cell_size
    y = grid.origin_y + float(r) * grid.cell_size
    yaw = float(t) * 2.0 * 3.14159265 / float(n_theta)  # the settle's heading convention
    ca = wp.cos(yaw)
    sa = wp.sin(yaw)
    # the bare body: rear wheel's back to front wheel's front, outer tread edge to outer tread edge
    x_lo = robot.wheel_pos[2][0] - robot.wheel_radius
    x_hi = robot.wheel_radius
    y_hi = robot.half_track + robot.wheel_half_width
    h = 0.5 * grid.cell_size
    nu = int(wp.ceil((x_hi - x_lo) / h))
    nv = int(wp.ceil(2.0 * y_hi / h))
    tallest = float(0.0)
    for i in range(nu + 1):
        u = x_lo + float(i) * (x_hi - x_lo) / float(nu)
        for j in range(nv + 1):
            v = -y_hi + float(j) * 2.0 * y_hi / float(nv)
            cc = wp.clamp(
                int(wp.round((x + ca * u - sa * v - grid.origin_x) / grid.cell_size)), 0, nx - 1
            )
            rr = wp.clamp(
                int(wp.round((y + sa * u + ca * v - grid.origin_y) / grid.cell_size)), 0, ny - 1
            )
            tallest = wp.max(tallest, face[rr, cc])
    if tallest > robot.wheel_radius:
        hazard[r, c, t] = 1.0


@wp.kernel
def _robust_kernel(
    blocked: wp.array3d(dtype=wp.float32),
    hazard: wp.array3d(dtype=wp.float32),
    violation: wp.array3d(dtype=wp.float32),
    graded_tilt: wp.array3d(dtype=wp.float32),
    dr: int,
    dc: int,
    dt: int,
    soft_weight: wp.float32,
    robust: wp.array3d(dtype=wp.float32),
    robust_tilt: wp.array3d(dtype=wp.float32),
):
    """Robust feasibility over the (y, x, theta) disturbance tube the closed loop cannot correct
    before the next replan -- split by what the tube is protecting against.

    HAZARDS -- a wall (a face the wheel cannot mount) or a settle that does not resolve -- are
    eroded HARD: a pose is blocked if any pose in the tube is. Driving into a wall is never a price
    worth paying, so no cost can buy it back.

    SOFT blocks -- tilt past the envelope, the belly short of clearance on smooth ground -- are
    vetoed at the pose itself, and the tube CHARGES for them instead: soft_weight times the worst
    violation [rad] anywhere in the tube, added to the graded tilt. Rough ground speckles both,
    and a hard erosion by the 27-pose box took `bumpy` from 6/6 reached to 2/6; on its own the
    erosion of belly-on-mound poses, 2.2% of the window, took it to 3/6. A charge proportional to
    HOW FAR over keeps a steep slope beside the route expensive and a marginal speckle nearly free.
    dr=dc=dt=0 is plain feasibility.
    """
    r, c, t = wp.tid()
    ny = blocked.shape[0]
    nx = blocked.shape[1]
    nth = blocked.shape[2]
    near_wall = float(0.0)
    excess = float(0.0)
    for i in range(-dr, dr + 1):
        rr = wp.clamp(r + i, 0, ny - 1)
        for j in range(-dc, dc + 1):
            cc = wp.clamp(c + j, 0, nx - 1)
            for k in range(-dt, dt + 1):
                tt = (t + k + nth) % nth  # heading wraps
                near_wall = wp.max(near_wall, hazard[rr, cc, tt])
                excess = wp.max(excess, violation[rr, cc, tt])
    robust[r, c, t] = wp.max(blocked[r, c, t], near_wall)
    robust_tilt[r, c, t] = graded_tilt[r, c, t] + soft_weight * excess


@wp.kernel
def _solid_kernel(
    hazard: wp.array3d(dtype=wp.float32),
    solid: wp.array2d(dtype=wp.float32),
):
    """A cell no pose can stand on at ANY heading: inside a wall, as far as the robot can tell."""
    r, c = wp.tid()
    s = float(1.0)
    for t in range(hazard.shape[2]):
        s = wp.min(s, hazard[r, c, t])
    solid[r, c] = s


@wp.func
def _clear_line(solid: wp.array2d(dtype=wp.float32), r0: int, c0: int, r1: int, c1: int) -> bool:
    """No solid cell between two cells, sampled every half cell along the straight line."""
    n = 2 * wp.max(wp.abs(r1 - r0), wp.abs(c1 - c0))
    for s in range(1, n + 1):
        f = float(s) / float(n)
        rr = int(wp.round(float(r0) + f * float(r1 - r0)))
        cc = int(wp.round(float(c0) + f * float(c1 - c0)))
        if solid[rr, cc] > 0.5:
            return False
    return True


@wp.kernel
def _escape_kernel(
    V: wp.array3d(dtype=wp.float32),  # the clamped cost-to-go
    solid: wp.array2d(dtype=wp.float32),
    vcap: wp.float32,
    reach: int,  # [cells] how far a no-route pose looks for a routable one
    cell: wp.float32,
    per_m: wp.float32,  # [m-equiv per m] price of getting there
    per_bin: wp.float32,  # [m-equiv per heading bin] price of turning to it
    V_escape: wp.array3d(dtype=wp.float32),
):
    """V, with every no-route pose given the cheapest way back to a routable one nearby.

    MPPI reads V at each rollout pose, and where V is capped it had nothing to follow but a straight
    line to the goal. That line is exactly wrong in the two failures it caused: pressed against a
    wall with the goal on the far side (`false_door`: 93% of the stuck frames had no route at the
    robot's own pose), and half-way through turning round in a corridor, where the sideways
    headings have no route and the line points back into the dead end. Here a capped pose instead
    costs the routable pose within `reach` that is cheapest to get to -- its V, plus `per_m` per
    metre and `per_bin` per heading bin to reach it -- so the gradient leads out, not into the wall.
    Poses with a route are untouched, and a pose with nothing routable within `reach` keeps the cap,
    so the straight-line exploration fallback still arms where the goal genuinely has no route.

    Only along a CLEAR line: a routable pose on the far side of a wall is not a way out. Without
    this check a pose inside `corridor`, near its dead end, escaped through the 0.4 m corridor
    wall to the open ground outside, and the pull went straight into the wall.
    """
    r, c, t = wp.tid()
    v = V[r, c, t]
    lim = 0.9 * vcap  # the same "no route" test MPPI applies
    if v < lim:
        V_escape[r, c, t] = v
        return
    ny = V.shape[0]
    nx = V.shape[1]
    nth = V.shape[2]
    best = vcap
    for i in range(-reach, reach + 1):
        rr = r + i
        if rr < 0 or rr >= ny:
            continue
        for j in range(-reach, reach + 1):
            cc = c + j
            if cc < 0 or cc >= nx:
                continue
            d2 = float(i * i + j * j)
            if d2 > float(reach * reach):
                continue
            if not _clear_line(solid, r, c, rr, cc):
                continue
            move = per_m * cell * wp.sqrt(d2)
            for k in range(nth):
                vn = V[rr, cc, k]
                if vn < lim:
                    dk = wp.abs(k - t)
                    turn = float(wp.min(dk, nth - dk))
                    best = wp.min(best, vn + move + per_bin * turn)
    V_escape[r, c, t] = wp.min(best, vcap)


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


@wp.kernel
def _pose_cost_kernel(
    blocked: wp.array3d(dtype=wp.float32),  # [row, col, heading]
    graded_tilt: wp.array3d(dtype=wp.float32),  # [row, col, heading]
    pose_cost: wp.array3d(dtype=wp.float32),  # [row, col, heading]
):
    """Pack this robot's feasibility into the one signed field the solver reads.

    The veto rides in the sign and the graded cost in the magnitude (terrain_value_field.margin,
    POSE COST), which halves the loads in the relax kernel's inner loop. The clamp at zero is not
    defensive noise: a negative penalty would read as a veto and quietly make a passable pose
    impassable, so the encoding's one precondition is enforced where it is produced.
    """
    r, c, t = wp.tid()
    pen = wp.max(graded_tilt[r, c, t], 0.0)
    pose_cost[r, c, t] = wp.where(blocked[r, c, t] > 0.5, -1.0 - pen, pen)


@wp.kernel
def _seed_goal_kernel(
    goal_rc: wp.array(dtype=wp.int32),  # [2]
    inf: wp.float32,
    seeds: wp.array3d(dtype=wp.float32),  # [row, col, heading]
):
    """Seed the goal cell at every heading, on DEVICE.

    The goal cell is resolved inside the captured graph (`_goal_cell_kernel`), so the seeding has
    to be too -- reading it back to call a host-side seeder would put a sync in the middle of the
    graph and defeat the point of capturing it.
    """
    r, c, t = wp.tid()
    seeds[r, c, t] = wp.where(r == goal_rc[0] and c == goal_rc[1], 0.0, inf)


@wp.kernel
def _seed_goal_and_boundary_kernel(
    goal_xy: wp.array(dtype=wp.float32),  # [2], this window's frame -- UNCLAMPED on purpose
    coarse_value: wp.array3d(dtype=wp.float32),  # [cy, cx, 1], the coarse layer's cost-to-go
    coarse_origin: wp.array(dtype=wp.float32),  # [2], the coarse grid in THIS window's frame
    coarse_cell: wp.float32,
    origin_x: wp.float32,  # this window's own origin, same frame
    origin_y: wp.float32,
    cell_size: wp.float32,
    band: wp.int32,  # ring thickness, in fine cells
    inf: wp.float32,
    seeds: wp.array3d(dtype=wp.float32),  # [row, col, heading]
):
    """Seed the goal AND the window's border, in one pass because each writes every cell.

    A border cell is seeded at what the coarse layer says it costs to carry on from there, so the
    fine solve pays the real price of each exit and prefers the right one instead of treating
    every way out of the window as equally good. See `terrain_value_field.hierarchical`; this is
    that kernel fused with the goal seeding, which the fine layer does on device because the goal
    is resolved inside the captured graph.

    The goal is taken UNCLAMPED here, unlike the single-layer path. Clamping an out-of-window goal
    onto the border is what makes one layer work at all -- it becomes a carrot the window drags
    along -- but with a coarse layer it is actively wrong: a zero-cost seed on the border beats
    every finite coarse value, so the ring is overridden and the robot chases the exit nearest the
    goal even when the coarse layer knows that exit is a dead end. When the goal is outside, the
    ring IS the goal information and nothing else should be seeded.

    The coarse value is read from the NEAREST coarse cell, never interpolated: unreachable cells
    hold +inf, and blending that with a finite neighbour yields a large finite number -- a cell
    that reads as reachable at an invented price, which is worse than either truth.

    """
    r, c, t = wp.tid()
    rows = seeds.shape[0]
    cols = seeds.shape[1]
    gc = int((goal_xy[0] - origin_x) / cell_size)
    gr = int((goal_xy[1] - origin_y) / cell_size)
    if gr >= 0 and gr < rows and gc >= 0 and gc < cols:
        # The goal is in the window, so the window is not missing anything and the ring is not
        # seeded at all. It would not merely be redundant: the coarse layer is omnidirectional and
        # pays no turn cost, so it UNDERSTATES distance in the fine layer's own metric, and a ring
        # priced that way reads as a shortcut. The fine solve would route the robot out of the
        # window and back to reach a goal sitting a few metres in front of it. Seeding the ring
        # at a 5 m margin instead was tried: no better on false_door (2/3 either way), and it
        # took pocket from 3/3 to 2/3 with a run three times slower.
        seeds[r, c, t] = wp.where(r == gr and c == gc, 0.0, inf)
        return
    v = inf
    if r < band or r >= rows - band or c < band or c >= cols - band:
        # origin + c*cell, NOT + (c + 0.5)*cell: this file places a pose at `origin + c * cell`
        # (see `_margin_kernel`) and resolves the goal the same way, so the half cell that the
        # engine's `_locate` convention would add puts this lookup half a cell from where every
        # other kernel here thinks cell c is. It feeds a NEAREST-cell read of the coarse field
        # rather than a smooth interpolation, so at --coarsen 1 the 0.1 m offset flips the
        # rounding for about half the ring and reads a neighbour's value.
        x = origin_x + float(c) * cell_size
        y = origin_y + float(r) * cell_size
        cc = int(wp.round((x - coarse_origin[0]) / coarse_cell))
        cr = int(wp.round((y - coarse_origin[1]) / coarse_cell))
        if cr >= 0 and cr < coarse_value.shape[0] and cc >= 0 and cc < coarse_value.shape[1]:
            v = coarse_value[cr, cc, 0]
    seeds[r, c, t] = v


@wp.kernel
def _descent_kernel(
    V: wp.array3d(dtype=wp.float32),  # [rows, cols, headings], the field the controller follows
    row: wp.int32,
    col: wp.int32,
    radius: wp.int32,  # [cells]
    vcap: wp.float32,
    out: wp.array(dtype=wp.float32),  # [2]: the bearing [rad] to the cheapest cell, its value
):
    """Where the field falls away from a cell: the bearing to the cheapest routable cell within
    `radius`, best over headings, the cell itself excluded. One thread; the disc is small."""
    rows = V.shape[0]
    cols = V.shape[1]
    best = vcap
    br = float(0.0)
    bc = float(0.0)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr * dr + dc * dc > radius * radius or (dr == 0 and dc == 0):
                continue
            rr = row + dr
            cc = col + dc
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                continue
            for t in range(V.shape[2]):
                if V[rr, cc, t] < best:
                    best = V[rr, cc, t]
                    br = float(dr)
                    bc = float(dc)
    out[0] = wp.atan2(br, bc)
    out[1] = best


class CostToGo:
    # The escape field's prices. Getting back to routable ground runs through ground the router
    # refused, so it is dearer than the 1 per metre of an ordinary route -- dear enough that no
    # escape reads cheaper than a real route beside it, cheap enough to stay far under the cap.
    ESCAPE_REACH_M = 1.0  # [m] the robot's own length: far enough to leave a wall's margin
    ESCAPE_PER_M = 3.0  # [m-equiv per m]
    ESCAPE_PER_BIN = 0.3  # [m-equiv per heading bin], the deployed plan_pivot_cost

    def __init__(
        self,
        grid_params: GridParams,
        robot_params: RobotParams,
        solver_params: SolverParams,
        n_theta: int = 24,
        step: float | None = None,  # None = the longest arc that CLOSES; see below
        flatness_weight: float = 2.0,  # planner strength: how much detour to trade for flat ground
        robust_margin_m: float = 0.0,  # lateral disturbance tube -> erode the feasible set by this
        robust_margin_deg: float = 0.0,  # heading disturbance tube (orientation-aware erosion)
        # [per rad] of the worst soft violation inside the tube, as graded tilt. Tilt and belly
        # clearance are charged, walls and unresolved settles are eroded hard -- `_robust_kernel`.
        robust_soft_weight: float = 10.0,
        obstacle_step_m: float = 0.0,  # hard-block cells with a local step taller than this [m];
        # 0 = OFF. Catches thin vertical obstacles (sticks/poles) the settle straddles.
        pivot_cost: float = 0.0,  # [m-equiv] per heading bin; > 0 adds point-turn primitives so
        # a goal behind the robot routes as pivot-then-drive instead of a wide loop. 0 = OFF.
        # --- probabilistic feasibility (z-margin). z_veto = 0 is EXACTLY the old behaviour:
        # the hard thresholds still veto, nothing is added, and `compute` need not be passed a
        # sigma. Above 0 a pose must additionally hold z_veto standard deviations of room on
        # every test, measured against the elevation belief's own per-cell MEASUREMENT sd.
        z_veto: float = 0.0,
        # Irreducible MAP error, and only that. It was 0.02, which is above every fused sd in the
        # window -- 0.45 cm at 2 m, 1.16 cm at 10 m after the ~20 returns a cell gets -- so
        # `max(sigma, floor)` always took the floor and the per-cell sigma never entered the
        # answer at all. The margin looked adaptive and was a flat 4.4 deg of roll everywhere,
        # `doubt` came out identically zero, and the optimistic and pessimistic readings were the
        # same number. At 0.005 the fused sd WINS at range, so the margin is small where the
        # robot has looked and grows where it has not, which is what the design was for.
        #
        # The model error the old floor was silently carrying -- settle-vs-real tilt, off by 7 deg
        # at the extremes on `bumpy` -- now lives in RobotParams' envelope instead, where it is
        # one number against a measured table rather than three interacting ones.
        sigma_floor_m: float = 0.005,
        # Start charging for proximity to a boundary below this many sigmas. It is denominated in
        # SIGMAS, so it does not survive a change of `sigma_floor_m` unless it is rescaled with
        # it: dropping the floor 0.02 -> 0.005 shrinks sigma_roll from 2.22 deg to 0.55 and
        # inflates every z fourfold, which left the penalty band 1.1 deg wide -- a ramp too narrow
        # to steer by. 16 restores the same ANGULAR band the old pair had: the penalty starts
        # 8.8 deg below the limit and the veto bites 1.1 deg below it.
        z_charge: float = 16.0,
        # [m-equiv] per sigma of shortfall below z_charge. 0 was "veto only", which makes the
        # feasibility a CLIFF: a pose at 2.01 sigmas is free and one at 1.99 is impossible, with
        # no gradient anywhere between. That is why enforcing the veto parked the robot at 8.0 m
        # of 14 on `bumpy` -- sitting at the edge cost nothing, so nothing pushed it off. A
        # penalty cannot make anything unreachable; it only makes roomy ground cheaper than
        # marginal ground, and it is what finally gives `z_charge` something to do.
        charge_per_sigma: float = 0.5,
        profile: bool = False,  # opt-in per-stage CUDA-event timing (tiny event nodes + per-call sync)
        device: wp.Device | str | None = None,
    ) -> None:

        self.device = wp.get_device(device)
        self.flatness_weight = flatness_weight
        self.z_veto = float(z_veto)
        self.sigma_floor_m = float(sigma_floor_m)
        self.z_charge = float(z_charge)
        self.charge_per_sigma = float(charge_per_sigma)
        self.n_theta = int(n_theta)
        self.robot = robot_params.build(self.device)
        self.grid = grid_params.build()
        self.bounds = grid_params.bounds  # (xmin, xmax, ymin, ymax) the solver takes
        # robust-feasibility tube in lattice cells / theta-bins (0 -> no erosion = plain feasibility)
        self._mr = int(round(robust_margin_m / self.grid.cell_size))
        self._mc = self._mr
        self._mt = int(round(robust_margin_deg / (360.0 / n_theta)))
        self._eroded = self._mr > 0 or self._mt > 0
        self.robust_soft_weight = float(robust_soft_weight)
        self._escape_reach = max(1, int(round(self.ESCAPE_REACH_M / self.grid.cell_size)))
        # a turn costs what the lattice charges for one when it has point turns; the deployed
        # value otherwise, so an escape that is mostly turning still reads cheaper than the cap
        self._escape_per_bin = float(pivot_cost) if pivot_cost > 0.0 else self.ESCAPE_PER_BIN
        # tall-step obstacle gate: block cells within a robot footprint of a step > obstacle_step_m.
        self._step_gate = float(obstacle_step_m)
        self._foot_r = max(1, int(round(robot_params.half_track / self.grid.cell_size)))

        # A lattice arc has to end on a heading BIN or the table records a heading the robot
        # never reaches -- up to half a bin of error on every move, compounding, with feasibility
        # then evaluated at a pose the robot will not occupy. It closes when the sharpest turn,
        # step / min_turn_radius, is a whole number of bins, and an EVEN number of them, because
        # the half-rate arcs have to land on a bin too.
        #
        # Closing is necessary and NOT sufficient: the arcs turn by {0, +-bins/2, +-bins} bins, so
        # repeated they reach only the multiples of bins/2, which is every heading iff
        # gcd(bins/2, n_theta) == 1. Point turns are the only +-1 move and this planner prices
        # them out, so an escalation that lands on a bad `bins` silently splits the heading ring.
        # That shipped: bins=4 at n_theta=16 reaches only the EVEN bins, leaving the odd half at
        # the cap with `blocked` all zero -- half the states unreachable on open ground. It cost
        # `pocket` the run, because the bearing to its goal fell on an odd bin, so the value field
        # called the one heading that pointed at the goal a dead end.
        #
        # The step still has to go somewhere -- an arc that snaps back onto its own state is a
        # self-loop and nothing propagates (tests/planning/test_closure.py pins |dr| + |dc| > 0).
        # But the bound that guarantees THAT is the half-diagonal: a state sits at a cell centre,
        # and the farthest a point can be from that centre and still be in the cell is
        # cell * sqrt(2) / 2, in the diagonal directions. One whole cell is that with a margin,
        # and it is what this takes. The old bound was TWO cells -- a round number, not a derived
        # one -- and bins=2 here misses it by 2% (0.3927 against 0.40), which is the entire
        # reason the search escalated into a split ring.
        #
        # The search stops at a quarter turn: past 90 degrees in ONE primitive the arc is a large
        # committed manoeuvre to collision-check as a unit, and routing gets coarse enough that
        # the lattice stops being worth its cost.
        if step is None:
            bins, bins_max = 2, max(2, (n_theta // 4) // 2 * 2)
            min_step = self.grid.cell_size
            usable = lambda b: (
                closing_step(n_theta, robot_params.min_turn_radius, b) >= min_step
                and math.gcd(b // 2, n_theta) == 1
            )
            while not usable(bins) and bins + 2 <= bins_max:
                bins += 2
            if not usable(bins):
                # Both ways out are real decisions, not fallbacks: a coarser heading ring, or
                # admitting the point turns the skid-steer actually has.
                raise ValueError(
                    f"no connected lattice for n_theta={n_theta}, "
                    f"min_turn_radius={robot_params.min_turn_radius}, "
                    f"cell_size={self.grid.cell_size}: every closing step up to a quarter turn "
                    f"either fails to clear one cell (< {min_step:.4f} m) or splits the heading "
                    f"ring. Lower n_theta, or set pivot_cost > 0."
                )
            step = closing_step(n_theta, robot_params.min_turn_radius, bins)
        self.step = float(step)

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
        # Heading bin `it` means exactly it*dth -- the solver's convention. It used to be the bin
        # MIDPOINT, which tilts every primitive half a bin off the grid and costs the left/right
        # symmetry of the fan; the settle poses have to move with it or feasibility would be
        # produced at one set of headings and consumed at another.
        ph = (tt * 2.0 * np.pi / n_theta).ravel().astype(np.float32)
        self.settle_sim.start_pose.assign(np.stack([px, py, ph], 1))
        self.settle_sim.target_wheel_omega.zero_()
        self._mu = Heightmap(
            np.full((self.grid.cells_y, self.grid.cells_x), 0.8, np.float32),
            (self.grid.origin_x, self.grid.origin_y),
            self.grid.cell_size,
        )
        self.settle_sim.set_friction(self._mu)

        # Two-layer routing is opt-in and is armed by `set_coarse`, not by the constructor: the
        # coarse grid's SHAPE and cell size are constant, and its origin and values live in device
        # arrays the captured graph reads -- so a coarse layer anchored to the world, which the
        # window slides across, moves under the graph without invalidating it.
        self._coarse_in: wp.array | None = None
        self._coarse_origin: wp.array | None = None  # [2], in this window's frame
        self._coarse_cell = 0.0
        self._band = 0

        self.solver = ValueSolver(
            self.grid.cell_size,
            self.grid.cells_y,
            self.grid.cells_x,
            n_theta=n_theta,
            turn_radius=float(self.robot.min_turn_radius),
            step=self.step,
            # helhest spells "no point turns" as 0.0; the solver spells it as an infinite price,
            # which leaves the two primitives out of the table instead of pricing them out.
            pivot_cost=math.inf if pivot_cost <= 0.0 else float(pivot_cost),
            device=self.device,
        )

        self.V = wp.zeros(
            (self.grid.cells_y, self.grid.cells_x, n_theta),
            dtype=wp.float32,
            device=self.device,
        )
        self.blocked = wp.zeros_like(self.V)
        self.zmargin = wp.zeros_like(self.V)  # safety margin in sigmas, per pose
        self.doubt = wp.zeros_like(self.V)  # > 0 where a pose is blocked by IGNORANCE alone
        self.V_optimistic = wp.zeros_like(self.V)  # filled by solve_gap()
        self.V_pessimistic = wp.zeros_like(self.V)  # ditto; `compute` reuses self.V
        self.doubt_pessimistic = wp.zeros_like(self.V)
        self.robust_blocked = wp.zeros_like(self.V)  # blocked after the disturbance-tube erosion
        self.graded_tilt = wp.zeros_like(self.V)
        # blocked HARD: a wall, or a settle that did not resolve. Also what MPPI vetoes hard --
        # WITHOUT the tube: the tube is the router's margin, and vetoing it in the controller froze
        # the robot beside every wall (each move, even a turn in place, clipped it)
        self.hazard = wp.zeros_like(self.V)
        self.violation = wp.zeros_like(self.V)  # [rad] how badly a soft test fails, 0 if none does
        self.robust_tilt = wp.zeros_like(self.V)  # graded_tilt + the tube's tilt charge
        # V with a way back out of every no-route pose: what MPPI follows (see _escape_kernel)
        self.V_escape = wp.zeros_like(self.V)
        self._descent_out = wp.zeros(2, dtype=wp.float32, device=self.device)
        self._solid = wp.zeros((ny, nx), dtype=wp.float32, device=self.device)  # inside a wall
        # what the solver actually reads: veto in the sign, graded cost in the magnitude
        self._pose_cost = wp.zeros_like(self.V)
        self._seeds = wp.zeros_like(self.V)
        self._step = wp.zeros((ny, nx), dtype=wp.float32, device=self.device)  # per-cell prominence
        self._face = wp.zeros(
            (ny, nx), dtype=wp.float32, device=self.device
        )  # per-cell face height

        self._elev_in = wp.zeros((ny, nx), dtype=wp.float32, device=self.device)
        # Per-cell measurement sd. Defaults to zero, which the floor then lifts to
        # `sigma_floor_m` everywhere -- so an unsupplied sigma is a uniform-uncertainty map, not
        # a claim of perfect knowledge.
        self._sigma_in = wp.zeros((ny, nx), dtype=wp.float32, device=self.device)
        # Pose-drift variance per cell (`var_h - var_meas` from the belief), and the footprint
        # spread reduced from it. Zero everywhere is a map of one age, which contributes nothing
        # -- so a caller that passes no drift gets exactly the no-drift behaviour.
        self._drift_in = wp.zeros((ny, nx), dtype=wp.float32, device=self.device)
        self._spread = wp.zeros((ny, nx), dtype=wp.float32, device=self.device)
        # The settle differences heights under the three wheels and midway back, so the spread has
        # to cover every contact: the furthest of them from the pose centre.
        self._drift_r = max(
            1,
            int(
                round(max(robot_params.rear_offset, robot_params.half_track) / self.grid.cell_size)
            ),
        )
        # Stable mask buffer the captured graph reads. Defaults to all-measured, so a caller that
        # passes no mask gets exactly the pre-mask behaviour.
        self._measured_in = wp.full((ny, nx), 1.0, dtype=wp.float32, device=self.device)
        # Device scalars so `compute` can retune them between CUDA-graph replays: a captured
        # graph freezes host floats at record time, and the two-solve gap needs to vary them.
        self._z_veto_d = wp.array([self.z_veto], dtype=wp.float32, device=self.device)
        self._sigma_scale_d = wp.array([1.0], dtype=wp.float32, device=self.device)
        self._goal_xy = wp.zeros(2, dtype=wp.float32, device=self.device)
        self._goal_rc = wp.zeros(2, dtype=wp.int32, device=self.device)
        self._prims: tuple | None = None  # host copy of the primitive tables, for policy walks
        self._graph = None

        self._prof = StageProfiler(
            self.device, ("settle", "feasibility", "route", "clamp"), profile
        )
        self._n_compute = 0

    def solve_gap(
        self,
        elevation: wp.array,
        goal_xy: tuple[float, float],
        sigma: wp.array,
        measured: wp.array | None = None,
        drift: wp.array | None = None,
    ) -> None:
        """Solve twice -- believing the map, then as if it were certain -- and keep both.

        The difference between the two value functions is what IGNORANCE is costing, in the
        plan cost's own units. It is the signal neither solve gives alone: a pessimistic
        planner never explores, because not knowing is expensive; an optimistic one always
        does, because not knowing is free. The gap between them is the honest quantity, and
        `gap_at` reads it at the robot's own pose.

        Results land in `V_pessimistic` / `V_optimistic` and `doubt_pessimistic`, because
        `compute` reuses `self.V` and `self.doubt` for whichever solve ran last.

        Costs one extra solve. Measured on the deployed 16 m window
        (`studies/planner/RESULTS.md` section 6.7) that is 7.9 ms for the first and as little as
        2.0 ms for a coarser second -- about 3% of a 69 ms sensor frame.
        """
        self.compute(elevation, goal_xy, measured, sigma, drift)
        wp.copy(self.V_pessimistic, self.V)
        wp.copy(self.doubt_pessimistic, self.doubt)
        # sigma_scale=0 discounts the measurement sd AND the drift spread together: the
        # optimistic reading asks what the robot would believe of a certain map, and a map with a
        # stale half is not one.
        self.compute(elevation, goal_xy, measured, sigma, drift, sigma_scale=0.0)
        wp.copy(self.V_optimistic, self.V)

    def gap_at(self, x: float, y: float, yaw: float) -> dict:
        """What ignorance costs at one pose: `V_pessimistic - V_optimistic`, in metres.

        Three things come off it, and they are the whole of the plan's section 4.3:

        - a **trigger**: explore only when `gap` is worth the detour. Equal value functions mean
          uncertainty is costing nothing here and the robot should simply drive.
        - a **target**: roll the optimistic policy out from here and collect the high-`doubt`
          poses along it -- those are the cells whose resolution would unlock the better route.
        - a **safety net**: `unreachable_by_ignorance` is true when the goal is out of reach on
          the believed map but in reach on a certain one. That is the blind-cell failure
          diagnosing itself, with a defined response (go look, or relax `z_veto`) instead of a
          planner that simply reports no path.

        Call after `solve_gap`. Nearest-cell lookup: this is one scalar of control data, not a
        field, so the host is the right place for it.
        """
        r, c, t = self._pose_index(x, y, yaw)
        v_pess = float(self.V_pessimistic.numpy()[r, c, t])
        v_opt = float(self.V_optimistic.numpy()[r, c, t])
        unreachable = self._vcap * 0.99
        return {
            "v_pessimistic": v_pess,
            "v_optimistic": v_opt,
            "gap_m": v_pess - v_opt,
            "reachable_pessimistic": v_pess < unreachable,
            "reachable_optimistic": v_opt < unreachable,
            "unreachable_by_ignorance": v_pess >= unreachable and v_opt < unreachable,
        }

    def doubt_targets(
        self,
        x: float,
        y: float,
        yaw: float,
        max_steps: int = 40,
        top_k: int = 8,
    ) -> list[dict]:
        """Where to look: the doubtful poses along the route the robot would take if it knew.

        Follows the OPTIMISTIC policy greedily from the robot's pose and collects the poses
        carrying `doubt` -- blocked by ignorance rather than by terrain. Those are the cells
        whose resolution would unlock the better route, so resolving them is what the plan
        means by decision-focused sensing: sense where the decision rests, not where entropy is
        highest. Maximising information gain instead sends the robot to look at whatever is
        least observed, which is usually the far edge of the map.

        This is the cheap form of `SENSITIVITY_PLAN.md`'s C4 -- a policy rollout rather than an
        adjoint. Call after `solve_gap`. The primitive tables come to the host once and the walk
        is 40 steps of table lookup, so it is control data, not a field.
        """
        if self._prims is None:
            sv = self.solver
            self._prims = (
                sv._prim_dr.numpy(),
                sv._prim_dc.numpy(),
                sv._prim_heading.numpy(),
                sv._prim_cost.numpy(),
            )
        prim_dr, prim_dc, prim_head, prim_cost = self._prims
        v_opt = self.V_optimistic.numpy()
        doubt = self.doubt_pessimistic.numpy()
        ny, nx = self.grid.cells_y, self.grid.cells_x
        cap = self._vcap * 0.99

        r, c, t = self._pose_index(x, y, yaw)
        hits: list[dict] = []
        seen = set()
        for _ in range(max_steps):
            if v_opt[r, c, t] >= cap:
                break  # the optimistic route does not reach the goal from here either
            if doubt[r, c, t] > 0.0 and (r, c, t) not in seen:
                seen.add((r, c, t))
                hits.append(
                    {
                        "x": float(self.grid.origin_x + c * self.grid.cell_size),
                        "y": float(self.grid.origin_y + r * self.grid.cell_size),
                        "heading": float((t + 0.5) * 2.0 * np.pi / self.n_theta),
                        "doubt": float(doubt[r, c, t]),
                    }
                )
            best, best_next = np.inf, None
            for pmt in range(self.solver.n_prim):
                nr, nc = r + int(prim_dr[t, pmt]), c + int(prim_dc[t, pmt])
                if not (0 <= nr < ny and 0 <= nc < nx):
                    continue
                nt = int(prim_head[t, pmt])
                total = float(prim_cost[t, pmt]) + float(v_opt[nr, nc, nt])
                if total < best:
                    best, best_next = total, (nr, nc, nt)
            if best_next is None or best >= v_opt[r, c, t] + 1e-6:
                break  # no primitive makes progress: the goal cell, or a dead end
            r, c, t = best_next
        hits.sort(key=lambda h: -h["doubt"])
        return hits[:top_k]

    def descent_bearing(self, x: float, y: float, radius_m: float) -> float:
        """The bearing [rad, this window's frame] from (x, y) to the cheapest routable cell of
        `V_escape` within `radius_m`, or nan when nothing in reach has a route.

        Where the way on lies relative to the robot -- the one number the turn-first brake needs,
        so it is reduced on device and read back as two floats rather than as the field.
        """
        r, c, _ = self._pose_index(x, y, 0.0)
        radius = max(int(round(radius_m / self.grid.cell_size)), 1)
        wp.launch(
            _descent_kernel,
            dim=1,
            inputs=[self.V_escape, r, c, radius, 0.9 * self._vcap],
            outputs=[self._descent_out],
            device=self.device,
        )
        bearing, best = self._descent_out.numpy()
        return float(bearing) if best < 0.9 * self._vcap else float("nan")

    def _pose_index(self, x: float, y: float, yaw: float) -> tuple[int, int, int]:
        """World pose -> (row, col, heading bin), clamped into the window."""
        c = int(
            np.clip(round((x - self.grid.origin_x) / self.grid.cell_size), 0, self.grid.cells_x - 1)
        )
        r = int(
            np.clip(round((y - self.grid.origin_y) / self.grid.cell_size), 0, self.grid.cells_y - 1)
        )
        t = int(np.floor(yaw / (2.0 * np.pi / self.n_theta))) % self.n_theta
        return r, c, t

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
            outputs=[self.blocked, self.hazard, self.violation, self.graded_tilt],
            device=self.device,
        )
        if self.z_veto > 0.0 or self.charge_per_sigma > 0.0:
            footprint_drift_spread(self._drift_in, self._drift_r, out=self._spread)
            wp.launch(
                _margin_kernel,
                dim=self.V.shape,
                inputs=[
                    sim.derived,
                    sim.clearance,
                    self._sigma_in,
                    self._spread,
                    self.grid,
                    self.robot,
                    self.n_theta,
                    self.sigma_floor_m,
                    self._z_veto_d,
                    self._sigma_scale_d,
                    self.z_charge,
                    self.charge_per_sigma,
                ],
                outputs=[self.blocked, self.graded_tilt, self.zmargin, self.doubt],
                device=self.device,
            )
        if self._eroded:  # a tilt over an unmountable face is a wall, not a slope
            wp.launch(
                _face_kernel,
                dim=(self.grid.cells_y, self.grid.cells_x),
                inputs=[self._elev_in, self._measured_in],
                outputs=[self._face],
                device=self.device,
            )
            wp.launch(
                _step_hazard_kernel,
                dim=self.V.shape,
                inputs=[self._face, self.grid, self.robot, self.blocked],
                outputs=[self.hazard],
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
                outputs=[self.blocked, self.hazard],
                device=self.device,
            )
        self._prof.mark(2)  # feasibility done
        if self._eroded:  # hazards eroded hard by the disturbance tube, soft blocks charged over it
            wp.launch(
                _robust_kernel,
                dim=self.V.shape,
                inputs=[
                    self.blocked,
                    self.hazard,
                    self.violation,
                    self.graded_tilt,
                    self._mr,
                    self._mc,
                    self._mt,
                    self.robust_soft_weight,
                ],
                outputs=[self.robust_blocked, self.robust_tilt],
                device=self.device,
            )
        feas = self.robust_blocked if self._eroded else self.blocked
        tilt = self.robust_tilt if self._eroded else self.graded_tilt
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
        wp.launch(
            _pose_cost_kernel,
            dim=self.V.shape,
            inputs=[feas, tilt],
            outputs=[self._pose_cost],
            device=self.device,
        )
        if self._coarse_in is None:
            wp.launch(
                _seed_goal_kernel,
                dim=self.V.shape,
                inputs=[self._goal_rc, self.solver._inf],
                outputs=[self._seeds],
                device=self.device,
            )
        else:
            wp.launch(
                _seed_goal_and_boundary_kernel,
                dim=self.V.shape,
                inputs=[
                    self._goal_xy,
                    self._coarse_in,
                    self._coarse_origin,
                    self._coarse_cell,
                    self.bounds[0],
                    self.bounds[2],
                    self.grid.cell_size,
                    self._band,
                    self.solver._inf,
                ],
                outputs=[self._seeds],
                device=self.device,
            )
        # capture=False: `compute` has already opened a ScopedCapture around this whole pipeline,
        # and value_iterate would try to nest a second one. Its `capture_while` still builds a
        # device-side conditional node inside the OUTER capture, so the loop stays on the GPU.
        result = self.solver.value_iterate(
            self._pose_cost, self._seeds, self.flatness_weight, capture=False
        )
        self._prof.mark(3)  # value iteration done (goal cell + solve)
        wp.launch(
            _clamp3d_kernel,
            dim=self.V.shape,
            inputs=[result, self._vcap],
            outputs=[self.V],
            device=self.device,
        )
        wp.launch(
            _solid_kernel,
            dim=(self.grid.cells_y, self.grid.cells_x),
            inputs=[self.hazard],
            outputs=[self._solid],
            device=self.device,
        )
        wp.launch(
            _escape_kernel,
            dim=self.V.shape,
            inputs=[
                self.V,
                self._solid,
                self._vcap,
                self._escape_reach,
                self.grid.cell_size,
                self.ESCAPE_PER_M,
                self._escape_per_bin,
                self.V_escape,
            ],
            device=self.device,
        )
        self._prof.mark(4)  # clamp + escape done

    def set_coarse(
        self,
        coarse_grid: GridParams,
        band: int | None = None,
    ) -> None:
        """Arm two-layer routing: this window's border is seeded from a coarser layer's V.

        `coarse_grid` is the coarse layer's geometry expressed in THIS window's frame. Its origin
        is the initial one: a layer bound to a window that recenters with this one keeps it, a
        layer anchored to the world passes each frame's through `compute(coarse_origin=...)`.

        `band` is the ring thickness in FINE cells and defaults to the furthest a single
        primitive reaches. A thinner ring can be stepped clean over by one arc, which seeds
        nothing and silently returns the single-layer behaviour.

        Call before the first `compute`. Arming it later would invalidate a graph already built
        around the single-layer seeding, so it refuses rather than replay a stale one.
        """
        if self._graph is not None:
            raise RuntimeError("set_coarse must be called before the first compute()")
        cy, cx = int(coarse_grid.cells_y), int(coarse_grid.cells_x)
        self._coarse_in = wp.zeros((cy, cx, 1), dtype=wp.float32, device=self.device)
        self._coarse_origin = wp.array(
            [float(coarse_grid.origin_x), float(coarse_grid.origin_y)],
            dtype=wp.float32,
            device=self.device,
        )
        self._coarse_cell = float(coarse_grid.cell_size)
        self._band = int(self.solver.reach_cells if band is None else band)
        if self._band < 1:
            raise ValueError(f"band must be >= 1 fine cell, got {self._band}")

    def compute(
        self,
        elevation: wp.array,
        goal_xy: tuple[float, float],
        measured: wp.array | None = None,
        sigma: wp.array | None = None,
        drift: wp.array | None = None,
        z_veto: float | None = None,
        sigma_scale: float | None = None,
        coarse_value: wp.array | None = None,
        coarse_origin: tuple[float, float] | None = None,
    ) -> wp.array:
        """elevation [ny, nx] device wp.array + goal -> clamped V[ny, nx, n_theta]. The entire solve
        (settle + value iteration) is captured ONCE as a CUDA graph and replayed each call with the
        new terrain/goal (copied into stable device buffers first) -- no host syncs in the loop.

        `measured` [ny, nx] (1 = observed, 0 = blind) is read ONLY by the step field (the
        obstacle_step_m gate, and the robust tube's step-hazard test), to keep the caller's
        blind-cell fill from reading as a real step. Omit it and every cell counts as observed.

        `drift` [ny, nx] is the belief's pose-drift variance (`var_h - var_meas`) [m^2], negative
        where nothing was measured. Two contacts share whatever drift accrued over their common
        interval, so only the SPREAD across the footprint reaches a margin -- see
        `terrain_value_field.drift`. Omit it and every cell is treated as one age, which is what
        the old behaviour assumed without saying so.

        `sigma` [ny, nx] is the elevation belief's per-cell MEASUREMENT sd (its `meas_sd`, not
        its `sigma`: pose drift is common-mode and cancels in the attitude differences). It is
        read only when `z_veto` or `charge_per_sigma` is non-zero; omitted, every cell falls back
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
        if drift is None:
            self._drift_in.zero_()  # one age everywhere -> no spread -> no extra uncertainty
        else:
            assert (
                drift.device == self.device
            ), f"drift must be a wp.array on {self.device}, got {drift.device}"
            wp.copy(self._drift_in, drift)
        if self._step_gate > 0.0 or self._eroded:  # only the step field reads the mask
            if measured is None:
                self._measured_in.fill_(1.0)
            else:
                assert (
                    measured.device == self.device
                ), f"measured must be a wp.array on {self.device}, got {measured.device}"
                wp.copy(self._measured_in, measured)
        self._z_veto_d.assign(np.array([self.z_veto if z_veto is None else z_veto], np.float32))
        self._sigma_scale_d.assign(
            np.array([1.0 if sigma_scale is None else sigma_scale], np.float32)
        )
        if coarse_value is not None:
            if self._coarse_in is None:
                raise RuntimeError("pass coarse_value only after set_coarse()")
            wp.copy(self._coarse_in, coarse_value)
        if coarse_origin is not None:
            if self._coarse_origin is None:
                raise RuntimeError("pass coarse_origin only after set_coarse()")
            self._coarse_origin.assign(np.asarray(coarse_origin[:2], np.float32))
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
