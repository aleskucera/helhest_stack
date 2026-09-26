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

import math
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..engine.robot import Robot  # the built struct, passed straight into the hazard kernel
from ..engine.terrain import Grid
from ..profiling import StageProfiler
from .clearance import ClearanceParams
from .clearance import ClearanceRoute
from .settle_producer import SettleProducer
from .terrain_value_field import closing_step
from .terrain_value_field import TerrainValueField
from .terrain_value_field.hierarchical import goal_cell_kernel
from .terrain_value_field.hierarchical import seed_goal_and_ring_kernel
from .terrain_value_field.hierarchical import seed_goal_kernel
from .terrain_value_field.margin import pack_pose_cost_kernel

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
def _unpack_kernel(
    pose_cost: wp.array3d(dtype=wp.float32),  # [row, col, heading] classified, veto in the sign
    penalty: wp.array3d(dtype=wp.float32),  # [row, col, heading] the sigma charge, exact
    flat_tilt: wp.array3d(dtype=wp.float32),  # [row, col, heading] the producer's flatness cost
    blocked: wp.array3d(dtype=wp.float32),
    graded_tilt: wp.array3d(dtype=wp.float32),
):
    """The classified field back into the veto and the graded cost the tube works on. The
    penalty is read from its own field: a vetoed state's magnitude does not survive -1 - p."""
    r, c, t = wp.tid()
    blocked[r, c, t] = wp.where(pose_cost[r, c, t] < 0.0, 1.0, 0.0)
    graded_tilt[r, c, t] = flat_tilt[r, c, t] + penalty[r, c, t]


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
        # the clearance law (planning/clearance.py): price poses in travel time instead of removing
        # them with the spatial tube. None = the tube vetoes.
        clearance: ClearanceParams | None = None,
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
        # Odin's physics: the settle at every pose, as terrain_value_field constraints
        self.producer = SettleProducer(
            grid_params,
            robot_params,
            solver_params,
            n_theta,
            self.sigma_floor_m,
            obstacle_step_m=self._step_gate,
            device=self.device,
        )
        self.settle_sim = self.producer.settle_sim
        # the cost-to-go machinery: classification in sigmas, then the value iteration
        self.field = TerrainValueField(
            ny,
            nx,
            self.grid.cell_size,
            n_theta,
            z_veto=self.z_veto,
            z_charge=self.z_charge,
            charge_per_sigma=self.charge_per_sigma,
            penalty_scale=self.flatness_weight,
            turn_radius=float(self.robot.min_turn_radius),
            step=self.step,
            # helhest spells "no point turns" as 0.0; the solver spells it as an infinite price,
            # which leaves the two primitives out of the table instead of pricing them out.
            pivot_cost=math.inf if pivot_cost <= 0.0 else float(pivot_cost),
            device=self.device,
        )
        self.solver = self.field.solver

        # Two-layer routing is opt-in and is armed by `set_coarse`, not by the constructor: the
        # coarse grid's SHAPE and cell size are constant, and its origin and values live in device
        # arrays the captured graph reads -- so a coarse layer anchored to the world, which the
        # window slides across, moves under the graph without invalidating it.
        self._coarse_in: wp.array | None = None
        self._coarse_origin: wp.array | None = None  # [2], in this window's frame
        self._coarse_cell = 0.0
        self._band = 0

        self.V = wp.zeros((ny, nx, n_theta), dtype=wp.float32, device=self.device)
        self.blocked = wp.zeros_like(self.V)
        self.zmargin = self.field.z  # safety margin in sigmas, per pose
        self.doubt = self.field.doubt  # > 0 where a pose is blocked by IGNORANCE alone
        self.V_optimistic = wp.zeros_like(self.V)  # filled by solve_gap()
        self.V_pessimistic = wp.zeros_like(self.V)  # ditto; `compute` reuses self.V
        self.doubt_pessimistic = wp.zeros_like(self.V)
        self.robust_blocked = wp.zeros_like(self.V)  # blocked after the disturbance-tube erosion
        self.graded_tilt = wp.zeros_like(self.V)  # flatness + the sigma charge
        # blocked HARD: a wall, or a settle that did not resolve. Also what MPPI vetoes hard --
        # WITHOUT the tube: the tube is the router's margin, and vetoing it in the controller froze
        # the robot beside every wall (each move, even a turn in place, clipped it)
        self.hazard = self.producer.hazard
        self.violation = self.producer.violation  # [rad] how badly a soft test fails
        self.robust_tilt = wp.zeros_like(self.V)  # graded_tilt + the tube's tilt charge
        # blocked under the heading bin alone, and its tilt: what routes when the clearance law
        # replaces the spatial tube
        self._loose_blocked = wp.zeros_like(self.V)
        self._loose_tilt = wp.zeros_like(self.V)
        self.clearance_route = (
            ClearanceRoute(
                clearance,
                robot_params,
                self.grid.cell_size,
                tuple(self.V.shape),
                self._mt,
                self.flatness_weight,
                self.solver,
                self.device,
            )
            if clearance is not None
            else None
        )
        # V with a way back out of every no-route pose: what MPPI follows (see _escape_kernel)
        self.V_escape = wp.zeros_like(self.V)
        self._descent_out = wp.zeros(2, dtype=wp.float32, device=self.device)
        self._solid = wp.zeros((ny, nx), dtype=wp.float32, device=self.device)  # inside a wall
        # what the solver actually reads: veto in the sign, graded cost in the magnitude
        self._pose_cost = self.field.pose_cost
        self._seeds = self.field.seeds
        self._spread = self.producer.spread
        self._face = wp.zeros((ny, nx), dtype=wp.float32, device=self.device)  # face height

        self._elev_in = wp.zeros((ny, nx), dtype=wp.float32, device=self.device)
        # Per-cell measurement sd. Defaults to zero, which the floor then lifts to
        # `sigma_floor_m` everywhere -- so an unsupplied sigma is a uniform-uncertainty map, not
        # a claim of perfect knowledge.
        self._sigma_in = wp.zeros((ny, nx), dtype=wp.float32, device=self.device)
        # Pose-drift variance per cell (`var_h - var_meas` from the belief). Zero everywhere is a
        # map of one age, which contributes nothing -- so a caller that passes no drift gets
        # exactly the no-drift behaviour.
        self._drift_in = wp.zeros((ny, nx), dtype=wp.float32, device=self.device)
        # Stable mask buffer the captured graph reads. Defaults to all-measured, so a caller that
        # passes no mask gets exactly the pre-mask behaviour.
        self._measured_in = wp.full((ny, nx), 1.0, dtype=wp.float32, device=self.device)
        # A device scalar so `compute` can retune it between CUDA-graph replays: a captured graph
        # freezes host floats at record time, and the two-solve gap needs to vary it.
        self._sigma_scale_d = wp.array([1.0], dtype=wp.float32, device=self.device)
        self._goal_xy = wp.zeros(2, dtype=wp.float32, device=self.device)
        self._goal_rc = wp.zeros(2, dtype=wp.int32, device=self.device)
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
        """Record the whole pipeline on stable owned buffers: terrain -> settle -> constraints ->
        classification -> tube -> goal cell (on device) -> value iteration -> clamp into V. Used
        both to build the captured graph (capture=True) and for the eager CPU fallback
        (capture=False)."""
        self._prof.mark(0)
        self.producer.settle(self._elev_in)
        self._prof.mark(1)  # settle done
        constraints = self.producer.run(
            self._elev_in, self._measured_in, self._sigma_in, self._drift_in, self._sigma_scale_d
        )
        self.field.classify(constraints)
        wp.launch(
            _unpack_kernel,
            dim=self.V.shape,
            inputs=[self.field.pose_cost, self.field.penalty, self.producer.tilt],
            outputs=[self.blocked, self.graded_tilt],
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
        if self.clearance_route is not None:  # the spatial tube becomes a price in seconds
            wp.launch(
                _robust_kernel,
                dim=self.V.shape,
                inputs=[
                    self.blocked,
                    self.hazard,
                    self.violation,
                    self.graded_tilt,
                    0,
                    0,
                    self._mt,
                    self.robust_soft_weight,
                ],
                outputs=[self._loose_blocked, self._loose_tilt],
                device=self.device,
            )
            self.clearance_route.charge(self.hazard, self._loose_blocked, self._loose_tilt)
            feas, tilt = self._loose_blocked, self._loose_tilt
        wp.launch(
            goal_cell_kernel,
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
            pack_pose_cost_kernel,
            dim=self.V.shape,
            inputs=[feas, tilt],
            outputs=[self.field.pose_cost],
            device=self.device,
        )
        if self._coarse_in is None:
            wp.launch(
                seed_goal_kernel,
                dim=self.V.shape,
                inputs=[self._goal_rc, self.solver._inf],
                outputs=[self._seeds],
                device=self.device,
            )
        else:
            wp.launch(
                seed_goal_and_ring_kernel,
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
        result = self.field.iterate(capture=False)
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
        self.field.set_z_veto(self.z_veto if z_veto is None else z_veto)
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
