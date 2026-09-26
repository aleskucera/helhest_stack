"""The clearance speed law: how fast the robot may move given the room around it.

    v_allowed(clearance) = max(v_min, (clearance - c0) / t_react)

applied to the body's FASTEST point, |v| + tail * |wz|: the tail trails 1.1 m behind the drive
axle, so a turn at walking pace can swing it sideways faster than the axle moves.

- `c0` [m] is the error that does not shrink with speed: map cells, sparse returns on a wall
  edge, tracking at low speed.
- `t_react` [s] is seconds of error per unit speed. It is small (0.125 s) because driving parallel
  to a wall does not eat clearance; only a heading or yaw-tracking error does.
- `t_turn` [s] is the same for the tail's swing. Turning is where this robot's model is least
  accurate, so it is counted longer.

One law, used in three places so they agree: the route prices poses in travel time
(`ClearanceRoute` below), MPPI charges rollouts for the time the law would cost
(control/mppi.py), and the governor enforces it on the plan the robot drives
(control/governor.py). Nothing is forbidden short of contact; closeness only costs time.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import warp as wp

from ..engine.robot import RobotParams


@dataclass(frozen=True)
class ClearanceParams:
    """The clearance law and the knobs of its three users. Built by `helhest.planner_config`."""

    t_react: float = 0.125  # [s] error per unit speed of the fastest body point
    c0: float = 0.15  # [m] fixed margin
    v_min: float = 0.15  # [m/s] the law never asks for less than a crawl
    v_cruise: float = 1.5  # [m/s] the route's time unit
    t_turn: float = 0.25  # [s] the tail swing's own t_react
    route_turn: bool = True  # the route charges turning arcs for their tail near walls
    mppi_weight: float = 1.0  # MPPI's price for lost time; 1 = exact in goal-cost units
    lookahead_s: float = 1.0  # [s] of the plan the governor checks
    decel: float = 2.0  # [m/s^2] braking the governor may count on before a tight step
    v_blind: float = 0.3  # [m/s] while the plan sweeps never-measured ground; 0 = off
    blind_area: float = 0.05  # [m^2] of newly swept unseen ground that counts
    search_m: float = 1.5  # [m] how far around the footprint the governor looks for walls

    @property
    def turn_ratio(self) -> float:
        """How much more the tail's swing counts than forward speed."""
        return self.t_turn / self.t_react

    def allowed(self, clearance: float) -> float:
        """[m/s] allowed speed of the fastest body point at `clearance` [m]."""
        return max(self.v_min, (clearance - self.c0) / self.t_react)


@wp.func
def allowed_speed(clearance: float, c0: float, t_react: float, v_min: float) -> float:
    """The law inside kernels; see `ClearanceParams.allowed`."""
    return wp.max(v_min, (clearance - c0) / t_react)


@wp.func
def is_wall_face(
    elevation: wp.array2d(dtype=wp.float32),
    measured: wp.array2d(dtype=wp.float32),
    r: int,
    c: int,
    face_h: float,
) -> bool:
    """A measured cell rising more than `face_h` above a measured 4-neighbour: a step the wheel
    cannot mount. Unmeasured cells are never faces; the caller keeps r, c off the border."""
    if measured[r, c] < 0.5:
        return False
    h = elevation[r, c]
    rise = float(0.0)
    if measured[r - 1, c] > 0.5:
        rise = wp.max(rise, h - elevation[r - 1, c])
    if measured[r + 1, c] > 0.5:
        rise = wp.max(rise, h - elevation[r + 1, c])
    if measured[r, c - 1] > 0.5:
        rise = wp.max(rise, h - elevation[r, c - 1])
    if measured[r, c + 1] > 0.5:
        rise = wp.max(rise, h - elevation[r, c + 1])
    return rise > face_h


@wp.kernel
def clearance_map_kernel(
    elevation: wp.array2d(dtype=wp.float32),
    measured: wp.array2d(dtype=wp.float32),  # 1 = real data
    cell: wp.float32,
    face_h: wp.float32,
    reach: int,  # [cells] distances past this read as `reach * cell`
    out: wp.array2d(dtype=wp.float32),  # [m] distance from each cell centre to the nearest face
):
    """Per-cell distance to the nearest wall face, so MPPI reads a footprint's clearance with a
    few lookups per rollout step instead of a search."""
    r, c = wp.tid()
    ny = elevation.shape[0]
    nx = elevation.shape[1]
    best = float(reach) * cell
    for i in range(-reach, reach + 1):
        rr = r + i
        if rr < 1 or rr >= ny - 1:
            continue
        for j in range(-reach, reach + 1):
            cc = c + j
            if cc < 1 or cc >= nx - 1:
                continue
            d = wp.sqrt(float(i * i + j * j)) * cell
            if d < best:
                if is_wall_face(elevation, measured, rr, cc, face_h):
                    best = d
    out[r, c] = best


# --- the route's share: price every pose in travel time ----------------------------------------
#
# The route's share of the clearance law (above): price every pose in travel time.
#
# With the clearance law on, the cost-to-go no longer removes poses near walls with its spatial
# tube; only contact and the heading bin block. Each free pose is charged instead, and two prices
# come out of one nearest-wall search:
#
# - Time: the solver charges a step `arc * (1 + flatness * pen)`, so `pen = (v_cruise / v - 1) /
#   flatness` makes it `arc * v_cruise / v` -- seconds, in metres at cruise speed. The route then
#   swings wide of an obstacle wherever wide is faster, and pays the time where there is no wide
#   option.
# - Turn: a turning arc of radius R swings the tail at v * lever / R, so near walls it is slower
#   than a straight move through the same cells. The solver charges that extra time
#   (terrain_value_field `ValueSolver.set_turn_price`) from a per-pose multiplier filled here.
#
# A pose's clearance is `(d - 0.5) * cell`, `d` the Chebyshev distance [cells] to the nearest contact
# pose within the heading bin: between d-1 and d cells of room.
@wp.func
def _contact_near(hazard: wp.array3d(dtype=wp.float32), r: int, c: int, t: int, dt: int) -> bool:
    """Contact at cell (r, c), clamped to the window, for any heading within `dt` bins of t."""
    rr = wp.clamp(r, 0, hazard.shape[0] - 1)
    cc = wp.clamp(c, 0, hazard.shape[1] - 1)
    nth = hazard.shape[2]
    found = bool(False)
    for k in range(-dt, dt + 1):
        if hazard[rr, cc, (t + k + nth) % nth] > 0.5:
            found = True
    return found


@wp.kernel
def _clearance_route_kernel(
    hazard: wp.array3d(dtype=wp.float32),  # contact / unresolved settle, per pose
    loose: wp.array3d(dtype=wp.float32),  # blocked under the heading bin alone
    reach_time: int,  # [cells] past this the time price is zero
    reach_turn: int,  # [cells] past this the turn price is zero; 0 = no turn price
    dt: int,  # the heading bin [bins]
    cell: wp.float32,
    v_cruise: wp.float32,
    t_react: wp.float32,
    v_min: wp.float32,
    c0: wp.float32,
    inv_flatness: wp.float32,
    tilt: wp.array3d(dtype=wp.float32),  # the loose set's graded tilt, charged in place
    turn_time: wp.array3d(dtype=wp.float32),  # v_cruise / v_allowed, uncapped; 0 past reach_turn
):
    r, c, t = wp.tid()
    turn_time[r, c, t] = 0.0
    if loose[r, c, t] > 0.5:
        return
    reach = wp.max(reach_time, reach_turn)
    # search outward ring by ring and stop at the first contact: most poses are far from walls
    best = reach + 1
    for d in range(1, reach + 1):
        found = bool(False)
        for i in range(-d, d + 1):
            if _contact_near(hazard, r - d, c + i, t, dt) or _contact_near(
                hazard, r + d, c + i, t, dt
            ):
                found = True
        for i in range(-d + 1, d):
            if _contact_near(hazard, r + i, c - d, t, dt) or _contact_near(
                hazard, r + i, c + d, t, dt
            ):
                found = True
        if found:
            best = d
            break
    if best > reach:
        return
    v = allowed_speed((float(best) - 0.5) * cell, c0, t_react, v_min)
    if best <= reach_time:
        tilt[r, c, t] = tilt[r, c, t] + (v_cruise / wp.min(v_cruise, v) - 1.0) * inv_flatness
    if best <= reach_turn:
        turn_time[r, c, t] = v_cruise / v


class ClearanceRoute:
    """The time and turn prices for one routing grid; `CostToGo(clearance=...)` owns one."""

    def __init__(
        self,
        params: ClearanceParams,
        robot: RobotParams,
        cell: float,
        shape: tuple[int, int, int],
        heading_bins: int,
        flatness_weight: float,
        solver,
        device,
    ) -> None:
        if flatness_weight <= 0.0:
            raise ValueError("the time price rides on the solver's penalty: flatness_weight > 0")
        self.params = params
        self.cell = float(cell)
        self.heading_bins = int(heading_bins)
        self.inv_flatness = 1.0 / float(flatness_weight)
        self.device = device
        self.reach_time = self._reach(params.v_cruise)
        lever = params.turn_ratio * (robot.rear_offset + robot.wheel_radius)
        if not params.route_turn:
            lever = 0.0
        elif not hasattr(solver, "set_turn_price"):
            warnings.warn(
                "terrain_value_field has no ValueSolver.set_turn_price: the route's turn price "
                "is OFF -- update the terrain_value_field pin"
            )
            lever = 0.0
        self.lever = lever
        # the sharpest turn slows the tail at clearances where driving straight is not slowed
        k_max = lever / float(robot.min_turn_radius)
        self.reach_turn = self._reach(params.v_cruise * (1.0 + k_max)) if lever > 0.0 else 0
        self.turn_time = wp.zeros(shape, dtype=wp.float32, device=device)
        if lever > 0.0:
            solver.set_turn_price(self.turn_time, lever)

    def _reach(self, speed: float) -> int:
        """[cells] beyond which the law allows `speed`."""
        p = self.params
        return max(1, int(math.ceil((speed * p.t_react + p.c0) / self.cell + 0.5)))

    def charge(self, hazard: wp.array, loose: wp.array, tilt: wp.array) -> None:
        """Add the time price to `tilt` in place and fill `turn_time`, for the free poses of
        `loose` (blocked under the heading bin alone)."""
        p = self.params
        wp.launch(
            _clearance_route_kernel,
            dim=tilt.shape,
            inputs=[
                hazard,
                loose,
                self.reach_time,
                self.reach_turn,
                self.heading_bins,
                self.cell,
                float(p.v_cruise),
                float(p.t_react),
                float(p.v_min),
                float(p.c0),
                self.inv_flatness,
            ],
            outputs=[tilt, self.turn_time],
            device=self.device,
        )
