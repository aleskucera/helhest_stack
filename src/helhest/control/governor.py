"""Clearance speed governor: drive only as fast as the room around the plan can absorb.

The speed law, used in two places so the route and the robot agree:

    v_allowed(clearance) = max(v_min, (clearance - c0) / t_react)

`c0` [m] is the error that does NOT shrink with speed -- map cells (8 cm on the robot), sparse
returns on a wall edge, tracking when slow. Without it, closeness was free down to ~0.2 m at
t_react 0.125 (1.6 m/s allowed), so shortest-time routes hugged corners and the governor braked
hard at the last moment: slalom's first wall, 1.8 -> 0.17 m/s in 0.6 s, 0.20 m from the corner.

`t_react` [s] is SECONDS OF ERROR AT THE FASTEST POINT'S SPEED -- the part of the old spatial
margin that scales with speed. It is not a reaction time: driving parallel to a wall does not eat
clearance, only a heading or yaw-tracking error does, roughly sin(15 deg) = 0.25 of the speed. So
a 0.5 s reaction is t_react ~ 0.125 s; t_react 0.5 s slowed a 2.6 m corridor from 184 to 280-340
frames in sim, where the deployed robot drove it at 1.2 m/s with 0.26-0.37 m and never touched. The ROUTE prices every step by travel time under this law
(`CostToGo(time_cost=...)`, capped at a cruise speed there, since only the relative price
matters), so it swings wide of an obstacle whenever the wider path is faster. This governor then
ENFORCES the law on the one plan the robot drives: it reads the footprint's clearance to the
nearest wall face along MPPI's next `lookahead_s` and scales both wheels by one factor, which
keeps the plan's curvature and only slows it down.

Speed means the body's FASTEST point, not the axle: the tail sits `rear_offset + wheel_radius`
behind the drive axle, and a pivot at walking pace swings it sideways at over 1 m/s (measured:
a 27 deg turn at 0.1-0.4 m/s put it 0.02 m from a corridor wall).

A wall face is a cell rising more than a wheel radius above a 4-neighbour -- the same test the
cost-to-go uses for an unmountable step. Unmeasured cells are ignored: the router owns blind
ground, and the governor must not brake on the frontier.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from ..engine.robot import RobotParams
from ..engine.terrain import Grid


def speed_law(clearance: float, t_react: float, v_min: float, c0: float = 0.0) -> float:
    """[m/s] allowed speed of the body's fastest point at `clearance` [m]: the room past the fixed
    margin `c0`, divided by the seconds of error `t_react`."""
    return max(v_min, (clearance - c0) / t_react)


@wp.kernel
def _plan_clearance_kernel(
    controlled: wp.array2d(dtype=wp.vec3f),  # [T+1, B] rollout poses (x, y, yaw) on `grid`
    elevation: wp.array2d(dtype=wp.float32),
    measured: wp.array2d(dtype=wp.float32),  # 1 = real data
    grid: Grid,
    x_lo: wp.float32,  # footprint in the body frame [m], origin at the drive axle
    x_hi: wp.float32,
    half_w: wp.float32,
    face_h: wp.float32,  # [m] a rise this tall to a 4-neighbour is a wall face
    search: int,  # [cells] how far around the footprint to look
    out: wp.array(dtype=wp.float32),  # [K] clearance per plan step, `search` cells if none
    blind: wp.array(dtype=wp.float32),  # [K] never-measured area [m^2] the footprint newly covers
):
    """Clearance [m] from the footprint rectangle at plan step k to the nearest wall-face cell.
    One thread per plan step; rollout 0 is the nominal (MPPI keeps candidate 0 noise-free)."""
    k = wp.tid()
    p = controlled[k, 0]
    ca = wp.cos(p[2])
    sa = wp.sin(p[2])
    ny = elevation.shape[0]
    nx = elevation.shape[1]
    # the footprint's centre, and the cell it sits in
    xm = 0.5 * (x_lo + x_hi)
    cx = p[0] + ca * xm
    cy = p[1] + sa * xm
    c0 = int(wp.round((cx - grid.origin_x) / grid.cell_size))
    r0 = int(wp.round((cy - grid.origin_y) / grid.cell_size))
    best = float(search) * grid.cell_size
    hx = 0.5 * (x_hi - x_lo)
    # the footprint NOW (plan step 0): the ground under the robot is never measured -- the sensor
    # cannot see it -- so only ground the footprint is about to cover counts as blind
    p0 = controlled[0, 0]
    ca0 = wp.cos(p0[2])
    sa0 = wp.sin(p0[2])
    cx_0 = p0[0] + ca0 * xm
    cy_0 = p0[1] + sa0 * xm
    unseen = float(0.0)
    for i in range(-search, search + 1):
        r = r0 + i
        if r < 1 or r >= ny - 1:
            continue
        for j in range(-search, search + 1):
            c = c0 + j
            if c < 1 or c >= nx - 1:
                continue
            if measured[r, c] < 0.5:
                if k > 0:
                    gx = grid.origin_x + float(c) * grid.cell_size
                    gy = grid.origin_y + float(r) * grid.cell_size
                    ex = gx - cx
                    ey = gy - cy
                    in_k = wp.abs(ca * ex + sa * ey) <= hx and wp.abs(-sa * ex + ca * ey) <= half_w
                    fx = gx - cx_0
                    fy = gy - cy_0
                    in_0 = (
                        wp.abs(ca0 * fx + sa0 * fy) <= hx and wp.abs(-sa0 * fx + ca0 * fy) <= half_w
                    )
                    if in_k and not in_0:
                        unseen += grid.cell_size * grid.cell_size
                continue
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
            if rise <= face_h:
                continue
            # the wall cell in the footprint's frame, centred on the rectangle
            dx = grid.origin_x + float(c) * grid.cell_size - cx
            dy = grid.origin_y + float(r) * grid.cell_size - cy
            u = wp.abs(ca * dx + sa * dy) - hx
            v = wp.abs(-sa * dx + ca * dy) - half_w
            d = wp.sqrt(wp.max(u, 0.0) * wp.max(u, 0.0) + wp.max(v, 0.0) * wp.max(v, 0.0))
            d = d + wp.min(wp.max(u, v), 0.0)
            best = wp.min(best, d)
    out[k] = best
    blind[k] = unseen


@wp.kernel
def clearance_map_kernel(
    elevation: wp.array2d(dtype=wp.float32),
    measured: wp.array2d(dtype=wp.float32),  # 1 = real data
    cell: wp.float32,
    face_h: wp.float32,  # [m] a rise this tall to a 4-neighbour is a wall face
    reach: int,  # [cells] distances past this read as `reach * cell`
    out: wp.array2d(dtype=wp.float32),  # [m] distance from each cell centre to the nearest face
):
    """Per-cell distance to the nearest wall-face cell, the same face test as the governor, so
    MPPI can read a footprint's clearance with a few lookups per rollout step instead of a search.
    Built once per frame on the MPPI terrain."""
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
            if d >= best or measured[rr, cc] < 0.5:
                continue
            h = elevation[rr, cc]
            rise = float(0.0)
            if measured[rr - 1, cc] > 0.5:
                rise = wp.max(rise, h - elevation[rr - 1, cc])
            if measured[rr + 1, cc] > 0.5:
                rise = wp.max(rise, h - elevation[rr + 1, cc])
            if measured[rr, cc - 1] > 0.5:
                rise = wp.max(rise, h - elevation[rr, cc - 1])
            if measured[rr, cc + 1] > 0.5:
                rise = wp.max(rise, h - elevation[rr, cc + 1])
            if rise > face_h:
                best = d
    out[r, c] = best


class ClearanceGovernor:
    """Scale MPPI's first command so the body's fastest point moves no faster than the clearance
    along the next `lookahead_s` of the plan allows. Device work is one tiny kernel; the only
    readback is `lookahead` floats, a host control value."""

    def __init__(
        self,
        robot: RobotParams,
        plan_dt: float,
        t_react: float = 0.125,
        v_min: float = 0.15,
        c0: float = 0.15,  # [m] fixed margin: error that does not shrink with speed
        t_turn: float | None = None,  # [s] the tail swing's own t_react; None = t_react
        lookahead_s: float = 1.0,
        decel: float = 2.0,  # [m/s^2] how fast the fastest body point can shed speed
        search_m: float = 1.5,
        # [m/s] the fastest point's speed while the footprint is about to cover ground nobody has
        # measured -- beside and behind the robot the sensor has never looked. 0 = off.
        v_blind: float = 0.3,
        blind_area: float = 0.05,  # [m^2] of newly covered unseen ground that counts
        device: str | None = None,
    ) -> None:
        self.device = wp.get_device(device)
        self.t_react = float(t_react)
        self.v_min = float(v_min)
        self.c0 = float(c0)
        # a turn is the least certain thing this robot does (the drivetrain realised ~0.74x the
        # commanded differential; turn_boost compensates, per terrain), so the tail's swing is
        # charged against a longer reaction time than driving straight: turning in a gap costs more
        self.turn_ratio = (t_react if t_turn is None else float(t_turn)) / float(t_react)
        self.r = float(robot.wheel_radius)
        self.half_track = float(robot.half_track)
        self.tail = float(robot.rear_offset + robot.wheel_radius)
        side = robot.half_track + 0.5 * (robot.wheel_width or 2.0 * robot.wheel_radius)
        self.x_lo, self.x_hi, self.half_w = -self.tail, float(robot.wheel_radius), float(side)
        self.face_h = float(robot.wheel_radius)
        self.search_m = float(search_m)
        self.steps = max(1, int(math.ceil(lookahead_s / plan_dt)))
        self.plan_dt = float(plan_dt)
        self.decel = float(decel)
        self._out = wp.zeros(self.steps + 1, dtype=wp.float32, device=self.device)
        self._blind = wp.zeros(self.steps + 1, dtype=wp.float32, device=self.device)
        self.v_blind = float(v_blind)
        self.blind_area = float(blind_area)
        self.blind = False  # last frame's: was the plan about to sweep unseen ground
        self.clearance = float("inf")  # last frame's, for logging
        self.v_cap = float("inf")

    def cap(
        self,
        wl: float,
        wr: float,
        controlled: wp.array,
        elevation: wp.array,
        measured: wp.array,
        grid: Grid,
    ) -> tuple[float, float]:
        """(wl, wr) [rad/s], scaled by one factor so the fastest body point obeys the law."""
        k = min(self.steps + 1, controlled.shape[0])
        search = max(1, int(math.ceil(self.search_m / float(grid.cell_size))))
        wp.launch(
            _plan_clearance_kernel,
            dim=k,
            inputs=[
                controlled,
                elevation,
                measured,
                grid,
                self.x_lo,
                self.x_hi,
                self.half_w,
                self.face_h,
                search,
            ],
            outputs=[self._out, self._blind],
            device=self.device,
        )
        per_step = self._out.numpy()[:k]
        unseen = self._blind.numpy()[:k] >= self.blind_area
        self.clearance = float(np.min(per_step))
        # Step i is reached i * plan_dt from now, and the robot can brake on the way: it only has
        # to be at that step's allowed speed WHEN it gets there. Taking the plain minimum instead
        # made a plan that is tight only at its far end brake to the floor NOW, every frame, and
        # stalled the robot 1-2.5 s at pocket's corner while MPPI kept proposing the same spin.
        t_i = np.arange(k) * self.plan_dt
        allowed = np.maximum(self.v_min, (per_step - self.c0) / self.t_react)
        # a wall the sensor never saw is invisible to the clearance above: measured, a turn at
        # the start swung the tail to 0.08 m from one while the map read 0.27 m of room
        if self.v_blind > 0.0:
            allowed = np.where(unseen, np.minimum(allowed, self.v_blind), allowed)
        self.blind = bool(unseen.any())
        self.v_cap = float(np.min(allowed + self.decel * t_i))
        v = self.r * 0.5 * abs(wl + wr)
        wz = self.r * abs(wr - wl) / (2.0 * self.half_track)  # alpha 1: over-estimates the swing
        fastest = v + self.turn_ratio * self.tail * wz
        if fastest <= self.v_cap:
            return wl, wr
        s = self.v_cap / fastest
        return wl * s, wr * s
