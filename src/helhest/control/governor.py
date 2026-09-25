"""Clearance speed governor: drive only as fast as the room around the plan can absorb.

The speed law, used in two places so the route and the robot agree:

    v_allowed(clearance) = max(v_min, clearance / t_react)

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


def speed_law(clearance: float, t_react: float, v_min: float) -> float:
    """[m/s] allowed speed of the body's fastest point at `clearance` [m]."""
    return max(v_min, clearance / t_react)


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
    for i in range(-search, search + 1):
        r = r0 + i
        if r < 1 or r >= ny - 1:
            continue
        for j in range(-search, search + 1):
            c = c0 + j
            if c < 1 or c >= nx - 1:
                continue
            if measured[r, c] < 0.5:
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
        lookahead_s: float = 1.0,
        search_m: float = 1.5,
        device: str | None = None,
    ) -> None:
        self.device = wp.get_device(device)
        self.t_react = float(t_react)
        self.v_min = float(v_min)
        self.r = float(robot.wheel_radius)
        self.half_track = float(robot.half_track)
        self.tail = float(robot.rear_offset + robot.wheel_radius)
        side = robot.half_track + 0.5 * (robot.wheel_width or 2.0 * robot.wheel_radius)
        self.x_lo, self.x_hi, self.half_w = -self.tail, float(robot.wheel_radius), float(side)
        self.face_h = float(robot.wheel_radius)
        self.search_m = float(search_m)
        self.steps = max(1, int(math.ceil(lookahead_s / plan_dt)))
        self._out = wp.zeros(self.steps + 1, dtype=wp.float32, device=self.device)
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
            outputs=[self._out],
            device=self.device,
        )
        self.clearance = float(np.min(self._out.numpy()[:k]))
        self.v_cap = speed_law(max(self.clearance, 0.0), self.t_react, self.v_min)
        v = self.r * 0.5 * abs(wl + wr)
        wz = self.r * abs(wr - wl) / (2.0 * self.half_track)  # alpha 1: over-estimates the swing
        fastest = v + self.tail * wz
        if fastest <= self.v_cap:
            return wl, wr
        s = self.v_cap / fastest
        return wl * s, wr * s
