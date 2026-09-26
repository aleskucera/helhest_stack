"""Clearance speed governor: enforce the clearance law (planning/clearance.py) on the plan the
robot drives.

It reads the footprint's clearance to the nearest wall face along MPPI's next `lookahead_s` and
scales both wheels by one factor, which keeps the plan's curvature and only slows it down. Plan
step i is reached i * dt from now and the robot can brake on the way, so a step only limits the
speed braking cannot shed by then; a plan that is tight only at its far end is not braked now.

While the plan is about to sweep ground nobody has measured -- beside and behind the robot the
sensor has never looked -- the fastest point is capped at `v_blind`: a wall there is invisible to
the clearance. Ground under the footprint now is exempt; it is never measured.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from ..engine.robot import RobotParams
from ..engine.terrain import Grid
from ..planning.clearance import ClearanceParams
from ..planning.clearance import is_wall_face


@wp.func
def _in_rect(
    gx: float, gy: float, cx: float, cy: float, ca: float, sa: float, hx: float, hw: float
) -> bool:
    ex = gx - cx
    ey = gy - cy
    return wp.abs(ca * ex + sa * ey) <= hx and wp.abs(-sa * ex + ca * ey) <= hw


@wp.kernel
def _plan_clearance_kernel(
    controlled: wp.array2d(dtype=wp.vec3f),  # [T+1, B] rollout poses (x, y, yaw) on `grid`
    elevation: wp.array2d(dtype=wp.float32),
    measured: wp.array2d(dtype=wp.float32),  # 1 = real data
    grid: Grid,
    x_lo: wp.float32,  # footprint in the body frame [m], origin at the drive axle
    x_hi: wp.float32,
    half_w: wp.float32,
    face_h: wp.float32,
    search: int,  # [cells] how far around the footprint to look
    out: wp.array(dtype=wp.float32),  # [K] clearance per plan step, `search` cells if none
    blind: wp.array(dtype=wp.float32),  # [K] never-measured area [m^2] the footprint newly covers
):
    """Per plan step k (rollout 0, MPPI's noise-free nominal): the clearance [m] from the footprint
    rectangle to the nearest wall face, and the unseen area it covers that step 0 does not."""
    k = wp.tid()
    p = controlled[k, 0]
    ca = wp.cos(p[2])
    sa = wp.sin(p[2])
    ny = elevation.shape[0]
    nx = elevation.shape[1]
    xm = 0.5 * (x_lo + x_hi)
    hx = 0.5 * (x_hi - x_lo)
    cx = p[0] + ca * xm
    cy = p[1] + sa * xm
    p0 = controlled[0, 0]
    ca0 = wp.cos(p0[2])
    sa0 = wp.sin(p0[2])
    cx0 = p0[0] + ca0 * xm
    cy0 = p0[1] + sa0 * xm
    rc = int(wp.round((cy - grid.origin_y) / grid.cell_size))
    cc = int(wp.round((cx - grid.origin_x) / grid.cell_size))
    best = float(search) * grid.cell_size
    unseen = float(0.0)
    for i in range(-search, search + 1):
        r = rc + i
        if r < 1 or r >= ny - 1:
            continue
        for j in range(-search, search + 1):
            c = cc + j
            if c < 1 or c >= nx - 1:
                continue
            gx = grid.origin_x + float(c) * grid.cell_size
            gy = grid.origin_y + float(r) * grid.cell_size
            if measured[r, c] < 0.5:
                if k > 0:
                    if _in_rect(gx, gy, cx, cy, ca, sa, hx, half_w):
                        if not _in_rect(gx, gy, cx0, cy0, ca0, sa0, hx, half_w):
                            unseen += grid.cell_size * grid.cell_size
                continue
            if not is_wall_face(elevation, measured, r, c, face_h):
                continue
            # signed distance from the wall cell to the rectangle, in the footprint's frame
            dx = gx - cx
            dy = gy - cy
            u = wp.abs(ca * dx + sa * dy) - hx
            v = wp.abs(-sa * dx + ca * dy) - half_w
            d = wp.sqrt(wp.max(u, 0.0) * wp.max(u, 0.0) + wp.max(v, 0.0) * wp.max(v, 0.0))
            best = wp.min(best, d + wp.min(wp.max(u, v), 0.0))
    out[k] = best
    blind[k] = unseen


class ClearanceGovernor:
    """Scale MPPI's first command so the body's fastest point obeys the clearance law along the
    next `params.lookahead_s` of the plan. One small kernel; the readback is a few floats."""

    def __init__(
        self,
        robot: RobotParams,
        plan_dt: float,
        params: ClearanceParams,
        device: str | None = None,
    ) -> None:
        self.device = wp.get_device(device)
        self.params = params
        self.plan_dt = float(plan_dt)
        self.r = float(robot.wheel_radius)
        self.half_track = float(robot.half_track)
        self.tail = float(robot.rear_offset + robot.wheel_radius)
        side = robot.half_track + 0.5 * (robot.wheel_width or 2.0 * robot.wheel_radius)
        self.x_lo, self.x_hi, self.half_w = -self.tail, float(robot.wheel_radius), float(side)
        self.face_h = float(robot.wheel_radius)
        self.steps = max(1, int(math.ceil(params.lookahead_s / plan_dt)))
        self._out = wp.zeros(self.steps + 1, dtype=wp.float32, device=self.device)
        self._blind = wp.zeros(self.steps + 1, dtype=wp.float32, device=self.device)
        # last frame's readings, for logging
        self.clearance = float("inf")
        self.v_cap = float("inf")
        self.blind = False

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
        p = self.params
        k = min(self.steps + 1, controlled.shape[0])
        search = max(1, int(math.ceil(p.search_m / float(grid.cell_size))))
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
        unseen = self._blind.numpy()[:k] >= p.blind_area
        allowed = np.maximum(p.v_min, (per_step - p.c0) / p.t_react)
        if p.v_blind > 0.0:
            allowed = np.where(unseen, np.minimum(allowed, p.v_blind), allowed)
        self.clearance = float(np.min(per_step))
        self.blind = bool(unseen.any())
        self.v_cap = float(np.min(allowed + p.decel * np.arange(k) * self.plan_dt))
        v = self.r * 0.5 * abs(wl + wr)
        wz = self.r * abs(wr - wl) / (2.0 * self.half_track)  # alpha 1: over-estimates the swing
        fastest = v + p.turn_ratio * self.tail * wz
        if fastest <= self.v_cap:
            return wl, wr
        s = self.v_cap / fastest
        return wl * s, wr * s
