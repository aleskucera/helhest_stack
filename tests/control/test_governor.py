"""The clearance speed governor: the footprint's distance to a wall face along the plan, and the
one-factor slowdown that makes the body's fastest point obey v = clearance / t_react."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from helhest.control.governor import ClearanceGovernor
from helhest.control.governor import speed_law
from helhest.engine import GridParams
from helhest.engine import RobotParams

CELL = 0.05
N = 160  # 8 m square, origin at the min corner
ROBOT = RobotParams(wheel_width=0.1)


def _scene(wall_y: float) -> tuple[wp.array, wp.array, object]:
    """Flat ground with a 1 m wall occupying y >= wall_y."""
    ys = (np.arange(N) + 0.5) * CELL
    h = np.zeros((N, N), np.float32)
    h[ys >= wall_y, :] = 1.0
    grid = GridParams(N, N, CELL, 0.0, 0.0).build()
    return wp.array(h), wp.array(np.ones((N, N), np.float32)), grid


def _plan(y: float, yaw: float = 0.0, steps: int = 12) -> wp.array:
    """Rollout 0 driving +x from x = 3 at height y; other rollouts are ignored."""
    p = np.zeros((steps, 2, 3), np.float32)
    p[:, 0, 0] = 3.0 + 0.05 * np.arange(steps)
    p[:, 0, 1] = y
    p[:, 0, 2] = yaw
    return wp.array(p, dtype=wp.vec3f)


def _gov() -> ClearanceGovernor:
    return ClearanceGovernor(ROBOT, plan_dt=0.1, t_react=0.5, v_min=0.15, lookahead_s=1.0)


def test_clearance_is_the_footprint_distance_to_the_wall_face():
    elev, meas, grid = _scene(wall_y=5.0)
    side = ROBOT.half_track + 0.5 * ROBOT.wheel_width
    gov = _gov()
    gov.cap(1.0, 1.0, _plan(y=4.0), elev, meas, grid)
    # the wall face is the first wall cell, centred half a cell past y = 5.0
    assert gov.clearance == pytest.approx(5.0 + 0.5 * CELL - (4.0 + side), abs=CELL)


def test_open_ground_is_never_slowed():
    elev, meas, grid = _scene(wall_y=7.9)
    gov = _gov()
    assert gov.cap(4.0, 4.0, _plan(y=2.0), elev, meas, grid) == (4.0, 4.0)


def test_a_close_wall_scales_both_wheels_to_the_law():
    elev, meas, grid = _scene(wall_y=4.8)
    gov = _gov()
    wl, wr = gov.cap(4.0, 3.0, _plan(y=4.0), elev, meas, grid)
    assert wl / 4.0 == pytest.approx(wr / 3.0)  # one factor: the curvature is kept
    r, b, tail = ROBOT.wheel_radius, ROBOT.half_track, ROBOT.rear_offset + ROBOT.wheel_radius
    fastest = r * 0.5 * abs(wl + wr) + tail * r * abs(wr - wl) / (2.0 * b)
    assert fastest == pytest.approx(speed_law(gov.clearance, 0.5, 0.15), rel=1e-4)


def test_a_pivot_is_slowed_by_its_tail_swing():
    elev, meas, grid = _scene(wall_y=4.8)
    gov = _gov()
    wl, wr = gov.cap(-2.0, 2.0, _plan(y=4.0), elev, meas, grid)  # no forward speed at all
    assert abs(wr) < 2.0 and wl == pytest.approx(-wr)
