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
    assert fastest == pytest.approx(speed_law(gov.clearance, 0.5, 0.15, gov.c0), rel=1e-4)


def test_a_pivot_is_slowed_by_its_tail_swing():
    elev, meas, grid = _scene(wall_y=4.8)
    gov = _gov()
    wl, wr = gov.cap(-2.0, 2.0, _plan(y=4.0), elev, meas, grid)  # no forward speed at all
    assert abs(wr) < 2.0 and wl == pytest.approx(-wr)


def test_the_clearance_map_is_the_distance_to_the_wall_face():
    from helhest.control.governor import clearance_map_kernel

    elev, meas, grid = _scene(wall_y=5.0)
    out = wp.zeros((N, N), dtype=wp.float32)
    wp.launch(clearance_map_kernel, dim=(N, N), inputs=[elev, meas, CELL, 0.35, 40], outputs=[out])
    m = out.numpy()
    face_row = int(5.0 / CELL)  # the first wall cell
    for y in (4.0, 4.5, 4.9):
        r = int(y / CELL)
        assert m[r, N // 2] == pytest.approx((face_row - r) * CELL, abs=1e-5)
    assert m[int(1.0 / CELL), N // 2] == pytest.approx(40 * CELL)  # past the reach


def test_a_plan_tight_only_at_its_far_end_is_not_braked_now():
    """The robot can brake on the way: a tight step a second out only limits the speed that
    braking cannot shed by then. This is the pocket-corner stall."""
    elev, meas, grid = _scene(wall_y=5.0)
    steps = 11
    p = np.zeros((steps, 2, 3), np.float32)
    p[:, 0, 0] = 3.0
    p[:, 0, 1] = np.linspace(3.0, 4.63, steps)  # nose (0.35 m ahead) ends ~0.05 m from the face
    p[:, 0, 2] = np.pi / 2
    gov = _gov()
    wl, wr = gov.cap(4.0, 4.0, wp.array(p, dtype=wp.vec3f), elev, meas, grid)
    assert gov.clearance < 0.1  # the far end really is tight
    assert (wl, wr) == (4.0, 4.0)  # 1.4 m/s now; 1 s of braking at 2 m/s^2 sheds far more


def test_a_longer_turn_allowance_slows_a_pivot_more_but_not_a_straight_drive():
    elev, meas, grid = _scene(wall_y=4.8)
    plain = ClearanceGovernor(ROBOT, plan_dt=0.1, t_react=0.125, v_min=0.15)
    turn = ClearanceGovernor(ROBOT, plan_dt=0.1, t_react=0.125, v_min=0.15, t_turn=0.25)
    assert (
        turn.cap(-2.0, 2.0, _plan(y=4.0), elev, meas, grid)[1]
        < plain.cap(-2.0, 2.0, _plan(y=4.0), elev, meas, grid)[1]
    )
    assert turn.cap(3.0, 3.0, _plan(y=4.0), elev, meas, grid) == plain.cap(
        3.0, 3.0, _plan(y=4.0), elev, meas, grid
    )


def test_sweeping_ground_nobody_has_measured_is_slow_but_driving_onto_seen_ground_is_not():
    """Beside and behind the robot the sensor has never looked. A turn that swings the tail over
    that ground is capped; driving straight onto measured ground ahead is not; the ground under
    the robot now is exempt (it is never measured)."""
    elev, meas, grid = _scene(wall_y=7.9)
    m = np.ones((N, N), np.float32)
    ys = (np.arange(N) + 0.5) * CELL
    xs = (np.arange(N) + 0.5) * CELL
    m[np.ix_(ys < 3.6, xs < 3.2)] = 0.0  # blind: beside, behind and under the robot at (3, 4)
    meas = wp.array(m)
    gov = ClearanceGovernor(ROBOT, plan_dt=0.1, t_react=0.125, v_min=0.15, v_blind=0.3)
    straight = _plan(y=4.0)  # drives +x: the tail follows over ground that was under the body
    assert gov.cap(3.0, 3.0, straight, elev, meas, grid) == (3.0, 3.0) and not gov.blind
    p = np.zeros((12, 2, 3), np.float32)
    p[:, 0, 0], p[:, 0, 1] = 3.0, 4.0
    p[:, 0, 2] = np.linspace(0.0, 0.8, 12)  # turns left on the spot: the tail swings right and down
    wl, wr = gov.cap(-2.0, 2.0, wp.array(p, dtype=wp.vec3f), elev, meas, grid)
    assert gov.blind and abs(wr) < 2.0
