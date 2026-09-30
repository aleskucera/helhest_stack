"""The clearance speed governor: the footprint's distance to a wall face along the plan, and the
one-factor slowdown that makes the body's fastest point obey v = clearance / t_react."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import warp as wp

from helhest.control.governor import ClearanceGovernor
from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.planning.clearance import clearance_map_kernel
from helhest.planning.clearance import ClearanceParams

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
    return ClearanceGovernor(ROBOT, plan_dt=0.1, params=ClearanceParams(t_react=0.5, t_turn=0.5))


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
    assert fastest == pytest.approx(gov.params.allowed(gov.clearance), rel=1e-4)


def test_a_pivot_is_slowed_by_its_tail_swing():
    elev, meas, grid = _scene(wall_y=4.8)
    gov = _gov()
    wl, wr = gov.cap(-2.0, 2.0, _plan(y=4.0), elev, meas, grid)  # no forward speed at all
    assert abs(wr) < 2.0 and wl == pytest.approx(-wr)


def test_the_clearance_map_is_the_distance_to_the_wall_face():
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
    plain = ClearanceGovernor(ROBOT, plan_dt=0.1, params=ClearanceParams(t_turn=0.125))
    turn = ClearanceGovernor(ROBOT, plan_dt=0.1, params=ClearanceParams(t_turn=0.25))
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
    gov = ClearanceGovernor(ROBOT, plan_dt=0.1, params=ClearanceParams(v_blind=0.3))
    straight = _plan(y=4.0)  # drives +x: the tail follows over ground that was under the body
    assert gov.cap(3.0, 3.0, straight, elev, meas, grid) == (3.0, 3.0) and not gov.blind
    p = np.zeros((12, 2, 3), np.float32)
    p[:, 0, 0], p[:, 0, 1] = 3.0, 4.0
    p[:, 0, 2] = np.linspace(0.0, 0.8, 12)  # turns left on the spot: the tail swings right and down
    wl, wr = gov.cap(-2.0, 2.0, wp.array(p, dtype=wp.vec3f), elev, meas, grid)
    assert gov.blind and abs(wr) < 2.0


def _clearance_at(elev: np.ndarray, meas: np.ndarray, y: float) -> float:
    out = wp.zeros((N, N), dtype=wp.float32)
    wp.launch(
        clearance_map_kernel,
        dim=(N, N),
        inputs=[wp.array(elev), wp.array(meas), CELL, 0.35, 40],
        outputs=[out],
    )
    return float(out.numpy()[int(y / CELL), N // 2])


def test_a_wall_whose_foot_was_never_measured_is_still_a_face():
    """The map handed in is the inpainted one: the unseen strip in front of the wall is filled at
    ground height, and the wall top rises above that fill. Requiring a MEASURED low side hid it."""
    h = np.zeros((N, N), np.float32)
    ys = (np.arange(N) + 0.5) * CELL
    h[ys >= 5.0, :] = 1.0
    m = np.ones((N, N), np.float32)
    m[(ys >= 4.8) & (ys < 5.0), :] = 0.0  # the wall's foot, never seen; the fill put it at 0
    face_row = int(5.0 / CELL)
    assert _clearance_at(h, m, 4.0) == pytest.approx((face_row - int(4.0 / CELL)) * CELL, abs=1e-5)


def test_a_filled_plateau_is_never_a_face():
    """Beyond a wall the fill can form a wall-height plateau that ends in a step onto measured
    floor. Nobody saw a wall there, so only a MEASURED cell may be a face."""
    h = np.zeros((N, N), np.float32)
    ys = (np.arange(N) + 0.5) * CELL
    m = np.ones((N, N), np.float32)
    h[ys >= 5.0, :] = 1.0
    m[ys >= 5.0, :] = 0.0  # the plateau is fill, not measurement
    assert _clearance_at(h, m, 4.0) == pytest.approx(40 * CELL)  # nothing within reach


def test_braking_that_ramps_in_under_the_jerk_limit_sheds_less_early():
    """With the output jerk-limited, braking takes decel/jerk to build up: the far-end-tight plan
    above must now be slowed, where braking at once would not."""
    elev, meas, grid = _scene(wall_y=5.0)
    steps = 11
    p = np.zeros((steps, 2, 3), np.float32)
    p[:, 0, 0] = 3.0
    p[:, 0, 1] = np.linspace(3.0, 4.63, steps)
    p[:, 0, 2] = np.pi / 2
    plan = wp.array(p, dtype=wp.vec3f)
    at_once = ClearanceGovernor(ROBOT, 0.1, ClearanceParams(t_react=0.5, t_turn=0.5, decel=1.0))
    ramped = ClearanceGovernor(
        ROBOT, 0.1, ClearanceParams(t_react=0.5, t_turn=0.5, decel=1.0, wheel_jerk=5.0)
    )
    fast = at_once.cap(4.0, 4.0, plan, elev, meas, grid)
    slow = ramped.cap(4.0, 4.0, plan, elev, meas, grid)
    assert ramped.v_cap < at_once.v_cap
    # speed shed by t: 0.5*j*t^2 inside the ramp, decel*(t - ramp/2) after it
    t = np.array([0.3, 1.0])
    jerk = 5.0 * ROBOT.wheel_radius
    ramp = 1.0 / jerk
    expect = np.where(t < ramp, 0.5 * jerk * t * t, 1.0 * (t - 0.5 * ramp))
    assert np.allclose(ramped._shed(t), expect)
    assert slow[0] <= fast[0]


def _wall_ahead() -> tuple[wp.array, wp.array, object]:
    """Flat ground with a 1 m wall occupying x >= 6.5, across the robot's heading."""
    xs = (np.arange(N) + 0.5) * CELL
    h = np.zeros((N, N), np.float32)
    h[:, xs >= 6.5] = 1.0
    grid = GridParams(N, N, CELL, 0.0, 0.0).build()
    return wp.array(h), wp.array(np.ones((N, N), np.float32)), grid


def _bending_away(steps: int = 26) -> wp.array:
    """Rollout 0 starting at x = 4 heading +x at the wall 2.5 m ahead, and bending left away --
    the plan that keeps promising a swerve the robot then does not make."""
    p = np.zeros((steps, 2, 3), np.float32)
    s = 0.2 * np.arange(steps)  # 2 m/s at plan_dt 0.1
    yaw = np.minimum(s / 0.6, np.pi / 2)  # 90 deg over the first ~0.9 m
    p[:, 0, 0] = 4.0 + np.cumsum(np.r_[0.0, 0.2 * np.cos(yaw[:-1])])
    p[:, 0, 1] = 4.0 + np.cumsum(np.r_[0.0, 0.2 * np.sin(yaw[:-1])])
    p[:, 0, 2] = yaw
    return wp.array(p, dtype=wp.vec3f)


def _gov_straight(straight: bool) -> ClearanceGovernor:
    params = ClearanceParams(t_react=0.08, t_turn=0.16, lookahead_s=2.5, decel=1.0)
    return ClearanceGovernor(
        ROBOT, plan_dt=0.1, params=dataclasses.replace(params, straight=straight)
    )


def test_carrying_straight_on_slows_a_plan_that_promises_to_bend_away():
    elev, meas, grid = _wall_ahead()
    w = 2.0 / ROBOT.wheel_radius  # 2 m/s
    plan_only = _gov_straight(False).cap(w, w, _bending_away(), elev, meas, grid)
    gov = _gov_straight(True)
    capped = gov.cap(w, w, _bending_away(), elev, meas, grid)
    assert plan_only == (w, w)  # the plan's bend shows room enough for full speed
    assert gov.v_cap_straight < 2.0  # straight on, the wall is 2.5 m ahead
    assert capped[0] == pytest.approx(capped[1])  # forward speed only
    assert ROBOT.wheel_radius * capped[0] == pytest.approx(gov.v_cap_straight, rel=1e-5)


def test_carrying_straight_on_never_caps_turning_away_from_a_wall():
    """Facing a wall, MPPI pivots away. The straight-on line runs into the wall, but that is no
    reason not to turn: capping the turn pinned the robot in drive_sim's pocket for a minute."""
    elev, meas, grid = _wall_ahead()
    pivot = np.zeros((26, 2, 3), np.float32)
    pivot[:, 0, 0], pivot[:, 0, 1] = 6.0, 4.0  # nose 0.15 m from the wall face
    pivot[:, 0, 2] = np.linspace(0.0, np.pi / 2, 26)
    plan = wp.array(pivot, dtype=wp.vec3f)
    plain = _gov_straight(False).cap(-2.0, 2.0, plan, elev, meas, grid)
    straight = _gov_straight(True).cap(-2.0, 2.0, plan, elev, meas, grid)
    assert straight == pytest.approx(plain)


def test_the_straight_on_check_is_off_by_default():
    assert not ClearanceParams().straight


def _flat_with(
    cells: list[tuple[float, float]], height: float
) -> tuple[wp.array, wp.array, object]:
    """Flat ground with single raised cells at the given (x, y) [m]."""
    h = np.zeros((N, N), np.float32)
    for x, y in cells:
        h[int(y / CELL), int(x / CELL)] = height
    grid = GridParams(N, N, CELL, 0.0, 0.0).build()
    return wp.array(h), wp.array(np.ones((N, N), np.float32)), grid


def test_scattered_grass_tufts_are_not_walls():
    # kolecko3: isolated face cells 0.2-0.5 m tall held the robot at v_min for 20 s
    tufts = [(x, y) for x in np.arange(3.0, 4.6, 0.3) for y in (4.5, 4.9, 5.3)]
    elev, meas, grid = _flat_with(tufts, 0.4)
    gov = _gov()
    assert gov.cap(4.0, 4.0, _plan(y=4.0), elev, meas, grid) == (4.0, 4.0)


def test_a_thin_tall_post_is_still_a_wall():
    elev, meas, grid = _flat_with([(3.5, 4.7)], 1.0)
    gov = _gov()
    gov.cap(4.0, 4.0, _plan(y=4.0), elev, meas, grid)
    assert gov.clearance < 0.5


def test_a_low_one_cell_thick_wall_is_still_a_wall():
    kerb = [(x, 4.7) for x in np.arange(2.0, 6.0, CELL)]
    elev, meas, grid = _flat_with(kerb, 0.45)
    gov = _gov()
    gov.cap(4.0, 4.0, _plan(y=4.0), elev, meas, grid)
    assert gov.clearance < 0.5
