"""The path QP (control/qp_path.py): the corridor around a reference, the smoothest path inside it,
and the speed planned along that path."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
from helhest.control.qp_path import extend_route
from helhest.control.qp_path import PathQp
from helhest.control.qp_path import QpPathParams
from helhest.control.qp_path import resample_reference
from helhest.control.qp_path import sharp_turn
from helhest.control.qp_path import trim_at_edge
from helhest.engine import GridParams
from helhest.planning.clearance import clearance_map_kernel

CELL = 0.05
N = 200  # 10 m square, origin at the min corner
DS = 0.15


def _box_scene() -> tuple[wp.array, object]:
    """Flat ground, one 1 m box centred at (5, 5); returns the wall-distance map and its grid."""
    xs = (np.arange(N) + 0.5) * CELL
    XX, YY = np.meshgrid(xs, xs)
    h = np.zeros((N, N), np.float32)
    h[(np.abs(XX - 5.0) <= 0.5) & (np.abs(YY - 5.0) <= 0.5)] = 1.0
    grid = GridParams(N, N, CELL, 0.0, 0.0).build()
    clear = wp.zeros((N, N), dtype=wp.float32)
    ones = wp.array(np.ones((N, N), np.float32))
    wp.launch(
        clearance_map_kernel,
        dim=(N, N),
        inputs=[wp.array(h), ones, CELL, 0.35, 60],
        outputs=[clear],
    )
    return clear, grid


def _qp(**kw) -> PathQp:
    params = QpPathParams(horizon_m=6.5, **kw)
    return PathQp(params, wheel_radius=0.35, half_width=0.42, tail=1.1, head=0.35)


def _straight_through(y: float = 5.0) -> np.ndarray:
    return resample_reference(np.stack([np.linspace(2.0, 9.0, 50), np.full(50, y)], 1), DS, 6.5)


def test_the_corridor_opens_beside_the_box_and_not_through_it():
    clear, grid = _box_scene()
    q = _qp(margin=0.25)
    ref = _straight_through()
    lo, hi = q.corridor(ref, clear, grid)
    at = lambda x: int(round((x - 2.0) / DS))  # noqa: E731
    assert lo[at(2.5)] < 0.0 < hi[at(2.5)]  # far from the box: free both ways
    # beside the box the reference is blocked, so the corridor is the free range nearest it,
    # clear of the face by the half width + margin (0.5 + 0.42 + 0.25), on one side only
    side = lo[at(5.0)] if lo[at(5.0)] > 0.0 else -hi[at(5.0)]
    assert side == pytest.approx(0.5 + 0.42 + 0.25, abs=0.1)


def test_the_path_stays_inside_the_corridor_and_under_the_curvature_limit():
    clear, grid = _box_scene()
    q = _qp(margin=0.25)
    ref = _straight_through()
    lo, hi = q.corridor(ref, clear, grid)
    path = q.solve(ref, lo, hi, e0=0.0, psi0=0.0, v0=1.0, stop_at_end=False)
    assert path is not None
    assert np.all(path.offset >= path.lo - 1e-3) and np.all(path.offset <= path.hi + 1e-3)
    assert np.abs(path.kappa).max() <= q.p.kappa_max + 1e-3
    assert path.offset[0] == pytest.approx(0.0, abs=1e-3)  # starts where the robot is


def test_no_room_anywhere_is_reported_not_guessed():
    q = _qp()
    ref = _straight_through()
    lo, hi = np.full(len(ref), 1.0), np.full(len(ref), -1.0)
    assert q.solve(ref, lo, hi, e0=0.0, psi0=0.0, v0=1.0, stop_at_end=False) is None


def test_the_speed_plan_respects_acceleration_braking_and_a_tight_turn():
    q = _qp()
    n = 40
    kappa = np.zeros(n)
    kappa[25:30] = 1.0  # a 1 m radius bend 3.75 m ahead
    v = q.speed_plan(kappa, np.full(n, 5.0), v0=0.5, stop_at_end=True)
    assert v[0] == pytest.approx(0.5)
    dv2 = np.diff(v**2) / DS
    assert dv2.max() <= 2.0 * q.p.a_acc + 1e-6  # never speeds up faster than allowed
    assert dv2.min() >= -2.0 * q.p.a_dec - 1e-6  # never brakes harder than allowed
    assert v[27] <= np.sqrt(q.p.a_lat / 1.0) + 1e-6  # slow enough in the bend
    assert v[-1] == 0.0


def test_the_tail_swing_term_slows_a_turn_near_a_wall_like_the_governor():
    q = _qp()
    kappa = np.full(30, 0.8)
    slack = np.full(30, 0.4)  # 0.4 m of body clearance: (0.4 - c0) / t_react ~ 3.1 m/s allowed
    straight = q.speed_plan(kappa * 0.0, slack, v0=2.0, stop_at_end=False)
    turning = q.speed_plan(kappa, slack, v0=2.0, stop_at_end=False, tail_swing=2.0 * 1.1)
    assert turning[10:].max() < straight[10:].max()
    allowed = (0.4 - q.p.c0) / q.p.t_react / (1.0 + 2.0 * 1.1 * 0.8)
    assert turning[15] == pytest.approx(min(allowed, np.sqrt(q.p.a_lat / 0.8)), rel=1e-6)


def test_the_route_continues_downhill_on_the_coarse_layer_to_the_horizon():
    # coarse value = distance to a goal 20 m east; the fine route stopped after 5 m
    cell, origin = 0.6, (-6.0, -18.3)  # a row of cell centres lies on y = 0
    centres = (np.arange(60) + 0.5) * cell
    XX, YY = np.meshgrid(origin[0] + centres, origin[1] + centres)
    coarse_v = np.hypot(XX - 20.0, YY)
    fine = np.stack([np.linspace(0.0, 5.0, 26), np.zeros(26)], 1)
    out = extend_route(fine, coarse_v, origin, cell, 8.0)
    length = np.hypot(*np.diff(out, axis=0).T).sum()
    assert 8.0 <= length < 8.0 + 2.0 * cell
    assert np.all(np.diff(out[:, 0]) > 0.0)  # keeps heading for the goal, never folds back
    assert np.abs(out[:, 1]).max() < cell


def test_a_route_already_long_enough_is_left_alone():
    fine = np.stack([np.linspace(0.0, 9.0, 46), np.zeros(46)], 1)
    out = extend_route(fine, np.zeros((10, 10)), (0.0, 0.0), 0.6, 8.0)
    assert out is fine or np.array_equal(out, fine)


def test_the_plan_stops_where_the_corridor_jumps_sides():
    # free to the left of the route, then (a route through an obstacle) only to the right
    q = _qp()
    ref = _straight_through()
    n = len(ref)
    lo, hi = np.full(n, -1.0), np.full(n, 1.0)
    lo[20:25], hi[20:25] = 0.6, 2.0
    lo[25:30], hi[25:30] = -2.0, -0.6
    path = q.solve(ref, lo, hi, e0=0.0, psi0=0.0, v0=1.0, stop_at_end=False)
    assert path is not None and len(path.xy) == 25  # planned up to the jump, not past it
    assert path.v[-1] == 0.0  # and able to stop there


def test_the_fine_route_is_cut_where_it_reaches_the_window_edge():
    # straight to the edge at x = 5, then wandering along it
    wander = np.array([[4.9, 0.2], [4.9, -0.2], [4.8, 0.1], [4.9, -0.1]])
    pts = np.r_[np.stack([np.linspace(0.0, 4.9, 50), np.zeros(50)], 1), wander]
    out = trim_at_edge(pts, (-5.0, -5.0), (5.0, 5.0), 0.4)
    assert out[-1, 0] >= 4.6 and len(out) <= 50  # stops on arrival, before the wandering


def test_the_path_starts_at_the_robot_along_its_heading():
    # the governor checks the robot's footprint along the path's heading: a path that leaves 10 deg
    # off the robot's swings the 1.1 m tail ~0.2 m sideways, into a wall that is not there
    q = _qp()
    stairs = np.array([[0.0, 0.0]] + [[0.24 * (k + 1), 0.24 * (k // 3)] for k in range(40)])
    ref = resample_reference(stairs, DS, 6.5)
    assert np.allclose(ref[0, :2], 0.0, atol=1e-9)
    # arc-length spacing; a chord across a smoothed corner is a fraction of a mm shorter
    assert np.allclose(np.hypot(*np.diff(ref[:, :2], axis=0).T), DS, atol=1e-3)
    for psi in (0.0, -0.35, 0.35):
        n = len(ref)
        path = q.solve(ref, np.full(n, -3.0), np.full(n, 3.0), 0.0, psi, 1.0, False)
        assert abs(path.heading[0] - (ref[0, 2] + psi)) < np.radians(2.0)


def test_the_path_is_never_shifted_past_a_bend_s_radius():
    # a drivable 0.6 m-radius left bend, and a corridor that wants the path 0.7 m to its inside:
    # shifted that far the path would loop (offset past the radius); it must stop short instead
    q = _qp()
    a = np.linspace(0.0, np.pi / 2, 30)
    bend = np.r_[
        np.stack([np.linspace(0.0, 3.0, 21), np.zeros(21)], 1),
        np.stack([3.0 + 0.6 * np.sin(a[1:]), 0.6 - 0.6 * np.cos(a[1:])], 1),
        np.stack([np.full(15, 3.6), np.linspace(0.75, 3.0, 15)], 1),
    ]
    ref = resample_reference(bend, DS, 6.0)
    n = len(ref)
    lo, hi = np.full(n, 0.7), np.full(n, 3.0)
    lo[:8] = -0.05  # room to get there
    path = q.solve(ref, lo, hi, 0.0, 0.0, 1.0, False)
    assert path is not None
    kr = ref[: len(path.offset), 3]
    radius = 1.0 / np.maximum(np.abs(kr), 1e-6)
    inside = path.offset * np.sign(kr)  # > 0: shifted toward the bend's centre
    assert np.all(inside <= radius - 1.0 / q.p.kappa_max + 1e-6)  # never tighter than a turn
    assert np.all(np.abs(np.diff(path.heading)) < np.radians(60.0))  # no segment doubles back


def test_the_route_ends_at_the_goal_not_at_its_coarse_cell():
    cell, origin = 0.6, (-6.0, -18.3)
    centres = (np.arange(60) + 0.5) * cell
    XX, YY = np.meshgrid(origin[0] + centres, origin[1] + centres)
    coarse_v = np.hypot(XX - 6.2, YY - 0.1)  # the goal sits off its cell's centre
    fine = np.stack([np.linspace(0.0, 4.0, 21), np.zeros(21)], 1)
    out = extend_route(fine, coarse_v, origin, cell, 8.0, goal=(6.2, 0.1))
    assert np.allclose(out[-1], (6.2, 0.1))
    assert np.hypot(*(out - (6.2, 0.1)).T).min() < 1e-9 and out[:, 0].max() <= 6.2 + 1e-9


def test_in_a_passage_the_path_keeps_off_the_wall_the_route_grazes():
    # a passage whose walls sit 0.0 and 1.2 m left of the route: the route grazes one wall, the
    # path should keep comfort_m (0.3 m) beyond the hard margin from it
    q = _qp()
    ref = _straight_through()
    n = len(ref)
    lo, hi = np.full(n, -3.0), np.full(n, 3.0)
    lo[15:30], hi[15:30] = 0.0, 1.2
    path = q.solve(ref, lo, hi, 0.0, 0.0, 1.0, False)
    assert path is not None
    assert path.offset[22] > 0.25


def test_where_no_comfort_fits_the_path_takes_the_middle():
    q = _qp()
    ref = _straight_through()
    n = len(ref)
    lo, hi = np.full(n, -3.0), np.full(n, 3.0)
    lo[15:30], hi[15:30] = 0.0, 0.4  # 0.4 m of room: less than twice comfort_m
    path = q.solve(ref, lo, hi, 0.0, 0.0, 1.0, False)
    assert path is not None
    assert abs(path.offset[22] - 0.2) < 0.06


def test_in_open_ground_the_comfort_pull_changes_nothing():
    q, plain = _qp(), _qp(w_comfort=0.0)
    ref = _straight_through()
    n = len(ref)
    lo, hi = np.full(n, -3.0), np.full(n, 3.0)
    a = q.solve(ref, lo, hi, 0.1, 0.1, 1.0, False)
    b = plain.solve(ref, lo, hi, 0.1, 0.1, 1.0, False)
    assert np.allclose(a.offset, b.offset, atol=1e-6)


def test_a_pivot_in_the_route_is_found_and_an_arc_is_not():
    # 3 m east, then straight north: the lattice's pivot, after smoothing still a ~90 deg corner
    corner = np.r_[
        np.stack([np.linspace(0.0, 3.0, 21), np.zeros(21)], 1),
        np.stack([np.full(20, 3.0), np.linspace(0.15, 3.0, 20)], 1),
    ]
    ref = resample_reference(corner, DS, 6.0)
    j = sharp_turn(ref, DS, 0.6, 60.0)
    assert j is not None and 2.0 < j * DS < 3.0  # found just before the corner
    a = np.linspace(0.0, np.pi / 2, 60)
    arc = np.stack([2.0 * np.sin(a), 2.0 - 2.0 * np.cos(a)], 1)  # 2 m radius: an arc, not a pivot
    assert sharp_turn(resample_reference(arc, DS, 6.0), DS, 0.6, 60.0) is None


def test_turning_on_the_spot_stops_first_then_turns_the_way_it_is_told():
    q = _qp()
    assert q.pivot(1.0, 0.8, 1.0, 0.365, 0)[:2] == (0.0, 0.0)  # still moving: stop first
    wl, wr, d = q.pivot(1.0, 0.0, 1.0, 0.365, 0)
    assert wr > 0.0 > wl and wr == -wl and d == 1  # no side given: the shorter way, left
    wl, wr, d = q.pivot(1.0, 0.0, 1.0, 0.365, -1)
    assert wl > 0.0 > wr and d == -1  # told right: the long way round
    sweep = q.pivot_sweep(0.0, 0.0, 0.0, 1.0, -1)
    assert np.isclose(sweep[-1, 2], -(2.0 * np.pi - 1.0))  # ends at the target, going right
