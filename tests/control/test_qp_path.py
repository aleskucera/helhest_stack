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
