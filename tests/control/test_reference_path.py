"""The online reference path (control/reference_path.py): the curve MPPI tracks, built on the
belief every frame. A 1 m box in the middle of a 10 m window, the robot 5 m west of it, the goal
3.5 m east: the curve must round the box at the asked clearance, within the curvature bounds,
and warm-start from a pose along itself without changing shape."""
from __future__ import annotations

import heapq
import time

import numpy as np
import pytest
import warp as wp
from helhest.control.reference_path import forbidden_distance
from helhest.control.reference_path import ReferencePath
from helhest.control.reference_path import ReferencePathParams
from helhest.engine import GridParams

N, CELL, N_THETA = 125, 0.08, 8
BOX = (6.0, 5.0, 0.5)  # centre x, y, half size
START = np.array([1.0, 5.0, 0.0], np.float32)
GOAL = np.array([9.5, 5.0])
HALF_W, REAR = 0.415, 0.75
# the box is forbidden here exactly to its faces, with none of the growth the settle adds around
# a real one (ReferencePathParams.forbidden_growth_m)
PRM = ReferencePathParams(forbidden_growth_m=0.0)


def _box_distance(p: np.ndarray) -> np.ndarray:
    q = np.abs(p - BOX[:2]) - BOX[2]
    return np.hypot(*np.maximum(q, 0.0).T) + np.minimum(np.maximum(q[:, 0], q[:, 1]), 0.0)


def _route_field(margin: float, goal: np.ndarray = GOAL) -> np.ndarray:
    """A cost-to-go [N, N, N_THETA]: 8-connected metres to `goal`, cells closer than `margin`
    to the box blocked, the same for every heading."""
    xs = (np.arange(N) + 0.5) * CELL
    XX, YY = np.meshgrid(xs, xs)
    free = _box_distance(np.column_stack([XX.ravel(), YY.ravel()])).reshape(N, N) >= margin
    V = np.full((N, N), 90.0, np.float32)
    g = (int(goal[1] / CELL), int(goal[0] / CELL))
    V[g] = 0.0
    heap = [(0.0, g)]
    while heap:
        d, (r, c) = heapq.heappop(heap)
        if d > V[r, c]:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if (dr or dc) and 0 <= rr < N and 0 <= cc < N and free[rr, cc]:
                    nd = d + CELL * (2**0.5 if dr and dc else 1.0)
                    if nd < V[rr, cc]:
                        V[rr, cc] = nd
                        heapq.heappush(heap, (nd, (rr, cc)))
    return np.repeat(V[:, :, None], N_THETA, axis=2)


@pytest.fixture(scope="module")
def scene():
    grid = GridParams(N, N, CELL, 0.0, 0.0).build()
    xs = (np.arange(N) + 0.5) * CELL
    XX, YY = np.meshgrid(xs, xs)
    inside = (np.abs(XX - BOX[0]) <= BOX[2]) & (np.abs(YY - BOX[1]) <= BOX[2])
    # the settle's verdict for a box: forbidden at every heading inside it
    blocked = wp.array(np.repeat(inside[:, :, None], N_THETA, axis=2).astype(np.float32))
    clear = wp.zeros((N, N), dtype=wp.float32)
    forbidden_distance(blocked, N_THETA, CELL, 1.6, wp.zeros((N, N), dtype=wp.float32), clear)
    V = wp.array(_route_field(0.6), dtype=wp.float32)
    return grid, clear, V


def _clearance(curve: np.ndarray) -> np.ndarray:
    """The lesser of the front and rear disc clearances along a curve, wheels' side to the box."""
    d = np.diff(curve, axis=0)
    t = d / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
    rear = curve[:-1] - REAR * t
    return np.minimum(_box_distance(curve[:-1]), _box_distance(rear)) - HALF_W


def _curvature(curve: np.ndarray) -> np.ndarray:
    d = np.diff(curve, axis=0)
    th = np.arctan2(d[:, 1], d[:, 0])
    seg = np.hypot(d[:, 0], d[:, 1])
    dth = (np.diff(th) + np.pi) % (2 * np.pi) - np.pi
    return dth / (0.5 * (seg[:-1] + seg[1:]))


def test_the_curve_rounds_the_box_at_the_asked_clearance_within_the_bounds(scene):
    grid, clear, V = scene
    prm = PRM
    rp = ReferencePath(N_THETA, HALF_W, REAR, prm)
    t0 = time.time()
    n = rp.update(V, grid, clear, grid, START, warm=False)
    wp.synchronize()
    cold_ms = (time.time() - t0) * 1000.0
    assert n >= 10, "no curve"
    curve = rp.curve()
    assert np.allclose(curve[0], START[:2], atol=1e-5)
    assert np.hypot(*(curve[-1] - GOAL)) < 0.3, f"ends {curve[-1]}, not at the goal"
    clr = _clearance(curve)
    assert clr.min() >= prm.clearance_m - 0.1, f"clearance {clr.min():.2f}"
    k = _curvature(curve)
    assert np.abs(k).max() <= prm.kappa_max + 0.05
    # the apex passes the box's face at the asked clearance, not far wider
    apex = np.interp(BOX[0], curve[:, 0], curve[:, 1])
    assert abs(abs(apex - BOX[1]) - (BOX[2] + HALF_W + prm.clearance_m)) < 0.25, f"apex {apex}"
    print(
        f"cold start {cold_ms:.0f} ms (includes compiling), {n} points, ds {rp.ds.numpy()[0]:.3f}"
    )


def test_a_warm_start_from_along_the_curve_keeps_its_shape(scene):
    grid, clear, V = scene
    rp = ReferencePath(N_THETA, HALF_W, REAR, PRM)
    rp.update(V, grid, clear, grid, START, warm=False)
    first = rp.curve()
    # the robot advanced to the 4th point, heading along the curve
    d = first[5] - first[3]
    pose = np.array([first[4, 0], first[4, 1], np.arctan2(d[1], d[0])], np.float32)
    t0 = time.time()
    n = rp.update(V, grid, clear, grid, pose, warm=True)
    wp.synchronize()
    warm_ms = (time.time() - t0) * 1000.0
    assert n >= 8
    second = rp.curve()
    # every point of the new curve lies close to the old one
    dist = np.min(np.linalg.norm(second[:, None, :] - first[None, :, :], axis=2), axis=1)
    assert dist.mean() < 0.1 and dist.max() < 0.3, f"warm start wandered: {dist.max():.2f}"
    print(f"warm update {warm_ms:.1f} ms")


def test_no_descent_means_no_curve(scene):
    grid, clear, _ = scene
    flat = wp.array(np.full((N, N, N_THETA), 90.0, np.float32))  # V capped everywhere
    rp = ReferencePath(N_THETA, HALF_W, REAR, PRM)
    assert rp.update(flat, grid, clear, grid, START, warm=False) == 0


def test_an_end_that_runs_ahead_keeps_the_warm_start(scene):
    """The window scrolls and the end of the string runs on along the same corridor: that is no
    new way out, and restarting there threw away a converged curve at four times the steps."""
    grid, clear, _ = scene
    rp = ReferencePath(N_THETA, HALF_W, REAR, PRM)
    near = wp.array(_route_field(0.6, np.array([8.0, 5.0])), dtype=wp.float32)
    assert rp.update(near, grid, clear, grid, START, warm=False) >= 10
    far = wp.array(_route_field(0.6, np.array([9.7, 5.0])), dtype=wp.float32)
    assert rp.update(far, grid, clear, grid, START) >= 10
    assert int(rp.cold.numpy()[0]) == 0, "the end ran ahead on the same corridor, and it restarted"


def test_a_new_way_out_restarts(scene):
    """A string that leaves last frame's curve is a new homotopy: the old profile is no start."""
    grid, clear, V = scene
    rp = ReferencePath(N_THETA, HALF_W, REAR, PRM)
    assert rp.update(V, grid, clear, grid, START, warm=False) >= 10
    north = wp.array(_route_field(0.6, np.array([5.0, 9.0])), dtype=wp.float32)
    assert rp.update(north, grid, clear, grid, START) >= 3
    assert int(rp.cold.numpy()[0]) == 1, "a new way out kept the old profile"
