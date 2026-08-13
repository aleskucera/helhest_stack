"""Pose integration against the closed-form arc.

Run:  python -m tests.engine.integrator

On flat ground with constant wheel speeds the body twist is constant, so the exact path is a
circular arc with a closed form. That makes this an EXACTNESS test, not a convergence test: the
engine must land on the analytic arc at every dt, because a composition of exact arcs is exact.

Forward Euler could not do this -- it takes the chord and is first order in dt, which measured
10.9 / 18.7 / 17.7 cm of error at dt = 0.1 s over a 2.5 s horizon at 2.1 m/s (gentle / hard /
tight turn), halving each time dt halved. The second test then confirms the property that
motivated the change: refining dt on real, bumpy terrain no longer moves the trajectory.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams

CELL, CELLS = 0.1, 241
ORIGIN = (-6.0, -12.0)
HORIZON = 2.5  # [s] fixed, so refining dt only changes the number of steps


def _device() -> str:
    return "cuda" if wp.get_cuda_device_count() > 0 else "cpu"


def _rollout(heights: np.ndarray, dt: float, wheel_left: float, wheel_right: float):
    steps = int(round(HORIZON / dt))
    device = _device()
    sim = ForwardSimulator(
        RobotParams(),
        SolverParams(dt=dt),
        GridParams(CELLS, CELLS, CELL, *ORIGIN),
        1,
        steps,
        device=device,
    )
    sim.set_uniform_friction(0.6)
    sim.set_terrain(wp.array(heights, dtype=wp.float32, device=device))
    omega = np.tile(
        np.float32([wheel_left, wheel_right, 0.5 * (wheel_left + wheel_right)]), (steps, 1, 1)
    )
    controlled, derived, *_ = sim.rollout(omega, (0.0, 0.0, 0.0))
    return controlled[:, 0], derived[:, 0], sim.turning.numpy()[0, 0]


def _analytic_arc(wheel_left: float, wheel_right: float, alpha: float, x_icr: float):
    """Closed-form endpoint of a constant twist from the origin, in the engine's own model."""
    rp = RobotParams()
    forward = rp.wheel_radius * (wheel_left + wheel_right) / 2.0
    yaw_rate = rp.wheel_radius * (wheel_right - wheel_left) / (2.0 * rp.half_track * alpha)
    lateral = -x_icr * yaw_rate
    swept = yaw_rate * HORIZON
    x = (forward * np.sin(swept) + lateral * (np.cos(swept) - 1.0)) / yaw_rate
    y = (forward * (1.0 - np.cos(swept)) + lateral * np.sin(swept)) / yaw_rate
    return float(x), float(y), float(swept)


def selftest_arc_exactness() -> None:
    """Flat ground, constant wheel speeds: the engine must sit ON the analytic arc at every dt."""
    flat = np.zeros((CELLS, CELLS), np.float32)
    print(f"{'turn':>16} {'dt [s]':>8} {'engine (x, y)':>22} {'exact (x, y)':>22} {'err [cm]':>9}")
    worst = 0.0
    for name, (left, right) in (
        ("gentle (5,7)", (5.0, 7.0)),
        ("hard (4,8)", (4.0, 8.0)),
        ("tight (2,8)", (2.0, 8.0)),
    ):
        for dt in (0.1, 0.05, 0.0125):
            controlled, _, turning = _rollout(flat, dt, left, right)
            exact_x, exact_y, swept = _analytic_arc(
                left, right, float(turning[0]), float(turning[1])
            )
            end = controlled[-1]
            error = float(np.hypot(end[0] - exact_x, end[1] - exact_y))
            worst = max(worst, error)
            print(
                f"{name:>16} {dt:8.4f} {str(np.round(end[:2], 4)):>22} "
                f"{str(np.round([exact_x, exact_y], 4)):>22} {error * 100:9.3f}"
            )
            assert abs(end[2] - swept) < 1e-4, "yaw must integrate exactly for a constant twist"
    print(f"worst deviation from the analytic arc: {worst * 100:.3f} cm")
    assert worst < 1e-3, "the pose integrator is no longer exact on a constant twist"
    print("arc exactness  OK")


def _bumpy() -> np.ndarray:
    axis_x = ORIGIN[0] + CELL * (np.arange(CELLS) + 0.5)
    axis_y = ORIGIN[1] + CELL * (np.arange(CELLS) + 0.5)
    X, Y = np.meshgrid(axis_x, axis_y)
    rng = np.random.default_rng(3)
    H = sum(
        rng.uniform(-0.12, 0.28)
        * np.exp(
            -((X - rng.uniform(-2, 18)) ** 2 + (Y - rng.uniform(-8, 8)) ** 2)
            / (2.0 * rng.uniform(0.25, 0.8) ** 2)
        )
        for _ in range(60)
    )
    return np.ascontiguousarray(H, np.float32)


def selftest_dt_convergence() -> None:
    """On real terrain the twist changes per step, so exactness is gone -- but dt should barely bite.

    The remaining dt dependence is terrain sampling (the twist is re-derived each step from a new
    pose), not integration error. Refining dt 8x must move the endpoint by centimetres, where the
    Euler version moved by 9-19 cm on the same worlds.
    """
    heights = _bumpy()
    print(f"{'turn':>16} {'dt [s]':>8} {'final (x, y)':>22} {'shift vs dt/2 [cm]':>20}")
    worst = 0.0
    for name, (left, right) in (("straight (6,6)", (6.0, 6.0)), ("turning (5,7)", (5.0, 7.0))):
        previous = None
        for dt in (0.1, 0.05, 0.025, 0.0125):
            controlled, _, _ = _rollout(heights, dt, left, right)
            end = controlled[-1]
            shift = ""
            if previous is not None:
                moved = float(np.hypot(end[0] - previous[0], end[1] - previous[1]))
                worst = max(worst, moved)
                shift = f"{moved * 100:20.3f}"
            print(f"{name:>16} {dt:8.4f} {str(np.round(end[:2], 4)):>22} {shift}")
            previous = end
    print(f"worst endpoint shift from halving dt: {worst * 100:.2f} cm")
    assert worst < 0.05, "dt still moves the trajectory more than 5 cm -- integration or sampling?"
    print("dt convergence  OK")


if __name__ == "__main__":
    wp.init()
    selftest_arc_exactness()
    print()
    selftest_dt_convergence()
