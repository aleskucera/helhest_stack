"""Command transport delay: the wheels act on a command issued `command_delay` seconds ago.

Run:  python -m tests.engine.delay

The real drivetrain responds ~200 ms after a command is issued -- essentially pure delay, with the
first-order lag fitting at 0.02-0.05 s, which is a no-op at dt = 0.1 (see
scripts/fit_actuator_lag.py). `motor_lag_step` cannot express that: it blends toward the CURRENT
target, so it can shape a response but never shift it in time.

The central test is an exact equivalence rather than a plausibility check: running a command
sequence WITH delay n must be bit-identical to running the explicitly time-shifted sequence with
no delay. That pins the indexing and the history convention at once.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams

CELL, CELLS = 0.1, 161
ORIGIN = (-6.0, -8.0)
BATCH, STEPS = 4, 20
DT = 0.1


def _device() -> str:
    return "cuda" if wp.get_cuda_device_count() > 0 else "cpu"


def _terrain() -> np.ndarray:
    axis_x = ORIGIN[0] + CELL * (np.arange(CELLS) + 0.5)
    axis_y = ORIGIN[1] + CELL * (np.arange(CELLS) + 0.5)
    X, Y = np.meshgrid(axis_x, axis_y)
    rng = np.random.default_rng(11)
    H = sum(
        rng.uniform(-0.1, 0.25)
        * np.exp(
            -((X - rng.uniform(-1, 9)) ** 2 + (Y - rng.uniform(-5, 5)) ** 2)
            / (2.0 * rng.uniform(0.3, 0.8) ** 2)
        )
        for _ in range(25)
    )
    return np.ascontiguousarray(H, np.float32)


def _run(commands: np.ndarray, delay_steps: int, history: np.ndarray | None = None):
    """One rollout with `delay_steps` of transport delay; returns (controlled, derived)."""
    device = _device()
    sim = ForwardSimulator(
        RobotParams(),
        SolverParams(dt=DT, command_delay=delay_steps * DT),
        GridParams(CELLS, CELLS, CELL, *ORIGIN),
        BATCH,
        STEPS,
        device=device,
    )
    sim.set_uniform_friction(0.6)
    sim.set_terrain(wp.array(_terrain(), dtype=wp.float32, device=device))
    if history is not None:
        sim.command_history.assign(np.ascontiguousarray(history, np.float32))
    start = np.stack(
        [np.zeros(BATCH), np.linspace(-1.0, 1.0, BATCH), np.zeros(BATCH)], axis=1
    ).astype(np.float32)
    sim.start_pose.assign(start)
    sim.target_wheel_omega.assign(np.ascontiguousarray(commands, np.float32))
    sim.init_current_wheel_omega.zero_()
    sim.rollout_launch()
    return sim.controlled.numpy(), sim.derived.numpy()


def _commands(seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    omega = rng.uniform(0.5, 4.0, (STEPS, BATCH, 3))
    omega[..., 2] = omega[..., :2].mean(-1)
    return np.ascontiguousarray(omega, np.float32)


def selftest_shift_equivalence() -> None:
    """Delay n == running the hand-shifted command sequence with no delay, bit for bit."""
    commands = _commands()
    rng = np.random.default_rng(99)
    worst = 0.0
    for n in (1, 2, 3):
        history = np.ascontiguousarray(rng.uniform(0.5, 4.0, (n, BATCH, 3)), np.float32)
        delayed = _run(commands, n, history)

        shifted = np.empty_like(commands)
        shifted[:n] = history
        shifted[n:] = commands[: STEPS - n]
        plain = _run(shifted, 0)

        diff = max(float(np.abs(a - b).max()) for a, b in zip(delayed, plain))
        worst = max(worst, diff)
        print(f"  delay {n} step(s): max|delayed - hand-shifted| = {diff:.3e}")
        assert diff == 0.0, "the delayed rollout is not the shifted rollout"
    print(f"shift equivalence  OK (worst {worst:.1e})")


def selftest_zero_delay_unchanged() -> None:
    """command_delay = 0 must reproduce the pre-delay engine exactly."""
    commands = _commands()
    a = _run(commands, 0)
    b = _run(commands, 0, history=np.zeros((1, BATCH, 3), np.float32))
    diff = max(float(np.abs(x - y).max()) for x, y in zip(a, b))
    print(f"  zero delay, history written vs not: max|d| = {diff:.3e}")
    assert diff == 0.0, "command_history leaked into the zero-delay path"
    print("zero delay  OK")


def selftest_motion_is_delayed() -> None:
    """With nothing in flight, a standing start must not move for the first n steps."""
    commands = _commands()
    for n in (0, 2, 4):
        history = np.zeros((max(n, 1), BATCH, 3), np.float32)
        controlled, _ = _run(commands, n, history)
        moved = np.abs(controlled[1 : n + 1, :, :2] - controlled[0, :, :2]).max() if n else 0.0
        after = float(np.abs(controlled[n + 1, :, :2] - controlled[0, :, :2]).max())
        print(f"  delay {n}: motion during the delay {moved:.2e} m, first step after {after:.4f} m")
        assert moved == 0.0, "the robot moved while its commands were still in flight"
        assert after > 1e-3, "the robot never started moving"
    print("delayed start  OK")


if __name__ == "__main__":
    wp.init()
    selftest_zero_delay_unchanged()
    print()
    selftest_shift_equivalence()
    print()
    selftest_motion_is_delayed()
