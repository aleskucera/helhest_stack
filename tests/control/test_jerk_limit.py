"""The jerk-limited output tracker: its bounds, that it settles, and that the rollouts run it too."""

from __future__ import annotations

import numpy as np
import warp as wp

from helhest.control.command import condition_command
from helhest.control.command import jerk_limited_step
from helhest.control.mppi import _rate_limit_kernel

ACCEL, JERK = 3.0, 5.0


def _track(targets: np.ndarray, dt: float, v0: float = 0.0, a0: float = 0.0) -> np.ndarray:
    """Commands and accelerations [n, 2] for one joint following `targets`."""
    v, a, out = np.float32(v0), np.float32(a0), []
    for target in targets:
        nv = jerk_limited_step(target, v, a, ACCEL, ACCEL, JERK, dt)
        v, a = nv, (nv - v) / dt
        out.append((v, a))
    return np.asarray(out, np.float64)


def test_bounds_hold_whatever_the_planner_asks():
    """Random target jumps from random moving states: acceleration and jerk never exceed the caps
    (the landing snap is only taken when it is itself within the jerk limit)."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        dt = rng.uniform(0.05, 0.1)
        targets = np.repeat(rng.uniform(-4.0, 4.0, 8), rng.integers(2, 30, 8))
        tr = _track(targets, dt, rng.uniform(-4.0, 4.0), rng.uniform(-ACCEL, ACCEL))
        accel = tr[:, 1]
        assert np.abs(accel).max() <= ACCEL + 1e-4
        assert np.abs(np.diff(accel)).max() / dt <= JERK + 1e-3


def test_a_step_ramps_and_lands_without_overshoot():
    tr = _track(np.full(60, 4.0), dt=0.067)
    assert tr[:, 0].max() <= 4.0 + 5e-3
    assert tr[-1, 0] == np.float32(4.0) and tr[-1, 1] == 0.0
    # full ramp: 0.6 s up to 3 rad/s^2, ~0.7 s at it, 0.6 s down -- about 1.9 s in all
    t_arrive = 0.067 * np.argmax(tr[:, 0] >= 3.96)
    assert 1.6 < t_arrive < 2.1


def test_condition_command_without_jerk_is_the_plain_rate_limit():
    prev = np.array([1.0, 1.0, 1.0], np.float32)
    plain = condition_command(3.0, 3.0, prev, max_omega=10.0, max_slew=3.0, dt=0.1)
    assert np.allclose(plain, prev + 0.3)
    jerky = condition_command(
        3.0, 3.0, prev, max_omega=10.0, max_slew=3.0, dt=0.1, max_jerk=5.0
    )  # from rest in acceleration: only j*dt^2 = 0.05 in the first tick
    assert np.allclose(jerky, prev + 0.05)


def test_the_rollouts_run_the_same_tracker():
    """_rate_limit_kernel must reproduce jerk_limited_step, or the planner scores commands the
    wheels never get."""
    rng = np.random.default_rng(1)
    T, B, dt = 25, 64, 0.1
    raw = rng.uniform(-4.0, 4.0, (T, B, 3)).astype(np.float32)
    state = np.array([[1.5, -2.0, 0.8, -1.2]], np.float32)  # (wL, wR, aL, aR)
    out = wp.zeros((T, B), dtype=wp.vec3, device="cpu")
    wp.launch(
        _rate_limit_kernel,
        dim=B,
        inputs=[
            wp.array(raw, dtype=wp.vec3, device="cpu"),
            wp.array(state, dtype=wp.vec4, device="cpu"),
            ACCEL,
            ACCEL,
            JERK,
            dt,
        ],
        outputs=[out],
        device="cpu",
    )
    got = out.numpy()
    for wheel in (0, 1):
        v = np.full(B, state[0, wheel], np.float32)
        a = np.full(B, state[0, 2 + wheel], np.float32)
        for t in range(T):
            nv = jerk_limited_step(raw[t, :, wheel], v, a, ACCEL, ACCEL, JERK, dt)
            v, a = nv, (nv - v) / dt
            assert np.allclose(got[t, :, wheel], v, atol=1e-4)
