"""The spin prior's wheel speed: floored at `spin_min`, capped at `spin_max` (0 = wmax), and
turning only the committed way once a spin is under way."""

from __future__ import annotations

import numpy as np
import warp as wp

from helhest.control.mppi import _sample_target_wheel_omega_kernel

T, N_SPIN = 6, 400


def _spin_rates(
    spin_max: float, wmax: float = 6.0, spin_min: float = 2.0, side: float = 0.0
) -> np.ndarray:
    """Right-wheel speed of every spin candidate (> 0 = turning left); candidate 0 is the
    nominal, so the band is 1..N_SPIN."""
    n_cand = 1 + N_SPIN
    out = wp.zeros((T, n_cand), dtype=wp.vec3)
    wp.launch(
        _sample_target_wheel_omega_kernel,
        dim=(T, n_cand),
        inputs=[
            wp.zeros((T, 2), dtype=float),  # nominal U
            0.5,  # sigma
            1.0,  # sigma_knot
            wp.zeros(1, dtype=float),  # wlo: forward only, as on the robot
            wmax,
            n_cand,
            1,  # n_wide: only the nominal comes before the spin band
            0,  # n_straight
            N_SPIN,
            spin_min,
            spin_max,
            0,  # n_pivot
            4,  # n_knots
            wp.array([7], dtype=int),
            wp.array([side], dtype=float),  # spin_side
        ],
        outputs=[out],
    )
    u = out.numpy()[:, 1:, :2]
    assert np.allclose(u[..., 0], -u[..., 1]), "a spin is a pure differential"
    return u[..., 1]


def _spin_speeds(spin_max: float, wmax: float = 6.0, spin_min: float = 2.0) -> np.ndarray:
    return np.abs(_spin_rates(spin_max, wmax, spin_min))


def test_spin_speed_stays_in_the_band() -> None:
    w = _spin_speeds(spin_max=4.0)
    assert w.min() >= 2.0 - 1e-6 and w.max() <= 4.0 + 1e-6
    assert w.max() > 3.8, "the band is sampled up to its ceiling"


def test_zero_means_up_to_wmax() -> None:
    w = _spin_speeds(spin_max=0.0)
    assert w.max() > 5.8 and w.max() <= 6.0 + 1e-6


def test_a_ceiling_below_the_floor_holds_the_floor() -> None:
    w = _spin_speeds(spin_max=1.0)
    assert np.allclose(w, 2.0, atol=1e-6)


def test_uncommitted_spins_go_both_ways() -> None:
    w = _spin_rates(spin_max=4.0)
    assert (w > 0).any() and (w < 0).any()


def test_a_committed_spin_keeps_its_side_and_band() -> None:
    for side in (1.0, -1.0):
        w = _spin_rates(spin_max=4.0, side=side)
        assert np.all(np.sign(w) == side)
        assert np.abs(w).min() >= 2.0 - 1e-6 and np.abs(w).max() <= 4.0 + 1e-6
