"""The footprint drift spread, and what a producer does with it.

Why this exists at all: a planner's margins are DIFFERENCES between nearby cells, and the drift
two cells share cancels in a difference. Only the part they do not share survives, which is the
drift accrued since the older of them was last seen. Feeding a planner the absolute height
variance vetoes a perfectly measured patch because a minute passed; feeding it the measurement
variance alone assumes every cell under the robot was measured at the same instant.

Measured on a real run, that assumption is wrong by 3.4x in the freshest part of the map -- the
age spread across a footprint is 1.11 s at the median within 5 m of the robot, which is 0.095 m
of sd on a height difference against the 0.028 m measurement alone would claim.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from terrain_value_field import build_grid
from terrain_value_field.drift import footprint_drift_spread
from terrain_value_field.producers import GeometricProducer

N = 41
CELL = 0.1
Q_Z = 7.430e-3  # DriftRates.q_z, variance per second


def _a(x):
    return wp.array(np.ascontiguousarray(x, np.float32), dtype=wp.float32)


# -- the reduction ---------------------------------------------------------------------------


def test_cells_of_one_age_have_no_spread_however_old_they_are():
    """The whole point. A map measured in one sweep five minutes ago is still a good map."""
    for age in (0.0, 1.0, 300.0):
        d = np.full((N, N), Q_Z * age, np.float32)
        assert float(footprint_drift_spread(_a(d), 3).numpy().max()) == pytest.approx(0.0)


def test_a_seam_between_two_ages_shows_the_difference_and_only_near_the_seam():
    d = np.full((N, N), Q_Z * 1.0, np.float32)
    d[:, N // 2 :] = Q_Z * 61.0  # the right half was last seen a minute earlier
    s = footprint_drift_spread(_a(d), 3).numpy()
    assert s[:, N // 2].max() == pytest.approx(
        Q_Z * 60.0, rel=1e-5
    ), "the full age gap, at the seam"
    assert s[:, 0].max() == pytest.approx(0.0), "and nothing far to the left of it"
    assert s[:, -1].max() == pytest.approx(0.0), "or far to the right"
    # the seam lies BETWEEN columns N//2-1 and N//2, so a cell is affected exactly when its
    # +-radius window straddles it: columns N//2-radius .. N//2+radius-1
    reach = np.where(s[N // 2] > 0)[0]
    assert reach.min() == N // 2 - 3 and reach.max() == N // 2 + 3 - 1
    assert len(reach) == 2 * 3, "one footprint radius each way, and no further"


def test_unmeasured_cells_are_skipped_rather_than_read_as_freshly_measured():
    """A negative drift means no height exists there. Counting it as zero would read as a brand
    new measurement and SHRINK the spread exactly where the map is worst."""
    d = np.full((N, N), Q_Z * 50.0, np.float32)
    d[N // 2, N // 2] = -1.0  # a hole in the middle
    s = footprint_drift_spread(_a(d), 3).numpy()
    assert float(s.max()) == pytest.approx(0.0), "one hole among equals is not a spread"


def test_a_cell_with_no_measured_neighbour_at_all_has_no_spread():
    d = np.full((N, N), -1.0, np.float32)
    assert float(footprint_drift_spread(_a(d), 3).numpy().max()) == pytest.approx(0.0)


def test_a_negative_radius_is_rejected():
    with pytest.raises(ValueError, match="radius"):
        footprint_drift_spread(_a(np.zeros((N, N))), -1)


# -- what the producer does with it ----------------------------------------------------------


def _produce(height, sd, drift=None, **kw):
    p = GeometricProducer(N, N, 1, **kw)
    g = build_grid(N, N, CELL, -N * CELL / 2, -N * CELL / 2)
    return p(_a(height), _a(sd), g, drift=None if drift is None else _a(drift))


def test_a_map_of_one_age_is_judged_exactly_as_if_no_drift_were_passed():
    h = np.zeros((N, N), np.float32)
    sd = np.full((N, N), 0.02, np.float32)
    plain = _produce(h, sd).sigma.numpy().copy()
    aged = _produce(h, sd, drift=np.full((N, N), Q_Z * 120.0, np.float32)).sigma.numpy()
    np.testing.assert_allclose(plain, aged, rtol=1e-5)


def test_a_seam_widens_the_sigma_the_margins_are_judged_against():
    h = np.zeros((N, N), np.float32)
    sd = np.full((N, N), 0.02, np.float32)
    d = np.full((N, N), Q_Z * 1.0, np.float32)
    d[:, N // 2 :] = Q_Z * 61.0
    plain = _produce(h, sd).sigma.numpy()
    seamed = _produce(h, sd, drift=d).sigma.numpy()
    at_seam = seamed[:, :, N // 2, 0]
    assert (at_seam > plain[:, :, N // 2, 0] * 1.5).all(), "both constraints widen at the seam"
    np.testing.assert_allclose(seamed[:, :, 0, 0], plain[:, :, 0, 0], rtol=1e-5)


def test_half_the_spread_per_cell_makes_a_PAIR_carry_the_whole_of_it():
    """The reason it is half and not the whole. Both constraints are differences built from two
    cells' sds, so half each sums to the full spread -- the bound on |drift_A - drift_B|. Giving
    each cell the whole spread would double the variance, and since the spread usually dominates
    the measurement term that is a real 1.41x shrink of every margin.
    """
    meas, spread = 0.02, Q_Z * 1.11  # the measured median age spread within 5 m
    inflated_var = meas**2 + 0.5 * spread
    pair_var = 2.0 * inflated_var
    assert pair_var == pytest.approx(2.0 * meas**2 + spread)
    assert math.sqrt(pair_var) == pytest.approx(0.095, abs=0.001), "the measured 0.095 m"
    assert math.sqrt(2.0 * meas**2) == pytest.approx(0.028, abs=0.001), "against 0.028 m"


def test_the_drift_map_must_match_the_grid():
    h = np.zeros((N, N), np.float32)
    sd = np.full((N, N), 0.02, np.float32)
    with pytest.raises(ValueError, match="drift"):
        _produce(h, sd, drift=np.zeros((N, N + 1), np.float32))
