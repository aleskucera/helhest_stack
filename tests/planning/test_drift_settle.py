"""Pose drift reaching the settle producer's margins, through the footprint SPREAD.

The settle differences heights: roll from the two side wheels, pitch from the rear against the
front pair, clearance from the belly against the ground beneath it. Pose drift is one shared
random walk, so it cancels in every one of those differences and only the part accrued since the
older contact was last seen survives -- `|drift_A - drift_B|`, which the footprint's max-min
spread bounds.

That matters because a footprint is not measured all at once: on a real run the age spread across
one is 1.11 s at the median within 5 m of the robot -- the sensor's sparsity, not revisits -- with
a p90 near 88 s wherever the robot crosses its own earlier track.

What the spread COSTS depends on Odin's own drift rate, which `studies/calib/fit_drift.py` puts at
q_z = 7.5e-05 m^2/s, about 1% of the shipped dead-reckoning default. At that rate the median
spread is worth 1.07x on a height difference and a revisit seam is worth 3.4x. A seam correction,
then -- but seams are where a map goes wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams
from helhest.planning.costtogo import CostToGo

N = 41
CELL = 0.24
GOAL = (N * CELL / 2 - CELL, 0.0)
Q_Z = 7.430e-3  # DriftRates.q_z, variance per second


def _grid() -> GridParams:
    return GridParams(
        cells_x=N, cells_y=N, cell_size=CELL, origin_x=-N * CELL / 2, origin_y=-N * CELL / 2
    )


def _terrain(amplitude: float = 0.25) -> wp.array:
    xs = np.linspace(-N * CELL / 2, N * CELL / 2, N)
    t = amplitude * np.sin(xs[None, :] * 0.8) + 0.8 * amplitude * np.cos(xs[:, None] * 0.6)
    return wp.array(t.astype(np.float32), dtype=wp.float32)


def _solve(drift: np.ndarray | None, k_sigma: float = 2.0, sigma_m: float = 0.025):
    ctg = CostToGo(_grid(), RobotParams(), SolverParams(), n_theta=12, k_sigma=k_sigma)
    ctg.compute(
        _terrain(),
        GOAL,
        sigma=wp.array(np.full((N, N), sigma_m, np.float32), dtype=wp.float32),
        drift=None if drift is None else wp.array(drift.astype(np.float32), dtype=wp.float32),
    )
    return ctg


def test_omitting_the_drift_is_the_old_behaviour_exactly():
    a = _solve(None).zmargin.numpy()
    b = _solve(np.zeros((N, N), np.float32)).zmargin.numpy()
    np.testing.assert_allclose(a, b, rtol=1e-5)


def test_a_map_of_one_age_costs_nothing_however_old():
    """The property that lets the robot drive back over ground it measured minutes ago. Shared
    drift cancels in a difference, so a uniformly aged map is judged exactly as a fresh one."""
    fresh = _solve(np.zeros((N, N), np.float32)).zmargin.numpy()
    for age in (1.0, 60.0, 600.0):
        aged = _solve(np.full((N, N), Q_Z * age, np.float32)).zmargin.numpy()
        np.testing.assert_allclose(aged, fresh, rtol=1e-5, err_msg=f"age {age} s changed z")


def test_a_seam_between_two_ages_lowers_the_margin_and_blocks_more():
    """Where old data meets new, the drifts do NOT cancel and the difference is genuinely less
    certain. That is the case `meas_sd` alone is blind to."""
    flat = np.zeros((N, N), np.float32)
    seam = np.zeros((N, N), np.float32)
    seam[:, N // 2 :] = Q_Z * 60.0
    a, b = _solve(flat), _solve(seam)
    za, zb = a.zmargin.numpy(), b.zmargin.numpy()
    assert zb.mean() < za.mean(), "a seam must cost margin somewhere"
    assert b.blocked.numpy().mean() > a.blocked.numpy().mean(), "and veto more poses"


def test_the_margin_falls_monotonically_with_the_size_of_the_seam():
    zs = []
    for age in (0.0, 5.0, 30.0, 120.0):
        d = np.zeros((N, N), np.float32)
        d[:, N // 2 :] = Q_Z * age
        zs.append(float(_solve(d).zmargin.numpy().mean()))
    assert zs == sorted(zs, reverse=True), f"not monotone in seam age: {zs}"


def test_the_optimistic_reading_discounts_the_drift_too():
    """`sigma_scale = 0` is "what would I believe if the map were certain". A map with no
    measurement error but a stale half is not certain, so the discount has to reach the spread
    as well -- otherwise the optimistic solve keeps a pessimism the pair never gives up."""
    seam = np.zeros((N, N), np.float32)
    seam[:, N // 2 :] = Q_Z * 120.0
    ctg = CostToGo(_grid(), RobotParams(), SolverParams(), n_theta=12, k_sigma=2.0)
    sig = wp.array(np.full((N, N), 0.025, np.float32), dtype=wp.float32)
    d = wp.array(seam, dtype=wp.float32)
    ctg.compute(_terrain(), GOAL, sigma=sig, drift=d, sigma_scale=1.0)
    believed = ctg.zmargin.numpy().mean()
    ctg.compute(_terrain(), GOAL, sigma=sig, drift=d, sigma_scale=0.0)
    optimistic = ctg.zmargin.numpy().mean()
    assert optimistic > believed, "the optimistic reading must be the less cautious one"


def test_the_spread_reaches_one_footprint_each_way_and_no_further():
    """The radius is the furthest contact from the pose centre -- the rear wheel -- because that
    is how far apart the heights being differenced actually are."""
    ctg = _solve(np.zeros((N, N), np.float32))
    expect = max(1, int(round(RobotParams().rear_offset / CELL)))
    assert ctg._drift_r == expect
    seam = np.zeros((N, N), np.float32)
    seam[:, N // 2 :] = Q_Z * 60.0
    ctg2 = _solve(seam)
    s = ctg2._spread.numpy()
    band = np.where(s[N // 2] > 0)[0]
    assert band.min() == N // 2 - expect
    assert band.max() == N // 2 + expect - 1
    assert s[:, 0].max() == pytest.approx(0.0), "nothing far from the seam"
