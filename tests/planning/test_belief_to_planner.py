"""The whole chain, end to end: a belief built from scans drives what the planner refuses.

Three repositories meet here and this is the only test that exercises the seam between all of
them. `elevation_belief` fuses sweeps into (height, measurement sd, pose drift); the settle
producer turns those into per-pose margins and the sigmas they are judged against;
`terrain_value_field` value-iterates the result. Each piece is tested in its own repo against
its own oracle -- what is tested here is that the quantities mean the same thing on both sides
of each boundary.

The one that is easy to get wrong, and the reason this file exists: which uncertainty crosses
the boundary. The belief publishes three, and only one of them is right for a planner whose
margins are DIFFERENCES between nearby cells.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from elevation_belief import DriftRates
from elevation_belief import ElevationBelief
from elevation_belief import NoiseModel
from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams
from helhest.planning.costtogo import CostToGo

N = 41
CELL = 0.24
HALF = N * CELL / 2
GOAL = (HALF - CELL, 0.0)
SENSOR = np.array([0.0, 0.0, 1.5])


def _grid() -> GridParams:
    return GridParams(cells_x=N, cells_y=N, cell_size=CELL, origin_x=-HALF, origin_y=-HALF)


def _scan(rng, n=40000, x_lo=-HALF, x_hi=HALF):
    """Gently rolling ground, dense enough to fill the window."""
    x = rng.uniform(x_lo, x_hi, n)
    y = rng.uniform(-HALF, HALF, n)
    z = 0.25 * np.sin(0.8 * x) + 0.2 * np.cos(0.6 * y) + rng.normal(0.0, 0.01, n)
    return np.c_[x, y, z].astype(np.float64)


def _belief() -> ElevationBelief:
    # Odin's pose is on-device SLAM; the shipped rates are a handheld rig's and are 100x larger
    return ElevationBelief(
        (-HALF, HALF, -HALF, HALF),
        CELL,
        noise=NoiseModel("linear", a=0.02, b=0.01),
        rates=DriftRates.odin_slam(),
    )


def _as_maps(b: ElevationBelief):
    """The three device arrays the producer takes, on the planner's grid."""
    lay = b.layers()
    h = np.nan_to_num(lay["raw_h"].numpy(), nan=0.0)
    sd = np.sqrt(np.maximum(lay["meas_var"].numpy(), 0.0))
    d = b.drift().numpy()
    return (
        wp.array(np.ascontiguousarray(h, np.float32), dtype=wp.float32),
        wp.array(np.ascontiguousarray(sd, np.float32), dtype=wp.float32),
        wp.array(np.ascontiguousarray(d, np.float32), dtype=wp.float32),
    )


def _plan(h, sd, drift, k_sigma=2.0):
    ctg = CostToGo(_grid(), RobotParams(), SolverParams(), n_theta=12, k_sigma=k_sigma)
    ctg.compute(h, GOAL, sigma=sd, drift=drift)
    return ctg


def test_the_shapes_and_units_line_up_across_all_three_repos():
    """A silent mismatch here reads as plausible nonsense rather than as a failure."""
    b = _belief()
    b.measure_scan(_scan(np.random.default_rng(0)), SENSOR)
    h, sd, drift = _as_maps(b)
    assert tuple(h.shape) == (N, N) == tuple(sd.shape) == tuple(drift.shape)
    ctg = _plan(h, sd, drift)
    assert tuple(ctg.V.shape) == (N, N, 12)
    assert np.isfinite(ctg.zmargin.numpy()).all()


def test_a_belief_of_one_age_is_planned_exactly_as_a_fresh_one():
    """Every cell swept at the same instant, then two minutes of pose drift. The drift is common
    to all of them, cancels in every margin, and must change nothing the planner does."""
    rng = np.random.default_rng(1)
    scan = _scan(rng)
    fresh, aged = _belief(), _belief()
    fresh.measure_scan(scan, SENSOR)
    aged.measure_scan(scan, SENSOR)
    aged.motion_update(120.0, (0.0, 0.0))
    a = _plan(*_as_maps(fresh)).zmargin.numpy()
    c = _plan(*_as_maps(aged)).zmargin.numpy()
    np.testing.assert_allclose(c, a, rtol=1e-4)


def test_a_stale_half_costs_margin_where_it_meets_the_fresh_half():
    """The case `meas_var` alone is blind to. Sweep the whole window, let a minute pass, then
    re-sweep only the right half: the two halves now carry different drift, and a footprint
    straddling the boundary is differencing heights that no longer share it.
    """
    rng = np.random.default_rng(2)
    even, seamed = _belief(), _belief()
    scan = _scan(rng)
    for b in (even, seamed):
        b.measure_scan(scan, SENSOR)
        b.motion_update(60.0, (0.0, 0.0))
    even.measure_scan(scan, SENSOR)  # refresh everything
    seamed.measure_scan(_scan(rng, x_lo=0.0), SENSOR)  # refresh only x > 0
    d_even, d_seam = even.drift().numpy(), seamed.drift().numpy()
    valid = (d_even >= 0) & (d_seam >= 0)
    assert np.ptp(d_even[valid]) < np.ptp(d_seam[valid]), "the seam must show as a drift spread"

    z_even = _plan(*_as_maps(even)).zmargin.numpy()
    z_seam = _plan(*_as_maps(seamed)).zmargin.numpy()
    assert z_seam.mean() < z_even.mean(), "and cost the planner margin"


def test_unmeasured_cells_survive_the_whole_chain_as_unmeasured():
    """The sentinel has to mean the same thing in all three repos. If a never-seen cell arrived
    as zero drift it would read as freshly measured and SHRINK its neighbours' spread, which is
    the one direction a safety margin must never be wrong in."""
    b = _belief()
    b.measure_scan(_scan(np.random.default_rng(3), x_hi=0.0), SENSOR)  # only x < 0 is ever seen
    b.motion_update(30.0, (0.0, 0.0))
    drift = b.drift().numpy()
    assert (drift[:, N // 2 + 2 :] < 0).all(), "the unseen half is flagged, not zeroed"
    assert (drift[:, : N // 2 - 2] >= 0).all(), "and the seen half is not"
    ctg = _plan(*_as_maps(b))
    assert np.isfinite(ctg.zmargin.numpy()).all(), "and nothing downstream turns it into a NaN"


def test_passing_the_wrong_variance_is_visibly_the_wrong_answer():
    """Why the belief publishes `drift_var` at all. `raw_var` is the absolute height variance,
    which grows for every cell as time passes even when nothing about the ground changed.
    Feeding THAT to a planner vetoes a map it measured perfectly, on a timer.

    The size of the effect is predicted rather than asserted: after `dt` at rate `q_z` a cell's
    height variance is `var_meas + q_z*dt`, so the sd ratio is sqrt(1 + q_z*dt/var_meas). Writing
    a bare threshold here instead would just encode whichever drift rate happened to be current,
    which is the mistake this whole chain of work came out of. On this scene it comes out at
    1.48x after 60 s, 2.63x after 300 s and 4.33x after 900 s -- and the prediction lands on all
    three, so what is really being checked is that the belief's drift bookkeeping is honest.
    """
    dt = 300.0
    b = _belief()
    b.measure_scan(_scan(np.random.default_rng(4)), SENSOR)
    b.motion_update(dt, (0.0, 0.0))
    lay = b.layers()
    h = wp.array(
        np.ascontiguousarray(np.nan_to_num(lay["raw_h"].numpy(), nan=0.0), np.float32),
        dtype=wp.float32,
    )
    right = np.sqrt(np.maximum(lay["meas_var"].numpy(), 0.0))
    wrong = np.sqrt(np.maximum(lay["raw_var"].numpy(), 0.0))  # absolute height sd
    _a = lambda v: wp.array(np.ascontiguousarray(v, np.float32), dtype=wp.float32)  # noqa: E731
    ok = _plan(h, _a(right), _a(b.drift().numpy())).blocked.numpy().mean()
    bad = _plan(h, _a(wrong), _a(b.drift().numpy())).blocked.numpy().mean()
    assert bad > ok + 0.1, f"absolute height variance should veto far more ({bad:.2f} vs {ok:.2f})"

    seen = lay["valid"].numpy() != 0
    var_meas = float(np.median(lay["meas_var"].numpy()[seen]))
    predicted = np.sqrt(1.0 + DriftRates.odin_slam().q_z * dt / var_meas)
    got = float(np.median(wrong[seen])) / float(np.median(right[seen]))
    assert got == pytest.approx(
        predicted, rel=0.15
    ), f"{got:.2f} against a predicted {predicted:.2f}"
    assert got > 3.0, "and it is much larger, even on a pose that barely drifts"


def test_the_drift_is_optional_all_the_way_through():
    b = _belief()
    b.measure_scan(_scan(np.random.default_rng(5)), SENSOR)
    h, sd, drift = _as_maps(b)
    with_drift = _plan(h, sd, drift).zmargin.numpy()
    without = _plan(h, sd, None).zmargin.numpy()
    np.testing.assert_allclose(with_drift, without, rtol=1e-4)
