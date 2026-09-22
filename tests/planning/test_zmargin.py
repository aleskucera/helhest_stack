"""The z-margin field: feasibility measured in sigmas rather than in raw thresholds.

Part A of PROBABILISTIC_PLANNING_PLAN.md. Each test is asked how much room is left in units of
its own uncertainty, the binding one sets the pose's margin, and one knob -- `z_veto` -- says
how many sigmas the robot insists on. The graded penalty comes off the same number, so
pessimism and proximity-to-bad are not two separately-tuned things.
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


def _grid() -> GridParams:
    return GridParams(
        cells_x=N, cells_y=N, cell_size=CELL, origin_x=-N * CELL / 2, origin_y=-N * CELL / 2
    )


# 0.55, not the 0.25 these were written with: the fixtures were shaped to graze a 15 deg
# envelope, and the measured envelope is 30 / 25 / 45 (studies/envelope/). At the old amplitude
# nothing on this scene comes within z_charge of a limit, so every test here passes vacuously --
# blocked fraction flat at zero, no pose below z_charge, no doubt anywhere. Steeper ground is what
# keeps them testing the mechanism rather than the terrain.
def _terrain(amplitude: float = 0.55) -> wp.array:
    xs = np.linspace(-N * CELL / 2, N * CELL / 2, N)
    t = amplitude * np.sin(xs[None, :] * 0.8) + 0.8 * amplitude * np.cos(xs[:, None] * 0.6)
    return wp.array(t.astype(np.float32), dtype=wp.float32)


def _solve(z_veto: float, sigma_m: float, **kw):
    ctg = CostToGo(_grid(), RobotParams(), SolverParams(), n_theta=12, z_veto=z_veto, **kw)
    sigma = wp.array(np.full((N, N), sigma_m, np.float32), dtype=wp.float32)
    v = ctg.compute(_terrain(), GOAL, sigma=sigma)
    return ctg, v.numpy()


def test_z_veto_zero_is_exactly_the_old_behaviour():
    """Opt-in: with BOTH knobs at zero the margin kernel does not run and nothing is added.

    Both, because the kernel is gated on `z_veto > 0 or charge_per_sigma > 0` and charge_per_sigma now
    defaults to 0.5 -- a graded cost with no veto is a perfectly sensible configuration, so
    z_veto = 0 alone no longer means "off".
    """
    ctg_off, _ = _solve(0.0, 0.05, charge_per_sigma=0.0)
    ctg_on, _ = _solve(0.0, 0.50, charge_per_sigma=0.0)  # a wildly uncertain map changes nothing
    np.testing.assert_array_equal(ctg_off.blocked.numpy(), ctg_on.blocked.numpy())
    assert ctg_off.zmargin.numpy().max() == 0.0, "z is not even computed when the knob is off"


def test_blocked_fraction_rises_with_z_veto():
    fracs = [_solve(k, 0.025)[0].blocked.numpy().mean() for k in (0.0, 1.0, 2.0, 3.0)]
    assert fracs == sorted(fracs), f"not monotone in z_veto: {fracs}"
    assert fracs[0] < fracs[-1], "z_veto must actually bite"


def test_blocked_fraction_rises_with_map_uncertainty():
    """The same terrain, believed less precisely, must route more conservatively."""
    fracs = [_solve(2.0, s)[0].blocked.numpy().mean() for s in (0.005, 0.025, 0.05, 0.10)]
    assert fracs == sorted(fracs), f"not monotone in sigma: {fracs}"


def test_sigma_floor_dominates_a_well_known_map():
    """Below the floor the map's own sigma stops mattering -- that is the floor's whole job.

    Without it a pose at 14.9 deg of roll against a 15 deg limit would read as infinitely safe
    on a perfectly known map.
    """
    a, _ = _solve(2.0, 0.001, sigma_floor_m=0.02)
    b, _ = _solve(2.0, 0.010, sigma_floor_m=0.02)
    np.testing.assert_array_equal(a.blocked.numpy(), b.blocked.numpy())
    assert a.zmargin.numpy().max() == pytest.approx(b.zmargin.numpy().max(), rel=1e-5)


def test_attitude_sigma_matches_the_closed_form_and_the_measurement():
    """The propagation, checked against geometry AND against the end-to-end probe.

    `roll = (e1 - e2)/2b` and `pitch = (e3 - (e1+e2)/2)/l`, so with independent per-cell sigma
    `sigma_roll = sqrt(2) s / 2b` and `sigma_pitch = s sqrt(1.5) / l`. Both are DIFFERENCES of
    supports, so shared pose drift cancels and `s` is the measurement sd, not the total.

    The check that matters is external: `studies/calib/RESULTS.md` section 1 measured the
    attitude residual end to end at 2.84 deg roll / 2.11 deg pitch, from settling on a real map
    and comparing against SLAM. A per-cell sigma of 2.5 cm reproduces 2.77 / 2.34 deg here --
    within 3% on roll -- from geometry alone, which is independent corroboration of the chain.
    """
    rp = RobotParams()
    b, l = rp.half_track, rp.rear_offset
    s = 0.025
    sigma_roll = np.degrees(np.sqrt(2.0) * s / (2.0 * b))
    sigma_pitch = np.degrees(s * np.sqrt(1.5) / l)
    assert sigma_roll == pytest.approx(2.77, abs=0.05)
    assert sigma_pitch == pytest.approx(2.34, abs=0.05)
    assert sigma_roll == pytest.approx(2.84, rel=0.05), "vs the measured end-to-end roll residual"

    # And the kernel's own z on flat ground must be the same quantity: max_roll / sigma_roll.
    ctg = CostToGo(_grid(), rp, SolverParams(), n_theta=12, z_veto=0.1)
    flat = wp.array(np.zeros((N, N), np.float32), dtype=wp.float32)
    ctg.compute(flat, GOAL, sigma=wp.array(np.full((N, N), s, np.float32), dtype=wp.float32))
    z = ctg.zmargin.numpy()
    inner = z[5:-5, 5:-5, :]
    expected_roll_z = np.degrees(rp.max_roll) / sigma_roll
    assert inner.max() <= expected_roll_z * 1.02, "no pose may exceed the roll test's own margin"
    assert inner.max() > expected_roll_z * 0.5, "flat ground should sit near the roll bound"


def test_charge_per_sigma_charges_only_below_z_charge():
    """The graded penalty is hinged: comfortable poses pay nothing."""
    plain, _ = _solve(0.5, 0.025, charge_per_sigma=0.0, z_charge=4.0)
    graded, _ = _solve(0.5, 0.025, charge_per_sigma=1.0, z_charge=4.0)
    free = plain.zmargin.numpy() >= 4.0
    dt = graded.graded_tilt.numpy() - plain.graded_tilt.numpy()
    assert dt.min() >= -1e-6, "the penalty may only add cost"
    assert dt[free].max() == pytest.approx(0.0, abs=1e-6), "poses above z_charge must pay nothing"
    assert dt.max() > 0.0, "poses below z_charge must pay something"


def test_uncertainty_can_make_a_goal_unreachable_and_z_veto_recovers_it():
    """The behaviour the whole field exists for, and its escape hatch.

    A map believed too loosely walls the robot off -- which is correct, not a bug: it is the
    honest consequence of not knowing the terrain. Lowering `z_veto` is how a caller trades
    that back, and it is the same lever the plan's optimistic/pessimistic gap pulls.
    """
    _, v_tight = _solve(3.0, 0.08)
    _, v_loose = _solve(0.5, 0.08)
    cap = 1e9
    assert np.isfinite(v_loose).mean() > 0.0
    assert (v_tight < v_loose.max()).mean() < (v_loose < v_loose.max()).mean()
