"""Lattice closure, the realized-arc-length cost, the turn surcharge and point turns."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from terrain_value_field import arc_control_set
from terrain_value_field import closing_step
from terrain_value_field import DEFAULT_PIVOT_ARCS
from terrain_value_field.solver import ValueSolver

RES = 0.1
SWEEP = 24
NSEG = 32


def build(n_theta, turn_radius, bins=2, **kw):
    step = closing_step(n_theta, turn_radius, bins)
    return step, arc_control_set(n_theta, RES, step, turn_radius, SWEEP, NSEG, **kw)


def heading_errors(n_theta, turn_radius, step, cs):
    """Recorded heading change minus the one the arc actually turns through, per primitive."""
    head = cs[3]
    dth = 2.0 * math.pi / n_theta
    out = []
    for it in range(n_theta):
        for p, frac in enumerate((-1.0, -0.5, 0.0, 0.5, 1.0)):
            recorded = ((head[it, p] - it + n_theta // 2) % n_theta - n_theta // 2) * dth
            out.append(recorded - frac * step / turn_radius)
    return np.array(out)


def duplicate_count(cs):
    n_prim, dr, dc, head, *_ = cs
    return sum(
        n_prim - len({(int(dr[it, p]), int(dc[it, p]), int(head[it, p])) for p in range(n_prim)})
        for it in range(dr.shape[0])
    )


# -- closure -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_theta,turn_radius,bins",
    [(16, 0.5, 2), (16, 0.5, 4), (32, 0.5, 2), (16, 0.6, 2), (8, 0.5, 2), (32, 1.0, 4)],
)
def test_a_closing_step_records_the_heading_the_arc_reaches(n_theta, turn_radius, bins):
    step, cs = build(n_theta, turn_radius, bins)
    assert np.abs(heading_errors(n_theta, turn_radius, step, cs)).max() < 1e-9


@pytest.mark.parametrize("n_theta,turn_radius,bins", [(16, 0.5, 2), (16, 0.5, 4), (32, 0.5, 2)])
def test_a_closing_step_has_no_duplicate_primitives(n_theta, turn_radius, bins):
    _, cs = build(n_theta, turn_radius, bins)
    assert duplicate_count(cs) == 0


def test_odd_bins_are_rejected_because_the_half_rate_arcs_must_close_too():
    with pytest.raises(ValueError, match="even"):
        closing_step(16, 0.5, bins=3)


def test_a_non_closing_step_warns_and_names_the_nearest_closing_one():
    with pytest.warns(UserWarning, match="does not close") as rec:
        arc_control_set(8, RES, 0.3, 0.5, SWEEP, NSEG)
    assert f"{closing_step(8, 0.5, 2):.4f}" in str(rec[0].message)


def test_the_documented_broken_default_is_as_bad_as_advertised():
    # the combination the module docstring cites: 17.2 deg of heading error, 20% duplicates
    with pytest.warns(UserWarning):
        cs = arc_control_set(8, RES, 0.3, 0.5, SWEEP, NSEG)
    assert math.degrees(np.abs(heading_errors(8, 0.5, 0.3, cs)).max()) == pytest.approx(
        17.2, abs=0.1
    )
    assert duplicate_count(cs) == 8  # of 40


def test_the_solver_default_step_closes():
    # constructing a solver with no explicit step must not trip the closure warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ValueSolver(RES, 32, 32, n_theta=16, turn_radius=0.6)


# -- cost ----------------------------------------------------------------------------------


def length_over_chord(turned: float, samples: int = 200_000) -> float:
    """Arc length / chord for a circular arc of this central angle, by dense polyline
    integration of a unit circle. Independent of the closed form the cost uses."""
    a = np.linspace(0.0, turned, samples)
    x, y = np.cos(a), np.sin(a)
    arc = float(np.hypot(np.diff(x), np.diff(y)).sum())
    return arc / float(math.hypot(x[-1] - x[0], y[-1] - y[0]))


@pytest.mark.parametrize("bins,turn_deg", [(2, 45.0), (4, 90.0)])
def test_cost_is_the_length_of_the_arc_through_the_recorded_endpoint(bins, turn_deg):
    # For any circular arc, length/chord depends only on how far it turns. So the cost divided by
    # the recorded endpoint's distance must equal that ratio -- checked here against a numerically
    # integrated circle, not against the formula the cost is computed with.
    step, cs = build(16, 0.5, bins=bins)
    _, dr, dc, _, cost, *_ = cs
    expected = length_over_chord(math.radians(turn_deg))
    for p in (0, 4):  # the two sharpest arcs, which turn `turn_deg`
        chord = math.hypot(dr[0, p] * RES, dc[0, p] * RES)
        assert cost[0, p] / chord == pytest.approx(expected, rel=1e-4)
        assert cost[0, p] != pytest.approx(step, rel=1e-3)  # NOT the nominal step


def test_the_straight_primitive_costs_the_distance_it_actually_covers():
    _, cs = build(16, 0.5, bins=2)
    _, dr, dc, _, cost, *_ = cs
    for it in range(16):
        chord = math.hypot(dr[it, 2] * RES, dc[it, 2] * RES)
        assert cost[it, 2] == pytest.approx(chord, rel=1e-5)


def test_cost_tracks_the_snapped_endpoint_rather_than_the_nominal_step():
    # the bias this replaced: every arc charged `step` while realized displacement spread ~13%
    step, cs = build(16, 0.5, bins=2)
    _, _, _, _, cost, *_ = cs
    assert cost[0, :5].std() > 0.01 * step


# -- turn surcharge ------------------------------------------------------------------------


def test_the_turn_surcharge_is_turn_weight_per_radian():
    n_theta, turn_radius = 16, 0.5
    step = closing_step(n_theta, turn_radius, 2)
    base = arc_control_set(n_theta, RES, step, turn_radius, SWEEP, NSEG)[4]
    w = 0.7
    charged = arc_control_set(n_theta, RES, step, turn_radius, SWEEP, NSEG, turn_weight=w)[4]
    for it in range(n_theta):
        for p, frac in enumerate((-1.0, -0.5, 0.0, 0.5, 1.0)):
            turned = abs(frac) * step / turn_radius
            assert charged[it, p] - base[it, p] == pytest.approx(w * turned, abs=1e-5)


def test_a_large_enough_turn_weight_makes_the_straight_the_cheapest_move():
    _, cs = build(16, 0.5, bins=2, turn_weight=2.0)
    cost = cs[4]
    for it in range(16):
        assert cost[it, 2] == cost[it].min()  # p2 is the straight


def test_cost_rises_monotonically_with_how_sharply_the_move_turns():
    _, cs = build(16, 0.5, bins=2, turn_weight=2.0)
    cost = cs[4]
    for it in range(16):
        straight, gentle, sharp = cost[it, 2], cost[it, 3], cost[it, 4]
        assert straight < gentle < sharp
        assert cost[it, 1] == pytest.approx(gentle, abs=0.02)  # left/right roughly symmetric


# -- point turns ---------------------------------------------------------------------------


def test_pivots_are_on_by_default():
    _, cs = build(16, 0.5)
    assert cs[0] == 7
    assert cs[4][0, 5] == cs[4][0, 6]  # left and right cost the same


@pytest.mark.parametrize(
    "n_theta,turn_radius,bins",
    [(16, 0.5, 2), (16, 0.5, 4), (32, 0.5, 2), (16, 1.0, 2), (32, 0.75, 4), (8, 0.5, 2)],
)
def test_a_pivot_costs_a_fixed_multiple_of_the_arc_that_turns_as_far(n_theta, turn_radius, bins):
    # the property the default exists to hold: the pivot/arc trade must not drift with the step,
    # the heading resolution or the robot. Pricing off `step` instead doubled it from bins=2 to 4.
    _, cs = build(n_theta, turn_radius, bins)
    arc_per_bin = turn_radius * (2.0 * math.pi / n_theta)  # sharpest arc runs at turn_radius
    assert cs[4][0, 5] / arc_per_bin == pytest.approx(DEFAULT_PIVOT_ARCS)


def test_a_half_turn_by_pivot_costs_that_multiple_of_the_u_turn_arc():
    n_theta, turn_radius = 16, 0.5
    _, cs = build(n_theta, turn_radius, 2)
    by_pivot = cs[4][0, 5] * (n_theta // 2)
    by_arc = math.pi * turn_radius  # half circle at the min turn radius
    assert by_pivot / by_arc == pytest.approx(DEFAULT_PIVOT_ARCS)


def test_an_infinite_pivot_cost_leaves_the_primitives_out_entirely():
    # a robot that cannot turn on the spot should not pay to relax two moves it will never take
    _, cs = build(16, 0.5, pivot_cost=math.inf)
    assert cs[0] == 5


def test_pivots_turn_one_bin_in_place_at_the_price_asked():
    _, cs = build(16, 0.5, pivot_cost=1.5)
    n_prim, dr, dc, head, cost, _, _, _, sweep_n = cs
    assert n_prim == 7
    for it in range(16):
        for p, dbin in ((5, -1), (6, +1)):
            assert (dr[it, p], dc[it, p]) == (0, 0)  # in place
            assert head[it, p] == (it + dbin) % 16
            assert cost[it, p] == pytest.approx(1.5)
            assert sweep_n[it, p] == 1


def test_a_free_pivot_is_rejected_as_a_zero_cost_cycle():
    with pytest.raises(ValueError, match="pivot_cost"):
        build(16, 0.5, pivot_cost=0.0)


def test_a_half_turn_costs_a_pivot_per_bin():
    # the price already scales with angle: n_theta/2 pivots to face the other way
    _, cs = build(16, 0.5, pivot_cost=1.5)
    assert cs[4][0, 5] * (16 // 2) == pytest.approx(12.0)


def test_a_heavy_pivot_outprices_every_arc():
    _, cs = build(16, 0.5, pivot_cost=4.0 * closing_step(16, 0.5, 2))
    cost = cs[4]
    assert cost[0, 5] > cost[0, :5].max()


@pytest.mark.filterwarnings("ignore:arc_control_set")  # it also fails closure; not the point here
def test_a_step_too_small_for_the_cell_is_rejected():
    """Arcs that snap back onto their own state are self-loops: nothing propagates and the solve
    reports the goal unreachable on an open map. A silent wrong answer, so it raises."""
    with pytest.raises(ValueError, match="move nowhere"):
        arc_control_set(16, 0.5, 0.05, 0.6, SWEEP, NSEG)  # step 0.05 m on 0.5 m cells


def test_every_arc_of_a_closing_set_actually_moves():
    for n_theta, turn_radius, bins in [(16, 0.5, 2), (32, 0.5, 2), (8, 0.5, 2), (16, 1.0, 4)]:
        _, cs = build(n_theta, turn_radius, bins)
        dr, dc, head = cs[1], cs[2], cs[3]
        for it in range(n_theta):
            for p in range(5):  # the arcs; pivots legitimately stay put
                assert (dr[it, p], dc[it, p]) != (0, 0) or head[it, p] != it


def test_the_swept_heading_offsets_match_the_arc_that_was_integrated():
    """Re-integrate each arc and check the recorded offset against the heading it is genuinely
    facing when it first enters each cell. Independent of how the table was built."""
    n_theta, turn_radius, nseg = 16, 0.5, 64
    step = closing_step(n_theta, turn_radius, 2)
    cs = arc_control_set(n_theta, RES, step, turn_radius, 40, nseg)
    _, _, _, _, _, sdr, sdc, sdt, sn = cs
    dth = 2.0 * math.pi / n_theta
    for it in (0, 5, 11):
        th0 = (it + 0.5) * dth
        for p, frac in enumerate((-1.0, -0.5, 0.0, 0.5, 1.0)):
            x = y = 0.0
            first = {}
            for s in range(1, nseg):
                cth = th0 + frac * (step / turn_radius) * (s - 0.5) / (nseg - 1)
                x += (step / (nseg - 1)) * math.cos(cth)
                y += (step / (nseg - 1)) * math.sin(cth)
                first.setdefault((round(y / RES), round(x / RES)), cth)
            for s in range(int(sn[it, p])):
                cell = (int(sdr[it, p, s]), int(sdc[it, p, s]))
                want = (int(math.floor((first[cell] % (2 * math.pi)) / dth)) - it) % n_theta
                assert int(sdt[it, p, s]) == want, f"bin {it}, prim {p}, cell {cell}"


def test_a_turning_arc_crosses_cells_at_a_heading_it_did_not_start_in():
    # if this were not so, the offsets would all be 0 and the table would be pointless
    step = closing_step(16, 0.5, 2)
    _, _, _, _, _, _, _, sdt, sn = arc_control_set(16, RES, step, 0.5, 40, 64)
    sharp = [int(sdt[0, 4, s]) for s in range(int(sn[0, 4]))]
    straight = [int(sdt[0, 2, s]) for s in range(int(sn[0, 2]))]
    assert set(straight) == {0}, "a straight arc never leaves its heading bin"
    assert max(sharp) == 2, "the sharpest arc turns two bins over one step"
    assert sum(1 for o in sharp if o != 0) > len(sharp) // 2, "and most of it is spent turning"


def test_an_oversized_sweep_is_rejected_rather_than_truncated():
    """Silently keeping only max_sweep cells would leave holes in the collision check."""
    with pytest.raises(ValueError, match="max_sweep"):
        arc_control_set(16, 0.02, closing_step(16, 0.5, 2), 0.5, 4, 64)
