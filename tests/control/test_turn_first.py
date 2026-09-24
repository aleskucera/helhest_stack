"""The turn-first brake: a large turn is made in place, not as an arc that advances."""

from __future__ import annotations

import math

import numpy as np
import pytest

from helhest.control.command import turn_first


def test_a_small_heading_error_changes_nothing():
    assert turn_first(3.0, 5.0, math.radians(30.0)) == (3.0, 5.0)


def test_a_turn_behind_keeps_the_differential_and_drops_the_advance():
    wl, wr = turn_first(0.0, 5.9, math.radians(150.0), min_scale=0.1)
    # the planner's arc: mean 2.95, differential 5.9; braked: mean 0.295, the same differential
    assert wr - wl == pytest.approx(5.9)
    assert 0.5 * (wl + wr) == pytest.approx(0.295)
    assert wl < 0.0, "at a full brake the inner wheel runs backward: a spin, not an arc"


def test_the_brake_ramps_between_start_and_full():
    means = [0.5 * sum(turn_first(4.0, 4.0, math.radians(e))) for e in (45.0, 77.5, 110.0, 170.0)]
    np.testing.assert_allclose(means, [4.0, 4.0 * 0.55, 0.4, 0.4], rtol=1e-6)


def test_the_sign_of_the_error_does_not_matter():
    assert turn_first(2.0, 6.0, math.radians(120.0)) == turn_first(2.0, 6.0, math.radians(-120.0))


def test_a_spin_under_way_keeps_its_direction_while_the_way_is_behind():
    # last frame spun left (wr > wl); the planner now proposes right, with the route at 170 deg
    wl, wr = turn_first(2.6, -2.6, math.radians(170.0), prev_diff=5.0)
    assert wr > wl, "the tie was re-decided; the spin must keep its direction"
    assert wr - wl == pytest.approx(5.2)  # the planner's magnitude, the previous sign


def test_the_commitment_lets_go_once_the_way_is_off_to_one_side():
    wl, wr = turn_first(2.6, -2.6, math.radians(60.0), prev_diff=5.0)
    assert wr < wl, "at 60 deg the planner's choice is unambiguous and wins"


def test_no_previous_spin_means_no_commitment():
    wl, wr = turn_first(2.6, -2.6, math.radians(170.0), prev_diff=None)
    assert wr < wl
    wl, wr = turn_first(2.6, -2.6, math.radians(170.0), prev_diff=0.3)  # last frame drove straight
    assert wr < wl
