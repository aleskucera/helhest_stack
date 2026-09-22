"""Part B: the doubt field and the optimistic/pessimistic gap.

PROBABILISTIC_PLANNING_PLAN.md sections 4.2 and 4.3. A pessimistic planner never explores,
because not knowing is expensive; an optimistic one always does, because not knowing is free.
Solving both and taking the difference is what makes exploration purposeful, and it is the only
one of the three that gives a threshold in the plan cost's own units.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams
from helhest.planning.costtogo import CostToGo

N = 51
CELL = 0.24
START = (-N * CELL / 2 + 1.0, 0.0, 0.0)
GOAL = (N * CELL / 2 - CELL, 0.0)
FLOOR = 0.02


def _ctg(**kw) -> CostToGo:
    grid = GridParams(
        cells_x=N, cells_y=N, cell_size=CELL, origin_x=-N * CELL / 2, origin_y=-N * CELL / 2
    )
    kw.setdefault("k_sigma", 2.0)
    kw.setdefault("sigma_floor_m", FLOOR)
    return CostToGo(grid, RobotParams(), SolverParams(), n_theta=12, **kw)


def _terrain() -> wp.array:
    # Amplitude stays where it was. The envelope doubled to 30 / 25 / 45 (studies/envelope/) and
    # the temptation is to steepen this to compensate -- but measured, the goal stops being
    # reachable at all somewhere between 0.32 and 0.38, so a steeper scene tests nothing. These
    # are tests about IGNORANCE, so what scales with the envelope is the SIGMA they feed, not the
    # ground they stand on.
    xs = np.linspace(-N * CELL / 2, N * CELL / 2, N)
    t = 0.25 * np.sin(xs[None, :] * 0.8) + 0.2 * np.cos(xs[:, None] * 0.6)
    return wp.array(t.astype(np.float32), dtype=wp.float32)


def _sigma(value: float) -> wp.array:
    return wp.array(np.full((N, N), value, np.float32), dtype=wp.float32)


# `far` scales with the envelope: at 30 / 25 / 45 a 6 cm sigma no longer blocks anything on this
# scene, so the frontier it describes would be invisible to the doubt field.
def _frontier_sigma(near: float = 0.01, far: float = 0.14) -> wp.array:
    """Well known behind the robot, uncertain ahead -- the shape a forward sensor produces."""
    s = np.full((N, N), near, np.float32)
    s[:, N // 2 :] = far
    return wp.array(s, dtype=wp.float32)


def test_doubt_separates_ignorance_from_bad_ground():
    """A pose that fails on ANY map is bad terrain; looking at it cannot help."""
    ctg = _ctg()
    ctg.compute(_terrain(), GOAL, sigma=_sigma(0.12))
    doubt = ctg.doubt.numpy()
    blocked = ctg.blocked.numpy() > 0.5
    assert (doubt[~blocked] == 0.0).all(), "an accepted pose carries no doubt"
    assert doubt.max() > 0.0, "an uncertain map must produce some ignorance-blocked poses"


def test_a_certain_map_produces_no_doubt():
    """At the floor the two readings coincide, so nothing is blocked by ignorance."""
    ctg = _ctg()
    ctg.compute(_terrain(), GOAL, sigma=_sigma(FLOOR / 4))
    assert ctg.doubt.numpy().max() == pytest.approx(0.0, abs=1e-6)


def test_optimistic_scale_matches_supplying_no_sigma_at_all():
    """`sigma_scale = 0` is the definition of the optimistic solve: floor everywhere."""
    a, b = _ctg(), _ctg()
    a.compute(_terrain(), GOAL, sigma=_sigma(0.10), sigma_scale=0.0)
    b.compute(_terrain(), GOAL)  # no sigma -> floor everywhere
    np.testing.assert_allclose(a.zmargin.numpy(), b.zmargin.numpy(), rtol=1e-5)


def test_gap_is_zero_when_the_map_is_good_enough():
    """Nothing to gain by looking: the trigger must stay silent."""
    ctg = _ctg()
    ctg.solve_gap(_terrain(), GOAL, _sigma(FLOOR / 2))
    r = ctg.gap_at(*START)
    assert r["gap_m"] == pytest.approx(0.0, abs=1e-4)
    assert r["reachable_pessimistic"] and not r["unreachable_by_ignorance"]


def test_gap_opens_as_the_map_gets_worse():
    gaps = []
    for s in (FLOOR / 2, 0.06, 0.12):
        ctg = _ctg()
        ctg.solve_gap(_terrain(), GOAL, _sigma(s))
        gaps.append(ctg.gap_at(*START)["gap_m"])
    assert gaps == sorted(gaps), f"gap must not shrink as the map worsens: {gaps}"
    assert gaps[-1] > gaps[0]


def test_an_ignorance_blocked_goal_diagnoses_itself():
    """The blind-cell failure, with a defined response instead of "no path".

    Unreachable on the believed map, reachable on a certain one: that is not bad terrain, it is
    not knowing, and the planner can say so rather than reporting no route.
    """
    ctg = _ctg()
    ctg.solve_gap(_terrain(), GOAL, _sigma(0.16))
    r = ctg.gap_at(*START)
    assert not r["reachable_pessimistic"]
    assert r["reachable_optimistic"]
    assert r["unreachable_by_ignorance"]


def test_targets_land_in_the_uncertain_half_not_the_known_one():
    """Decision-focused, not entropy-focused: look where the route needs it."""
    ctg = _ctg()
    ctg.solve_gap(_terrain(), GOAL, _frontier_sigma())
    targets = ctg.doubt_targets(*START)
    assert targets, "a frontier map must offer somewhere to look"
    assert all(t["x"] > 0.0 for t in targets), f"targets strayed into the known half: {targets}"
    assert targets == sorted(targets, key=lambda t: -t["doubt"]), "must rank by doubt"


def test_no_targets_when_nothing_is_doubtful():
    ctg = _ctg()
    ctg.solve_gap(_terrain(), GOAL, _sigma(FLOOR / 4))
    assert ctg.doubt_targets(*START) == []


def test_solve_gap_keeps_both_value_functions():
    """`compute` reuses self.V, so the pessimistic solve must be preserved explicitly."""
    ctg = _ctg()
    ctg.solve_gap(_terrain(), GOAL, _sigma(0.06))
    vp = ctg.V_pessimistic.numpy()
    vo = ctg.V_optimistic.numpy()
    assert not np.allclose(vp, vo), "the two solves must differ on an uncertain map"
    np.testing.assert_allclose(vo, ctg.V.numpy(), rtol=1e-6)  # V holds the last solve run
