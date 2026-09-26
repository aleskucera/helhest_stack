"""Explore or push: what ignorance costs, against what it costs to go and resolve it.

Both sides are metres of driving, so the comparison needs no tuned threshold. The gap comes from
solving twice -- believing the map, then as if it were certain -- and is an upper bound on what
any amount of looking could save. The cost of looking comes from a third solve of the same field
seeded on the DOUBTED states instead of the goal.

The scenes below are built so each branch is reached for a different reason, because a decision
rule that happens to be right on one map is not a decision rule.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from helhest.planning.terrain_value_field import omni_control_set
from helhest.planning.terrain_value_field import TerrainValueField
from helhest.planning.terrain_value_field.field import Constraints

N, CELL = 40, 0.5
ROBOT = (20, 3)
GOAL = (20, 36)

CERTAIN_GROUND = (0.9, 0.02)  # (margin, sigma) -> z = 45, far above z_veto
UNCERTAIN = (0.9, 0.9)  # z = 1.0 believed (vetoed), 90 certain -> pure doubt
WALL = (-1.0, 0.02)  # bad ground either way -> no doubt to resolve


def _field() -> TerrainValueField:
    return TerrainValueField(N, N, CELL, n_theta=1, z_veto=2.0, control_set=omni_control_set(CELL))


def _constraints(patches) -> Constraints:
    """`patches` is a list of (row_slice, col_slice, (margin, sigma))."""
    m = np.full((1, N, N, 1), CERTAIN_GROUND[0], np.float32)
    s = np.full((1, N, N, 1), CERTAIN_GROUND[1], np.float32)
    for rs, cs, (mv, sv) in patches:
        m[0, rs, cs, 0] = mv
        s[0, rs, cs, 0] = sv
    return Constraints(
        margin=wp.array(m, dtype=wp.float32),
        sigma=wp.array(s, dtype=wp.float32),
        floor=wp.array([0.01], dtype=wp.float32),
    )


def _ask(patches, **kw) -> dict:
    f = _field()
    f.seed_cell(*GOAL)
    return f.value_of_looking(_constraints(patches), *ROBOT, **kw)


# -- the cases that have no number ----------------------------------------------------------


def test_a_fully_observed_map_has_nothing_to_look_at():
    r = _ask([])
    assert r["doubted_states"] == 0
    assert r["gap"] == pytest.approx(0.0), "nothing is hidden, so nothing is being lost"
    assert r["cost_to_look"] == float("inf"), "and nowhere to go"
    assert not r["worth_looking"]


def test_a_map_walled_off_by_real_ground_has_nothing_to_learn():
    """Bad ground is bad on any reading. The goal is unreachable and looking will not change it."""
    r = _ask([(slice(None), slice(18, 21), WALL)])
    assert r["walled_off"]
    assert not r["blocked_by_ignorance"]
    assert r["doubted_states"] == 0
    assert not r["worth_looking"]


def test_a_route_blocked_only_by_ignorance_is_the_strongest_signal_there_is():
    """Unreachable believing the map, reachable if the doubts go the robot's way. Not knowing is
    costing the ENTIRE route, so the gap is infinite and no detour can fail to be worth it."""
    r = _ask([(slice(None), slice(18, 21), UNCERTAIN)])
    assert r["blocked_by_ignorance"]
    assert not r["walled_off"]
    assert r["gap"] == float("inf")
    assert r["cost_to_look"] < float("inf"), "the doubted band is reachable to look at"
    assert r["worth_looking"]


# -- the case with a real trade ---------------------------------------------------------------


def _gap_scene(col0=8, col1=11):
    """An uncertain band across the map with a clear way round at the very top. Believing the map
    the robot must take the long way; if the band turned out fine it would go straight through.

    The band sits CLOSE to the robot and the way round is far, which is what makes looking the
    better bet here. Put the same band at column 18 with five clear rows and the numbers flip --
    7.5 m to go and look against a 6.2 m detour -- and pushing on is right. The rule has no
    opinion about which of those a map should be; it just reports the two distances.
    """
    return [(slice(0, N - 2), slice(col0, col1), UNCERTAIN)]


def test_a_detour_that_ignorance_forces_shows_up_as_a_finite_gap():
    r = _ask(_gap_scene())
    assert not r["blocked_by_ignorance"] and not r["walled_off"]
    assert r["v"] < float("inf") and r["v_certain"] < r["v"], "the long way round costs more"
    assert 0.0 < r["gap"] < float("inf")
    assert r["gap"] == pytest.approx(r["v"] - r["v_certain"], rel=1e-5)
    assert r["doubted_states"] > 0


def test_looking_wins_when_the_doubt_is_nearer_than_the_detour_it_forces():
    r = _ask(_gap_scene())
    assert r["cost_to_look"] < r["gap"], "the band is close and the detour is long"
    assert r["worth_looking"]


def test_looking_loses_when_the_doubt_is_far_and_the_detour_is_cheap():
    """The same shape of scene, but the uncertain band is a short hop from the goal end and
    barely forces a detour at all. Pushing on is then the right call."""
    f = _field()
    f.seed_cell(*GOAL)
    patches = [(slice(0, 4), slice(30, 33), UNCERTAIN)]  # a small patch out of the way
    r = f.value_of_looking(_constraints(patches), *ROBOT)
    assert r["doubted_states"] > 0, "there IS something doubted"
    assert r["gap"] < r["cost_to_look"], "it just is not worth the trip"
    assert not r["worth_looking"]


def test_min_doubt_filters_what_counts_as_worth_looking_at():
    f = _field()
    f.seed_cell(*GOAL)
    con = _constraints(_gap_scene())
    loose = f.value_of_looking(con, *ROBOT, min_doubt=0.0)
    strict = f.value_of_looking(con, *ROBOT, min_doubt=1.0e6)
    assert loose["doubted_states"] > 0
    assert strict["doubted_states"] == 0
    assert strict["cost_to_look"] == float("inf"), "nothing clears the bar, so nowhere to go"


# -- it must not disturb the field it was asked about ------------------------------------------


def test_asking_leaves_the_believed_solve_in_place():
    """Three solves run inside, and the last of them is seeded on doubt rather than on the goal.
    Anything left behind would describe a question nobody asked."""
    f = _field()
    f.seed_cell(*GOAL)
    con = _constraints(_gap_scene())
    f.solve_pair(con)
    want = {n: getattr(f, n).numpy().copy() for n in ("V", "V_certain", "z", "doubt", "pose_cost")}
    seeds = f._seeds.numpy().copy()
    f.value_of_looking(con, *ROBOT)
    for n, v in want.items():
        np.testing.assert_array_equal(getattr(f, n).numpy(), v, err_msg=f"{n} was left disturbed")
    np.testing.assert_array_equal(f._seeds.numpy(), seeds, "the goal seeding was not restored")


def test_seed_doubt_makes_the_field_the_distance_to_the_nearest_doubt():
    f = _field()
    f.seed_cell(*GOAL)
    con = _constraints(_gap_scene())
    f.solve_pair(con)
    f.seed_doubt()
    v = f.solve(con, certain=True).numpy()[:, :, 0]
    inside, mid, far = v[20, 9], v[20, 20], v[20, 35]  # the band spans columns 8..10
    assert inside == pytest.approx(0.0), "a doubted state costs nothing to reach: it IS the target"
    assert 0.0 < mid < far, "and the field grows with distance from the band"
    assert mid == pytest.approx((20 - 10) * CELL, rel=0.05), "straight-line, on open ground"
