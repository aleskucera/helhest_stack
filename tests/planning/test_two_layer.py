"""Two layers: a coarse field that says which way, a fine one that says how.

The case this exists for was measured, not imagined. Driving `pillars` closed-loop, the robot
committed to the corridor along the south wall at a point where the goal region was still beyond
its ~6 m sensor horizon, followed it east, and wedged itself in the corner between the south and
east walls -- a pocket the fine router then correctly reported as unreachable (V at its own cell
pinned to the cap). It did not fail to find a route; it took one whose dead end it could not yet
see.

A coarse layer can see it, because coverage is a question of whether ANY return landed in a cell:
pooled to 1.0 m the same map is 85% known at 10 m where the 0.2 m grid is 26%. So these tests are
about one property -- that the fine window's border is priced by what lies beyond it.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams
from helhest.planning.coarse import CoarseRouter
from helhest.planning.costtogo import CostToGo

CELL = 0.2
WORLD = 100  # 20 m at 0.2 m
FINE = 30  # 6 m fine window
WALL = 1.0


def _world_grid() -> GridParams:
    return GridParams(cells_x=WORLD, cells_y=WORLD, cell_size=CELL, origin_x=0.0, origin_y=0.0)


def _dev(v) -> wp.array:
    return wp.array(np.ascontiguousarray(v, np.float32), dtype=wp.float32)


def _blocking_wall() -> tuple[np.ndarray, np.ndarray]:
    """Flat ground, and a wall at x = 12 m running from y = 4 m to the top of the world.

    Everything east of it is reachable only by rounding its SOUTHERN end. The asymmetry is the
    point: with the goal due east, the way there is south, so "head at the goal" and "head the
    right way" are different directions and a test can tell them apart. The wall sits outside a
    6 m window centred at (7, 12), so the fine layer cannot see it at all.
    """
    h = np.zeros((WORLD, WORLD), np.float32)
    c = int(12.0 / CELL)
    h[int(4.0 / CELL) :, c : c + 2] = WALL
    return h, np.ones_like(h)


ROBOT = (7.0, 12.0)
GOAL = (18.0, 12.0)  # due east of the robot, reachable only by going south


def _gap_wall(opening: float = 1.8) -> tuple[np.ndarray, np.ndarray]:
    """A wall right across the world at x = 12, with an `opening`-metre doorway at y = 12.

    Modelled on the `gap` stress world, whose wall has a 1.8 m opening -- the case that showed
    the first version of this layer sealing every passage narrower than about three coarse cells.
    Everything is measured here: the question is what the pooling makes of geometry it CAN see.
    """
    h = np.zeros((WORLD, WORLD), np.float32)
    c = int(12.0 / CELL)
    h[:, c : c + 2] = WALL
    lo = int((12.0 - opening / 2) / CELL)
    hi = int((12.0 + opening / 2) / CELL)
    h[lo:hi, c : c + 2] = 0.0
    return h, np.ones_like(h)


def _fine_window(cx: float, cy: float) -> GridParams:
    """A FINE-window GridParams centred on (cx, cy), in world coordinates."""
    return GridParams(
        cells_x=FINE,
        cells_y=FINE,
        cell_size=CELL,
        origin_x=cx - FINE * CELL / 2,
        origin_y=cy - FINE * CELL / 2,
    )


def _crop(a: np.ndarray, g: GridParams) -> np.ndarray:
    r0 = int(round(g.origin_y / CELL))
    c0 = int(round(g.origin_x / CELL))
    return np.ascontiguousarray(a[r0 : r0 + FINE, c0 : c0 + FINE])


def _solve(h, m, robot_xy, goal_xy, two_layer: bool):
    """The fine field at `robot_xy`, with or without the coarse layer priced into its border.

    Both layers work in the FINE window's own frame, which is how the caller uses them: the
    windows recenter together so the offset between them is a constant.
    """
    fg = _fine_window(*robot_xy)
    local = GridParams(cells_x=FINE, cells_y=FINE, cell_size=CELL, origin_x=0.0, origin_y=0.0)
    ctg = CostToGo(local, RobotParams(), SolverParams(), n_theta=16, device="cuda")
    goal_l = (goal_xy[0] - fg.origin_x, goal_xy[1] - fg.origin_y)
    if not two_layer:
        return ctg.compute(_dev(_crop(h, fg)), goal_l), None
    coarse = CoarseRouter(_world_grid(), factor=5, max_step_m=0.25, device="cuda")
    vc = coarse.solve(_dev(h), _dev(m), (goal_xy[0], goal_xy[1]))
    # the coarse grid is in WORLD coordinates; express its origin in the fine window's frame
    cg = GridParams(
        cells_x=coarse.grid.cells_x,
        cells_y=coarse.grid.cells_y,
        cell_size=coarse.grid.cell_size,
        origin_x=coarse.grid.origin_x - fg.origin_x,
        origin_y=coarse.grid.origin_y - fg.origin_y,
    )
    ctg.set_coarse(cg)
    return ctg.compute(_dev(_crop(h, fg)), goal_l, coarse_value=vc), coarse


def _heads(V: np.ndarray) -> str:
    """Which way the field sends the robot from the centre of the window.

    The decision, rather than a border statistic: comparing the cheapest cell on each side cannot
    separate two adjacent sides, because they share the corner that is usually cheapest on both.
    """
    v = V.numpy().min(2)
    mid = v.shape[0] // 2
    out = {
        "east": v[mid, mid + 5],
        "west": v[mid, mid - 5],
        "north": v[mid + 5, mid],
        "south": v[mid - 5, mid],
    }
    return min(out, key=out.get)


def _at_robot(V: np.ndarray) -> float:
    v = V.numpy().min(2)
    return float(v[v.shape[0] // 2, v.shape[1] // 2])


# --------------------------------------------------------------------------------------- coarse


def test_the_coarse_layer_sees_a_wall_the_fine_window_cannot():
    h, m = _blocking_wall()
    coarse = CoarseRouter(_world_grid(), factor=5, max_step_m=0.25, device="cuda")
    coarse.solve(_dev(h), _dev(m), GOAL)
    k = coarse.grid.cell_size
    col = int(12.0 / k)
    frac = coarse.passable.numpy()
    assert (frac[int(5.0 / k) :, col] < 0.5).all(), "no room to cross the solid part of the wall"
    assert frac[0, col] >= 0.5, "and the ground past its southern end is still crossable"


def test_the_coarse_layer_routes_around_the_wall_rather_than_through_it():
    h, m = _blocking_wall()
    coarse = CoarseRouter(_world_grid(), factor=5, max_step_m=0.25, device="cuda")
    v = coarse.solve(_dev(h), _dev(m), GOAL).numpy()[:, :, 0]
    k = coarse.grid.cell_size
    at = lambda x, y: v[int(y / k), int(x / k)]  # noqa: E731
    # the solver spells unreachable as its own large sentinel, not as a float inf
    assert at(12.0, 12.0) >= coarse.solver._inf, "the wall itself must be impassable"
    goal_side = at(16.0, 12.0)
    assert goal_side < 4.0
    # 11 m as the crow flies, but the way round costs ~26: that difference is the whole point
    assert at(*ROBOT) > 20.0, f"the detour must be priced, got {at(*ROBOT):.2f}"


def test_a_gap_narrower_than_a_coarse_cell_pair_still_reads_as_a_way_through():
    """The regression this layer was rewritten for.

    A 1.8 m doorway is under two 1.0 m coarse cells wide, so no coarse cell fits inside it at
    most grid alignments. Pooling PASSABILITY with ANY does not need one to: a block holding any
    climbable fine cell is a block with a way through it. Pooling height with MAX did need one,
    and sealed the doorway.
    """
    h, m = _gap_wall(1.8)
    coarse = CoarseRouter(_world_grid(), factor=5, max_step_m=0.25, device="cuda")
    v = coarse.solve(_dev(h), _dev(m), (18.0, 12.0)).numpy()[:, :, 0]
    k = coarse.grid.cell_size
    at = lambda x, y: v[int(y / k), int(x / k)]  # noqa: E731
    west = at(8.0, 12.0)
    assert west < coarse.solver._inf, "the doorway must connect the two sides"
    # straight through is ~10 m; anything near the 40 m way round the world is not the doorway
    assert west < 16.0, f"it should route THROUGH the gap, not around the world: {west:.1f} m"
    assert at(12.0, 3.0) >= coarse.solver._inf, "and the solid part of the wall stays blocked"


def test_a_doorway_the_pooling_cannot_see_is_the_one_case_it_may_seal():
    """The limit, stated rather than hidden. A doorway narrower than one FINE cell leaves no
    climbable fine cell, so nothing survives to pool -- being wrong here is being wrong in the
    permissive layer's safe direction only because the robot could not have fitted anyway."""
    h, m = _gap_wall(0.2)
    coarse = CoarseRouter(_world_grid(), factor=5, max_step_m=0.25, device="cuda")
    v = coarse.solve(_dev(h), _dev(m), (18.0, 12.0)).numpy()[:, :, 0]
    k = coarse.grid.cell_size
    assert v[int(12.0 / k), int(8.0 / k)] > 16.0, "a 0.2 m slit is not a route"


# ------------------------------------------------------------- unmeasured ground, and its price


def test_unmeasured_ground_near_the_frontier_is_free():
    h = np.zeros((WORLD, WORLD), np.float32)
    m = np.ones_like(h)
    m[:, int(10.0 / CELL) :] = 0.0  # nothing east of 10 m has ever been measured
    coarse = CoarseRouter(_world_grid(), factor=5, frontier_m=3.0, void_penalty=1.0, device="cuda")
    coarse.solve(_dev(h), _dev(m), (18.0, 10.0))
    cost = coarse._pose_cost.numpy()[:, :, 0]
    k = coarse.grid.cell_size
    assert cost[int(10.0 / k), int(11.5 / k)] == 0.0, "just past the frontier is still free"


def test_unmeasured_ground_far_past_the_frontier_is_priced_but_never_blocked():
    """What stops the layer routing around the outside of the world. On `gap` it costed an 18 m
    detour through 483 cells that have no world in them and called it a route."""
    h = np.zeros((WORLD, WORLD), np.float32)
    m = np.ones_like(h)
    m[:, int(10.0 / CELL) :] = 0.0
    coarse = CoarseRouter(_world_grid(), factor=5, frontier_m=3.0, void_penalty=1.0, device="cuda")
    v = coarse.solve(_dev(h), _dev(m), (18.0, 10.0)).numpy()[:, :, 0]
    cost = coarse._pose_cost.numpy()[:, :, 0]
    k = coarse.grid.cell_size
    deep = cost[int(10.0 / k), int(17.0 / k)]
    assert deep == pytest.approx(1.0), f"deep void should carry the penalty, got {deep}"
    assert deep > 0.0, "priced, not vetoed -- the sign carries the veto"
    assert v[int(10.0 / k), int(2.0 / k)] < coarse.solver._inf, "and the goal stays reachable"


def test_the_void_penalty_makes_the_real_way_round_the_cheaper_one():
    """A wall with a doorway, and nothing measured beyond the wall. Un-penalised, the layer would
    rather leave the world than use the doorway, because empty space is free and shorter."""
    h, m = _gap_wall(1.8)
    m = m.copy()
    m[: int(6.0 / CELL), :] = 0.0  # the whole southern strip is unseen: a tempting way round
    free = CoarseRouter(_world_grid(), factor=5, void_penalty=0.0, device="cuda")
    priced = CoarseRouter(_world_grid(), factor=5, void_penalty=2.0, device="cuda")
    k = free.grid.cell_size
    a = free.solve(_dev(h), _dev(m), (18.0, 12.0)).numpy()[int(12.0 / k), int(8.0 / k), 0]
    b = priced.solve(_dev(h), _dev(m), (18.0, 12.0)).numpy()[int(12.0 / k), int(8.0 / k), 0]
    assert b >= a, "pricing the void can only raise the cost of a route that uses it"


# ------------------------------------------------------------------------------------ the pair


def test_one_layer_drives_at_the_goal_and_into_the_wall():
    """The failure, stated as the thing the single layer actually does.

    The goal is outside the window, so it clamps onto the east border at zero cost, and the field
    reports the goal as 3.1 m away when the real cost is 26. The robot heads east into a wall it
    has no way to know about. This is the sim failure in miniature: it committed to a corridor
    whose dead end was beyond its horizon.
    """
    h, m = _blocking_wall()
    V, _ = _solve(h, m, ROBOT, GOAL, two_layer=False)
    assert _heads(V) == "east"
    assert _at_robot(V) < 5.0, "and believes the goal is close, which is the lie that causes it"


def test_two_layers_drive_the_way_that_leads_somewhere_and_price_it_honestly():
    """The point of the pair, and the same scene.

    The wall is outside the fine window and invisible to it. The coarse layer prices every exit
    by what lies beyond, so the fine field sends the robot SOUTH around the wall's end -- and its
    value at the robot agrees with the coarse layer's to a few per cent, rather than being a
    fiction about a goal it cannot reach.
    """
    h, m = _blocking_wall()
    V, coarse = _solve(h, m, ROBOT, GOAL, two_layer=True)
    assert _heads(V) == "south"
    k = coarse.grid.cell_size
    truth = float(coarse.V.numpy()[int(ROBOT[1] / k), int(ROBOT[0] / k), 0])
    assert _at_robot(V) == pytest.approx(
        truth, rel=0.1
    ), f"the fine field should inherit the real cost: {_at_robot(V):.2f} vs {truth:.2f}"


def test_arming_the_coarse_layer_changes_nothing_when_the_goal_is_inside_the_window():
    """A window that contains the goal is missing nothing, so the ring must not be seeded at all.

    Not merely redundant: the coarse layer is omnidirectional and pays no turn cost, so it
    understates distance in the fine layer's own metric. A ring priced that way reads as a
    shortcut, and the fine solve routes the robot out of the window and back to reach a goal
    sitting two metres in front of it. That is what this caught the first time it was run.
    """
    h = np.zeros((WORLD, WORLD), np.float32)
    m = np.ones_like(h)
    one, _ = _solve(h, m, ROBOT, (7.6, 12.4), two_layer=False)
    two, _ = _solve(h, m, ROBOT, (7.6, 12.4), two_layer=True)
    np.testing.assert_allclose(two.numpy(), one.numpy(), rtol=1e-4, atol=1e-4)


def test_arming_after_the_first_compute_is_refused():
    """The graph is captured around whichever seeding was in force, so arming later would replay
    a stale one. Better to say so than to plan on a field nobody asked for."""
    h = np.zeros((WORLD, WORLD), np.float32)
    local = GridParams(cells_x=FINE, cells_y=FINE, cell_size=CELL, origin_x=0.0, origin_y=0.0)
    ctg = CostToGo(local, RobotParams(), SolverParams(), n_theta=16, device="cuda")
    ctg.compute(_dev(_crop(h, _fine_window(*ROBOT))), (3.0, 3.0))
    with pytest.raises(RuntimeError, match="before the first compute"):
        ctg.set_coarse(local)


def test_passing_coarse_values_without_arming_is_refused():
    h = np.zeros((WORLD, WORLD), np.float32)
    local = GridParams(cells_x=FINE, cells_y=FINE, cell_size=CELL, origin_x=0.0, origin_y=0.0)
    ctg = CostToGo(local, RobotParams(), SolverParams(), n_theta=16, device="cuda")
    stray = wp.zeros((4, 4, 1), dtype=wp.float32)
    with pytest.raises(RuntimeError, match="set_coarse"):
        ctg.compute(_dev(_crop(h, _fine_window(*ROBOT))), (3.0, 3.0), coarse_value=stray)
