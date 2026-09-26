"""Two layers: the fine window routes correctly around something it cannot see.

The scene is the whole point. A wall at x = 10 m forces a detour north, and the goal is at
x = 16 m. The fine window spans +-6.9 m about the robot, so BOTH the wall and the goal are
outside it and every cell the fine solve can see is free. Any preference it shows for one exit
over another therefore came from the coarse layer, and nowhere else.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from helhest.grid import build_grid
from helhest.planning.terrain_value_field import omni_control_set
from helhest.planning.terrain_value_field import TerrainValueField
from helhest.planning.terrain_value_field.field import Constraints

COARSE_CELL, COARSE_N = 0.5, 80  # 40 m of world, cell (0,0) centred at (-20, -20)
FINE_CELL, FINE_N = 0.2, 70  # 13.8 m window about the robot
COARSE_ORIGIN = -20.0
FINE_ORIGIN = -(FINE_N - 1) / 2 * FINE_CELL  # -6.9, so the window is centred on the robot
UNREACHABLE = 1.0e29

GOAL_XY = (16.0, 0.0)
WALL_X = (9.5, 10.5)
WALL_Y_TOP = 6.0  # the wall runs from the south edge up to here; you must pass north of it


def _constraints(blocked: np.ndarray, n_theta: int) -> Constraints:
    """A free/blocked mask as margins: free states sit far above z_veto, blocked ones below."""
    n = blocked.shape[0]
    m = np.full((1, n, n, n_theta), 0.9, np.float32)
    m[0, blocked, :] = -1.0
    return Constraints(
        margin=wp.array(m, dtype=wp.float32),
        sigma=wp.array(np.full((1, n, n, n_theta), 0.05, np.float32), dtype=wp.float32),
        floor=wp.array([0.01], dtype=wp.float32),
    )


def _coarse_cell(x: float, y: float) -> tuple[int, int]:
    return (
        round((y - COARSE_ORIGIN) / COARSE_CELL),
        round((x - COARSE_ORIGIN) / COARSE_CELL),
    )


@pytest.fixture(scope="module")
def layers():
    blocked = np.zeros((COARSE_N, COARSE_N), bool)
    _, c0 = _coarse_cell(WALL_X[0], -20.0)
    r1, c1 = _coarse_cell(WALL_X[1], WALL_Y_TOP)
    blocked[: r1 + 1, c0 : c1 + 1] = True

    coarse = TerrainValueField(
        COARSE_N,
        COARSE_N,
        COARSE_CELL,
        n_theta=1,
        z_veto=2.0,
        control_set=omni_control_set(COARSE_CELL),
    )
    gr, gc = _coarse_cell(*GOAL_XY)
    coarse.seed_cell(gr, gc)
    coarse.solve(_constraints(blocked, 1), certain=True)

    fine = TerrainValueField(
        FINE_N,
        FINE_N,
        FINE_CELL,
        n_theta=16,
        z_veto=2.0,
        turn_radius=0.6,
        free_blocked_seeds=False,
    )
    coarse_grid = build_grid(COARSE_N, COARSE_N, COARSE_CELL, COARSE_ORIGIN, COARSE_ORIGIN)
    fine_grid = build_grid(FINE_N, FINE_N, FINE_CELL, FINE_ORIGIN, FINE_ORIGIN)
    return coarse, fine, coarse_grid, fine_grid, blocked


def test_the_window_really_cannot_see_the_wall_or_the_goal(layers):
    """If it could, the rest of this file would prove nothing."""
    half = FINE_ORIGIN + (FINE_N - 1) * FINE_CELL
    assert WALL_X[0] > half, "the wall must lie outside the window"
    assert GOAL_XY[0] > half, "so must the goal"


def test_without_seeds_the_fine_window_has_no_plan_at_all(layers):
    _, fine, _, _, _ = layers
    fine.seed_values(np.full((FINE_N, FINE_N, 16), fine.solver_inf, np.float32))
    v = fine.solve(_constraints(np.zeros((FINE_N, FINE_N), bool), 16)).numpy()
    assert (v >= UNREACHABLE).all(), "no sources, so nothing is reachable"


def test_the_coarse_layer_routes_around_the_wall(layers):
    coarse, _, _, _, _ = layers
    v = coarse.V.numpy()[:, :, 0]
    north = v[_coarse_cell(8.0, 8.0)]  # north of the wall: a short way round
    south = v[_coarse_cell(8.0, -8.0)]  # south of it: the long way
    assert north < UNREACHABLE and south < UNREACHABLE
    assert north < south - 5.0, "the detour must be visibly cheaper to the north"


def test_the_fine_window_finds_a_plan_it_could_not_have_found_alone(layers):
    coarse, fine, cg, fg, _ = layers
    fine.seed_from_coarse(coarse.V, cg, fg)
    v = fine.solve(_constraints(np.zeros((FINE_N, FINE_N), bool), 16)).numpy()
    here = v[FINE_N // 2, FINE_N // 2, :].min()
    assert here < UNREACHABLE, "the robot should have a route although the goal is out of sight"
    # and it should roughly agree with the coarse layer about what that route costs
    coarse_here = coarse.V.numpy()[_coarse_cell(0.0, 0.0) + (0,)]
    assert here == pytest.approx(coarse_here, rel=0.35)


def test_the_cheapest_way_out_is_the_one_that_goes_around(layers):
    """The load-bearing test. Inside the window every cell is free and identical, so a preference
    for the northern exits can only have come down from the coarse layer."""
    coarse, fine, cg, fg, _ = layers
    fine.seed_from_coarse(coarse.V, cg, fg)
    seeds = fine._seeds.numpy()[:, :, 0]
    band = fine.solver.reach_cells
    north = seeds[-band:, :].min()  # +y edge: toward the gap
    south = seeds[:band, :].min()  # -y edge: away from it
    assert north < UNREACHABLE and south < UNREACHABLE
    assert north < south - 5.0, f"north {north:.1f} should beat south {south:.1f}"

    v = fine.solve(_constraints(np.zeros((FINE_N, FINE_N), bool), 16)).numpy()
    assert v[-band:, :, :].min() < v[:band, :, :].min() - 5.0


def test_the_seed_band_is_thick_enough_that_an_arc_cannot_jump_it(layers):
    """A move here spans 0.4712 m = 2.36 cells, so a one-cell ring is thinner than a single
    primitive and an arc can land beyond it without ever touching a seed."""
    coarse, fine, cg, fg, _ = layers
    free = _constraints(np.zeros((FINE_N, FINE_N), bool), 16)
    assert fine.solver.reach_cells >= 2, "a single cell is thinner than one move"

    fine.seed_from_coarse(coarse.V, cg, fg, band=1)
    thin_seeds = int((fine._seeds.numpy() < UNREACHABLE).sum())
    thin_v = fine.solve(free).numpy()[FINE_N // 2, FINE_N // 2, :].min()

    fine.seed_from_coarse(coarse.V, cg, fg)  # default band = reach_cells
    assert int((fine._seeds.numpy() < UNREACHABLE).sum()) > thin_seeds
    full_v = fine.solve(free).numpy()[FINE_N // 2, FINE_N // 2, :].min()
    assert full_v <= thin_v + 1e-4, "a thicker ring can only offer more ways out, never fewer"


def test_a_heading_bearing_coarse_layer_is_refused(layers):
    _, fine, cg, fg, _ = layers
    with pytest.raises(ValueError, match="heading-free"):
        fine.seed_from_coarse(fine.V, cg, fg)  # fine.V has 16 headings
