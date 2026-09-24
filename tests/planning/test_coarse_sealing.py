"""Two ways the coarse layer let a route through a wall it had measured, from false_door.

A wall seen from one side is a flat strip of measured cells on top with the ground at its foot
in shadow, and a block holding only such cells pools as crossable. And with every wall block
sealed, a diagonal move still passed between two vetoed orthogonal neighbours: through the
corner where two walls meet.
"""

from __future__ import annotations

import numpy as np
import warp as wp
from terrain_value_field import omni_control_set
from terrain_value_field.solver import ValueSolver

from helhest.engine import GridParams
from helhest.planning.coarse import _omni_no_corner_cutting
from helhest.planning.coarse import CoarseRouter

CELL = 0.2
FACTOR = 3
N = 60  # 12 m


def _router() -> CoarseRouter:
    grid = GridParams(cells_x=N, cells_y=N, cell_size=CELL, origin_x=0.0, origin_y=0.0)
    return CoarseRouter(grid, factor=FACTOR, max_step_m=0.25, device="cuda")


def _dev(v: np.ndarray) -> wp.array:
    return wp.array(np.ascontiguousarray(v, np.float32), dtype=wp.float32)


def test_a_wall_seen_only_from_its_top_is_not_a_plateau():
    """The wall's two cells are the first two of a block whose third cell lies in shadow: the
    block's measured cells are wall top, level with each other, and the block pooled to 0.5
    -- crossable -- by the step test alone."""
    h = np.zeros((N, N), np.float32)
    m = np.ones((N, N), np.float32)
    c = 30  # a block boundary: cells 30, 31 are wall, 32 unmeasured
    h[:, c : c + 2] = 1.0
    m[:, c + 2 :] = 0.0
    r = _router()
    r.solve(_dev(h), _dev(m), (11.0, 6.0))
    frac = r.passable.numpy()
    block = c // FACTOR
    assert (frac[:, block] < 0.5).all(), "the wall-top block pooled as crossable"
    assert (frac[:, block - 1] >= 0.5).all(), "the ground before the wall is still crossable"


def test_ground_is_read_from_the_blocks_around_not_the_block_alone():
    """The same wall, but every block on the wall's own column holds wall top only: the ground
    it stands over is in the blocks beside it, and that is where the rule must look."""
    h = np.zeros((N, N), np.float32)
    m = np.ones((N, N), np.float32)
    c = 30
    h[:, c : c + 3] = 1.0  # a whole block of wall top
    m[:, c + 3 :] = 0.0
    r = _router()
    r.solve(_dev(h), _dev(m), (11.0, 6.0))
    assert (r.passable.numpy()[:, c // FACTOR] < 0.5).all()


def _solve(control_set, cost: np.ndarray) -> np.ndarray:
    n = cost.shape[0]
    solver = ValueSolver(1.0, n, n, n_theta=1, control_set=control_set(1.0), device="cuda")
    seeds = np.full((n, n, 1), solver._inf, np.float32)
    seeds[n - 1, n - 1, 0] = 0.0
    return solver.value_iterate(_dev(cost[:, :, None]), _dev(seeds), 1.0).numpy()[:, :, 0]


def test_a_diagonal_does_not_cut_the_corner_where_two_walls_meet():
    """An L of vetoed cells closes the top-left room, except that the corner cell itself is
    free -- as false_door's far corners were. The stock set steps from the room to the corner
    cell and out, between two vetoed cells; the corner-safe set does not."""
    cost = np.zeros((5, 5), np.float32)
    cost[2, 0:2] = -1.0  # the row of wall ...
    cost[0:2, 2] = -1.0  # ... the column of wall, and (2, 2) where they meet is free
    leaky = _solve(omni_control_set, cost)
    sealed = _solve(_omni_no_corner_cutting, cost)
    assert leaky[1, 1] < 100.0, "the stock set should leak here, or this test proves nothing"
    assert sealed[1, 1] >= 1.0e5, "the corner-safe set cut the corner"
    assert sealed[3, 3] < 100.0, "outside the room the field is untouched"


def test_a_doorway_one_cell_wide_is_still_crossed():
    cost = np.zeros((5, 5), np.float32)
    cost[:, 2] = -1.0
    cost[2, 2] = 0.0  # the doorway
    v = _solve(_omni_no_corner_cutting, cost)
    assert v[2, 0] < 100.0, "the doorway must connect the two sides"
    # 2 across to the doorway, 1 more straight -- leaving it, the diagonal along the wall would
    # cut the corner of the wall cell beside it -- then a diagonal and a last straight step
    np.testing.assert_allclose(v[2, 0], 4.0 + np.sqrt(2.0), rtol=1e-5)
