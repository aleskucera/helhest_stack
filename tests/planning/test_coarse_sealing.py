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


# ------------------------------------------------------------------------------------- memory


def _two_frames(memory: bool) -> CoarseRouter:
    """A wall seen in one frame, then the window moves 8 m east and the wall leaves it."""
    fine = GridParams(cells_x=N, cells_y=N, cell_size=CELL, origin_x=0.0, origin_y=0.0)
    mem = GridParams(cells_x=150, cells_y=150, cell_size=CELL, origin_x=0.0, origin_y=0.0)
    r = CoarseRouter(
        fine, factor=FACTOR, max_step_m=0.25, memory_grid=mem if memory else None, device="cuda"
    )
    h = np.zeros((N, N), np.float32)
    m = np.ones((N, N), np.float32)
    h[:, 30:32] = 1.0  # a wall at x = 6 m across the whole window
    goal = (25.0, 6.0) if memory else (11.0, 6.0)
    r.solve(_dev(h), _dev(m), goal, (0.0, 0.0))
    r.solve(_dev(np.zeros((N, N), np.float32)), _dev(m), goal, (8.0, 0.0) if memory else (0.0, 0.0))
    return r


def test_an_anchored_map_remembers_a_wall_the_window_left():
    r = _two_frames(memory=True)
    seen, frac = r.seen.numpy(), r.passable.numpy()
    col = 30 // FACTOR  # the wall's block column, in the memory's own lattice
    rows = slice(0, N // FACTOR)
    assert (seen[rows, col] > 0.5).all() and (frac[rows, col] < 0.5).all(), "the wall was forgotten"
    v = r.V.numpy()[:, :, 0]
    inf = r.solver._inf
    assert (v[rows, col] >= inf).all(), "the remembered wall must still be impassable"
    # the ground west of it routes round the wall's ends, never through: 19 m as the crow flies
    assert v[N // FACTOR // 2, col - 2] > 19.0 + 4.0
    # and the blocks the window now covers were re-pooled from flat ground
    assert (frac[rows, 40 // FACTOR : (40 + N) // FACTOR] >= 0.5).all()


def test_a_window_bound_layer_forgets_what_scrolls_out():
    r = _two_frames(memory=False)
    frac = r.passable.numpy()
    assert (frac[: N // FACTOR, : N // FACTOR] >= 0.5).all(), "a moved window holds new ground"


def test_the_window_offset_pools_into_the_memory_blocks_it_covers():
    fine = GridParams(cells_x=N, cells_y=N, cell_size=CELL, origin_x=0.0, origin_y=0.0)
    mem = GridParams(cells_x=150, cells_y=150, cell_size=CELL, origin_x=0.0, origin_y=0.0)
    r = CoarseRouter(fine, factor=FACTOR, max_step_m=0.25, memory_grid=mem, device="cuda")
    h = np.zeros((N, N), np.float32)
    m = np.ones((N, N), np.float32)
    h[:, 30:32] = 1.0
    r.solve(_dev(h), _dev(m), (25.0, 20.0), (6.0, 12.0))  # the window at (30, 60) fine cells
    seen, frac = r.seen.numpy(), r.passable.numpy()
    rows = slice(60 // FACTOR, (60 + N) // FACTOR)
    assert (frac[rows, (30 + 30) // FACTOR] < 0.5).all(), "the wall landed in the wrong blocks"
    assert seen[: 60 // FACTOR].sum() == 0 and seen[:, : 30 // FACTOR].sum() == 0, "nothing else"


def test_an_anchored_grid_with_its_own_origin_seeds_the_goal_where_it_is():
    """The first anchored run reached its goal for the wrong reason: the goal and the window
    were passed already offset by the grid's origin, the goal kernel subtracted it again, and
    the seed landed clamped in the grid's far corner. Coordinates are the grid's own."""
    fine = GridParams(cells_x=N, cells_y=N, cell_size=CELL, origin_x=0.0, origin_y=0.0)
    mem = GridParams(cells_x=150, cells_y=150, cell_size=CELL, origin_x=-9.0, origin_y=-15.0)
    r = CoarseRouter(fine, factor=FACTOR, max_step_m=0.25, memory_grid=mem, device="cuda")
    h = np.zeros((N, N), np.float32)
    m = np.ones((N, N), np.float32)
    h[:, 30:32] = 1.0  # a wall at x = 6 m in the window, the window's origin at (0, 0) world
    # off a block edge, where float32 and Python would round the cell index differently
    v = r.solve(_dev(h), _dev(m), (3.1, 3.1), (0.0, 0.0)).numpy()[:, :, 0]
    rr, cc = np.unravel_index(np.argmin(v), v.shape)
    assert (rr, cc) == (int((3.1 + 15.0) / 0.6), int((3.1 + 9.0) / 0.6)), "the goal seed moved"
    assert v[rr, cc] == 0.0
    # and the wall is where the window put it: x = 6 world -> column (6 + 9) / 0.6
    assert (r.passable.numpy()[int(15.0 / 0.6) : int(27.0 / 0.6), int(15.0 / 0.6)] < 0.5).all()


def test_a_poorer_view_does_not_overwrite_a_remembered_wall():
    """Seen whole, a block holding a wall is sealed. Seen again from the ground side only -- the
    cells the belief re-measures first when the wall comes back into view -- it stays sealed.
    Seen whole again, and flat this time, it opens: the world may have changed."""
    fine = GridParams(cells_x=N, cells_y=N, cell_size=CELL, origin_x=0.0, origin_y=0.0)
    mem = GridParams(cells_x=N, cells_y=N, cell_size=CELL, origin_x=0.0, origin_y=0.0)
    r = CoarseRouter(fine, factor=FACTOR, max_step_m=0.25, memory_grid=mem, device="cuda")
    goal = (11.0, 6.0)
    wall = np.zeros((N, N), np.float32)
    wall[:, 31:33] = 1.0  # inside block column 10 (cells 30..32), with ground in cell 30
    full = np.ones((N, N), np.float32)
    r.solve(_dev(wall), _dev(full), goal, (0.0, 0.0))
    assert (r.passable.numpy()[:, 10] < 0.5).all()
    part = full.copy()
    part[:, 31:] = 0.0  # only the ground before the wall is measured now, and it is flat
    r.solve(_dev(np.zeros((N, N), np.float32)), _dev(part), goal, (0.0, 0.0))
    assert (r.passable.numpy()[:, 10] < 0.5).all(), "three flat cells overrode nine"
    r.solve(_dev(np.zeros((N, N), np.float32)), _dev(full), goal, (0.0, 0.0))
    assert (r.passable.numpy()[:, 10] >= 0.5).all(), "a whole view of flat ground must open it"


# ------------------------------------------------------------------------------------- bridge


def _wall_with_shadow(gap: int) -> CoarseRouter:
    """A sealed wall across the window with `gap` UNSEEN blocks in it, bounded by sealed
    blocks on both sides; the goal beyond the wall."""
    fine = GridParams(cells_x=N, cells_y=N, cell_size=CELL, origin_x=0.0, origin_y=0.0)
    r = CoarseRouter(fine, factor=FACTOR, max_step_m=0.25, bridge_m=1.2, device="cuda")
    h = np.zeros((N, N), np.float32)
    m = np.ones((N, N), np.float32)
    h[:, 30:32] = 1.0  # the wall, in block column 10
    r0 = 9 * FACTOR
    m[r0 : r0 + gap * FACTOR, 27:36] = 0.0  # the shadow: block rows 9.., columns 9..11 unseen
    r.solve(_dev(h), _dev(m), (11.0, 6.0), (0.0, 0.0))
    return r


def test_a_short_shadow_in_a_sealed_wall_is_the_wall():
    r = _wall_with_shadow(2)
    v = r.V.numpy()[:, :, 0]
    assert (r.bridged.numpy()[9:11, 10] > 0.5).all(), "the shadow was not bridged"
    assert (v[:, 10] >= r.solver._inf).all(), "the wall must be whole"
    assert (r.bridged.numpy()[9:11, 9] < 0.5).all(), "unseen ground beside the wall is not wall"


def test_a_shadow_wider_than_a_doorway_is_left_open():
    r = _wall_with_shadow(3)
    assert (r.bridged.numpy()[:, 10] < 0.5).all()
    v = r.V.numpy()[:, :, 0]
    assert (v[9:12, 10] < r.solver._inf).all(), "three blocks may be a door: it stays open"


def test_a_shadow_at_a_walls_end_is_not_bridged():
    """Sealed on one side only: the wall may genuinely end there."""
    fine = GridParams(cells_x=N, cells_y=N, cell_size=CELL, origin_x=0.0, origin_y=0.0)
    r = CoarseRouter(fine, factor=FACTOR, max_step_m=0.25, bridge_m=1.2, device="cuda")
    h = np.zeros((N, N), np.float32)
    m = np.ones((N, N), np.float32)
    h[: 9 * FACTOR, 30:32] = 1.0  # the wall stops at block row 9 ...
    m[9 * FACTOR : 11 * FACTOR, 27:36] = 0.0  # ... and rows 9-10 beside its end are unseen
    r.solve(_dev(h), _dev(m), (11.0, 6.0), (0.0, 0.0))
    assert (r.bridged.numpy() < 0.5).all()
