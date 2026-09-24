"""A drop seen across its own shadow is an edge, in both layers."""

from __future__ import annotations

import numpy as np
import warp as wp

from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams
from helhest.planning.coarse import CoarseRouter
from helhest.planning.costtogo import CostToGo

N, CELL = 50, 0.2


def _cliff(drop: float, shadow_cells: int = 4):
    """Flat ground for y < 5 m, `drop` lower beyond, the band between unmeasured."""
    h = np.zeros((N, N), np.float32)
    m = np.ones((N, N), np.float32)
    edge = 25
    h[edge:, :] = -drop
    m[edge : edge + shadow_cells, :] = 0.0
    return h, m


def test_the_fine_layer_blocks_the_body_off_a_cliff_edge():
    grid = GridParams(N, N, CELL, 0.0, 0.0)
    ctg = CostToGo(grid, RobotParams(), SolverParams(), n_theta=8, drop_m=0.5, device="cuda")
    h, m = _cliff(1.0)
    ctg.compute(wp.array(h), (5.0, 2.0), measured=wp.array(m))
    blocked = ctg.blocked.numpy().max(2)
    # the last measured row before the shadow (y = 4.8) and the body's reach behind it are blocked
    assert blocked[24, 25] > 0.5 and blocked[20, 25] > 0.5, "the edge and the tail's reach"
    assert blocked[10, 25] < 0.5, "well back from the edge is free"
    assert (ctg.hazard.numpy()[24, 25] > 0.5).any(), "an edge is a hazard, eroded hard"


def test_a_gentle_shadowed_slope_is_not_an_edge():
    grid = GridParams(N, N, CELL, 0.0, 0.0)
    ctg = CostToGo(grid, RobotParams(), SolverParams(), n_theta=8, drop_m=0.5, device="cuda")
    h, m = _cliff(0.3)  # 0.3 m lower across a 0.8 m band: a slope, not a cliff
    ctg.compute(wp.array(h), (5.0, 2.0), measured=wp.array(m))
    assert ctg.blocked.numpy().max(2)[20, 25] < 0.5


def test_the_coarse_layer_seals_the_edge_and_keeps_the_slope():
    grid = GridParams(N, N, CELL, 0.0, 0.0)
    for drop, sealed in ((1.0, True), (0.3, False)):
        r = CoarseRouter(grid, factor=3, max_step_m=0.25, drop_m=0.5, device="cuda")
        h, m = _cliff(drop)
        r.solve(wp.array(h), wp.array(m), (5.0, 2.0))
        edge_row = 24 // 3  # the block holding the last measured row before the shadow
        assert (r.passable.numpy()[edge_row, 3:14] < 0.5).all() == sealed


def _wall_with_shadow():
    """A 1 m wall two cells thick across the window with an unmeasured band behind it (its
    shadow), and a 1.8 m doorway. The wall's top is higher than the ground beyond the shadow:
    it must NOT read as a cliff edge, or the doorway is sealed."""
    h = np.zeros((N, N), np.float32)
    m = np.ones((N, N), np.float32)
    h[25:27, :] = 1.0
    h[25:27, 20:29] = 0.0  # the doorway
    m[27:31, :] = 0.0
    m[27:31, 20:29] = 1.0
    return h, m


def test_a_wall_top_over_its_own_shadow_is_not_an_edge():
    grid = GridParams(N, N, CELL, 0.0, 0.0)
    with_drop = CostToGo(grid, RobotParams(), SolverParams(), n_theta=8, drop_m=0.5, device="cuda")
    without = CostToGo(grid, RobotParams(), SolverParams(), n_theta=8, drop_m=0.0, device="cuda")
    h, m = _wall_with_shadow()
    a = with_drop.compute(wp.array(h), (4.8, 8.0), measured=wp.array(m)).numpy()
    b = without.compute(wp.array(h), (4.8, 8.0), measured=wp.array(m)).numpy()
    assert (with_drop._drop.numpy() == 0.0).all(), "the wall top was read as a cliff edge"
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-4)  # the doorway is untouched


def test_the_coarse_layer_keeps_a_doorway_beside_a_shadowed_wall():
    grid = GridParams(N, N, CELL, 0.0, 0.0)
    h, m = _wall_with_shadow()
    r = CoarseRouter(grid, factor=3, max_step_m=0.25, drop_m=0.5, device="cuda")
    v = r.solve(wp.array(h), wp.array(m), (4.8, 8.0)).numpy()[:, :, 0]
    assert v[2, 8] < r.solver._inf, "the doorway must still connect the two sides"


def test_a_block_top_with_its_far_side_in_shadow_is_not_an_edge():
    """A 1 m block six cells wide, seen from the south: its top is measured, its north face
    and the ground behind it are in shadow, and the ground further north is measured again.
    The top is a metre above the ground beside it; it is an obstacle, not a cliff edge, and
    reading it as one blocked the body's reach around every ridge crest."""
    h = np.zeros((N, N), np.float32)
    m = np.ones((N, N), np.float32)
    h[20:26, 10:40] = 1.0  # the block
    m[26:31, 10:40] = 0.0  # its shadow to the north
    grid = GridParams(N, N, CELL, 0.0, 0.0)
    ctg = CostToGo(grid, RobotParams(), SolverParams(), n_theta=8, drop_m=0.5, device="cuda")
    ctg.compute(wp.array(h), (5.0, 8.0), measured=wp.array(m))
    assert (ctg._drop.numpy() == 0.0).all(), "a block top was read as a cliff edge"
    r = CoarseRouter(grid, factor=3, max_step_m=0.25, drop_m=0.5, device="cuda")
    v = r.solve(wp.array(h), wp.array(m), (5.0, 8.0)).numpy()[:, :, 0]
    assert v[2, 8] < r.solver._inf, "the way round the block is still open"
