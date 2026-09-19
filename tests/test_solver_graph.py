"""The recorded solve: it must agree with the eager one, and survive a new map.

Caching a CUDA graph is a contract about what may change between replays. The pointers and the
scalar the graph was recorded with may not; the array CONTENTS must, or the whole thing is
useless -- the map is new every frame. These pin both halves.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from terrain_value_field import TerrainValueField
from terrain_value_field.field import Constraints

N = 48
CELL = 0.1
NT = 16

pytestmark = pytest.mark.skipif(
    not wp.get_device().is_cuda, reason="graph capture needs a CUDA device"
)


def _field():
    return TerrainValueField(N, N, CELL, n_theta=NT, k_sigma=2.0)


def _constraints(wall: tuple[int, int, int, int] | None):
    m = np.full((1, N, N, NT), 1.0, np.float32)
    if wall is not None:
        r0, r1, c0, c1 = wall
        m[0, r0:r1, c0:c1, :] = -1.0
    return Constraints(
        margin=wp.array(m, dtype=wp.float32),
        sigma=wp.array(np.full((1, N, N, NT), 0.05, np.float32), dtype=wp.float32),
        floor=wp.array([0.01], dtype=wp.float32),
    )


UNREACHABLE = 1.0e29  # the solver's +inf sentinel is 1e30, not numpy.inf


def _finite(v):
    """Values with unreachable states folded to -1, so comparisons read plainly."""
    a = v.numpy()
    return np.where(a >= UNREACHABLE, -1.0, a)


def test_the_recorded_solve_matches_the_eager_one_bit_for_bit():
    f = _field()
    f.seed_cell(N // 2, N // 2)
    con = _constraints((10, 38, 12, 16))
    f.solve(con)
    recorded = _finite(f.V).copy()
    eager = _finite(
        f.solver.value_iterate(f.blocked, f.penalty, f._seeds, f.penalty_scale, capture=False)
    )
    assert np.array_equal(recorded, eager)


def test_the_graph_is_recorded_once_and_replayed():
    f = _field()
    f.seed_cell(N // 2, N // 2)
    con = _constraints(None)
    f.solve(con)
    first = f.solver._graph
    assert first is not None
    for _ in range(3):
        f.solve(con)
    assert f.solver._graph is first  # replayed, not re-recorded


def test_a_new_map_through_the_same_buffers_gives_a_new_answer():
    # the contract the cache rests on: contents change every frame, pointers do not
    f = _field()
    f.seed_cell(N // 2, N // 2)
    open_ground = f.solve(_constraints(None))
    v_open = _finite(open_ground).copy()
    graph = f.solver._graph

    walled = f.solve(_constraints((0, N, N // 2, N // 2 + 3)))  # wall clean across the map
    v_walled = _finite(walled).copy()

    assert f.solver._graph is graph  # same recording
    assert not np.array_equal(v_open, v_walled)  # but a different answer
    # everything past the wall is unreachable now, and was not before
    assert (v_walled[:, -1, :] < 0).all()
    assert (v_open[:, -1, :] >= 0).all()


def test_the_new_answer_is_the_one_an_uncached_solve_gives():
    f = _field()
    g = _field()  # a second field that sees the walled map FIRST, so it never cached the other
    for h in (f, g):
        h.seed_cell(N // 2, N // 2)
    f.solve(_constraints(None))  # f records its graph on open ground
    walled = _constraints((0, N, N // 2, N // 2 + 3))
    replayed = _finite(f.solve(walled)).copy()
    fresh = _finite(g.solve(walled)).copy()
    assert np.array_equal(replayed, fresh)


def test_changing_the_seeds_is_honoured_by_the_replay():
    f = _field()
    con = _constraints(None)
    f.seed_cell(2, 2)
    corner = _finite(f.solve(con)).copy()
    graph = f.solver._graph
    f.seed_cell(N - 3, N - 3)
    far = _finite(f.solve(con)).copy()
    assert f.solver._graph is graph
    assert not np.array_equal(corner, far)
    assert far[N - 3, N - 3, :].max() == pytest.approx(0.0)
