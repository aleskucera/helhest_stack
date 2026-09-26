"""The recorded solve: it must agree with the eager one, and survive a new map.

Caching a CUDA graph is a contract about what may change between replays. The pointers and the
scalar the graph was recorded with may not; the array CONTENTS must, or the whole thing is
useless -- the map is new every frame. These pin both halves.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from helhest.planning.terrain_value_field import arc_control_set
from helhest.planning.terrain_value_field import closing_step
from helhest.planning.terrain_value_field import TerrainValueField
from helhest.planning.terrain_value_field.field import Constraints

N = 48
CELL = 0.1
NT = 16

pytestmark = pytest.mark.skipif(
    not wp.get_device().is_cuda, reason="graph capture needs a CUDA device"
)


def _field():
    return TerrainValueField(N, N, CELL, n_theta=NT, z_veto=2.0)


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
    eager = _finite(f.solver.value_iterate(f.pose_cost, f._seeds, f.penalty_scale, capture=False))
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


def test_a_finished_solve_reports_convergence():
    f = _field()
    f.seed_cell(N // 2, N // 2)
    f.solve(_constraints(None))
    assert f.converged()
    assert f.solver.bodies_used() < f.solver._cap


def test_a_capped_solve_reports_that_it_did_not_converge():
    """The failure this exists to catch: a capped solve leaves V too high, and 'unreachable'
    then means the same thing it means for a genuinely walled-off map."""
    f = _field()
    f.seed_cell(2, 2)  # corner seed, so the route across the map is long
    f.solver._cap = 2  # set BEFORE the first solve: the cap is baked into the captured graph
    f.solve(_constraints(None))
    assert not f.converged()
    assert f.solver.bodies_used() == 2
    # and the damage is real: most of the map reads unreachable although nothing blocks it
    assert (_finite(f.V) < 0).mean() > 0.5


def test_the_solver_checks_each_swept_cell_at_the_heading_it_is_crossed_at():
    """The defect this closes: an arc that turns 45 degrees was judged entirely at the heading it
    STARTED in, although two thirds of its swept cells are crossed one or two bins later.

    The scene makes every ODD heading bin impassable and every even one clear. From bin 0 the
    sharpest arc ends at bin 2 -- legal at both ends -- but crosses four cells at odd bins in
    between. Honouring `sweep_dt` rejects it; ignoring it lets the robot drive through.
    """
    nt, cell, R = 16, 0.1, 0.5
    step = closing_step(nt, R, 2)
    cs = arc_control_set(nt, cell, step, R, 40, 64)
    blind = (*cs[:7], np.zeros_like(cs[7]), cs[8])  # the same set, heading offsets discarded

    def solve_with(control_set):
        f = TerrainValueField(N, N, cell, n_theta=nt, z_veto=2.0, control_set=control_set)
        m = np.empty((1, N, N, nt), np.float32)
        for t in range(nt):
            m[0, :, :, t] = -1.0 if t % 2 else 1.0  # odd bins impassable, even bins clear
        con = Constraints(
            margin=wp.array(m, dtype=wp.float32),
            sigma=wp.array(np.full((1, N, N, nt), 0.05, np.float32), dtype=wp.float32),
            floor=wp.array([0.01], dtype=wp.float32),
        )
        f.seed_cell(N // 2, N // 2)
        return _finite(f.solve(con)).copy()

    seeing = solve_with(cs)
    blindly = solve_with(blind)
    assert not np.array_equal(seeing, blindly), "sweep_dt is not reaching the kernel"
    # honouring the heading can only refuse moves the blind version allowed, so on this scene it
    # is nowhere cheaper and somewhere dearer -- unreachable (-1) counts as the dearest of all
    reachable_both = (seeing >= 0) & (blindly >= 0)
    assert (seeing[reachable_both] >= blindly[reachable_both] - 1e-5).all()
    assert (seeing < 0).sum() > (blindly < 0).sum(), "and it blocks routes the blind one took"
