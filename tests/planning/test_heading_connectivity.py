"""Every heading bin must be reachable, or half the lattice is unreachable on open ground.

A lattice arc has to CLOSE on a heading bin, and `closing_step` is what guarantees it. Closing is
not sufficient. The arcs turn by {0, +-bins/2, +-bins} bins, so repeated they reach only the
multiples of bins/2 -- every heading iff gcd(bins/2, n_theta) == 1. The point turns are the only
+-1 move and this planner prices them out, so a step that closes can still split the heading ring
into gcd(bins/2, n_theta) rings that never meet.

That shipped. `min_turn_radius=0.5`, `n_theta=16`, `cell=0.2` put `closing_step(bins=2)` at
0.3927 m against a 0.40 m "clears a couple of cells" guard -- a 2% miss -- so the search
escalated to bins=4, whose turns {2, 4} generate the EVEN bins and nothing else. The odd half of
every cell sat at the value cap with `blocked` all zero: not vetoed, never visited. It cost
`pocket` the run, because the bearing to its goal fell on bin 9, so the field called the one
heading pointing at the goal a dead end while the ground in between was flat and clear.

The failure is silent by construction -- the goal cell is seeded at every heading, so states on
the orphaned ring still take a value wherever they can drive straight in, and the field degrades
instead of erroring. Hence a test that looks at the whole ring on ground with nothing in it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from helhest import dynamics
from helhest.engine import GridParams
from helhest.planning.costtogo import CostToGo

N = 41
CELL = 0.2
NT = 16


def _open_ground() -> wp.array:
    return wp.zeros((N, N), dtype=wp.float32)


def _ctg(**kw) -> CostToGo:
    grid = GridParams(
        cells_x=N, cells_y=N, cell_size=CELL, origin_x=-N * CELL / 2, origin_y=-N * CELL / 2
    )
    return CostToGo(
        grid,
        dynamics.robot_params(0.10),
        dynamics.planning_solver(dt=1.0 / 14.5, command_delay=0.0),
        n_theta=NT,
        device="cuda",
        **kw,
    )


def _finite_by_parity(ctg: CostToGo) -> tuple[float, float]:
    """Fraction of states with a route, split by heading parity -- the signature of the split."""
    ctg.compute(_open_ground(), (N * CELL / 2 - CELL, 0.0))
    v = ctg.V.numpy()
    reach = v < v.max() * 0.9
    return float(reach[:, :, 0::2].mean()), float(reach[:, :, 1::2].mean())


def test_the_default_step_reaches_every_heading():
    """The regression guard. On ground with nothing in it, no heading is a dead end."""
    even, odd = _finite_by_parity(_ctg())
    assert odd > 0.9 * even, f"odd headings reachable {odd:.1%} vs even {even:.1%}"


def test_the_chosen_step_generates_the_whole_heading_ring():
    """Stated as the arithmetic, so a future change to the search is checked against the reason
    rather than against a number that happened to work."""
    ctg = _ctg()
    bins = round(ctg.step / (ctg.robot.min_turn_radius * 2.0 * math.pi / NT))
    assert bins % 2 == 0, f"half-rate arcs land off-bin at bins={bins}"
    assert math.gcd(bins // 2, NT) == 1, (
        f"bins={bins} turns by multiples of {bins // 2}, which reaches only "
        f"1 heading in {math.gcd(bins // 2, NT)} of {NT}"
    )


def test_the_arc_must_still_leave_its_own_cell():
    """The constraint the connectivity rule has to coexist with: arcs that stay inside one cell
    cannot propagate. The bound is the cell diagonal -- the farthest a point in a cell can be
    from its centre -- not a round number of cells."""
    assert _ctg().step >= math.sqrt(2.0) * CELL


def test_a_disconnected_step_is_refused_rather_than_solved():
    """bins=4 at n_theta=16: it closes, and it splits the ring in two. The control set rejects it
    instead of handing the solver a table in which half the states cannot be entered."""
    with pytest.raises(ValueError, match="disconnected"):
        _ctg(step=0.5 * (2.0 * math.pi / NT) * 4)


def test_point_turns_reconnect_a_split_ring():
    """Why the fix does not live in `pivot_cost`: the +-1 move does bridge the split, so
    connectivity would silently depend on a price. Pinned so that stays a deliberate choice."""
    even, odd = _finite_by_parity(_ctg(step=0.5 * (2.0 * math.pi / NT) * 4, pivot_cost=0.5))
    assert odd > 0.9 * even
