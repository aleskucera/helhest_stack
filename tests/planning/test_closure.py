"""The lattice closes: every arc ends on a heading bin the robot actually reaches.

An arc is integrated in continuous space and then snapped to the grid, so the heading the table
records is the one the robot reaches only if the turn lands on a bin boundary. When it does not,
the recorded end heading is wrong by up to half a bin on EVERY move -- and since feasibility is
indexed by heading, it then gets evaluated at a pose the robot will not occupy. Distinct turn
rates also collapse onto one lattice transition, so the solver relaxes duplicates.

This never held here until the step was derived rather than picked: at n_theta=12, step=0.72 the
error was 15.0 deg per move, and at n_theta=16, step=0.30 it was 11.2 deg.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams
from helhest.planning.costtogo import CostToGo

# (24, 0.50) is NOT here: it has no connected lattice at all, which is its own test below.
CONFIGS = [(12, 0.24), (16, 0.24), (24, 0.10), (24, 0.20), (32, 0.15)]


def _ctg(n_theta: int, cell: float, n: int = 25) -> CostToGo:
    grid = GridParams(
        cells_x=n, cells_y=n, cell_size=cell, origin_x=-n * cell / 2, origin_y=-n * cell / 2
    )
    return CostToGo(grid, RobotParams(), SolverParams(), n_theta=n_theta)


@pytest.mark.parametrize("n_theta,cell", CONFIGS)
def test_the_default_step_closes_on_the_lattice(n_theta, cell):
    """`arc_control_set` warns when handed a step that does not close, so nothing is enough."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ctg(n_theta, cell)


@pytest.mark.parametrize("n_theta,cell", CONFIGS)
def test_every_arc_records_the_heading_it_actually_reaches(n_theta, cell):
    """Checked against the arcs' own turn rates rather than against the warning."""
    ctg = _ctg(n_theta, cell)
    head = ctg.solver._prim_heading.numpy()
    dth = 2.0 * math.pi / n_theta
    sharpest = ctg.step / RobotParams().min_turn_radius
    for it in range(n_theta):
        for p, frac in enumerate((-1.0, -0.5, 0.0, 0.5, 1.0)):
            recorded = ((head[it, p] - it + n_theta // 2) % n_theta - n_theta // 2) * dth
            assert recorded == pytest.approx(frac * sharpest, abs=1e-9)


@pytest.mark.parametrize("n_theta,cell", CONFIGS)
def test_no_two_arcs_collapse_onto_the_same_move(n_theta, cell):
    ctg = _ctg(n_theta, cell)
    dr, dc = ctg.solver._prim_dr.numpy(), ctg.solver._prim_dc.numpy()
    head = ctg.solver._prim_heading.numpy()
    for it in range(n_theta):
        moves = {(int(dr[it, p]), int(dc[it, p]), int(head[it, p])) for p in range(5)}
        assert len(moves) == 5, f"bin {it} has duplicate arcs"


@pytest.mark.parametrize("n_theta,cell", CONFIGS)
def test_no_single_primitive_turns_more_than_a_quarter_circle(n_theta, cell):
    """Past 90 degrees an arc is a committed manoeuvre, awkward to collision-check as one unit
    and coarse enough to route with that the lattice stops earning its cost."""
    ctg = _ctg(n_theta, cell)
    assert math.degrees(ctg.step / RobotParams().min_turn_radius) <= 90.0 + 1e-6


@pytest.mark.parametrize("n_theta,cell", CONFIGS)
def test_every_arc_moves_at_least_one_cell(n_theta, cell):
    """An arc snapping back onto its own state is a self-loop: nothing propagates, and the solve
    reports the goal unreachable on open ground."""
    ctg = _ctg(n_theta, cell)
    dr, dc = ctg.solver._prim_dr.numpy(), ctg.solver._prim_dc.numpy()
    assert (np.abs(dr[:, :5]) + np.abs(dc[:, :5]) > 0).all()


def test_an_explicit_non_closing_step_still_warns():
    """The default is derived, but a caller can still pin one -- and should be told."""
    with pytest.warns(UserWarning, match="does not close"):
        grid = GridParams(cells_x=25, cells_y=25, cell_size=0.24, origin_x=-3.0, origin_y=-3.0)
        CostToGo(grid, RobotParams(), SolverParams(), n_theta=12, step=0.72)


def test_a_config_with_no_connected_lattice_says_so():
    """n_theta=24 on 0.50 m cells: closing and connected cannot both be had under a quarter turn.

    bins=2 turns 30 deg over 0.2618 m, which is half a cell and snaps back onto its own state.
    The next two closing steps reach the grid but not the whole heading ring -- bins=4 turns by
    {2, 4} of 24 and bins=6 by {3, 6}, generating every 2nd and every 3rd heading. So there is
    nothing to choose, and the honest answers are a coarser heading ring or the point turns the
    skid-steer actually has.

    This config used to be in CONFIGS and used to pass. It picked bins=6 and shipped a lattice in
    THREE disconnected pieces, with two thirds of every cell's headings unreachable on open
    ground -- and the suite was satisfied because closure was all anyone checked.
    """
    grid = GridParams(cells_x=25, cells_y=25, cell_size=0.50, origin_x=-6.25, origin_y=-6.25)
    with pytest.raises(ValueError, match="no connected lattice"):
        CostToGo(grid, RobotParams(), SolverParams(), n_theta=24)
