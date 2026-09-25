"""Slow in narrow places: the spatial tube MARKS instead of vetoing.

With `narrow_cost` set, the router routes on the heading bin alone and a pose only the spatial
tube removes is marked narrow and charged. A 2.4 m corridor the deployed tube closes off-centre
must stay routable, with exactly those poses marked.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from helhest.control.mppi import CostParams
from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams
from helhest.planner_config import planner_config
from helhest.planning.costtogo import CostToGo

N = 66
CELL = 0.24
CLEAR = 2.4


def _corridor() -> np.ndarray:
    ys = (np.arange(N) - N // 2) * CELL
    t = np.full((N, N), -0.5, np.float32)
    t[np.abs(ys) > CLEAR / 2, :] = 0.5  # 1 m walls either side of a corridor along +x
    return t


def _solve(narrow_cost: float | None) -> CostToGo:
    ctg = CostToGo(
        GridParams(cells_x=N, cells_y=N, cell_size=CELL, origin_x=0.0, origin_y=0.0),
        RobotParams(wheel_width=0.1),
        SolverParams(),
        n_theta=24,
        robust_margin_m=0.2,
        robust_margin_deg=15.0,
        narrow_cost=narrow_cost,
    )
    goal = ((N // 2) * CELL + 6.0, (N // 2) * CELL)
    ctg.compute(wp.array(_corridor(), dtype=wp.float32), goal)
    return ctg


def test_narrow_marks_what_the_spatial_tube_would_veto_and_keeps_the_route():
    veto = _solve(None)
    marked = _solve(0.15)
    cap = veto._vcap
    r, c = N // 2 - 2, N // 2  # 0.48 m off the centre line: dead under the deployed tube
    assert (veto.V.numpy()[r, c] >= 0.9 * cap).all(), "the scene no longer shows the dead field"
    assert (marked.V.numpy()[r, c] < 0.9 * cap).any(), "narrow mode left the pose unroutable"
    narrow = marked.narrow.numpy() > 0.5
    strict = marked.robust_blocked.numpy() > 0.5
    loose = marked._loose_blocked.numpy() > 0.5
    assert narrow.any()
    assert np.array_equal(narrow, strict & ~loose)
    # the veto path is untouched: no narrow poses, the same V as before the option existed
    assert not (veto.narrow.numpy() > 0.5).any()


def test_off_by_default_everywhere():
    assert CostParams().build().narrow == 0.0
    cfg = planner_config({})
    assert cfg.cost.narrow == 0.0 and "narrow_cost" not in cfg.costtogo
    on = planner_config({"plan_narrow_speed": 0.4})
    assert on.cost.narrow > 0.0 and on.cost.narrow_speed == 0.4
    assert on.costtogo["narrow_cost"] == 0.15 and on.costtogo["narrow_reach_m"] == 0.6


def test_the_route_charge_grades_toward_the_middle():
    ctg = _solve(0.15)
    free = ctg._loose_blocked.numpy() < 0.5
    charge = ctg._loose_tilt.numpy()[:, N // 2, 0]  # heading along the corridor, one column
    rows = [r for r in range(N // 2 - 4, N // 2 + 1) if free[r, N // 2, 0]]
    # from the wall side toward the centre line the charge never rises, and it does fall
    seq = [charge[r] for r in rows]
    assert all(a >= b - 1e-6 for a, b in zip(seq, seq[1:])) and seq[0] > seq[-1]
