"""The route priced in travel time under the clearance speed law, and its turn price.

With the clearance law on (`clearance`), the spatial tube no longer removes poses near walls:
the route uses the heading bin alone and charges every pose the time the law would cost there. A
2.4 m corridor the old veto closes off-centre must stay routable, charged more toward the walls.
The turn price (`ClearanceParams.route_turn`) adds the tail swing's time to turning arcs near walls.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams
from helhest.planner_config import planner_config
from helhest.planning.clearance import ClearanceParams
from helhest.planning.costtogo import CostToGo

N = 66
CELL = 0.24
CLEAR = 2.4


def _corridor() -> np.ndarray:
    ys = (np.arange(N) - N // 2) * CELL
    t = np.full((N, N), -0.5, np.float32)
    t[np.abs(ys) > CLEAR / 2, :] = 0.5  # 1 m walls either side of a corridor along +x
    return t


def _solve_time() -> CostToGo:
    ctg = CostToGo(
        GridParams(cells_x=N, cells_y=N, cell_size=CELL, origin_x=0.0, origin_y=0.0),
        RobotParams(wheel_width=0.1),
        SolverParams(),
        n_theta=24,
        robust_margin_m=0.2,
        robust_margin_deg=15.0,
        clearance=ClearanceParams(t_react=0.5, c0=0.0, route_turn=False),
    )
    goal = ((N // 2) * CELL + 6.0, (N // 2) * CELL)
    ctg.compute(wp.array(_corridor(), dtype=wp.float32), goal)
    return ctg


def test_the_time_price_keeps_the_corridor_routable_and_charges_toward_the_walls():
    ctg = _solve_time()
    cap = ctg._vcap
    r, c = N // 2 - 2, N // 2  # 0.48 m off the centre line: dead under the veto
    assert (ctg.V.numpy()[r, c] < 0.9 * cap).any()
    charge = ctg._loose_tilt.numpy()[:, c, 0]
    free = ctg._loose_blocked.numpy()[:, c, 0] < 0.5
    rows = [q for q in range(N // 2 - 4, N // 2 + 1) if free[q]]
    seq = [charge[q] for q in rows]
    assert all(a >= b - 1e-6 for a, b in zip(seq, seq[1:])) and seq[0] > seq[-1]
    # the centre line is past the law's reach: no charge. The first free pose from the wall is at
    # most two cells from a contact pose, so it pays at least the two-cell multiplier
    # v_cruise / v = 1.5 / (0.48 / 0.5), as a penalty (multiplier - 1) / flatness_weight
    assert seq[-1] < 1e-3
    assert seq[0] >= (1.5 / (2 * CELL / 0.5) - 1.0) / ctg.flatness_weight - 1e-4


def test_the_route_turn_price_charges_turning_next_to_walls_only():
    """With the turn price on, V can only rise (it adds cost to turning arcs, never removes any),
    and it rises by more beside the walls than on the centre line."""

    def solve(ratio: float) -> CostToGo:
        ctg = CostToGo(
            GridParams(cells_x=N, cells_y=N, cell_size=CELL, origin_x=0.0, origin_y=0.0),
            RobotParams(wheel_width=0.1),
            SolverParams(),
            n_theta=24,
            robust_margin_m=0.2,
            robust_margin_deg=15.0,
            clearance=ClearanceParams(c0=0.1, t_turn=0.125 * max(ratio, 1.0), route_turn=ratio > 0),
        )
        ctg.compute(
            wp.array(_corridor(), dtype=wp.float32), ((N // 2) * CELL + 6.0, (N // 2) * CELL)
        )
        return ctg

    off, on = solve(0.0), solve(2.0)
    v0, v1 = off.V.numpy(), on.V.numpy()
    ok = (v0 < 0.9 * off._vcap) & (v1 < 0.9 * on._vcap)
    assert (v1[ok] >= v0[ok] - 1e-4).all()
    assert (v1[ok] > v0[ok] + 1e-3).any()
    assert on.clearance_route.turn_time.numpy().max() > 0.0
    assert off.clearance_route.turn_time.numpy().max() == 0.0


def test_off_by_default_and_complete_when_on():
    off = planner_config({})
    assert off.clearance is None and "clearance" not in off.costtogo and off.cost.clearance is None
    on = planner_config({"plan_clear_t_react": 0.125})
    c = on.clearance
    assert (c.t_react, c.c0, c.turn_ratio) == (0.125, 0.15, 2.0) and c.route_turn
    assert on.costtogo["clearance"] is c and on.cost.clearance is c
