"""Point-turn lattice primitives: a goal BEHIND the robot in a corridor too narrow for a
forward-arc U-turn must be unreachable without pivots and routable with them.

Run:  python -m tests.planning.test_pivot   (also collected by pytest)
"""
from __future__ import annotations

import numpy as np
import warp as wp

from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams
from helhest.planning.costtogo import CostToGo

_N_THETA = 16


def _corridor_v(pivot_cost: float, device: str) -> tuple[float, float]:
    """V at the start pose facing AWAY from the goal, and CostToGo's unreachable cap.

    Corridor 2.2 m wide: straight driving and a pivot-in-place fit, but a min-turn-radius
    forward U-turn (center excursion ~2*R = 1 m + envelope margins) does not.
    """
    cell, nx, ny = 0.1, 80, 44
    gp = GridParams(nx, ny, cell, 0.0, -2.2)
    ys = (np.arange(ny) + 0.5) * cell - 2.2
    H = np.zeros((ny, nx), np.float32)
    H[np.abs(ys) > 1.1, :] = 2.0  # walls
    Hd = wp.array(np.ascontiguousarray(H), dtype=wp.float32, device=device)
    ctg = CostToGo(
        gp,
        RobotParams(),
        SolverParams(dt=0.1, k_turn=2.0, newton_iters=6, atol=1e-4),
        n_theta=_N_THETA,
        pivot_cost=pivot_cost,
        device=device,
    )
    V = ctg.compute(Hd, (7.0, 0.0)).numpy()
    r = int(round((0.0 - (-2.2)) / cell))  # y = 0
    c = int(round(1.5 / cell))  # x = 1.5
    t_away = int(np.floor(np.pi / (2.0 * np.pi / _N_THETA))) % _N_THETA  # heading -x
    return float(V[r, c, t_away]), float(ctg._vcap)


def test_pivot_unlocks_goal_behind() -> None:
    wp.init()
    device = "cuda" if wp.get_cuda_device_count() > 0 else "cpu"
    v_off, cap = _corridor_v(0.0, device)
    v_on, _ = _corridor_v(0.3, device)
    assert v_off >= cap * 0.9, f"corridor U-turn should be impossible without pivots (V={v_off})"
    assert v_on < cap * 0.9, f"pivot route should be finite (V={v_on}, cap={cap})"
    # route ~= pivot half-turn (8 bins x 0.3) + ~5.5 m of corridor; generous band for
    # discretization + tilt grading near the walls
    expected = 8 * 0.3 + 5.5
    assert 0.5 * expected < v_on < 2.0 * expected, f"V={v_on} vs expected ~{expected}"


if __name__ == "__main__":
    test_pivot_unlocks_goal_behind()
    print("pivot corridor  OK")
