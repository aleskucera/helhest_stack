"""The step gate blocks a pole the settle straddles, at every heading, and nothing else.

It went untested until the reverse-study strip deleted `_foot_r` with the drop gate's lines, and
every `obstacle_step_m > 0` configuration then crashed on its first compute.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams
from helhest.planning.costtogo import CostToGo

N = 41
CELL = 0.24


def _blocked(step_m: float) -> np.ndarray:
    grid = GridParams(N, N, CELL, -N * CELL / 2, -N * CELL / 2)
    ctg = CostToGo(grid, RobotParams(), SolverParams(), n_theta=12, obstacle_step_m=step_m)
    terrain = np.zeros((N, N), np.float32)
    terrain[N // 2, N // 2] = 0.8  # one cell: a pole thinner than the gap between the supports
    ctg.compute(wp.array(terrain, dtype=wp.float32, device=ctg.device), (N * CELL / 2 - CELL, 0.0))
    return ctg.blocked.numpy()


def test_pole_is_blocked_at_every_heading_only_with_the_gate() -> None:
    gated = _blocked(0.3)
    assert gated[N // 2, N // 2].all()
    assert not gated[2, 2].any()  # open ground far from the pole stays free
    assert gated.sum() > _blocked(0.0).sum()
