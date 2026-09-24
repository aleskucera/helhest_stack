"""The robust tube splits by cause: walls and unresolved settles are eroded HARD, tilt and belly
clearance are CHARGED.

Eroding every blocked pose by the (y, x, theta) box took `bumpy` from 6/6 reached to 2/6 --
rough ground speckles the tilt envelope and puts mound tops under the belly, and a 27-pose box
around each leaves little route. But the tube exists so the robot never drives into a wall, and
to the settle a wall is not always a hazard: a wheel lifted onto its edge reads as TILT, a wall
under the body as belly clearance. So a soft block over a face the wheel cannot mount is re-filed
as a hazard, and the wall scene below must erode exactly as the old box did.
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
GOAL = (N * CELL / 2 - CELL, 0.0)
WEIGHT = 10.0


def _grid() -> GridParams:
    return GridParams(
        cells_x=N, cells_y=N, cell_size=CELL, origin_x=-N * CELL / 2, origin_y=-N * CELL / 2
    )


def _solve(terrain: np.ndarray) -> CostToGo:
    ctg = CostToGo(
        _grid(),
        RobotParams(),
        SolverParams(),
        n_theta=12,
        robust_margin_m=CELL,
        robust_margin_deg=30.0,
        robust_soft_weight=WEIGHT,
    )
    ctg.compute(wp.array(terrain.astype(np.float32), dtype=wp.float32), GOAL)
    return ctg


def _box_max(field: np.ndarray, dr: int, dt: int) -> np.ndarray:
    """Max over the tube: rows/cols clamped at the edge, heading wrapping."""
    ny, nx, _ = field.shape
    padded = np.pad(field, ((dr, dr), (dr, dr), (0, 0)), mode="edge")
    out = np.full_like(field, -np.inf)
    for i in range(-dr, dr + 1):
        for j in range(-dr, dr + 1):
            shifted = padded[dr + i : dr + i + ny, dr + j : dr + j + nx]
            for k in range(-dt, dt + 1):
                out = np.maximum(out, np.roll(shifted, -k, axis=2))
    return out


def _wall() -> np.ndarray:
    t = np.zeros((N, N))
    t[:, N // 2 : N // 2 + 2] = 1.0  # a 1 m wall across the window
    return t


def _slopes() -> np.ndarray:
    xs = np.linspace(-N * CELL / 2, N * CELL / 2, N)
    return 0.55 * np.sin(xs[None, :] * 0.8) + 0.44 * np.cos(xs[:, None] * 0.6)


def test_a_wall_erodes_exactly_as_the_old_box_did():
    ctg = _solve(_wall())
    blocked = ctg.blocked.numpy() > 0.5
    hazard = ctg.hazard.numpy() > 0.5
    assert blocked.any()
    # the wheel on the wall's edge tilts the settle, the wall under the body reads as belly
    # clearance; the face re-files every one of those
    assert np.array_equal(hazard, blocked), "a wall pose was left soft, and would be charged"
    old = _box_max(blocked.astype(np.float32), ctg._mr, ctg._mt) > 0.5
    assert np.array_equal(ctg.robust_blocked.numpy() > 0.5, old)


def test_soft_blocks_are_charged_by_how_far_over_not_eroded():
    ctg = _solve(_slopes())
    blocked = ctg.blocked.numpy() > 0.5
    hazard = ctg.hazard.numpy() > 0.5
    soft = blocked & ~hazard
    assert soft.any(), "the scene no longer grazes the envelope: the test would be vacuous"

    robust = ctg.robust_blocked.numpy() > 0.5
    want = blocked | (_box_max(hazard.astype(np.float32), ctg._mr, ctg._mt) > 0.5)
    assert np.array_equal(robust, want), "only hazards may spread; soft blocks veto the pose alone"

    charge = ctg.robust_tilt.numpy() - ctg.graded_tilt.numpy()
    excess = _box_max(ctg.violation.numpy(), ctg._mr, ctg._mt)
    np.testing.assert_allclose(charge, WEIGHT * excess, atol=1e-5)
    # the poses the old box would have closed are open, and pay instead
    spared = ~robust & (_box_max(blocked.astype(np.float32), ctg._mr, ctg._mt) > 0.5)
    assert spared.any() and (charge[spared] > 0.0).all()
