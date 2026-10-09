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
import pytest
import warp as wp

from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams
from helhest.planning.costtogo import _escape_kernel
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


def test_the_vetoed_field_holds_no_soft_block():
    """What MPPI vetoes hard is `hazard`: on ground where every block is soft, nothing at all."""
    ctg = _solve(_slopes())
    assert not (ctg.hazard.numpy() > 0.5).any() and (ctg.blocked.numpy() > 0.5).any()


def test_an_escape_never_runs_through_a_wall():
    """A routable pose one wall away is no way out; the same pose through a gap in the wall is.

    Straight on the kernel, with the field built by hand: in any settled scene at this resolution
    the routable ground behind a wall lies beyond the escape reach, and the test would pass with
    the line-of-sight check deleted (it did)."""
    ny, nx, nth, cap = 9, 9, 4, 100.0
    V = np.full((ny, nx, nth), cap, np.float32)
    V[:, 6:, :] = 10.0  # routable beyond the wall ...
    solid = np.zeros((ny, nx), np.float32)
    solid[:, 5] = 1.0  # ... a wall one cell thick ...
    solid[4, 5] = 0.0  # ... with a gap in row 4
    out = wp.zeros((ny, nx, nth), dtype=wp.float32)
    wp.launch(
        _escape_kernel,
        dim=(ny, nx, nth),
        inputs=[wp.array(V), wp.array(solid), cap, 3, 1.0, 1.0, 0.0],
        outputs=[out],
    )
    E = out.numpy()
    assert (E[0, 3, :] >= 0.9 * cap).all(), "escaped through the wall"
    # through the gap: 2 cells straight along row 4 to column 6, at 1 per cell
    np.testing.assert_allclose(E[4, 4, :], 10.0 + 2.0)


def test_a_no_route_pose_escapes_to_the_cheapest_routable_one_nearby():
    ctg = _solve(_wall())
    V, E = ctg.V.numpy(), ctg.V_escape.numpy()
    cap = ctg._vcap
    lim = 0.9 * cap
    routable = V < lim
    assert routable.any() and (~routable).any()
    np.testing.assert_array_equal(E[routable], V[routable])  # a pose with a route is untouched

    # brute force over the no-route poses, against the kernel
    ny, nx, nth = V.shape
    K, cell = ctg._escape_reach, ctg.grid.cell_size
    rng = np.random.default_rng(0)
    capped = np.argwhere(~routable)
    for r, c, t in capped[rng.choice(len(capped), size=min(300, len(capped)), replace=False)]:
        best = cap
        for i in range(-K, K + 1):
            for j in range(-K, K + 1):
                rr, cc = r + i, c + j
                if not (0 <= rr < ny and 0 <= cc < nx) or i * i + j * j > K * K:
                    continue
                for k in range(nth):
                    if V[rr, cc, k] < lim:
                        dk = abs(k - t)
                        turn = min(dk, nth - dk)
                        cand = (
                            V[rr, cc, k]
                            + ctg.ESCAPE_PER_M * cell * np.sqrt(i * i + j * j)
                            + ctg._escape_per_bin * turn
                        )
                        best = min(best, cand)
        assert E[r, c, t] == pytest.approx(min(best, cap), rel=1e-5, abs=1e-4)

    # Both outcomes occur: poses in the wall's margin on the goal's side find a way out, and the
    # far side -- cut off entirely, the wall spans the window -- keeps the cap, so the straight-line
    # exploration fallback still arms where there is genuinely no route.
    escaped = ~routable & (E < lim)
    assert escaped.any() and (~routable & (E >= lim)).any()
