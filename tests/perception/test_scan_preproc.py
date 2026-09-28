"""The node's scan entry path: `ScanPreprocessor.run` and `transform_points`.

`run` is the first thing every sweep goes through on the robot -- sensor->base transform, the z
crop, the self-footprint box and the range crop, fused into one kernel that compacts survivors in
nondeterministic order. A gate that drops the wrong side of its bound, or a transform applied
transposed, silently reshapes every map downstream, so each is checked here against the same
arithmetic done in numpy. Outputs are compared as sorted row sets because the append order is
atomic.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from helhest.perception import ScanPreprocessor
from helhest.perception import transform_points

# the robot's own gates (ros/config/odin.params.yaml)
SELF_BOX = (-0.05, 0.55, -0.75, 0.75)  # (x_min, x_max, y_min, y_max) [m], base frame
Z_RANGE = (-1.0, 1.5)  # [m], base frame
MAX_RANGE = 8.0  # [m], xy distance in the base frame


def _base_T_sensor() -> np.ndarray:
    """Yawed 90 deg and lifted: a transform that is wrong if applied transposed or not at all."""
    T = np.eye(4)
    T[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    T[:3, 3] = (0.2, -0.1, 0.6)
    return T


def _reference(
    points_sensor: np.ndarray,
    base_T_sensor: np.ndarray,
    z_range: tuple[float, float] | None,
    self_box: tuple[float, float, float, float] | None,
    max_range: float,
) -> np.ndarray:
    p = points_sensor @ base_T_sensor[:3, :3].T + base_T_sensor[:3, 3]
    keep = np.ones(len(p), bool)
    if z_range is not None:
        keep &= (p[:, 2] >= z_range[0]) & (p[:, 2] <= z_range[1])
    if self_box is not None:
        x0, x1, y0, y1 = self_box
        keep &= ~((p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1))
    if max_range > 0.0:
        keep &= p[:, 0] ** 2 + p[:, 1] ** 2 <= max_range**2
    return p[keep]


def _sorted_rows(a: np.ndarray) -> np.ndarray:
    return a[np.lexsort(a.T[::-1])]


def _run(
    points: np.ndarray,
    times: np.ndarray | None = None,
    z_range: tuple[float, float] | None = Z_RANGE,
    self_box: tuple[float, float, float, float] | None = SELF_BOX,
    max_range: float = MAX_RANGE,
) -> tuple[np.ndarray, float, float]:
    pre = ScanPreprocessor(len(points))
    buf, count, _, t_min, t_span = pre.run(
        points, times, _base_T_sensor(), z_range=z_range, self_box=self_box, max_range=max_range
    )
    return buf.numpy()[:count], t_min, t_span


def test_each_gate_rejects_exactly_its_own_point() -> None:
    """One point per rule, placed in the BASE frame and mapped back to the sensor frame."""
    base = np.array(
        [
            [2.0, 1.0, 0.0],  # kept
            [2.0, 1.0, -1.2],  # below the z crop
            [2.0, 1.0, 1.7],  # above the z crop
            [0.3, 0.0, 0.0],  # on the robot's own body
            [9.0, 0.0, 0.0],  # beyond the range crop
            [-0.3, 0.0, 0.0],  # just behind the self box: kept
        ]
    )
    T = _base_T_sensor()
    sensor = (base - T[:3, 3]) @ T[:3, :3]
    got, _, _ = _run(sensor)
    np.testing.assert_allclose(_sorted_rows(got), _sorted_rows(base[[0, 5]]), atol=1e-5)


def test_a_random_sweep_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    sensor = rng.uniform((-10.0, -10.0, -2.0), (10.0, 10.0, 2.0), (20000, 3))
    got, _, _ = _run(sensor)
    want = _reference(sensor, _base_T_sensor(), Z_RANGE, SELF_BOX, MAX_RANGE)
    assert 0 < len(want) < len(sensor)  # every gate has something to do
    assert len(got) == len(want)
    np.testing.assert_allclose(_sorted_rows(got), _sorted_rows(want), atol=1e-5)


def test_disabled_gates_keep_everything() -> None:
    rng = np.random.default_rng(1)
    sensor = rng.uniform(-12.0, 12.0, (5000, 3))
    got, _, _ = _run(sensor, z_range=None, self_box=None, max_range=0.0)
    want = sensor @ _base_T_sensor()[:3, :3].T + _base_T_sensor()[:3, 3]
    np.testing.assert_allclose(_sorted_rows(got), _sorted_rows(want), atol=1e-5)


def test_sweep_time_bounds_come_from_the_survivors_only() -> None:
    """The deskew's alpha is normalised over what survives, so a rejected point's stamp must not
    widen the span."""
    base = np.array([[2.0, 1.0, 0.0], [3.0, -1.0, 0.2], [0.3, 0.0, 0.0], [2.0, 1.0, 5.0]])
    times = np.array([0.02, 0.07, 0.0, 0.1])  # the extremes sit on the two rejected points
    T = _base_T_sensor()
    _, t_min, t_span = _run((base - T[:3, 3]) @ T[:3, :3], times)
    assert abs(t_min - 0.02) < 1e-7 and abs(t_span - 0.05) < 1e-6


def test_no_times_reports_a_zero_span() -> None:
    _, t_min, t_span = _run(np.array([[1.0, 2.0, 0.0], [2.0, 3.0, 0.0]]))
    assert t_min == 0.0 and t_span == 0.0


def test_an_empty_sweep_returns_nothing() -> None:
    got, _, t_span = _run(np.zeros((0, 3)))
    assert len(got) == 0 and t_span == 0.0


def test_transform_points_matches_numpy() -> None:
    rng = np.random.default_rng(2)
    p = rng.uniform(-5.0, 5.0, (1000, 3)).astype(np.float32)
    T = _base_T_sensor()
    got = transform_points(wp.array(p, dtype=wp.vec3), len(p), T).numpy()
    np.testing.assert_allclose(got, p @ T[:3, :3].T + T[:3, 3], atol=1e-5)
