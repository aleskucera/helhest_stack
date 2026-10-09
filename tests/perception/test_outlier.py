"""The node's `_denoise`: `StatisticalOutlierFilter` at the robot's settings.

What it must do: strip isolated specks floating above the ground, and keep the ground at every
range -- dense in front of the robot, thinned out far away. The near patch is the regression: a
range-normalised mean-distance test once removed every return inside ~1 m.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from helhest.perception import OutlierFilterConfig
from helhest.perception import StatisticalOutlierFilter


def _config() -> OutlierFilterConfig:
    # the node's defaults (navigation_node outlier_*)
    return OutlierFilterConfig(search_radius_m=0.25, min_neighbors=6)


def _patch(range_m: float, spacing_m: float) -> np.ndarray:
    """A flat 1.5 m square of ground centred `range_m` ahead of the sensor."""
    xs = np.arange(range_m - 0.75, range_m + 0.75, spacing_m)
    ys = np.arange(-0.75, 0.75, spacing_m)
    X, Y = np.meshgrid(xs, ys)
    return np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)].astype(np.float32)


def _scene() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    close = _patch(0.9, 0.02)  # right in front of the robot, densest of all
    near = _patch(3.0, 0.03)
    far = _patch(8.0, 0.10)  # returns thin out with range
    specks = np.c_[
        rng.uniform(1.0, 8.0, 40), rng.uniform(-0.7, 0.7, 40), rng.uniform(0.6, 1.5, 40)
    ].astype(np.float32)
    return close, near, far, specks


def _kept_fraction(part: np.ndarray, out: np.ndarray) -> float:
    kept = {tuple(p) for p in np.round(out, 4)}
    return float(np.mean([tuple(p) in kept for p in np.round(part, 4)]))


def _filter(points: np.ndarray) -> np.ndarray:
    return StatisticalOutlierFilter(_config()).apply(wp.array(points, dtype=wp.vec3)).numpy()


def test_specks_go_and_the_ground_stays_at_every_range() -> None:
    close, near, far, specks = _scene()
    out = _filter(np.r_[close, near, far, specks])
    assert _kept_fraction(specks, out) == 0.0
    assert _kept_fraction(close, out) == 1.0, "ground in front of the robot must reach the map"
    assert _kept_fraction(near, out) == 1.0
    assert _kept_fraction(far, out) == 1.0


def test_device_in_device_out_and_numpy_in_numpy_out() -> None:
    _, near, _, _ = _scene()
    f = StatisticalOutlierFilter(_config())
    assert isinstance(f.apply(wp.array(near, dtype=wp.vec3)), wp.array)
    assert isinstance(f.apply(near), np.ndarray)


def test_too_few_points_pass_through_untouched() -> None:
    few = np.array([[1.0, 0.0, 0.0], [5.0, 5.0, 3.0]], np.float32)
    np.testing.assert_array_equal(StatisticalOutlierFilter(_config()).apply(few), few)
