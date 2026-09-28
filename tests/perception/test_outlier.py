"""The node's `_denoise`: `StatisticalOutlierFilter` at the robot's settings.

What it must do: strip isolated specks floating above the ground, and NOT strip distant ground
just because a lidar's returns thin out with range. The second is the reason the mean neighbour
distance is divided by the range from `sensor_origin`, and why the node sets that origin every
call -- so the test places a dense near patch and a sparse far patch whose spacing grows with
range, and checks the far one survives only when the origin is the real one.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from helhest.perception import OutlierFilterConfig
from helhest.perception import StatisticalOutlierFilter

SENSOR = (0.0, 0.0, 0.5)  # [m] sensor origin, base frame


def _config(sensor_origin: tuple[float, float, float] = SENSOR) -> OutlierFilterConfig:
    # the node's defaults (navigation_node outlier_*)
    return OutlierFilterConfig(
        search_radius_m=0.25, min_neighbors=6, std_multiplier=1.0, sensor_origin=sensor_origin
    )


def _patch(range_m: float, spacing_m: float) -> np.ndarray:
    """A flat 1.5 m square of ground centred `range_m` ahead of the sensor."""
    xs = np.arange(range_m - 0.75, range_m + 0.75, spacing_m)
    ys = np.arange(-0.75, 0.75, spacing_m)
    X, Y = np.meshgrid(xs, ys)
    return np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)].astype(np.float32)


def _scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    near = _patch(1.5, 0.03)
    far = _patch(5.0, 0.10)  # ~3x the spacing at ~3x the range
    specks = np.c_[
        rng.uniform(1.0, 5.0, 40), rng.uniform(-0.7, 0.7, 40), rng.uniform(0.6, 1.5, 40)
    ].astype(np.float32)
    return near, far, specks


def _kept_fraction(part: np.ndarray, out: np.ndarray) -> float:
    kept = {tuple(p) for p in np.round(out, 4)}
    return float(np.mean([tuple(p) in kept for p in np.round(part, 4)]))


def _filter(points: np.ndarray, sensor_origin: tuple[float, float, float] = SENSOR) -> np.ndarray:
    f = StatisticalOutlierFilter(_config(sensor_origin))
    return f.apply(wp.array(points, dtype=wp.vec3)).numpy()


def test_specks_go_and_the_ground_stays() -> None:
    near, far, specks = _scene()
    out = _filter(np.r_[near, far, specks])
    assert _kept_fraction(specks, out) == 0.0
    # mean + 1 sigma trims the tail by design (patch edges); the bulk must survive
    assert _kept_fraction(near, out) > 0.75
    assert _kept_fraction(far, out) > 0.9


def test_the_range_normalisation_is_what_keeps_the_far_ground() -> None:
    near, far, specks = _scene()
    pts = np.r_[near, far, specks]
    at_sensor = _kept_fraction(far, _filter(pts))
    # an origin 100 m behind makes the divisor ~constant: raw spacing decides, and far ground loses
    unnormalised = _kept_fraction(far, _filter(pts, sensor_origin=(-100.0, 0.0, 0.5)))
    assert at_sensor > unnormalised + 0.3, (at_sensor, unnormalised)


def test_device_in_device_out_and_numpy_in_numpy_out() -> None:
    near, _, _ = _scene()
    f = StatisticalOutlierFilter(_config())
    assert isinstance(f.apply(wp.array(near, dtype=wp.vec3)), wp.array)
    assert isinstance(f.apply(near), np.ndarray)


def test_too_few_points_pass_through_untouched() -> None:
    few = np.array([[1.0, 0.0, 0.0], [5.0, 5.0, 3.0]], np.float32)
    np.testing.assert_array_equal(StatisticalOutlierFilter(_config()).apply(few), few)
