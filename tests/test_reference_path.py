"""The reference trajectory (studies/closed_loop/reference.py): what the harness scores a run
against. Deterministic, symmetric where the world is, and within its own three bounds."""
from __future__ import annotations

import sys

import numpy as np
import pytest

sys.path.insert(0, "studies/closed_loop")
import reference as R  # noqa: E402

PRM = R.ReferenceParams()


@pytest.fixture(scope="module")
def box() -> R.Reference:
    return R.reference_for_world("box", PRM)


def test_the_box_reference_is_symmetric_about_the_box(box: R.Reference):
    """Start, box and goal lie on one line, so the ideal path's two halves mirror each other."""
    p = box.points
    u = np.linspace(0.5, 5.5, 60)
    near = np.interp(u, p[:, 0], p[:, 1])
    far = np.interp(12.0 - u, p[:, 0], p[:, 1])
    assert np.abs(near - far).mean() < 0.03, "the halves differ by more than 3 cm on average"
    assert np.abs(near - far).max() < 0.08


def test_the_box_reference_keeps_its_bounds(box: R.Reference):
    assert box.clearance.min() >= PRM.clearance_m - 0.02
    assert np.abs(box.kappa).max() <= PRM.kappa_max + 0.05
    rate = np.abs(np.diff(box.kappa)) / PRM.spacing_m
    assert rate.max() <= PRM.kappa_rate_max + 0.3, "curvature changes faster than the clothoid"


def test_the_box_reference_starts_along_the_start_heading_and_ends_at_the_goal(box: R.Reference):
    assert np.allclose(box.points[0], (0.0, 0.0))
    assert np.allclose(box.points[-1], (12.0, 0.0))
    assert abs(box.points[1, 1]) < 1e-6, "the second point is off the start heading"


def test_the_box_reference_is_barely_longer_than_the_taut_string(box: R.Reference):
    """Length dominates the smoothing: the curve rounds the corners, it does not bulge."""
    taut = np.hypot(*np.diff(box.taut, axis=0).T).sum()
    assert taut < box.s[-1] < taut + 0.2  # 11 cm of rounding at R 1.5


def test_the_gap_reference_threads_the_doorway_at_its_middle():
    """The 1.8 m door is narrower than the body plus twice the clearance asked: the reference
    goes through its middle at the clearance the door allows, and keeps the asked one elsewhere."""
    ref = R.reference_for_world("gap", PRM)
    y_at_door = np.interp(6.0, ref.points[:, 0], ref.points[:, 1])
    assert abs(y_at_door) < 0.1
    allowed = 0.5 * 1.8 - R.half_width_m()  # [m] centred in the door
    assert ref.offset_used < PRM.clearance_m + R.half_width_m()
    assert ref.clearance.min() >= allowed - 0.04
    far = np.abs(ref.points[:, 0] - 6.0) > 3.0
    assert ref.clearance[far].min() >= PRM.clearance_m - 0.02


def test_the_same_world_gives_the_same_curve_twice():
    a = R.reference_for_world("box", PRM).points
    b = R.reference_for_world("box", PRM).points
    assert np.allclose(a, b)


def test_the_rear_axle_keeps_the_clearance_too(box: R.Reference):
    """The body is two discs, front and rear axle: on a curve the rear axle runs inside the front
    one's track, so it is the rear disc that sets the offset at an inside corner."""
    rear, _, _ = R.rear_discs(box.points, R.rear_axle_m())
    rear_clear = box.scene.distance(rear)[0] - R.half_width_m()
    assert rear_clear.min() >= PRM.clearance_m - 0.02
    assert R.rear_axle_m() == 0.75
