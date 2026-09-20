"""The inpaint's fixed-cell mask, which used to be the one host round trip inside it.

`multigrid_inpaint` is the ground-referenced fill a planner needs for cells it has never seen --
a zero fill reads as flat ground and walls the routing window off where the real ground sits
below zero. In a control loop it runs every frame, so a device->host->device hop to decide which
cells are NaN is a per-frame cost for a question the GPU can answer in place.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from helhest.perception import multigrid_inpaint
from helhest.perception.heightmap.postprocess import _fixed_mask_from


def _holey(rng, n=48):
    h = 0.3 * np.sin(np.linspace(0, 6, n))[None, :] + np.zeros((n, 1), np.float32)
    h = h.astype(np.float32)
    h[rng.random((n, n)) < 0.4] = np.nan
    h[0, :] = 0.0  # keep at least one fixed row so the diffusion has a boundary
    return h


def test_the_mask_is_exactly_isfinite():
    rng = np.random.default_rng(0)
    for _ in range(3):
        a = _holey(rng)
        got = _fixed_mask_from(wp.array(a, dtype=wp.float32)).numpy()
        np.testing.assert_array_equal(got, np.isfinite(a).astype(np.int32))


def test_an_all_nan_map_fixes_nothing_and_an_all_finite_one_fixes_everything():
    """The two ends, because a mask kernel that is wrong only at them still passes a random test."""
    n = 16
    allnan = np.full((n, n), np.nan, np.float32)
    assert _fixed_mask_from(wp.array(allnan, dtype=wp.float32)).numpy().sum() == 0
    solid = np.zeros((n, n), np.float32)
    assert _fixed_mask_from(wp.array(solid, dtype=wp.float32)).numpy().sum() == n * n


def test_inpainting_fills_every_hole_and_leaves_the_measured_cells_alone():
    """What the callers actually rely on, unchanged by how the mask is built."""
    a = _holey(np.random.default_rng(1))
    out = multigrid_inpaint(wp.array(a.copy(), dtype=wp.float32)).numpy()
    assert np.isfinite(out).all(), "a hole survived the inpaint"
    known = np.isfinite(a)
    np.testing.assert_allclose(out[known], a[known], rtol=1e-5, atol=1e-6)


def test_the_numpy_and_device_entry_points_agree():
    a = _holey(np.random.default_rng(2))
    from_np = multigrid_inpaint(a.copy())
    from_wp = multigrid_inpaint(wp.array(a.copy(), dtype=wp.float32)).numpy()
    assert isinstance(from_np, np.ndarray)
    np.testing.assert_allclose(from_np, from_wp, rtol=1e-4, atol=1e-5)
