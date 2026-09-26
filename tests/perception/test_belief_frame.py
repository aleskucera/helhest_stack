"""BeliefFrame is drive_sim's belief sequence, bit for bit.

The reference below is drive_sim's per-frame code as it stood before the helper existed
(studies/closed_loop/drive_sim.py at 429d427): every sim result so far was produced by it, and the
node must plan on exactly what those results tested.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

pytest.importorskip("elevation_belief")

from elevation_belief import DriftRates  # noqa: E402
from elevation_belief import ElevationBelief  # noqa: E402
from elevation_belief import NoiseModel  # noqa: E402

from helhest.perception import multigrid_inpaint  # noqa: E402
from helhest.perception.belief_frame import BeliefFrame  # noqa: E402

SPAN, CELL, DT = 8.0, 0.2, 1.0 / 14.5


def _scans(n_frames: int = 6) -> list[tuple[np.ndarray, np.ndarray, tuple[float, float]]]:
    """A sensor driving along x over a bumpy floor with a box, sampled like a ray fan."""
    rng = np.random.default_rng(0)
    out = []
    for f in range(n_frames):
        rx, ry = 0.3 * f, 0.05 * f
        origin = np.array([rx, ry, 0.9], np.float32)
        xy = rng.uniform(-3.5, 3.5, (6000, 2)) + np.array([rx, ry])
        z = 0.1 * np.sin(1.3 * xy[:, 0]) * np.cos(0.9 * xy[:, 1])
        box = (np.abs(xy[:, 0] - 2.0) < 0.4) & (np.abs(xy[:, 1]) < 0.4)
        z = np.where(box, 0.6, z) + rng.normal(0.0, 0.01, len(z))
        pts = np.column_stack([xy, z]).astype(np.float32)
        out.append((pts, origin, (rx, ry)))
    return out


@wp.kernel
def _measured_ref(valid: wp.array2d(dtype=wp.int32), out: wp.array2d(dtype=wp.float32)):
    i, j = wp.tid()
    out[i, j] = wp.where(valid[i, j] != 0, 1.0, 0.0)


@wp.kernel
def _sd_ref(var: wp.array2d(dtype=wp.float32), out: wp.array2d(dtype=wp.float32)):
    i, j = wp.tid()
    out[i, j] = wp.sqrt(wp.max(var[i, j], 0.0))


def _reference(scans) -> list[np.ndarray]:
    """drive_sim's inline sequence at 429d427."""
    belief = ElevationBelief(
        (-SPAN / 2, SPAN / 2, -SPAN / 2, SPAN / 2),
        CELL,
        noise=NoiseModel("linear", a=0.012, b=0.004),
        rates=DriftRates.odin_slam(),
    )
    n = belief.nx
    scratch = wp.zeros((n, n), dtype=wp.float32)
    measured = wp.zeros((n, n), dtype=wp.float32)
    sd = wp.zeros((n, n), dtype=wp.float32)
    for f, (pts, origin, xy) in enumerate(scans):
        pts_d = wp.array(pts, dtype=wp.vec3f)
        belief.recenter(xy)
        if f:
            belief.motion_update(DT, xy)
        belief.carve(pts_d, origin, max_range=6.0)
        belief.measure_scan(pts_d, origin)
    lay = belief.layers()
    wp.copy(scratch, lay["raw_h"])
    height = multigrid_inpaint(scratch)
    wp.launch(_measured_ref, dim=(n, n), inputs=[lay["valid"]], outputs=[measured])
    wp.launch(_sd_ref, dim=(n, n), inputs=[lay["meas_var"]], outputs=[sd])
    return [height.numpy(), measured.numpy(), sd.numpy(), belief.drift().numpy()]


def test_belief_frame_is_drive_sims_sequence_bit_for_bit():
    scans = _scans()
    frame = BeliefFrame((-SPAN / 2, SPAN / 2, -SPAN / 2, SPAN / 2), CELL, carve_range=6.0)
    for pts, origin, xy in scans:
        frame.update(wp.array(pts, dtype=wp.vec3f), origin, xy, DT)
    got = [a.numpy() for a in frame.layers()]
    want = _reference(scans)
    for name, g, w in zip(("height", "measured", "sd", "drift"), got, want):
        np.testing.assert_array_equal(g, w, err_msg=name)
    assert got[1].mean() > 0.3, "the scans must actually cover the window"
    assert got[0].max() > 0.4, "and the box must be in the map"
