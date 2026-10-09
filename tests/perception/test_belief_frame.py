"""BeliefFrame is drive_sim's belief sequence, and fills what the belief never measured.

The reference below is drive_sim's per-frame code as it stood before the helper existed
(studies/closed_loop/drive_sim.py at 429d427). The helper matches it bit for bit on every measured
cell and on the mask, sd and drift. It deliberately differs on the UNMEASURED cells: the belief
stores 0 there, not NaN, so drive_sim's inpaint never ran and blind ground read as a plateau at 0.
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
    meas = want[1] > 0.5
    np.testing.assert_array_equal(got[0][meas], want[0][meas], err_msg="height, measured cells")
    for name, g, w in zip(("measured", "sd", "drift"), got[1:], want[1:]):
        np.testing.assert_array_equal(g, w, err_msg=name)
    assert got[1].mean() > 0.3, "the scans must actually cover the window"
    assert got[0].max() > 0.4, "and the box must be in the map"


def test_pooling_takes_measured_ground_over_the_fill_and_the_conservative_sd():
    """Two 2x2 blocks. Left: one cell unmeasured with a HIGHER inpainted height and a huge sd --
    neither may speak for the block. Right: nothing measured -> the inpainted max and the largest
    sd stand in."""
    frame = BeliefFrame((-SPAN / 2, SPAN / 2, -SPAN / 2, SPAN / 2), CELL)
    n = frame.measured.shape[0]
    h = np.zeros((n, n), np.float32)
    m = np.zeros((n, n), np.float32)
    sd = np.full((n, n), 0.01, np.float32)
    dr = np.full((n, n), -1.0, np.float32)
    h[0:2, 0:2] = [[0.10, 0.30], [0.20, 0.90]]  # 0.90 is the fill
    m[0:2, 0:2] = [[1, 1], [1, 0]]
    sd[0:2, 0:2] = [[0.01, 0.03], [0.02, 5.0]]
    dr[0:2, 0:2] = [[1e-4, 3e-4], [2e-4, -1.0]]
    h[0:2, 2:4] = [[0.4, 0.5], [0.6, 0.7]]
    sd[0:2, 2:4] = [[1.0, 2.0], [3.0, 4.0]]
    frame.height = wp.array(h, dtype=wp.float32)
    frame.measured.assign(m)
    frame.sd.assign(sd)
    frame.drift = wp.array(dr, dtype=wp.float32)
    out = [wp.zeros((1, 2), dtype=wp.float32) for _ in range(4)]
    frame.pool(0, 0, 2, *out)
    ph, pm, psd, pdr = (o.numpy()[0] for o in out)
    np.testing.assert_allclose(ph, [0.30, 0.7])
    np.testing.assert_array_equal(pm, [1.0, 0.0])
    np.testing.assert_allclose(psd, [0.03, 4.0])
    np.testing.assert_allclose(pdr, [3e-4, -1.0])


def test_blind_ground_is_filled_from_the_ground_around_it_not_zero():
    """Ground at -0.5 m, measured only near the robot: the rest of the window must read as ground
    (the inpaint's job), not as the 0 the belief stores in cells it never measured."""
    frame = BeliefFrame((-SPAN / 2, SPAN / 2, -SPAN / 2, SPAN / 2), CELL)
    rng = np.random.default_rng(1)
    xy = rng.uniform(-1.0, 1.0, (3000, 2))
    pts = np.column_stack([xy, np.full(len(xy), -0.5)]).astype(np.float32)
    frame.update(wp.array(pts, dtype=wp.vec3f), np.array([0.0, 0.0, 0.4], np.float32), (0, 0), DT)
    height, measured, _, _ = (a.numpy() for a in frame.layers())
    assert measured.mean() < 0.2, "most of the window is blind, or this proves nothing"
    np.testing.assert_allclose(height[measured < 0.5], -0.5, atol=0.02)
