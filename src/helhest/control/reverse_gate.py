"""The map-knowledge gate for reverse driving: no rear sensor, so the robot may only back over
ground it has already measured.

A robot-width strip straight behind the base, from `near_m` to `far_m`, is sampled on the
measured mask; reverse is open while at least `min_fraction` of it is measured. Out-of-window
samples count as blind. One thread, the strip is a few hundred cells; the result is one float
read back, not the mask.
"""

from __future__ import annotations

import numpy as np
import warp as wp


@wp.kernel
def _strip_kernel(
    measured: wp.array2d(dtype=wp.float32),  # [ny, nx], 1 = observed
    origin_x: wp.float32,  # the mask's origin, same frame as the pose
    origin_y: wp.float32,
    cell: wp.float32,
    x: wp.float32,  # base pose
    y: wp.float32,
    yaw: wp.float32,
    near: wp.float32,  # [m] the strip runs from `near` to `far` behind the base ...
    far: wp.float32,
    half_width: wp.float32,  # ... and this far to each side
    out: wp.array(dtype=wp.float32),  # [1] measured fraction of the strip
):
    tid = wp.tid()
    if tid != 0:
        return
    c = wp.cos(yaw)
    s = wp.sin(yaw)
    # rounded, not truncated: 1.2 / 0.2 is 5.999 in float32
    n_d = int((far - near) / cell + 0.5) + 1
    n_l = int(2.0 * half_width / cell + 0.5) + 1
    hit = float(0.0)
    for i in range(n_d):
        d = near + float(i) * cell
        for j in range(n_l):
            lat = -half_width + float(j) * cell
            px = x - c * d - s * lat
            py = y - s * d + c * lat
            col = int(wp.floor((px - origin_x) / cell))
            row = int(wp.floor((py - origin_y) / cell))
            if row >= 0 and row < measured.shape[0] and col >= 0 and col < measured.shape[1]:
                hit += measured[row, col]
    out[0] = hit / float(n_d * n_l)


class ReverseGate:
    """`clear(...)` -> True while the strip behind the robot is measured enough to back over."""

    def __init__(
        self,
        far_m: float = 1.5,
        near_m: float = 0.3,
        half_width_m: float = 0.6,
        min_fraction: float = 0.95,
        device: wp.Device | str | None = None,
    ) -> None:
        self.far_m, self.near_m = float(far_m), float(near_m)
        self.half_width_m, self.min_fraction = float(half_width_m), float(min_fraction)
        self.device = wp.get_device(device)
        self._out = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.fraction = 0.0  # the last strip's measured fraction, for logging

    def clear(
        self,
        measured: wp.array,
        origin_xy: tuple[float, float],
        cell: float,
        pose: tuple[float, float, float],
    ) -> bool:
        wp.launch(
            _strip_kernel,
            dim=1,
            inputs=[
                measured,
                float(origin_xy[0]),
                float(origin_xy[1]),
                float(cell),
                float(pose[0]),
                float(pose[1]),
                float(pose[2]),
                self.near_m,
                self.far_m,
                self.half_width_m,
            ],
            outputs=[self._out],
            device=self.device,
        )
        self.fraction = float(self._out.numpy()[0])
        return self.fraction >= self.min_fraction


def strip_measured_fraction(measured: np.ndarray, origin_xy, cell, pose, **kw) -> float:
    """Host reference of the kernel, for tests."""
    near, far = kw.get("near_m", 0.3), kw.get("far_m", 1.5)
    hw = kw.get("half_width_m", 0.6)
    x, y, yaw = pose
    ds = np.arange(near, far + 1e-6, cell)
    lats = np.arange(-hw, hw + 1e-6, cell)
    px = x - np.cos(yaw) * ds[:, None] - np.sin(yaw) * lats[None, :]
    py = y - np.sin(yaw) * ds[:, None] + np.cos(yaw) * lats[None, :]
    cols = np.floor((px - origin_xy[0]) / cell).astype(int)
    rows = np.floor((py - origin_xy[1]) / cell).astype(int)
    inb = (rows >= 0) & (rows < measured.shape[0]) & (cols >= 0) & (cols < measured.shape[1])
    hit = np.zeros(px.shape, np.float32)
    hit[inb] = measured[rows[inb], cols[inb]]
    return float(hit.mean())
