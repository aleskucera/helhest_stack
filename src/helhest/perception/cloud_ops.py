"""Device-resident point-cloud ops (Warp).

Keeps clouds on the GPU between stages — no host round trip. `transform_points`
applies a host 4x4 pose on device. `ScanPreprocessor` fuses a raw sensor sweep's
whole entry path — sensor->base transform, z / self-footprint / range rejection
and compaction — into one kernel over the device cloud. Only the point count
crosses back to the host.
"""

from __future__ import annotations

import numpy as np
import warp as wp

wp.init()


@wp.kernel
def _transform_kernel(
    src: wp.array(dtype=wp.vec3),
    n: wp.int32,
    m: wp.mat44,
    out: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    if i >= n:
        return
    p = src[i]
    out[i] = wp.vec3(
        m[0, 0] * p[0] + m[0, 1] * p[1] + m[0, 2] * p[2] + m[0, 3],
        m[1, 0] * p[0] + m[1, 1] * p[1] + m[1, 2] * p[2] + m[1, 3],
        m[2, 0] * p[0] + m[2, 1] * p[1] + m[2, 2] * p[2] + m[2, 3],
    )


def transform_points(points: wp.array, n: int, pose: np.ndarray) -> wp.array:
    """Apply the host 4x4 `pose` to the first `n` device points; return a new device array."""
    m = wp.mat44(*[float(v) for v in np.asarray(pose, dtype=np.float32).reshape(-1)])
    out = wp.empty(n, dtype=wp.vec3, device=points.device)
    wp.launch(_transform_kernel, dim=n, inputs=[points, n, m], outputs=[out], device=points.device)
    return out


@wp.kernel
def _scan_gate_kernel(
    src: wp.array(dtype=wp.vec3),  # raw points in the SENSOR frame
    n: wp.int32,
    m: wp.mat44,  # base_T_sensor
    z_min: wp.float32,
    z_max: wp.float32,
    z_enable: wp.int32,
    self_x_min: wp.float32,
    self_x_max: wp.float32,
    self_y_min: wp.float32,
    self_y_max: wp.float32,
    self_enable: wp.int32,
    range_max_sq: wp.float32,  # <= 0 disables the range crop
    range_min_sq: wp.float32,  # <= 0 disables the near cut
    out_pts: wp.array(dtype=wp.vec3),
    counter: wp.array(dtype=wp.int32),
):
    """Transform one raw point into the base frame and apply every entry gate in a single pass.

    Fusing the three rejections matters more than the transform: done separately on the host each
    one is a full boolean mask plus a fancy-index COPY of the whole cloud. Here a point that fails
    any test simply never gets an output slot.
    """
    i = wp.tid()
    if i >= n:
        return
    p = src[i]
    if range_min_sq > 0.0 and wp.dot(p, p) < range_min_sq:
        return  # too close to the sensor, measured in the sensor frame
    x = m[0, 0] * p[0] + m[0, 1] * p[1] + m[0, 2] * p[2] + m[0, 3]
    y = m[1, 0] * p[0] + m[1, 1] * p[1] + m[1, 2] * p[2] + m[1, 3]
    z = m[2, 0] * p[0] + m[2, 1] * p[1] + m[2, 2] * p[2] + m[2, 3]
    if z_enable != 0 and (z < z_min or z > z_max):
        return
    if self_enable != 0:
        if x >= self_x_min and x <= self_x_max and y >= self_y_min and y <= self_y_max:
            return  # the robot's own wheels/body
    if range_max_sq > 0.0 and x * x + y * y > range_max_sq:
        return
    idx = wp.atomic_add(counter, 0, 1)  # append order is nondeterministic; callers don't rely on it
    out_pts[idx] = wp.vec3(x, y, z)


class ScanPreprocessor:
    """Whole raw-sweep entry path on device: transform + gates + compaction.

    The host only ever sees the surviving point COUNT (one scalar readback) — the cloud itself is
    uploaded once and never comes back. Buffers are sized for `max_points` and reused; the
    returned array is valid only until the next `run()`.
    """

    def __init__(self, max_points: int, device: wp.context.Device | None = None) -> None:
        self.device = wp.get_device(device)
        self.max_points = int(max_points)
        with wp.ScopedDevice(self.device):
            self._src = wp.empty(self.max_points, dtype=wp.vec3)
            self._out = wp.empty(self.max_points, dtype=wp.vec3)
            self._counter = wp.zeros(1, dtype=wp.int32)

    def run(
        self,
        points: np.ndarray,  # (N, 3) sensor frame — the one unavoidable host->device upload
        base_T_sensor: np.ndarray,
        *,
        z_range: tuple[float, float] | None,  # None disables the z crop
        self_box: tuple[float, float, float, float] | None,  # (x_min, x_max, y_min, y_max)
        max_range: float,  # <= 0 disables the range crop
        # [m] <= 0 disables. 3-D distance from the sensor. Close returns are the densest and
        # the most precise, so the belief's "adopt a higher reading at once" rule ratchets the
        # ground under the robot upward on them: +5-7 cm within 1 m on the Robotour drive.
        min_range: float,
    ) -> tuple[wp.array, int]:
        """Return `(points_device, count)`."""
        n = int(points.shape[0])
        if n > self.max_points:
            raise ValueError(f"n={n} exceeds max_points={self.max_points}")
        if n == 0:
            return self._out, 0
        m = wp.mat44(*[float(v) for v in np.asarray(base_T_sensor, dtype=np.float32).reshape(-1)])
        zr = z_range if z_range is not None else (0.0, 0.0)
        sb = self_box if self_box is not None else (0.0, 0.0, 0.0, 0.0)
        with wp.ScopedDevice(self.device):
            wp.copy(
                self._src,
                wp.array(np.ascontiguousarray(points, np.float32), dtype=wp.vec3),
                count=n,
            )
            self._counter.zero_()
            wp.launch(
                _scan_gate_kernel,
                dim=n,
                inputs=[
                    self._src,
                    n,
                    m,
                    float(zr[0]),
                    float(zr[1]),
                    int(z_range is not None),
                    float(sb[0]),
                    float(sb[1]),
                    float(sb[2]),
                    float(sb[3]),
                    int(self_box is not None),
                    float(max_range * max_range) if max_range > 0.0 else 0.0,
                    float(min_range * min_range) if min_range > 0.0 else 0.0,
                ],
                outputs=[self._out, self._counter],
            )
            wp.synchronize()  # the single readback: the count
            count = int(self._counter.numpy()[0])
        return self._out, count
