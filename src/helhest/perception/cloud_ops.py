"""Device-resident point-cloud ops (Warp): transform and axis-aligned crop.

Keeps clouds on the GPU between stages — no host round trip. `transform_points`
applies a host 4x4 pose on device; `BoxCrop` crops to an xy box (with optional
recenter + z cutoff) and compacts, mirroring the host `pose_math` helpers but
without ever leaving the device. Only point counts cross back to the host.

`ScanPreprocessor` fuses a raw sensor sweep's whole entry path — sensor->base
transform, z / self-footprint / range rejection, compaction, and the
constant-velocity deskew — into two kernels over the device cloud.
"""

from __future__ import annotations

import numpy as np
import warp as wp

wp.init()

_Z_UNBOUNDED = 1.0e30  # sentinel "no z cutoff" (float32-safe)


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
def _box_crop_kernel(
    points: wp.array(dtype=wp.vec3),
    n: wp.int32,
    cx: wp.float32,
    cy: wp.float32,
    half_x: wp.float32,
    half_y: wp.float32,
    sx: wp.float32,  # recenter shift subtracted from each kept point (0 = no shift)
    sy: wp.float32,
    sz: wp.float32,
    z_max: wp.float32,  # drop points whose (recentered) z exceeds this
    out: wp.array(dtype=wp.vec3),
    counter: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if i >= n:
        return
    p = points[i]
    if wp.abs(p[0] - cx) > half_x or wp.abs(p[1] - cy) > half_y:  # xy box, inclusive
        return
    q = wp.vec3(p[0] - sx, p[1] - sy, p[2] - sz)
    if q[2] > z_max:
        return
    idx = wp.atomic_add(counter, 0, 1)  # append order is nondeterministic; callers don't rely on it
    out[idx] = q


class BoxCrop:
    """Crop a device cloud to an xy box (optional recenter + z cutoff), compacting on device.

    Preallocated output buffer sized for up to `max_points` input points, reused
    every call — the returned array is valid only until the next `crop()`.
    """

    def __init__(self, max_points: int, device: wp.context.Device | None = None) -> None:
        self.device = wp.get_device(device)
        self.max_points = int(max_points)
        with wp.ScopedDevice(self.device):
            self._out = wp.empty(self.max_points, dtype=wp.vec3)
            self._counter = wp.zeros(1, dtype=wp.int32)

    def crop(
        self,
        points: wp.array,
        n: int,
        center: tuple[float, float],
        half: float | tuple[float, float],
        *,
        recenter: np.ndarray | None = None,
        z_max: float | None = None,
    ) -> tuple[wp.array, int]:
        """Keep the first `n` points inside the xy box; return `(out_device, count)`.

        `center` is the box center `(cx, cy)`; `half` a single or `(half_x, half_y)`
        half-extent. `recenter` (a 3-vector, e.g. the robot translation) is
        subtracted from every kept point; `z_max` then drops points whose recentered
        z exceeds it. z is otherwise unbounded.
        """
        if n > self.max_points:
            raise ValueError(f"n={n} exceeds max_points={self.max_points}")
        if n == 0:
            return self._out, 0
        cx, cy = float(center[0]), float(center[1])
        half_x, half_y = (half, half) if np.isscalar(half) else (float(half[0]), float(half[1]))
        sx, sy, sz = (
            (0.0, 0.0, 0.0)
            if recenter is None
            else (float(recenter[0]), float(recenter[1]), float(recenter[2]))
        )
        zm = _Z_UNBOUNDED if z_max is None else float(z_max)
        with wp.ScopedDevice(self.device):
            self._counter.zero_()
            wp.launch(
                _box_crop_kernel,
                dim=n,
                inputs=[points, n, cx, cy, float(half_x), float(half_y), sx, sy, sz, zm],
                outputs=[self._out, self._counter],
            )
            wp.synchronize()
            n_out = int(self._counter.numpy()[0])
        return self._out, n_out


@wp.kernel
def _scan_gate_kernel(
    src: wp.array(dtype=wp.vec3),  # raw points in the SENSOR frame
    times: wp.array(dtype=wp.float32),  # per-point sweep time; ignored when has_times == 0
    has_times: wp.int32,
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
    out_pts: wp.array(dtype=wp.vec3),
    out_times: wp.array(dtype=wp.float32),
    counter: wp.array(dtype=wp.int32),
    t_bounds: wp.array(dtype=wp.float32),  # [min, max] over the SURVIVING points
):
    """Transform one raw point into the base frame and apply every entry gate in a single pass.

    Fusing the three rejections matters more than the transform: done separately on the host each
    one is a full boolean mask plus a fancy-index COPY of the whole cloud. Here a point that fails
    any test simply never gets an output slot. The sweep-time bounds are reduced in the same pass,
    because the deskew's alpha normalisation is defined over the surviving points, not the raw
    sweep.
    """
    i = wp.tid()
    if i >= n:
        return
    p = src[i]
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
    if has_times != 0:
        t = times[i]
        out_times[idx] = t
        wp.atomic_min(t_bounds, 0, t)
        wp.atomic_max(t_bounds, 1, t)


@wp.kernel
def _deskew_kernel(
    pts: wp.array(dtype=wp.vec3),  # compacted base-frame points, rewritten IN PLACE
    times: wp.array(dtype=wp.float32),
    n: wp.int32,
    t_min: wp.float32,
    inv_span: wp.float32,
    axis: wp.vec3,  # so3 log of the sweep rotation, normalised
    theta: wp.float32,  # its magnitude; <= 0 means pure translation
    t_delta: wp.vec3,
    r_delta_t: wp.mat33,  # R_delta TRANSPOSED (the host form right-multiplies by R_delta)
):
    """Motion-compensate one point to the sweep-end pose (constant velocity).

    Mirrors `localization.pose_math.deskew_scan` exactly: p' = R_dᵀ·(R(α)·p + (α−1)·t_d) with
    R(α) = exp(α·log R_d) applied by Rodrigues about the shared screw axis.
    """
    i = wp.tid()
    if i >= n:
        return
    p = pts[i]
    alpha = (times[i] - t_min) * inv_span
    if theta > 1.0e-9:
        angle = alpha * theta
        c1 = wp.cross(axis, p)
        c2 = wp.cross(axis, c1)
        p = p + wp.sin(angle) * c1 + (1.0 - wp.cos(angle)) * c2
    p = p + (alpha - 1.0) * t_delta
    pts[i] = r_delta_t * p


class ScanPreprocessor:
    """Whole raw-sweep entry path on device: transform + gates + compaction + deskew.

    The host only ever sees the surviving point COUNT and the sweep-time bounds (three scalars,
    one readback) — the cloud itself is uploaded once and never comes back. Buffers are sized for
    `max_points` and reused; the returned array is valid only until the next `run()`.
    """

    def __init__(self, max_points: int, device: wp.context.Device | None = None) -> None:
        self.device = wp.get_device(device)
        self.max_points = int(max_points)
        with wp.ScopedDevice(self.device):
            self._src = wp.empty(self.max_points, dtype=wp.vec3)
            self._src_t = wp.empty(self.max_points, dtype=wp.float32)
            self._out = wp.empty(self.max_points, dtype=wp.vec3)
            self._out_t = wp.empty(self.max_points, dtype=wp.float32)
            self._counter = wp.zeros(1, dtype=wp.int32)
            self._t_bounds = wp.zeros(2, dtype=wp.float32)

    def run(
        self,
        points: np.ndarray,  # (N, 3) sensor frame — the one unavoidable host->device upload
        point_times: np.ndarray | None,  # (N,) sweep times, or None when the cloud has no t field
        base_T_sensor: np.ndarray,
        *,
        z_range: tuple[float, float] | None,  # None disables the z crop
        self_box: tuple[float, float, float, float] | None,  # (x_min, x_max, y_min, y_max)
        max_range: float,  # <= 0 disables the range crop
    ) -> tuple[wp.array, int, wp.array, float, float]:
        """Return `(points_device, count, times_device, t_min, t_span)`.

        `t_min`/`t_span` are 0.0 when the cloud carries no per-point times or all survivors share
        one stamp; the caller skips the deskew in that case.
        """
        n = int(points.shape[0])
        if n > self.max_points:
            raise ValueError(f"n={n} exceeds max_points={self.max_points}")
        if n == 0:
            return self._out, 0, self._out_t, 0.0, 0.0
        has_times = point_times is not None
        m = wp.mat44(*[float(v) for v in np.asarray(base_T_sensor, dtype=np.float32).reshape(-1)])
        zr = z_range if z_range is not None else (0.0, 0.0)
        sb = self_box if self_box is not None else (0.0, 0.0, 0.0, 0.0)
        with wp.ScopedDevice(self.device):
            wp.copy(self._src, wp.array(np.ascontiguousarray(points, np.float32), dtype=wp.vec3),
                    count=n)
            if has_times:
                wp.copy(
                    self._src_t,
                    wp.array(np.ascontiguousarray(point_times, np.float32), dtype=wp.float32),
                    count=n,
                )
            self._counter.zero_()
            # seed the reduction so the first atomic_min/max wins
            self._t_bounds.assign(np.array([np.inf, -np.inf], np.float32))
            wp.launch(
                _scan_gate_kernel,
                dim=n,
                inputs=[
                    self._src, self._src_t, int(has_times), n, m,
                    float(zr[0]), float(zr[1]), int(z_range is not None),
                    float(sb[0]), float(sb[1]), float(sb[2]), float(sb[3]),
                    int(self_box is not None),
                    float(max_range * max_range) if max_range > 0.0 else 0.0,
                ],
                outputs=[self._out, self._out_t, self._counter, self._t_bounds],
            )
            wp.synchronize()  # the single readback: count + sweep bounds (3 scalars)
            count = int(self._counter.numpy()[0])
            lo, hi = (float(v) for v in self._t_bounds.numpy())
        if not has_times or count == 0 or not np.isfinite(lo) or not np.isfinite(hi):
            return self._out, count, self._out_t, 0.0, 0.0
        return self._out, count, self._out_t, lo, max(hi - lo, 0.0)

    def deskew(self, count: int, t_min: float, t_span: float, sweep_delta: np.ndarray) -> None:
        """Deskew the compacted buffer IN PLACE. No-op for an empty or zero-length sweep."""
        if count == 0 or t_span <= 0.0:
            return
        # Deferred import: `localization.localizer` imports THIS module, so pulling pose_math in at
        # module scope would close the cycle. Reused rather than reimplemented so the screw-axis
        # decomposition can't drift away from the host `deskew_scan` this kernel mirrors.
        from ..localization.pose_math import _so3_log

        d = np.asarray(sweep_delta, dtype=np.float64)
        r_delta = d[:3, :3]
        omega = _so3_log(r_delta)
        theta = float(np.linalg.norm(omega))
        axis = omega / theta if theta > 1.0e-9 else np.zeros(3)
        with wp.ScopedDevice(self.device):
            wp.launch(
                _deskew_kernel,
                dim=count,
                inputs=[
                    self._out, self._out_t, int(count), float(t_min), float(1.0 / t_span),
                    wp.vec3(*[float(v) for v in axis]), float(theta),
                    wp.vec3(*[float(v) for v in d[:3, 3]]),
                    wp.mat33(*[float(v) for v in r_delta.T.reshape(-1)]),
                ],
            )
