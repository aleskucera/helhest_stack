"""GPU ray-cast LiDAR against analytic primitives (Warp).

A simulation utility (not part of the perception pipeline): one thread per beam
casts against a bounded ground plane plus a set of axis-aligned box obstacles,
keeping the nearest hit. The sensor has a movable pose (position + yaw), so it
can drive around; range noise and beam dropout are applied in-kernel. Used by
the demos to feed realistic scans into the terrain pipeline / dynamic filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

wp.init()

_MISS = wp.constant(1.0e30)


@wp.func
def _hit_bounded_plane(
    o: wp.vec3,
    d: wp.vec3,
    axis: int,
    value: float,
    a_lo: float,
    a_hi: float,  # bounds on the first other axis
    b_lo: float,
    b_hi: float,  # bounds on the second other axis
) -> float:
    """Distance to an axis-aligned plane, valid only inside a 2D window (else miss)."""
    denom = d[axis]
    if wp.abs(denom) < 1.0e-9:
        return _MISS
    t = (value - o[axis]) / denom
    if t <= 1.0e-3:
        return _MISS
    p = o + t * d
    # The two axes that are NOT `axis`.
    a = (axis + 1) % 3
    b = (axis + 2) % 3
    if p[a] < a_lo or p[a] > a_hi or p[b] < b_lo or p[b] > b_hi:
        return _MISS
    return t


@wp.func
def _hit_aabb(o: wp.vec3, d: wp.vec3, lo: wp.vec3, hi: wp.vec3) -> float:
    """Slab-method nearest entry distance for a ray/AABB (miss → _MISS)."""
    tmin = float(-1.0e30)
    tmax = float(1.0e30)
    for a in range(3):
        if wp.abs(d[a]) < 1.0e-12:
            if o[a] < lo[a] or o[a] > hi[a]:
                return _MISS
        else:
            inv = 1.0 / d[a]
            t1 = (lo[a] - o[a]) * inv
            t2 = (hi[a] - o[a]) * inv
            tmin = wp.max(tmin, wp.min(t1, t2))
            tmax = wp.min(tmax, wp.max(t1, t2))
    if tmax < wp.max(tmin, 0.0) or tmax <= 1.0e-3:
        return _MISS
    if tmin > 1.0e-3:
        return tmin
    return tmax


@wp.kernel
def raycast_kernel(
    origin: wp.vec3,
    yaw: wp.float32,
    dirs: wp.array(dtype=wp.vec3),  # beam directions in the sensor's local frame
    ground_z: wp.float32,
    gx_lo: wp.float32,
    gx_hi: wp.float32,
    gy_lo: wp.float32,
    gy_hi: wp.float32,
    boxes_lo: wp.array(dtype=wp.vec3),
    boxes_hi: wp.array(dtype=wp.vec3),
    n_boxes: wp.int32,
    noise_base: wp.float32,
    noise_quad: wp.float32,
    noise_max: wp.float32,
    dropout: wp.float32,
    min_range: wp.float32,
    max_range: wp.float32,
    seed: wp.int32,
    out_points: wp.array(dtype=wp.vec3),
    out_valid: wp.array(dtype=wp.int32),
):
    """Nearest primitive hit per beam, with dropout + range-dependent noise."""
    i = wp.tid()
    # Rotate the local beam into the world by the sensor yaw (about +z).
    dl = dirs[i]
    c = wp.cos(yaw)
    s = wp.sin(yaw)
    d = wp.vec3(c * dl[0] - s * dl[1], s * dl[0] + c * dl[1], dl[2])

    t = _MISS
    tg = _hit_bounded_plane(origin, d, 2, ground_z, gx_lo, gx_hi, gy_lo, gy_hi)
    if tg < t:
        t = tg
    for j in range(n_boxes):
        tb = _hit_aabb(origin, d, boxes_lo[j], boxes_hi[j])
        if tb < t:
            t = tb

    # No hit, or the surface is outside the sensor's [min, max] range window.
    if t >= _MISS or t < min_range or t > max_range:
        out_valid[i] = 0
        return

    # Draw dropout first, then noise, so the stream is deterministic per beam.
    state = wp.rand_init(seed, i)
    if wp.randf(state) < dropout:
        out_valid[i] = 0
        return
    # Range precision degrades with distance: sigma(r) = base + quad*r², capped.
    sigma = noise_base + noise_quad * t * t
    if sigma > noise_max:
        sigma = noise_max
    t = t + sigma * wp.randn(state)  # noise is along the beam
    out_points[i] = origin + t * d
    out_valid[i] = 1


@dataclass
class GroundSpec:
    z: float
    x_range: tuple[float, float]
    y_range: tuple[float, float]


class PrimitiveLidar:
    """Ray-cast LiDAR: a movable sensor over a ground plane + box obstacles.

    Beam directions (in the sensor's local frame, looking down +x) and the
    ground are fixed at construction. Each `scan()` places the sensor at a pose
    (position + yaw), casts against the current set of box obstacles, and returns
    the surviving hit points as an (N, 3) numpy array. Range noise and dropout
    are applied on-device.
    """

    def __init__(
        self,
        directions: np.ndarray,
        *,
        ground: GroundSpec,
        noise_std: float = 0.0,
        range_noise_quad: float = 0.0,
        range_noise_max: float | None = None,
        dropout: float = 0.0,
        min_range: float = 0.0,
        max_range: float | None = None,
        device: wp.context.Device | None = None,
    ):
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError(f"directions must be (B, 3); got {directions.shape}")
        self.device = device if device is not None else wp.get_device()
        self.ground = ground
        # Range noise 1σ(r) = noise_std + range_noise_quad·r², capped at range_noise_max.
        self.noise_std = float(noise_std)
        self.range_noise_quad = float(range_noise_quad)
        self.range_noise_max = 1.0e30 if range_noise_max is None else float(range_noise_max)
        self.dropout = float(dropout)
        self.min_range = float(min_range)
        # None → effectively unlimited (the miss sentinel already caps real hits).
        self.max_range = 1.0e30 if max_range is None else float(max_range)

        n = len(directions)
        with wp.ScopedDevice(self.device):
            self._dirs = wp.array(np.ascontiguousarray(directions, dtype=np.float32), dtype=wp.vec3)
            self._out_pts = wp.empty(n, dtype=wp.vec3)
            self._out_valid = wp.empty(n, dtype=wp.int32)
        self._n = n

    def scan(
        self,
        origin: np.ndarray,
        yaw: float,
        boxes_lo: np.ndarray,
        boxes_hi: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        """Cast from `origin` with heading `yaw` against boxes `[boxes_lo, boxes_hi]`.

        `boxes_lo`/`boxes_hi` are (M, 3) AABB corners (M may be 0). Returns kept
        hit points (N, 3).
        """
        boxes_lo = np.ascontiguousarray(boxes_lo, dtype=np.float32).reshape(-1, 3)
        boxes_hi = np.ascontiguousarray(boxes_hi, dtype=np.float32).reshape(-1, 3)
        n_boxes = len(boxes_lo)
        g = self.ground
        with wp.ScopedDevice(self.device):
            # Warp arrays must be non-empty; pad to length 1 when there are no boxes.
            lo_wp = wp.array(boxes_lo if n_boxes else np.zeros((1, 3), np.float32), dtype=wp.vec3)
            hi_wp = wp.array(boxes_hi if n_boxes else np.zeros((1, 3), np.float32), dtype=wp.vec3)
            wp.launch(
                raycast_kernel,
                dim=self._n,
                inputs=[
                    wp.vec3(*np.asarray(origin, dtype=np.float64).tolist()),
                    float(yaw),
                    self._dirs,
                    float(g.z),
                    float(g.x_range[0]),
                    float(g.x_range[1]),
                    float(g.y_range[0]),
                    float(g.y_range[1]),
                    lo_wp,
                    hi_wp,
                    int(n_boxes),
                    self.noise_std,
                    self.range_noise_quad,
                    self.range_noise_max,
                    self.dropout,
                    self.min_range,
                    self.max_range,
                    int(seed),
                ],
                outputs=[self._out_pts, self._out_valid],
            )
            wp.synchronize()
            valid = self._out_valid.numpy().astype(bool)
            return self._out_pts.numpy()[valid]
