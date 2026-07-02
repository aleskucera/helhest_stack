"""On-device rolling point-cloud accumulator (Warp).

The temporal-fusion stage of the pipeline (scans → **accumulate + carve** →
heightmap → traversability). Robot-agnostic: it takes point clouds plus an
optional carve mask (e.g. from `DynamicPointFilter.carve`) — the same class
backs both the offline sim demo and on-robot accumulation.

Keeps the accumulated map resident on the GPU across frames so the per-frame
pipeline (carve → add new returns → crop to a radius → voxel-thin) never rounds
through host memory. One `step()` fuses all four into a single masked voxel pass:
carved map points and out-of-radius / invalid points are simply skipped, and the
survivors are re-binned to one centroid per occupied voxel.
"""

from __future__ import annotations

import warp as wp

wp.init()

_MAX_CELLS = 20_000_000


@wp.kernel
def _voxel_accumulate_masked_kernel(
    points: wp.array(dtype=wp.vec3),
    valid: wp.array(dtype=wp.int32),
    cx: wp.float32,
    cy: wp.float32,
    r2: wp.float32,
    min_corner: wp.vec3,
    inv_voxel: wp.float32,
    dx: wp.int32,
    dy: wp.int32,
    dz: wp.int32,
    sums: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=wp.int32),
):
    """Sum each kept point into its voxel cell.

    Skips invalid points and points outside the xy crop radius, so the carve /
    crop / bin steps fuse into this one launch (no separate mask pass).
    """
    i = wp.tid()
    if valid[i] == 0:
        return
    p = points[i]
    if (p[0] - cx) * (p[0] - cx) + (p[1] - cy) * (p[1] - cy) > r2:
        return
    ix = int((p[0] - min_corner[0]) * inv_voxel)
    iy = int((p[1] - min_corner[1]) * inv_voxel)
    iz = int((p[2] - min_corner[2]) * inv_voxel)
    if ix < 0 or ix >= dx or iy < 0 or iy >= dy or iz < 0 or iz >= dz:
        return
    idx = (ix * dy + iy) * dz + iz
    wp.atomic_add(sums, idx, p)
    wp.atomic_add(counts, idx, 1)


@wp.kernel
def _voxel_compact_kernel(
    sums: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=wp.int32),
    out_counter: wp.array(dtype=wp.int32),
    out_points: wp.array(dtype=wp.vec3),
):
    """Write one centroid per occupied voxel into a compact output."""
    v = wp.tid()
    c = counts[v]
    if c > 0:
        slot = wp.atomic_add(out_counter, 0, 1)
        out_points[slot] = sums[v] / float(c)


class DeviceMapAccumulator:
    """Rolling map kept on-device: carve + add + crop + voxel-thin per frame.

    Fixed robot-centric voxel grid (`radius` half-extent in xy, `z_bounds` in z),
    so bounds come from the robot pose with no host readback. `step()` returns the
    new map as a `wp.array(vec3)` living on `device`.
    """

    def __init__(
        self,
        voxel_size: float,
        radius: float,
        *,
        z_bounds: tuple[float, float] = (-2.0, 6.0),
        device: wp.context.Device | None = None,
    ):
        self.device = wp.get_device(device)
        self.voxel = float(voxel_size)
        self.radius = float(radius)
        self.z0, self.z1 = float(z_bounds[0]), float(z_bounds[1])
        self.dx = int(2.0 * self.radius / self.voxel) + 1
        self.dy = self.dx
        self.dz = int((self.z1 - self.z0) / self.voxel) + 1
        n_vx = self.dx * self.dy * self.dz
        if n_vx > _MAX_CELLS:
            raise ValueError(
                f"voxel grid has {n_vx} cells (>{_MAX_CELLS}); coarsen voxel_size or shrink radius"
            )
        with wp.ScopedDevice(self.device):
            self._sums = wp.zeros(n_vx, dtype=wp.vec3)
            self._counts = wp.zeros(n_vx, dtype=wp.int32)
            self._counter = wp.zeros(1, dtype=wp.int32)

    def step(
        self,
        map_wp: wp.array | None,
        carve_mask: wp.array | None,
        points_wp: wp.array,
        valid_wp: wp.array,
        center: tuple[float, float],
    ) -> wp.array:
        """Return the new map: (carved map ∪ valid new points), cropped + voxel-thinned.

        `map_wp` is the previous map (None on the first frame). `carve_mask` (int32,
        len == map) marks map points to keep (None → keep all). `points_wp` are the
        new scan's per-beam points with `valid_wp` (int32) selecting real returns.
        """
        cx, cy = float(center[0]), float(center[1])
        n_map = 0 if map_wp is None else len(map_wp)
        n_pts = len(points_wp)
        n = n_map + n_pts
        with wp.ScopedDevice(self.device):
            if n == 0:
                return wp.zeros(0, dtype=wp.vec3)

            combined = wp.empty(n, dtype=wp.vec3)
            mask_in = wp.empty(n, dtype=wp.int32)
            if n_map:
                wp.copy(combined, map_wp, 0, 0, n_map)
                keep = carve_mask if carve_mask is not None else wp.full(n_map, 1, dtype=wp.int32)
                wp.copy(mask_in, keep, 0, 0, n_map)
            wp.copy(combined, points_wp, n_map, 0, n_pts)
            wp.copy(mask_in, valid_wp, n_map, 0, n_pts)

            self._sums.zero_()
            self._counts.zero_()
            self._counter.zero_()
            min_corner = wp.vec3(cx - self.radius, cy - self.radius, self.z0)
            # Carve + xy-radius crop + voxel-bin fused into one masked pass.
            wp.launch(
                _voxel_accumulate_masked_kernel,
                dim=n,
                inputs=[
                    combined,
                    mask_in,
                    cx,
                    cy,
                    self.radius * self.radius,
                    min_corner,
                    1.0 / self.voxel,
                    self.dx,
                    self.dy,
                    self.dz,
                ],
                outputs=[self._sums, self._counts],
            )
            out = wp.empty(n, dtype=wp.vec3)
            wp.launch(
                _voxel_compact_kernel,
                dim=len(self._counts),
                inputs=[self._sums, self._counts],
                outputs=[self._counter, out],
            )
            wp.synchronize()
            n_out = int(self._counter.numpy()[0])
            new_map = wp.empty(n_out, dtype=wp.vec3)
            if n_out:
                wp.copy(new_map, out, 0, 0, n_out)
            return new_map
