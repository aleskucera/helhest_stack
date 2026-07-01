from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from ..sensor import LidarSensorConfig
from .kernels import classify_kernel
from .kernels import render_depth_kernel

_DEPTH_SENTINEL = 1.0e18


@dataclass
class DynamicFilterConfig:
    """Tuning for `DynamicPointFilter` (see that class for how the filter works).

    Most fields describe the range image and should match the sensor — prefer
    `from_sensor()`, which pulls the FOV, resolution, and min-range from a
    `LidarSensorConfig` so they can't silently disagree with it. Only `margin_m`
    / `margin_rel` are genuinely filter-specific.
    """

    # Range-image angular resolution. Azimuth spans the full 360°; elevation spans
    # [el_min_deg, el_max_deg]. Default roughly a 128-beam sensor.
    az_bins: int = 1024
    el_bins: int = 128
    # Elevation FOV (deg) the bins span. A point outside is ignored (kept) — too
    # narrow a band silently under-carves, so the default is the full hemisphere.
    el_min_deg: float = -90.0
    el_max_deg: float = 90.0
    # A point is carved if the other cloud's surface along its bearing is farther
    # than the point by more than `margin_m + range * margin_rel`. The relative
    # term absorbs angular quantization on slanted surfaces + registration error.
    margin_m: float = 0.3
    margin_rel: float = 0.02
    # Ignore returns closer than this (sensor self-hits / degenerate directions).
    min_range_m: float = 0.5

    @classmethod
    def from_sensor(
        cls,
        sensor: LidarSensorConfig,
        *,
        margin_m: float = 0.3,
        margin_rel: float = 0.02,
        az_bins: int | None = None,
        el_bins: int | None = None,
    ) -> DynamicFilterConfig:
        """Build from a `LidarSensorConfig`: FOV, min-range, and default range-image
        resolution come from the sensor; only the margins are supplied here.

        `az_bins`/`el_bins` default to the sensor's `columns`/`channels`; override
        for a coarser or finer range image than the sensor's native resolution.
        """
        return cls(
            az_bins=az_bins if az_bins is not None else sensor.columns,
            el_bins=el_bins if el_bins is not None else sensor.channels,
            el_min_deg=sensor.el_fov_deg[0],
            el_max_deg=sensor.el_fov_deg[1],
            margin_m=margin_m,
            margin_rel=margin_rel,
            min_range_m=sensor.min_range_m,
        )


class DynamicPointFilter:
    """Map-frame visibility / ray-carving filter for removing moving objects.

    Compares the accumulated map against a new scan through spherical range images
    rendered from the sensor origin: a map point is *carved* when the scan reached
    farther along its bearing (the beam passed through it → free space). Feed the
    scan's per-beam free-space frontier (surface hit, or the max-range point on a
    miss) and this works even with no background behind a point — e.g. the top of
    a person against open sky. It removes things that MOVE (a beam has to see
    through where something was); a motionless object is a static pillar and kept.

    Two entry points:
      * `carve(map, scan, origin)` → the `map_keep` mask (device- or numpy-native,
        matching the input type). The per-frame mapping call.
      * `filter(map, scan, origin)` → `(scan_keep, map_keep)` numpy masks — also
        drops incoming scan points in front of known geometry.
    """

    def __init__(
        self,
        config: DynamicFilterConfig | None = None,
        *,
        device: wp.context.Device | None = None,
    ):
        self.config = config or DynamicFilterConfig()
        self.device = device if device is not None else wp.get_device()
        self._n_bins = self.config.az_bins * self.config.el_bins
        # Reusable range-image buffers, reset per render (avoids a per-call alloc).
        # Two, because filter() keeps the map and scan depth images live at once.
        with wp.ScopedDevice(self.device):
            self._depth = [wp.empty(self._n_bins, dtype=wp.float32) for _ in range(2)]

    @classmethod
    def from_sensor(
        cls,
        sensor: LidarSensorConfig,
        *,
        margin_m: float = 0.3,
        margin_rel: float = 0.02,
        az_bins: int | None = None,
        el_bins: int | None = None,
        device: wp.context.Device | None = None,
    ) -> DynamicPointFilter:
        """Build a filter whose range image matches `sensor` (FOV, resolution,
        min-range); only the carve margins are supplied here."""
        config = DynamicFilterConfig.from_sensor(
            sensor, margin_m=margin_m, margin_rel=margin_rel, az_bins=az_bins, el_bins=el_bins
        )
        return cls(config, device=device)

    # ------------------------------------------------------------------
    # Internal helpers (all operate on device wp.arrays)
    # ------------------------------------------------------------------

    def _grid_args(self) -> list[int | float]:
        """Shared angular-grid args for the render/classify kernels:
        (az_bins, el_bins, az_min, az_span, el_min, el_max, min_range)."""
        cfg = self.config
        return [
            int(cfg.az_bins),
            int(cfg.el_bins),
            float(-math.pi),
            float(2.0 * math.pi),
            float(math.radians(cfg.el_min_deg)),
            float(math.radians(cfg.el_max_deg)),
            float(cfg.min_range_m),
        ]

    def _to_wp(self, points: np.ndarray | wp.array) -> tuple[wp.array, int]:
        """Upload if numpy; pass through if already a device array."""
        if isinstance(points, wp.array):
            return points, len(points)
        arr = wp.array(np.ascontiguousarray(points, dtype=np.float32), dtype=wp.vec3)
        return arr, len(points)

    def _render(
        self,
        points_wp: wp.array,
        n: int,
        origin: wp.vec3,
        rot: wp.mat33,
        grid: list[int | float],
        depth: wp.array,
    ) -> wp.array:
        """Render points into `depth` as a nearest-range image (sentinel-reset first)."""
        depth.fill_(_DEPTH_SENTINEL)
        if n > 0:
            wp.launch(
                render_depth_kernel,
                dim=n,
                inputs=[points_wp, origin, rot, *grid],
                outputs=[depth],
            )
        return depth

    def _classify(
        self,
        points_wp: wp.array,
        n: int,
        origin: wp.vec3,
        rot: wp.mat33,
        grid: list[int | float],
        other_depth: wp.array,
    ) -> wp.array:
        """Per-point keep mask vs `other_depth` (0 = in front of it → remove)."""
        keep = wp.empty(n, dtype=wp.int32)
        wp.launch(
            classify_kernel,
            dim=n,
            inputs=[
                points_wp,
                origin,
                rot,
                *grid,
                float(self.config.margin_m),
                float(self.config.margin_rel),
                other_depth,
            ],
            outputs=[keep],
        )
        return keep

    @staticmethod
    def _pose(
        sensor_origin: np.ndarray, sensor_rotation: np.ndarray | None
    ) -> tuple[wp.vec3, wp.mat33]:
        R = np.eye(3) if sensor_rotation is None else np.asarray(sensor_rotation, dtype=np.float64)
        rot = wp.mat33(R.flatten().tolist())
        origin = wp.vec3(*np.asarray(sensor_origin, dtype=np.float64).tolist())
        return origin, rot

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def carve(
        self,
        map_points: np.ndarray | wp.array,
        scan_points: np.ndarray | wp.array,
        sensor_origin: np.ndarray,
        *,
        sensor_rotation: np.ndarray | None = None,
    ) -> np.ndarray | wp.array:
        """Carve-only: return the `map_keep` mask (points the scan sees through → 0).

        Half the work of `filter()` — one range image + one classify — and
        device-native: accepts numpy OR `wp.array` inputs and returns a matching
        type (a numpy bool mask, or a device `wp.array(int32)` when given device
        arrays, so the whole loop can stay on the GPU).
        """
        was_np = isinstance(map_points, np.ndarray)
        n_map = len(map_points)
        if n_map == 0:
            return np.ones(0, dtype=bool) if was_np else wp.zeros(0, dtype=wp.int32)

        origin, rot = self._pose(sensor_origin, sensor_rotation)
        grid = self._grid_args()
        with wp.ScopedDevice(self.device):
            map_wp, _ = self._to_wp(map_points)
            scan_wp, n_scan = self._to_wp(scan_points)
            scan_depth = self._render(scan_wp, n_scan, origin, rot, grid, self._depth[0])
            map_keep = self._classify(map_wp, n_map, origin, rot, grid, scan_depth)
            if was_np:
                wp.synchronize()
                return map_keep.numpy().astype(bool)
            return map_keep  # device mask, stream-ordered with downstream kernels

    def filter(
        self,
        map_points: np.ndarray,
        scan_points: np.ndarray,
        sensor_origin: np.ndarray,
        *,
        sensor_rotation: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return `(scan_keep, map_keep)` boolean masks (numpy).

        `sensor_rotation` is the world→sensor rotation (3x3); it aligns the range
        image with the sensor's beams so the elevation band matches its FOV
        regardless of robot tilt. Identity if omitted. For carve-only device use,
        see `carve()`.
        """
        n_scan = len(scan_points)
        n_map = len(map_points)
        # Nothing to compare against → keep everything.
        if n_map == 0 or n_scan == 0:
            return np.ones(n_scan, dtype=bool), np.ones(n_map, dtype=bool)

        origin, rot = self._pose(sensor_origin, sensor_rotation)
        grid = self._grid_args()
        with wp.ScopedDevice(self.device):
            map_wp, _ = self._to_wp(map_points)
            scan_wp, _ = self._to_wp(scan_points)
            map_depth = self._render(map_wp, n_map, origin, rot, grid, self._depth[0])
            scan_depth = self._render(scan_wp, n_scan, origin, rot, grid, self._depth[1])
            scan_keep = self._classify(scan_wp, n_scan, origin, rot, grid, map_depth)
            map_keep = self._classify(map_wp, n_map, origin, rot, grid, scan_depth)
            wp.synchronize()
            return scan_keep.numpy().astype(bool), map_keep.numpy().astype(bool)
