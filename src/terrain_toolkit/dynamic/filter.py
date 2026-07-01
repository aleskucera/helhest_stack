from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from .kernels import classify_kernel
from .kernels import render_depth_kernel

_DEPTH_SENTINEL = 1.0e18


@dataclass
class DynamicFilterConfig:
    """Configuration for `DynamicPointFilter`.

    The filter compares a new scan against the accumulated map by visibility: it
    renders each into a spherical range image from the current sensor origin and
    flags points that sit in front of the other's surface. It removes MOVING
    objects only — a beam has to see *through* where something was for it to be
    carved. A motionless person is geometrically a static pillar and is kept.
    """

    # Angular resolution of the range image. Match roughly to the sensor: azimuth
    # over the full 360°, elevation over the vertical FOV below.
    az_bins: int = 900
    el_bins: int = 64
    # Vertical field of view (degrees) the elevation bins span. Points outside are
    # ignored by the visibility test (kept). Widen for high-FOV sensors.
    el_min_deg: float = -25.0
    el_max_deg: float = 25.0
    # A point is dynamic if the other cloud's surface along its bearing is farther
    # than the point by more than `margin_m + range * margin_rel`. The relative
    # term absorbs angular quantization on slanted surfaces + registration error.
    margin_m: float = 0.3
    margin_rel: float = 0.02
    # Ignore returns closer than this (sensor self-hits / degenerate directions).
    min_range_m: float = 0.5


class DynamicPointFilter:
    """Map-frame visibility filter for removing moving objects (e.g. people).

    Given the accumulated map, a new scan, and the current sensor pose (all in a
    common world frame), returns two boolean keep-masks:

      * `scan_keep` — False for scan points in front of known static geometry
        (a person walking into previously-free space): drop before fusing.
      * `map_keep`  — False for map points the current scan now sees through
        (a ghost/trail the person left): carve from the map.

    Degrades gracefully: with a stationary sensor this is exactly per-beam range
    background subtraction; with slow motion the tracked pose keeps the two range
    images aligned. Accepts numpy arrays; returns numpy bool masks.
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

    def filter(
        self,
        map_points: np.ndarray,
        scan_points: np.ndarray,
        sensor_origin: np.ndarray,
        *,
        sensor_rotation: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return `(scan_keep, map_keep)` boolean masks.

        `sensor_rotation` is the world→sensor rotation (3x3); it aligns the range
        image with the sensor's beams so the elevation band matches its FOV
        regardless of robot tilt. Identity if omitted.
        """
        cfg = self.config
        n_scan = len(scan_points)
        n_map = len(map_points)
        # Nothing to compare against → keep everything.
        if n_map == 0:
            return np.ones(n_scan, dtype=bool), np.ones(n_map, dtype=bool)
        if n_scan == 0:
            return np.ones(n_scan, dtype=bool), np.ones(n_map, dtype=bool)

        R = np.eye(3) if sensor_rotation is None else np.asarray(sensor_rotation, dtype=np.float64)
        rot = wp.mat33(R.flatten().tolist())
        origin = wp.vec3(*np.asarray(sensor_origin, dtype=np.float64).tolist())

        az_min = -math.pi
        az_span = 2.0 * math.pi
        el_min = math.radians(cfg.el_min_deg)
        el_max = math.radians(cfg.el_max_deg)

        grid_args = [
            int(cfg.az_bins),
            int(cfg.el_bins),
            float(az_min),
            float(az_span),
            float(el_min),
            float(el_max),
            float(cfg.min_range_m),
        ]

        with wp.ScopedDevice(self.device):
            map_wp = wp.array(np.ascontiguousarray(map_points, dtype=np.float32), dtype=wp.vec3)
            scan_wp = wp.array(np.ascontiguousarray(scan_points, dtype=np.float32), dtype=wp.vec3)

            fill = np.full(self._n_bins, _DEPTH_SENTINEL, dtype=np.float32)
            map_depth = wp.array(fill, dtype=wp.float32)
            scan_depth = wp.array(fill.copy(), dtype=wp.float32)

            wp.launch(
                render_depth_kernel,
                dim=n_map,
                inputs=[map_wp, origin, rot, *grid_args],
                outputs=[map_depth],
            )
            wp.launch(
                render_depth_kernel,
                dim=n_scan,
                inputs=[scan_wp, origin, rot, *grid_args],
                outputs=[scan_depth],
            )

            scan_keep = wp.empty(n_scan, dtype=wp.int32)
            map_keep = wp.empty(n_map, dtype=wp.int32)
            wp.launch(
                classify_kernel,
                dim=n_scan,
                inputs=[
                    scan_wp,
                    origin,
                    rot,
                    *grid_args,
                    float(cfg.margin_m),
                    float(cfg.margin_rel),
                    map_depth,
                ],
                outputs=[scan_keep],
            )
            wp.launch(
                classify_kernel,
                dim=n_map,
                inputs=[
                    map_wp,
                    origin,
                    rot,
                    *grid_args,
                    float(cfg.margin_m),
                    float(cfg.margin_rel),
                    scan_depth,
                ],
                outputs=[map_keep],
            )
            wp.synchronize()

            return (
                scan_keep.numpy().astype(bool),
                map_keep.numpy().astype(bool),
            )
