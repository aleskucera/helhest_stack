from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .kernels import compact_inliers_kernel
from .kernels import neighbor_count_kernel


@dataclass
class OutlierFilterConfig:
    """Configuration for `StatisticalOutlierFilter`."""

    # [m] neighbour search radius
    search_radius_m: float = 0.25
    # Points with fewer than this many neighbours inside `search_radius_m` are rejected as
    # isolated. Nothing else is: a mean-distance test over the same ball once sat on top of
    # this and removed every return within ~1 m of the sensor (see `StatisticalOutlierFilter`).
    min_neighbors: int = 10


def _hashgrid_dims_from_extent(extent: np.ndarray, radius: float) -> tuple[int, int, int]:
    extent = np.maximum(extent, radius)
    cells = np.ceil(extent / max(radius, 1.0e-6)).astype(int)
    cells = np.clip(cells, 8, 256)
    return int(cells[0]), int(cells[1]), int(cells[2])


def _hashgrid_dims_from_points(points_np: np.ndarray, radius: float) -> tuple[int, int, int]:
    mins = points_np.min(axis=0)
    maxs = points_np.max(axis=0)
    return _hashgrid_dims_from_extent(maxs - mins, radius)


def _hashgrid_dims_from_bounds(
    bounds: tuple[float, float, float, float, float, float],
    radius: float,
) -> tuple[int, int, int]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    extent = np.array([xmax - xmin, ymax - ymin, zmax - zmin], dtype=np.float64)
    return _hashgrid_dims_from_extent(extent, radius)


class StatisticalOutlierFilter:
    """GPU-native isolated-point removal: drop points with fewer than `min_neighbors` neighbours
    within `search_radius_m`.

    This used to add a statistical test -- reject where the range-normalised mean distance to
    the neighbours in the ball exceeded mean + 1 sd over the cloud. Averaged over a FIXED ball,
    that distance is ~2/3 of the radius at any density (15-17 cm at 0.25 m on Odin clouds, from
    1400 neighbours near the robot to 7 at 10 m), so divided by range it was ~0.16/range and the
    test removed every return inside ~1 m: 19% of each cloud, and never a speck past 1.5 m. The
    count gate is what removes specks.

    Compaction happens on the GPU; only the output count is read back. Accepts numpy or
    `wp.array` input; returns the matching type.
    """

    def __init__(
        self,
        config: OutlierFilterConfig | None = None,
        *,
        bounds: tuple[float, float, float, float, float, float] | None = None,
        device: wp.context.Device | None = None,
    ):
        self.config = config or OutlierFilterConfig()
        self.device = wp.get_device(device)
        self._grid: wp.HashGrid | None = None

        # If the caller knows the point-cloud extent ahead of time, precreate
        # the hashgrid so we never touch the CPU to size it. Otherwise the
        # first apply() does a one-time numpy readback to pick dims.
        if bounds is not None:
            dims = _hashgrid_dims_from_bounds(bounds, self.config.search_radius_m)
            with wp.ScopedDevice(self.device):
                self._grid = wp.HashGrid(*dims, device=self.device)

        # Per-point outputs, grown on demand.
        self._valid: wp.array | None = None
        self._out_pts: wp.array | None = None
        self._capacity: int = 0

        with wp.ScopedDevice(self.device):
            self._out_counter = wp.zeros(1, dtype=wp.int32)

    def _ensure_grid(self, radius: float, pts_wp: wp.array) -> wp.HashGrid:
        if self._grid is None or self._grid.device != self.device:
            # One-time readback to size the grid when no bounds were supplied.
            dims = _hashgrid_dims_from_points(pts_wp.numpy(), radius)
            self._grid = wp.HashGrid(*dims, device=self.device)
        return self._grid

    def _ensure_buffers(self, n: int) -> None:
        if self._capacity >= n and self._valid is not None:
            return
        with wp.ScopedDevice(self.device):
            self._valid = wp.empty(n, dtype=wp.int32)
            self._out_pts = wp.empty(n, dtype=wp.vec3)
        self._capacity = n

    def apply(self, points: np.ndarray | wp.array) -> np.ndarray | wp.array:
        """Return `points` with outliers removed. Input and output types match."""
        cfg = self.config
        return_numpy = isinstance(points, np.ndarray)

        if return_numpy:
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"points must be (N, 3); got {points.shape}")
            n = len(points)
            pts_np_f32 = np.ascontiguousarray(points, dtype=np.float32)
            pts_wp = wp.array(pts_np_f32, dtype=wp.vec3, device=self.device)
        else:
            n = len(points)
            pts_wp = points

        if n <= cfg.min_neighbors:
            return points

        with wp.ScopedDevice(self.device):
            grid = self._ensure_grid(cfg.search_radius_m, pts_wp)
            grid.build(points=pts_wp, radius=float(cfg.search_radius_m))

            self._ensure_buffers(n)
            wp.launch(
                neighbor_count_kernel,
                dim=n,
                inputs=[grid.id, pts_wp, float(cfg.search_radius_m), int(cfg.min_neighbors)],
                outputs=[self._valid],
            )
            self._out_counter.zero_()
            wp.launch(
                compact_inliers_kernel,
                dim=n,
                inputs=[pts_wp, self._valid],
                outputs=[self._out_counter, self._out_pts],
            )
            wp.synchronize()
            n_out = int(self._out_counter.numpy()[0])

            if return_numpy:
                return self._out_pts.numpy()[:n_out].astype(points.dtype, copy=True)

            # GPU path: allocate a right-sized output and copy from the compact buffer.
            out = wp.empty(n_out, dtype=wp.vec3, device=self.device)
            if n_out > 0:
                wp.copy(out, self._out_pts, 0, 0, n_out)
            return out
