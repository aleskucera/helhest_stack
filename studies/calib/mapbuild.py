"""Device-resident elevation accumulation for the calibration harness.

Points land on the GPU once per frame and stay there: the per-cell fold is a Warp kernel,
never a numpy scatter (CLAUDE.md section 6). Two levels are kept because the two calibration
probes need different things:

  per-FRAME  MAX height per cell, from one sweep       -> one independent observation of a cell
  per-MAP    mean over frame maxima, plus sum of squares -> the belief, and its across-frame spread

Two deliberate choices.

`max` within a frame, because that is what the deployed pipeline uses (`primary: "max"`,
_pipeline_common.py:187) and what the physics wants: a wheel rests on the highest point
beneath it, not the average. A mean over the vertical column is simply wrong -- measured on
ostrich4, averaging floor returns together with a person standing beside the robot put the
cells under the wheels at +1.5 m against a real floor at -0.44 m.

Mean ACROSS frames for the belief, not the running max the pipeline keeps. A running max is
monotone and biased high, and being an extreme it has no natural standard deviation; each
frame's own max is an independent estimate of the same surface, so averaging them gives a
belief whose across-frame spread IS the per-cell sigma. That the deployed statistic has no
sigma to publish is itself an argument for the new mapper's mean+sigma form.

Points are gated to a z-window about the robot's own height before any of this: a wheel can
only contact terrain within roughly [z_robot - z_below, z_robot + z_above], and everything
above that is overhead structure the wheel never touches.
"""

from __future__ import annotations

import numpy as np
import warp as wp


@wp.kernel
def accumulate_frame_kernel(
    points: wp.array(dtype=wp.vec3f),
    sensor_x: wp.float32,
    sensor_y: wp.float32,
    sensor_z: wp.float32,
    z_below: wp.float32,
    z_above: wp.float32,
    origin_x: wp.float32,
    origin_y: wp.float32,
    cell_size: wp.float32,
    max_range: wp.float32,
    frame_max: wp.array2d(dtype=wp.float32),
    frame_sum: wp.array2d(dtype=wp.float32),
    frame_cnt: wp.array2d(dtype=wp.float32),
    frame_rng: wp.array2d(dtype=wp.float32),
):
    """Scatter one sweep's points into per-cell max / sum / count / range sum.

    Both reductions are collected so the per-cell statistic can be switched without a second
    pass: `max` is what the deployed pipeline publishes, `mean` is a lower-variance estimator of
    the same surface, and the gap between them is measurable.

    Range is the horizontal sensor-to-point distance, carried so the per-cell observation can
    later be binned by it: a dTOF return degrades with range, and the mapper's sigma model has
    to reproduce that.
    """
    i = wp.tid()
    p = points[i]
    if p[2] < sensor_z - z_below or p[2] > sensor_z + z_above:
        return  # overhead structure / sub-floor noise: no wheel can contact it
    dx = p[0] - sensor_x
    dy = p[1] - sensor_y
    rng = wp.sqrt(dx * dx + dy * dy)
    if rng > max_range:
        return
    c = int((p[0] - origin_x) / cell_size)
    r = int((p[1] - origin_y) / cell_size)
    if r < 0 or r >= frame_max.shape[0] or c < 0 or c >= frame_max.shape[1]:
        return
    wp.atomic_max(frame_max, r, c, p[2])
    wp.atomic_add(frame_sum, r, c, p[2])
    wp.atomic_add(frame_cnt, r, c, 1.0)
    wp.atomic_add(frame_rng, r, c, rng)


@wp.kernel
def fold_frame_kernel(
    frame_max: wp.array2d(dtype=wp.float32),
    frame_sum: wp.array2d(dtype=wp.float32),
    frame_cnt: wp.array2d(dtype=wp.float32),
    frame_rng: wp.array2d(dtype=wp.float32),
    min_points: wp.float32,
    use_max: wp.int32,
    map_sum: wp.array2d(dtype=wp.float32),
    map_sumsq: wp.array2d(dtype=wp.float32),
    map_nobs: wp.array2d(dtype=wp.float32),
    map_rng: wp.array2d(dtype=wp.float32),
    elev: wp.array2d(dtype=wp.float32),
    measured: wp.array2d(dtype=wp.float32),
):
    """Fold this frame's per-cell max into the running belief, and republish `elev`.

    A cell needs `min_points` returns in the sweep to count as observed by it, which drops the
    single-ray grazing hits that otherwise dominate the far field.
    """
    r, c = wp.tid()
    n = frame_cnt[r, c]
    if n < min_points:
        return
    if use_max != 0:
        h = frame_max[r, c]
    else:
        h = frame_sum[r, c] / n
    wp.atomic_add(map_sum, r, c, h)
    wp.atomic_add(map_sumsq, r, c, h * h)
    wp.atomic_add(map_nobs, r, c, 1.0)
    wp.atomic_add(map_rng, r, c, frame_rng[r, c] / n)
    k = map_nobs[r, c]
    elev[r, c] = map_sum[r, c] / k
    measured[r, c] = 1.0


@wp.kernel
def fill_unobserved_kernel(
    measured: wp.array2d(dtype=wp.float32),
    fill_z: wp.float32,
    elev: wp.array2d(dtype=wp.float32),
):
    """Give never-observed cells an explicit constant, so the settle sees a defined surface.

    This is the blind-cell fill the planner is meant to stop needing; here it is only a harness
    convenience, and probes are rejected unless their whole footprint is measured.
    """
    r, c = wp.tid()
    if measured[r, c] < 0.5:
        elev[r, c] = fill_z


class MapAccumulator:
    """Causal elevation accumulator: `add_frame` in bag order, read `elev` at any point."""

    def __init__(
        self,
        cells_y: int,
        cells_x: int,
        cell_size: float,
        origin_x: float,
        origin_y: float,
        max_range: float = 15.0,
        min_points_per_cell: int = 2,
        z_below: float = 1.5,
        z_above: float = 0.5,
        stat: str = "max",  # per-cell per-frame reduction: "max" (the pipeline) or "mean"
        device: wp.Device | str | None = None,
    ) -> None:
        self.device = wp.get_device(device)
        self.cells_y = cells_y
        self.cells_x = cells_x
        self.cell_size = cell_size
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.max_range = float(max_range)
        self.min_points = float(min_points_per_cell)
        self.z_below = float(z_below)
        self.z_above = float(z_above)
        if stat not in ("max", "mean"):
            raise ValueError(f"stat must be 'max' or 'mean', got {stat!r}")
        self.use_max = 1 if stat == "max" else 0
        shape = (cells_y, cells_x)
        z = lambda: wp.zeros(shape, dtype=wp.float32, device=self.device)  # noqa: E731
        self.frame_max, self.frame_sum = z(), z()
        self.frame_cnt, self.frame_rng = z(), z()
        self.map_sum, self.map_sumsq, self.map_nobs, self.map_rng = z(), z(), z(), z()
        self.elev = z()
        self.measured = z()
        self._pts = None  # grown on demand; reused across frames

    def _upload(self, points: np.ndarray) -> wp.array:
        """Host cloud -> device vec3 array, reusing the buffer when it is already big enough."""
        n = len(points)
        if self._pts is None or len(self._pts) < n:
            self._pts = wp.zeros(max(n, 65536), dtype=wp.vec3f, device=self.device)
        wp.copy(self._pts, wp.array(points, dtype=wp.vec3f, device=self.device), count=n)
        return self._pts

    def add_frame(self, points: np.ndarray, sensor_xyz: np.ndarray) -> None:
        """Fold one sweep into the belief. `sensor_xyz` sets the range origin and the z-window."""
        n = len(points)
        if n == 0:
            return
        pts = self._upload(points)
        self.frame_max.fill_(-1.0e9)  # max-reduction identity; empty cells are gated by count
        self.frame_sum.zero_()
        self.frame_cnt.zero_()
        self.frame_rng.zero_()
        wp.launch(
            accumulate_frame_kernel,
            dim=n,
            inputs=[
                pts,
                float(sensor_xyz[0]),
                float(sensor_xyz[1]),
                float(sensor_xyz[2]),
                self.z_below,
                self.z_above,
                self.origin_x,
                self.origin_y,
                self.cell_size,
                self.max_range,
            ],
            outputs=[self.frame_max, self.frame_sum, self.frame_cnt, self.frame_rng],
            device=self.device,
        )
        wp.launch(
            fold_frame_kernel,
            dim=(self.cells_y, self.cells_x),
            inputs=[
                self.frame_max,
                self.frame_sum,
                self.frame_cnt,
                self.frame_rng,
                self.min_points,
                self.use_max,
            ],
            outputs=[
                self.map_sum,
                self.map_sumsq,
                self.map_nobs,
                self.map_rng,
                self.elev,
                self.measured,
            ],
            device=self.device,
        )

    def elevation(self, fill_z: float) -> wp.array:
        """The belief, with never-observed cells set to `fill_z`."""
        wp.launch(
            fill_unobserved_kernel,
            dim=(self.cells_y, self.cells_x),
            inputs=[self.measured, float(fill_z)],
            outputs=[self.elev],
            device=self.device,
        )
        return self.elev
