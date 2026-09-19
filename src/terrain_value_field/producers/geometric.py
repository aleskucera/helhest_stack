"""A producer built from geometry alone: slope and step, from a (mean, sigma) heightmap.

The point of shipping this is that it is not a toy. A great many robots decide traversability
exactly this way -- too steep, or too tall a step -- and for them this is the whole feasibility
model, with the uncertainty handled properly. It is also the reference for what a producer owes
the library: per-state margins, their sigmas, and their floors, in whatever units it likes.

Both constraints propagate the map's own per-cell sd through a linear estimator, so a cell the
mapper is unsure about widens the margin it needs rather than being silently trusted:

  slope  a central difference over the footprint half-width `h`. Its sd follows from the two
         cells it differences, which is why a smoother map is not automatically a safer one.
  step   the largest departure from the local PLANE, not the footprint's peak-to-trough. Those
         differ, and the difference matters: on a smooth 20 degree incline a 0.8 m footprint
         spans 0.29 m top to bottom with no step present at all, so a peak-to-trough measure
         re-reports the slope and the two constraints stop being independent. Removing the
         plane the slope term already fitted leaves roughness, which is what a step limit is
         actually about.

Neither uses a body model, so neither depends on a physics engine. A robot that has one -- a
settle, a contact solve -- should write its own producer and keep this as the shape to follow.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..field import Constraints
from ..grid import Grid
from ..grid import sample

SLOPE = wp.constant(0)
STEP = wp.constant(1)


@wp.kernel
def geometric_margins_kernel(
    height: wp.array2d(dtype=wp.float32),
    height_sd: wp.array2d(dtype=wp.float32),
    grid: Grid,
    footprint_m: wp.float32,
    max_slope: wp.float32,  # [rad]
    max_step: wp.float32,  # [m]
    n_theta: wp.int32,
    margin: wp.array4d(dtype=wp.float32),
    sigma: wp.array4d(dtype=wp.float32),
):
    """Slope and step margins, with the map's own uncertainty carried into each.

    Isotropic: neither constraint depends on which way the robot faces, so every heading bin of
    a cell gets the same answer. That is not a limitation of the library -- the state space
    still carries heading for the control set's sake -- it is a statement about THIS producer,
    and a producer whose constraints are direction-aware simply fills the bins differently.
    """
    r, c, t = wp.tid()
    x = grid.origin_x + float(c) * grid.cell_size
    y = grid.origin_y + float(r) * grid.cell_size
    h = footprint_m

    # --- slope: central differences across the footprint ---------------------------------
    hxp = sample(height, grid, x + h, y)
    hxm = sample(height, grid, x - h, y)
    hyp = sample(height, grid, x, y + h)
    hym = sample(height, grid, x, y - h)
    gx = (hxp - hxm) / (2.0 * h)
    gy = (hyp - hym) / (2.0 * h)
    slope = wp.atan(wp.sqrt(gx * gx + gy * gy))

    sxp = sample(height_sd, grid, x + h, y)
    sxm = sample(height_sd, grid, x - h, y)
    syp = sample(height_sd, grid, x, y + h)
    sym = sample(height_sd, grid, x, y - h)
    # A difference of two independent cells: variances add, then divide by the baseline. The
    # baseline is why a wide footprint is not only more conservative but also less UNCERTAIN.
    var_gx = (sxp * sxp + sxm * sxm) / (4.0 * h * h)
    var_gy = (syp * syp + sym * sym) / (4.0 * h * h)
    # d(atan(g))/dg <= 1, so taking 1 is the conservative linearisation and costs nothing.
    sd_slope = wp.sqrt(var_gx + var_gy)

    margin[SLOPE, r, c, t] = max_slope - slope
    sigma[SLOPE, r, c, t] = sd_slope

    # --- step: the largest departure from the plane the slope term just fitted -------------
    h0 = sample(height, grid, x, y)
    n = int(wp.ceil(h / grid.cell_size))
    worst = float(0.0)
    sd_worst = float(0.0)
    for dr in range(-n, n + 1):
        for dc in range(-n, n + 1):
            rr = wp.clamp(r + dr, 0, height.shape[0] - 1)
            cc = wp.clamp(c + dc, 0, height.shape[1] - 1)
            dx = float(dc) * grid.cell_size
            dy = float(dr) * grid.cell_size
            resid = wp.abs(height[rr, cc] - (h0 + gx * dx + gy * dy))
            if resid > worst:
                worst = resid
                sd_worst = height_sd[rr, cc]
    step = worst
    # Difference of the offending cell against the plane's own centre. Reading the sd off the
    # arg-max is the one-hot estimate and runs HIGH where two cells nearly tie for worst; a
    # producer that cares can fold the maximum instead of taking it.
    sd_centre = sample(height_sd, grid, x, y)
    sd_step = wp.sqrt(sd_worst * sd_worst + sd_centre * sd_centre)

    margin[STEP, r, c, t] = max_step - step
    sigma[STEP, r, c, t] = sd_step


class GeometricProducer:
    """Slope-and-step feasibility from a (mean, sd) heightmap. Owns its output buffers."""

    N_CONSTRAINTS = 2

    def __init__(
        self,
        rows: int,
        cols: int,
        n_theta: int,
        *,
        footprint_m: float = 0.4,
        max_slope_rad: float = 0.45,
        max_step_m: float = 0.15,
        slope_floor_rad: float = 0.02,
        step_floor_m: float = 0.01,
        device: wp.Device | str | None = None,
    ) -> None:
        self.device = wp.get_device(device)
        self.footprint_m = float(footprint_m)
        self.max_slope = float(max_slope_rad)
        self.max_step = float(max_step_m)
        self.n_theta = int(n_theta)
        shape = (self.N_CONSTRAINTS, int(rows), int(cols), int(n_theta))
        self.margin = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self.sigma = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self.floor = wp.array(
            np.array([slope_floor_rad, step_floor_m], np.float32),
            dtype=wp.float32,
            device=self.device,
        )

    def __call__(self, height: wp.array, height_sd: wp.array, grid: Grid) -> Constraints:
        # A map whose shape disagrees with the grid reads out of bounds and returns plausible
        # nonsense rather than failing -- `locate` clamps to the GRID's extent, not the array's.
        # Caught the hard way by a test that passed a (1, N) broadcast row as an (N, N) map.
        want = (grid.cells_y, grid.cells_x)
        for name, arr in (("height", height), ("height_sd", height_sd)):
            if tuple(arr.shape) != want:
                raise ValueError(f"{name} is {tuple(arr.shape)}, but the grid is {want}")
        if tuple(self.margin.shape[1:3]) != want:
            raise ValueError(f"producer built for {tuple(self.margin.shape[1:3])}, grid is {want}")
        wp.launch(
            geometric_margins_kernel,
            dim=self.margin.shape[1:],
            inputs=[
                height,
                height_sd,
                grid,
                self.footprint_m,
                self.max_slope,
                self.max_step,
                self.n_theta,
            ],
            outputs=[self.margin, self.sigma],
            device=self.device,
        )
        return Constraints(margin=self.margin, sigma=self.sigma, floor=self.floor)
