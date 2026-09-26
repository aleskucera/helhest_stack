"""The world-to-cell mapping, and bilinear sampling on it -- the one grid every kernel shares.

`origin` is the CENTRE of cell (0, 0). The host-side `engine.GridParams` keeps the map's min
corner, which is what a caller measuring a window computes, and adds the half cell once in
`build()`; everything on the device sees cell centres. There used to be two structs here, half a
cell apart, and both defects that produced were real (tests/planning/test_grid_conventions.py).

Dependency-free on purpose: the engine, the planners and the cost-to-go library all read it, so it
sits below all of them.
"""

from __future__ import annotations

import warp as wp


@wp.struct
class Grid:
    """A regular 2-D grid. `origin` is the centre of cell (0, 0)."""

    cells_x: wp.int32
    cells_y: wp.int32
    cell_size: wp.float32  # meters per cell
    origin_x: wp.float32
    origin_y: wp.float32


def build_grid(
    cells_x: int, cells_y: int, cell_size: float, origin_x: float, origin_y: float
) -> Grid:
    """A `Grid` from a CENTRE-of-cell-(0, 0) origin. From a min corner, use `GridParams.build()`."""
    g = Grid()
    g.cells_x, g.cells_y = int(cells_x), int(cells_y)
    g.cell_size = float(cell_size)
    g.origin_x, g.origin_y = float(origin_x), float(origin_y)
    return g


@wp.func
def locate(grid: Grid, x: wp.float32, y: wp.float32):
    """World (x, y) -> bilinear stencil, packed as vec4(x_idx, y_idx, frac_x, frac_y): the lower-left
    corner index (as a float -- cast back with int()) and the in-cell offset toward +1, in [0,1].
    The ONE place the cell-centre mapping `(x - origin)/cell_size` lives -- shared by
    `sample_field`, its analytic gradient, and the d/dH adjoint scatter so the convention can never
    drift. Packed in a PLAIN vec4, not a struct: an int-member struct round-trip zeroes the auto-grad
    of `frac` w.r.t. (x, y), which silently kills `sample_field`'s POSITION gradient (e.g. friction
    sampled at a pose-dependent contact point -- a cross-step term that grows with the rollout)."""
    fx = (x - grid.origin_x) / grid.cell_size
    fy = (y - grid.origin_y) / grid.cell_size
    x_idx = wp.clamp(int(wp.floor(fx)), 0, grid.cells_x - 2)
    y_idx = wp.clamp(int(wp.floor(fy)), 0, grid.cells_y - 2)
    frac_x = wp.clamp(fx - float(x_idx), 0.0, 1.0)
    frac_y = wp.clamp(fy - float(y_idx), 0.0, 1.0)
    return wp.vec4(float(x_idx), float(y_idx), frac_x, frac_y)


@wp.func
def sample_field(
    field: wp.array2d(dtype=wp.float32),
    grid: Grid,
    x: wp.float32,
    y: wp.float32,
):
    """Bilinear-interpolate a 2D grid field (elevation, envelope, friction, ...) at world (x, y),
    clamped at the border. Differentiable w.r.t. BOTH the field values and the sample position
    (x, y) -- see `locate`."""
    c = locate(grid, x, y)
    xi = int(c[0])
    yi = int(c[1])
    frac_x = c[2]
    frac_y = c[3]

    v00 = field[yi, xi]
    v10 = field[yi, xi + 1]
    v01 = field[yi + 1, xi]
    v11 = field[yi + 1, xi + 1]

    return (
        (1.0 - frac_x) * (1.0 - frac_y) * v00
        + frac_x * (1.0 - frac_y) * v10
        + (1.0 - frac_x) * frac_y * v01
        + frac_x * frac_y * v11
    )


@wp.func
def world_of(grid: Grid, row: wp.int32, col: wp.int32):
    """Cell indices -> the world point at that cell's centre."""
    return wp.vec2(
        grid.origin_x + float(col) * grid.cell_size,
        grid.origin_y + float(row) * grid.cell_size,
    )
