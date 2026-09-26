"""The world-to-cell mapping, and bilinear sampling on it.

Deliberately tiny and dependency-free. A value field needs to know where a cell is and how to
read a map between cells; it does not need a physics engine's terrain module to do that, and
depending on one would tie this library to a single robot's stack.
"""

from __future__ import annotations

import warp as wp


@wp.struct
class Grid:
    """A regular 2-D grid. `origin` is the centre of cell (0, 0).

    Worth stating because a neighbour disagrees: `helhest.engine.terrain.Grid` carries these same
    five fields and takes its origin to be the map's MIN CORNER, so the centre of cell i sits at
    origin + (i + 0.5) * cell_size there and at origin + i * cell_size here -- half a cell apart.
    The two are distinct warp types, so passing one where the other is expected is a type error;
    passing loose FLOATS between them is not, and that is where the half cell would hide.
    Converting from a corner-origin grid means adding cell_size / 2 to both origins.
    """

    cells_x: wp.int32
    cells_y: wp.int32
    cell_size: wp.float32
    origin_x: wp.float32
    origin_y: wp.float32


def build_grid(
    cells_x: int, cells_y: int, cell_size: float, origin_x: float, origin_y: float
) -> Grid:
    g = Grid()
    g.cells_x, g.cells_y = int(cells_x), int(cells_y)
    g.cell_size = float(cell_size)
    g.origin_x, g.origin_y = float(origin_x), float(origin_y)
    return g


@wp.func
def locate(grid: Grid, x: wp.float32, y: wp.float32):
    """World (x, y) -> vec4(col, row, frac_x, frac_y): the lower-left cell and the offset in it.

    The single place the cell-centre convention lives, so the sampler and anything that scatters
    into the map cannot drift apart. Packed in a plain vec4 rather than a struct: an int-bearing
    struct round-trip zeroes the position gradient in Warp's autodiff, which silently kills the
    derivative of a sampled field with respect to where it was sampled.
    """
    fx = (x - grid.origin_x) / grid.cell_size
    fy = (y - grid.origin_y) / grid.cell_size
    xi = wp.clamp(int(wp.floor(fx)), 0, grid.cells_x - 2)
    yi = wp.clamp(int(wp.floor(fy)), 0, grid.cells_y - 2)
    return wp.vec4(
        float(xi), float(yi), wp.clamp(fx - float(xi), 0.0, 1.0), wp.clamp(fy - float(yi), 0.0, 1.0)
    )


@wp.func
def sample(
    field: wp.array2d(dtype=wp.float32), grid: Grid, x: wp.float32, y: wp.float32
) -> wp.float32:
    """Bilinear read of a [rows, cols] field at a world point, clamped at the border."""
    c = locate(grid, x, y)
    xi, yi = int(c[0]), int(c[1])
    fx, fy = c[2], c[3]
    a = field[yi, xi] * (1.0 - fx) + field[yi, xi + 1] * fx
    b = field[yi + 1, xi] * (1.0 - fx) + field[yi + 1, xi + 1] * fx
    return a * (1.0 - fy) + b * fy


@wp.func
def world_of(grid: Grid, row: wp.int32, col: wp.int32):
    """Cell indices -> the world point at that cell's centre."""
    return wp.vec2(
        grid.origin_x + float(col) * grid.cell_size,
        grid.origin_y + float(row) * grid.cell_size,
    )
