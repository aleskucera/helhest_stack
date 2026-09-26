from dataclasses import dataclass

import warp as wp

from ..grid import Grid
from ..grid import locate as _locate
from ..grid import sample_field


@dataclass
class GridParams:
    cells_x: int
    cells_y: int
    cell_size: float
    origin_x: float
    origin_y: float

    @property
    def bounds(self) -> tuple:
        """World extent (xmin, xmax, ymin, ymax) -- the convention the geodesic solvers take."""
        return (
            self.origin_x,
            self.origin_x + self.cells_x * self.cell_size,
            self.origin_y,
            self.origin_y + self.cells_y * self.cell_size,
        )

    def build(self) -> Grid:
        """Host min-corner -> device CENTRE-of-cell-(0,0), which is the one convention the
        kernels use.

        The half cell is added HERE, once, rather than subtracted in every kernel that locates a
        point. `GridParams.origin_x` keeps its public meaning -- the map's min corner, which is
        what a caller measuring a window computes -- while the struct the kernels read carries
        the cell centre (`helhest.grid.Grid`). Sampling is unchanged to the
        bit: `(x - (o + c/2))/c` is exactly the `(x - o)/c - 0.5` this replaced.

        Two half-cell defects came out of having the two conventions coexist: `_margin_kernel`
        placed a pose at `origin + c*cell` and then sampled sigma through the min-corner
        `_locate`, and the cost-to-go's boundary ring read the coarse field half a cell off (see
        `tests/planning/test_grid_conventions.py`, and 518876a). One convention is what stops
        that recurring.
        """
        grid = Grid()
        grid.cells_x, grid.cells_y = int(self.cells_x), int(self.cells_y)
        grid.cell_size = float(self.cell_size)
        grid.origin_x = float(self.origin_x) + 0.5 * float(self.cell_size)
        grid.origin_y = float(self.origin_y) + 0.5 * float(self.cell_size)
        return grid


@wp.func
def sample_height_grad(
    elevation: wp.array2d(dtype=wp.float32),
    grid: Grid,
    x: wp.float32,
    y: wp.float32,
):
    c = _locate(grid, x, y)
    xi = int(c[0])
    yi = int(c[1])
    frac_x = c[2]
    frac_y = c[3]
    h00 = elevation[yi, xi]
    h10 = elevation[yi, xi + 1]
    h01 = elevation[yi + 1, xi]
    h11 = elevation[yi + 1, xi + 1]

    h = (
        (1.0 - frac_x) * (1.0 - frac_y) * h00
        + frac_x * (1.0 - frac_y) * h10
        + (1.0 - frac_x) * frac_y * h01
        + frac_x * frac_y * h11
    )

    gx = ((1.0 - frac_y) * (h10 - h00) + frac_y * (h11 - h01)) / grid.cell_size
    gy = ((1.0 - frac_x) * (h01 - h00) + frac_x * (h11 - h10)) / grid.cell_size
    return wp.vec3(h, gx, gy)


@wp.func
def sample_normal(
    elevation: wp.array2d(dtype=wp.float32),
    grid: Grid,
    x: float,
    y: float,
):
    e = grid.cell_size
    dhdx = (sample_field(elevation, grid, x + e, y) - sample_field(elevation, grid, x - e, y)) / (
        2.0 * e
    )
    dhdy = (sample_field(elevation, grid, x, y + e) - sample_field(elevation, grid, x, y - e)) / (
        2.0 * e
    )
    return wp.normalize(wp.vec3(-dhdx, -dhdy, 1.0))
