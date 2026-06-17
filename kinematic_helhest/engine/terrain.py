"""Device-side heightmap: bilinear height + central-difference normal.

Mirrors the numpy `heightmap.Heightmap` exactly (same clamping, same stencil) so
the numpy version stays usable as a finite-difference oracle. The wheel-envelope
surface is precomputed on the host (`heightmap.wheel_envelope`) and handed in as
the height grid; this module just samples it.

The height grid `elevation` is passed to kernels as a plain `wp.array` (not a struct
member) so the tape accumulates its adjoint cleanly for the Phase-5 d/dh gradient.
The grid's non-differentiated metadata (origin, cell, size) rides in a small
`Grid` struct alongside it.
"""
from dataclasses import dataclass

import numpy as np
import warp as wp


@wp.struct
class Grid:
    """Non-differentiated grid metadata, passed with the plain `elevation` array."""

    x0: wp.float32
    y0: wp.float32
    cell: wp.float32
    nx: wp.int32
    ny: wp.int32


@dataclass
class Terrain:
    """Host-side bag: a differentiated grid `elevation` (plain array) + its Grid."""

    elevation: wp.array  # array2d(float32) [ny, nx]
    g: Grid


@dataclass
class GridParams:
    """Fixed grid the Simulator preallocates for: dims, cell, node origin, wheel R."""

    nx: int
    ny: int
    cell: float
    x0: float
    y0: float
    R: float = 0.35  # wheel radius -> envelope neighborhood radius

    @classmethod
    def from_heightmap(cls, hm, R=0.35):
        return cls(int(hm.nx), int(hm.ny), float(hm.cell), float(hm.x0), float(hm.y0), R)

    def build(self) -> Grid:
        g = Grid()
        g.x0, g.y0, g.cell = float(self.x0), float(self.y0), float(self.cell)
        g.nx, g.ny = int(self.nx), int(self.ny)
        return g


def to_terrain(hm, device="cpu", requires_grad=False) -> Terrain:
    """Build a device Terrain from a numpy `heightmap.Heightmap`."""
    elevation = wp.array(
        np.ascontiguousarray(hm.H, dtype=np.float32),
        dtype=wp.float32, device=device, requires_grad=requires_grad,
    )
    g = Grid()
    g.x0 = float(hm.x0)
    g.y0 = float(hm.y0)
    g.cell = float(hm.cell)
    g.nx = int(hm.nx)
    g.ny = int(hm.ny)
    return Terrain(elevation, g)


def bounds_to_origin(bounds, res):
    """Cell-center grid bounds (xmin, xmax, ymin, ymax) -> node origin (x0, y0).

    An external rasterizer (e.g. terrain_toolkit) stores a cell's value at its
    CENTER: x = xmin + (j+0.5)*res. `sample_height` here treats grid values as
    sitting on NODES: value at x = x0 + ix*cell. So the node origin is the center
    of cell (0,0): x0 = xmin + 0.5*res. Using xmin directly biases every sample by
    half a cell.
    """
    xmin, _, ymin, _ = bounds
    return xmin + 0.5 * res, ymin + 0.5 * res


def terrain_from_device(elevation, x0, y0, cell) -> Terrain:
    """Wrap an already-on-device height grid as a Terrain, no numpy round-trip.

    `elevation` is a device `wp.array2d(float32)` of shape [ny, nx] (row=y, col=x), e.g.
    handed over from another Warp library. The grid metadata is supplied directly;
    pair with `bounds_to_origin` to convert a cell-center-bounds raster.
    """
    ny, nx = elevation.shape
    g = Grid()
    g.x0 = float(x0)
    g.y0 = float(y0)
    g.cell = float(cell)
    g.nx = int(nx)
    g.ny = int(ny)
    return Terrain(elevation, g)


@wp.func
def sample_height(elevation: wp.array2d(dtype=wp.float32), g: Grid, x: float, y: float):
    fx = (x - g.x0) / g.cell
    fy = (y - g.y0) / g.cell
    ix = wp.clamp(int(wp.floor(fx)), 0, g.nx - 2)
    iy = wp.clamp(int(wp.floor(fy)), 0, g.ny - 2)
    tx = wp.clamp(fx - float(ix), 0.0, 1.0)
    ty = wp.clamp(fy - float(iy), 0.0, 1.0)
    h00 = elevation[iy, ix]
    h10 = elevation[iy, ix + 1]
    h01 = elevation[iy + 1, ix]
    h11 = elevation[iy + 1, ix + 1]
    return ((1.0 - tx) * (1.0 - ty) * h00 + tx * (1.0 - ty) * h10
            + (1.0 - tx) * ty * h01 + tx * ty * h11)


@wp.func
def sample_height_grad(elevation: wp.array2d(dtype=wp.float32), g: Grid, x: float, y: float):
    """Bilinear height AND its exact in-cell gradient, from one 4-corner fetch.

    Returns vec3(h, dH/dx, dH/dy). The gradient is the analytic derivative of the
    bilinear `sample_height` (NOT the wider central-difference of sample_normal),
    so it is exactly d(sample_height)/d(x,y) -- what the settle Jacobian needs.
    """
    fx = (x - g.x0) / g.cell
    fy = (y - g.y0) / g.cell
    ix = wp.clamp(int(wp.floor(fx)), 0, g.nx - 2)
    iy = wp.clamp(int(wp.floor(fy)), 0, g.ny - 2)
    tx = wp.clamp(fx - float(ix), 0.0, 1.0)
    ty = wp.clamp(fy - float(iy), 0.0, 1.0)
    h00 = elevation[iy, ix]
    h10 = elevation[iy, ix + 1]
    h01 = elevation[iy + 1, ix]
    h11 = elevation[iy + 1, ix + 1]
    h = ((1.0 - tx) * (1.0 - ty) * h00 + tx * (1.0 - ty) * h10
         + (1.0 - tx) * ty * h01 + tx * ty * h11)
    gx = ((1.0 - ty) * (h10 - h00) + ty * (h11 - h01)) / g.cell
    gy = ((1.0 - tx) * (h01 - h00) + tx * (h11 - h10)) / g.cell
    return wp.vec3(h, gx, gy)


@wp.func
def sample_normal(elevation: wp.array2d(dtype=wp.float32), g: Grid, x: float, y: float):
    e = g.cell
    dhdx = (sample_height(elevation, g, x + e, y) - sample_height(elevation, g, x - e, y)) / (2.0 * e)
    dhdy = (sample_height(elevation, g, x, y + e) - sample_height(elevation, g, x, y - e)) / (2.0 * e)
    return wp.normalize(wp.vec3(-dhdx, -dhdy, 1.0))


@wp.kernel
def _probe(elevation: wp.array2d(dtype=wp.float32), g: Grid,
           xs: wp.array(dtype=wp.float32), ys: wp.array(dtype=wp.float32),
           out_h: wp.array(dtype=wp.float32), out_n: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    out_h[i] = sample_height(elevation, g, xs[i], ys[i])
    out_n[i] = sample_normal(elevation, g, xs[i], ys[i])


# `_probe` (above) is the device sampler reused by tests/engine/terrain.py and
# the planning device-path check; the sampling-vs-oracle self-test lives there.
