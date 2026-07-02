"""Point cloud -> heightmap rasterizer: the real-sensor front-end that turns a 3D lidar point cloud
into the [ny, nx] elevation grid the planner consumes (the role synthetic `lidar_scan` plays today).

This is THE sim-to-real seam -- the one place a frame transpose or a handedness flip silently corrupts
everything downstream -- so it is guarded by an asymmetric round-trip test
(tests/perception/test_rasterize.py).

Convention (matches Heightmap / GridParams / the engine): REP-103 x-forward, y-left; the grid is
H[ny, nx] = H[row, col] with row -> y and col -> x. (x0, y0) is the grid's MIN CORNER, so cell
(r, c)'s CENTER is world (x0 + (c + 0.5) * cell, y0 + (r + 0.5) * cell), and world point (x, y) bins
to its nearest cell center (round((y - y0) / cell - 0.5), round((x - x0) / cell - 0.5)) -- the same
`(x - x0)/cell - 0.5` mapping the engine's sample_field uses (so a rasterized grid needs no shift).
"""

import numpy as np


def rasterize(points, x0, y0, cell, ny, nx):
    """[N, 3] (x, y, z) world points -> (H[ny, nx], known[ny, nx]). H = the MAX z per cell (the top
    surface -- what clearance / obstacle height care about); cells with no point are H = 0, known = False.
    Points outside the grid are dropped."""
    p = np.asarray(points, np.float64)
    ci = np.round((p[:, 0] - x0) / cell - 0.5).astype(np.int64)  # col <- x (nearest cell center)
    ri = np.round((p[:, 1] - y0) / cell - 0.5).astype(np.int64)  # row <- y
    inb = (ri >= 0) & (ri < ny) & (ci >= 0) & (ci < nx)
    flat = ri[inb] * nx + ci[inb]
    H = np.full(ny * nx, -np.inf)
    np.maximum.at(H, flat, p[inb, 2])  # max-z per cell
    known = np.isfinite(H)
    H = np.where(known, H, 0.0).reshape(ny, nx).astype(np.float32)
    return H, known.reshape(ny, nx)


def heightmap_to_points(H, x0, y0, cell):
    """Inverse for testing: one point at each cell CENTER (x0 + (c+0.5)*cell, y0 + (r+0.5)*cell).
    rasterize() of these recovers H exactly -- the round-trip that pins the frame convention to the
    engine (Heightmap / GridParams) end to end."""
    ny, nx = H.shape
    rr, cc = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    xs = (x0 + (cc + 0.5) * cell).ravel()
    ys = (y0 + (rr + 0.5) * cell).ravel()
    return np.stack([xs, ys, H.ravel()], 1)
