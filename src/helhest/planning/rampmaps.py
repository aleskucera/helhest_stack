"""Symmetric full-width ridges for the lattice-planner steepness benchmark.

A benchmark map is flat ground (z = 0) crossed by ONE symmetric trapezoidal ridge spanning the
full Y width: ramp up at `up_deg`, a flat plateau, ramp down at the same angle, both feet back at
z = 0. Spanning the full width is the point -- there is no lateral detour, so the robot must climb
and descend or the goal is unreachable, and V collapses to a clean feasibility verdict.

The series holds the plateau HEIGHT fixed and sweeps only `up_deg`, so the horizontal run
(height / tan(up_deg)) is what changes. That isolates steepness from every other variable and
sweeps the robot's own envelope (RobotParams: max_pitch_up 25 deg, max_pitch_down 15 deg,
max_roll 15 deg): shallow ridges are drivable, the descent blocks before the climb does, and at
the steep end the ridge is a near-vertical 0.75 m step where the settle's residual / clearance
terms rather than tilt are what block it. Crossing at an angle trades pitch for roll, and roll is
the stricter limit, so a correct planner attacks the ridge head-on.

The steepest angle a grid can hold faithfully is max_renderable_deg() -- 75.1 deg at cell 0.1 m
for a 0.75 m rise. bump_ridge() raises rather than quietly returning a shallower ramp than asked
for; reach past it with a finer cell, not by ignoring it.

Terrain here is built in numpy on purpose: this is offline scene construction at a boundary (same
as every builder in helhest.heightmap and helhest.worlds), not a per-frame compute stage. The
elevation crosses to the device once, via
wp.array(np.ascontiguousarray(hm.H, np.float32), dtype=wp.float32, device=device) -- the
ascontiguousarray is required because Heightmap upcasts its grid to float64.

Self-test:
    python -m helhest.planning.rampmaps
"""

from __future__ import annotations

import math

import numpy as np

from ..heightmap import Heightmap

DEFAULT_HEIGHT = 0.75  # [m] plateau height, held constant across the whole series
# [m] flat top. Exceeds the wheelbase (RobotParams.rear_offset = 0.75 m) so the robot can sit
# level on the crest; a shorter plateau runs climb pitch straight into descend pitch and
# confounds the two envelope limits.
DEFAULT_PLATEAU = 1.5
DEFAULT_EXTENT_Y = 8.0  # [m] a full-width ridge admits no detour, so wide Y is only wasted solve
DEFAULT_MARGIN = 2.0  # [m] flat ground between the grid edge and the ridge foot, for start/goal
# [m] Generate directly at the routing resolution. Do NOT feed these through benchmarks._common
# .coarsen: it max-pools, which turns a constant-slope face into a staircase and biases it up by
# half a cell -- fatal when the face angle IS the independent variable.
DEFAULT_CELL = 0.1


def ramp_run(up_deg: float, height: float = DEFAULT_HEIGHT) -> float:
    """Horizontal run [m] of one ramp face rising `height` at `up_deg`."""
    return height / math.tan(math.radians(up_deg))


def ridge_extent(
    up_deg: float,
    height: float = DEFAULT_HEIGHT,
    plateau: float = DEFAULT_PLATEAU,
    margin: float = DEFAULT_MARGIN,
    cell: float = DEFAULT_CELL,
) -> float:
    """Grid extent along X [m] holding this ridge plus `margin` of flat ground each end.

    The series driver calls this with its SHALLOWEST angle and reuses the answer for every map, so
    all maps share one grid and the results stay directly comparable.

    Returned already aligned to a whole number of cells, with two cells of slack: start and goal
    sit `margin` in from the grid EDGE, so without that the rounding could push them past the ramp
    foot, and a cell of clearance each end also keeps their bilinear stencil entirely on flat
    ground.
    """
    span = 2.0 * ramp_run(up_deg, height) + plateau
    return (math.ceil((span + 2.0 * margin) / cell) + 2) * cell


def max_renderable_deg(height: float = DEFAULT_HEIGHT, cell: float = DEFAULT_CELL) -> float:
    """Steepest face [deg] this cell size renders faithfully: the one whose run is two cells.

    Measured, not assumed. Sampling the ideal profile at cell centers reproduces the requested
    angle exactly while the run stays at or above two cells, and degrades below that because the
    cell centers no longer straddle the face at the right places -- at cell 0.1 m a 0.75 m rise
    asked for at 80 deg (1.3 cells) comes out at 77.9 deg, and at 82 deg (1.05 cells) at 75.8 deg.
    """
    return math.degrees(math.atan(height / (2.0 * cell)))


def face_angle_deg(hm: Heightmap) -> float:
    """Steepest rendered rise along +X as an angle [deg] -- what the terrain gradient actually is.

    The benchmark records this next to the angle that was requested, so a map that the grid could
    not hold is visible in the results rather than silently mislabelled.
    """
    return math.degrees(math.atan(float(np.max(np.diff(hm.H[0]))) / hm.cell))


def bump_ridge(
    up_deg: float,
    extent_x: float,
    height: float = DEFAULT_HEIGHT,
    plateau: float = DEFAULT_PLATEAU,
    extent_y: float = DEFAULT_EXTENT_Y,
    cell: float = DEFAULT_CELL,
) -> Heightmap:
    """Flat ground crossed by one symmetric trapezoidal ridge, centred on the world origin.

    `extent_x` is explicit rather than derived per angle: every map in a series must share a grid,
    so the caller sizes it once from the shallowest angle via ridge_extent().

    The grid is built from cell counts directly instead of via heightmap._grid, which appends one
    cell past its requested limit -- fine for a free-floating scene, wrong when the extent has to
    match a GridParams exactly.
    """
    if not 0.0 < up_deg < 90.0:
        raise ValueError(f"up_deg must be in (0, 90) deg, got {up_deg}")
    run = ramp_run(up_deg, height)
    if run < 2.0 * cell:
        raise ValueError(
            f"a {up_deg:.1f} deg face rising {height:.2f} m has a {run * 100:.1f} cm run, under two "
            f"{cell * 100:.0f} cm cells -- the grid would render it shallower than asked. Steepest "
            f"renderable here is {max_renderable_deg(height, cell):.1f} deg; use a finer cell."
        )
    span = 2.0 * run + plateau
    if extent_x < span:
        raise ValueError(
            f"extent_x {extent_x:.2f} m cannot hold a {up_deg:.1f} deg ridge spanning {span:.2f} m"
        )

    nx = int(round(extent_x / cell))
    ny = int(round(extent_y / cell))
    x0, y0 = -0.5 * nx * cell, -0.5 * ny * cell
    xs = x0 + (np.arange(nx) + 0.5) * cell  # cell centers, matching Heightmap's convention
    # np.interp clamps to the end values outside the breakpoints, both 0.0 here, so the flat ground
    # either side needs no separate masking.
    breakpoints = [-run - 0.5 * plateau, -0.5 * plateau, 0.5 * plateau, run + 0.5 * plateau]
    profile = np.interp(xs, breakpoints, [0.0, height, height, 0.0])
    return Heightmap(np.tile(profile, (ny, 1)), (x0, y0), cell)


def start_goal(
    hm: Heightmap, margin: float = DEFAULT_MARGIN
) -> tuple[tuple[float, float, float], tuple[float, float]]:
    """Start pose (x, y, yaw) and goal (x, y) on opposite X edges, centred in Y.

    Measured from the GRID EDGES, not the ridge, so a whole series sharing one extent also shares
    one start and one goal however the ridge in the middle changes shape.
    """
    y_center = hm.y0 + 0.5 * hm.ny * hm.cell
    x_lo = hm.x0 + margin
    x_hi = hm.x0 + hm.nx * hm.cell - margin
    return (x_lo, y_center, 0.0), (x_hi, y_center)


if __name__ == "__main__":
    extent = ridge_extent(5.0)
    hm = bump_ridge(5.0, extent)
    assert hm.H.shape == (hm.ny, hm.nx), "grid shape mismatch"
    assert abs(hm.H.max() - DEFAULT_HEIGHT) < 1e-9, f"peak {hm.H.max()} != {DEFAULT_HEIGHT}"
    assert hm.H.min() == 0.0, "ground is not flat at z=0"
    assert np.allclose(hm.H, hm.H[0]), "ridge is not invariant along Y (not full-width)"
    assert np.allclose(hm.H[0], hm.H[0][::-1]), "profile is not symmetric"

    # Every angle the guard admits must render exactly, right up to the 2-cell limit.
    for deg in (5.0, 15.0, 25.0, 45.0, 70.0, 75.0):
        measured = face_angle_deg(bump_ridge(deg, extent))
        assert abs(measured - deg) < 0.01, f"{deg} deg face rendered as {measured:.2f} deg"

    start, goal = start_goal(hm)
    assert abs(hm.sample(start[0], start[1])) < 1e-9, "start is not on flat ground"
    assert abs(hm.sample(goal[0], goal[1])) < 1e-9, "goal is not on flat ground"

    for bad, why in ((80.0, "unrenderable at this cell"), (0.0, "degenerate angle")):
        try:
            bump_ridge(bad, extent)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} deg should have raised: {why}")
    try:
        bump_ridge(5.0, 4.0)
    except ValueError:
        pass
    else:
        raise AssertionError("an extent too small for the ridge should have raised")

    print(
        f"rampmaps self-test OK: {hm.nx}x{hm.ny} cells at {hm.cell} m "
        f"({extent:.2f} x {DEFAULT_EXTENT_Y:.1f} m), peak {hm.H.max():.3f} m, "
        f"steepest renderable {max_renderable_deg():.1f} deg, start {start}, goal {goal}"
    )
