"""Straight dead-straight corridors for the lattice-planner pivot-requirement benchmark.

A benchmark map is flat ground (z = 0) run the length of the grid, flanked by two parallel
full-length walls `width` apart. The corridor admits no lateral detour -- crossing to the other
side of a wall is not an option -- and the start pose faces AWAY from the goal (`yaw = pi`, goal
sits in +X, start faces -X), so a forward-only robot can only re-aim by driving a full U-turn
inside the corridor. Sweeping only `width` (everything else held fixed) isolates corridor width as
the variable and crosses two feasibility boundaries in turn:

  * wide enough for a forward-arc U-turn (turn_radius-limited primitives, see lattice_solver.py)
    to fit inside the walls -- no pivot needed, forward-only V is finite;
  * too narrow for that U-turn but still wide enough for the robot's own footprint to rotate in
    place -- V is +inf without pivot primitives (`pivot_cost = 0`), finite with them;
  * too narrow to even fit the robot's footprint to pivot -- infeasible outright; bump_corridor()
    raises rather than silently building an unpivotable map (min_pivotable_width()).

This generalizes the fixed corridor in tests/planning/test_pivot.py (width 2.2 m, wall height
2.0 m) into a bump_ridge()-style parametrized family: 2.2 m is exactly the validated middle regime
above, kept as an anchor in DEFAULT_WIDTHS.

Wall height 2.0 m matches test_pivot.py and the convention in worlds.py (_WALL) / heightmap.py
(demo_terrain's wall): tall enough that the settle's residual/envelope check (CostToGo's
_feasibility_kernel) blocks it at every heading, not just an over-tilt at some -- a heading-
independent hard wall, the way a real corridor wall is.

Terrain here is built in numpy on purpose: this is offline scene construction at a boundary (same
as every builder in helhest.heightmap and helhest.worlds and rampmaps.py), not a per-frame compute
stage.

Self-test:
    python -m helhest.planning.turnmaps
"""

from __future__ import annotations

import math

import numpy as np

from ..heightmap import Heightmap

DEFAULT_WALL_HEIGHT = 2.0  # [m] tall enough to hard-block the settle at every heading
DEFAULT_LENGTH = 8.0  # [m] corridor length (X), matches tests/planning/test_pivot.py's extent_x
DEFAULT_MARGIN = 1.0  # [m] flat run-up along X from the grid edge to start/goal
DEFAULT_SIDE_MARGIN = 0.4  # [m] extra WALL thickness past the clear width, out to the grid edge
DEFAULT_CELL = 0.1  # [m] Generate directly at the routing resolution, same as rampmaps.py

# Literal RobotParams default (engine/robot.py), kept as a plain float rather than importing
# RobotParams so this module stays engine-independent, like rampmaps.py and lattice_solver.py.
DEFAULT_REAR_OFFSET = 0.75

# [m] A handful of representative widths spanning the two feasibility boundaries described above
# (pivot-only / forward-U-turn-fits), anchored on the 2.2 m point validated by
# tests/planning/test_pivot.py. All safely above min_pivotable_width() (1.5 m) -- the dedicated
# too-narrow-to-pivot case is exercised separately, in the self-test's guard check below. Not a
# fine sweep like rampmaps' 5 deg steps -- no series driver consumes this yet, so keep it small.
DEFAULT_WIDTHS = (1.6, 2.2, 3.0, 4.0)


def min_pivotable_width(rear_offset: float = DEFAULT_REAR_OFFSET) -> float:
    """Narrowest corridor [m] the robot's own footprint can rotate in place inside.

    Below this the robot does not fit to pivot at all, regardless of pivot_cost -- a hard
    geometric floor, not a planner tuning knob. Analogous to rampmaps.max_renderable_deg.

    Driven by LENGTH, not track width: engine/robot.py's Robot.build() places every pose at the
    front axle (wheel_center = pose + R @ wheel_pos, with the front wheels AT wheel_pos local
    x=0), and the rear wheel sits `rear_offset` behind that -- by far the vehicle's longest reach
    from the point the lattice's pivot primitive actually rotates about. Mid-pivot, at a broadside
    heading, that reach swings straight across the corridor, so it (not half_track) sets the floor.
    This intentionally excludes wheel/chassis radius padding: the exact worst-case circle
    (2 * (rear_offset + wheel_radius) = 2.20 m for the default robot) lands EXACTLY on the 2.2 m
    corridor tests/planning/test_pivot.py already validates as pivotable -- i.e. the settle's real
    tolerance clears a bit inside that exact circle, so using it here would wrongly flag a corridor
    already proven to work. This coarser, wheel-center-only floor stays safely under that, catching
    only the clearly-too-narrow cases; the settle (a pivot_cost > 0 CostToGo solve) is the
    authoritative test for anything closer to the edge."""
    return 2.0 * rear_offset


def corridor_extent(
    width: float, side_margin: float = DEFAULT_SIDE_MARGIN, cell: float = DEFAULT_CELL
) -> float:
    """Grid extent along Y [m] holding this corridor width plus `side_margin` of WALL each side
    (not flat ground -- bump_corridor puts wall_height everywhere past width/2, all the way to
    the grid edge; side_margin only keeps that wall off the exact boundary row).

    Width is the swept axis here (Y), the way ramp face angle drove ridge_extent's X-extent in
    rampmaps.py. Cell-aligned with two cells of slack, same rounding pattern as ridge_extent."""
    return (math.ceil((width + 2.0 * side_margin) / cell) + 2) * cell


def bump_corridor(
    width: float,
    length: float = DEFAULT_LENGTH,
    extent_y: float | None = None,
    wall_height: float = DEFAULT_WALL_HEIGHT,
    cell: float = DEFAULT_CELL,
) -> Heightmap:
    """Flat ground run the length of the grid, flanked by two parallel full-length walls `width`
    apart, centred on the world origin.

    `extent_y` is explicit rather than always derived from `width`, the way rampmaps.bump_ridge
    takes `extent_x` explicit: a caller sweeping a whole width series can pin one shared grid
    (sized via corridor_extent() from the WIDEST width in the sweep, which needs the most Y-room),
    keeping V and path length directly comparable across the series. None derives it from this
    map's own width alone.
    """
    floor = min_pivotable_width()
    if width <= floor:
        raise ValueError(
            f"a {width:.2f} m corridor cannot fit the robot's footprint to pivot -- narrowest "
            f"pivotable width is {floor:.2f} m (2 * rear_offset); widen the corridor rather than "
            f"asking bump_corridor() to build an unpivotable one."
        )
    if extent_y is None:
        extent_y = corridor_extent(width, cell=cell)
    elif extent_y < width + 2.0 * cell:
        raise ValueError(
            f"extent_y {extent_y:.2f} m cannot hold a {width:.2f} m wide corridor "
            f"(needs at least {width + 2.0 * cell:.2f} m for the walls to render)"
        )

    nx = int(round(length / cell))
    ny = int(round(extent_y / cell))
    x0, y0 = -0.5 * nx * cell, -0.5 * ny * cell
    ys = y0 + (np.arange(ny) + 0.5) * cell  # cell centers, matching Heightmap's convention
    H = np.zeros((ny, nx), np.float64)
    H[np.abs(ys) > 0.5 * width, :] = wall_height  # two parallel walls, full corridor length
    return Heightmap(H, (x0, y0), cell)


def start_goal(
    hm: Heightmap, margin: float = DEFAULT_MARGIN
) -> tuple[tuple[float, float, float], tuple[float, float]]:
    """Start pose (x, y, yaw) and goal (x, y) on opposite X edges, centred in Y.

    Start faces AWAY from the goal (yaw = pi, goal sits in +X) -- this is what forces the pivot:
    a forward-only robot cannot re-aim onto the goal without a full U-turn, which the corridor is
    sized (via width) to admit or deny. Measured from the GRID EDGES, not the walls, so a whole
    width series sharing one extent_y also shares one start and one goal however narrow the
    corridor in the middle gets -- same convention as rampmaps.start_goal.
    """
    y_center = hm.y0 + 0.5 * hm.ny * hm.cell
    x_lo = hm.x0 + margin
    x_hi = hm.x0 + hm.nx * hm.cell - margin
    return (x_lo, y_center, math.pi), (x_hi, y_center)


if __name__ == "__main__":
    floor = min_pivotable_width()
    assert abs(floor - 2.0 * DEFAULT_REAR_OFFSET) < 1e-9

    for width in DEFAULT_WIDTHS:
        hm = bump_corridor(width)
        assert hm.H.shape == (hm.ny, hm.nx), "grid shape mismatch"

        ys = hm.y0 + (np.arange(hm.ny) + 0.5) * hm.cell
        wall_rows = np.abs(ys) > 0.5 * width
        assert np.all(hm.H[wall_rows, :] == DEFAULT_WALL_HEIGHT), "wall rows not at wall_height"
        assert np.all(hm.H[~wall_rows, :] == 0.0), "corridor floor is not flat at z=0"
        assert np.allclose(hm.H, hm.H[::-1, :]), f"corridor at width={width} is not Y-symmetric"

        start, goal = start_goal(hm)
        assert abs(hm.sample(start[0], start[1])) < 1e-9, "start is not on flat ground"
        assert abs(hm.sample(goal[0], goal[1])) < 1e-9, "goal is not on flat ground"
        assert start[2] == math.pi, "start must face away from the goal"
        assert goal[0] > start[0], "goal must be ahead of start in +X"

    for bad in (floor, 0.5 * floor):
        try:
            bump_corridor(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"width {bad} deg should have raised: at/under the pivot floor")

    try:
        bump_corridor(2.2, extent_y=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("an extent_y too small for the corridor should have raised")

    hm = bump_corridor(2.2, length=8.0, extent_y=4.4)
    print(
        f"turnmaps self-test OK: {len(DEFAULT_WIDTHS)} widths in {DEFAULT_WIDTHS}, "
        f"pivot floor {floor:.2f} m, anchor corridor {hm.nx}x{hm.ny} cells at {hm.cell} m "
        f"({hm.nx * hm.cell:.1f} x {hm.ny * hm.cell:.1f} m)"
    )
