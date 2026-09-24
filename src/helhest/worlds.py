"""Adversarial test worlds for stress-testing the planner.

Each builder returns a Heightmap; WORLDS maps a name -> (builder, start (x,y,yaw), goal (x,y))
for the stress harness. They target different weaknesses:

  gap      a wall with one narrow gap -> threading clearance (and robustness under slip)
  slalom   alternating walls -> a forced S-weave, repeated tight clearance
  pillars  a field of pillars -> dense local avoidance
  pocket   a U-shaped cul-de-sac opening AWAY from the start -> GLOBAL routing (cost-to-go);
           a greedy Euclidean planner drives into the closed side and stalls
  ridge    a diagonal barrier with one notch -> direction-dependent crossing
  bumpy    rough terrain, some bumps tall enough to high-center -> tilt / settle feasibility
  corridor a mouth aimed at the goal into a corridor capped out of sight, and a side door ->
           commit, discover the dead end, turn round in 2.8 m, back out, take the other way
  false_door  a door aimed at the goal into a closed room, and a side gap -> the same recovery
           with room to turn, so the backtrack is tested without the tight turn

Render them:  python -m helhest.worlds [--out /tmp/worlds.png]
"""

import argparse
import math
from dataclasses import dataclass

import numpy as np

from .heightmap import _grid
from .heightmap import Heightmap

_WALL = 1.0  # impassable obstacle height (drive in -> infeasible settle)


@dataclass(frozen=True)
class Box:
    """One obstacle as a solid box, in world metres.

    The heightmap builders below rasterise obstacles into cells, which is what the
    planner consumes. A physics simulator wants the solid instead: extruding a
    rasterised wall gives one sliver quad per cell across the height discontinuity
    (0.06 m wide, 1.0 m tall), so a wheel finds a handful of badly-conditioned
    contacts on a face that should produce a dense manifold.

    ``OBSTACLES`` below carries the same geometry the builders draw, so a consumer
    that can use solids does not have to recover them from the grid.
    ``test_obstacles_match_heightmaps`` keeps the two in step.

    The box spans ``z`` in ``[0, h]``; ``yaw`` rotates it about its centre.
    """

    cx: float
    cy: float
    hx: float
    hy: float
    h: float = _WALL
    yaw: float = 0.0


def _box(H, XX, YY, cx, cy, hx, hy, h=_WALL):
    H[(np.abs(XX - cx) <= hx) & (np.abs(YY - cy) <= hy)] = h


def gap_world(cell=0.06):
    xlim, ylim = (-2.0, 14.0), (-5.0, 5.0)
    XX, YY = _grid(xlim, ylim, cell)
    H = np.zeros_like(XX)
    H[(np.abs(XX - 6.0) <= 0.2) & (np.abs(YY) >= 0.9)] = _WALL  # wall, 1.8 m gap at |y| < 0.9
    return Heightmap(H, (xlim[0], ylim[0]), cell)


def slalom_world(cell=0.06):
    xlim, ylim = (-2.0, 19.0), (-5.0, 5.0)
    XX, YY = _grid(xlim, ylim, cell)
    H = np.zeros_like(XX)
    H[(np.abs(XX - 4.0) <= 0.2) & (YY <= 1.2)] = _WALL  # open top (lane y > 1.2)
    H[(np.abs(XX - 9.0) <= 0.2) & (YY >= -1.2)] = _WALL  # open bottom
    # open CENTER -> exit aligned to goal
    H[(np.abs(XX - 14.0) <= 0.2) & (np.abs(YY) >= 1.2)] = _WALL
    return Heightmap(H, (xlim[0], ylim[0]), cell)


def pillars_world(cell=0.06):
    xlim, ylim = (-2.0, 16.0), (-5.0, 5.0)
    XX, YY = _grid(xlim, ylim, cell)
    H = np.zeros_like(XX)
    # 0.9 m pillars on a staggered ~3 m grid -> ~2 m clear corridors; last row flanks the center
    # so the robot exits aligned with the goal (a straight run-up, like gap)
    for cx, cy in [
        (4.0, -2.0),
        (4.0, 2.0),
        (7.0, 0.0),
        (7.0, -4.0),
        (7.0, 4.0),
        (10.0, -2.0),
        (10.0, 2.0),
        (13.0, -2.5),
        (13.0, 2.5),
    ]:
        _box(H, XX, YY, cx, cy, 0.45, 0.45)
    return Heightmap(H, (xlim[0], ylim[0]), cell)


def pocket_world(cell=0.06):
    xlim, ylim = (-2.0, 16.0), (-5.0, 5.0)
    XX, YY = _grid(xlim, ylim, cell)
    H = np.zeros_like(XX)
    H[(np.abs(XX - 7.0) <= 0.2) & (np.abs(YY) <= 2.5)] = _WALL  # closed side (faces start)
    H[(np.abs(YY - 2.5) <= 0.2) & (XX >= 7.0) & (XX <= 11.0)] = _WALL  # top
    H[(np.abs(YY + 2.5) <= 0.2) & (XX >= 7.0) & (XX <= 11.0)] = _WALL  # bottom; opening at x > 11
    return Heightmap(H, (xlim[0], ylim[0]), cell)


def ridge_world(cell=0.06):
    xlim, ylim = (-2.0, 14.0), (-5.0, 5.0)
    XX, YY = _grid(xlim, ylim, cell)
    H = np.zeros_like(XX)
    line = YY - 0.3 * (XX - 6.0)  # gentle diagonal ridge across the middle
    H[np.abs(line) <= 0.3] = _WALL
    H[(np.abs(line) <= 0.3) & (np.abs(XX - 6.0) <= 1.0)] = 0.0  # notch near x = 6
    return Heightmap(H, (xlim[0], ylim[0]), cell)


# The two trap worlds. From the start, the opening aimed at the goal is the obvious way and its
# dead end is 18 m out -- past the harness's 10 m sensing -- so the robot can only find it by
# driving in. Both are 22 m to the goal and share an extent, so they are directly comparable.
def corridor_world(cell=0.06):
    xlim, ylim = (-2.0, 24.0), (-9.0, 9.0)
    XX, YY = _grid(xlim, ylim, cell)
    H = np.zeros_like(XX)
    _box(H, XX, YY, 5.0, -5.2, 0.2, 3.8)  # front wall; mouth at |y| < 1.4 ...
    _box(H, XX, YY, 5.0, 3.3, 0.2, 1.9)
    _box(H, XX, YY, 5.0, 8.0, 0.2, 1.0)  # ... and a 1.8 m side door at y 5.2..7.0
    _box(H, XX, YY, 11.5, 1.6, 6.7, 0.2)  # the corridor, 2.8 m clear
    _box(H, XX, YY, 11.5, -1.6, 6.7, 0.2)
    _box(H, XX, YY, 18.0, 0.0, 0.2, 1.8)  # capped 18 m out
    return Heightmap(H, (xlim[0], ylim[0]), cell)


def false_door_world(cell=0.06):
    xlim, ylim = (-2.0, 24.0), (-9.0, 9.0)
    XX, YY = _grid(xlim, ylim, cell)
    H = np.zeros_like(XX)
    _box(H, XX, YY, 5.0, -5.0, 0.2, 4.0)  # front wall; a 2.0 m door at |y| < 1.0 ...
    _box(H, XX, YY, 5.0, 3.3, 0.2, 2.3)
    _box(H, XX, YY, 5.0, 8.2, 0.2, 0.8)  # ... and a 1.8 m side gap at y 5.6..7.4
    _box(H, XX, YY, 11.5, 5.0, 6.7, 0.2)  # a room behind the door, 12.6 x 9.6 m clear,
    _box(H, XX, YY, 11.5, -5.0, 6.7, 0.2)
    _box(H, XX, YY, 18.0, 0.0, 0.2, 5.2)  # closed at the back
    return Heightmap(H, (xlim[0], ylim[0]), cell)


def behind_world(cell=0.06):
    """Open ground, the goal 6 m straight BEHIND the start. The ground behind is unmeasured at the
    start -- the sensor looks ahead -- so a reverse that respects the map-knowledge gate must turn
    and drive forward; one that does not backs blind."""
    xlim, ylim = (-9.0, 5.0), (-5.0, 5.0)
    XX, YY = _grid(xlim, ylim, cell)
    return Heightmap(np.zeros_like(XX), (xlim[0], ylim[0]), cell)


def blind_wall_world(cell=0.06):
    """`behind`, with a wall 1.4 m behind the start that the robot has never seen (it is outside
    the sensor's view until the robot turns). Backing blind hits it; turning finds it and goes
    round its end."""
    xlim, ylim = (-9.0, 5.0), (-5.0, 5.0)
    XX, YY = _grid(xlim, ylim, cell)
    H = np.zeros_like(XX)
    _box(H, XX, YY, -1.4, 0.0, 0.2, 3.0)  # spans |y| < 3, open past either end
    return Heightmap(H, (xlim[0], ylim[0]), cell)


def cliff_corridor_world(cell=0.06):
    """The 2.8 m corridor with its south wall replaced by a 1 m DROP. A turn or a reverse that
    strays south goes over the edge; the settle sees it as a wheel with no ground under it."""
    xlim, ylim = (-2.0, 24.0), (-9.0, 9.0)
    XX, YY = _grid(xlim, ylim, cell)
    H = np.zeros_like(XX)
    _box(H, XX, YY, 11.5, -5.2, 6.7, 3.8, h=-1.0)  # the south wall is a cliff, y < -1.4 ...
    _box(H, XX, YY, 5.0, -5.2, 0.2, 3.8)  # ... stamped first, so the walls keep their cells
    _box(H, XX, YY, 5.0, 3.3, 0.2, 1.9)
    _box(H, XX, YY, 5.0, 8.0, 0.2, 1.0)
    _box(H, XX, YY, 11.5, 1.6, 6.7, 0.2)  # the north wall stays
    _box(H, XX, YY, 18.0, 0.0, 0.2, 1.8)
    return Heightmap(H, (xlim[0], ylim[0]), cell)


def narrow_corridor_world(cell=0.06):
    """A 2.2 m dead end, 6 m deep: narrower than the robot can spin in even from rest, so the only
    way out is 6 m of reverse -- four times the gate's strip, over ground measured on the way in."""
    xlim, ylim = (-2.0, 20.0), (-9.0, 9.0)
    XX, YY = _grid(xlim, ylim, cell)
    H = np.zeros_like(XX)
    _box(H, XX, YY, 5.0, -5.05, 0.2, 3.95)  # front wall; mouth at |y| < 1.1 ...
    _box(H, XX, YY, 5.0, 3.15, 0.2, 2.05)
    _box(H, XX, YY, 5.0, 8.0, 0.2, 1.0)  # ... and the 1.8 m side door at y 5.2..7.0
    _box(H, XX, YY, 8.0, 1.3, 3.2, 0.2)  # the dead end, 2.2 m clear, capped at x = 11
    _box(H, XX, YY, 8.0, -1.3, 3.2, 0.2)
    _box(H, XX, YY, 11.0, 0.0, 0.2, 1.5)
    return Heightmap(H, (xlim[0], ylim[0]), cell)


def bumpy_world(cell=0.06, seed=0):
    xlim, ylim = (-2.0, 16.0), (-5.0, 5.0)
    XX, YY = _grid(xlim, ylim, cell)
    H = np.zeros_like(XX)
    rng = np.random.default_rng(seed)
    # mounds of mixed height -- a few tall enough to be real obstacles to route AROUND, many gentle
    # -- with flat ground still left between them, so the flat path is harder to find but exists
    for _ in range(22):
        cx, cy = rng.uniform(1.5, 12.5), rng.uniform(-4.0, 4.0)
        amp, wid = rng.uniform(0.25, 0.8), rng.uniform(0.35, 0.7)
        H += amp * np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2 * wid**2))
    return Heightmap(H, (xlim[0], ylim[0]), cell)


# The same obstacles the builders above rasterise, as solids. Kept beside the
# builders so the two are edited together; test_obstacles_match_heightmaps
# rasterises these and diffs against the builder output.
#
# bumpy has none -- it is summed Gaussian mounds, genuinely continuous terrain
# that a heightmap represents correctly and a box cannot.
OBSTACLES: dict[str, tuple[Box, ...]] = {
    # one slab with a 1.8 m gap at |y| < 0.9, split into the two halves
    "gap": (
        Box(6.0, 3.2, 0.2, 2.3),
        Box(6.0, -3.2, 0.2, 2.3),
    ),
    # three partial-width slabs forcing an S-weave
    "slalom": (
        Box(4.0, -2.15, 0.2, 3.35),  # open above y = 1.2
        Box(9.0, 2.15, 0.2, 3.35),  # open below y = -1.2
        Box(14.0, 3.35, 0.2, 2.15),  # open centre
        Box(14.0, -3.35, 0.2, 2.15),
    ),
    "pillars": tuple(
        Box(cx, cy, 0.45, 0.45)
        for cx, cy in (
            (4.0, -2.0),
            (4.0, 2.0),
            (7.0, 0.0),
            (7.0, -4.0),
            (7.0, 4.0),
            (10.0, -2.0),
            (10.0, 2.0),
            (13.0, -2.5),
            (13.0, 2.5),
        )
    ),
    # U opening away from the start
    "pocket": (
        Box(7.0, 0.0, 0.2, 2.5),  # closed side, faces the start
        Box(9.0, 2.5, 2.0, 0.2),  # top
        Box(9.0, -2.5, 2.0, 0.2),  # bottom
    ),
    # diagonal band with a notch near x = 6, as two rotated slabs. The notch is a
    # vertical cut while a box ends perpendicular to its axis, so the two ends are
    # off by the ridge angle -- approximate here, unlike the others.
    "ridge": (
        Box(
            1.5,
            0.3 * (1.5 - 6.0),
            3.5 / math.cos(0.2914567944778671),
            0.2873478855663454,
            yaw=0.2914567944778671,
        ),
        Box(
            10.5,
            0.3 * (10.5 - 6.0),
            3.5 / math.cos(0.2914567944778671),
            0.2873478855663454,
            yaw=0.2914567944778671,
        ),
    ),
    "bumpy": (),
    "behind": (),
    "blind_wall": (Box(-1.4, 0.0, 0.2, 3.0),),
    # the cliff is terrain, not a solid: the heightfield carries it
    "cliff_corridor": (
        Box(5.0, -5.2, 0.2, 3.8),
        Box(5.0, 3.3, 0.2, 1.9),
        Box(5.0, 8.0, 0.2, 1.0),
        Box(11.5, 1.6, 6.7, 0.2),
        Box(18.0, 0.0, 0.2, 1.8),
    ),
    "narrow_corridor": (
        Box(5.0, -5.05, 0.2, 3.95),
        Box(5.0, 3.15, 0.2, 2.05),
        Box(5.0, 8.0, 0.2, 1.0),
        Box(8.0, 1.3, 3.2, 0.2),
        Box(8.0, -1.3, 3.2, 0.2),
        Box(11.0, 0.0, 0.2, 1.5),
    ),
    "corridor": (
        Box(5.0, -5.2, 0.2, 3.8),
        Box(5.0, 3.3, 0.2, 1.9),
        Box(5.0, 8.0, 0.2, 1.0),
        Box(11.5, 1.6, 6.7, 0.2),
        Box(11.5, -1.6, 6.7, 0.2),
        Box(18.0, 0.0, 0.2, 1.8),
    ),
    "false_door": (
        Box(5.0, -5.0, 0.2, 4.0),
        Box(5.0, 3.3, 0.2, 2.3),
        Box(5.0, 8.2, 0.2, 0.8),
        Box(11.5, 5.0, 6.7, 0.2),
        Box(11.5, -5.0, 6.7, 0.2),
        Box(18.0, 0.0, 0.2, 5.2),
    ),
}


def stamp(boxes, XX, YY):
    """Stamp `boxes` onto the sample points `XX`, `YY` -- the inverse of reading OBSTACLES."""
    H = np.zeros_like(XX)
    for b in boxes:
        dx, dy = XX - b.cx, YY - b.cy
        if b.yaw:
            c, s_ = np.cos(-b.yaw), np.sin(-b.yaw)
            dx, dy = c * dx - s_ * dy, s_ * dx + c * dy
        H[(np.abs(dx) <= b.hx) & (np.abs(dy) <= b.hy)] = b.h
    return H


def rasterise(boxes, xlim, ylim, cell):
    """Stamp `boxes` into a fresh height grid over `xlim`/`ylim`."""
    return stamp(boxes, *_grid(xlim, ylim, cell))


def footprint(r) -> tuple[float, float, float, float]:
    """(x_min, x_max, y_min, y_max) of the robot's TRUE extent in its base frame, from RobotParams.

    Base origin at the front axle; the rear wheel sits `rear_offset` behind it; wheels reach
    `wheel_radius` fore and aft of their axles and `half_track + wheel_width / 2` to each side. No
    margin: this is for asking whether the robot TOUCHED something, and padding it would turn a
    graze into a pass.
    """
    side = r.half_track + 0.5 * r.wheel_width
    return (-(r.rear_offset + r.wheel_radius), r.wheel_radius, -side, side)


def _rect_sdf(p: np.ndarray, hx: float, hy: float) -> np.ndarray:
    """Signed distance from points p [N, 2] to a centred axis-aligned rectangle; < 0 inside."""
    q = np.abs(p) - np.array([hx, hy])
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0)


def obstacle_clearance(
    world: str, x: float, y: float, yaw: float, fp: tuple[float, float, float, float]
) -> float:
    """Signed distance [m] from the robot's footprint to the nearest solid obstacle; < 0 = overlap.

    Exact geometry from OBSTACLES, not a height threshold on the raster, so it is the physical
    question -- did the chassis reach a wall -- rather than a proxy for it. Inf for a world with no
    solids (bumpy: continuous mounds, where the hazard is tilt, not contact).

    Two convex rectangles: the robot's boundary is sampled against each box's SDF, and each box's
    corners against the robot's -- the second catches a box corner poking into the robot's side,
    which boundary samples alone can step over. Resolution is the boundary spacing, ~5 cm.
    """
    boxes = OBSTACLES.get(world, ())
    if not boxes:
        return float("inf")
    x0, x1, y0, y1 = fp
    n = 32
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    edge = np.concatenate(
        [
            np.stack([xs, np.full(n, y0)], 1),
            np.stack([xs, np.full(n, y1)], 1),
            np.stack([np.full(n, x0), ys], 1),
            np.stack([np.full(n, x1), ys], 1),
        ]
    )
    c, s = np.cos(yaw), np.sin(yaw)
    world_pts = edge @ np.array([[c, s], [-s, c]]) + np.array([x, y])  # base -> world
    centre = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
    half = (0.5 * (x1 - x0), 0.5 * (y1 - y0))
    best = float("inf")
    for b in boxes:
        cb, sb = np.cos(b.yaw), np.sin(b.yaw)
        d = world_pts - np.array([b.cx, b.cy])
        local = np.stack([d[:, 0] * cb + d[:, 1] * sb, -d[:, 0] * sb + d[:, 1] * cb], 1)
        best = min(best, float(_rect_sdf(local, b.hx, b.hy).min()))
        # the box's corners, into the robot's frame
        corners = np.array([[sx * b.hx, sy * b.hy] for sx in (-1, 1) for sy in (-1, 1)])
        cw = corners @ np.array([[cb, sb], [-sb, cb]]) + np.array([b.cx, b.cy])  # box -> world
        dw = cw - np.array([x, y])
        cr = np.stack([dw[:, 0] * c + dw[:, 1] * s, -dw[:, 0] * s + dw[:, 1] * c], 1) - centre
        best = min(best, float(_rect_sdf(cr, *half).min()))
    return best


WORLDS = {
    "gap": (gap_world, (0.0, 0.0, 0.0), (11.0, 0.0)),
    "slalom": (slalom_world, (0.0, 0.0, 0.0), (17.0, 0.0)),
    "pillars": (pillars_world, (0.0, 0.0, 0.0), (15.0, 0.0)),
    "pocket": (pocket_world, (0.0, 0.0, 0.0), (9.0, 0.0)),
    "ridge": (ridge_world, (0.0, -4.0, 0.0), (9.0, 2.5)),
    "bumpy": (bumpy_world, (0.0, 0.0, 0.0), (14.0, 0.0)),
    "corridor": (corridor_world, (0.0, 0.0, 0.0), (22.0, 0.0)),
    "false_door": (false_door_world, (0.0, 0.0, 0.0), (22.0, 0.0)),
    # reverse: the goal behind, a wall behind that was never seen, an edge beside the turn, and a
    # dead end too narrow to spin in
    "behind": (behind_world, (0.0, 0.0, 0.0), (-6.0, 0.0)),
    "blind_wall": (blind_wall_world, (0.0, 0.0, 0.0), (-6.0, 0.0)),
    "cliff_corridor": (cliff_corridor_world, (0.0, 0.0, 0.0), (22.0, 0.0)),
    "narrow_corridor": (narrow_corridor_world, (0.0, 0.0, 0.0), (16.0, 0.0)),
}


def matching_friction(hm, value=0.8):
    """Uniform-friction Heightmap matching a scene's grid exactly (avoids the dim mismatch)."""
    return Heightmap(np.full((hm.ny, hm.nx), value, np.float32), (hm.x0, hm.y0), hm.cell)


def _plot_all(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(21, 8))
    for ax, (name, (builder, start, goal)) in zip(axes.ravel(), WORLDS.items()):
        hm = builder()
        ext = [hm.x0, hm.x0 + hm.nx * hm.cell, hm.y0, hm.y0 + hm.ny * hm.cell]
        ax.imshow(hm.H, origin="lower", extent=ext, cmap="terrain", vmin=0.0, vmax=1.0)
        ax.plot(start[0], start[1], "o", color="white", mec="k", ms=9)
        ax.plot(goal[0], goal[1], "*", color="red", ms=16)
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
    fig.suptitle("Stress-test worlds (white = start, red star = goal, bright = obstacle)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/worlds.png")
    args = ap.parse_args()
    _plot_all(args.out)


if __name__ == "__main__":
    main()
