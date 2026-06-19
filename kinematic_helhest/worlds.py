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

Render them:  python -m kinematic_helhest.worlds [--out /tmp/worlds.png]
"""
import argparse

import numpy as np

from .heightmap import Heightmap
from .heightmap import _grid

_WALL = 1.0  # impassable obstacle height (drive in -> infeasible settle)


def _box(H, XX, YY, cx, cy, hx, hy, h=_WALL):
    H[(np.abs(XX - cx) <= hx) & (np.abs(YY - cy) <= hy)] = h


def gap_world(cell=0.06):
    xlim, ylim = (-1.0, 8.0), (-3.5, 3.5)
    XX, YY = _grid(xlim, ylim, cell)
    H = np.zeros_like(XX)
    H[(np.abs(XX - 3.5) <= 0.15) & (np.abs(YY) >= 0.55)] = _WALL  # wall, gap |y| < 0.55 (1.1 m)
    return Heightmap(H, (xlim[0], ylim[0]), cell)


def slalom_world(cell=0.06):
    xlim, ylim = (-1.0, 9.0), (-3.0, 3.0)
    XX, YY = _grid(xlim, ylim, cell)
    H = np.zeros_like(XX)
    H[(np.abs(XX - 2.0) <= 0.15) & (YY <= 0.8)] = _WALL    # gap on top
    H[(np.abs(XX - 4.5) <= 0.15) & (YY >= -0.8)] = _WALL   # gap on bottom
    H[(np.abs(XX - 7.0) <= 0.15) & (YY <= 0.8)] = _WALL    # gap on top
    return Heightmap(H, (xlim[0], ylim[0]), cell)


def pillars_world(cell=0.06):
    xlim, ylim = (-1.0, 9.0), (-3.0, 3.0)
    XX, YY = _grid(xlim, ylim, cell)
    H = np.zeros_like(XX)
    for cx, cy in [(2.0, 0.8), (2.1, -1.6), (3.4, -0.2), (3.6, 1.9), (5.0, -1.1),
                   (5.1, 1.0), (6.6, 0.1), (6.7, -1.9), (6.8, 2.0)]:
        _box(H, XX, YY, cx, cy, 0.3, 0.3)
    return Heightmap(H, (xlim[0], ylim[0]), cell)


def pocket_world(cell=0.06):
    xlim, ylim = (-1.0, 9.0), (-3.5, 3.5)
    XX, YY = _grid(xlim, ylim, cell)
    H = np.zeros_like(XX)
    H[(np.abs(XX - 4.0) <= 0.15) & (np.abs(YY) <= 1.6)] = _WALL         # closed side (faces the start)
    H[(np.abs(YY - 1.6) <= 0.15) & (XX >= 4.0) & (XX <= 6.2)] = _WALL    # top
    H[(np.abs(YY + 1.6) <= 0.15) & (XX >= 4.0) & (XX <= 6.2)] = _WALL    # bottom; opening at x > 6.2
    return Heightmap(H, (xlim[0], ylim[0]), cell)


def ridge_world(cell=0.06):
    xlim, ylim = (-1.0, 9.0), (-3.5, 3.5)
    XX, YY = _grid(xlim, ylim, cell)
    H = np.zeros_like(XX)
    line = YY - 0.55 * (XX - 2.0)                  # diagonal ridge along y = 0.55 (x - 2)
    H[np.abs(line) <= 0.25] = _WALL
    H[(np.abs(line) <= 0.25) & (np.abs(XX - 4.0) <= 0.6)] = 0.0   # notch
    return Heightmap(H, (xlim[0], ylim[0]), cell)


def bumpy_world(cell=0.06, seed=0):
    xlim, ylim = (-1.0, 9.0), (-3.0, 3.0)
    XX, YY = _grid(xlim, ylim, cell)
    H = np.zeros_like(XX)
    rng = np.random.default_rng(seed)
    for _ in range(40):
        cx, cy = rng.uniform(0.5, 8.0), rng.uniform(-2.5, 2.5)
        amp, wid = rng.uniform(0.1, 0.9), rng.uniform(0.25, 0.6)
        H += amp * np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2 * wid ** 2))
    return Heightmap(H, (xlim[0], ylim[0]), cell)


WORLDS = {
    "gap":     (gap_world,     (0.0, 0.0, 0.0),  (6.0, 0.0)),
    "slalom":  (slalom_world,  (0.0, 0.0, 0.0),  (8.5, 0.0)),
    "pillars": (pillars_world, (0.0, 0.0, 0.0),  (8.0, 0.0)),
    "pocket":  (pocket_world,  (0.0, 0.0, 0.0),  (5.0, 0.0)),
    "ridge":   (ridge_world,   (0.0, -2.5, 0.0), (7.0, 2.8)),
    "bumpy":   (bumpy_world,   (0.0, 0.0, 0.0),  (8.0, 0.0)),
}


def matching_friction(hm, value=0.8):
    """Uniform-friction Heightmap matching a scene's grid exactly (avoids the dim mismatch)."""
    return Heightmap(np.full((hm.ny, hm.nx), value, np.float32), (hm.x0, hm.y0), hm.cell)


def _plot_all(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (name, (builder, start, goal)) in zip(axes.ravel(), WORLDS.items()):
        hm = builder()
        ext = [hm.x0, hm.x0 + hm.nx * hm.cell, hm.y0, hm.y0 + hm.ny * hm.cell]
        ax.imshow(hm.H, origin="lower", extent=ext, cmap="terrain", vmin=0.0, vmax=1.0)
        ax.plot(start[0], start[1], "o", color="white", mec="k", ms=9)
        ax.plot(goal[0], goal[1], "*", color="red", ms=16)
        ax.set_title(name); ax.set_aspect("equal")
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    fig.suptitle("Stress-test worlds (white = start, red star = goal, bright = obstacle)")
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/worlds.png")
    args = ap.parse_args()
    _plot_all(args.out)


if __name__ == "__main__":
    main()
