"""The carve measured against ground truth, scored by HEIGHT rather than by the valid flag.

`carve` runs immediately before `measure_scan`, so a retired cell is re-initialised from the
ground behind it in the same frame and reads valid again. Watching `valid` therefore sees
nothing. What matters to a planner is whether the GHOST is gone -- whether the cell has come
back down to the ground it is actually standing on.
"""

import numpy as np, warp as wp
from elevation_belief import DriftRates, ElevationBelief, NoiseModel

wp.init()
d = np.load("studies/dynamic/out/person.npz")
pts, cnt, robot, person, stamp, mount = (
    d["points"],
    d["counts"],
    d["robot"],
    d["person"],
    d["stamp"],
    d["mount"],
)
HX, HY = float(d["person_hx"]), float(d["person_hy"])
off = np.r_[0, np.cumsum(cnt)]
SPAN, CELL = 14.0, 0.2
N = int(SPAN / CELL)
GHOST = 0.35  # [m] above local ground = still a ghost as far as a planner is concerned


def qm(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def run(carve_range, persist=8.0, margin=0.10):
    bel = ElevationBelief(
        (-SPAN / 2, SPAN / 2, -SPAN / 2, SPAN / 2),
        CELL,
        noise=NoiseModel("linear", a=0.012, b=0.004),
        rates=DriftRates.odin_slam(),
    )
    occ, clear_frame = {}, {}
    for k in range(len(cnt)):
        p = pts[off[k] : off[k + 1]].astype(np.float64)
        R = qm(robot[k, 3:7])
        t = robot[k, 0:3]
        sensor = t + R @ mount
        w = p @ R.T + sensor
        rxy = (float(t[0]), float(t[1]))
        bel.recenter(rxy)
        if k:
            bel.motion_update(float(stamp[k] - stamp[k - 1]), rxy)
        if carve_range > 0:
            bel.carve(w, sensor, max_range=carve_range, margin=margin, persist=persist)
        bel.measure_scan(w, sensor)
        px, py = person[k, 0], person[k, 1]
        for cx in np.arange(px - HX, px + HX + CELL, CELL):
            for cy in np.arange(py - HY, py + HY + CELL, CELL):
                j = int((cx - bel.xmin) / CELL)
                i = int((cy - bel.ymin) / CELL)
                if 0 <= i < N and 0 <= j < N:
                    occ[(i, j)] = k
        h = np.nan_to_num(bel.layers()["raw_h"].numpy())
        for c, last in occ.items():
            if last < k and c not in clear_frame and h[c] < GHOST:
                clear_frame[c] = k
    return (
        bel,
        occ,
        clear_frame,
        np.nan_to_num(bel.layers()["raw_h"].numpy()),
        bel.layers()["valid"].numpy() != 0,
    )


print(
    f"{'carve':>7s} {'trail':>6s} {'still a ghost at the end':>26s} {'median clear lag':>18s} "
    f"{'ground disturbed':>17s}"
)
for cr in (0.0, 1.0, 3.0, 6.0, 10.0):
    bel, occ, clr, h, valid = run(cr)
    final = {c for c, l in occ.items() if l == len(cnt) - 1}
    trail = [c for c in occ if c not in final]
    ghosts = [c for c in trail if h[c] >= GHOST]
    lags = [clr[c] - occ[c] for c in trail if c in clr]
    gm = valid.copy()
    for c in occ:
        gm[c] = False
    dist = int((h[gm] >= GHOST).sum())
    print(
        f"{cr:>6.1f}m {len(trail):>6d} {len(ghosts):>4d} ({100*len(ghosts)/len(trail):>3.0f}%) "
        f"{'':>12s} {(f'{np.median(lags):.0f} frames = {np.median(lags)/14.5:.2f} s' if lags else '--'):>18s} "
        f"{dist:>10d} of {int(gm.sum())}"
    )

print("\n=== where are the 28 that survive a 10 m carve? ===")
bel, occ, clr, h, valid = run(10.0)
final = {c for c, l in occ.items() if l == len(cnt) - 1}
trail = [c for c in occ if c not in final]
ghosts = [c for c in trail if h[c] >= GHOST]
cleared = [c for c in trail if h[c] < GHOST]
py_end = person[-1, 1]


def ydist(c):  # how far along the walk, relative to where the person STOPPED
    return abs((bel.ymin + (c[0] + 0.5) * CELL) - py_end)


gy = np.array([ydist(c) for c in ghosts])
cy = np.array([ydist(c) for c in cleared])
print(f"  distance from where the person came to rest:")
print(
    f"    surviving ghosts : median {np.median(gy):5.2f} m   min {gy.min():.2f}  max {gy.max():.2f}"
)
print(
    f"    cleared cells    : median {np.median(cy):5.2f} m   min {cy.min():.2f}  max {cy.max():.2f}"
)
print(f"\n  a cell right behind where the person stopped cannot be seen through --")
print(
    f"  the person is standing in the way. {int((gy < 1.0).sum())} of {len(ghosts)} are within 1 m of it."
)
