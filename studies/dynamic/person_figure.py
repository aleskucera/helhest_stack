"""See the ghost: the same walk mapped with the carve off and with it reaching 6 m.

Top row is the shipped behaviour. The person leaves a wall behind it that never goes away, and
by the last frame it is a 7 m barrier across ground the robot can drive on.

  python studies/dynamic/person_figure.py --shot /tmp/person.png
"""

from __future__ import annotations

import argparse

import numpy as np
import warp as wp
from elevation_belief import DriftRates
from elevation_belief import ElevationBelief
from elevation_belief import NoiseModel

SPAN, CELL = 14.0, 0.2
N = int(SPAN / CELL)
DATA = "studies/dynamic/out/person_clear.npz"


def quat_mat(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def replay(d, carve_range, marks):
    pts, cnt = d["points"], d["counts"]
    robot, person, stamp, mount = d["robot"], d["person"], d["stamp"], d["mount"]
    off = np.r_[0, np.cumsum(cnt)]
    bel = ElevationBelief(
        (-SPAN / 2, SPAN / 2, -SPAN / 2, SPAN / 2),
        CELL,
        noise=NoiseModel("linear", a=0.012, b=0.004),
        rates=DriftRates.odin_slam(),
    )
    shots = {}
    for k in range(len(cnt)):
        p = pts[off[k] : off[k + 1]].astype(np.float64)
        R = quat_mat(robot[k, 3:7])
        t = robot[k, 0:3]
        sensor = t + R @ mount
        w = p @ R.T + sensor
        rxy = (float(t[0]), float(t[1]))
        bel.recenter(rxy)
        if k:
            bel.motion_update(float(stamp[k] - stamp[k - 1]), rxy)
        if carve_range > 0:
            bel.carve(w, sensor, max_range=carve_range, margin=0.10, persist=8.0)
        bel.measure_scan(w, sensor)
        if k in marks:
            lay = bel.layers()
            shots[k] = (
                np.where(lay["valid"].numpy() != 0, lay["raw_h"].numpy(), np.nan),
                bel.xmin,
                bel.ymin,
                t[:2].copy(),
                person[: k + 1, :2].copy(),
            )
    return shots


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shot", default="/tmp/person.png")
    ap.add_argument("--carve", type=float, default=6.0)
    a = ap.parse_args()
    wp.init()
    d = np.load(DATA)
    marks = [30, 70, 110, 259]
    rows = [
        ("carve OFF  (the shipped 1.0 m is the same)", replay(d, 0.0, marks)),
        (f"carve {a.carve:.0f} m", replay(d, a.carve, marks)),
    ]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(marks), figsize=(4.4 * len(marks), 9.4))
    for r, (label, shots) in enumerate(rows):
        for c, k in enumerate(marks):
            h, xmin, ymin, rxy, track = shots[k]
            ax = axes[r, c]
            ax.set_facecolor("0.82")
            im = ax.imshow(
                h,
                origin="lower",
                extent=(xmin, xmin + SPAN, ymin, ymin + SPAN),
                cmap="terrain",
                vmin=-0.2,
                vmax=1.6,
            )
            # the track is drawn OFFSET and hollow: the whole point is what is underneath it
            ax.plot(track[:, 0] - 0.55, track[:, 1], "--", color="magenta", lw=1.3, alpha=0.95)
            ax.plot(track[-1, 0], track[-1, 1], "o", ms=13, mfc="none", mec="magenta", mew=2.4)
            ax.plot(*rxy, "o", ms=10, mfc="deepskyblue", mec="k")
            ax.set_xlim(rxy[0] - 1.2, rxy[0] + 5.2)
            ax.set_ylim(rxy[1] - 4.6, rxy[1] + 4.6)
            ax.set_title(f"frame {k}   t = {k/14.5:.1f} s", fontsize=10)
            ax.tick_params(labelsize=7)
            if c == 0:
                ax.set_ylabel(label, fontsize=12)
        fig.colorbar(im, ax=axes[r, :].tolist(), fraction=0.02, label="height above ground [m]")
    fig.suptitle(
        "a person walks across the robot's view at 3 m, then stands still\n"
        "dashed magenta = the walk (drawn 0.55 m to the left so the map under it shows)\n"
        "hollow circle = where the person actually is   blue = robot   grey = never measured",
        fontsize=13,
    )
    fig.savefig(a.shot, dpi=105, bbox_inches="tight")
    print(f"wrote {a.shot}")


if __name__ == "__main__":
    main()
