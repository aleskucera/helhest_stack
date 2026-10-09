"""Compare arms of a batch.sh run: per world, frames to goal, the closest pass to a wall, degrees
turned while any part of the footprint is within 0.5 m of a wall (exact world geometry), and for
slalom the heading when crossing the third gap (x = 14 m).

  python studies/clearance/analyze.py out_dir base new
Run files are named <world>_<arm>_r<n>.npz.
"""

from __future__ import annotations

import glob
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import worlds  # noqa: E402,F401  (registers the study worlds)
from helhest import dynamics  # noqa: E402
from helhest import worlds as W  # noqa: E402

WORLDS = (
    "slalom",
    "false_door",
    "pocket",
    "cornerL24",
    "corridor24",
    "corridor26",
    "pillars",
    "gap",
)
NEAR_M = 0.5


def summarize(out: str, world: str, arm: str, fp) -> str:
    runs = []
    for f in sorted(glob.glob(f"{out}/{world}_{arm}_r*.npz")):
        reached = "REACHED" in pathlib.Path(f[:-4] + ".log").read_text()
        d = np.load(f)
        t, m = d["trail"], d["hist_meta"]
        yaw = np.interp(np.arange(len(t)), m[:, 0], np.unwrap(m[:, 3]))
        cl = np.array(
            [W.obstacle_clearance(world, t[i, 0], t[i, 1], yaw[i], fp) for i in range(len(t))]
        )
        turned = np.degrees(np.abs(np.diff(yaw))[cl[:-1] < NEAR_M].sum())
        gap = np.degrees(yaw[int(np.argmin(np.abs(t[:, 0] - 14.0)))])
        runs.append((reached, len(t), float(cl.min()), turned, gap))
    if not runs:
        return "-"
    s = (
        f"{min(r[1] for r in runs)}-{max(r[1] for r in runs)} fr, "
        f"{min(r[2] for r in runs):.2f} m, " + "/".join(f"{r[3]:.0f}" for r in runs) + " deg"
    )
    if world == "slalom":
        s += ", gap " + "/".join(f"{r[4]:+.0f}" for r in runs)
    missed = sum(not r[0] for r in runs)
    return s + (f", {missed} NOT REACHED" if missed else "")


def main() -> None:
    out, arms = sys.argv[1], sys.argv[2:]
    fp = W.footprint(dynamics.robot_params(0.1))
    print("| world | " + " | ".join(arms) + " |")
    print("|---|" + "---|" * len(arms))
    for world in WORLDS:
        print(f"| {world} | " + " | ".join(summarize(out, world, a, fp) for a in arms) + " |")


if __name__ == "__main__":
    main()
