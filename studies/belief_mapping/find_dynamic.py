"""Which Odin bags have something moving in them? Triage for the belief map's carving check.

A column of the world (0.2 m cells) that holds something tall in only SOME of the frames that
observe it is what a moving object leaves behind; static structure is tall nearly every time it is
seen, and ground never. Per bag this counts such columns, and draws them over the driven path so
the candidates can be confirmed by eye.

    python studies/belief_mapping/find_dynamic.py [bag ...] --out studies/belief_mapping/out0

Heights are taken against each column's lowest return over the whole bag, so a column the robot
only ever saw the top of reads as ground; that under-counts, it does not invent movers.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import warp as wp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from calib import bagio  # noqa: E402

CELL = 0.2  # [m]
TALL = wp.constant(0.3)  # [m] above the column's lowest return
TOP = wp.constant(2.0)  # [m] above it: a ceiling or a tree canopy is not a mover
MAX_RANGE = wp.constant(8.0)  # [m] from the sensor, where the dToF coverage is good


@wp.kernel
def _min_z_kernel(
    pts: wp.array(dtype=wp.vec3f),
    x0: float,
    y0: float,
    cell: float,
    ground: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    p = pts[i]
    c = int(wp.floor((p[0] - x0) / cell))
    r = int(wp.floor((p[1] - y0) / cell))
    if r >= 0 and r < ground.shape[0] and c >= 0 and c < ground.shape[1]:
        wp.atomic_min(ground, r, c, p[2])


@wp.kernel
def _frame_kernel(
    pts: wp.array(dtype=wp.vec3f),
    sensor: wp.vec3f,
    x0: float,
    y0: float,
    cell: float,
    ground: wp.array2d(dtype=wp.float32),
    seen: wp.array2d(dtype=wp.int32),
    tall: wp.array2d(dtype=wp.int32),
):
    i = wp.tid()
    p = pts[i]
    if wp.length(wp.vec2(p[0] - sensor[0], p[1] - sensor[1])) > MAX_RANGE:
        return
    c = int(wp.floor((p[0] - x0) / cell))
    r = int(wp.floor((p[1] - y0) / cell))
    if r < 0 or r >= seen.shape[0] or c < 0 or c >= seen.shape[1]:
        return
    seen[r, c] = 1
    rise = p[2] - ground[r, c]
    if rise > TALL and rise < TOP:
        tall[r, c] = 1


@wp.kernel
def _accumulate_kernel(
    seen: wp.array2d(dtype=wp.int32),
    tall: wp.array2d(dtype=wp.int32),
    n_seen: wp.array2d(dtype=wp.int32),
    n_tall: wp.array2d(dtype=wp.int32),
):
    r, c = wp.tid()
    n_seen[r, c] += seen[r, c]
    n_tall[r, c] += tall[r, c]


def scan_bag(name: str, out: pathlib.Path) -> dict:
    path = bagio.bag_path(name)
    frames = bagio.read_frames(path, bagio.read_odometry(path))
    if not frames:
        return {"bag": name, "frames": 0}
    lo = np.min([f.points.min(axis=0) for f in frames], axis=0)
    hi = np.max([f.points.max(axis=0) for f in frames], axis=0)
    x0, y0 = float(lo[0]), float(lo[1])
    nx = int(np.ceil((hi[0] - x0) / CELL)) + 1
    ny = int(np.ceil((hi[1] - y0) / CELL)) + 1
    ground = wp.full((ny, nx), 1.0e9, dtype=wp.float32)
    dev = [wp.array(f.points, dtype=wp.vec3f) for f in frames]
    for p in dev:
        wp.launch(_min_z_kernel, dim=len(p), inputs=[p, x0, y0, CELL, ground])
    seen = wp.zeros((ny, nx), dtype=wp.int32)
    tall = wp.zeros((ny, nx), dtype=wp.int32)
    n_seen = wp.zeros((ny, nx), dtype=wp.int32)
    n_tall = wp.zeros((ny, nx), dtype=wp.int32)
    for f, p in zip(frames, dev):
        seen.zero_()
        tall.zero_()
        s = wp.vec3f(*[float(v) for v in f.sensor_xyz])
        wp.launch(_frame_kernel, dim=len(p), inputs=[p, s, x0, y0, CELL, ground, seen, tall])
        wp.launch(_accumulate_kernel, dim=(ny, nx), inputs=[seen, tall, n_seen, n_tall])
    ns, nt = n_seen.numpy(), n_tall.numpy()
    ratio = np.where(ns > 0, nt / np.maximum(ns, 1), 0.0)
    # observed often enough to judge, and tall in some but far from all of those frames
    transient = (ns >= 15) & (ratio > 0.05) & (ratio < 0.4)
    static = (ns >= 15) & (ratio >= 0.8)
    _draw(name, transient, static, frames, x0, y0, out)
    return {
        "bag": name,
        "frames": len(frames),
        "transient": int(transient.sum()),
        "static": int(static.sum()),
        "share": float(transient.sum() / max(static.sum(), 1)),
    }


def _draw(name, transient, static, frames, x0, y0, out: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    ext = (x0, x0 + transient.shape[1] * CELL, y0, y0 + transient.shape[0] * CELL)
    img = np.zeros((*transient.shape, 3))
    img[static] = (0.55, 0.55, 0.55)
    img[transient] = (0.85, 0.2, 0.1)
    ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
    path = np.array([f.sensor_xyz[:2] for f in frames])
    ax.plot(path[:, 0], path[:, 1], color=(0.06, 0.54, 0.49), lw=1.5)
    ax.set_title(f"{name}: static grey, transient red, path teal")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out / f"dynamic_{name}.png", dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bags", nargs="*")
    p.add_argument("--out", default="studies/belief_mapping/out0")
    a = p.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bags = a.bags or sorted(
        d.name for d in pathlib.Path("bags").iterdir() if (d / "metadata.yaml").exists()
    )
    wp.init()
    print("| bag | frames | transient columns | static columns | transient / static |")
    print("|---|---|---|---|---|")
    for b in bags:
        try:
            r = scan_bag(b, out)
        except Exception as e:  # a bag without the cloud topic (steps_air) is not an error here
            print(f"| {b} | skipped: {type(e).__name__}: {e} | | | |")
            continue
        if r["frames"] == 0:
            print(f"| {b} | 0 | | | |")
            continue
        print(
            f"| {b} | {r['frames']} | {r['transient']} | {r['static']} | {r['share']:.3f} |",
            flush=True,
        )


if __name__ == "__main__":
    main()
