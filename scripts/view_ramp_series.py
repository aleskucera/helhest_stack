"""View the ramp-series benchmark maps: top-down elevation + cross-section, one angle at a time.

    python scripts/view_ramp_series.py
    python scripts/view_ramp_series.py --angle 25
    python scripts/view_ramp_series.py --angle 25 --out /tmp/ramp25.png

Opens an interactive window; LEFT/RIGHT (or p/n) step through the sweep. With --out it renders
that one map headless to a PNG instead.

Reads whatever scripts/make_ramp_series.py wrote, so run that first. Elevation is shown on one
colour scale across the whole series, so stepping through compares like with like.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np


def load_series(map_dir: pathlib.Path) -> list[dict]:
    """Every map in the directory, in sweep order (the filenames sort that way by construction)."""
    paths = sorted(map_dir.glob("bump_a*.npz"))
    if not paths:
        raise SystemExit(f"no bump_a*.npz in {map_dir} -- run scripts/make_ramp_series.py first")
    maps = []
    for path in paths:
        with np.load(path) as data:
            maps.append({k: data[k] for k in data.files} | {"name": path.name})
    return maps


def draw(fig, axes, maps: list[dict], idx: int, vmax: float):
    """Redraw both panels for maps[idx] and return the elevation image.

    Called on every key press, so it clears as it goes. The colour scale is fixed across the
    series, so the caller can build one colorbar from the first returned image and keep it.
    """
    m = maps[idx]
    heights, cell = m["heights"], float(m["cell"])
    ny, nx = heights.shape
    x0, y0 = float(m["origin"][0]), float(m["origin"][1])
    extent = [x0, x0 + nx * cell, y0, y0 + ny * cell]
    start, goal = m["start"], m["goal"]
    xs = x0 + (np.arange(nx) + 0.5) * cell

    for ax in axes:
        ax.clear()

    top = axes[0]
    # terrain's bottom quarter is ocean blue; vmin = -vmax/3 puts z=0 at the start of its LAND
    # ramp, so flat ground reads as ground rather than as water.
    im = top.imshow(
        heights,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="terrain",
        vmin=-vmax / 3.0,
        vmax=vmax,
    )
    top.plot(start[0], start[1], "o", color="lime", ms=9, mec="black", label="start")
    top.plot(goal[0], goal[1], "X", color="red", ms=11, mec="black", label="goal")
    # the start pose's heading -- the ridge is crossed head-on only if the planner keeps this
    top.annotate(
        "",
        xy=(start[0] + 1.2, start[1]),
        xytext=(start[0], start[1]),
        arrowprops=dict(arrowstyle="->", color="lime", lw=2),
    )
    top.set_xlabel("x [m]")
    top.set_ylabel("y [m]")
    top.legend(loc="upper right", fontsize=8)

    run = float(m["height"]) / np.tan(np.radians(float(m["up_deg"])))
    prof = axes[1]
    prof.plot(xs, heights[0], color="tab:blue", lw=1.5)
    prof.fill_between(xs, 0.0, heights[0], color="tab:blue", alpha=0.25)
    # zoomed to the ridge, not the full extent: at 1:1 a 0.75 m rise across 23 m of map is an
    # invisible sliver, and the shape is the whole point of this panel (the top one gives context)
    prof.set_xlim(-(run + 0.5 * float(m["plateau"]) + 1.0), run + 0.5 * float(m["plateau"]) + 1.0)
    prof.set_aspect("equal")  # 1:1, so the face angle can be judged by eye
    prof.set_xlabel("x [m]")
    prof.set_ylabel("z [m]")
    prof.grid(alpha=0.3)
    fig.suptitle(
        f"{m['name']}   [{idx + 1}/{len(maps)}]   "
        f"face {float(m['up_deg']):.0f} deg (rendered {float(m['measured_deg']):.2f})   "
        f"height {float(m['height']):.2f} m   run {run:.2f} m   "
        f"plateau {float(m['plateau']):.2f} m   grid {nx}x{ny} @ {cell} m"
    )
    fig.canvas.draw_idle()
    return im


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dir", type=pathlib.Path, default=pathlib.Path("/tmp/ramp_series"))
    ap.add_argument("--angle", type=float, default=None, help="open on this face angle [deg]")
    ap.add_argument(
        "--out", type=pathlib.Path, default=None, help="render to PNG instead of a window"
    )
    args = ap.parse_args()

    import matplotlib

    if args.out is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    maps = load_series(args.dir)
    vmax = max(float(m["heights"].max()) for m in maps)
    idx = 0
    if args.angle is not None:
        idx = min(range(len(maps)), key=lambda i: abs(float(maps[i]["up_deg"]) - args.angle))

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), height_ratios=[3, 1])
    im = draw(fig, axes, maps, idx, vmax)
    fig.colorbar(im, ax=axes[0], label="z [m]", fraction=0.025, pad=0.02)

    if args.out is not None:
        fig.savefig(args.out, dpi=120, bbox_inches="tight")
        print(f"saved {args.out} ({maps[idx]['name']})")
        return

    state = {"idx": idx}

    def on_key(event) -> None:
        if event.key in ("right", "n"):
            state["idx"] = (state["idx"] + 1) % len(maps)
        elif event.key in ("left", "p"):
            state["idx"] = (state["idx"] - 1) % len(maps)
        else:
            return
        draw(fig, axes, maps, state["idx"], vmax)

    fig.canvas.mpl_connect("key_press_event", on_key)
    print(f"{len(maps)} maps from {args.dir}. LEFT/RIGHT (or p/n) to step, q to quit.")
    plt.show()


if __name__ == "__main__":
    main()
