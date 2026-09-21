"""Every stress world, one panel each: the map the robot built and the line it drove on it.

One run of `sweep.sh` produces one npz per world. Each is drawn at its OWN extent, because the
belief is a rolling window and every run ends somewhere different -- the map is what the robot
had in front of it when it stopped, not a survey of the world. Ground it drove over earlier has
scrolled out of the window and is grey.

  python studies/closed_loop/sweep_figure.py --dir studies/closed_loop/out/sweep
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["gap", "slalom", "pillars", "pocket", "ridge", "bumpy"]


def _panel(ax, d, name: str) -> str:
    x0, y0 = d["bounds"]
    cell = float(d["cell"])
    n = d["height"].shape[0]
    seen = d["seen"]
    trail = d["trail"]
    goal = d["goal"]
    reached = bool(d["reached"])

    ax.set_facecolor("0.85")  # never measured, or outside the window the run ended in
    im = ax.imshow(
        np.where(seen, d["height"], np.nan),
        origin="lower",
        extent=(x0, x0 + n * cell, y0, y0 + n * cell),
        cmap="terrain",
        vmin=-0.1,
        vmax=1.2,
    )
    ax.plot(trail[:, 0], trail[:, 1], "-", color="deepskyblue", lw=2.2)
    ax.plot(*trail[0], "o", color="deepskyblue", ms=7, mec="k", zorder=5)
    ax.plot(*trail[-1], "o", color="white", ms=8, mec="k", zorder=5)
    ax.plot(goal[0], goal[1], "*", color="magenta", ms=18, mec="k", zorder=5)

    # the axes cover the union of the map and the whole drive, so a trail that has scrolled out
    # of the belief window is still visible against the grey
    xs = np.r_[trail[:, 0], goal[0], x0, x0 + n * cell]
    ys = np.r_[trail[:, 1], goal[1], y0, y0 + n * cell]
    pad = 1.0
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)

    end = float(np.hypot(trail[-1, 0] - goal[0], trail[-1, 1] - goal[1]))
    path = float(np.hypot(*np.diff(trail, axis=0).T).sum()) if len(trail) > 1 else 0.0
    verdict = "REACHED" if reached else f"stopped {end:.2f} m short"
    ax.set_title(
        f"{name}   {verdict}\n{len(trail)} frames, {path:.1f} m driven",
        fontsize=10,
        color=("#0b6b2f" if reached else "#a11"),
    )
    return (
        f"  {name:<8s} {'yes' if reached else 'no':>4s} {len(trail):>7d} {path:>8.1f} "
        f"{end:>8.2f} {float(np.ptp(d['body_z'])):>8.3f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default="studies/closed_loop/out/sweep")
    p.add_argument("--shot", default="studies/closed_loop/out/sweep.png")
    a = p.parse_args()

    found = [(w, pathlib.Path(a.dir) / f"{w}.npz") for w in ORDER]
    found = [(w, f) for w, f in found if f.exists()]
    if not found:
        raise SystemExit(f"no world npz in {a.dir} -- run sweep.sh first")

    cols = 3
    rows = (len(found) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 5.2 * rows), squeeze=False)
    print(
        f"  {'world':<8s} {'reach':>4s} {'frames':>7s} {'driven':>8s} {'short':>8s} {'z ptp':>8s}"
    )
    for ax, (w, f) in zip(axes.ravel(), found):
        print(_panel(ax, np.load(f), w))
    for ax in axes.ravel()[len(found) :]:
        ax.axis("off")
    fig.suptitle(
        "the planner on every stress world -- ostrich physics, the Odin dToF, a belief map\n"
        "blue = driven   white = where it stopped   magenta star = goal   "
        "grey = outside the window the run ended in",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(a.shot, dpi=110)
    print(f"\nwrote {a.shot}")


if __name__ == "__main__":
    main()
