"""What the robot believed, and where it drove on it.

Four panels from one run of `drive_sim.py --out`: the height the belief settled on, which cells
were ever measured, what the lattice refused, and the cost-to-go the MPPI actually followed. The
trail is the same in all four, so a detour can be read against the thing that caused it.

  python studies/closed_loop/drive_figure.py --data studies/closed_loop/out/drive.npz
"""

from __future__ import annotations

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def render(d, out: str) -> None:
    # Three grids share a centre but not a size: the belief (height, seen), the routing window
    # (blocked, V) and the coarse layer. Each panel is drawn at its OWN extent -- drawing them on
    # one would silently misplace the routing field by the crop offset.
    x0, y0 = d["bounds"]
    cell = float(d["cell"])
    n = d["height"].shape[0]
    ext = (x0, x0 + n * cell, y0, y0 + n * cell)
    rx0, ry0 = d["route_bounds"]
    nr = d["V"].shape[0]
    ext_r = (rx0, rx0 + nr * cell, ry0, ry0 + nr * cell)
    seen = d["seen"]
    trail = d["trail"]
    goal = d["goal"]
    mask = lambda v: np.where(seen, v, np.nan)  # noqa: E731

    blocked = d["blocked"]
    v = d["V"].min(2)
    # unreachable goes below the scale so it does not share a colour with a merely expensive cell
    cap = np.nanmax(v[np.isfinite(v)]) if np.isfinite(v).any() else 1.0
    vshow = np.where(v >= cap * 0.9, -1.0, v)

    cV = d["coarse_V"]
    if cV.size:
        ck = float(d["coarse_cell"])
        cshow = np.where(cV >= 1.0e29, -1.0, cV)
        ext_c = (x0, x0 + cV.shape[1] * ck, y0, y0 + cV.shape[0] * ck)
    else:
        cshow, ext_c = np.zeros((1, 1)), ext

    panels = [
        ("height the belief settled on  [m]", mask(d["height"]), "terrain", None, ext),
        ("ever measured", seen.astype(float), "Greys_r", (0, 1), ext),
        ("coarse: which way round  [m]", cshow, "magma", "route", ext_c),
        ("blocked  (fraction of headings)", blocked.mean(2), "Reds", (0, 1), ext_r),
        ("cost-to-go the MPPI followed  [m]", vshow, "viridis", "route", ext_r),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(26, 5.4))
    for ax, (title, img, cmap, lim, extent) in zip(axes, panels):
        if lim == "route":
            cm = plt.get_cmap(cmap).copy()
            cm.set_under("black")  # no route, as distinct from never measured
            kw = dict(cmap=cm, vmin=0.0)
        elif lim is None:
            kw = dict(cmap=cmap)
        else:
            kw = dict(cmap=cmap, vmin=lim[0], vmax=lim[1])
        ax.set_facecolor("0.82")  # never measured
        im = ax.imshow(img, origin="lower", extent=extent, **kw)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.plot(trail[:, 0], trail[:, 1], "-", color="deepskyblue", lw=2.0)
        ax.plot(*trail[0], "o", color="deepskyblue", ms=7, mec="k")
        ax.plot(*trail[-1], "o", color="white", ms=8, mec="k")
        ax.plot(goal[0], goal[1], "*", color="magenta", ms=17, mec="k")
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    end = float(np.hypot(trail[-1, 0] - goal[0], trail[-1, 1] - goal[1]))
    fig.suptitle(
        "driving on a belief: the map is what the robot built, in the frame it ended in\n"
        f"blue = driven   white = where it stopped   magenta star = goal   "
        f"{'REACHED' if bool(d['reached']) else f'stopped {end:.2f} m short'}"
        "        grey = never measured",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="studies/closed_loop/out/drive.npz")
    p.add_argument("--shot", default="/tmp/drive.png")
    a = p.parse_args()
    render(np.load(a.data), a.shot)


if __name__ == "__main__":
    main()
