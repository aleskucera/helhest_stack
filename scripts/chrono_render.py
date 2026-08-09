"""Draw what the Chrono runs did: soil deformation, wheel tracks, and the paths.

    python scripts/chrono_render.py /tmp/chrono_capture.npz --outdir /tmp/render

Runs in the project venv (Chrono's environment has no matplotlib). Input comes from
`scripts/chrono_capture.py`. Chrono's own Irrlicht/VSG visualisers are not in this build -- it is
headless and they were disabled -- so this draws from the recorded state instead, which has the
advantage of showing the SCM height field directly rather than a shaded mesh of it.
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

CHASSIS_BOXES = ((-0.13, 0.48, 0.56), (-0.61, 0.48, 0.24))  # (centre_x, length, width)
BG, FG, GRID = "#12151c", "#e8eaed", "#2c313c"


def rot(pts: np.ndarray, yaw: float, x: float, y: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.column_stack([x + pts[:, 0] * c - pts[:, 1] * s, y + pts[:, 0] * s + pts[:, 1] * c])


def robot_patches(pose, wheel_pos, radius, half_width, colour, alpha=1.0, lw=1.0):
    x, y, _z, yaw = pose
    out = []
    for cx, ln, wd in CHASSIS_BOXES:
        box = np.array([[cx - ln / 2, -wd / 2], [cx + ln / 2, -wd / 2],
                        [cx + ln / 2, wd / 2], [cx - ln / 2, wd / 2]])
        out.append(Polygon(rot(box, yaw, x, y), closed=True, facecolor="none",
                           edgecolor=colour, lw=lw, alpha=alpha))
    for wx, wy, _wz in wheel_pos:
        w = np.array([[wx - radius, wy - half_width], [wx + radius, wy - half_width],
                      [wx + radius, wy + half_width], [wx - radius, wy + half_width]])
        out.append(Polygon(rot(w, yaw, x, y), closed=True, facecolor=colour,
                           edgecolor="none", alpha=0.55 * alpha))
    return out


def style(ax, title: str) -> None:
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.tick_params(colors=FG, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_title(title, color=FG, fontsize=10, pad=8)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.5)


def soil_figure(d, case: str, path: pathlib.Path) -> None:
    gx, gy = d["grid_x"], d["grid_y"]
    soils, poses, tracks = d[f"{case}/soils"], d[f"{case}/poses"], d[f"{case}/tracks"]
    fig, ax = plt.subplots(figsize=(9, 8), facecolor=BG, layout="constrained")
    # plotted as POSITIVE rut depth so undisturbed ground is black and sinks into the page
    # background, leaving only what the wheels actually did
    depth = -soils[-1] * 100.0
    m = ax.pcolormesh(gx, gy, depth, cmap="inferno", shading="auto",
                      vmin=0.0, vmax=float(depth.max()))
    cb = fig.colorbar(m, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("rut depth [cm]", color=FG, fontsize=9)
    cb.ax.tick_params(colors=FG, labelsize=8)
    for k in range(tracks.shape[1]):
        ax.plot(tracks[:, k, 0], tracks[:, k, 1], color="#7fd1ff", lw=0.8, alpha=0.8)
    ax.plot(poses[:, 0], poses[:, 1], color=FG, lw=1.4, label="chassis path")
    for i in np.linspace(0, len(poses) - 1, 7).astype(int):
        for p in robot_patches(poses[i], d["wheel_pos"], d["wheel_radius"], d["half_width"],
                               "#7fd1ff", alpha=0.85):
            ax.add_patch(p)
    style(ax, f"{case} — deformed soil after 6 s (SCM, Bekker–Wong + Janosi–Hanamoto)")
    ax.set_xlabel("x [m]", color=FG)
    ax.set_ylabel("y [m]", color=FG)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8, loc="lower right")
    fig.savefig(path, dpi=140, facecolor=BG)
    plt.close(fig)


def paths_figure(d, cases, path: pathlib.Path) -> None:
    fig, axes = plt.subplots(
        1, len(cases), figsize=(4.2 * len(cases), 5.4), facecolor=BG, layout="constrained"
    )
    for ax, (case, label) in zip(np.atleast_1d(axes), cases):
        poses = d[f"{case}/poses"]
        ax.plot(poses[:, 0], poses[:, 1], color="#7fd1ff", lw=1.6)
        for i in np.linspace(0, len(poses) - 1, 6).astype(int):
            for p in robot_patches(poses[i], d["wheel_pos"], d["wheel_radius"], d["half_width"],
                                   FG, alpha=0.5, lw=0.8):
                ax.add_patch(p)
        style(ax, label)
        ax.set_xlabel("x [m]", color=FG)
    np.atleast_1d(axes)[0].set_ylabel("y [m]", color=FG)
    fig.suptitle("same command (wL 1, wR 3 rad/s) — the rear wheel decides the turn",
                 color=FG, fontsize=12)
    fig.savefig(path, dpi=140, facecolor=BG)
    plt.close(fig)


def animate(d, case: str, path: pathlib.Path) -> None:
    gx, gy = d["grid_x"], d["grid_y"]
    soils, poses = d[f"{case}/soils"], d[f"{case}/poses"]
    times = d[f"{case}/times"]
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG)
    vmax = float(-soils[-1].min() * 100.0)
    mesh = ax.pcolormesh(gx, gy, -soils[0] * 100.0, cmap="inferno", shading="auto",
                         vmin=0.0, vmax=vmax)
    (trail,) = ax.plot([], [], color=FG, lw=1.2)
    style(ax, "")
    ax.set_xlabel("x [m]", color=FG)
    ax.set_ylabel("y [m]", color=FG)
    cb = fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("rut depth [cm]", color=FG, fontsize=9)
    cb.ax.tick_params(colors=FG, labelsize=8)
    patches: list = []

    def frame(i):
        nonlocal patches
        mesh.set_array((-soils[i] * 100.0).ravel())
        k = min(int(i / max(len(soils) - 1, 1) * (len(poses) - 1)), len(poses) - 1)
        trail.set_data(poses[: k + 1, 0], poses[: k + 1, 1])
        for p in patches:
            p.remove()
        patches = robot_patches(poses[k], d["wheel_pos"], d["wheel_radius"], d["half_width"],
                                "#7fd1ff")
        for p in patches:
            ax.add_patch(p)
        ax.set_title(f"{case} — t = {times[i]:.2f} s", color=FG, fontsize=10, pad=8)
        return [mesh, trail, *patches]

    anim = animation.FuncAnimation(fig, frame, frames=len(soils), blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=8), savefig_kwargs={"facecolor": BG})
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    d = np.load(args.capture)
    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    soil_figure(d, "scm_fixed_turn", out / "chrono_soil_turn.png")
    soil_figure(d, "scm_fixed_straight", out / "chrono_soil_straight.png")
    paths_figure(
        d,
        [("rigid_fixed_turn", "rigid ground, fixed rear axle"),
         ("rigid_caster_turn", "rigid ground, rear caster"),
         ("scm_fixed_turn", "SCM soil, fixed rear axle"),
         ("scm_caster_turn", "SCM soil, rear caster")],
        out / "chrono_paths.png",
    )
    animate(d, "scm_fixed_turn", out / "chrono_turn.gif")
    for f in sorted(out.iterdir()):
        print(f"  {f}  ({f.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
