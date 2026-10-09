"""Generate the lattice-planner steepness benchmark: one map per ramp angle.

    python scripts/make_ramp_series.py
    python scripts/make_ramp_series.py --angles 10 20 30 --out /tmp/steep --plot

Each map is flat ground crossed by a symmetric full-width ridge (see helhest.planning.rampmaps):
the plateau HEIGHT is held constant and only the face angle changes, so the series isolates
steepness from every other variable. Start and goal sit on opposite X edges and a full-width ridge
admits no detour, so the robot must climb and descend or the goal is unreachable.

Every map shares ONE grid, sized from the SHALLOWEST angle in the sweep (the shallowest ramp has
the longest run, so it needs the most room), which keeps V and path length directly comparable
across the series. Start and goal are measured from the grid edges, so they are identical in every
map too.

The default sweep stops at 75 deg because that is the steepest face a 0.1 m cell renders faithfully
for a 0.75 m rise -- see rampmaps.max_renderable_deg. Asking for more raises rather than quietly
producing a shallower ramp; pass a finer --cell to go steeper.

Maps are written as float32 .npz. The vendored heightmap/ package writes 8-bit PNG + YAML, which
would quantize a 0.75 m rise to ~3 mm and put a staircase on what has to be a clean constant-slope
face -- this benchmark resolves a feasibility boundary against a 25 deg envelope limit, so the
elevation is kept lossless.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

from helhest.planning.rampmaps import DEFAULT_CELL
from helhest.planning.rampmaps import DEFAULT_EXTENT_Y
from helhest.planning.rampmaps import DEFAULT_HEIGHT
from helhest.planning.rampmaps import DEFAULT_MARGIN
from helhest.planning.rampmaps import DEFAULT_PLATEAU
from helhest.planning.rampmaps import bump_ridge
from helhest.planning.rampmaps import face_angle_deg
from helhest.planning.rampmaps import ridge_extent
from helhest.planning.rampmaps import start_goal

# deg. Brackets the robot's envelope (max_pitch_down 15, max_pitch_up 25) and hits both limits
# exactly, which is the transition the benchmark exists to resolve.
DEFAULT_ANGLES = tuple(float(d) for d in range(5, 76, 5))


def map_path(out_dir: pathlib.Path, up_deg: float) -> pathlib.Path:
    """<out>/bump_a0125.npz -- angle in tenths of a degree, zero-padded so a plain sorted glob
    comes back in sweep order, and with no dot in the stem."""
    return out_dir / f"bump_a{round(up_deg * 10):04d}.npz"


def plot_series(paths: list[pathlib.Path], out: pathlib.Path) -> None:
    """Contact sheet of every profile, for eyeballing that the faces steepen while the peak holds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    for path in paths:
        with np.load(path) as data:
            heights, cell, origin = data["heights"], float(data["cell"]), data["origin"]
            xs = origin[0] + (np.arange(heights.shape[1]) + 0.5) * cell
            for ax in axes:
                ax.plot(xs, heights[0], lw=1.0, label=f"{float(data['up_deg']):.0f} deg")
    # full extent first (stretched, so the shallow ramps are readable), then the crest at true
    # 1:1 aspect, where the face angles can actually be judged by eye
    axes[0].set_title("ramp series -- constant height, swept face angle")
    axes[1].set_xlim(-3.0, 3.0)
    axes[1].set_aspect("equal")
    axes[1].set_title("crest, 1:1 aspect")
    for ax in axes:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--angles", type=float, nargs="+", default=list(DEFAULT_ANGLES), help="deg")
    ap.add_argument("--height", type=float, default=DEFAULT_HEIGHT, help="plateau height [m]")
    ap.add_argument("--plateau", type=float, default=DEFAULT_PLATEAU, help="flat top length [m]")
    ap.add_argument("--cell", type=float, default=DEFAULT_CELL, help="grid resolution [m]")
    ap.add_argument("--extent-y", type=float, default=DEFAULT_EXTENT_Y, help="grid width [m]")
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN, help="flat ground each end [m]")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/tmp/ramp_series"))
    ap.add_argument(
        "--plot", action="store_true", help="also write a contact sheet of the profiles"
    )
    args = ap.parse_args()

    angles = sorted(args.angles)
    extent_x = ridge_extent(angles[0], args.height, args.plateau, args.margin, args.cell)
    args.out.mkdir(parents=True, exist_ok=True)

    paths = []
    print(f"{'deg':>6} {'measured':>9} {'run_m':>7} {'grid':>10} {'file':>18}")
    for up_deg in angles:
        hm = bump_ridge(up_deg, extent_x, args.height, args.plateau, args.extent_y, args.cell)
        start, goal = start_goal(hm, args.margin)
        measured = face_angle_deg(hm)
        path = map_path(args.out, up_deg)
        np.savez(
            path,
            heights=np.ascontiguousarray(hm.H, np.float32),
            origin=np.array([hm.x0, hm.y0], np.float64),
            cell=hm.cell,
            up_deg=up_deg,
            measured_deg=measured,
            height=args.height,
            plateau=args.plateau,
            start=np.array(start, np.float64),
            goal=np.array(goal, np.float64),
        )
        paths.append(path)
        run = args.height / np.tan(np.radians(up_deg))
        print(f"{up_deg:6.1f} {measured:9.2f} {run:7.3f} {hm.nx:4d}x{hm.ny:<4d} {path.name:>18}")

    print(
        f"\n{len(paths)} maps in {args.out}  "
        f"({extent_x:.2f} x {args.extent_y:.1f} m at {args.cell} m, peak {args.height:.2f} m)\n"
        f"start {start[:2]} -> goal {goal}, shared by every map"
    )
    if args.plot:
        plot_series(paths, args.out / "series.png")


if __name__ == "__main__":
    main()
