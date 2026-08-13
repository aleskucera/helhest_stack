"""Figures for the talk (TALK.md in the paper repo) -- act 1: the map is a belief.

One hero image, generated from the study's own simulated-sensing pipeline rather than
drawn: (a) what the mapper stores (holes are NaN), (b) what the planner is fed (holes
painted by inpainting), (c) the honest error bars (sigma, huge where painted).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import warp as wp

from studies.sensing.lidar_belief import CELL
from studies.sensing.lidar_belief import fractal_terrain
from studies.sensing.lidar_belief import LidarSim
from studies.sensing.lidar_belief import NoiseParams
from studies.sensing.lidar_belief import simulate_belief
from helhest.perception.heightmap.builder import HeightMapBuilder

OUT = Path(__file__).resolve().parents[1] / "out" / "bench"
DPI = 220

# muted ink for annotations; data colors come from the perceptual ramps
INK = "#37474f"
HOLE_GRAY = "#d6d6d6"


def harmonic_inpaint(z: np.ndarray, observed: np.ndarray, iters: int = 800) -> np.ndarray:
    """Diffusion fill: each hole cell relaxes to its neighbor mean, observed cells pinned.

    This is the same class of fill the deployed mappers use (elevation_mapping_cupy's
    OpenCV Navier-Stokes/Telea plugins); a Jacobi Laplace solve is the minimal honest
    stand-in and avoids an OpenCV dependency in the study env.
    """
    filled = np.where(observed, z, np.nanmean(z[observed]))
    for _ in range(iters):
        up = np.roll(filled, 1, axis=0)
        dn = np.roll(filled, -1, axis=0)
        lf = np.roll(filled, 1, axis=1)
        rt = np.roll(filled, -1, axis=1)
        avg = (up + dn + lf + rt) / 4.0
        filled = np.where(observed, filled, avg)
    return filled


def fig_act1(seed: int, realizations: int, device: str, path: Path) -> None:
    ny = nx = 90
    truth = fractal_terrain(ny, nx, CELL, seed=seed)
    xs = np.linspace(1.0, 8.0, 40)
    traj = np.stack([xs, np.full_like(xs, 4.5), np.zeros_like(xs)], axis=1)
    sim = LidarSim(truth, 0.0, 0.0, CELL, device)
    builder = HeightMapBuilder(
        CELL, (0.0, nx * CELL, 0.0, ny * CELL), device=wp.get_device(device)
    )
    p = NoiseParams()

    maps = []
    for r in range(realizations):
        m, _ = simulate_belief(sim, traj, p, seed * 1000 + r, builder)
        maps.append(m)
    stack = np.stack(maps)
    belief = stack[0]
    observed = np.isfinite(belief)
    # the sigma the mapper would publish: empirical spread where seen, capped-out where not
    sigma_obs = np.nanstd(stack, axis=0)
    sigma_cap = 0.10  # [m] the unobserved-cell cap the planning stack uses
    sigma = np.where(observed, np.where(np.isfinite(sigma_obs), sigma_obs, sigma_cap), sigma_cap)
    painted = harmonic_inpaint(belief, observed)

    frac_holes = 1.0 - observed.mean()
    med_sigma_cm = 100.0 * np.nanmedian(np.where(observed, sigma, np.nan))

    extent = (0.0, nx * CELL, 0.0, ny * CELL)
    vmin, vmax = np.nanpercentile(truth, 2), np.nanpercentile(truth, 98)
    terrain_cmap = plt.get_cmap("cividis").copy()
    terrain_cmap.set_bad(HOLE_GRAY)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), constrained_layout=True)
    ax_a, ax_b, ax_c = axes

    ax_a.imshow(
        np.where(observed, belief, np.nan), origin="lower", extent=extent,
        cmap=terrain_cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    ax_a.set_title(
        f"(a) what the mapper stores\nholes are honest: NaN ({frac_holes:.0%} of cells)",
        loc="left", fontsize=10,
    )

    im_b = ax_b.imshow(
        painted, origin="lower", extent=extent,
        cmap=terrain_cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    # outline the painted regions so the audience sees where the terrain is invented
    ax_b.contour(
        (~observed).astype(float), levels=[0.5], origin="lower", extent=extent,
        colors=[INK], linewidths=0.6, linestyles="dotted",
    )
    ax_b.set_title(
        "(b) what the planner is fed\ninpainted: invented ground inside the dotted line",
        loc="left", fontsize=10,
    )

    im_c = ax_c.imshow(
        sigma * 100.0, origin="lower", extent=extent, cmap="Oranges",
        vmin=0.0, vmax=sigma_cap * 100.0, interpolation="nearest",
    )
    ax_c.set_title(
        f"(c) the honest error bars\n$\\sigma\\approx${med_sigma_cm:.0f} cm measured, "
        f"{sigma_cap*100:.0f} cm where painted",
        loc="left", fontsize=10,
    )

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        # the robot's traverse, same in every panel
        ax.plot(traj[:, 0], traj[:, 1], color="#c62828", lw=1.4)
    ax_a.annotate(
        "robot's path", xy=(traj[20, 0], traj[20, 1]), xytext=(2.2, 2.7),
        color="#c62828", fontsize=9,
        arrowprops=dict(arrowstyle="-", color="#c62828", lw=0.8),
    )
    # one scale bar, leftmost panel
    ax_a.plot([0.4, 1.4], [0.45, 0.45], color=INK, lw=2.0)
    ax_a.text(0.9, 0.62, "1 m", ha="center", color=INK, fontsize=9)

    cb = fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.02)
    cb.set_label("height [m]", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb2 = fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.02)
    cb2.set_label("$\\sigma$ [cm]", fontsize=9)
    cb2.ax.tick_params(labelsize=8)

    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}  (holes {frac_holes:.1%}, median sigma {med_sigma_cm:.1f} cm)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--realizations", type=int, default=8)
    args = ap.parse_args()
    wp.init()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_act1(args.seed, args.realizations, args.device, OUT / "talk_act1_map.png")


if __name__ == "__main__":
    main()
