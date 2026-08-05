"""The Study B figure: when is first-order attribution valid?

(a) the placeholder sigma field, with the decoy patch marked -- high uncertainty placed
    deliberately off the driven line, so entropy-directed and attribution-directed sensing
    disagree about it.
(b) global adequacy: sd_MC / sd_FOSM per rollout against the sigma scale. The self-check
    lives here too -- every curve must pass through 1 at the left edge.
(c) THE CENTRAL PANEL. Per-cell ratio against the perturbation measured in validity radii
    (sigma / slack). The claim is that the vertical line at 1 separates valid from invalid
    attribution, independently of region, sigma, or gradient magnitude.
(d) the same per-cell ratios grouped by region, showing that region alone does NOT predict
    the breakdown -- the radius does. This is the panel that says the criterion is
    transferable rather than a property of this scene.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .scene import REGION_NAMES

REGION_COLOR = {"flat": "#9aa5b1", "slope": "#2f7ec4", "curb": "#d1495b", "rock": "#e8a33d"}
BAD = 2.0  # a per-cell ratio worse than this counts as a failed attribution


def figure(scene, sigma_np, decoy, b1: list[dict], b2: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    _panel_sigma(axes[0, 0], scene, sigma_np, decoy)
    _panel_global(axes[0, 1], b1)
    _panel_criterion(axes[1, 0], b2)
    _panel_region(axes[1, 1], b2)
    fig.suptitle(
        "Study B -- first-order attribution is valid exactly inside its own validity radius",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _panel_sigma(ax, scene, sigma_np, decoy) -> None:
    ny, nx = scene.shape
    ext = (
        scene.origin_x,
        scene.origin_x + nx * scene.cell,
        scene.origin_y,
        scene.origin_y + ny * scene.cell,
    )
    im = ax.imshow(1e2 * sigma_np, origin="lower", extent=ext, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="placeholder sigma [cm]")
    ax.contour(
        decoy.astype(float), levels=[0.5], origin="lower", extent=ext, colors=["w"], linewidths=1.4
    )
    ax.set_title("(a) placeholder uncertainty; white = unobserved decoy patch")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def _panel_global(ax, b1) -> None:
    corr = [r for r in b1 if r["corr_len"] > 0]
    rollouts = sorted({r["rollout"] for r in corr})
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(rollouts):
        rs = sorted((r for r in corr if r["rollout"] == name), key=lambda r: r["sigma_scale"])
        ax.plot(
            [r["sigma_scale"] for r in rs],
            [r["ratio"] for r in rs],
            marker="o",
            ms=3.5,
            lw=1.4,
            color=cmap(i % 10),
            label=name,
        )
    ax.axhline(1.0, color="k", ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sigma scale (x the placeholder field)")
    ax.set_ylabel("sd_MC / sd_FOSM")
    ax.set_title("(b) global adequacy per rollout (left edge = self-check)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25, which="both")


def _panel_criterion(ax, b2) -> None:
    """Ratio vs sigma/slack. Every point from every region and every sigma scale."""
    finite = [r for r in b2 if np.isfinite(r["ratio"]) and r["sigma_over_slack"] > 0]
    xmax = 1e3  # exact-tie cells run to the slack floor; clip so the decade of interest reads
    for code, name in enumerate(REGION_NAMES[:-1]):
        rs = [r for r in finite if r["region"] == name]
        if rs:
            ax.scatter(
                [min(r["sigma_over_slack"], xmax) for r in rs],
                [r["ratio"] for r in rs],
                s=11,
                alpha=0.5,
                color=REGION_COLOR[name],
                label=name,
            )
    ax.axvline(1.0, color="k", lw=1.4)
    ax.axhline(1.0, color="k", ls="--", lw=1.0)
    ax.axhline(BAD, color="#a00", ls=":", lw=1.2)
    inside = [r["ratio"] for r in finite if r["sigma_over_slack"] < 1.0]
    ax.annotate(
        f"inside the radius:\n{len(inside)} cells, "
        f"{np.mean(np.array(inside) > BAD):.0%} worse than {BAD:g}x",
        xy=(0.03, 0.93),
        xycoords="axes fraction",
        fontsize=8,
        va="top",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(right=2 * xmax)
    ax.set_xlabel("perturbation in validity radii   sigma / slack  (clipped at 1e3)")
    ax.set_ylabel("per-cell  sd_MC / sd_FOSM")
    ax.set_title("(c) the criterion: valid left of the line")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25, which="both")


def _panel_region(ax, b2) -> None:
    """The same failures grouped by region -- region does not separate them, the radius does."""
    scales = sorted({r["sigma_scale"] for r in b2})
    width = 0.8 / max(len(scales), 1)
    names = [n for n in REGION_NAMES[:-1] if any(r["region"] == n for r in b2)]
    for j, gain in enumerate(scales):
        fracs, xs = [], []
        for i, name in enumerate(names):
            v = [
                r["ratio"]
                for r in b2
                if r["region"] == name and r["sigma_scale"] == gain and np.isfinite(r["ratio"])
            ]
            if v:
                fracs.append(100 * np.mean(np.array(v) > BAD))
                xs.append(i + (j - (len(scales) - 1) / 2) * width)
        ax.bar(xs, fracs, width=width * 0.9, label=f"sigma x{gain:g}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel(f"cells worse than {BAD:g}x  [%]")
    ax.set_title("(d) region alone does not predict the breakdown")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
