"""The stratified Study-A figure.

Four panels, each carrying one of the study's claims:

  (a) the scene: what "flat / slope / curb / rock" actually are, with the rollouts on top and
      the adjoint's own support shaded -- so the strata are inspectable, not asserted.
  (b) adjoint vs finite difference, per region, on the settle path. A correct adjoint is the
      identity line; the historic ~47% error would sit visibly off it at slope 0.53.
  (c) the kinked fraction against the perturbation size, with and without the dilation. This
      is the paper-relevant panel: the cliff at the dilation's arg-max validity radius.
  (d) the same scatter for the belly-clearance term, where the min-over-belly-points tie puts
      a large share of the mass on the horizontal axis (finite difference sees sensitivity,
      adjoint reports a hard zero).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .harness import TERM_NAMES
from .scene import REGION_NAMES

REGION_COLOR = {"flat": "#9aa5b1", "slope": "#2f7ec4", "curb": "#d1495b", "rock": "#e8a33d"}


def figure(scene, harness, store: dict, rows: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    _panel_scene(axes[0, 0], scene, harness, store)
    _panel_scatter(axes[0, 1], scene, store, term="settle", title="(b) settle path")
    _panel_kink(axes[1, 0], rows)
    _panel_scatter(axes[1, 1], scene, store, term="clear", title="(d) belly-clearance path")
    fig.suptitle(
        "Study A -- IFT settle adjoint vs finite differences on non-uniform terrain",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _extent(scene) -> tuple[float, float, float, float]:
    ny, nx = scene.shape
    return (
        scene.origin_x,
        scene.origin_x + nx * scene.cell,
        scene.origin_y,
        scene.origin_y + ny * scene.cell,
    )


def _panel_scene(ax, scene, harness, store) -> None:
    ext = _extent(scene)
    ax.imshow(scene.elevation, origin="lower", extent=ext, cmap="pink", aspect="auto")
    # Region outlines, so the strata are visible as regions rather than as a legend claim.
    for code, name in enumerate(REGION_NAMES[:-1]):
        if name == "flat":
            continue
        ax.contour(
            (scene.region == code).astype(float),
            levels=[0.5],
            origin="lower",
            extent=ext,
            colors=[REGION_COLOR[name]],
            linewidths=1.4,
        )
    grads = store[(True, "elevation")]["grads"]
    support = np.abs(grads).max(axis=(0, 1))
    ax.contourf(
        (support > 0.01 * support.max()).astype(float),
        levels=[0.5, 1.5],
        origin="lower",
        extent=ext,
        colors=["k"],
        alpha=0.18,
    )
    traj = harness.sim.controlled.numpy()  # [T+1, B, 3]
    for b in range(traj.shape[1]):
        ax.plot(traj[:, b, 0], traj[:, b, 1], "k-", lw=1.0)
        ax.plot(traj[0, b, 0], traj[0, b, 1], "k.", ms=5)
    handles = [
        plt.Line2D([], [], color=REGION_COLOR[n], lw=1.4, label=n)
        for n in ("slope", "curb", "rock")
    ]
    handles.append(plt.Line2D([], [], color="k", lw=6, alpha=0.18, label="adjoint support"))
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_title("(a) scene, rollouts and gradient support")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def _panel_scatter(ax, scene, store, term: str, title: str) -> None:
    """Adjoint vs central FD at the smallest epsilon, coloured by region."""
    sweep = store[(True, "elevation")]
    k = TERM_NAMES.index(term)
    cells, is_zero = sweep["cells"], sweep["is_zero"]
    adj = sweep["grads"][k][:, cells[:, 0], cells[:, 1]].T
    fd = 0.5 * (sweep["d_plus"][0, :, k, :] + sweep["d_minus"][0, :, k, :])
    codes = scene.region[cells[:, 0], cells[:, 1]]
    scale = np.abs(fd[~is_zero]).max()

    for code, name in enumerate(REGION_NAMES[:-1]):
        sel = (codes == code) & ~is_zero
        if not sel.any():
            continue
        ax.scatter(
            fd[sel].ravel(), adj[sel].ravel(), s=9, alpha=0.55, label=name, color=REGION_COLOR[name]
        )
    lim = 1.1 * scale
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.0, label="exact")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("central finite difference  dJ/dh")
    ax.set_ylabel("adjoint  dJ/dh")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)


def _panel_kink(ax, rows: list[dict]) -> None:
    """Kinked fraction vs perturbation size, dilation off (A2) vs on (A3)."""
    for level, style in (("A2", "--"), ("A3", "-")):
        for r in rows:
            if r["level"] != level or r["term"] != "settle" or r["region"] == "zero-control":
                continue
            eps = np.geomspace(3e-4, 3e-2, len(r["kinked_curve"]))
            ax.plot(
                eps,
                100 * np.array(r["kinked_curve"]),
                style,
                color=REGION_COLOR[r["region"]],
                lw=1.6,
                marker="o" if level == "A3" else None,
                ms=3.5,
            )
    ax.axvline(3.6e-3, color="k", lw=1.2, ls=":")
    ax.annotate(
        "dilation arg-max\nvalidity radius 3.6 mm",
        xy=(3.6e-3, 50),
        xytext=(4.5e-3, 30),
        fontsize=8,
    )
    ax.set_xscale("log")
    ax.set_xlabel("perturbation size eps [m]")
    ax.set_ylabel("kinked pairs [%]")
    ax.set_title("(c) where the forward stops being differentiable")
    handles = [plt.Line2D([], [], color=REGION_COLOR[n], lw=1.6, label=n) for n in REGION_COLOR]
    handles += [
        plt.Line2D([], [], color="k", ls="--", lw=1.6, label="A2: dilation OFF"),
        plt.Line2D([], [], color="k", ls="-", marker="o", ms=3.5, lw=1.6, label="A3: dilation ON"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper left", ncol=2)
    ax.grid(alpha=0.25)
