"""The stratified Study-A figure.

Four panels, each carrying one of the study's claims:

  (a) the scene: what "flat / slope / curb / rock" actually are, with the rollouts on top and
      the adjoint's own support shaded -- so the strata are inspectable, not asserted.
  (b) adjoint vs finite difference, per region, on the settle path. A correct adjoint is the
      identity line; the historic ~47% error would sit visibly off it at slope 0.53.
  (c) the kinked fraction against the perturbation size, with and without the dilation. This
      is the paper-relevant panel: the cliff at the dilation's arg-max validity radius.
  (d) the belly-clearance term BEFORE and AFTER the tie fix, on one pair of axes: the tied
      `min` scatters into a cross (one arm where the adjoint reports a hard zero and the
      finite difference does not), the tie-free hinge sum lies on the identity.
  (e) the dilation contact margin -- how far terrain must move for the wheel's contact cell to
      change. This is the linearisation-validity flag Study B stratifies on, replacing the
      min N_i margin, which cannot reach zero inside the planner's tilt envelope.
  (f) that margin's distribution per region: it must be discriminative to be useful.
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


def figure(scene, harness, store: dict, rows: list[dict], diag: dict, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.5))
    _panel_scene(axes[0, 0], scene, harness, store)
    _panel_scatter(axes[0, 1], scene, store, ("settle",), "(b) settle path")
    _panel_kink(axes[0, 2], rows)
    _panel_scatter(axes[1, 0], scene, store, ("clear", "clear_soft"), "(d) belly clearance")
    _panel_margin_map(axes[1, 1], scene, diag)
    _panel_margin_dist(axes[1, 2], scene, diag)
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
    # vmin below zero so the flat lanes are not crushed against the bottom of the ramp
    ax.imshow(scene.elevation, origin="lower", extent=ext, cmap="pink", aspect="auto", vmin=-0.12)
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


def _panel_scatter(ax, scene, store, terms: tuple[str, ...], title: str) -> None:
    """Adjoint vs central FD at the smallest epsilon. One term -> coloured by region. Two terms
    -> the first is drawn grey (the 'before'), the second coloured (the 'after')."""
    sweep = store[(True, "elevation")]
    cells, is_zero = sweep["cells"], sweep["is_zero"]
    codes = scene.region[cells[:, 0], cells[:, 1]]
    scale = 0.0

    for n, term in enumerate(terms):
        k = TERM_NAMES.index(term)
        adj = sweep["grads"][k][:, cells[:, 0], cells[:, 1]].T
        fd = 0.5 * (sweep["d_plus"][0, :, k, :] + sweep["d_minus"][0, :, k, :])
        # Two terms differ in scale (a clearance vs a summed margin violation), so normalise
        # each by its own peak -- the panel is about SHAPE, not magnitude.
        norm = np.abs(fd[~is_zero]).max() if len(terms) > 1 else 1.0
        scale = max(scale, np.abs(fd[~is_zero]).max() / norm)
        for code, name in enumerate(REGION_NAMES[:-1]):
            sel = (codes == code) & ~is_zero
            if not sel.any():
                continue
            grey = len(terms) > 1 and n == 0
            ax.scatter(
                fd[sel].ravel() / norm,
                adj[sel].ravel() / norm,
                s=16 if grey else 9,
                alpha=0.35 if grey else 0.6,
                label=(f"{term}  ({name})" if len(terms) == 1 else None),
                color="#8a8a8a" if grey else REGION_COLOR[name],
                marker="x" if grey else "o",
                linewidths=0.8 if grey else 0,
            )
    if len(terms) > 1:
        ax.scatter([], [], marker="x", color="#8a8a8a", s=16, label=f"{terms[0]} (tied min)")
        for name in ("flat", "slope", "curb", "rock"):
            ax.scatter([], [], color=REGION_COLOR[name], s=9, label=f"{terms[1]} ({name})")
    lim = 1.1 * scale
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.0, label="exact")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("central finite difference  dJ/dh")
    ax.set_ylabel("adjoint  dJ/dh")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.25)


def _panel_margin_map(ax, scene, diag) -> None:
    """Where the dilation's arg-max is nearly tied -- i.e. where freezing it stops being the
    true derivative. Clipped at the flat-ground cap step, which is its geometric maximum."""
    m = 1e3 * diag["contact_margin"]
    cap_step = 1e3 * (0.35 - np.sqrt(0.35**2 - scene.cell**2))
    im = ax.imshow(
        m,
        origin="lower",
        extent=_extent(scene),
        cmap="magma",
        vmin=0.0,
        vmax=cap_step,
        aspect="auto",
    )
    plt.colorbar(im, ax=ax, label="winner - runner-up [mm]")
    ax.set_title("(e) dilation contact margin (dark = arg-max undecided)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def _panel_margin_dist(ax, scene, diag) -> None:
    """The flag is only useful if it separates the regions. Cumulative, so the left tail --
    the cells where the frozen gradient is invalid -- is what you read."""
    m = 1e3 * diag["contact_margin"]
    for code, name in enumerate(REGION_NAMES[:-1]):
        v = np.sort(m[scene.region == code].ravel())
        if v.size:
            ax.plot(
                v,
                100 * np.arange(v.size) / v.size,
                color=REGION_COLOR[name],
                lw=1.8,
                label=f"{name}  (n={v.size})",
            )
    ax.set_xlim(0.0, 10.0)  # the geometric max is ~3.6 mm; the left tail is what matters
    ax.set_xlabel("contact margin [mm]")
    ax.set_ylabel("cells below [%]")
    ax.set_title("(f) is the flag discriminative?")
    ax.legend(fontsize=8, loc="lower right")
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
