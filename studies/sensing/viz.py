"""Figures for the active-sensing experiment: uncertainty is not relevance.

    .venv/bin/python -m studies.sensing.viz --case studies/out/sensing/case_seed7.npz

The robot may take exactly ONE look before it commits to one of K candidate plans on a
partially observed map. Every policy uses the same look, the same reveal geometry and the same
re-costing; they differ ONLY in the per-cell score field that ranks the viewpoints:

  ENTROPY        sigma^2 -- look where the map is least certain. Knows nothing about the plans.
  SWATH_VAR      geometry -- look where the candidate paths spread. Knows the plans, not the cost.
  DISAGREEMENT   Var_k(dJ_k/dh) * sigma^2 (OURS) -- look where the plans' costs DISAGREE, i.e.
                 the cells that can still change WHICH plan wins.
  ORACLE         not a policy: it is allowed to see the truth, and marks the ceiling.

Figure 1 (`pipeline_seed{N}.png`) is the showcase for one seed; figure 2 (`summary.png`) is the
evidence over every seed in results.json. Both read numbers only -- no tau, score or cost is
recomputed here.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

# Headless: every figure is written to disk, never shown. Must precede the pyplot import.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Wedge

OUT = Path("studies/out/sensing")

# Row order is the argument: nothing -> blind -> uncertainty -> geometry -> ours -> ceiling.
POLICIES: tuple[str, ...] = ("none", "random", "entropy", "swath_var", "disagreement", "oracle")
SCORE_PANELS: tuple[str, ...] = ("entropy", "swath_var", "disagreement")

# Dark2-derived; distinguishable under deuteranopia/protanopia (no red-vs-green pair carries
# meaning on its own -- the emphasised trajectory is also thicker and labelled).
COLOR: dict[str, str] = {
    "none": "#8c8c8c",
    "random": "#a6761d",
    "entropy": "#7570b3",
    "swath_var": "#1b9e77",
    "disagreement": "#d95f02",
    "oracle": "#2b2b2b",
}
LABEL: dict[str, str] = {
    "none": "none (no look)",
    "random": "random",
    "entropy": "entropy (uncertainty)",
    "swath_var": "swath var (geometry)",
    "disagreement": "disagreement (ours)",
    "oracle": "oracle (peeks at truth)",
}
PANEL_TITLE: dict[str, str] = {
    "entropy": "(d) ENTROPY: where the map is least certain",
    "swath_var": "(e) SWATH VAR: where the plans spread",
    "disagreement": "(f) DISAGREEMENT (ours): where the costs disagree",
}

ELEV_CMAP = "viridis"  # one perceptually-uniform map for every elevation panel
SCORE_CMAP = "magma"  # one sequential map for all three score fields
COST_CMAP = "RdYlGn_r"
FOV_HALF_DEG = 60.0  # the look's field of view, half-angle
FOV_RANGE_M = 8.0
SCORE_GAMMA = 0.7  # mild lift so diffuse fields stay visible without flattening concentration

REQUIRED_CASE_KEYS: tuple[str, ...] = (
    "truth",
    "belief",
    "belief_after_disagreement",
    "sigma",
    "meas",
    "obs",
    "cell",
    "origin",
    "robot_pose",
    "traj",
    "j_true",
    "j_belief",
    "viewpoints",
    "tau_before",
    *[f"score_{p}" for p in SCORE_PANELS],
    *[f"vp_{p}" for p in ("entropy", "swath_var", "disagreement", "random", "oracle")],
    *[f"reveal_{p}" for p in SCORE_PANELS],
    *[f"jafter_{p}" for p in POLICIES],
    *[f"tau_{p}" for p in POLICIES],
)


# --------------------------------------------------------------------------------------- io


def _load_case(path: Path) -> dict[str, np.ndarray]:
    """Load the per-seed npz and fail loudly (not silently) on a schema mismatch."""
    with np.load(path) as handle:
        case = {name: handle[name] for name in handle.files}
    missing = [k for k in REQUIRED_CASE_KEYS if k not in case]
    if missing:
        raise KeyError(f"{path}: missing keys {missing} (schema mismatch, not adapting)")
    return case


def _load_results(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text())
    if "rows" not in payload:
        raise KeyError(f"{path}: missing 'rows' (schema mismatch, not adapting)")
    return list(payload["rows"])


def _seed_label(path: Path) -> str:
    match = re.search(r"seed(\d+)", path.stem)
    return match.group(1) if match else path.stem


# ---------------------------------------------------------------------------------- helpers


def _extent(cell: float, origin: np.ndarray, shape: tuple[int, int]) -> tuple[float, ...]:
    """imshow extent [m] for a grid whose cell (iy,ix) centre is origin + (i+0.5)*cell."""
    ny, nx = shape
    return (
        float(origin[0]),
        float(origin[0] + nx * cell),
        float(origin[1]),
        float(origin[1] + ny * cell),
    )


def _centres(cell: float, origin: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, ...]:
    ny, nx = shape
    xs = origin[0] + (np.arange(nx) + 0.5) * cell
    ys = origin[1] + (np.arange(ny) + 0.5) * cell
    return np.meshgrid(xs, ys)


def _frame(ax: Axes, ext: tuple[float, ...], title: str, xlabel: bool, ylabel: bool) -> None:
    ax.set_title(title, fontsize=10.5)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.tick_params(labelsize=9)
    if xlabel:
        ax.set_xlabel("x [m]", fontsize=9.5)
    else:
        ax.set_xticklabels([])
    if ylabel:
        ax.set_ylabel("y [m]", fontsize=9.5)
    else:
        ax.set_yticklabels([])


def _draw_pose(ax: Axes, pose: np.ndarray, span: float, color: str, zorder: float = 6.0) -> None:
    """Robot as a dot plus a heading arrow scaled to the panel so it reads at any zoom."""
    x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    length = 0.07 * span
    ax.annotate(
        "",
        xy=(x + length * np.cos(yaw), y + length * np.sin(yaw)),
        xytext=(x, y),
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.0, "shrinkA": 0, "shrinkB": 0},
        zorder=zorder,
    )
    ax.plot([x], [y], "o", ms=6.0, mfc=color, mec="white", mew=1.2, zorder=zorder)


def _fov_radius(span: float) -> float:
    """On a small map the nominal 8 m wedge swallows the whole panel, so cap it at half the
    map. The wedge is a schematic of the heading; the reveal overlay is the real footprint."""
    return min(FOV_RANGE_M, 0.5 * span)


def _draw_fov(ax: Axes, pose: np.ndarray, color: str, radius: float) -> None:
    heading = np.degrees(float(pose[2]))
    wedge = Wedge(
        (float(pose[0]), float(pose[1])),
        radius,
        heading - FOV_HALF_DEG,
        heading + FOV_HALF_DEG,
        facecolor=color,
        edgecolor=color,
        alpha=0.14,
        lw=1.2,
        zorder=4,
    )
    wedge.set_clip_on(True)  # the wedge runs off the map on purpose; clip it to the axes
    ax.add_patch(wedge)


def _box_blur(field: np.ndarray, radius: int) -> np.ndarray:
    """Display-only smoothing so contours of a cell-noisy field stay readable (never plotted
    as a number, and never applied to anything the figure quotes)."""
    if radius < 1:
        return field
    kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
    smoothed = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 0, field)
    return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 1, smoothed)


def _normalise_score(score: np.ndarray) -> np.ndarray:
    """Each field lives in its own units -- clip at its own 99.5th pct and gamma-lift."""
    values = np.asarray(score, float)
    high = float(np.nanpercentile(values, 99.5))
    if not np.isfinite(high) or high <= 0.0:
        high = float(np.nanmax(values)) if np.isfinite(np.nanmax(values)) else 1.0
    high = high if high > 0.0 else 1.0
    return np.clip(values / high, 0.0, 1.0) ** SCORE_GAMMA


def _bootstrap_ci(values: np.ndarray, n_resample: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(n_resample, values.size))
    means = values[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# --------------------------------------------------------------------------------- figure 1


def figure_pipeline(case: dict[str, np.ndarray], out_dir: Path, label: str) -> Path:
    truth = np.asarray(case["truth"], float)
    belief = np.asarray(case["belief"], float)
    sigma = np.asarray(case["sigma"], float)
    obs = np.asarray(case["obs"], bool)
    cell = float(np.asarray(case["cell"]).reshape(-1)[0])
    origin = np.asarray(case["origin"], float).reshape(-1)
    pose = np.asarray(case["robot_pose"], float).reshape(-1)
    traj = np.asarray(case["traj"], float)
    j_true = np.asarray(case["j_true"], float).reshape(-1)
    tau_before = float(np.asarray(case["tau_before"]).reshape(-1)[0])
    tau = {p: float(np.asarray(case[f"tau_{p}"]).reshape(-1)[0]) for p in POLICIES}

    ext = _extent(cell, origin, truth.shape)
    span = max(ext[1] - ext[0], ext[3] - ext[2])
    grid_x, grid_y = _centres(cell, origin, truth.shape)
    elev_lo, elev_hi = np.percentile(np.concatenate([truth.ravel(), belief.ravel()]), [1.0, 99.0])
    cost_norm = Normalize(vmin=float(j_true.min()), vmax=float(j_true.max()))
    best = int(np.argmin(j_true))

    fov_radius = _fov_radius(span)
    blur = max(1, min(truth.shape) // 30)

    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.5), constrained_layout=True, dpi=140)
    # Reserve a strip at the bottom for the caption and one at the top for the suptitle; the
    # layout engine otherwise lets the panels grow into both.
    fig.get_layout_engine().set(rect=(0.0, 0.055, 1.0, 0.90))

    # (a) the world as it is, and every plan the robot is choosing between -------------------
    ax = axes[0, 0]
    ax.imshow(
        truth, origin="lower", extent=ext, cmap=ELEV_CMAP, vmin=elev_lo, vmax=elev_hi, zorder=1
    )
    cmap_cost = plt.get_cmap(COST_CMAP)
    for k in range(traj.shape[1]):
        if k == best:
            continue
        ax.plot(
            traj[:, k, 0],
            traj[:, k, 1],
            color=cmap_cost(cost_norm(j_true[k])),
            lw=1.3,
            alpha=0.95,
            zorder=3,
        )
    ax.plot(
        traj[:, best, 0],
        traj[:, best, 1],
        color="white",
        lw=4.2,
        alpha=0.9,
        solid_capstyle="round",
        zorder=4,
    )
    ax.plot(
        traj[:, best, 0],
        traj[:, best, 1],
        color=cmap_cost(cost_norm(j_true[best])),
        lw=2.4,
        zorder=5,
        label=f"true best plan (k={best}, J={j_true[best]:.2f})",
    )
    _draw_pose(ax, pose, span, "#d62728")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9, borderpad=0.4)
    _frame(
        ax,
        ext,
        f"(a) Ground truth + {traj.shape[1]} candidate plans\n(colour = true cost; the robot"
        " cannot see any of this)",
        xlabel=False,
        ylabel=True,
    )
    bar = fig.colorbar(
        ScalarMappable(norm=cost_norm, cmap=COST_CMAP), ax=ax, fraction=0.046, pad=0.02
    )
    bar.set_label("true plan cost J", fontsize=9)
    bar.ax.tick_params(labelsize=9)

    # (b) the same world as the robot has it: a belief plus a hole where nothing was seen -----
    ax = axes[0, 1]
    im = ax.imshow(
        belief, origin="lower", extent=ext, cmap=ELEV_CMAP, vmin=elev_lo, vmax=elev_hi, zorder=1
    )
    unseen = np.zeros((*obs.shape, 4), float)
    unseen[~obs] = (1.0, 1.0, 1.0, 0.45)  # desaturate, do not hide, the unobserved region
    ax.imshow(unseen, origin="lower", extent=ext, zorder=2, interpolation="nearest")
    # Smoothed before contouring: a real observed mask is speckled with occlusion shadows and
    # its raw outline is unreadable. The pale overlay above already shows the mask exactly.
    ax.contour(
        grid_x,
        grid_y,
        _box_blur(obs.astype(float), blur),
        levels=[0.5],
        colors="#111111",
        linewidths=1.4,
        zorder=3,
    )
    smooth_sigma = _box_blur(sigma, blur)
    levels = np.unique(np.round(np.percentile(smooth_sigma, [70.0, 92.0]), 4))
    if levels.size:
        cs = ax.contour(
            grid_x,
            grid_y,
            smooth_sigma,
            levels=levels,
            colors="#d62728",
            linewidths=1.1,
            alpha=0.9,
            zorder=5,
        )
        ax.clabel(cs, inline=True, fontsize=9, fmt="%.2f")
    for k in range(traj.shape[1]):
        ax.plot(traj[:, k, 0], traj[:, k, 1], color="#222222", lw=0.7, alpha=0.35, zorder=4)
    _draw_pose(ax, pose, span, "#d62728")
    _frame(
        ax,
        ext,
        "(b) What the robot knows\n(pale = unobserved, red = sigma [m])",
        xlabel=False,
        ylabel=False,
    )
    bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    bar.set_label("elevation [m]", fontsize=9)
    bar.ax.tick_params(labelsize=9)

    # (c) the outcome of that one look, per policy -------------------------------------------
    ax = axes[0, 2]
    rows = list(range(len(POLICIES)))
    for row, policy in zip(rows, POLICIES):
        y = len(POLICIES) - 1 - row
        after = tau[policy]
        colour = COLOR[policy]
        ax.annotate(
            "",
            xy=(after, y),
            xytext=(tau_before, y),
            arrowprops={
                "arrowstyle": "-|>",
                "color": colour,
                "lw": 2.0,
                "alpha": 0.85,
                "shrinkA": 3,
                "shrinkB": 0,
            },
        )
        ax.plot([tau_before], [y], "o", ms=5.0, mfc="white", mec="#999999", mew=1.2, zorder=4)
        hollow = policy == "oracle"
        ax.plot(
            [after],
            [y],
            "o",
            ms=9.0 if policy in ("disagreement", "oracle") else 7.0,
            mfc="white" if hollow else colour,
            mec=colour,
            mew=1.8,
            zorder=5,
        )
        ax.text(
            after + 0.012,
            y + 0.22,
            f"{after:.2f}",
            fontsize=9.5,
            color=colour,
            fontweight="bold" if policy == "disagreement" else "normal",
            va="bottom",
            ha="left",
        )
    ax.axvline(tau_before, color="#999999", ls="--", lw=1.1, zorder=1)
    ax.set_yticks(rows)
    ax.set_yticklabels([LABEL[p] for p in reversed(POLICIES)], fontsize=9.5)
    for tick, policy in zip(ax.get_yticklabels(), reversed(POLICIES)):
        tick.set_color(COLOR[policy])
        if policy == "disagreement":
            tick.set_fontweight("bold")
    lo = min(tau_before, min(tau.values()))
    hi = max(tau_before, max(tau.values()))
    pad = 0.10 * max(hi - lo, 0.05)
    ax.set_xlim(lo - pad, hi + 2.6 * pad)
    ax.set_ylim(-0.7, len(POLICIES) - 0.1)
    ax.set_xlabel("Kendall tau of re-ranked plans vs truth", fontsize=9.5)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", color="#dddddd", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(
        f"(c) One look, seed {label}: tau {tau_before:.2f} -> entropy {tau['entropy']:.2f},"
        f" ours {tau['disagreement']:.2f} (oracle {tau['oracle']:.2f})",
        fontsize=10.5,
    )
    ax.text(
        0.99,
        0.02,
        "dashed = before looking",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#777777",
    )

    # (d,e,f) the three score fields, treated identically so they can be compared -------------
    backdrop = np.ma.masked_invalid(belief)
    score_images = []
    for col, policy in enumerate(SCORE_PANELS):
        ax = axes[1, col]
        ax.imshow(
            backdrop,
            origin="lower",
            extent=ext,
            cmap="Greys",
            vmin=elev_lo - 0.9 * (elev_hi - elev_lo),
            vmax=elev_hi + 1.6 * (elev_hi - elev_lo),
            zorder=1,
        )
        for k in range(traj.shape[1]):
            ax.plot(traj[:, k, 0], traj[:, k, 1], color="#3b3b3b", lw=0.7, alpha=0.35, zorder=2)
        norm_score = _normalise_score(np.asarray(case[f"score_{policy}"], float))
        image = ax.imshow(
            norm_score,
            origin="lower",
            extent=ext,
            cmap=SCORE_CMAP,
            vmin=0.0,
            vmax=1.0,
            # Cap the alpha: a saturated field (real sigma is often flat) must not go opaque
            # and hide the plans underneath.
            alpha=np.clip(0.12 + 0.73 * norm_score, 0.0, 0.85),
            zorder=3,
            interpolation="nearest",
        )
        score_images.append(image)
        # A real reveal mask is speckled by occlusion, so fill it rather than contour it --
        # an outline of shadow speckle is noise, not information.
        reveal = np.asarray(case[f"reveal_{policy}"], bool)
        revealed = np.zeros((*reveal.shape, 4), float)
        revealed[reveal] = (0.0, 0.83, 1.0, 0.16)
        ax.imshow(revealed, origin="lower", extent=ext, zorder=5, interpolation="nearest")
        if reveal.any():
            ax.contour(
                grid_x,
                grid_y,
                _box_blur(reveal.astype(float), blur),
                levels=[0.5],
                colors="#00d4ff",
                linewidths=1.3,
                zorder=5,
            )
        vp = np.asarray(case[f"vp_{policy}"], float).reshape(-1)
        # The robot's starting pose, so the turn/creep to the chosen viewpoint is visible.
        ax.plot([pose[0]], [pose[1]], "o", ms=5.0, mfc="none", mec="#d62728", mew=1.4, zorder=6)
        _draw_fov(ax, vp, "#00d4ff", fov_radius)
        _draw_pose(ax, vp, span, "#00d4ff", zorder=7)
        _frame(
            ax,
            ext,
            f"{PANEL_TITLE[policy]}\nits look -> tau {tau[policy]:.2f}"
            f"  (before {tau_before:.2f})",
            xlabel=True,
            ylabel=col == 0,
        )

    bar = fig.colorbar(score_images[-1], ax=list(axes[1, :]), fraction=0.022, pad=0.01, shrink=0.62)
    bar.set_label("cell score (each field self-normalised)", fontsize=9)
    bar.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Uncertainty is not relevance: one look, three ways to choose it"
        f"   --   seed {label},  {traj.shape[1]} candidate plans",
        fontsize=14,
    )
    fig.text(
        0.012,
        0.004,
        "Bottom row: each score field is clipped at its own 99.5th percentile and gamma-lifted"
        f" ({SCORE_GAMMA}), so the three panels compare SHAPE, not units -- the colourbar is\n"
        f"common but the units are not. Cyan = the chosen viewpoint, a"
        f" {2 * FOV_HALF_DEG:.0f} deg / {fov_radius:.1f} m schematic of where it faces, and the"
        " shaded cells that look actually revealed;"
        " the hollow red ring is where the robot started.",
        fontsize=9,
        color="#555555",
        ha="left",
        va="bottom",
        linespacing=1.4,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"pipeline_seed{label}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------- figure 2


def figure_summary(rows: list[dict[str, object]], out_dir: Path) -> Path:
    taus = {p: np.array([float(r["policies"][p]["tau"]) for r in rows]) for p in POLICIES}
    before = np.array([float(r["tau_before"]) for r in rows])
    n_seeds = len(rows)
    means = {p: float(taus[p].mean()) for p in POLICIES}
    cis = {p: _bootstrap_ci(taus[p], seed=i) for i, p in enumerate(POLICIES)}

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True, dpi=140)
    rng = np.random.default_rng(0)

    # (a) where each policy lands after its one look -----------------------------------------
    ax = axes[0]
    for i, policy in enumerate(POLICIES):
        values = taus[policy]
        jitter = i + rng.uniform(-0.16, 0.16, values.size)
        ax.plot(jitter, values, "o", ms=3.0, mfc=COLOR[policy], mec="none", alpha=0.18, zorder=2)
        mean = means[policy]
        lo, hi = cis[policy]
        hollow = policy == "oracle"
        ax.errorbar(
            [i],
            [mean],
            yerr=[[mean - lo], [hi - mean]],
            fmt="o",
            ms=10.0 if policy in ("disagreement", "oracle") else 8.0,
            mfc="white" if hollow else COLOR[policy],
            mec=COLOR[policy],
            mew=2.0,
            ecolor=COLOR[policy],
            elinewidth=2.0,
            capsize=5.0,
            zorder=4,
        )
        ax.text(
            i,
            hi + 0.022,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=COLOR[policy],
            fontweight="bold" if policy == "disagreement" else "normal",
        )
    ax.axhline(float(before.mean()), color="#999999", ls="--", lw=1.2, zorder=1)
    ax.text(
        len(POLICIES) - 0.55,
        float(before.mean()),
        f"before looking ({before.mean():.3f})",
        fontsize=9,
        color="#777777",
        ha="right",
        va="bottom",
    )
    ax.annotate(
        "ceiling:\npeeks at truth",
        xy=(len(POLICIES) - 1.12, float(taus["oracle"].mean())),
        xytext=(len(POLICIES) - 2.1, float(taus["oracle"].mean()) + 0.14),
        fontsize=9,
        color="#2b2b2b",
        ha="center",
        va="bottom",
        arrowprops={"arrowstyle": "->", "color": "#2b2b2b", "lw": 1.0},
    )
    ax.set_xticks(range(len(POLICIES)))
    ax.set_xticklabels([LABEL[p].replace(" (", "\n(") for p in POLICIES], fontsize=9)
    for tick, policy in zip(ax.get_xticklabels(), POLICIES):
        tick.set_color(COLOR[policy])
    ax.set_ylabel("post-look Kendall tau (plan ranking vs truth)", fontsize=9.5)
    ax.set_xlim(-0.6, len(POLICIES) - 0.4)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", color="#e4e4e4", lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # The headline is read off the data, never asserted: a figure must not claim a separation
    # its own error bars deny.
    rival = max(("random", "entropy", "swath_var"), key=lambda p: means[p])
    if cis["disagreement"][0] > cis[rival][1]:
        verdict = "Decision-focused sensing beats uncertainty and geometry"
    elif means["disagreement"] > means[rival]:
        verdict = "Decision-focused sensing leads, but its CI overlaps the best baseline"
    else:
        verdict = f"Decision-focused sensing does not beat {LABEL[rival].split(' (')[0]} here"
    ax.set_title(
        f"{verdict}\nn = {n_seeds} seeds, one look each; dot = mean, bar = 95% bootstrap CI",
        fontsize=11,
    )

    # (b) the same comparison seed by seed, which is where the claim actually lives -----------
    ax = axes[1]
    pairs = (
        ("entropy", taus["disagreement"] - taus["entropy"]),
        ("swath_var", taus["disagreement"] - taus["swath_var"]),
    )
    span = float(np.abs(np.concatenate([d for _, d in pairs])).max())
    span = span if span > 0 else 0.05
    bins = np.linspace(-1.08 * span, 1.08 * span, 26)
    for baseline, diff in pairs:
        ax.hist(
            diff,
            bins=bins,
            color=COLOR[baseline],
            alpha=0.45,
            label=(
                f"vs {LABEL[baseline].split(' (')[0]}:  mean {diff.mean():+.3f},"
                f"  {100.0 * float((diff > 0).mean()):.0f}% of seeds > 0"
            ),
            zorder=2,
        )
        ax.hist(diff, bins=bins, color=COLOR[baseline], histtype="step", lw=1.6, zorder=3)
        ax.axvline(float(diff.mean()), color=COLOR[baseline], ls=":", lw=1.8, zorder=4)
    ax.axvline(0.0, color="#333333", lw=1.3, zorder=5)
    ax.set_xlabel("per-seed tau(disagreement) - tau(baseline)", fontsize=9.5)
    ax.set_ylabel("seeds", fontsize=9.5)
    ax.tick_params(labelsize=9)
    ax.set_ylim(0.0, 1.38 * ax.get_ylim()[1])  # headroom so the legend never sits on a bar
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", color="#e4e4e4", lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    won = [float((diff > 0).mean()) for _, diff in pairs]
    if min(won) >= 0.6:
        paired = "Paired per seed: the gain holds seed by seed, not just on average"
    else:
        paired = "Paired per seed: a positive mean over a spread that straddles zero"
    ax.set_title(f"{paired}\ndotted = mean difference, solid = no difference", fontsize=11)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------------- fixture


def _fractal(
    rng: np.random.Generator, shape: tuple[int, int], beta: float, scale: float
) -> np.ndarray:
    """1/f^beta noise -- a cheap stand-in for terrain while the real pipeline is being built."""
    ny, nx = shape
    freq = np.hypot(np.fft.fftfreq(ny)[:, None], np.fft.fftfreq(nx)[None, :])
    freq[0, 0] = 1.0
    amplitude = freq ** (-beta / 2.0)
    amplitude[0, 0] = 0.0
    field = np.fft.ifft2(np.fft.fft2(rng.standard_normal(shape)) * amplitude).real
    field -= field.mean()
    return scale * field / (field.std() + 1e-9)


def _kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Tau-a over K=16 items; O(K^2) is free here and keeps scipy out of the fixture."""
    n = a.size
    concordant = 0.0
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            concordant += float(s)
            total += 1.0
    return concordant / total if total else 0.0


def _fixture(root: Path, n_seeds: int = 120, seed: int = 7) -> tuple[Path, Path]:
    """Write a schema-conforming case npz + results.json of synthetic content."""
    rng = np.random.default_rng(seed)
    ny, nx = 90, 130
    cell = 0.2
    # The robot sits INSIDE the map, not at its edge: the ground behind it is unobserved too,
    # which is exactly the terrain an uncertainty-driven policy is tempted by.
    origin = np.array([-12.0, -9.0], np.float64)
    grid_x, grid_y = _centres(cell, origin, (ny, nx))
    pose = np.array([0.0, 0.0, 0.0], np.float32)

    truth = _fractal(rng, (ny, nx), 3.0, 0.28).astype(np.float32)

    # K plans: a fan of constant-curvature arcs from the robot, 14 m long.
    n_plans, n_steps = 16, 40
    curvature = np.linspace(-0.16, 0.16, n_plans)
    step = 11.0 / n_steps
    traj = np.zeros((n_steps + 1, n_plans, 2), np.float32)
    for k, kappa in enumerate(curvature):
        x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        for t in range(1, n_steps + 1):
            yaw += kappa * step
            x += step * np.cos(yaw)
            y += step * np.sin(yaw)
            traj[t, k] = (x, y)

    # Per-plan footprint kernel: how much plan k's cost can care about each cell.
    kernel = np.zeros((n_plans, ny, nx))
    for k in range(n_plans):
        d_k = np.full((ny, nx), np.inf)
        for t in range(n_steps + 1):
            d_k = np.minimum(d_k, np.hypot(grid_x - traj[t, k, 0], grid_y - traj[t, k, 1]))
        kernel[k] = np.exp(-(d_k**2) / (2.0 * 0.55**2))

    radius = np.hypot(grid_x - pose[0], grid_y - pose[1])
    bearing = np.arctan2(grid_y - pose[1], grid_x - pose[0])
    obs = (radius < 4.5) & (np.abs(np.arctan2(np.sin(bearing), np.cos(bearing))) < np.radians(55.0))
    belief = np.where(obs, truth, 0.35 * _fractal(rng, (ny, nx), 3.4, 0.28)).astype(np.float32)
    sigma = np.where(obs, 0.015, 0.05 + 0.13 * (1.0 - np.exp(-radius / 7.0)))
    # A patch behind the robot that was never observed at all: genuinely the least certain
    # ground on the map, and completely irrelevant to which plan wins.
    sigma *= 1.0 + 1.6 * np.exp(-((grid_x + 6.0) ** 2 + (grid_y - 4.0) ** 2) / (2.0 * 3.0**2))
    sigma *= 1.0 + 0.5 * np.clip(_fractal(rng, (ny, nx), 2.6, 1.0), -0.6, 1.6)
    sigma = np.abs(sigma).astype(np.float32)
    meas = (truth + rng.normal(0.0, 0.02, (ny, nx))).astype(np.float32)

    # The three fields, each built from its own stated definition so the panels differ for the
    # right reason: entropy ignores the plans, swath knows only geometry, ours weights the
    # geometry by how much the COST there moves (a stand-in for dJ_k/dh).
    grad_y, grad_x = np.gradient(belief.astype(np.float64), cell)
    roughness = np.hypot(grad_x, grad_y)
    roughness /= np.percentile(roughness, 98.0) + 1e-9
    sensitivity = kernel * (0.25 + np.clip(roughness, 0.0, 2.0))[None, :, :]

    score_entropy = (sigma.astype(np.float64) ** 2 * (~obs)).astype(np.float32)
    score_swath = (kernel.var(axis=0) * (~obs)).astype(np.float32)
    score_disagreement = (sensitivity.var(axis=0) * sigma.astype(np.float64) ** 2 * (~obs)).astype(
        np.float32
    )

    # V candidate viewpoints: turn in place, or creep a little first. The headings must spread
    # wider than the FOV or every policy ends up looking at the same wedge.
    vp_yaw = np.linspace(-2.4, 2.4, 12)
    stations = ((0.0, 0.0), (2.6, 0.0))
    viewpoints = np.array(
        [(sx, sy, float(yaw)) for sx, sy in stations for yaw in vp_yaw], np.float32
    )
    n_vp = viewpoints.shape[0]

    def wedge_mask(vp: np.ndarray) -> np.ndarray:
        rel = np.arctan2(grid_y - vp[1], grid_x - vp[0]) - vp[2]
        rel = np.arctan2(np.sin(rel), np.cos(rel))
        reach = np.hypot(grid_x - vp[0], grid_y - vp[1])
        return (reach < FOV_RANGE_M) & (np.abs(rel) < np.radians(FOV_HALF_DEG)) & (~obs)

    def best_viewpoint(score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gains = [float(score[wedge_mask(v)].sum()) for v in viewpoints]
        pick = viewpoints[int(np.argmax(gains))]
        return pick, wedge_mask(pick)

    vp_entropy, reveal_entropy = best_viewpoint(score_entropy)
    vp_swath, reveal_swath = best_viewpoint(score_swath)
    vp_disagreement, reveal_disagreement = best_viewpoint(score_disagreement)
    vp_random = viewpoints[int(rng.integers(n_vp))]

    belief_after = belief.copy()
    belief_after[reveal_disagreement] = meas[reveal_disagreement]

    def cost(height: np.ndarray) -> np.ndarray:
        out = np.zeros(n_plans)
        for k in range(n_plans):
            iy = np.clip(((traj[:, k, 1] - origin[1]) / cell).astype(int), 0, ny - 1)
            ix = np.clip(((traj[:, k, 0] - origin[0]) / cell).astype(int), 0, nx - 1)
            out[k] = np.abs(np.diff(height[iy, ix])).sum() + 0.4 * np.abs(height[iy, ix]).sum()
        return out

    j_true = cost(truth).astype(np.float32)
    j_belief = cost(belief).astype(np.float32)
    tau_before = _kendall_tau(j_belief, j_true)

    # FABRICATED, and deliberately so: each policy's look drags every believed cost a fixed
    # fraction of the way to its true cost, with the fraction ordered the way the experiment is
    # expected to come out. Reproducing the actual result from a toy cost is not this file's job
    # -- the fixture exists to exercise the drawing code against the frozen schema.
    correction = {
        "none": 0.0,
        "random": 0.22,
        "entropy": 0.30,
        "swath_var": 0.55,
        "disagreement": 0.82,
        "oracle": 0.95,
    }
    jafter = {
        name: (j_belief + a * (j_true - j_belief)).astype(np.float32)
        for name, a in correction.items()
    }
    taus = {name: _kendall_tau(j, j_true) for name, j in jafter.items()}

    # The oracle's look is the wedge covering the most actual belief error under the plans.
    error_under_plans = np.abs(truth - belief) * kernel.max(axis=0) * (~obs)
    vp_oracle, _ = best_viewpoint(error_under_plans)

    case = {
        "truth": truth,
        "belief": belief,
        "belief_after_disagreement": belief_after,
        "sigma": sigma,
        "meas": meas,
        "obs": obs,
        "cell": np.float64(cell),
        "origin": origin,
        "robot_pose": pose,
        "traj": traj,
        "j_true": j_true,
        "j_belief": j_belief,
        "score_entropy": score_entropy,
        "score_swath_var": score_swath,
        "score_disagreement": score_disagreement,
        "vp_entropy": vp_entropy,
        "vp_swath_var": vp_swath,
        "vp_disagreement": vp_disagreement,
        "vp_random": vp_random.astype(np.float32),
        "vp_oracle": vp_oracle,
        "reveal_entropy": reveal_entropy,
        "reveal_swath_var": reveal_swath,
        "reveal_disagreement": reveal_disagreement,
        "viewpoints": viewpoints,
        "tau_before": np.float64(tau_before),
    }
    for name, j in jafter.items():
        case[f"jafter_{name}"] = j
    for name, value in taus.items():
        case[f"tau_{name}"] = np.float64(value)

    root.mkdir(parents=True, exist_ok=True)
    case_path = root / f"case_seed{seed}.npz"
    np.savez_compressed(case_path, **case)

    # results.json: the same story repeated over many seeds, with realistic spread/overlap.
    gain = {
        "none": 0.0,
        "random": 0.04,
        "entropy": 0.09,
        "swath_var": 0.16,
        "disagreement": 0.27,
        "oracle": 0.38,
    }
    noise = {
        "none": 0.0,
        "random": 0.10,
        "entropy": 0.09,
        "swath_var": 0.08,
        "disagreement": 0.07,
        "oracle": 0.04,
    }
    rows: list[dict[str, object]] = []
    rrng = np.random.default_rng(1234)
    for s in range(n_seeds):
        base = float(np.clip(rrng.normal(0.42, 0.12), -0.2, 0.85))
        row_policies: dict[str, dict[str, object]] = {}
        for policy in POLICIES:
            value = float(
                np.clip(base + rrng.normal(gain[policy], noise[policy]) * (1.0 - base), -1.0, 1.0)
            )
            row_policies[policy] = {
                "tau": value,
                "regret": float(max(0.0, rrng.normal(0.45 - 0.9 * gain[policy], 0.2))),
                "top1": bool(rrng.random() < 0.35 + gain[policy]),
                "time_cost": 0.0 if policy == "none" else float(rrng.uniform(1.5, 4.0)),
                "n_revealed": 0 if policy == "none" else int(rrng.integers(180, 420)),
                "gain": value - base,
            }
        rows.append({"seed": s, "tau_before": base, "policies": row_policies})
    results_path = root / "results.json"
    results_path.write_text(json.dumps({"rows": rows}, indent=1))
    return case_path, results_path


# -------------------------------------------------------------------------------------- cli


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", type=Path, default=None, help="case_seed{N}.npz")
    parser.add_argument("--results", type=Path, default=OUT / "results.json")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="write synthetic schema-conforming inputs into this dir and plot those",
    )
    args = parser.parse_args()

    case_path: Path | None = args.case
    results_path: Path = args.results
    if args.fixture is not None:
        case_path, results_path = _fixture(args.fixture)
        print(f"fixture: {case_path}  {results_path}")
    if case_path is None:
        found = sorted(OUT.glob("case_seed*.npz"))
        case_path = found[0] if found else None

    if case_path is not None and case_path.exists():
        path = figure_pipeline(_load_case(case_path), args.out, _seed_label(case_path))
        print(f"wrote {path}")
    else:
        print(f"no case npz (looked for {case_path or OUT / 'case_seed*.npz'}) -- skipped figure 1")

    if results_path.exists():
        print(f"wrote {figure_summary(_load_results(results_path), args.out)}")
    else:
        print(f"no {results_path} -- skipped figure 2")


if __name__ == "__main__":
    main()
