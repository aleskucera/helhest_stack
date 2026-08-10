"""Paper figures for the Clark/SSTA moment-propagation paper (IEEEtran, single column).

    .venv/bin/python -m studies.bench.plot_paper --out paper/figures

Two figures, both COMPUTED from the same objects the estimator uses (the real wheel-footprint
offset table, the real correlation kernel, the committed budget-curve json) rather than drawn as
schematics -- a teaching figure that is also a validation is worth twice as much, and a
schematic that disagrees with the code is a liability.

  jensen_explainer.png  Fig. 1. (a) the max over one wheel footprint at a realistic sigma: the
                        belief map evaluates the max of the means, the truth is the mean of the
                        max, and the gap is the optimism a mean-map planner pays. (b) that gap
                        vs sigma, for the real correlated kernel and for the independent-cell
                        model of the same sigma, with Clark's closed form overlaid on Monte
                        Carlo -- correlation suppresses the inflation several-fold, which is why
                        an estimator that keeps only per-cell (mean, sigma) cannot get it right.
  geometry.png          Fig. 1. the objects Section II names but cannot draw in equations: the
                        three wheel footprints on the grid, the K candidate cells under one of
                        them, the spherical-cap offsets that lift each candidate, and the
                        tripod the settle rests on. Drawn from RobotParams, not by hand.
  budget_curve.png      Fig. 3. matched-budget MC regret vs number of draws, against the
                        analytic estimator's regret, under the measured belief
                        (studies/out/bench/realistic_sigma_hybrid_all.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .clark import _norm_cdf  # noqa: E402
from .clark import _norm_pdf  # noqa: E402
from .clark import rho1_table  # noqa: E402
from .clark_conv import cylinder_offsets  # noqa: E402
from .clark_conv import WHEEL_HALF_WIDTH  # noqa: E402
from .clark import rho_lookup  # noqa: E402
from .ranking import CELL  # noqa: E402
from .ranking import OUT  # noqa: E402
from .risk import CORR_LEN  # noqa: E402
from helhest.engine import RobotParams  # noqa: E402
from helhest.engine.envelope import wheel_offset_table  # noqa: E402

FIG_W = 3.4  # [in] IEEEtran single column
DPI = 300
RNG = np.random.default_rng(7)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8,
            "legend.fontsize": 6.8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.3,
            "figure.dpi": DPI,
        }
    )


def real_footprints(n_foot: int = 24) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """`n_foot` REAL wheel footprints drawn from the benchmark's own belief map: candidate means
    (belief + cap), per-candidate sigma from the study's sigma field, and the kernel correlation
    matrix of their cells. Flat synthetic footprints are the adversarial extreme for a
    moment-matching fold (every candidate tied); the paper's numbers live on terrain, so the
    figure must too."""
    import warp as wp

    from .ranking import build_case

    wp.init()
    rp = RobotParams()
    radius_cells = int(np.ceil(rp.wheel_radius / CELL))
    off_dy, off_dx, off_cap = wheel_offset_table(radius_cells, CELL, rp.wheel_radius)
    table = rho1_table(CORR_LEN, CELL)
    rho = rho_lookup(table, off_dy[:, None] - off_dy[None, :], off_dx[:, None] - off_dx[None, :])

    scene, _, _, _, sigma, _, _, _ = build_case(0, "hybrid", "all")
    belief = np.asarray(scene.elevation, np.float64)
    sig = np.asarray(sigma, np.float64)
    ny, nx = belief.shape
    margin = radius_cells + 2
    out = []
    for _ in range(n_foot):
        iy = RNG.integers(margin, ny - margin)
        ix = RNG.integers(margin, nx - margin)
        cy, cx = iy + off_dy, ix + off_dx
        out.append((belief[cy, cx] + np.asarray(off_cap, np.float64), sig[cy, cx], rho))
    return out


def clark_max_moments(mu: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    """Clark's recursion over K candidates, descending mean order (the same fold clark.py runs,
    reimplemented here in scalar form so the figure does not depend on the batched plumbing)."""
    order = np.argsort(-mu)
    mu, cov = mu[order], cov[np.ix_(order, order)]
    m, v = mu[0], cov[0, 0]
    c = cov[0].copy()  # cov(running max, each original candidate)
    for i in range(1, len(mu)):
        a2 = max(v + cov[i, i] - 2.0 * c[i], 1e-18)
        a = np.sqrt(a2)
        alpha = (m - mu[i]) / a
        pa, pn, da = _norm_cdf(alpha), 1.0 - _norm_cdf(alpha), _norm_pdf(alpha)
        m_new = m * pa + mu[i] * pn + a * da
        ex2 = (m * m + v) * pa + (mu[i] ** 2 + cov[i, i]) * pn + (m + mu[i]) * a * da
        c = c * pa + cov[i] * pn
        m, v = m_new, max(ex2 - m_new**2, 0.0)
    return float(m), float(v)


def _mc_max(mu: np.ndarray, sd: np.ndarray, chol: np.ndarray, n_draws: int) -> np.ndarray:
    z = RNG.standard_normal((n_draws, len(mu)))
    return (mu[None] + sd[None] * (z @ chol.T)).max(axis=1)


def fig_jensen(path: Path) -> None:
    foots = real_footprints()
    k = len(foots[0][0])
    scales = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0])  # multiples of the map's own sigma field
    n_draws = 20_000
    chol = np.linalg.cholesky(foots[0][2] + 1e-9 * np.eye(k))
    eye = np.eye(k)

    sig_axis, gap_corr, gap_indep, gap_clark = [], [], [], []
    for sc in scales:
        gc, gi, gk, sbar = [], [], [], []
        for mu, sd0, rho in foots:
            sd = sc * sd0
            base = mu.max()
            gc.append(_mc_max(mu, sd, chol, n_draws).mean() - base)
            gi.append(_mc_max(mu, sd, eye, n_draws).mean() - base)
            gk.append(clark_max_moments(mu, np.outer(sd, sd) * rho)[0] - base)
            sbar.append(sd.mean())
        sig_axis.append(np.mean(sbar))
        gap_corr.append(np.mean(gc))
        gap_indep.append(np.mean(gi))
        gap_clark.append(np.mean(gk))
    sig_axis = np.array(sig_axis)

    # panel (a): one representative footprint at the map's own sigma
    mu, sd, rho = foots[0]
    draws = _mc_max(mu, sd, chol, 60_000)
    e_clark = clark_max_moments(mu, np.outer(sd, sd) * rho)[0]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FIG_W * 2.06, 1.75))

    ax_a.hist(draws, bins=70, color="#c6dbef", edgecolor="none", density=True)
    top = ax_a.get_ylim()[1]
    ax_a.set_xlim(mu.max() - 4.2 * sd.mean(), mu.max() + 5.0 * sd.mean())
    ax_a.axvline(mu.max(), color="#252525", ls="--", lw=1.2)
    ax_a.axvline(float(draws.mean()), color="#cb181d", lw=1.4)
    ax_a.annotate(
        "", xy=(float(draws.mean()), top * 0.62), xytext=(mu.max(), top * 0.62),
        arrowprops=dict(arrowstyle="<->", color="#cb181d", lw=0.9),
    )
    ax_a.text(float(draws.mean()), top * 0.66,
              f"  optimism {100*(draws.mean()-mu.max()):.1f} cm", ha="left", va="bottom",
              fontsize=6.6, color="#cb181d")
    ax_a.annotate(
        "belief map\n$\\max\\,\\mathbb{E}[h]$",
        xy=(mu.max(), top * 0.30), xytext=(mu.max() - 3.4 * sd.mean(), top * 0.42),
        fontsize=6.6, ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color="#525252", lw=0.7),
    )
    ax_a.plot([e_clark], [top * 0.045], marker="^", color="k", ms=5, ls="none", clip_on=False)
    ax_a.text(e_clark, top * 0.10, "Clark", fontsize=6.6, ha="center", va="bottom")
    ax_a.set_xlabel("wheel support height [m]")
    ax_a.set_ylabel("density")
    ax_a.set_yticks([])
    ax_a.set_title(f"(a) one real footprint, $K={k}$", loc="left")

    # (b) what the correlation actually is. The independent-vs-correlated point this panel
    # used to make is subsumed: the measured kernel IS the answer to "does correlation matter".
    import json as _json

    rho_tot = np.load(OUT / "rho_measured.npy")
    model = np.load(OUT / "belief_model.npz")
    rho_s = model["rho_stationary"]
    share = float(model["plane_share"])
    mid = rho_tot.shape[0] // 2
    lag = np.arange(mid + 1) * CELL
    assumed = rho1_table(CORR_LEN, CELL)
    ax_b.plot(lag, rho_tot[mid, mid:] / rho_tot[mid, mid], "-o", ms=2.5, color="#cb181d",
              label="measured, total")
    ax_b.plot(lag, rho_s[mid, mid:] / rho_s[mid, mid], "-o", ms=2.5, color="#2171b5",
              label="measured, after removing the plane")
    ax_b.plot(np.arange(len(assumed)) * CELL, assumed, "k--", label="assumed by prior work")
    ax_b.axhline(0.0, color="#bdbdbd", lw=0.6)
    ax_b.text(0.62, 0.42, f"a rank-3 plane\ncarries {share:.0%} of the\nvariance and never\ndecorrelates",
              fontsize=6.3, color="#cb181d", va="center")
    ax_b.set_xlabel("lag [m]")
    ax_b.set_ylabel(r"$\rho$")
    ax_b.set_xlim(0, 1.2)
    ax_b.legend(frameon=False, loc="upper right")
    ax_b.set_title("(b) the correlation the sensing produces", loc="left")

    fig.tight_layout(pad=0.35)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_geometry(path: Path) -> None:
    """The contact geometry, drawn from the robot's own parameters: (a) plan view of the three
    wheel footprints on the 0.10 m grid with one footprint's K candidate cells picked out;
    (b) section through that wheel, showing the cap offsets kappa_c that lift each candidate and
    the max that selects the support height."""
    rp = RobotParams()
    cell = CELL
    radius_cells = int(np.ceil(rp.wheel_radius / cell))
    off_dy, off_dx, off_cap = wheel_offset_table(radius_cells, cell, rp.wheel_radius)
    off_dy, off_dx = np.asarray(off_dy), np.asarray(off_dx)
    wheels = np.array([[0.0, rp.half_track], [0.0, -rp.half_track], [-rp.rear_offset, 0.0]])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(FIG_W * 2.06, 1.85))

    # --- (a) plan view ------------------------------------------------------------------
    # The wheels are drawn as RECTANGLES because that is what a cylinder looks like from
    # above: 2R along travel by the tread width across. A circle of radius R here would be
    # the SPHERE's silhouette, which is the envelope the experiments dilate with but not the
    # shape of the wheel. Both candidate sets are shown, because the paper uses the sphere's
    # for its results and the cylinder's for its cost figures.
    cyl_dy, cyl_dx, _ = cylinder_offsets(cell, rp.wheel_radius, WHEEL_HALF_WIDTH, 0.0)
    sphere_set = {(int(a), int(b)) for a, b in zip(off_dy, off_dx)}
    cyl_set = {(int(a), int(b)) for a, b in zip(cyl_dy, cyl_dx)}

    lim_x, lim_y = (-1.25, 0.55), (-0.85, 0.85)
    for gx in np.arange(-1.3, 0.65, cell):
        ax_a.axvline(gx, color="#e8e8e8", lw=0.3, zorder=0)
    for gy in np.arange(-0.9, 0.95, cell):
        ax_a.axhline(gy, color="#e8e8e8", lw=0.3, zorder=0)
    for cx, hx, hy in ((-0.13, 0.24, 0.28), (-0.61, 0.24, 0.12)):
        ax_a.add_patch(
            plt.Rectangle((cx - hx, -hy), 2 * hx, 2 * hy, fill=False, ec="#737373", lw=0.9,
                          zorder=4)
        )
    for i, j in ((0, 1), (0, 2), (1, 2)):
        ax_a.plot(wheels[[i, j], 0], wheels[[i, j], 1], color="#252525", lw=0.8, ls=":",
                  zorder=5)
    for i, (wx, wy) in enumerate(wheels):
        highlight = i == 0
        wx = round(wx / cell) * cell
        wy = round(wy / cell) * cell
        for dyc, dxc in zip(off_dy, off_dx):
            in_cyl = (int(dyc), int(dxc)) in cyl_set
            if highlight:
                fc = "#e6550d" if in_cyl else "#fdd0a2"
            else:
                fc = "#9ecae1" if in_cyl else "#deebf7"
            ax_a.add_patch(
                plt.Rectangle((wx + dxc * cell - cell / 2, wy + dyc * cell - cell / 2),
                              cell, cell, fc=fc, ec="#bdbdbd", lw=0.25, zorder=1)
            )
        # the wheel itself: a cylinder seen from above. Outline only -- a filled patch
        # would cover the cells the panel exists to show.
        ax_a.add_patch(
            plt.Rectangle((wx - rp.wheel_radius, wy - WHEEL_HALF_WIDTH),
                          2 * rp.wheel_radius, 2 * WHEEL_HALF_WIDTH,
                          fill=False, ec="#252525", lw=1.2, zorder=6)
        )
    ax_a.annotate("cylinder, $K=7$", xy=(0.12, 0.40), xytext=(-0.32, 0.72),
                  fontsize=6.5, ha="center", color="#a63603",
                  arrowprops=dict(arrowstyle="->", color="#a63603", lw=0.7))
    ax_a.annotate("sphere envelope, $K=37$", xy=(0.22, 0.10), xytext=(0.02, -0.72),
                  fontsize=6.5, ha="center", color="#8c6d31",
                  arrowprops=dict(arrowstyle="->", color="#8c6d31", lw=0.7))
    ax_a.text(-0.44, 0.14, "tripod", fontsize=6.5, color="#252525", ha="center")
    ax_a.set_xlim(*lim_x)
    ax_a.set_ylim(*lim_y)
    ax_a.set_aspect("equal")
    ax_a.set_xlabel("$x$ [m]")
    ax_a.set_ylabel("$y$ [m]")
    ax_a.set_title("(a) two contact elements on the belief grid", loc="left")

    # --- (b) section through the highlighted wheel ------------------------------------------
    mid = off_dy == 0
    xs = off_dx[mid] * cell
    caps = off_cap[mid]
    rng = np.random.default_rng(3)
    ground = 0.035 * np.sin(xs * 11.0) + 0.02 * rng.standard_normal(len(xs))
    lifted = ground + caps
    j = int(np.argmax(lifted))
    fine = np.linspace(xs.min() - 0.02, xs.max() + 0.02, 200)
    ax_b.fill_between(fine, -0.25, np.interp(fine, xs, ground), color="#d9d9d9", lw=0)
    ax_b.plot(xs, ground, "o-", color="#525252", ms=2.5, lw=0.9, label="belief $h_c$")
    for x, g, l in zip(xs, ground, lifted):
        ax_b.plot([x, x], [g, l], color="#fd8d3c", lw=0.8)
    ax_b.plot(xs, lifted, "s", color="#a63603", ms=2.6, label="$h_c + \\kappa_c$")
    ax_b.annotate("$\\kappa_c \\leq 0$: how far this cell\nsits from the wheel's contact",
                  xy=(xs[-1], 0.5 * (ground[-1] + lifted[-1])),
                  xytext=(xs[2], lifted.min() - 0.005),
                  fontsize=6.3, color="#a63603", ha="left", va="bottom",
                  arrowprops=dict(arrowstyle="->", color="#a63603", lw=0.7))
    ax_b.axhline(lifted[j], color="#cb181d", lw=1.2)
    ax_b.plot([xs[j]], [lifted[j]], marker="*", ms=8, color="#cb181d", zorder=5)
    ax_b.text(xs.min(), lifted[j] + 0.028, "$e_{w,t} = \\max_c\\,(h_c + \\kappa_c)$",
              fontsize=6.8, color="#cb181d", ha="left", va="bottom")
    ax_b.set_xlabel("distance along travel [m]")
    ax_b.set_ylabel("height [m]")
    ax_b.set_ylim(min(ground.min(), lifted.min()) - 0.055, lifted[j] + 0.075)
    ax_b.legend(frameon=False, loc="upper right", ncol=1, handletextpad=0.4,
                borderaxespad=0.2)
    ax_b.set_title("(b) section along travel: identical for both", loc="left")

    fig.tight_layout(pad=0.35)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_budget(path: Path) -> None:
    # clark_full.json's budget curve came from the old hand-built NoiseDraws kernel; the
    # paper's regrets now come from realistic_sigma.py's measured belief (BeliefModel), so the
    # figure has to read that file's budget_curve/n_star instead or the curve and the Clark
    # line would be drawn from two different noise models.
    data = json.loads((OUT / "realistic_sigma_hybrid_all.json").read_text())
    n = np.array([c["n"] for c in data["budget_curve"]])
    regret = np.array([c["mean_regret"] for c in data["budget_curve"]])
    clark = data["mean_regret"]["clark_cvar"]

    fig, ax = plt.subplots(figsize=(FIG_W, 1.9))
    ax.plot(n, regret, "o-", color="#2171b5", label="Monte-Carlo with $N$ draws", ms=3.5)
    ax.axhline(clark, color="#cb181d", lw=1.3, label=f"Clark (no draws): {clark:.3f}")
    ax.axvline(data["n_star"], color="#969696", ls=":", lw=1.0)
    ax.text(data["n_star"] * 1.06, max(regret) * 0.55, f"$N^* = {data['n_star']}$",
            fontsize=6.8, color="#525252")
    ax.set_xscale("log", base=2)
    ax.set_xticks(n)
    ax.set_xticklabels([str(v) for v in n])
    ax.set_xlabel("Monte-Carlo draws per plan")
    ax.set_ylabel("mean regret")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT), help="directory for the .png files")
    args = ap.parse_args()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    _style()
    fig_geometry(out / "geometry.png")
    fig_jensen(out / "jensen_explainer.png")
    fig_budget(out / "budget_curve.png")


if __name__ == "__main__":
    main()
