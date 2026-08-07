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
  budget_curve.png      Fig. 2. matched-budget MC regret vs number of draws, against the
                        analytic estimator's regret (studies/out/bench/clark_full.json).
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

    ax_b.plot(sig_axis * 100, np.array(gap_indep) * 100, "o-", color="#6baed6",
              label="independent cells (MC)", ms=3)
    ax_b.plot(sig_axis * 100, np.array(gap_corr) * 100, "o-", color="#cb181d",
              label="correlated (MC truth)", ms=3)
    ax_b.plot(sig_axis * 100, np.array(gap_clark) * 100, "k--", label="Clark, closed form")
    # a linearised estimator has no Jensen term at all -- the honest comparison for panel (b)
    # is not "Clark vs exact" but "Clark vs the zero that first-order propagation predicts".
    ax_b.axhline(0.0, color="#525252", lw=1.0, ls="-.", label="first-order: no gap at all")
    i_nom = int(np.argmin(np.abs(scales - 1.0)))
    ax_b.axvline(sig_axis[i_nom] * 100, color="#969696", ls=":", lw=1.0)
    ax_b.text(sig_axis[i_nom] * 100 * 1.03, max(gap_indep) * 100 * 0.15,
              "this map's $\\sigma$", fontsize=6.5, color="#525252")
    ax_b.set_xlabel("mean per-cell $\\sigma$ under the footprint [cm]")
    ax_b.set_ylabel("$\\mathbb{E}[\\max] - \\max \\mathbb{E}$ [cm]")
    ax_b.legend(frameon=False, loc="upper left")
    ax_b.set_title("(b) averaged over 24 footprints", loc="left")

    fig.tight_layout(pad=0.35)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    ratio = np.array(gap_clark) / np.array(gap_corr)
    print(f"wrote {path}")
    print("  Clark/MC gap ratio per sigma scale:", np.round(ratio, 3))
    print("  independent/correlated inflation:",
          np.round(np.array(gap_indep) / np.array(gap_corr), 2))


def fig_budget(path: Path) -> None:
    data = json.loads((OUT / "clark_full.json").read_text())["budget"]
    n = np.array([c["n"] for c in data["curve"]])
    regret = np.array([c["mean_regret"] for c in data["curve"]])
    clark = data["clark_cvar_regret"]

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
    fig_jensen(out / "jensen_explainer.png")
    fig_budget(out / "budget_curve.png")


if __name__ == "__main__":
    main()
