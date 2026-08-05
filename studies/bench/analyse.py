"""Paired analysis and the benchmark figure.

    .venv/bin/python -m studies.bench.analyse

The scenario is BIMODAL by construction -- half the seeds are hard (gap on the far side /
corridor blocked) and half are easy -- so the median lands inside one cluster and reports
neither. Expected time-to-goal over the seed mixture is the mean, and that is what a robot
running this policy repeatedly actually experiences. Both are shown, with the regimes split
out, because a summary that hides a 2x regime difference is not a summary.

Paired throughout: every policy sees the same seeds, so per-seed differences cancel the
scenario variation and a sign test on those differences is the honest small-n statement. With
n = 12 nothing here is asymptotic; the sign test is reported as a count, not a p-value dressed
up as significance.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "studies" / "out" / "bench"
ARMS = ("none", "sigma", "entropy", "cvar", "attribution")
COLOR = {
    "none": "#9aa5b1",
    "sigma": "#7a5aa8",
    "entropy": "#2f7ec4",
    "cvar": "#4c9a6a",
    "attribution": "#d1495b",
    "oracle": "#333333",
}
VARIANTS = {"gap": "A: aperture (gap in a wall)", "corridor": "B: opaque (corridor floor)"}


def sign_test(d: np.ndarray) -> tuple[int, int, float]:
    """Two-sided sign test on paired differences. Ties are dropped, as the test requires."""
    nz = d[d != 0]
    n, k = len(nz), int((nz < 0).sum())
    if n == 0:
        return 0, 0, 1.0
    tail = sum(comb(n, i) for i in range(min(k, n - k) + 1))
    return n, k, min(1.0, 2.0 * tail / 2**n)


def load(variant: str) -> list[dict]:
    return json.load(open(OUT / f"results_{variant}.json"))["rows"]


def analyse(variant: str) -> dict:
    rows = load(variant)
    # The regime split is DESCRIPTIVE and post-hoc: seeds where the null baseline actually ran
    # long. Splitting by seed parity (which side the gap is on) was the design intent, but at
    # larger n it stops predicting difficulty -- the exact gap position and approach angle
    # matter too, and some far-side seeds turn out easy. Labelling by the observed outcome is
    # honest about what the two clusters ARE; it is not a selection rule for the headline
    # numbers, which are computed over all seeds.
    orc0 = np.array([r["oracle"]["time"] for r in rows], float)
    nul0 = np.array([r["none"]["time"] for r in rows], float)
    hard = nul0 > 1.5 * orc0
    t = {a: np.array([r[a]["time"] for r in rows], float) for a in ARMS}
    t["oracle"] = np.array([r["oracle"]["time"] for r in rows], float)

    print(
        f"\n=== variant {variant}: {VARIANTS[variant]}  (n={len(rows)}, "
        f"{int(hard.sum())} hard / {int((~hard).sum())} easy) ==="
    )
    print(
        f"{'policy':<13}{'mean':>7}{'median':>8}{'hard':>7}{'easy':>7}"
        f"{'vs none':>9}{'headroom':>10}{'@crit':>7}{'@decoy':>8}{'reach':>8}"
    )
    res = {}
    for a in (*ARMS, "oracle"):
        v = t[a]
        extra = ""
        if a != "oracle":
            gap = t["none"].mean() - t["oracle"].mean()
            extra = (
                f"{v.mean() - t['none'].mean():>+9.0f}{(t['none'].mean() - v.mean()) / gap:>9.0%}"
            )
            crit = np.mean([r[a]["look_at_gap"] for r in rows])
            dec = np.mean([r[a]["look_at_decoy"] for r in rows])
            reach = sum(r[a]["reached"] for r in rows)
            extra += f"{crit:>7.2f}{dec:>8.2f}{reach:>5}/{len(rows):<3}"
        print(
            f"{a:<13}{v.mean():>7.0f}{np.median(v):>8.0f}"
            f"{v[hard].mean():>7.0f}{v[~hard].mean():>7.0f}{extra}"
        )
        res[a] = {
            "mean": float(v.mean()),
            "median": float(np.median(v)),
            "hard_mean": float(v[hard].mean()),
            "easy_mean": float(v[~hard].mean()),
        }

    print("\npaired, attribution minus X (negative = attribution faster)")
    for a in ARMS:
        if a == "attribution":
            continue
        d = t["attribution"] - t[a]
        n, k, p = sign_test(d)
        print(
            f"  vs {a:<12} mean {d.mean():>+7.0f}  faster on {k:>2}/{n:<2} non-tied seeds"
            f"   sign test p={p:.3f}"
        )
        res[f"paired_vs_{a}"] = {"mean_delta": float(d.mean()), "wins": k, "n": n, "p": p}
    return res


def figure(path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    data = {v: load(v) for v in VARIANTS}

    # (a) mean time by policy, per variant
    ax = axes[0]
    width = 0.36
    for i, (v, rows) in enumerate(data.items()):
        for j, a in enumerate(ARMS):
            t = np.array([r[a]["time"] for r in rows], float)
            ax.bar(
                j + (i - 0.5) * width,
                t.mean(),
                width * 0.9,
                color=COLOR[a],
                alpha=1.0 if i else 0.55,
                edgecolor="k",
                linewidth=0.4,
            )
        orc = np.mean([r["oracle"]["time"] for r in rows])
        ax.axhline(orc, color="k", ls=":" if i else "--", lw=1.2)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels(ARMS, rotation=20, ha="right")
    ax.set_ylabel("mean time to goal [frames]")
    ax.set_title("(a) faded = A aperture, solid = B opaque\ndashed/dotted = oracle")
    ax.grid(alpha=0.25, axis="y")

    # (b) the point of the whole benchmark: entropy vs attribution, per variant
    ax = axes[1]
    for i, (v, rows) in enumerate(data.items()):
        base = np.mean([r["none"]["time"] for r in rows])
        for j, a in enumerate(("entropy", "attribution")):
            t = np.mean([r[a]["time"] for r in rows])
            ax.bar(
                i + (j - 0.5) * 0.35,
                t - base,
                0.32,
                color=COLOR[a],
                edgecolor="k",
                lw=0.4,
                label=a if i == 0 else None,
            )
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels([v.split(":")[0] + "\n" + v.split("(")[1][:-1] for v in VARIANTS.values()])
    ax.set_ylabel("mean time vs never looking [frames]")
    ax.set_title("(b) below zero = sensing paid for itself")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    # (c) where the looks went
    ax = axes[2]
    for i, (v, rows) in enumerate(data.items()):
        for j, a in enumerate(("entropy", "attribution")):
            crit = np.mean([r[a]["look_at_gap"] for r in rows])
            dec = np.mean([r[a]["look_at_decoy"] for r in rows])
            x = i * 2.4 + j
            ax.bar(x, crit, 0.8, color=COLOR[a], edgecolor="k", lw=0.4)
            ax.bar(x, -dec, 0.8, color=COLOR[a], alpha=0.45, edgecolor="k", lw=0.4)
            ax.text(x, crit + 0.05, a[:4], ha="center", fontsize=8)
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xticks([0.5, 2.9])
    ax.set_xticklabels(["A aperture", "B opaque"])
    ax.set_ylabel("looks at critical cells  /  at the decoy")
    ax.set_title("(c) up = decision-critical, down = decoy")
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle(
        "Section-6 benchmark: decision-focused vs information-theoretic sensing", fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    res = {v: analyse(v) for v in VARIANTS}
    figure(OUT / "benchmark.png")
    (OUT / "analysis.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / 'benchmark.png'} and analysis.json")


if __name__ == "__main__":
    main()
