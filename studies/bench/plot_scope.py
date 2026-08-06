"""The scope-law figure: when does sensing need a derivative?

    .venv/bin/python -m studies.bench.plot_scope

Three panels from the fixed-sigma reruns (ranking_hybrid{,_all}.json, n=200 each):
(a) clean and (b) full-noise tau-vs-budget curves for entropy / corridor_mi / disagreement,
with the oracle as the ceiling; (c) the law itself — the adjoint's edge over corridor
masking against the headroom corridor masking leaves below the oracle. Floor compression
(tiny budgets: nothing helps) and ceiling compression (large budgets or heavy noise:
masking saturates the achievable) both kill the headroom, and with it the derivative's
value; in between the adjoint captures a large fraction of what masking leaves on the
table. The motion-coupled wide-cone result (holdout.json) is the ceiling case at sensor
scale and is annotated on (c) for the connection.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("studies/out/bench")
BUDGETS = (25, 100, 400)
POLICIES = ("entropy", "corridor_mi", "disagreement", "oracle")
COLORS = {
    "entropy": "tab:purple",
    "corridor_mi": "tab:green",
    "disagreement": "tab:orange",
    "oracle": "0.3",
}
LABELS = {
    "entropy": "entropy (uncertainty)",
    "corridor_mi": "corridor-masked MI (no derivative)",
    "disagreement": "adjoint disagreement (ours)",
    "oracle": "oracle (peeks at truth)",
}


def _means(rows: list[dict], policy: str) -> list[float]:
    return [float(np.mean([r["policies"][policy][str(b)]["tau"] for r in rows])) for b in BUDGETS]


def main() -> None:
    arms = {
        "clean": json.load(open(OUT / "ranking_hybrid.json"))["rows"],
        "full noise": json.load(open(OUT / "ranking_hybrid_all.json"))["rows"],
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)

    for ax, (arm, rows) in zip(axes[:2], arms.items()):
        for p in POLICIES:
            style = dict(ls="--", marker="o", mfc="white") if p == "oracle" else dict(marker="o")
            ax.plot(BUDGETS, _means(rows, p), color=COLORS[p], label=LABELS[p], **style)
        ax.set_xscale("log")
        ax.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("look budget [cells]")
        ax.set_title(f"{arm}  (n=200)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Kendall tau of re-ranked plans vs truth")
    axes[0].legend(fontsize=8, loc="upper left")

    # (c) the law: edge over masking vs the headroom masking leaves below the oracle
    ax = axes[2]
    for arm, rows in arms.items():
        cor, dis, orc = (_means(rows, p) for p in ("corridor_mi", "disagreement", "oracle"))
        head = [o - c for o, c in zip(orc, cor)]
        edge = [d - c for d, c in zip(dis, cor)]
        mk = "s" if arm == "clean" else "D"
        ax.plot(head, edge, mk, ms=8, color="tab:orange", mfc="white" if arm == "clean" else None)
        for h, e, b in zip(head, edge, BUDGETS):
            ax.annotate(f"{arm[:5]}@{b}", (h, e), textcoords="offset points", xytext=(6, -3), fontsize=7)
    ax.axhline(0.0, color="0.6", lw=0.8)
    # the wide-cone motion-coupled holdout is the ceiling case at sensor scale: headroom
    # ~0.14 (oracle 0.53 vs corridor 0.39 on holdout A) yet edge ~0 -- cone reveals ~1100
    # cells at once, i.e. off this budget axis entirely; plotted as the boundary reminder
    ax.plot([0.14], [-0.006], "x", color="tab:red", ms=9)
    ax.annotate("wide-cone look\n(holdout A)", (0.14, -0.006), textcoords="offset points",
                xytext=(8, -14), fontsize=7, color="tab:red")
    ax.set_xlabel("headroom left by masking:  tau(oracle) − tau(corridor)")
    ax.set_ylabel("adjoint's edge:  tau(disagreement) − tau(corridor)")
    ax.set_title("the derivative's edge tracks the headroom")
    ax.grid(alpha=0.3)

    fig.suptitle("When does sensing need a derivative? Floor and ceiling compression bound the regime where it pays")
    out = OUT / "scope_law.png"
    fig.savefig(out, dpi=140)
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
