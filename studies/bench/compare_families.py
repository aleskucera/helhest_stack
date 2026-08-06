"""When is an adjoint actually NEEDED, and when does a distance transform do?

    .venv/bin/python -m studies.bench.compare_families

Section 7 found decision-focused sensing beating entropy decisively and geometry only barely --
"reveal what you are about to drive over" was as good as the taped adjoint at small budgets.
That was a property of the plan set, not of the method: a fan of constant-curvature arcs makes
"which plan wins" and "which cells does it cross" nearly the same question, so the derivative
has nothing to add that a distance transform does not already know.

This contrasts three plan families that differ ONLY in how much the plans' spatial coverage
overlaps, holding terrain, sigma, budgets, policies and seeds fixed:

  fan     16 separate corridors     -- geometry can rank by coverage alone
  hybrid  4 paths x 4 speed profiles -- geometry can rank ACROSS groups, never WITHIN one
  speed   1 path, 16 speed profiles -- geometry sees sixteen identical corridors

`tau_within` is the diagnostic that makes the claim falsifiable rather than rhetorical: it is
the ordering restricted to plans that cross exactly the same cells, so it isolates the part of
the decision no purely geometric score can address by construction.
"""

from __future__ import annotations

import json

import numpy as np

from .ranking import BUDGETS
from .ranking import OUT
from .ranking import sign_test

FAMILIES = {
    "fan": "16 separate corridors",
    "hybrid": "4 paths x 4 speed profiles",
    "speed": "1 shared path, 16 profiles",
}
SHOW = ("entropy", "swath", "swath_var", "attribution", "disagreement", "oracle")


def load(family: str) -> list[dict] | None:
    path = OUT / f"ranking_{family}.json"
    return json.load(open(path))["rows"] if path.exists() else None


def main() -> None:
    data = {f: load(f) for f in FAMILIES}
    missing = [f for f, r in data.items() if r is None]
    if missing:
        print(f"missing runs for {missing} -- run: ranking --family <name> --seeds 200")
    data = {f: r for f, r in data.items() if r}

    print("plan separation, measured (not assumed):")
    print(f"{'family':<10}{'endpoints apart':>17}{'coverage spread':>17}{'cost spread':>13}")
    for f, rows in data.items():
        print(
            f"{f:<10}{np.mean([r['path_spread'] for r in rows]):>14.3f} m"
            f"{np.mean([r['coverage_spread'] for r in rows]):>17.4f}"
            f"{np.mean([r['spread_true'] for r in rows]):>13.2f}"
        )

    for m in BUDGETS:
        print(f"\n=== tau at {m} revealed cells ===")
        print(f"{'policy':<14}" + "".join(f"{f:>12}" for f in data))
        for p in SHOW:
            row = f"{p:<14}"
            for f, rows in data.items():
                row += f"{np.mean([r['policies'][p][str(m)]['tau'] for r in rows]):>+12.3f}"
            print(row)
        print(f"{'adjoint - geom':<14}", end="")
        for f, rows in data.items():
            d = np.array(
                [
                    r["policies"]["disagreement"][str(m)]["tau"]
                    - r["policies"]["swath"][str(m)]["tau"]
                    for r in rows
                ]
            )
            k, w, pv = sign_test(d)
            print(f"{d.mean():>+8.3f}{'*' if pv < 0.01 else ' ':<4}", end="")
        print("     (* = sign test p < 0.01)")

    print("\n=== tau WITHIN path-identical groups -- what geometry cannot reach ===")
    print("(undefined for `fan`: no two plans share a path)")
    for f, rows in data.items():
        if not np.isfinite(rows[0]["tau_within_before"]):
            continue
        print(
            f"\n  {f} ({FAMILIES[f]}), no sensing: {np.mean([r['tau_within_before'] for r in rows]):+.3f}"
        )
        for p in SHOW:
            vals = [
                f"{np.mean([r['policies'][p][str(m)]['tau_within'] for r in rows]):+.3f}"
                for m in BUDGETS
            ]
            print(f"    {p:<14}" + "".join(f"{v:>10}" for v in vals))
        d = np.array(
            [
                r["policies"]["disagreement"][str(BUDGETS[-1])]["tau_within"]
                - r["policies"]["swath"][str(BUDGETS[-1])]["tau_within"]
                for r in rows
            ]
        )
        k, w, pv = sign_test(d)
        print(
            f"    disagreement - swath @{BUDGETS[-1]}: {d.mean():+.3f}, "
            f"better on {w}/{k}, p={pv:.2e}"
        )

    figure(data, OUT / "families.png")
    print(f"\nwrote {OUT / 'families.png'}")


def figure(data: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
    colour = {"fan": "#4c9a6a", "hybrid": "#7a3ea8", "speed": "#c9722a"}

    # (a) the whole point: how the adjoint's edge over geometry varies with plan overlap
    ax = axes[0]
    width = 0.26
    for i, (f, rows) in enumerate(data.items()):
        for j, m in enumerate(BUDGETS):
            d = np.array(
                [
                    r["policies"]["disagreement"][str(m)]["tau"]
                    - r["policies"]["swath"][str(m)]["tau"]
                    for r in rows
                ]
            )
            _, _, pv = sign_test(d)
            ax.bar(
                j + (i - 1) * width,
                d.mean(),
                width * 0.9,
                color=colour[f],
                edgecolor="k",
                lw=0.5,
                label=f if j == 0 else None,
            )
            if pv < 0.01:
                y = d.mean()
                ax.text(
                    j + (i - 1) * width,
                    y + (0.008 if y >= 0 else -0.022),
                    "*",
                    ha="center",
                    fontsize=11,
                )
    ax.axhline(0, color="k", lw=1.0)
    ax.margins(y=0.22)  # headroom so the significance stars stay inside the axes
    ax.set_xticks(range(len(BUDGETS)))
    ax.set_xticklabels([str(m) for m in BUDGETS])
    ax.set_xlabel("cells revealed")
    ax.set_ylabel("tau(adjoint) - tau(geometry)")
    ax.set_title("(a) above zero = the derivative earns its keep\n* = sign test p < 0.01")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    # (b) absolute tau at the largest budget
    ax = axes[1]
    shown = ("entropy", "swath", "attribution", "disagreement", "oracle")
    for i, (f, rows) in enumerate(data.items()):
        y = [np.mean([r["policies"][p][str(BUDGETS[-1])]["tau"] for r in rows]) for p in shown]
        ax.plot(range(len(shown)), y, marker="o", color=colour[f], label=f, lw=2)
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels(shown, rotation=20, ha="right")
    ax.set_ylabel(f"tau at {BUDGETS[-1]} cells")
    ax.set_title("(b) fully-shared paths make the problem\nEASY for geometry, not hard")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    # (c) the component geometry cannot reach, by construction
    ax = axes[2]
    rows = data.get("hybrid")
    if rows:
        for p, c in (
            ("entropy", "#2f7ec4"),
            ("swath", "#4c9a6a"),
            ("disagreement", "#7a3ea8"),
            ("oracle", "#333333"),
        ):
            y = [np.mean([r["policies"][p][str(m)]["tau_within"] for r in rows]) for m in BUDGETS]
            ax.plot(
                BUDGETS, y, marker="o", color=c, label=p, lw=2, ls="--" if p == "oracle" else "-"
            )
        ax.axhline(
            np.mean([r["tau_within_before"] for rows_ in [rows] for r in rows_]),
            color="k",
            ls=":",
            lw=0.9,
            label="no sensing",
        )
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(BUDGETS)
        ax.set_xticklabels([str(m) for m in BUDGETS])
    ax.set_xlabel("cells revealed")
    ax.set_ylabel("tau WITHIN path-identical groups")
    ax.set_title("(c) hybrid: ranking plans that cross\nexactly the same cells")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle(
        "When does map-cell attribution beat a distance transform?  (n=200 per family)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
