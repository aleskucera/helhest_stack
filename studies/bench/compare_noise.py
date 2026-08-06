"""What survives when the map is wrong for the reasons a real map is wrong?

    .venv/bin/python -m studies.bench.compare_noise

Section 7 ran on a belief corrupted by ONE source, modelled crudely: a disc of "observed"
inside which the map was exactly right, and reveals that handed over ground truth. That is a
perfect sensor with perfect localisation. `noise.py` adds the other two with the structure they
really have, and `verify_noise.py` gates that structure before any of this is believed.

The comparison holds the plan family, terrain, budgets, policies and seeds fixed and varies
only which sources are switched on, so every difference below is attributable.

The number to read first is the ORACLE. It is not a policy -- it is the best any sensing
strategy could do with that budget. Where the oracle falls, the ceiling itself has moved, and
no amount of cleverness in choosing cells can recover it.
"""

from __future__ import annotations

import json

import numpy as np

from .ranking import BUDGETS
from .ranking import OUT
from .ranking import sign_test

ARMS = ("clean", "sensor", "occlusion", "localisation", "all")
LABEL = {
    "clean": "section 7 (perfect sensor + pose)",
    "sensor": "+ sensor noise",
    "occlusion": "+ ray-cast shadows",
    "localisation": "+ pose error",
    "all": "all three",
}
SHOW = ("entropy", "random", "swath", "swath_var", "attribution", "disagreement", "oracle")


def load(arm: str):
    tag = "hybrid" if arm == "clean" else f"hybrid_{arm}"
    path = OUT / f"ranking_{tag}.json"
    return json.load(open(path))["rows"] if path.exists() else None


def main() -> None:
    data = {a: load(a) for a in ARMS}
    missing = [a for a, r in data.items() if r is None]
    if missing:
        print(f"missing: {missing}  -- run ranking --family hybrid --noise <arm> --seeds 200\n")
    data = {a: r for a, r in data.items() if r}
    if not data:
        return

    n = min(len(r) for r in data.values())
    print(f"plan family `hybrid`, n={n} seeds per arm\n")

    print(f"{'arm':<14}{'observed':>10}{'no sensing':>12}", end="")
    for p in ("entropy", "swath", "disagreement", "oracle"):
        print(f"{p[:12]:>14}", end="")
    print(f"   ({BUDGETS[-1]} cells revealed)")
    for a, rows in data.items():
        tb = np.mean([r["tau_before"] for r in rows])
        print(f"{a:<14}{'':>10}{tb:>+12.3f}", end="")
        for p in ("entropy", "swath", "disagreement", "oracle"):
            print(
                f"{np.mean([r['policies'][p][str(BUDGETS[-1])]['tau'] for r in rows]):>+14.3f}",
                end="",
            )
        print(f"   {LABEL[a]}")

    print("\nTHE CEILING MOVES. Oracle tau at each budget -- the best ANY policy could do:")
    print(f"{'arm':<14}" + "".join(f"{'@' + str(m):>10}" for m in BUDGETS))
    for a, rows in data.items():
        line = f"{a:<14}"
        for m in BUDGETS:
            line += f"{np.mean([r['policies']['oracle'][str(m)]['tau'] for r in rows]):>+10.3f}"
        print(line)

    print("\nDoes the adjoint still beat geometry?  disagreement - swath, paired per seed:")
    print(f"{'arm':<14}" + "".join(f"{'@' + str(m):>24}" for m in BUDGETS))
    for a, rows in data.items():
        line = f"{a:<14}"
        for m in BUDGETS:
            d = np.array(
                [
                    r["policies"]["disagreement"][str(m)]["tau"]
                    - r["policies"]["swath"][str(m)]["tau"]
                    for r in rows
                ]
            )
            k, w, p = sign_test(d)
            line += f"{d.mean():>+9.3f} {w:>3}/{k:<3} p={p:>6.0e}"
        print(line)

    print("\nDoes it still beat entropy?  disagreement - entropy, paired per seed:")
    print(f"{'arm':<14}" + "".join(f"{'@' + str(m):>24}" for m in BUDGETS))
    for a, rows in data.items():
        line = f"{a:<14}"
        for m in BUDGETS:
            d = np.array(
                [
                    r["policies"]["disagreement"][str(m)]["tau"]
                    - r["policies"]["entropy"][str(m)]["tau"]
                    for r in rows
                ]
            )
            k, w, p = sign_test(d)
            line += f"{d.mean():>+9.3f} {w:>3}/{k:<3} p={p:>6.0e}"
        print(line)

    print("\nWithin path-identical groups -- the part geometry cannot reach by construction:")
    print(
        f"{'arm':<14}{'no sensing':>12}{'entropy':>10}{'swath':>10}{'disagree':>10}{'oracle':>10}"
    )
    for a, rows in data.items():
        line = f"{a:<14}{np.mean([r['tau_within_before'] for r in rows]):>+12.3f}"
        for p in ("entropy", "swath", "disagreement", "oracle"):
            line += f"{np.mean([r['policies'][p][str(BUDGETS[-1])]['tau_within'] for r in rows]):>+10.3f}"
        print(line)

    figure(data, OUT / "noise.png")
    print(f"\nwrote {OUT / 'noise.png'}")


def figure(data: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [a for a in ARMS if a in data]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))
    colour = {
        "entropy": "#2f7ec4",
        "swath": "#4c9a6a",
        "disagreement": "#7a3ea8",
        "oracle": "#333333",
    }

    # (a) absolute tau at the largest budget, per arm
    ax = axes[0]
    x = np.arange(len(order))
    for p, c in colour.items():
        y = [np.mean([r["policies"][p][str(BUDGETS[-1])]["tau"] for r in data[a]]) for a in order]
        ax.plot(x, y, marker="o", color=c, lw=2.2, ls="--" if p == "oracle" else "-", label=p)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=18, ha="right")
    ax.set_ylabel(f"tau at {BUDGETS[-1]} cells")
    ax.set_title("(a) the dashed line is the CEILING.\nPose error nearly halves it")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # (b) how much of the achievable gain each policy captures
    ax = axes[1]
    for p, c in colour.items():
        if p == "oracle":
            continue
        y = []
        for a in order:
            tb = np.mean([r["tau_before"] for r in data[a]])
            oc = np.mean([r["policies"]["oracle"][str(BUDGETS[-1])]["tau"] for r in data[a]])
            v = np.mean([r["policies"][p][str(BUDGETS[-1])]["tau"] for r in data[a]])
            y.append((v - tb) / max(oc - tb, 1e-9))
        ax.plot(x, y, marker="o", color=c, lw=2.2, label=p)
    ax.axhline(1.0, color="k", ls="--", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=18, ha="right")
    ax.set_ylabel("fraction of the achievable gain")
    ax.set_title("(b) normalised by the moving ceiling:\nwho uses the budget best")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    # (c) adjoint minus geometry, per arm and budget
    ax = axes[2]
    width = 0.26
    for j, m in enumerate(BUDGETS):
        vals, sig = [], []
        for a in order:
            d = np.array(
                [
                    r["policies"]["disagreement"][str(m)]["tau"]
                    - r["policies"]["swath"][str(m)]["tau"]
                    for r in data[a]
                ]
            )
            _, _, p = sign_test(d)
            vals.append(d.mean())
            sig.append(p < 0.01)
        ax.bar(x + (j - 1) * width, vals, width * 0.9, edgecolor="k", lw=0.5, label=f"{m} cells")
        for xi, v, sg in zip(x + (j - 1) * width, vals, sig):
            if sg:
                ax.text(xi, v + (0.006 if v >= 0 else -0.016), "*", ha="center", fontsize=10)
    ax.axhline(0, color="k", lw=1.0)
    ax.margins(y=0.2)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=18, ha="right")
    ax.set_ylabel("tau(adjoint) - tau(geometry)")
    ax.set_title("(c) the adjoint's edge over a distance\ntransform, * = p < 0.01")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle(
        "Does decision-focused sensing survive realistic map error?  "
        f"(hybrid plans, n={min(len(r) for r in data.values())})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
