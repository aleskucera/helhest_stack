"""Does deciding WHETHER to look (C5) change the benchmark's conclusion?

    .venv/bin/python -m studies.bench.compare_c5

The n=32 benchmark found every policy losing to never-looking on a per-seed basis, and the
diagnosis was that the look budget was spent UNCONDITIONALLY -- 4 looks whether or not the
plan had anything to gain. `SENSITIVITY_PLAN.md` lists deciding whether to observe as C5, part
of the contribution rather than a patch, so it was added: each policy skips its look when the
best available bearing would resolve less than LOOK_THRESHOLD of its own objective's total.

This compares the two runs directly. It exists because adding a mechanism and then reporting
only the improved numbers is how benchmarks get rigged. The comparison is paired per seed, the
same threshold is applied to every arm against its own total, and the threshold was chosen a
priori and NOT swept -- so if the gate helps, the honest claim is "with this threshold", and
robustness is still open.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "studies" / "out" / "bench"
ARMS = ("none", "sigma", "entropy", "cvar", "attribution")


def sign_test(d: np.ndarray) -> tuple[int, int, float]:
    nz = d[d != 0]
    n, k = len(nz), int((nz < 0).sum())
    if n == 0:
        return 0, 0, 1.0
    tail = sum(comb(n, i) for i in range(min(k, n - k) + 1))
    return n, k, min(1.0, 2.0 * tail / 2**n)


def load(name: str):
    path = OUT / name
    return json.load(open(path))["rows"] if path.exists() else None


def compare(variant: str) -> None:
    before = load(f"results_{variant}_nogate.json")
    after = load(f"results_{variant}_c5.json")
    if before is None or after is None:
        print(f"  ({variant}: missing a run, skipped)")
        return

    n = min(len(before), len(after))
    before, after = before[:n], after[:n]
    print(f"\n=== variant {variant}  (n={n}) ===")
    print(
        f"{'policy':<13}{'mean before':>13}{'mean after':>12}{'delta':>8}"
        f"{'looks before':>14}{'looks after':>13}{'reached':>10}"
    )
    for a in ARMS:
        tb = np.array([r[a]["time"] for r in before], float)
        ta = np.array([r[a]["time"] for r in after], float)
        lb = np.mean([r[a]["n_looks"] for r in before])
        la = np.mean([r[a]["n_looks"] for r in after])
        reach = sum(r[a]["reached"] for r in after)
        print(
            f"{a:<13}{tb.mean():>13.0f}{ta.mean():>12.0f}{ta.mean() - tb.mean():>+8.0f}"
            f"{lb:>14.2f}{la:>13.2f}{reach:>7}/{n:<3}"
        )

    print("\n  with the gate, paired against never looking:")
    nul = np.array([r["none"]["time"] for r in after], float)
    for a in ARMS:
        if a == "none":
            continue
        t = np.array([r[a]["time"] for r in after], float)
        d = t - nul
        k, kk, p = sign_test(d)
        print(f"    {a:<12} mean {d.mean():>+7.0f}  faster on {kk:>2}/{k:<2}  sign p={p:.3f}")

    print("\n  with the gate, attribution against each baseline:")
    att = np.array([r["attribution"]["time"] for r in after], float)
    for a in ARMS:
        if a == "attribution":
            continue
        t = np.array([r[a]["time"] for r in after], float)
        d = att - t
        k, kk, p = sign_test(d)
        print(f"    vs {a:<10} mean {d.mean():>+7.0f}  faster on {kk:>2}/{k:<2}  sign p={p:.3f}")


def main() -> None:
    print("C5: deciding WHETHER to look, not only where")
    print("threshold = 0.25 of each policy's own objective total, applied to every arm, not swept")
    for v in ("gap", "corridor"):
        compare(v)


if __name__ == "__main__":
    main()
