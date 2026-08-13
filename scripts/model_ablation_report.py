"""What did each model change do to the DECISION, and what did it cost?"""

from __future__ import annotations

import json
import sys

import numpy as np


def tau_b(a: np.ndarray, b: np.ndarray, chunk: int = 512) -> float:
    n = len(a)
    conc = disc = ta = tb = 0
    idx = np.arange(n)
    for i in range(0, n, chunk):
        hi = min(i + chunk, n)
        da = np.sign(a[i:hi, None] - a[None, :]).astype(np.int8)
        db = np.sign(b[i:hi, None] - b[None, :]).astype(np.int8)
        up = idx[i:hi, None] < idx[None, :]
        p = (da * db)[up]
        conc += int((p > 0).sum())
        disc += int((p < 0).sum())
        ta += int(((da == 0) & (db != 0))[up].sum())
        tb += int(((db == 0) & (da != 0))[up].sum())
    den = np.sqrt((conc + disc + ta) * (conc + disc + tb))
    return float((conc - disc) / den) if den else float("nan")


class Run:
    def __init__(self, stem: str):
        with open(stem + ".json") as f:
            self.meta = json.load(f)
        self.blob = np.load(stem + ".npz")

    def has(self, level: str) -> bool:
        return level in self.meta["levels"]

    def scen(self, level: str) -> list[str]:
        return list(self.meta["levels"][level].keys())

    def J(self, level: str, s: str) -> np.ndarray:
        return self.blob[f"{level}|{s}|J"].astype(np.float64)

    def end(self, level: str, s: str) -> np.ndarray:
        return self.blob[f"{level}|{s}|end"].astype(np.float64)

    def u0(self, level: str, s: str) -> np.ndarray:
        return np.array(self.meta["levels"][level][s]["u0"])

    def ms(self, level: str) -> float:
        return float(np.mean([v["ms"] for v in self.meta["levels"][level].values()]))


def pair(ra: Run, la: str, rb: Run, lb: str, elite_frac: float = 0.02) -> dict:
    rows = []
    for s in ra.scen(la):
        Ja, Jb = ra.J(la, s), rb.J(lb, s)
        k = max(int(elite_frac * len(Ja)), 1)
        ea = set(np.argpartition(Ja, k - 1)[:k].tolist())
        eb = set(np.argpartition(Jb, k - 1)[:k].tolist())
        d = np.linalg.norm(rb.end(lb, s) - ra.end(la, s), axis=1)
        rows.append(
            {
                "tau": tau_b(Ja, Jb),
                "elite": len(ea & eb) / k,
                "argmin": float(np.argmin(Ja) == np.argmin(Jb)),
                "du0": float(np.linalg.norm(rb.u0(lb, s) - ra.u0(la, s))),
                "end": float(d.mean() * 100),
                "endp95": float(np.percentile(d, 95) * 100),
                "moved": float(np.mean(np.abs(Ja - Jb) > 1e-6)),
            }
        )
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def hdr(title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"{'':>26}{'tau_b':>8}{'elite':>7}{'argmin':>7}{'moved':>7}"
          f"{'d|u0|':>8}{'end cm':>8}{'p95 cm':>8}{'ms':>7}")


def row(label: str, agg: dict, ms: float) -> None:
    print(f"{label:>26}{agg['tau']:>8.4f}{agg['elite']:>7.0%}{agg['argmin']:>7.0%}"
          f"{agg['moved']:>7.0%}{agg['du0']:>8.3f}{agg['end']:>8.1f}"
          f"{agg['endp95']:>8.1f}{ms:>7.2f}")


def main() -> None:
    m, t = Run(sys.argv[1]), Run(sys.argv[2])

    hdr("every rung vs the pre-work simulator (main, no actuator lag)")
    for label, r, lv in [
        ("exact arc + certificates", t, "base"),
        ("  + tau_motor 0.19", t, "lag"),
        ("  + cylinder envelope", t, "cylinder"),
        ("  + shear/momentum", t, "traction"),
        ("  + both", t, "all"),
    ]:
        if r.has(lv):
            row(label, pair(m, "base", r, lv), r.ms(lv))

    hdr("each change against the rung below it")
    for label, ra, la, rb, lb in [
        ("main: + tau_motor 0.19", m, "base", m, "lag"),
        ("main -> arc + certificates", m, "base", t, "base"),
        ("  + tau_motor 0.19", t, "base", t, "lag"),
        ("  + cylinder envelope", t, "lag", t, "cylinder"),
        ("  + shear/momentum", t, "lag", t, "traction"),
    ]:
        if ra.has(la) and rb.has(lb):
            row(label, pair(ra, la, rb, lb), rb.ms(lb))

    print("\n=== the cylinder envelope, per world (vs the same model with a sphere) ===")
    print(f"{'world':>10}{'candidates moved':>18}{'tau_b':>9}{'elite':>7}"
          f"{'max dJ':>10}{'max end cm':>12}")
    for w in ["gap", "slalom", "pillars", "pocket", "ridge", "bumpy", "rocks"]:
        ss = [s for s in t.scen("cylinder") if s.split("|")[0] == w]
        mv, dj, de, tt, el = [], [], [], [], []
        for s in ss:
            a, b = t.J("lag", s), t.J("cylinder", s)
            mv.append(np.mean(np.abs(a - b) > 1e-6))
            dj.append(np.abs(a - b).max())
            de.append(np.linalg.norm(t.end("cylinder", s) - t.end("lag", s), axis=1).max() * 100)
            tt.append(tau_b(a, b))
            k = max(int(0.02 * len(a)), 1)
            el.append(len(set(np.argpartition(a, k - 1)[:k].tolist())
                          & set(np.argpartition(b, k - 1)[:k].tolist())) / k)
        print(f"{w:>10}{np.mean(mv):>17.1%}{np.mean(tt):>9.4f}{np.mean(el):>7.0%}"
              f"{np.max(dj):>10.2f}{np.max(de):>12.2f}")


if __name__ == "__main__":
    main()
