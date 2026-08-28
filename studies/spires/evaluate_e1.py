"""Mechanical evaluation of the pre-registered E1 criteria against a scored artifact.

Written AFTER the frozen pipeline produced its numbers and it reads only the committed
artifact json -- it computes no cost, opens no window, and touches nothing the freeze (A3)
covers. Its job is arithmetic on `e1_heldout.json` plus the one statistic the prereg asks for
that the runner does not emit: the by-sequence cluster-robust test on the pooled paired NLL
difference.

The criteria, verbatim from PREREG_risk_calibration.md (frozen 2026-08-28) with amendment A1:

  E1-i    clark |z|<=1 coverage in [0.55, 0.85].
  E1-ii   clark sd-ratio in [0.7, 1.4].
  E1-iii  clark mean NLL below fosm's AND below clark-diag's, in at least 2 of the 3 held-out
          sequences each  -- A1(1) restates this as "in at least 3 of the 4 held-out
          sequences each" -- with the pooled paired difference reported with a by-sequence
          cluster-robust test.
  E1-iv   |z|<=2 coverage is reported unconditionally.

A1(2): criteria E1-i..iii are evaluated on UNFLAGGED windows; flagged windows are reported
separately and unconditionally. A2(3): the same for the retention filter.

Everything below the `POST-HOC` marker in the output is exploratory and is labelled as such;
it is not part of any criterion.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ARMS = ("clark", "clark-diag", "fosm")
E1_I_BAND = (0.55, 0.85)
E1_II_BAND = (0.7, 1.4)
E1_III_MIN_SEQUENCES = 3  # A1(1): "at least 3 of the 4 held-out sequences"
E1_III_N_SEQUENCES = 4


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Lentz). NR 6.4, no scipy."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (tiny if abs(d) < tiny else d)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d = 1.0 / (tiny if abs(d) < tiny else d)
        c = tiny if abs(c) < tiny else c
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d = 1.0 / (tiny if abs(d) < tiny else d)
        c = tiny if abs(c) < tiny else c
        de = d * c
        h *= de
        if abs(de - 1.0) < 3e-16:
            break
    return h


def betainc_reg(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf_two_sided(t: float, dof: float) -> float:
    """Two-sided Student-t tail probability."""
    if dof <= 0 or not math.isfinite(t):
        return float("nan")
    return betainc_reg(0.5 * dof, 0.5, dof / (dof + t * t))


def cluster_robust_paired(d: np.ndarray, cluster: np.ndarray) -> dict:
    """Mean of paired differences with a CR1 cluster-robust SE, clustered by sequence.

    The unit of independence is the SEQUENCE: windows inside one sequence share that
    sequence's trajectory, its site's survey and its drift realization, so treating them as
    independent would understate the SE. CR1 finite-sample correction G/(G-1) (K = 1 estimated
    parameter, so the (N-1)/(N-K) factor is 1), Student-t with G-1 degrees of freedom.

    With a handful of clusters this test has very little power; that is a property of the
    pre-registered design (4 held-out sequences), not of the implementation, and the effective
    degrees of freedom are reported alongside the p-value so it cannot be read as more.
    """
    n = len(d)
    groups = sorted(set(cluster.tolist()))
    g = len(groups)
    mean = float(d.mean())
    u = np.array([float((d[cluster == k] - mean).sum()) for k in groups])
    v = float((u**2).sum()) / n**2
    correction = g / (g - 1) if g > 1 else float("nan")
    se = math.sqrt(correction * v) if g > 1 else float("nan")
    t = mean / se if se and math.isfinite(se) and se > 0 else float("nan")
    return {
        "n_windows": n, "n_clusters": g, "clusters": groups,
        "mean_difference": mean, "cluster_robust_se": se,
        "t": t, "dof": g - 1, "p_two_sided": t_sf_two_sided(t, g - 1),
        "note": "CR1, clustered by sequence; low power with few clusters",
    }


def evaluate(artifact: Path, out: Path) -> dict:
    r = json.loads(artifact.read_text())
    rows = r["windows"]
    unflagged = [w for w in rows if w["distinct"] and not w["overhang_flag"]]
    flagged = [w for w in rows if w["distinct"] and w["overhang_flag"]]
    seqs_all = sorted({w["sequence"] for w in rows})
    prim = r["primary"]
    by_seq = r["by_sequence_distinct_unflagged"]

    # --- E1-i / E1-ii / E1-iv, on the unflagged pool -----------------------------------------
    cov1 = prim["clark"]["cov1"]
    sdr = prim["clark"]["sd_ratio"]
    cov2 = prim["clark"]["cov2"]
    e1i = {"criterion": "clark |z|<=1 coverage in [0.55, 0.85]", "value": cov1,
           "band": list(E1_I_BAND), "n_windows": prim["n"],
           "verdict": "PASS" if E1_I_BAND[0] <= cov1 <= E1_I_BAND[1] else "FAIL"}
    e1ii = {"criterion": "clark sd-ratio in [0.7, 1.4]", "value": sdr,
            "band": list(E1_II_BAND), "n_windows": prim["n"],
            "verdict": "PASS" if E1_II_BAND[0] <= sdr <= E1_II_BAND[1] else "FAIL"}
    e1iv = {"criterion": "clark |z|<=2 coverage, reported unconditionally",
            "value_unflagged": cov2,
            "value_flagged": r["groups"]["distinct_flagged"].get("clark", {}).get("cov2"),
            "value_all": r["groups"]["distinct_all"].get("clark", {}).get("cov2"),
            "verdict": "REPORTED (no band)"}

    # --- E1-iii ------------------------------------------------------------------------------
    seq_rows = {}
    for s in seqs_all:
        blk = by_seq.get(s)
        if not blk or blk.get("n", 0) == 0:
            seq_rows[s] = {"n_unflagged": 0, "evaluable": False}
            continue
        seq_rows[s] = {
            "n_unflagged": blk["n"], "evaluable": True,
            "nll_clark": blk["clark"]["mean_nll"],
            "nll_fosm": blk["fosm"]["mean_nll"],
            "nll_clark_diag": blk["clark-diag"]["mean_nll"],
            "clark_below_fosm": blk["clark"]["mean_nll"] < blk["fosm"]["mean_nll"],
            "clark_below_clark_diag": blk["clark"]["mean_nll"] < blk["clark-diag"]["mean_nll"],
        }
    won_fosm = sum(1 for v in seq_rows.values() if v.get("clark_below_fosm"))
    won_diag = sum(1 for v in seq_rows.values() if v.get("clark_below_clark_diag"))
    n_eval = sum(1 for v in seq_rows.values() if v["evaluable"])

    cl = np.array([w["sequence"] for w in unflagged])
    d_fosm = np.array([w["clark"]["nll_recal"] - w["fosm"]["nll_recal"] for w in unflagged])
    d_diag = np.array([w["clark"]["nll_recal"] - w["clark-diag"]["nll_recal"] for w in unflagged])
    e1iii = {
        "criterion": ("clark mean NLL below fosm's AND below clark-diag's, in at least "
                      f"{E1_III_MIN_SEQUENCES} of the {E1_III_N_SEQUENCES} held-out sequences "
                      "each; pooled paired difference reported with a by-sequence "
                      "cluster-robust test"),
        "sequences_total": E1_III_N_SEQUENCES,
        "sequences_with_unflagged_windows": n_eval,
        "per_sequence": seq_rows,
        "sequences_clark_below_fosm": won_fosm,
        "sequences_clark_below_clark_diag": won_diag,
        "required": E1_III_MIN_SEQUENCES,
        "vs_fosm_verdict": "PASS" if won_fosm >= E1_III_MIN_SEQUENCES else "FAIL",
        "vs_clark_diag_verdict": "PASS" if won_diag >= E1_III_MIN_SEQUENCES else "FAIL",
        "verdict": "PASS" if (won_fosm >= E1_III_MIN_SEQUENCES
                              and won_diag >= E1_III_MIN_SEQUENCES) else "FAIL",
        "pooled_cluster_robust_clark_minus_fosm": cluster_robust_paired(d_fosm, cl),
        "pooled_cluster_robust_clark_minus_clark_diag": cluster_robust_paired(d_diag, cl),
    }

    res = {
        "evaluated_artifact": str(artifact),
        "prereg": "clark_paper/PREREG_risk_calibration.md, frozen 2026-08-28, amendments A1-A3",
        "pipeline_frozen_at": "spires e54c12a",
        "recalibration": r["recalibration"],
        "counts": {
            "windows_on_disk": r["n_windows_computed"] + len(r["skipped"]),
            "scored": r["n_windows_computed"],
            "skipped_by_retention_or_empty": len(r["skipped"]),
            "unflagged_scored": len(unflagged),
            "flagged_scored": len(flagged),
            "sequences": seqs_all,
        },
        "E1-i": e1i, "E1-ii": e1ii, "E1-iii": e1iii, "E1-iv": e1iv,
        "arms_unflagged": {a: prim[a] for a in ARMS},
        "arms_flagged": {a: r["groups"]["distinct_flagged"].get(a) for a in ARMS},
        "arms_all": {a: r["groups"]["distinct_all"].get(a) for a in ARMS},
        "by_sequence_unflagged": by_seq,
        "skipped": r["skipped"],
        "POST-HOC_exploratory_not_a_criterion": {
            "note": "computed for the report only; no pre-registered criterion uses these",
            "sign_test_windows_clark_below_fosm": int((d_fosm < 0).sum()),
            "sign_test_windows_total": len(d_fosm),
            "median_paired_nll_clark_minus_fosm": float(np.median(d_fosm)),
            "median_paired_nll_clark_minus_clark_diag": float(np.median(d_diag)),
        },
    }
    out.write_text(json.dumps(res, indent=2))
    return res


def _selftest() -> None:
    """The t tail against the closed form for dof = 3, where it is elementary."""
    for t in (0.0, 0.5, 1.0, 2.5, 7.0):
        closed = 1.0 - 2.0 * (
            0.5 + (1.0 / math.pi) * ((t / math.sqrt(3)) / (1 + t * t / 3)
                                     + math.atan(t / math.sqrt(3))) - 0.5
        )
        assert abs(t_sf_two_sided(t, 3) - closed) < 1e-12, (t, t_sf_two_sided(t, 3), closed)
    print("selftest: two-sided t tail matches the dof=3 closed form to < 1e-12")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    _selftest()
    a = Path(args.artifact)
    o = Path(args.out) if args.out else a.parent / "e1_heldout_criteria.json"
    res = evaluate(a, o)
    for k in ("E1-i", "E1-ii", "E1-iii", "E1-iv"):
        print(f"\n{k}: {res[k].get('verdict')}")
        print(json.dumps({x: y for x, y in res[k].items() if x != "per_sequence"}, indent=1))
    print("\nwrote", o)
