"""Post hoc readings of the frozen v2 held-out record (2026-09-11 review round).

Reads out/v2/heldout_score_{foresight,hindsight}.json (unflagged, scoreable
rows) and writes out/v2/posthoc_review.json. No new scoring: every number is
a function of fields the runner already wrote.

  1. reference CVaR: the runner's cvar_mc is the empirical mean of the worst
     10% of the 20k draws; its distance from the Gaussian CVaR (eq. cvar)
     applied to the referee's own (E_mc, sd_mc) is the floor any Gaussian-CVaR
     estimator fed exact moments retains.
  2. referee noise floor: relative standard error of E_mc, sd_mc / (E_mc
     sqrt(n)); of sd_mc, 1 / sqrt(2 (n - 1)) (Gaussian approximation).
  3. hybrid arm: the fold's mean with linearization's sd (no fitted
     parameter), CVaR_0.9 error against cvar_mc, paired against clark-corr.
  4. Table II with two significant figures on the medians.
  5. paired decision test: exact McNemar (two-sided binomial on the
     discordant fans) of clark-corr's top-1 hit against fosm, mc-32, mean-map.
  7. traverse-level statistics: per-traverse win fractions of clark-corr
     against fosm (rel_err_E; cvar_abs_err) and against mc-32 (both moments),
     the number of traverses with a majority, the traverse-level two-sided
     sign test, and a cluster bootstrap by traverse (2000 resamples, seed 0)
     of the median errors and of the fosm / clark-corr ratios.
  6. threshold verdicts: at thresholds tau set to the reference CVaR's
     quartiles and 90th percentile, the fraction of segments on which an
     arm's safe/unsafe verdict (cvar_arm < tau) disagrees with the
     reference's, split into unsafe-called-safe and safe-called-unsafe.

    PYTHONPATH=studies python -m baseprod.v2_posthoc_review [out_dir]
"""
from __future__ import annotations

import json
import sys
import time
from math import comb, exp, pi, sqrt
from pathlib import Path

import numpy as np

Q = 0.9
Z_Q = 1.2815515655446004                       # Phi^-1(0.9)
LAM_Q = exp(-0.5 * Z_Q * Z_Q) / sqrt(2 * pi) / (1 - Q)   # phi(z_q)/(1-q) = 1.754983
ARMS = ["clark-corr", "clark", "mc-32", "fosm", "mc-2", "mean-map", "step-form"]


def binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    pk = [comb(n, i) / 2 ** n for i in range(n + 1)]
    # relative tolerance: an absolute 1e-15 admitted every outcome up to 1e-15
    # when the observed tail was far below it (methodology review 2026-09-13,
    # F-11: hindsight hybrid sign test 2.4e-15 reported, 2.3e-18 exact)
    return min(1.0, sum(p for p in pk if p <= pk[k] * (1.0 + 1e-9)))


def p95(x):
    return float(np.quantile(x, 0.95, method="lower"))    # as REPORT_v2 / Table II


def main(out_dir: Path):
    out = {"_meta": {"written": time.strftime("%F %T"), "lambda_q": LAM_Q, "q": Q,
                     "source": "heldout_score_{foresight,hindsight}.json (unflagged, scoreable rows)"}}
    for cond in ("foresight", "hindsight"):
        rows = [r for r in json.loads((out_dir / f"heldout_score_{cond}.json").read_text())
                if not r.get("flagged") and r.get("scoreable")]
        E = np.array([r["mc"]["E_mc"] for r in rows])
        sd = np.array([r["mc"]["sd_mc"] for r in rows])
        cv = np.array([r["mc"]["cvar_mc"] for r in rows])
        n = np.array([r["mc"]["n_draws"] for r in rows], float)
        gauss = E + LAM_Q * sd
        rel_gap = np.abs(gauss - cv) / cv
        se_E = sd / (E * np.sqrt(n))
        se_sd = 1.0 / np.sqrt(2.0 * (n - 1.0))
        relE = {a: np.array([r["arms"][a]["rel_err_E"] for r in rows]) for a in ("clark-corr", "fosm")}
        hyb = np.abs(np.array([r["arms"]["clark"]["E"] + LAM_Q * r["arms"]["fosm"]["sd"] for r in rows]) - cv)
        cc = np.array([r["arms"]["clark-corr"]["cvar_abs_err"] for r in rows])
        d = {"n_windows": len(rows)}
        d["reference_cvar"] = {
            "definition": "empirical mean of the worst 10% of the referee draws (v2_arms.score_window)",
            "gaussian_on_referee_moments_rel_gap": {"median": float(np.median(rel_gap)), "p95": float(np.quantile(rel_gap, 0.95))}}
        d["referee_noise_floor"] = {
            "rel_se_E_mc": {"median": float(np.median(se_E)), "p95": float(np.quantile(se_E, 0.95))},
            "rel_se_sd_mc": {"median": float(np.median(se_sd))},
            "clark-corr_med_relE_over_med_se": float(np.median(relE["clark-corr"]) / np.median(se_E)),
            "fosm_med_relE_over_med_se": float(np.median(relE["fosm"]) / np.median(se_E)),
            "clark-corr_windows_within_2se": int(np.sum(relE["clark-corr"] < 2 * se_E)),
            "fosm_windows_within_2se": int(np.sum(relE["fosm"] < 2 * se_E))}
        d["hybrid_clarkE_fosmSD"] = {
            "cvar_abs_err": {"median": float(np.median(hyb)), "p95": p95(hyb)},
            "clark-corr_lower_in": int(np.sum(cc < hyb)), "hybrid_lower_in": int(np.sum(hyb < cc)),
            "sign_test_p": binom_two_sided(min(int(np.sum(cc < hyb)), int(np.sum(hyb < cc))),
                                           int(np.sum(cc != hyb)))}
        d["table_cvar"] = {a: {"median": float(np.median([r["arms"][a]["cvar_abs_err"] for r in rows])),
                               "p95": p95([r["arms"][a]["cvar_abs_err"] for r in rows])} for a in ARMS}
        d["table_cvar"]["hybrid"] = d["hybrid_clarkE_fosmSD"]["cvar_abs_err"]
        m = d["table_cvar"]
        d["ratios"] = {"mc32_over_clark-corr_median": m["mc-32"]["median"] / m["clark-corr"]["median"],
                       "fosm_over_clark-corr_median": m["fosm"]["median"] / m["clark-corr"]["median"],
                       "fosm_over_clark-corr_p95": m["fosm"]["p95"] / m["clark-corr"]["p95"],
                       "clark_over_clark-corr_median": m["clark"]["median"] / m["clark-corr"]["median"],
                       "clark_over_clark-corr_p95": m["clark"]["p95"] / m["clark-corr"]["p95"]}
        fans = [r for r in rows if r.get("fan", {}).get("scoreable")]
        mc = {"n_fans": len(fans), "top1": {a: int(sum(bool(r["fan"]["top1"][a]) for r in fans))
                                             for a in ARMS}}
        ours = [bool(r["fan"]["top1"]["clark-corr"]) for r in fans]
        for other in ("fosm", "mc-32", "mean-map"):
            th = [bool(r["fan"]["top1"][other]) for r in fans]
            n10 = sum(a and not b for a, b in zip(ours, th))
            n01 = sum(b and not a for a, b in zip(ours, th))
            mc[f"clark-corr_vs_{other}"] = {"ours_only": n10, "theirs_only": n01,
                                            "mcnemar_exact_p": binom_two_sided(min(n10, n01), n10 + n01)}
        d["decision_paired"] = mc
        taus = {q: float(np.quantile(cv, q)) for q in (0.25, 0.5, 0.75, 0.9)}
        tv = {"thresholds": taus, "arms": {}}
        for a in ARMS:
            arm = np.array([r["arms"][a]["cvar_arm"] for r in rows])
            tv["arms"][a] = {str(q): {"disagree": float(np.mean((arm < t) != (cv < t))),
                                      "unsafe_called_safe": float(np.mean((arm < t) & (cv >= t))),
                                      "safe_called_unsafe": float(np.mean((arm >= t) & (cv < t)))}
                             for q, t in taus.items()}
        d["threshold_verdicts"] = tv
        trs = sorted({r["traverse"] for r in rows})
        by = {t: [r for r in rows if r["traverse"] == t] for t in trs}
        def frac(fn):
            return [float(np.mean([fn(r) for r in by[t]])) for t in trs]
        wins = {"V2-1_E_vs_fosm": frac(lambda r: r["arms"]["clark-corr"]["rel_err_E"] < r["arms"]["fosm"]["rel_err_E"]),
                "V2-3_both_vs_mc-32": frac(lambda r: r["arms"]["clark-corr"]["rel_err_E"] < r["arms"]["mc-32"]["rel_err_E"]
                                           and r["arms"]["clark-corr"]["rel_err_sd"] < r["arms"]["mc-32"]["rel_err_sd"]),
                "cvar_vs_fosm": frac(lambda r: r["arms"]["clark-corr"]["cvar_abs_err"] < r["arms"]["fosm"]["cvar_abs_err"]),
                "cvar_vs_mc-32": frac(lambda r: r["arms"]["clark-corr"]["cvar_abs_err"] < r["arms"]["mc-32"]["cvar_abs_err"])}
        tl = {"n_traverses": len(trs), "traverses": trs}
        for k, v in wins.items():
            maj = int(sum(x > 0.5 for x in v))
            tl[k] = {"min": min(v), "median": float(np.median(v)), "majority_in": maj,
                     "sign_test_p": binom_two_sided(len(trs) - maj, len(trs))}
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(2000):
            rs = [r for t in rng.choice(trs, len(trs), replace=True) for r in by[t]]
            c = np.median([r["arms"]["clark-corr"]["rel_err_E"] for r in rs]); f = np.median([r["arms"]["fosm"]["rel_err_E"] for r in rs])
            cc_ = np.median([r["arms"]["clark-corr"]["cvar_abs_err"] for r in rs]); fc = np.median([r["arms"]["fosm"]["cvar_abs_err"] for r in rs])
            boot.append((c, f, f / c, fc / cc_))
        b = np.array(boot)
        ci = lambda i: [float(np.percentile(b[:, i], 2.5)), float(np.percentile(b[:, i], 97.5))]
        tl["cluster_bootstrap_95"] = {"resamples": 2000, "seed": 0, "median_relE_clark-corr": ci(0), "median_relE_fosm": ci(1),
                                      "ratio_relE_fosm_over_clark-corr": ci(2), "ratio_cvar_err_fosm_over_clark-corr": ci(3)}
        d["traverse_level"] = tl
        out[cond] = d
    (out_dir / "posthoc_review.json").write_text(json.dumps(out, indent=1))
    for cond in ("foresight", "hindsight"):
        d = out[cond]
        print(f"== {cond} n={d['n_windows']}")
        print("  gauss-vs-empirical CVaR rel gap median %.4f p95 %.4f" % tuple(d["reference_cvar"]["gaussian_on_referee_moments_rel_gap"].values()))
        f = d["referee_noise_floor"]
        print(f"  referee rel SE E median {f['rel_se_E_mc']['median']:.2e}; clark-corr/floor {f['clark-corr_med_relE_over_med_se']:.1f}x, "
              f"fosm {f['fosm_med_relE_over_med_se']:.0f}x; within 2 SE {f['clark-corr_windows_within_2se']}; sd SE {f['rel_se_sd_mc']['median']:.2e}")
        h = d["hybrid_clarkE_fosmSD"]
        print(f"  hybrid cvar err median {h['cvar_abs_err']['median']:.4f} p95 {h['cvar_abs_err']['p95']:.3f}; clark-corr lower in {h['clark-corr_lower_in']} (p={h['sign_test_p']:.2g})")
        print("  table:", {a: (round(v["median"], 4), round(v["p95"], 3)) for a, v in d["table_cvar"].items()})
        print("  ratios:", {k: round(v, 3) for k, v in d["ratios"].items()})
        print("  decisions:", d["decision_paired"])
        tl = d["traverse_level"]
        print("  traverse-level:", {k: (v["min"], v["majority_in"], f"{v['sign_test_p']:.1e}") for k, v in tl.items() if isinstance(v, dict) and "majority_in" in v},
              "bootstrap ratios E / CVaR:", [round(x, 1) for x in tl["cluster_bootstrap_95"]["ratio_relE_fosm_over_clark-corr"]],
              [round(x, 1) for x in tl["cluster_bootstrap_95"]["ratio_cvar_err_fosm_over_clark-corr"]])
        print("  threshold verdict disagreement (q25/q50/q75/q90):",
              {a: [round(v[str(q)]["disagree"], 3) for q in (0.25, 0.5, 0.75, 0.9)]
               for a, v in d["threshold_verdicts"]["arms"].items()})
    print("wrote", out_dir / "posthoc_review.json")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "out" / "v2")
