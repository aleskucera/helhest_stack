"""Add the C-FOSM (Clark-FOSM) arm to the tail decomposition, post hoc.

Written 2026-09-14, when the paper stopped reporting the fitted-scale arm
(clark-corr) and needed the tail readings for the arm it does propose.

No new draws and no re-scoring: v2_tail_decomp.py already stored, per window,
the fold's mean, linearization's standard deviation and the re-drawn
reference's CVaR at q = 0.90, 0.95, 0.99 (tail_decomp_windows.json).
Clark-FOSM is (clark.E, fosm.sd), so its Gaussian CVaR at each level, and
thus its error against the same reference, is a two-line derivation from that
file. The script asserts that the stored arms' medians reproduce from the
same rows before it reports the new one, which validates the window filter
and the lambda_q constants below against the committed summary.

Usage:  python -m studies.baseprod.v2_tail_decomp_cfosm studies/out/v2
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

QS = ("0.9", "0.95", "0.99")
# lambda_q = phi(Phi^-1(q)) / (1 - q); the same constant v2_tail_decomp.gcvar
# builds from scipy, written out so this script needs only numpy.
LAM = {"0.9": 0.17549833193248682 / 0.10,
       "0.95": 0.10313564315813918 / 0.05,
       "0.99": 0.02665214220345844 / 0.01}


def main(out_dir: Path):
    rows = [r for r in json.loads((out_dir / "tail_decomp" / "tail_decomp_windows.json").read_text())
            if r.get("scoreable") and not r["flagged"] and "arms" in r]   # v2_tail_decomp.summarize's filter
    ref = json.loads((out_dir / "tail_decomp" / "tail_decomp_summary.json").read_text())
    out = {"_meta": {"written": time.strftime("%F %T"),
                     "source": "tail_decomp/tail_decomp_windows.json (scoreable rows), no new draws",
                     "arm": "c-fosm = (clark E, fosm sd) through the Gaussian CVaR",
                     "lambda_q": LAM}}
    for cond in ("foresight", "hindsight"):
        rc = [r for r in rows if r["condition"] == cond]
        assert len(rc) == ref[cond]["n_windows"], (cond, len(rc), ref[cond]["n_windows"])
        # validate: the stored arms must reproduce the committed medians
        for a in ("clark", "clark-corr", "fosm"):
            for q in QS:
                got = float(np.median([r["arms"][a]["cvar_rel_err"][q] for r in rc]))
                want = ref[cond]["arms"][a]["cvar_rel_err_median"][q]
                assert abs(got - want) < 1e-12, (cond, a, q, got, want)
            E = np.array([r["arms"][a]["E"] for r in rc]); sd = np.array([r["arms"][a]["sd"] for r in rc])
            cv = np.array([r["mc"]["cvar"]["0.9"] for r in rc])
            chk = np.median(np.abs((E + LAM["0.9"] * sd) / cv - 1.0))
            assert abs(chk - ref[cond]["arms"][a]["cvar_rel_err_median"]["0.9"]) < 1e-9, (cond, a, chk)
        E = np.array([r["arms"]["clark"]["E"] for r in rc])
        sd = np.array([r["arms"]["fosm"]["sd"] for r in rc])
        sd_mc = np.array([r["mc"]["sd_mc"] for r in rc])
        E_mc = np.array([r["mc"]["E_mc"] for r in rc])
        d = {"n_windows": len(rc),
             "rel_err_E": float(np.median(np.abs(E / E_mc - 1.0))),
             "rel_err_sd": float(np.median(np.abs(sd / sd_mc - 1.0))),
             "sd_ratio_to_mc": float(np.median(sd / sd_mc)),
             "cvar_abs_err_median": {}, "cvar_abs_err_p95": {}, "cvar_rel_err_median": {}}
        for q in QS:
            cv = np.array([r["mc"]["cvar"][q] for r in rc])
            g = E + LAM[q] * sd
            d["cvar_abs_err_median"][q] = float(np.median(np.abs(g - cv)))
            d["cvar_abs_err_p95"][q] = float(np.quantile(np.abs(g - cv), 0.95, method="lower"))
            d["cvar_rel_err_median"][q] = float(np.median(np.abs(g / cv - 1.0)))
        out[cond] = {"c-fosm": d,
                     "fosm": ref[cond]["arms"]["fosm"],
                     "gaussian_cvar_on_true_moments_rel_err_median":
                         ref[cond]["decomp"]["gaussian_cvar_on_true_moments_rel_err_median"]}
        print(f"== {cond} n={len(rc)}")
        print("   c-fosm cvar rel err median q90/95/99: " + " / ".join(f"{d['cvar_rel_err_median'][q]:.4f}" for q in QS))
        print("   fosm   cvar rel err median q90/95/99: "
              + " / ".join(f"{ref[cond]['arms']['fosm']['cvar_rel_err_median'][q]:.4f}" for q in QS))
        print("   Gaussian floor on the referee's own moments: "
              + " / ".join(f"{ref[cond]['decomp']['gaussian_cvar_on_true_moments_rel_err_median'][q]:.4f}" for q in QS))
    p = out_dir / "tail_decomp" / "tail_decomp_cfosm.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "studies/out/v2"))
