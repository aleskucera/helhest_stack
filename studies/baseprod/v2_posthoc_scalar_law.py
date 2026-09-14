"""POST HOC (2026-09-14, after the ICRA-style review): does the alpha_w dependence of the
sd correction law show on the held-out record, or would a single scalar do as well?

Reads the committed held-out artifacts (out/v2/heldout_score_{foresight,hindsight}.json:
scoreable, unflagged windows; the arms 'clark' and 'clark-corr' as scored, law a = 0.110,
b = 1.0209 frozen from F1.json). Reports, per condition: the raw fold sd ratio and the
frozen law across six alpha_w-quantile bins (the bins of Fig. 3), the rise of each across
the bins, the single scalar 1 / median(raw ratio), and the median |sd ratio - 1| under the
scalar and under the law. Also the reference CVaR_0.9 median (units of the cost) for the
Table II caption. No new draws; no refit of the law.

    python -m studies.baseprod.v2_posthoc_scalar_law
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "out" / "v2"
A, B = 0.110, 1.0209


def law(x):
    return 1.0 - A * np.exp(-B * np.asarray(x, float))


def main():
    res = {"law": {"a": A, "b": B}, "conditions": {}}
    for cond in ("foresight", "hindsight"):
        d = json.loads((OUT / f"heldout_score_{cond}.json").read_text())
        rows = [w for w in (d if isinstance(d, list) else d["windows"]) if w.get("scoreable") and not w.get("flagged") and "arms" in w]
        aw = np.array([w["alpha_weighted_median"] for w in rows])
        raw = np.array([w["arms"]["clark"]["sd_ratio_to_mc"] for w in rows])
        cor = np.array([w["arms"]["clark-corr"]["sd_ratio_to_mc"] for w in rows])
        cv = np.array([w["mc"]["cvar_mc"] for w in rows])
        q = np.quantile(aw, np.linspace(0, 1, 7))
        bins = []
        for i in range(6):
            m = (aw >= q[i]) & (aw <= q[i + 1])
            bins.append({"alpha_w_median": float(np.median(aw[m])), "raw_median": float(np.median(raw[m])),
                         "law_at_median": float(law(np.median(aw[m]))), "corrected_median": float(np.median(cor[m])),
                         "n": int(m.sum())})
        scalar = 1.0 / float(np.median(raw))
        res["conditions"][cond] = {
            "n_windows": len(rows), "bins": bins,
            "raw_rise_over_bins": bins[-1]["raw_median"] - bins[0]["raw_median"],
            "law_rise_over_bins": bins[-1]["law_at_median"] - bins[0]["law_at_median"],
            "single_scalar": scalar,
            "median_abs_sd_err_scalar": float(np.median(np.abs(raw * scalar - 1.0))),
            "median_abs_sd_err_law": float(np.median(np.abs(cor - 1.0))),
            "corr_raw_ratio_vs_alpha_w": float(np.corrcoef(raw, aw)[0, 1]),
            "reference_cvar_median": float(np.median(cv)),
            "reference_cvar_p5_p95": [float(np.percentile(cv, 5)), float(np.percentile(cv, 95))],
        }
        c = res["conditions"][cond]
        print(f"{cond}: n {c['n_windows']}, raw rise {c['raw_rise_over_bins']:+.3f}, law rise {c['law_rise_over_bins']:+.3f}, "
              f"scalar {scalar:.3f}: median |err| scalar {c['median_abs_sd_err_scalar']:.4f} vs law {c['median_abs_sd_err_law']:.4f}; "
              f"reference CVaR_0.9 median {c['reference_cvar_median']:.3f}")
    (OUT / "posthoc_scalar_law.json").write_text(json.dumps(res, indent=1))
    print("wrote", OUT / "posthoc_scalar_law.json")


if __name__ == "__main__":
    main()
