"""POST-HOC analysis of E2-ii: the effect of excluding the one held-out
traverse whose DSM-to-RTK vertical registration failed.

Provenance rules of this file:
- INPUT: only the committed held-out artifacts
  (studies/out/e2/heldout_score_{foresight,hindsight}.json). No rebuild, no
  re-scoring, no new draws; the frozen per-condition k is read from the
  artifact and never refitted.
- The as-scored values are recomputed from the per-window rows over the
  EXACT pooled window list recorded in the artifact and ASSERTED equal to
  the committed criteria values before any exclusion is applied.
- STATUS: post-hoc. The exclusion was decided after the pre-registered
  verdicts were recorded. E2-ii FAILED as pre-registered and that verdict
  stands; this analysis quantifies how much of the failure the single
  defective traverse carries. The discriminator (per-traverse block-CV
  registration RMSE, register.py) is computed before scoring and references
  no scored outcome, but no registration gate was pre-registered.

The defective traverse: 2023-07-20_19-12-27. Its selected registration
model's block-CV RMSE is 3.3185 m against a median of 0.0463 m (max 0.3944)
across the other held-out traverses (FINAL_REPORT.md registration table);
its rover altitude wanders 10.42 m over an 835 s traverse at 0.081 m/s
horizontal speed. Any block-CV gate in (0.394, 3.319) m excludes exactly
this traverse and nothing else.

Run:  python -m studies.baseprod.posthoc_registration_exclusion
Writes: studies/out/e2/posthoc_registration_exclusion.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "out" / "e2"
EXCLUDED_TRAVERSE = "2023-07-20_19-12-27"
BAND = (0.7, 1.4)


def std(xs):
    # sample std (ddof=1), matching the pipeline's std(z)
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def pooled_stats(rows, k, arm="clark"):
    zs = [(r["j_truth"] - r["arms"][arm][0]) / (math.sqrt(k) * r["arms"][arm][1])
          for r in rows]
    return {
        "n_windows": len(zs),
        "sd_ratio": std(zs),
        "coverage_1sigma": sum(abs(z) <= 1.0 for z in zs) / len(zs),
        "mean_z": sum(zs) / len(zs),
    }


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def belief_referee_stats(rows):
    cl = [r["mc"]["arms"]["clark"] for r in rows]
    fo = [r["mc"]["arms"]["fosm"] for r in rows]
    m32 = [r["mc"]["arms"]["mc-32"] for r in rows]
    both = sum(c["rel_err_E"] < m["rel_err_E"] and c["rel_err_sd"] < m["rel_err_sd"]
               for c, m in zip(cl, m32))
    return {
        "n_windows": len(rows),
        "E2-iii_a_median_rel_err_E": median([c["rel_err_E"] for c in cl]),
        "E2-iii_a_clark_below_fosm_E": sum(c["rel_err_E"] < f["rel_err_E"] for c, f in zip(cl, fo)),
        "E2-iii_b_sd_ratio_to_mc": median([c["sd_ratio_to_mc"] for c in cl]),
        "E2-v_clark_below_mc32_both_moments": both,
        "clark_cvar_abs_err_median": median([c["cvar_abs_err"] for c in cl]),
    }


def bias_regression(rows, k):
    # residual (J_truth - E_clark) against bias*steps; slope predicted -1 by
    # the vehicle model (W_Z = 1, settle_map @ 1 = (1,0,0)).
    xs = [r["mu_minus_truth_med_m"] * r["n_steps"] for r in rows]
    ys = [r["j_truth"] - r["arms"]["clark"][0] for r in rows]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return {"n": n, "slope": slope, "intercept": intercept,
            "r2": 1.0 - ss_res / ss_tot, "predicted_slope": -1.0}


def run(cond):
    art = json.loads((OUT / f"heldout_score_{cond}.json").read_text())
    k = art["fit"]["k"]
    rows = {r["name"]: r for r in art["windows"]}
    pooled_names = art["pooled"]["windows"]
    pool = [rows[n] for n in pooled_names]

    as_scored = pooled_stats(pool, k)
    # assert reproduction of the committed criteria before excluding
    # (the criteria block stores values rounded to 4 decimals)
    c = art["criteria"]
    assert abs(as_scored["sd_ratio"] - c["E2-ii"]["value"]) < 5e-5, \
        (cond, as_scored["sd_ratio"], c["E2-ii"]["value"])
    assert abs(as_scored["coverage_1sigma"] - c["E2-i"]["value"]) < 5e-5

    kept = [r for r in pool if r["traverse"] != EXCLUDED_TRAVERSE]
    excluded = [r for r in pool if r["traverse"] == EXCLUDED_TRAVERSE]
    # 2026-09-14 addition (author's request for the Limitations paragraph):
    # the raw (k = 1) sd ratios of clark and fosm, and fosm at the frozen k,
    # both as scored (asserted against the pooled table) and excluding the
    # defective traverse. Same rows, same k, no new draws.
    pc, pf = art["pooled"]["clark"], art["pooled"]["fosm"]
    raw_c, raw_f = pooled_stats(pool, 1.0), pooled_stats(pool, 1.0, "fosm")
    rec_f = pooled_stats(pool, k, "fosm")
    assert abs(raw_c["sd_ratio"] - pc["raw"]["sd_ratio"]) < 5e-4, (cond, raw_c, pc["raw"])
    assert abs(raw_f["sd_ratio"] - pf["raw"]["sd_ratio"]) < 5e-4, (cond, raw_f, pf["raw"])
    assert abs(rec_f["coverage_1sigma"] - pf["recal"]["cov1"]) < 5e-4, (cond, rec_f, pf["recal"])
    assert abs(rec_f["sd_ratio"] - pf["recal"]["sd_ratio"]) < 5e-4
    extra = {
        "raw_k1_as_scored": {"clark": raw_c, "fosm": raw_f},
        "raw_k1_excluding_defective_traverse_POSTHOC": {
            "clark": pooled_stats(kept, 1.0), "fosm": pooled_stats(kept, 1.0, "fosm")},
        "fosm_frozen_k_as_scored": rec_f,
        "fosm_frozen_k_excluding_defective_traverse_POSTHOC": pooled_stats(kept, k, "fosm"),
    }
    return {
        **extra,
        "k_frozen": k,
        "excluded_traverse": EXCLUDED_TRAVERSE,
        "n_windows_excluded": len(excluded),
        "as_scored": as_scored,
        "excluding_defective_traverse_POSTHOC": pooled_stats(kept, k),
        "criterion_band_sd_ratio": list(BAND),
        "belief_referee_as_scored": belief_referee_stats(
            [r for r in pool if "mc" in r and r["mc"].get("arms")]),
        "belief_referee_excluding_POSTHOC": belief_referee_stats(
            [r for r in kept if "mc" in r and r["mc"].get("arms")]),
        "residual_vs_bias_x_steps_regression": bias_regression(pool, k),
    }


def main():
    result = {
        "status": "POST-HOC analysis of a pre-registered FAILURE. E2-ii "
                  "failed as pre-registered and that verdict stands; see "
                  "module docstring for the warrant and its limits.",
        "discriminator": {
            "quantity": "per-traverse block-CV registration RMSE (register.py, "
                        "computed before scoring, outcome-independent)",
            "excluded_traverse_value_m": 3.3185,
            "other_traverses_median_m": 0.0463,
            "other_traverses_max_m": 0.3944,
            "source": "FINAL_REPORT.md registration table",
            "gate_insensitivity": "any threshold in (0.394, 3.319) m excludes "
                                  "exactly this traverse",
        },
        "conditions": {cond: run(cond) for cond in ("foresight", "hindsight")},
    }
    out = OUT / "posthoc_registration_exclusion.json"
    out.write_text(json.dumps(result, indent=1))
    for cond, r in result["conditions"].items():
        print(cond, "as-scored sd_ratio %.4f -> excluding %.4f (n %d -> %d)" % (
            r["as_scored"]["sd_ratio"],
            r["excluding_defective_traverse_POSTHOC"]["sd_ratio"],
            r["as_scored"]["n_windows"],
            r["excluding_defective_traverse_POSTHOC"]["n_windows"]))
        print("  coverage %.4f -> %.4f" % (
            r["as_scored"]["coverage_1sigma"],
            r["excluding_defective_traverse_POSTHOC"]["coverage_1sigma"]))
        b = r["residual_vs_bias_x_steps_regression"]
        print("  regression slope %.4f (predicted -1), R^2 %.3f" % (b["slope"], b["r2"]))
        print("  raw sd_ratio clark %.2f -> %.2f, fosm %.2f -> %.2f; fosm recal cov %.4f -> %.4f, sd_ratio %.4f -> %.4f" % (
            r["raw_k1_as_scored"]["clark"]["sd_ratio"],
            r["raw_k1_excluding_defective_traverse_POSTHOC"]["clark"]["sd_ratio"],
            r["raw_k1_as_scored"]["fosm"]["sd_ratio"],
            r["raw_k1_excluding_defective_traverse_POSTHOC"]["fosm"]["sd_ratio"],
            r["fosm_frozen_k_as_scored"]["coverage_1sigma"],
            r["fosm_frozen_k_excluding_defective_traverse_POSTHOC"]["coverage_1sigma"],
            r["fosm_frozen_k_as_scored"]["sd_ratio"],
            r["fosm_frozen_k_excluding_defective_traverse_POSTHOC"]["sd_ratio"]))
        print("  cvar med %.4f -> %.4f" % (
            r["belief_referee_as_scored"]["clark_cvar_abs_err_median"],
            r["belief_referee_excluding_POSTHOC"]["clark_cvar_abs_err_median"]))
    print("wrote", out)


if __name__ == "__main__":
    main()
