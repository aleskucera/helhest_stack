"""Scoring: the QC flags, the two referees, and the (k, tau) fit -- one code path for the
design phase and the held-out run, and one code path for both belief conditions.

Two referees, kept separate exactly as PREREG_baseprod.md separates them:

  THE BELIEF REFEREE (E2-iii, PRIMARY -- the estimator claim). Monte Carlo drawn FROM THAT
  WINDOW'S OWN BELIEF, >= 20,000 draws, the identical belief every arm consumes; no truth, no
  recalibration, raw Var. Circular BY DESIGN: that is what makes it a test of the estimator
  alone with no mapper error in the loop.

  THE REALITY REFEREE (E2-i, E2-ii, E2-iv and the reported arm comparisons). The registered
  survey truth. It grades the WHOLE pipeline, mostly the mapper. NO pass/fail criterion
  attaches to any reality-side ARM comparison; it is reported with the prereg's fixed
  interpretation.

The recalibration is the E1-A2 family exactly -- Var' = k Var + (tau n_steps)^2, at most two
scalars, ML on UNFLAGGED DESIGN windows, applied identically to every variance-predicting arm,
E[cost] untouched (so mean-map is unaffected by construction). It is fitted PER CONDITION and
frozen. Truth-fitted per-window or per-traverse mean corrections are BANNED
(constants.BANNED_CORRECTIONS) and nothing in this module computes one.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from spires.risk_calibration import (SIGMA_FLOOR, Window, _fit_scalars, _mean_nll, arm_moments,
                                     build_nodes, cell_covariance, cell_variance_split, mc_cost,
                                     recalibrate_sd, resample_track)
from spires.vehicle import PAD_CAP, SETTLE_CONST_PER_STEP

from . import constants as K


# ---------------------------------------------------------------- one window
def case(p: Path, mc_draws: int = 0) -> dict | None:
    """Score one window product. Returns None if it carries no scoreable step at all."""
    d = np.load(p)
    win = Window(path=p, mu=d["mu"].astype(np.float64), sigma=d["sigma"].astype(np.float64),
                 raw_sd=d["raw_sd"].astype(np.float64), meas_sd=d["meas_sd"].astype(np.float64),
                 count=d["count"], x0=float(d["xmin"]), y0=float(d["ymin"]),
                 cell=float(d["cell"]), sx=float(d["sx"]), sy=float(d["sy"]),
                 gt_poses=d["gt_poses"])
    tmax = d["truth_max"].astype(np.float64)
    track = resample_track(win.gt_poses)
    if len(track) == 0:
        return None
    cell_flat, caps, c_eff = build_nodes(win, track)
    t_steps = len(track)
    mu_f, sg_f, tl_f = win.mu.ravel(), win.sigma.ravel(), tmax.ravel()
    cnt_f = win.count.ravel()
    usable = (np.isfinite(mu_f) & np.isfinite(sg_f) & (sg_f > SIGMA_FLOOR)
              & np.isfinite(tl_f) & (cnt_f > 0))

    # --- QC, on ALL element cells of the window, BEFORE any node filtering -----------------
    # (prereg: "computed for every window BEFORE scoring")
    real_all = caps > PAD_CAP + 1.0
    qc_cells = np.unique(cell_flat[real_all])
    obs_frac = float((cnt_f[qc_cells] > 0).mean())
    seen = qc_cells[cnt_f[qc_cells] > 0]
    sig_med_all = float(np.median(sg_f[seen])) if len(seen) else float("nan")
    void_frac = float((~np.isfinite(tl_f[qc_cells])).mean())
    flags = []
    if obs_frac < K.FLAG_MIN_OBS_FRAC:
        flags.append("coverage")
    if not (sig_med_all <= K.FLAG_MAX_SIGMA_M):
        flags.append("sigma")
    if void_frac > K.FLAG_MAX_INVALID_FRAC:
        flags.append("truth_invalid")

    node_ok = usable[cell_flat].all(axis=1).reshape(3 * t_steps, 4).all(axis=1)
    step_ok = node_ok.reshape(3, t_steps).all(axis=0)
    n_keep = int(step_ok.sum())
    base = {"name": f"{p.parent.name}/{p.stem}", "traverse": p.parent.name,
            "n_steps": n_keep, "n_steps_total": t_steps,
            "retained": n_keep / t_steps if t_steps else 0.0,
            "qc_elem_cells": int(len(qc_cells)),
            "qc_observed_frac": round(obs_frac, 4),
            "qc_sigma_med_m": round(sig_med_all, 5) if np.isfinite(sig_med_all) else None,
            "qc_truth_invalid_frac": round(void_frac, 5),
            "qc_flags": flags, "flagged": bool(flags)}
    if n_keep == 0:
        base["scoreable"] = False
        return base

    sel = np.repeat(np.tile(step_ok, 3), 4)
    cell_flat, caps, c_eff = cell_flat[sel], caps[sel], c_eff[sel]
    u_cells, inv = np.unique(cell_flat.ravel(), return_inverse=True)
    u_idx = inv.reshape(cell_flat.shape)
    vi, vg, vl = cell_variance_split(win)
    C = cell_covariance(u_cells, win, vi, vg, vl)
    mu_u = mu_f[u_cells]
    means = mu_u[u_idx] + caps
    env_t = (tl_f[u_cells][u_idx] + caps).max(axis=1)
    const = n_keep * SETTLE_CONST_PER_STEP
    j_truth = float(c_eff @ env_t) + const
    arms = {k: (e + const, sd) for k, (e, sd) in
            arm_moments(means, c_eff, u_idx, C, len(u_cells)).items()}

    # --- the condition variable, read directly out of the fold ----------------------------
    cov_self = C[u_idx[:, :, None], u_idx[:, None, :]]
    real = caps > PAD_CAP + 1.0
    m_masked = np.where(real, means, -np.inf)
    order = np.argsort(-m_masked, axis=1)
    i1, i2 = order[:, 0], order[:, 1]
    r = np.arange(means.shape[0])
    m1, m2 = m_masked[r, i1], m_masked[r, i2]
    v1, v2 = cov_self[r, i1, i1], cov_self[r, i2, i2]
    c12 = cov_self[r, i1, i2]
    denom = np.sqrt(np.maximum(v1 + v2 - 2.0 * c12, 1e-18))
    alpha = np.abs(m1 - m2) / denom
    ok2 = np.isfinite(alpha) & real[r, i2]
    wgt = np.abs(c_eff)[ok2]
    al = alpha[ok2]
    o = np.argsort(al)
    cw = np.cumsum(wgt[o]) / max(wgt.sum(), 1e-12)
    alpha_wmed = float(np.interp(0.5, cw, al[o])) if len(al) else float("nan")

    tl_u, sg_u = tl_f[u_cells], sg_f[u_cells]
    relief = float(np.percentile(tl_u, 95) - np.percentile(tl_u, 5))
    sigma_med = float(np.median(sg_u))
    base.update({
        "scoreable": True,
        "n_cells": int(len(u_cells)), "n_nodes": int(means.shape[0]),
        "j_truth": j_truth, "arms": {k: [v[0], v[1]] for k, v in arms.items()},
        "sigma_med_m": sigma_med, "relief_p95_p5_m": relief,
        "sigma_over_relief": sigma_med / relief if relief > 0 else float("nan"),
        "truth_cell_sd_m": float(np.std(tl_u)),
        "alpha_weighted_median": alpha_wmed,
        "frac_nodes_alpha_lt_1": float(np.average(al < 1.0, weights=wgt)) if len(al) else float("nan"),
        "frac_nodes_alpha_lt_2": float(np.average(al < 2.0, weights=wgt)) if len(al) else float("nan"),
        "mu_minus_truth_med_m": float(np.median(mu_u - tl_u)),
    })

    # --- E2-iii: the BELIEF referee, on this window's own belief ---------------------------
    if mc_draws:
        shim = SimpleNamespace(C=C, u_cells=u_cells, u_idx=u_idx, mu_u=mu_u,
                               caps=caps, c_eff=c_eff, const=const)
        e_mc, sd_mc, n_used, min_eig = mc_cost(shim, n_draws=mc_draws,
                                               seed=K.MC_SEED, chunk=K.MC_CHUNK)
        mc = {"n_draws": int(n_used), "E_mc": e_mc, "sd_mc": sd_mc,
              "min_eig_C": min_eig, "mc_se_of_mean_in_sd": 1.0 / math.sqrt(n_used),
              "arms": {}}
        for a in K.ARMS_SCORED:
            E, sd = arms[a]
            mc["arms"][a] = {
                "E": E, "sd": sd,
                "rel_err_E": abs(E - e_mc) / abs(e_mc) if e_mc else float("nan"),
                "rel_err_sd": abs(sd - sd_mc) / sd_mc if sd_mc > 0 else float("nan"),
                "E_err_in_sd": abs(E - e_mc) / sd_mc if sd_mc > 0 else float("nan"),
            }
        base["mc"] = mc
    return base


# ---------------------------------------------------------------- pooling
def sign_test(n_better: int, n: int) -> float:
    """Exact two-sided binomial sign test at p = 0.5."""
    if n == 0:
        return float("nan")
    m = min(n_better, n - n_better)
    tail = sum(math.comb(n, i) for i in range(m + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def _nll(r, arm, k, tau, suf):
    E, sd = r["arms"][arm]
    s = sd if suf == "raw" else recalibrate_sd(sd, r["n_steps"], k, tau)
    z = (r["j_truth"] - E) / s
    return 0.5 * math.log(2 * math.pi * s * s) + 0.5 * z * z


def fit(rows) -> tuple[dict, dict]:
    """ML fit of (k, tau) on the given windows, plus the pinned-tau (k = 1) alternative."""
    res = np.array([r["j_truth"] - r["arms"][K.FIT_ARM][0] for r in rows])
    sd = np.array([r["arms"][K.FIT_ARM][1] for r in rows])
    n = np.array([r["n_steps"] for r in rows], float)
    k, tau, nll_at_fit = _fit_scalars(res, sd, n)
    ts = np.geomspace(1e-6, 1e1, 4000)
    nll_t = _mean_nll(res, sd, n, np.ones(len(ts)), ts)
    j = int(np.argmin(nll_t))
    return ({"arm": K.FIT_ARM, "k": k, "tau_m": tau, "sd_multiplier_sqrt_k": math.sqrt(k),
             "mean_NLL_at_fit": nll_at_fit, "n_windows_fitted": len(rows)},
            {"k": 1.0, "tau_m": float(ts[j]), "mean_NLL_at_fit": float(nll_t[j]),
             "nll_penalty_vs_ML": round(float(nll_t[j]) - nll_at_fit, 4)})


def summarise(rows, k, tau, label) -> dict:
    if not rows:
        return {"n_windows": 0, "label": label}
    agg = {"label": label, "n_windows": len(rows), "k_used": k, "tau_used_m": tau,
           "windows": [r["name"] for r in rows]}
    for arm in K.ARMS_SCORED:
        E = np.array([r["arms"][arm][0] for r in rows])
        sd = np.array([r["arms"][arm][1] for r in rows])
        n = np.array([r["n_steps"] for r in rows], float)
        j = np.array([r["j_truth"] for r in rows])
        res = j - E
        for suf, s in (("raw", sd),
                       ("recal", np.array([recalibrate_sd(a, int(b), k, tau)
                                           for a, b in zip(sd, n)]))):
            z = res / s
            nll = 0.5 * np.log(2 * np.pi * s**2) + 0.5 * z**2
            agg.setdefault(arm, {})[suf] = {
                "mean_NLL": round(float(nll.mean()), 4),
                "cov1": round(float(np.mean(np.abs(z) <= 1)), 4),
                "cov2": round(float(np.mean(np.abs(z) <= 2)), 4),
                "sd_ratio": round(float(np.std(z, ddof=1)), 4) if len(z) > 1 else None,
                "mean_z": round(float(z.mean()), 4),
                "median_z": round(float(np.median(z)), 4),
                "median_sd": round(float(np.median(s)), 4),
                "mean_NLL_full": float(nll.mean()),
            }
    for a, b in (("clark", "fosm"), ("clark", "clark-diag")):
        for suf in ("raw", "recal"):
            dd = (np.array([_nll(r, a, k, tau, suf) for r in rows])
                  - np.array([_nll(r, b, k, tau, suf) for r in rows]))
            nb = int((dd < 0).sum())
            agg.setdefault(f"nll_{a}_minus_{b}", {})[suf] = {
                "mean": round(float(dd.mean()), 4), "median": round(float(np.median(dd)), 4),
                "n_windows_clark_better": nb, "n": len(dd),
                "sign_test_p_two_sided": round(sign_test(nb, len(dd)), 4)}
    raw_m = [agg[a]["raw"]["mean_NLL"] for a in K.ARMS_SCORED]
    rec_m = [agg[a]["recal"]["mean_NLL"] for a in K.ARMS_SCORED]
    sr, sc = max(raw_m) - min(raw_m), max(rec_m) - min(rec_m)
    agg["arm_NLL_spread"] = {
        "raw_nats": round(float(sr), 4), "recal_nats": round(float(sc), 4),
        "compression_factor": round(float(sr / sc), 1) if sc > 0 else None,
        "note": "max-min mean NLL across the three variance arms, before and after the "
                "recalibration; the factor is how much of the scale a ranking is read on "
                "the k-rescale destroys"}
    err = np.array([r["j_truth"] - r["arms"]["mean-map"][0] for r in rows])
    nst = np.array([r["n_steps"] for r in rows], float)
    agg["mean-map"] = {
        "note": "point prediction, no predicted variance: mean_NLL / cov1 / cov2 / sd_ratio / "
                "mean_z do not exist for this arm and are null by construction, not omitted",
        "mean_NLL": None, "cov1": None, "cov2": None, "sd_ratio": None, "mean_z": None,
        "mean_err": round(float(err.mean()), 4),
        "median_abs_err": round(float(np.median(np.abs(err))), 4),
        "median_abs_err_per_step_m": round(float(np.median(np.abs(err / nst))), 5)}
    for f in ("sigma_med_m", "relief_p95_p5_m", "sigma_over_relief", "truth_cell_sd_m",
              "alpha_weighted_median", "frac_nodes_alpha_lt_1", "frac_nodes_alpha_lt_2",
              "mu_minus_truth_med_m", "retained", "qc_observed_frac", "qc_sigma_med_m"):
        v = np.array([r[f] for r in rows if r.get(f) is not None], float)
        if not len(v):
            continue
        agg[f] = {q: round(float(np.percentile(v, pv)), 4)
                  for q, pv in (("min", 0), ("p5", 5), ("p25", 25), ("median", 50),
                                ("p75", 75), ("p95", 95), ("max", 100))}
    return agg


def mc_summary(rows) -> dict:
    """E2-iii: the belief referee, pooled. Raw Var, no truth, no recalibration."""
    rows = [r for r in rows if "mc" in r]
    if not rows:
        return {"n_windows": 0, "note": "no MC drawn"}
    out = {"n_windows": len(rows), "n_draws_per_window": rows[0]["mc"]["n_draws"],
           "mc_seed": K.MC_SEED, "arms": {}}
    for a in K.ARMS_SCORED:
        rE = np.array([r["mc"]["arms"][a]["rel_err_E"] for r in rows])
        rS = np.array([r["mc"]["arms"][a]["rel_err_sd"] for r in rows])
        out["arms"][a] = {
            "median_rel_err_E": float(np.median(rE)), "median_rel_err_sd": float(np.median(rS)),
            "p95_rel_err_E": float(np.percentile(rE, 95)),
            "p95_rel_err_sd": float(np.percentile(rS, 95)),
            "max_rel_err_E": float(rE.max()), "max_rel_err_sd": float(rS.max())}
    for a, b in (("clark", "fosm"), ("clark", "clark-diag")):
        wE = int(sum(r["mc"]["arms"][a]["rel_err_E"] < r["mc"]["arms"][b]["rel_err_E"]
                     for r in rows))
        wS = int(sum(r["mc"]["arms"][a]["rel_err_sd"] < r["mc"]["arms"][b]["rel_err_sd"]
                     for r in rows))
        wB = int(sum((r["mc"]["arms"][a]["rel_err_E"] < r["mc"]["arms"][b]["rel_err_E"])
                     and (r["mc"]["arms"][a]["rel_err_sd"] < r["mc"]["arms"][b]["rel_err_sd"])
                     for r in rows))
        n = len(rows)
        out[f"{a}_vs_{b}"] = {
            "n": n,
            "n_clark_better_E": wE, "sign_test_p_E": round(sign_test(wE, n), 5),
            "n_clark_better_sd": wS, "sign_test_p_sd": round(sign_test(wS, n), 5),
            "n_clark_better_BOTH_moments": wB, "sign_test_p_both": round(sign_test(wB, n), 5)}
    return out


# ---------------------------------------------------------------- criteria
def evaluate_criteria(pooled: dict, mc: dict, all_windows: list) -> dict:
    """E2-i, E2-ii, E2-iii, E2-iv, verbatim from PREREG_baseprod.md. Held-out only.

    No criterion attaches to any reality-side arm comparison, or to clark-vs-diag under
    either referee; those are REPORTED and carry the prereg's fixed interpretation.
    """
    out = {}
    c1 = pooled.get("clark", {}).get("recal", {})
    lo, hi = K.E2_I_COV1_BAND
    v = c1.get("cov1")
    out["E2-i"] = {"statement": f"clark |z| <= 1 coverage in [{lo}, {hi}]",
                   "value": v, "pass": bool(v is not None and lo <= v <= hi)}
    lo, hi = K.E2_II_SD_RATIO_BAND
    v = c1.get("sd_ratio")
    out["E2-ii"] = {"statement": f"clark sd-ratio in [{lo}, {hi}]",
                    "value": v, "pass": bool(v is not None and lo <= v <= hi)}
    ck = mc.get("arms", {}).get("clark", {})
    cf = mc.get("clark_vs_fosm", {})
    a_ok = bool(ck and ck["median_rel_err_E"] <= K.E2_III_MAX_MEDIAN_REL_ERR
                and ck["median_rel_err_sd"] <= K.E2_III_MAX_MEDIAN_REL_ERR)
    n = cf.get("n", 0)
    b_ok = bool(n and cf["n_clark_better_BOTH_moments"] > n / 2
                and cf["sign_test_p_both"] < K.E2_III_SIGN_TEST_ALPHA)
    out["E2-iii"] = {
        "statement": "(a) clark median relative errors vs the belief MC each <= 5%; "
                     "(b) clark's error below fosm's on BOTH moments in a majority of "
                     "windows, two-sided exact sign test p < 0.05",
        "a_median_rel_err_E": ck.get("median_rel_err_E"),
        "a_median_rel_err_sd": ck.get("median_rel_err_sd"), "a_pass": a_ok,
        "b_n_clark_better_both": cf.get("n_clark_better_BOTH_moments"), "b_n": n,
        "b_sign_test_p": cf.get("sign_test_p_both"), "b_pass": b_ok,
        "pass": bool(a_ok and b_ok)}
    scoreable = [r for r in all_windows if r.get("scoreable")]
    k, tau = pooled.get("k_used"), pooled.get("tau_used_m")
    if scoreable and k is not None:
        z = np.array([(r["j_truth"] - r["arms"]["clark"][0])
                      / recalibrate_sd(r["arms"]["clark"][1], r["n_steps"], k, tau)
                      for r in scoreable])
        out["E2-iv"] = {"statement": "clark |z| <= 2 coverage, reported unconditionally, "
                                     "flagged pool included (no pass/fail)",
                        "value": round(float(np.mean(np.abs(z) <= 2)), 4),
                        "n_windows": len(z), "pass": None}
    out["reality_side_arm_comparisons"] = {
        "criterion": None,
        "interpretation": ("Fixed before any result existed: the reality comparison between "
                           "arms conflates mapper and estimator. If the belief is faithful, "
                           "the fold's advantage is expected to appear against reality too; "
                           "if the belief is biased, ANY arm -- including cruder ones -- can "
                           "win, because an estimator faithful to a wrong belief amplifies "
                           "the belief's error while a crude one ignores the term the error "
                           "feeds. Such an outcome grades the mapper, not the estimator.")}
    return out


# ---------------------------------------------------------------- driver
def score_condition(win_dirs, condition: str, k=None, tau=None, mc_draws: int = 0,
                    label: str = "") -> dict:
    """Score every window under one condition. If (k, tau) is None it is fitted here."""
    rows = []
    for wd in win_dirs:
        for p in sorted(Path(wd).glob("window_*.npz")):
            r = case(p, mc_draws=mc_draws)
            if r is not None:
                rows.append(r)
    excluded = [r["name"] for r in rows
                if not r.get("scoreable") or r["retained"] < K.MIN_RETAINED]
    kept = [r for r in rows if r.get("scoreable") and r["retained"] >= K.MIN_RETAINED]
    unflagged = [r for r in kept if not r["flagged"]]
    fitted_here = k is None
    if fitted_here:
        f, pin = fit(unflagged if unflagged else kept)
        k, tau = f["k"], f["tau_m"]
    else:
        f, pin = {"k": k, "tau_m": tau, "note": "inherited, not refitted"}, None
    out = {"condition": condition, "label": label,
           "qc": {"rule": "PREREG_baseprod.md QC flags",
                  "n_window_products": len(rows), "n_retained": len(kept),
                  "n_unflagged": len(unflagged), "n_excluded": len(excluded),
                  "excluded": excluded,
                  "flagged": {r["name"]: r["qc_flags"] for r in kept if r["flagged"]}},
           "fit": f, "fit_pinned_tau_alternative": pin,
           "pooled": summarise(unflagged, k, tau, f"{condition}: pooled, UNFLAGGED windows"),
           "pooled_all_retained": summarise(kept, k, tau,
                                            f"{condition}: pooled, all retained windows"),
           "belief_referee_E2iii": mc_summary(unflagged),
           "belief_referee_E2iii_all_retained": mc_summary(kept),
           "windows": rows}
    return out
