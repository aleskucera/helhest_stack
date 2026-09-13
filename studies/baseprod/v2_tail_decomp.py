"""Tail levels and the approximation-error decomposition on the held-out windows
(post hoc, 2026-09-13; the referee is re-drawn, seed crc32 of the window name,
so the numbers are the runner's up to Monte Carlo noise).

Per unflagged window:
  1. the referee's empirical CVaR at q = 0.90, 0.95, 0.99 (mean of the worst
     10 / 5 / 1 %), and the Gaussian CVaR of each arm at those levels;
  2. the decomposition arm "mc-moments": the referee draws' own support
     moments (mean vector and covariance of the per-node maxima) fed to the
     same quadratic-form identities the fold uses. Its E equals the referee's
     up to noise (the mean identity holds for any distribution); its sd
     differs from the referee's only by the Gaussian-supports assumption of
     the variance identity. So
        |sd_clark - sd_mcmom|  is the fold's moment-matching error, and
        |sd_mcmom - sd_ref|    is the quadratic-form Gaussian error,
     and the referee's skewness and excess kurtosis of J are recorded.

    PYTHONPATH=studies_v2 BASEPROD_ROOT=... python -m baseprod.v2_tail_decomp <windows_root> <out_dir> [--workers N]
"""
from __future__ import annotations

import argparse
import json
import os
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import norm

from .v2_arms import arm_clark, arm_fosm
from .v2_correction import law
from .v2_gate_sweep import abstract_window
from .v2_moments import quadratic_moments

LAW = {"a": 0.110, "b": 1.0209}
N_REF = 20_000
QS = (0.90, 0.95, 0.99)


def gcvar(E, sd, q):
    return E + norm.pdf(norm.ppf(q)) / (1.0 - q) * sd


def draws_with_supports(mu_all, C_all, node_slices, G, n, seed, chunk=4096):
    rng = np.random.default_rng(seed)
    Lc = np.linalg.cholesky(C_all + 1e-12 * np.eye(len(C_all)))
    nodes = [np.asarray(i) for i in node_slices]
    cost = np.empty(n)
    s1 = np.zeros(len(nodes))
    s2 = np.zeros((len(nodes), len(nodes)))
    done = 0
    while done < n:
        b = min(chunk, n - done)
        h = mu_all + rng.standard_normal((b, len(mu_all))) @ Lc.T
        e = np.stack([h[:, idx].max(axis=1) for idx in nodes], axis=1)
        x = e @ G.T
        cost[done:done + b] = np.einsum("ij,ij->i", x, x)
        s1 += e.sum(axis=0)
        s2 += e.T @ e
        done += b
    m = s1 / n
    S = s2 / n - np.outer(m, m)
    return cost, m, S


def one_window(p: Path):
    r = abstract_window(p)
    if r is None:
        return None
    base, w = r
    if w is None:
        return base
    mu_all, C, slices, G, alpha_w = w["mu_all"], w["C"], w["slices"], w["G"], w["alpha_w"]
    seed = zlib.crc32(base["name"].encode()) % 2 ** 31
    ref, m_mc, S_mc = draws_with_supports(mu_all, C, slices, G, N_REF, seed)
    E_mc, sd_mc = float(ref.mean()), float(ref.std(ddof=1))
    z = (ref - E_mc) / sd_mc
    srt = np.sort(ref)
    cv = {str(q): float(srt[int(np.ceil(q * len(ref))):].mean()) for q in QS}
    base["mc"] = {"n_draws": N_REF, "seed": seed, "E_mc": E_mc, "sd_mc": sd_mc, "cvar": cv,
                  "skew": float(np.mean(z ** 3)), "excess_kurtosis": float(np.mean(z ** 4) - 3.0)}
    A = G.T @ G
    E_q, V_q = quadratic_moments(m_mc, S_mc, A)
    E_c, sd_c = arm_clark(mu_all, C, slices, G)
    corr = 1.0 / float(law(alpha_w, LAW["a"], LAW["b"]))
    E_f, sd_f = arm_fosm(mu_all, C, slices, G)
    arms = {"clark": (E_c, sd_c), "clark-corr": (E_c, sd_c * corr), "fosm": (E_f, sd_f),
            "mc-moments": (E_q, float(np.sqrt(max(V_q, 0.0))))}
    base["alpha_weighted_median"] = alpha_w
    base["arms"] = {}
    for a, (E, sd) in arms.items():
        base["arms"][a] = {"E": E, "sd": sd, "rel_err_E": abs(E / E_mc - 1), "rel_err_sd": abs(sd / sd_mc - 1),
                           "sd_ratio_to_mc": sd / sd_mc,
                           "cvar_abs_err": {str(q): abs(gcvar(E, sd, q) - cv[str(q)]) for q in QS},
                           "cvar_rel_err": {str(q): abs(gcvar(E, sd, q) / cv[str(q)] - 1) for q in QS}}
    base["decomp"] = {"fold_sd_err_vs_mcmom": abs(sd_c / arms["mc-moments"][1] - 1),
                      "quadform_gauss_sd_err": abs(arms["mc-moments"][1] / sd_mc - 1),
                      "gaussian_cvar_on_true_moments_rel_err": {str(q): abs(gcvar(E_mc, sd_mc, q) / cv[str(q)] - 1) for q in QS}}
    return base


def summarize(rows):
    out = {}
    ok = [r for r in rows if r.get("scoreable") and not r["flagged"] and "arms" in r]
    for cond in sorted({r["condition"] for r in ok}):
        rc = [r for r in ok if r["condition"] == cond]
        d = {"n_windows": len(rc),
             "referee": {"skew_median": float(np.median([r["mc"]["skew"] for r in rc])),
                         "excess_kurtosis_median": float(np.median([r["mc"]["excess_kurtosis"] for r in rc]))},
             "arms": {}, "decomp": {}}
        for a in ("clark-corr", "clark", "fosm", "mc-moments"):
            d["arms"][a] = {"rel_err_E": float(np.median([r["arms"][a]["rel_err_E"] for r in rc])),
                            "rel_err_sd": float(np.median([r["arms"][a]["rel_err_sd"] for r in rc])),
                            "sd_ratio_to_mc": float(np.median([r["arms"][a]["sd_ratio_to_mc"] for r in rc])),
                            "cvar_abs_err_median": {str(q): float(np.median([r["arms"][a]["cvar_abs_err"][str(q)] for r in rc])) for q in QS},
                            "cvar_abs_err_p95": {str(q): float(np.quantile([r["arms"][a]["cvar_abs_err"][str(q)] for r in rc], 0.95, method="lower")) for q in QS},
                            "cvar_rel_err_median": {str(q): float(np.median([r["arms"][a]["cvar_rel_err"][str(q)] for r in rc])) for q in QS}}
        d["decomp"] = {"fold_sd_err_vs_mcmom_median": float(np.median([r["decomp"]["fold_sd_err_vs_mcmom"] for r in rc])),
                       "quadform_gauss_sd_err_median": float(np.median([r["decomp"]["quadform_gauss_sd_err"] for r in rc])),
                       "gaussian_cvar_on_true_moments_rel_err_median": {str(q): float(np.median([r["decomp"]["gaussian_cvar_on_true_moments_rel_err"][str(q)] for r in rc])) for q in QS},
                       "gaussian_cvar_on_true_moments_rel_err_p95": {str(q): float(np.quantile([r["decomp"]["gaussian_cvar_on_true_moments_rel_err"][str(q)] for r in rc], 0.95)) for q in QS}}
        out[cond] = d
    return out


def print_summary(s):
    for cond, d in s.items():
        if cond.startswith("_"):
            continue
        print(f"\n== {cond}: {d['n_windows']} windows; referee skew median {d['referee']['skew_median']:.2f}, excess kurtosis {d['referee']['excess_kurtosis_median']:.2f}")
        for a, x in d["arms"].items():
            print(f"  {a:11s} relE {x['rel_err_E']:.2e} relSD {x['rel_err_sd']:.3f} sd/mc {x['sd_ratio_to_mc']:.3f} | CVaR abs err med q90/95/99 "
                  + " / ".join(f"{x['cvar_abs_err_median'][str(q)]:.4f}" for q in QS) + " | rel med " + " / ".join(f"{x['cvar_rel_err_median'][str(q)]:.3f}" for q in QS))
        dd = d["decomp"]
        print(f"  decomposition: fold sd err vs true-moment quadratic form {dd['fold_sd_err_vs_mcmom_median']:.3f}; Gaussian quadratic-form sd err on true moments {dd['quadform_gauss_sd_err_median']:.3f}")
        print("  Gaussian CVaR on the referee's own moments, rel err median q90/95/99: " + " / ".join(f"{dd['gaussian_cvar_on_true_moments_rel_err_median'][str(q)]:.3f}" for q in QS)
              + "; p95: " + " / ".join(f"{dd['gaussian_cvar_on_true_moments_rel_err_p95'][str(q)]:.3f}" for q in QS))


def _job(args):
    p, cond = args
    r = one_window(p)
    if r is not None:
        r["condition"] = cond
        print(f"[{time.strftime('%T')}] {cond} {r['name']} " + ("flagged" if r["flagged"] else "ok" if r.get("scoreable") else "unscoreable"), flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("windows_root")
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    root, out = Path(a.windows_root), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(p, cond) for cond in ("foresight", "hindsight") for tdir in sorted((root / f"{cond}_v2").iterdir()) if tdir.is_dir()
            for p in sorted(tdir.glob("window_*.npz"))]
    print(f"{len(jobs)} windows, {a.workers} workers", flush=True)
    with Pool(a.workers) as pool:
        rows = [r for r in pool.map(_job, jobs, chunksize=1) if r is not None]
    (out / "tail_decomp_windows.json").write_text(json.dumps(rows))
    s = summarize(rows)
    s["_meta"] = {"written": time.strftime("%F %T"), "levels": QS, "n_ref": N_REF, "windows_root": str(root),
                  "note": "post hoc; referee re-drawn with crc32 seeds; design (t1, t2) and held-out traverses both included"}
    (out / "tail_decomp_summary.json").write_text(json.dumps(s, indent=1))
    print_summary(s)
    print("wrote", out)


if __name__ == "__main__":
    main()
