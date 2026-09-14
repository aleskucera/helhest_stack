"""Softmax (log-sum-exp) arms on the v2 windows (post hoc, 2026-09-14).

The ICRA-style review (reviewer 2, W3) noted that Section III-C refutes the
means-only softmax and that no softmax is scored, while the paper's own
constant tau = a / 1.702 shows how to build a belief-scaled one. Both are
scored here on the same abstract windows and the same 20k referee as the
other post hoc arms (v2_unscented.py):

  lse-<tau>     the softened contact at ONE temperature tau [m] for every
                node: node mean LSE_tau(mu) = tau log sum exp(mu_i / tau),
                node weights w = softmax(mu / tau), first-order propagation
                e ~ LSE + w . (h - mu), so the supports are affine in the
                belief and their covariance is w_j' C[idx_j, idx_k] w_k; the
                cost's moments follow from the quadratic-form identities.
                Swept over TAUS; the best tau on these windows is an upper
                bound for the practice, as the UT row was.
  lse-belief    the same with a per-node temperature tau_j = a_j / 1.702,
                a_j the contest width (eq. diffwidth) of the node's two
                highest-mean candidates (floored at 1e-6 m): the softmax
                that reads the belief's widths, i.e. the construction the
                review proposes.

Scored beside clark-corr (law a = 0.110, b = 1.0209), clark, fosm and the
hybrid from the same window. Design and held-out windows both included.

    PYTHONPATH=studies_v2 BASEPROD_ROOT=... python -m baseprod.v2_softmax <windows_root> <out_dir> [--workers N]
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

from .v2_arms import arm_clark, arm_fosm, sample_costs
from .v2_correction import law
from .v2_gate_sweep import abstract_window
from .v2_moments import gaussian_cvar, quadratic_moments

LAW = {"a": 0.110, "b": 1.0209}
N_REF = 20_000
Q = 0.9
TAUS = (0.005, 0.01, 0.02, 0.05, 0.1)   # metres
LOGISTIC_MATCH = 1.702                   # Section III-C: tau = a / 1.702 matches Phi(Delta / a)


def softmax_supports(mu_all, C_all, node_slices, taus):
    """Node means LSE_tau(mu) and the affine supports' covariance for a per-node tau."""
    n = len(node_slices)
    m = np.empty(n)
    W = []  # per node: (idx, weights)
    for j, idx in enumerate(node_slices):
        idx = np.asarray(idx)
        mu = mu_all[idx]
        t = float(taus[j])
        z = (mu - mu.max()) / t
        w = np.exp(z); w /= w.sum()
        m[j] = mu.max() + t * np.log(np.exp(z).sum())
        W.append((idx, w))
    S = np.empty((n, n))
    for j, (ij, wj) in enumerate(W):
        for k, (ik, wk) in enumerate(W):
            if k < j:
                S[j, k] = S[k, j]; continue
            S[j, k] = wj @ C_all[np.ix_(ij, ik)] @ wk
    return m, S


def belief_taus(mu_all, C_all, node_slices):
    out = np.empty(len(node_slices))
    for j, idx in enumerate(node_slices):
        idx = np.asarray(idx)
        mu = mu_all[idx]
        if len(idx) < 2:
            out[j] = 1e-6; continue
        o = np.argsort(-mu)[:2]
        i1, i2 = idx[o[0]], idx[o[1]]
        a2 = C_all[i1, i1] + C_all[i2, i2] - 2.0 * C_all[i1, i2]
        out[j] = max(np.sqrt(max(a2, 0.0)) / LOGISTIC_MATCH, 1e-6)
    return out


def arm_softmax(mu_all, C_all, node_slices, G, taus):
    m, S = softmax_supports(mu_all, C_all, node_slices, taus)
    E, V = quadratic_moments(m, S, G.T @ G)
    return E, np.sqrt(V)


def one_window(p: Path):
    r = abstract_window(p)
    if r is None:
        return None
    base, w = r
    if w is None:
        return base
    mu_all, C, slices, G, alpha_w = w["mu_all"], w["C"], w["slices"], w["G"], w["alpha_w"]
    seed = zlib.crc32(base["name"].encode()) % 2 ** 31
    ref = sample_costs(mu_all, C, slices, G, N_REF, seed)
    E_mc, sd_mc = float(ref.mean()), float(ref.std(ddof=1))
    cvar_mc = float(np.sort(ref)[int(np.ceil(Q * len(ref))):].mean())
    base["mc"] = {"n_draws": N_REF, "seed": seed, "E_mc": E_mc, "sd_mc": sd_mc, "cvar_mc": cvar_mc}
    E_c, sd_c = arm_clark(mu_all, C, slices, G)
    corr = 1.0 / float(law(alpha_w, LAW["a"], LAW["b"]))
    E_f, sd_f = arm_fosm(mu_all, C, slices, G)
    arms = {"clark-corr": (E_c, sd_c * corr), "clark": (E_c, sd_c), "fosm": (E_f, sd_f), "hybrid": (E_c, sd_f)}
    n = len(slices)
    for t in TAUS:
        arms[f"lse-{t:g}"] = arm_softmax(mu_all, C, slices, G, np.full(n, t))
    bt = belief_taus(mu_all, C, slices)
    arms["lse-belief"] = arm_softmax(mu_all, C, slices, G, bt)
    base["belief_tau_median_m"] = float(np.median(bt))
    base["arms"] = {a: {"E": E, "sd": sd, "rel_err_E": abs(E / E_mc - 1), "rel_err_sd": abs(sd / sd_mc - 1),
                        "sd_ratio_to_mc": sd / sd_mc, "cvar_abs_err": abs(gaussian_cvar(E, sd, Q) - cvar_mc)}
                    for a, (E, sd) in arms.items()}
    return base


ARMS = ("clark-corr", "clark", "fosm", "hybrid") + tuple(f"lse-{t:g}" for t in TAUS) + ("lse-belief",)


def summarize(rows):
    out = {}
    ok = [r for r in rows if r.get("scoreable") and not r["flagged"] and "arms" in r]
    for cond in sorted({r["condition"] for r in ok}):
        rc = [r for r in ok if r["condition"] == cond]
        d = {"n_windows": len(rc), "belief_tau_median_m": float(np.median([r["belief_tau_median_m"] for r in rc])), "arms": {}}
        for a in ARMS:
            d["arms"][a] = {k: float(np.median([r["arms"][a][k] for r in rc])) for k in ("rel_err_E", "rel_err_sd", "sd_ratio_to_mc", "cvar_abs_err")}
            d["arms"][a]["cvar_abs_err_p95"] = float(np.quantile([r["arms"][a]["cvar_abs_err"] for r in rc], 0.95, method="lower"))
        best = min((a for a in ARMS if a.startswith("lse-") and a != "lse-belief"), key=lambda a: d["arms"][a]["rel_err_E"])
        d["best_global_tau_arm"] = best
        d["paired"] = {}
        for a in (best, "lse-belief"):
            d["paired"][a] = {"E_err_below_fosm": int(sum(r["arms"][a]["rel_err_E"] < r["arms"]["fosm"]["rel_err_E"] for r in rc)),
                              "clark_E_err_below": int(sum(r["arms"]["clark"]["rel_err_E"] < r["arms"][a]["rel_err_E"] for r in rc)),
                              "clark-corr_sd_err_below": int(sum(r["arms"]["clark-corr"]["rel_err_sd"] < r["arms"][a]["rel_err_sd"] for r in rc)),
                              "clark-corr_cvar_err_below": int(sum(r["arms"]["clark-corr"]["cvar_abs_err"] < r["arms"][a]["cvar_abs_err"] for r in rc)),
                              "hybrid_cvar_err_below": int(sum(r["arms"]["hybrid"]["cvar_abs_err"] < r["arms"][a]["cvar_abs_err"] for r in rc))}
        out[cond] = d
    return out


def print_summary(s):
    for cond, d in s.items():
        if cond.startswith("_"):
            continue
        print(f"\n== {cond}: {d['n_windows']} windows, belief tau median {d['belief_tau_median_m']*100:.2f} cm, best global tau arm {d['best_global_tau_arm']}")
        for a, x in d["arms"].items():
            print(f"  {a:11s} relE {x['rel_err_E']:.2e} relSD {x['rel_err_sd']:.3f} sd/mc {x['sd_ratio_to_mc']:.3f} cvar med {x['cvar_abs_err']:.4f} p95 {x['cvar_abs_err_p95']:.3f}")
        print("  paired:", d["paired"])


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
    rows_path = out / "softmax_rows.jsonl"
    rows = []
    with Pool(a.workers, maxtasksperchild=8) as pool, rows_path.open("w") as fh:
        for r in pool.imap_unordered(_job, jobs, chunksize=1):
            if r is None:
                continue
            fh.write(json.dumps(r) + "\n"); fh.flush()
            rows.append(r)
    (out / "softmax_windows.json").write_text(json.dumps(rows))
    s = summarize(rows)
    s["_meta"] = {"written": time.strftime("%F %T"), "n_ref": N_REF, "windows_root": str(root), "taus_m": list(TAUS),
                  "logistic_match": LOGISTIC_MATCH,
                  "note": "post hoc; softened contact (LSE) at one global temperature (swept) and at the belief-scaled per-node temperature a/1.702; first-order propagation through the softmax weights; design and held-out windows both included"}
    (out / "softmax_summary.json").write_text(json.dumps(s, indent=1))
    print_summary(s)
    print("wrote", out)


if __name__ == "__main__":
    main()
