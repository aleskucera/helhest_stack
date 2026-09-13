"""Unscented-transform baseline on the v2 windows (post hoc, 2026-09-13).

The reviewer's deterministic sampling-free baseline: the standard unscented
transform over the window's joint Gaussian of N candidate heights (2N + 1
sigma points, x = mu +/- sqrt(N + kappa) L e_i with L the Cholesky factor,
kappa = 3 - N clipped so that the centre weight stays in (0, 1], i.e. the
common choice kappa = 0 for large N), each point pushed through the true
pipeline (max per node, attitude quadratic), the cost's mean and variance
read from the weighted points. Scored like every other arm against the 20k
referee (re-drawn, seed crc32 of the window name), beside clark-corr,
clark and fosm from the same abstract window.

Cost: 2N + 1 cost evaluations per window against mc-32's 32 and the fold's
closed form; N is reported per window.

    PYTHONPATH=studies_v2 BASEPROD_ROOT=... python -m baseprod.v2_unscented <windows_root> <out_dir> [--workers N]
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
from .v2_moments import gaussian_cvar

LAW = {"a": 0.110, "b": 1.0209}
N_REF = 20_000
Q = 0.9


ALPHAS = (1.0, 0.3, 0.1, 0.03, 0.01)


def unscented_moments(mu_all, C_all, node_slices, G, alphas=ALPHAS, kappa=0.0, beta=2.0, chunk=2048):
    """Scaled UT (Julier 2002) for every alpha at once: sigma points
    mu +/- alpha sqrt(N + kappa) L e_i; mean weights w0 = lambda/(N+lambda),
    wi = 1/(2(N+lambda)), lambda = alpha^2 (N+kappa) - N; the centre's
    covariance weight adds (1 - alpha^2 + beta). alpha = 1 is the standard UT
    used first; alpha -> 0 tends to the mean map's mean and the delta
    method's variance."""
    N = len(mu_all)
    L = np.linalg.cholesky(C_all + 1e-12 * np.eye(N))
    nodes = [np.asarray(i) for i in node_slices]

    def costs(H):
        e = np.stack([H[:, idx].max(axis=1) for idx in nodes], axis=1)
        x = e @ G.T
        return np.einsum("ij,ij->i", x, x)

    c0 = costs(mu_all[None, :])[0]
    out = {}
    for al in alphas:
        lam = al * al * (N + kappa) - N
        s = np.sqrt(N + lam)
        w0m = lam / (N + lam)
        w0c = w0m + (1.0 - al * al + beta)
        wi = 1.0 / (2.0 * (N + lam))
        S1 = 0.0
        pts = []
        for a in range(0, N, chunk):
            D = s * L[:, a:a + chunk].T
            cp = costs(mu_all[None, :] + D)
            cm = costs(mu_all[None, :] - D)
            S1 += wi * (cp.sum() + cm.sum())
            pts.append(cp); pts.append(cm)
        E = w0m * c0 + S1
        dev = np.concatenate(pts) - E
        V = w0c * (c0 - E) ** 2 + wi * float((dev * dev).sum())
        out[str(al)] = (float(E), float(np.sqrt(max(V, 0.0))))
    return out, 2 * N + 1


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
    t0 = time.time()
    ut, n_pts = unscented_moments(mu_all, C, slices, G)
    t_ut = time.time() - t0
    E_u, sd_u = ut["1.0"]
    E_c, sd_c = arm_clark(mu_all, C, slices, G)
    corr = 1.0 / float(law(alpha_w, LAW["a"], LAW["b"]))
    E_f, sd_f = arm_fosm(mu_all, C, slices, G)
    sub = sample_costs(mu_all, C, slices, G, 32, seed + 132)
    arms = {"clark-corr": (E_c, sd_c * corr), "clark": (E_c, sd_c), "fosm": (E_f, sd_f),
            "unscented": (E_u, sd_u), "mc-32": (float(sub.mean()), float(sub.std(ddof=1)))}
    for al, (E, sd) in ut.items():
        arms[f"ut-{al}"] = (E, sd)
    base["arms"] = {a: {"E": E, "sd": sd, "rel_err_E": abs(E / E_mc - 1), "rel_err_sd": abs(sd / sd_mc - 1),
                        "sd_ratio_to_mc": sd / sd_mc, "cvar_abs_err": abs(gaussian_cvar(E, sd, Q) - cvar_mc)}
                    for a, (E, sd) in arms.items()}
    base["unscented"] = {"n_sigma_points": n_pts, "n_candidates": len(mu_all), "seconds": t_ut}
    return base


def summarize(rows):
    out = {}
    ok = [r for r in rows if r.get("scoreable") and not r["flagged"] and "arms" in r]
    for cond in sorted({r["condition"] for r in ok}):
        rc = [r for r in ok if r["condition"] == cond]
        d = {"n_windows": len(rc), "arms": {}, "sigma_points_median": float(np.median([r["unscented"]["n_sigma_points"] for r in rc]))}
        for a in ("clark-corr", "clark", "fosm", "unscented", "mc-32") + tuple(f"ut-{al}" for al in ALPHAS):
            d["arms"][a] = {k: float(np.median([r["arms"][a][k] for r in rc])) for k in ("rel_err_E", "rel_err_sd", "sd_ratio_to_mc", "cvar_abs_err")}
            d["arms"][a]["cvar_abs_err_p95"] = float(np.quantile([r["arms"][a]["cvar_abs_err"] for r in rc], 0.95, method="lower"))
        d["paired"] = {"clark-corr_E_below_ut": int(sum(r["arms"]["clark-corr"]["rel_err_E"] < r["arms"]["unscented"]["rel_err_E"] for r in rc)),
                       "clark-corr_sd_below_ut": int(sum(r["arms"]["clark-corr"]["rel_err_sd"] < r["arms"]["unscented"]["rel_err_sd"] for r in rc)),
                       "clark-corr_cvar_below_ut": int(sum(r["arms"]["clark-corr"]["cvar_abs_err"] < r["arms"]["unscented"]["cvar_abs_err"] for r in rc)),
                       "ut_E_below_fosm": int(sum(r["arms"]["unscented"]["rel_err_E"] < r["arms"]["fosm"]["rel_err_E"] for r in rc))}
        out[cond] = d
    return out


def print_summary(s):
    for cond, d in s.items():
        if cond.startswith("_"):
            continue
        print(f"\n== {cond}: {d['n_windows']} windows, median sigma points {d['sigma_points_median']:.0f}")
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
    with Pool(a.workers) as pool:
        rows = [r for r in pool.map(_job, jobs, chunksize=1) if r is not None]
    (out / "unscented_windows.json").write_text(json.dumps(rows))
    s = summarize(rows)
    s["_meta"] = {"written": time.strftime("%F %T"), "n_ref": N_REF, "windows_root": str(root),
                  "note": "post hoc; scaled UT (kappa = 0, beta = 2) at alpha in ALPHAS, 2N+1 sigma points; 'unscented' = alpha 1; design and held-out windows both included"}
    (out / "unscented_summary.json").write_text(json.dumps(s, indent=1))
    print_summary(s)
    print("wrote", out)


if __name__ == "__main__":
    main()
