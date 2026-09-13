"""Fold-order ablation on the DESIGN traverses (post hoc, 2026-09-13).

The Clark fold is pairwise and moment-matches after every pair, so the result
depends on the order in which a node's candidates are folded (the true
maximum does not). The paper folds in descending order of mean and says the
effect was not studied. This studies it: every node of every design window is
folded in five orders, and the window's E[J] and sd[J] are scored against the
same 20k-draw referee as v2_gate_sweep (seed crc32 of the window name).

  desc      descending mean (the paper's order)
  asc       ascending mean
  var_desc  descending variance
  var_asc   ascending variance
  random    a fixed random permutation per node (seed = window crc32 + node index)

    PYTHONPATH=studies_v2 BASEPROD_ROOT=... python -m baseprod.v2_fold_order <windows_root> <out_dir> [--workers N]
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

from .v2_arms import arm_fosm, sample_costs
from .v2_gate_sweep import abstract_window
from .v2_moments import _clark_pair, gaussian_cvar, quadratic_moments

ORDERS = ("desc", "asc", "var_desc", "var_asc", "random")
N_REF = 20_000
Q = 0.9


def fold_ordered(mu_all, C_all, node_slices, real, order_name, seed):
    n, N = len(node_slices), len(mu_all)
    M = np.zeros((N + n, N + n))
    M[:N, :N] = C_all
    mu_aug = np.concatenate([mu_all, np.zeros(n)])
    rng = np.random.default_rng(seed)
    for i, idx in enumerate(node_slices):
        idx = np.asarray(idx)[real[np.asarray(idx)]]           # real candidates only
        if len(idx) == 0:
            raise ValueError("node with no real candidate")
        if order_name == "desc":
            order = idx[np.argsort(-mu_all[idx], kind="stable")]
        elif order_name == "asc":
            order = idx[np.argsort(mu_all[idx], kind="stable")]
        elif order_name == "var_desc":
            order = idx[np.argsort(-np.diag(C_all)[idx], kind="stable")]
        elif order_name == "var_asc":
            order = idx[np.argsort(np.diag(C_all)[idx], kind="stable")]
        elif order_name == "random":
            order = rng.permutation(idx)
        else:
            raise ValueError(order_name)
        j0 = order[0]
        shift = mu_aug[j0]
        m_run, v_run = 0.0, M[j0, j0]
        c_run = M[j0, :N + i].copy()
        for j in order[1:]:
            m_run, v_run, c_run = _clark_pair(m_run, v_run, mu_aug[j] - shift, M[j, j],
                                              c_run[j], c_run, M[j, :N + i])
        k = N + i
        M[k, :N + i] = c_run
        M[:N + i, k] = c_run
        M[k, k] = v_run
        mu_aug[k] = m_run + shift
    return mu_aug[N:], M[N:, N:]


def one_window(p: Path) -> dict | None:
    r = abstract_window(p)
    if r is None:
        return None
    base, w = r
    if w is None:
        return base
    mu_all, C, slices, G, real = w["mu_all"], w["C"], w["slices"], w["G"], w["real"]
    seed = zlib.crc32(base["name"].encode()) % 2 ** 31
    ref = sample_costs(mu_all, C, slices, G, N_REF, seed)
    E_mc, sd_mc = float(ref.mean()), float(ref.std(ddof=1))
    tail = np.sort(ref)[int(np.ceil(Q * len(ref))):]
    cvar_mc = float(tail.mean())
    base["mc"] = {"n_draws": N_REF, "E_mc": E_mc, "sd_mc": sd_mc, "cvar_mc": cvar_mc, "seed": seed}
    A = G.T @ G
    base["orders"] = {}
    for o in ORDERS:
        m, S = fold_ordered(mu_all, C, slices, real, o, seed + 7)
        E, V = quadratic_moments(m, S, A)
        sd = float(np.sqrt(V))
        base["orders"][o] = {"E": E, "sd": sd, "rel_err_E": abs(E / E_mc - 1), "rel_err_sd": abs(sd / sd_mc - 1),
                             "sd_ratio_to_mc": sd / sd_mc, "cvar_abs_err": abs(gaussian_cvar(E, sd, Q) - cvar_mc)}
    E_f, sd_f = arm_fosm(mu_all, C, slices, G)
    base["fosm"] = {"E": E_f, "sd": sd_f, "rel_err_E": abs(E_f / E_mc - 1), "rel_err_sd": abs(sd_f / sd_mc - 1)}
    # sanity: desc must equal the frozen clark arm (pads skipped here, kept-as-leader there: same result)
    from .v2_arms import arm_clark
    E_c, sd_c = arm_clark(mu_all, C, slices, G)
    assert abs(base["orders"]["desc"]["E"] - E_c) <= 1e-9 * max(1, abs(E_c)), (base["name"], base["orders"]["desc"]["E"], E_c)
    assert abs(base["orders"]["desc"]["sd"] - sd_c) <= 1e-9 * max(1, sd_c), (base["name"], base["orders"]["desc"]["sd"], sd_c)
    return base


def summarize(rows):
    out = {}
    ok = [r for r in rows if r.get("scoreable") and not r["flagged"] and "orders" in r]
    for cond in sorted({r["condition"] for r in ok}):
        rc = [r for r in ok if r["condition"] == cond]
        d = {"n_windows": len(rc), "orders": {}}
        for o in ORDERS:
            g = [r["orders"][o] for r in rc]
            d["orders"][o] = {k: float(np.median([x[k] for x in g])) for k in ("rel_err_E", "rel_err_sd", "sd_ratio_to_mc", "cvar_abs_err")}
            d["orders"][o]["max_rel_err_E"] = float(max(x["rel_err_E"] for x in g))
            if o != "desc":
                dE = [abs(x["E"] / r["orders"]["desc"]["E"] - 1) for x, r in zip(g, rc)]
                dS = [abs(x["sd"] / r["orders"]["desc"]["sd"] - 1) for x, r in zip(g, rc)]
                d["orders"][o]["vs_desc"] = {"rel_dE_median": float(np.median(dE)), "rel_dE_max": float(max(dE)),
                                             "rel_dsd_median": float(np.median(dS)), "rel_dsd_max": float(max(dS)),
                                             "E_better_than_desc": int(sum(x["rel_err_E"] < r["orders"]["desc"]["rel_err_E"] for x, r in zip(g, rc)))}
        d["fosm"] = {k: float(np.median([r["fosm"][k] for r in rc])) for k in ("rel_err_E", "rel_err_sd")}
        out[cond] = d
    return out


def print_summary(s):
    for cond, d in s.items():
        if cond.startswith("_"):
            continue
        print(f"\n== {cond}: {d['n_windows']} windows (fosm relE {d['fosm']['rel_err_E']:.2e}, relSD {d['fosm']['rel_err_sd']:.3f})")
        print("  order     relE      relSD   sd/mc   cvarErr  maxrelE | vs desc: dE med / max, dsd med / max, E better")
        for o in ORDERS:
            x = d["orders"][o]
            v = x.get("vs_desc")
            tail = (f"{v['rel_dE_median']:.1e} / {v['rel_dE_max']:.1e}, {v['rel_dsd_median']:.1e} / {v['rel_dsd_max']:.1e}, {v['E_better_than_desc']}/{d['n_windows']}" if v else "-")
            print(f"  {o:9s} {x['rel_err_E']:.2e} {x['rel_err_sd']:.3f}  {x['sd_ratio_to_mc']:.4f}  {x['cvar_abs_err']:.4f}  {x['max_rel_err_E']:.3f} | {tail}")


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
    ap.add_argument("--traverses", nargs="*", default=["t1", "t2"])
    a = ap.parse_args()
    root, out = Path(a.windows_root), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(p, cond) for cond in ("foresight", "hindsight") for t in a.traverses
            for p in sorted((root / f"{cond}_v2" / t).glob("window_*.npz"))]
    print(f"{len(jobs)} design windows, {a.workers} workers", flush=True)
    with Pool(a.workers) as pool:
        rows = [r for r in pool.map(_job, jobs, chunksize=1) if r is not None]
    (out / "fold_order_windows.json").write_text(json.dumps(rows))
    s = summarize(rows)
    s["_meta"] = {"written": time.strftime("%F %T"), "orders": ORDERS, "n_ref": N_REF, "windows_root": str(root)}
    (out / "fold_order_summary.json").write_text(json.dumps(s, indent=1))
    print_summary(s)
    print("wrote", out)


if __name__ == "__main__":
    main()
