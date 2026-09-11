"""Gated fold sweep on the DESIGN traverses (post hoc, 2026-09-11).

Question (clark_paper review discussion): first-order propagation is the
alpha -> inf limit of the Clark fold. Gate each pairwise fold on its contest
depth: if the running leader's lead over the next candidate, in units of
the contest width, clears alpha_star, keep the leader's moments (the FOSM
step); otherwise fold (Clark). alpha_star = 0 is FOSM everywhere, alpha_star
= inf is the Clark fold everywhere; both endpoints must reproduce the frozen
arms bit-for-bit (asserted per window).

Two gate variables per fold step:
  exact     alpha = (m_run - m_j) / a,  a^2 = v_run + v_j - 2 c_j   (needs the covariance)
  marginal  alpha_lb = (m_run - m_j) / (sqrt(v_run) + sqrt(v_j)) <= alpha  (marginals only)

Reported per window and alpha_star: E, sd, Gaussian CVaR_0.9 and their
errors against a 20k-draw referee drawn from the window's belief (the v2
referee construction, v2_arms.sample_costs), the fraction of real folds
gated (unweighted and weighted by the attitude operator's column norm),
and the summed lift the gate declined, sum over gated folds of a*psi(alpha),
psi(x) = phi(x) - x Phi(-x), weighted the same way.

DESIGN traverses only (t1, t2), both conditions. Nothing here touches the
held-out record. Seeds: crc32 of the window name (the runner's hash() is
salted per process; this sweep is meant to be re-runnable).

    PYTHONPATH=studies python -m baseprod.v2_gate_sweep <windows_root> <out_dir> [--workers N]

<windows_root>/{foresight_v2,hindsight_v2}/{t1,t2}/window_*.npz
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

from spires.risk_calibration import SIGMA_FLOOR, Window, build_nodes, resample_track
from spires.vehicle import PAD_CAP
from . import constants as K
from .score import cell_covariance, cell_variance_split
from .v2_arms import arm_clark, arm_fosm, sample_costs
from .v2_correction import law, window_alpha_from_arrays
from .v2_moments import _clark_pair, gaussian_cvar, quadratic_moments
from .v2_score import _attitude_G

ALPHA_STARS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, float("inf")]
LAW = {"a": 0.110, "b": 1.0209}          # F1.json, frozen 2026-08-31
N_REF = 20_000
Q = 0.9


def psi(x):
    return norm.pdf(x) - x * norm.cdf(-x)


def abstract_window(p: Path):
    """case_v2's loading, QC and abstract form, verbatim, without scoring."""
    d = np.load(p)
    win = Window(path=p, mu=d["mu"].astype(np.float64), sigma=d["sigma"].astype(np.float64),
                 raw_sd=d["raw_sd"].astype(np.float64), meas_sd=d["meas_sd"].astype(np.float64),
                 count=d["count"], x0=float(d["xmin"]), y0=float(d["ymin"]),
                 cell=float(d["cell"]), sx=float(d["sx"]), sy=float(d["sy"]),
                 gt_poses=d["gt_poses"])
    track = resample_track(win.gt_poses)
    if len(track) == 0:
        return None
    cell_flat, caps, c_eff = build_nodes(win, track)
    t_steps = len(track)
    mu_f, sg_f = win.mu.ravel(), win.sigma.ravel()
    cnt_f = win.count.ravel()
    usable = (np.isfinite(mu_f) & np.isfinite(sg_f) & (sg_f > SIGMA_FLOOR) & (cnt_f > 0))
    real_all = caps > PAD_CAP + 1.0
    qc_cells = np.unique(cell_flat[real_all])
    obs_frac = float((cnt_f[qc_cells] > 0).mean())
    seen = qc_cells[cnt_f[qc_cells] > 0]
    sig_med_all = float(np.median(sg_f[seen])) if len(seen) else float("nan")
    flags = []
    if obs_frac < K.FLAG_MIN_OBS_FRAC:
        flags.append("coverage")
    if not (sig_med_all <= K.FLAG_MAX_SIGMA_M):
        flags.append("sigma")
    node_ok = usable[cell_flat].all(axis=1).reshape(3 * t_steps, 4).all(axis=1)
    step_ok = node_ok.reshape(3, t_steps).all(axis=0)
    n_keep = int(step_ok.sum())
    base = {"name": f"{p.parent.name}/{p.stem}", "traverse": p.parent.name,
            "n_steps": n_keep, "n_steps_total": t_steps,
            "retained": n_keep / t_steps if t_steps else 0.0,
            "qc_observed_frac": round(obs_frac, 4),
            "qc_sigma_med_m": round(sig_med_all, 5) if np.isfinite(sig_med_all) else None,
            "qc_flags": flags, "flagged": bool(flags)}
    if n_keep == 0 or n_keep / t_steps < K.MIN_RETAINED:
        base["scoreable"] = False
        return base, None
    base["scoreable"] = True
    sel = np.repeat(np.tile(step_ok, 3), 4)
    cell_flat, caps, c_eff = cell_flat[sel], caps[sel], c_eff[sel]
    u_cells, inv = np.unique(cell_flat.ravel(), return_inverse=True)
    u_idx = inv.reshape(cell_flat.shape)
    vi, vg, vl = cell_variance_split(win)
    C = cell_covariance(u_cells, win, vi, vg, vl)
    means = mu_f[u_cells][u_idx] + caps
    n_nodes, ncand = means.shape
    mu_all = means.ravel()
    Cflat = C[u_idx.ravel()][:, u_idx.ravel()]
    node_slices = [np.arange(ncand * i, ncand * (i + 1)) for i in range(n_nodes)]
    G = _attitude_G(c_eff, n_keep)
    cov_self = C[u_idx[:, :, None], u_idx[:, None, :]]
    alpha_w = window_alpha_from_arrays(means, cov_self, caps, c_eff, G)
    real = (caps > PAD_CAP + 1.0).ravel()
    return base, dict(mu_all=mu_all, C=Cflat, slices=node_slices, G=G,
                      real=real, alpha_w=alpha_w)


def fold_gated(mu_all, C_all, node_slices, real, wcol, alpha_star, gate="exact"):
    """v2_moments.fold_supports with a per-fold gate. Returns (m, S, stats).

    stats: n_real folds, n_gated, weighted fractions, declined lift.
    Pad candidates (real False) never count as folds: Clark returns the
    running moments for them regardless (alpha = +inf).
    """
    n = len(node_slices)
    N = len(mu_all)
    M = np.zeros((N + n, N + n))
    M[:N, :N] = C_all
    mu_aug = np.concatenate([mu_all, np.zeros(n)])
    n_fold = n_gate = 0
    w_fold = w_gate = 0.0
    declined = 0.0
    alphas = []
    for i, idx in enumerate(node_slices):
        order = idx[np.argsort(-mu_all[idx])]
        j0 = order[0]
        shift = mu_aug[j0]
        m_run, v_run = 0.0, M[j0, j0]
        c_run = M[j0, :N + i].copy()
        for j in order[1:]:
            if not real[j]:
                continue                       # pad: Clark keeps the running moments
            m_j, v_j = mu_aug[j] - shift, M[j, j]
            c_j = c_run[j] if j < N + i else 0.0
            a2 = v_run + v_j - 2.0 * c_j
            a = np.sqrt(max(a2, 0.0))
            lead = m_run - m_j
            al_exact = lead / a if a > 1e-8 else np.inf
            al_marg = lead / (np.sqrt(v_run) + np.sqrt(v_j))
            al = al_exact if gate == "exact" else al_marg
            n_fold += 1
            w_fold += wcol[i]
            alphas.append(al_exact)
            if al >= alpha_star:
                n_gate += 1
                w_gate += wcol[i]
                if np.isfinite(al_exact):
                    declined += wcol[i] * a * psi(al_exact)
                continue
            m_run, v_run, c_run = _clark_pair(m_run, v_run, m_j, v_j, c_j, c_run, M[j, :N + i])
        k = N + i
        M[k, :N + i] = c_run
        M[:N + i, k] = c_run
        M[k, k] = v_run
        mu_aug[k] = m_run + shift
    stats = {"n_fold": n_fold, "n_gated": n_gate,
             "frac_gated": n_gate / n_fold if n_fold else float("nan"),
             "frac_gated_w": w_gate / w_fold if w_fold else float("nan"),
             "declined_lift_w": float(declined)}
    return mu_aug[N:], M[N:, N:], stats, np.asarray(alphas)


def one_window(p: Path) -> dict | None:
    r = abstract_window(p)
    if r is None:
        return None
    base, w = r
    if w is None:
        return base
    mu_all, C, slices, G, real, alpha_w = (w["mu_all"], w["C"], w["slices"], w["G"],
                                           w["real"], w["alpha_w"])
    seed = zlib.crc32(base["name"].encode()) % 2 ** 31
    ref = sample_costs(mu_all, C, slices, G, N_REF, seed)
    E_mc, sd_mc = float(ref.mean()), float(ref.std(ddof=1))
    tail = np.sort(ref)[int(np.ceil(Q * len(ref))):]
    cvar_mc = float(tail.mean())
    base["mc"] = {"n_draws": N_REF, "E_mc": E_mc, "sd_mc": sd_mc, "cvar_mc": cvar_mc,
                  "cvar_q": Q, "seed": seed}
    base["alpha_weighted_median"] = alpha_w
    A = G.T @ G
    wcol = np.linalg.norm(G, axis=0)

    def err(E, sd):
        cv = gaussian_cvar(E, sd, Q)
        return {"E": E, "sd": sd, "rel_err_E": abs(E / E_mc - 1.0),
                "rel_err_E_signed": E / E_mc - 1.0,
                "rel_err_sd": abs(sd / sd_mc - 1.0), "sd_ratio_to_mc": sd / sd_mc,
                "cvar_arm": cv, "cvar_abs_err": abs(cv - cvar_mc)}

    # frozen arms, for the endpoint assertions and the reference columns
    E_c, sd_c = arm_clark(mu_all, C, slices, G)
    E_f, sd_f = arm_fosm(mu_all, C, slices, G)
    corr = 1.0 / float(law(alpha_w, LAW["a"], LAW["b"]))
    base["arms"] = {"clark": err(E_c, sd_c), "clark-corr": err(E_c, sd_c * corr),
                    "fosm": err(E_f, sd_f)}

    sweep = {}
    for gate in ("exact", "marginal"):
        sweep[gate] = {}
        for a_star in ALPHA_STARS:
            m, S, st, alphas = fold_gated(mu_all, C, slices, real, wcol, a_star, gate)
            E, V = quadratic_moments(m, S, A)
            row = err(E, float(np.sqrt(V)))
            row.update(st)
            sweep[gate][str(a_star)] = row
            if a_star == 0.0:
                assert abs(E - E_f) <= 1e-9 * max(1.0, abs(E_f)) and abs(row["sd"] - sd_f) <= 1e-9 * max(1.0, sd_f), \
                    ("alpha*=0 is not fosm", base["name"], gate, E, E_f, row["sd"], sd_f)
            if a_star == float("inf"):
                assert abs(E - E_c) <= 1e-9 * max(1.0, abs(E_c)) and abs(row["sd"] - sd_c) <= 1e-9 * max(1.0, sd_c), \
                    ("alpha*=inf is not clark", base["name"], gate, E, E_c, row["sd"], sd_c)
        if gate == "exact":
            fin = alphas[np.isfinite(alphas)]
            base["fold_alpha"] = {"n": int(len(alphas)), "n_finite": int(len(fin)),
                                  "quantiles": {str(q): float(np.quantile(fin, q))
                                                for q in (0.05, 0.25, 0.5, 0.75, 0.95)} if len(fin) else None,
                                  "hist_edges": [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 6, 1e9],
                                  "hist": np.histogram(fin, [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 6, 1e9])[0].tolist() if len(fin) else None}
    base["sweep"] = sweep
    return base


def _med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")


def summarize(rows):
    out = {}
    ok = [r for r in rows if r.get("scoreable") and not r["flagged"] and "sweep" in r]
    for cond in sorted({r["condition"] for r in ok}):
        rc = [r for r in ok if r["condition"] == cond]
        s = {"n_windows": len(rc), "arms": {}, "sweep": {}}
        for arm in ("clark", "clark-corr", "fosm"):
            s["arms"][arm] = {k: _med(r["arms"][arm][k] for r in rc)
                              for k in ("rel_err_E", "rel_err_E_signed", "rel_err_sd",
                                        "sd_ratio_to_mc", "cvar_abs_err")}
        for gate in ("exact", "marginal"):
            s["sweep"][gate] = {}
            prev = None
            for a_star in ALPHA_STARS:
                key = str(a_star)
                g = [r["sweep"][gate][key] for r in rc]
                d = {k: _med(x[k] for x in g)
                     for k in ("rel_err_E", "rel_err_E_signed", "rel_err_sd", "sd_ratio_to_mc",
                               "cvar_abs_err", "frac_gated", "frac_gated_w", "declined_lift_w")}
                d["frac_gated_pooled"] = (sum(x["n_gated"] for x in g)
                                          / max(sum(x["n_fold"] for x in g), 1))
                if prev is not None:
                    # per-window: does the error fall as alpha* rises (more folds)?
                    d["win_E_better_than_prev"] = int(sum(
                        x["rel_err_E"] <= y["rel_err_E"] for x, y in zip(g, prev)))
                    d["win_cvar_better_than_prev"] = int(sum(
                        x["cvar_abs_err"] <= y["cvar_abs_err"] for x, y in zip(g, prev)))
                s["sweep"][gate][key] = d
                prev = g
        # pooled per-fold alpha histogram
        hist = np.zeros(11, int)
        for r in rc:
            if r.get("fold_alpha", {}).get("hist"):
                hist += np.asarray(r["fold_alpha"]["hist"])
        s["fold_alpha_hist"] = {"edges": rc[0]["fold_alpha"]["hist_edges"], "counts": hist.tolist(),
                                "frac_ge_2": float(hist[6:].sum() / max(hist.sum(), 1)),
                                "frac_ge_3": float(hist[8:].sum() / max(hist.sum(), 1))}
        out[cond] = s
    return out


def print_summary(summ):
    for cond, s in summ.items():
        if cond.startswith("_"):
            continue
        print(f"\n== {cond}: {s['n_windows']} unflagged design windows ==")
        a = s["arms"]
        print("  frozen arms  relE      signed    relSD    sd/mc   cvarErr")
        for arm in ("fosm", "clark", "clark-corr"):
            x = a[arm]
            print(f"  {arm:11s} {x['rel_err_E']:.2e} {x['rel_err_E_signed']:+.2e} {x['rel_err_sd']:.3f} {x['sd_ratio_to_mc']:.3f} {x['cvar_abs_err']:.4f}")
        h = s["fold_alpha_hist"]
        print(f"  per-fold alpha (exact, pooled, {sum(h['counts'])} real folds): "
              f"P[alpha>=2] = {h['frac_ge_2']:.3f}, P[alpha>=3] = {h['frac_ge_3']:.3f}")
        print("  hist edges", h["edges"][:-1], "\n  counts    ", h["counts"])
        for gate in ("exact", "marginal"):
            print(f"  gate={gate}:  a*     relE      signed    relSD    sd/mc   cvarErr  gated(pooled) gated_w  declined_lift  E<=prev cvar<=prev")
            for a_star in ALPHA_STARS:
                d = s["sweep"][gate][str(a_star)]
                print(f"    {a_star:>6}  {d['rel_err_E']:.2e} {d['rel_err_E_signed']:+.2e} {d['rel_err_sd']:.3f} {d['sd_ratio_to_mc']:.3f} "
                      f"{d['cvar_abs_err']:.4f}   {d['frac_gated_pooled']:.3f}      {d['frac_gated_w']:.3f}   "
                      f"{d['declined_lift_w']:.2e}      {d.get('win_E_better_than_prev', '-'):>3}     {d.get('win_cvar_better_than_prev', '-'):>3}")


def _job(args):
    p, cond = args
    t0 = time.time()
    r = one_window(p)
    if r is not None:
        r["condition"] = cond
        r["wall_s"] = round(time.time() - t0, 1)
        print(f"[{time.strftime('%T')}] {cond} {r['name']} "
              + ("flagged" if r["flagged"] else
                 (f"relE fosm {r['arms']['fosm']['rel_err_E']:.1e} clark {r['arms']['clark']['rel_err_E']:.1e} "
                  f"P[a>=2] {r['fold_alpha']['hist'] and sum(r['fold_alpha']['hist'][6:]) / max(sum(r['fold_alpha']['hist']), 1):.2f}"
                  if r.get("scoreable") else "unscoreable"))
              + f" ({r['wall_s']}s)", flush=True)
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
    jobs = []
    for cond in ("foresight", "hindsight"):
        for t in a.traverses:
            jobs += [(p, cond) for p in sorted((root / f"{cond}_v2" / t).glob("window_*.npz"))]
    print(f"{len(jobs)} design windows, {a.workers} workers", flush=True)
    with Pool(a.workers) as pool:
        rows = [r for r in pool.map(_job, jobs, chunksize=1) if r is not None]
    (out / "design_sweep_windows.json").write_text(json.dumps(rows, indent=1))
    summ = summarize(rows)
    summ["_meta"] = {"alpha_stars": [str(x) for x in ALPHA_STARS], "n_ref": N_REF, "law": LAW,
                     "traverses": a.traverses, "written": time.strftime("%F %T"),
                     "windows_root": str(root)}
    (out / "design_sweep_summary.json").write_text(json.dumps(summ, indent=1))
    print_summary(summ)
    print("wrote", out)


if __name__ == "__main__":
    main()
