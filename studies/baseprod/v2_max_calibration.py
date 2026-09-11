"""Maximum-level calibration of the wheel support on BASEPROD against the DSM (2026-09-11).

Post hoc, not pre-registered. For every node (wheel-step corner) of every v2
window: the TRUE support, max over the element's candidates of (truth_max + cap),
the DSM registered per traverse by E2's register.py and carried in the window
npz as `truth_max`; the fold's E[support] and sd (spires.risk_calibration.clark_fold,
the v2 fold's own recursion); the max of the means (fosm's and the mean map's
support); and the node's contest depth alpha (top two candidates). The question
the simulated clearance campaign raised (clark_paper sim_campaign/rubble/clearance/
investigate/RESULT_investigation.md): does the truth's support lift over the max
of the means as the belief's Jensen lift predicts, or does it not?

Also the cost-level residual E2 reported, per window, for the record.

Design traverses (t1, t2) and held-out traverses are both read and LABELLED; the
held-out truth was already consumed by E2 (pre-registered), and this reading of
it is post hoc.

    PYTHONPATH=studies_v2 BASEPROD_ROOT=/local/kuceral4/baseprod \
        python -m baseprod.v2_max_calibration <windows_root> <out_dir> [--workers N]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from spires.risk_calibration import SIGMA_FLOOR, Window, build_nodes, clark_fold, resample_track
from spires.vehicle import PAD_CAP
from . import constants as K
from .paths import DESIGN_TRAVERSES
from .score import cell_covariance, cell_variance_split

ALPHA_BINS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 1e9]


def one_window(p: Path) -> dict | None:
    d = np.load(p)
    if "truth_max" not in d.files:
        return None
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
    mu_f, sg_f, tl_f, cnt_f = win.mu.ravel(), win.sigma.ravel(), tmax.ravel(), win.count.ravel()
    usable = (np.isfinite(mu_f) & np.isfinite(sg_f) & (sg_f > SIGMA_FLOOR)
              & np.isfinite(tl_f) & (cnt_f > 0))
    real_all = caps > PAD_CAP + 1.0
    qc_cells = np.unique(cell_flat[real_all])
    obs_frac = float((cnt_f[qc_cells] > 0).mean())
    seen = qc_cells[cnt_f[qc_cells] > 0]
    sig_med = float(np.median(sg_f[seen])) if len(seen) else float("nan")
    void_frac = float((~np.isfinite(tl_f[qc_cells])).mean())
    flags = []
    if obs_frac < K.FLAG_MIN_OBS_FRAC:
        flags.append("coverage")
    if not (sig_med <= K.FLAG_MAX_SIGMA_M):
        flags.append("sigma")
    if void_frac > K.FLAG_MAX_INVALID_FRAC:
        flags.append("truth_invalid")
    node_ok = usable[cell_flat].all(axis=1).reshape(3 * t_steps, 4).all(axis=1)
    step_ok = node_ok.reshape(3, t_steps).all(axis=0)
    n_keep = int(step_ok.sum())
    base = {"name": f"{p.parent.name}/{p.stem}", "traverse": p.parent.name,
            "design": p.parent.name in DESIGN_TRAVERSES or p.parent.name in ("t1", "t2"),
            "n_steps": n_keep, "n_steps_total": t_steps,
            "retained": n_keep / t_steps if t_steps else 0.0,
            "qc_observed_frac": round(obs_frac, 4), "qc_sigma_med_m": sig_med,
            "qc_truth_invalid_frac": round(void_frac, 5),
            "qc_flags": flags, "flagged": bool(flags)}
    if n_keep == 0 or n_keep / t_steps < K.MIN_RETAINED:
        base["scoreable"] = False
        return base
    base["scoreable"] = True
    sel = np.repeat(np.tile(step_ok, 3), 4)
    cell_flat, caps, c_eff = cell_flat[sel], caps[sel], c_eff[sel]
    u_cells, inv = np.unique(cell_flat.ravel(), return_inverse=True)
    u_idx = inv.reshape(cell_flat.shape)
    vi, vg, vl = cell_variance_split(win)
    C = cell_covariance(u_cells, win, vi, vg, vl)
    means = mu_f[u_cells][u_idx] + caps                  # (n_nodes, K), pads at PAD_CAP
    cov_self = C[u_idx[:, :, None], u_idx[:, None, :]]
    m_fold, v_fold, _lam = clark_fold(means, cov_self)
    real = caps > PAD_CAP + 1.0
    m_masked = np.where(real, means, -np.inf)
    max_mean = m_masked.max(axis=1)
    env_t = np.where(real, tl_f[u_cells][u_idx] + caps, -np.inf).max(axis=1)
    # contest depth of the top two real candidates, as v2_correction does
    order = np.argsort(-m_masked, axis=1)
    i1, i2 = order[:, 0], order[:, 1]
    r = np.arange(means.shape[0])
    a2 = cov_self[r, i1, i1] + cov_self[r, i2, i2] - 2.0 * cov_self[r, i1, i2]
    alpha = (m_masked[r, i1] - m_masked[r, i2]) / np.sqrt(np.maximum(a2, 1e-18))
    alpha = np.where(real[r, i2], alpha, np.inf)
    sd_win = np.sqrt(cov_self[r, i1, i1])
    n_real = real.sum(axis=1)
    base["nodes"] = {
        "truth_support": env_t.round(5).tolist(),
        "fold_E": m_fold.round(5).tolist(),
        "fold_sd": np.sqrt(np.maximum(v_fold, 0.0)).round(5).tolist(),
        "max_mean": max_mean.round(5).tolist(),
        "winner_sd": sd_win.round(5).tolist(),
        "alpha": np.where(np.isfinite(alpha), alpha, 1e9).round(4).tolist(),
        "n_real": n_real.tolist(),
        "c_eff": c_eff.round(6).tolist(),
    }
    # cost-level residual (E2's linear cost, without its constant): truth minus arms
    base["cost"] = {"j_truth": float(c_eff @ env_t), "fold_E": float(c_eff @ m_fold),
                    "mean_map": float(c_eff @ max_mean)}
    return base


def summarize(rows: list) -> dict:
    out = {}
    ok = [r for r in rows if r.get("scoreable") and not r["flagged"]]
    for split in ("design", "heldout", "all"):
        rs = [r for r in ok if split == "all" or (r["design"] == (split == "design"))]
        for cond in sorted({r["condition"] for r in rs}):
            rc = [r for r in rs if r["condition"] == cond]
            if not rc:
                continue
            T = np.concatenate([r["nodes"]["truth_support"] for r in rc])
            E = np.concatenate([r["nodes"]["fold_E"] for r in rc])
            S = np.concatenate([r["nodes"]["fold_sd"] for r in rc])
            M = np.concatenate([r["nodes"]["max_mean"] for r in rc])
            W = np.concatenate([r["nodes"]["winner_sd"] for r in rc])
            A = np.concatenate([r["nodes"]["alpha"] for r in rc])
            TR = np.concatenate([[r["traverse"]] * len(r["nodes"]["truth_support"]) for r in rc])
            fin = np.isfinite(T) & np.isfinite(E) & (S > 0)
            T, E, S, M, W, A, TR = T[fin], E[fin], S[fin], M[fin], W[fin], A[fin], TR[fin]
            # The DSM registration (E2 register.py) carries the wheel radius and any
            # datum constant in its constant term, so the truth support sits a
            # constant above the belief per traverse. Remove the per-traverse
            # median of (truth - max of means) and report it; what is left is the
            # LIFT structure, which is the object of this reading.
            offsets = {t: float(np.median((T - M)[TR == t])) for t in np.unique(TR)}
            off = np.array([offsets[t] for t in TR])
            e_fold, e_mm = T - E - off, T - M - off
            d = {"n_windows": len(rc), "n_nodes": int(len(T)),
                 "per_traverse_offset_m": offsets,
                 "offset_note": "truth minus max of means, per-traverse median, removed from every error below",
                 "fold": {"err_mean": float(e_fold.mean()), "err_median": float(np.median(e_fold)),
                          "err_sd": float(e_fold.std()), "z_mean": float((e_fold / S).mean()),
                          "within_1sd": float(np.mean(np.abs(e_fold / S) <= 1)),
                          "within_2sd": float(np.mean(np.abs(e_fold / S) <= 2)),
                          "P_truth_above": float(np.mean(e_fold > 0))},
                 "max_of_means": {"err_mean": float(e_mm.mean()), "err_median": float(np.median(e_mm)),
                                  "err_sd": float(e_mm.std()), "z_mean_winner_sd": float((e_mm / W).mean()),
                                  "within_1sd": float(np.mean(np.abs(e_mm / W) <= 1)),
                                  "P_truth_above": float(np.mean(e_mm > 0))},
                 "predicted_lift_median": float(np.median(E - M)),
                 "truth_lift_over_max_mean_median": float(np.median(e_mm)),
                 "by_alpha": []}
            for lo, hi in zip(ALPHA_BINS[:-1], ALPHA_BINS[1:]):
                m = (A >= lo) & (A < hi)
                if m.sum() < 50:
                    continue
                d["by_alpha"].append({"alpha_lo": lo, "alpha_hi": hi, "n": int(m.sum()),
                                      "predicted_lift_median": float(np.median(E[m] - M[m])),
                                      "truth_lift_median": float(np.median(e_mm[m])),
                                      "truth_lift_mean": float(e_mm[m].mean()),
                                      "fold_err_mean": float(e_fold[m].mean()),
                                      "fold_z_mean": float((e_fold[m] / S[m]).mean()),
                                      "fold_within_1sd": float(np.mean(np.abs(e_fold[m] / S[m]) <= 1))})
            jt = np.array([r["cost"]["j_truth"] for r in rc])
            d["cost_residual"] = {"fold_mean": float(np.mean(jt - [r["cost"]["fold_E"] for r in rc])),
                                  "mean_map_mean": float(np.mean(jt - [r["cost"]["mean_map"] for r in rc]))}
            out[f"{split}/{cond}"] = d
    return out


def print_summary(summ: dict) -> None:
    for key, d in summ.items():
        f, m = d["fold"], d["max_of_means"]
        print(f"\n== {key}: {d['n_windows']} windows, {d['n_nodes']} nodes (truth support - forecast)")
        print(f"  fold E:        mean {f['err_mean']:+.4f} median {f['err_median']:+.4f} sd {f['err_sd']:.4f}; "
              f"z {f['z_mean']:+.2f}, |z|<=1 {f['within_1sd']:.2f}, P(truth above) {f['P_truth_above']:.2f}")
        print(f"  max of means:  mean {m['err_mean']:+.4f} median {m['err_median']:+.4f} sd {m['err_sd']:.4f}; "
              f"z(winner sd) {m['z_mean_winner_sd']:+.2f}, P(truth above) {m['P_truth_above']:.2f}")
        print(f"  predicted lift median {d['predicted_lift_median']:+.4f}; truth lift over max of means median "
              f"{d['truth_lift_over_max_mean_median']:+.4f}")
        for b in d["by_alpha"]:
            print(f"    alpha [{b['alpha_lo']:g},{b['alpha_hi']:g}) n={b['n']:6d}: predicted lift {b['predicted_lift_median']:+.4f}, "
                  f"truth lift median {b['truth_lift_median']:+.4f} mean {b['truth_lift_mean']:+.4f}; "
                  f"fold err {b['fold_err_mean']:+.4f} z {b['fold_z_mean']:+.2f} |z|<=1 {b['fold_within_1sd']:.2f}")


def _job(args):
    p, cond = args
    r = one_window(p)
    if r is not None:
        r["condition"] = cond
        print(f"[{time.strftime('%T')}] {cond} {r['name']} " + ("flagged" if r["flagged"] else
              ("ok" if r.get("scoreable") else "unscoreable")), flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("windows_root")
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    root, out = Path(a.windows_root), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for cond in ("foresight", "hindsight"):
        for tdir in sorted((root / f"{cond}_v2").iterdir()):
            if tdir.is_dir():
                jobs += [(p, cond) for p in sorted(tdir.glob("window_*.npz"))]
    print(f"{len(jobs)} windows, {a.workers} workers", flush=True)
    with Pool(a.workers) as pool:
        rows = [r for r in pool.map(_job, jobs, chunksize=2) if r is not None]
    (out / "max_calibration_windows.json").write_text(json.dumps(rows))
    summ = summarize(rows)
    summ["_meta"] = {"written": time.strftime("%F %T"), "windows_root": str(root),
                     "note": "post hoc; held-out truth already consumed by E2 (pre-registered)"}
    (out / "max_calibration_summary.json").write_text(json.dumps(summ, indent=1))
    print_summary({k: v for k, v in summ.items() if not k.startswith("_")})
    print("wrote", out)


if __name__ == "__main__":
    main()
