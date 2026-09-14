"""Cross-site check of the sd correction on the Oxford Spires windows (post hoc, 2026-09-13).

The deficit law d(alpha_w) = 1 - a exp(-b alpha_w) was fitted on two BASEPROD
design traverses (F1.json: a = 0.110, b = 1.0209) and held out within BASEPROD.
This applies it, frozen, to a different dataset: the Oxford Spires windows of the
E1 campaign (Hesai lidar, handheld track, the same virtual vehicle and the same
v2 attitude cost), on the design site (keble-college, two sequences) and the two
virgin sites (blenheim-palace, christ-church). The referee is the v2 referee
(20k draws from each window's belief, seed crc32 of the window name). Reported
per site: the fold's sd ratio to the referee raw, under the BASEPROD law
(zero-shot cross-site), and under a law refitted on keble (within-dataset
transfer, the same rule as F1); the mean and CVaR errors of clark, clark-corr,
fosm and the hybrid; the windows' contest depth alpha_w.

The Spires window npz files carry the same fields the v2 driver reads (the
Window dataclass is the Spires one). `keble-college-02-unclamped` duplicates
-02-default's tracks and is skipped.

2026-09-14 rerun (author's request): the E1 overhang flag is joined as a
quality rule. The Spires walks pass under arches, cloisters and porches, and
the max-height-per-cell convention of both the belief and the TLS truth
aliases the roof onto the ground cell (RISK_CAL_DATA_AUDIT.md R1; the flag,
belief-to-ground gap > 1 m along the track, was computed in E1 before any
cross-site scoring). `--overhang-json` takes the E1 artifacts; a window
flagged there, or absent from them, is excluded from the CLEAN summary
(crosssite_summary_clean.json) and reported in crosssite_summary_overhang.json.
The BASEPROD sigma flag still marks every Spires window and is reported as
before. Same referee seeds, so the per-window numbers reproduce the
2026-09-13 run.

    PYTHONPATH=studies python -m baseprod.v2_crosssite <spires_out_root> <out_dir> [--workers N]
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
from .v2_correction import fit_law, law
from .v2_gate_sweep import abstract_window
from .v2_moments import gaussian_cvar

LAW_BASEPROD = {"a": 0.110, "b": 1.0209}
N_REF = 20_000
Q = 0.9
SITES = {"keble-college-02-default": "keble", "keble-college-03-default": "keble",
         "virgin-2024-03-14-blenheim-palace-01": "blenheim", "virgin-2024-03-14-blenheim-palace-02": "blenheim",
         "virgin-2024-03-18-christ-church-02": "christ-church", "virgin-2024-03-18-christ-church-03": "christ-church"}


def one_window(p: Path):
    r = abstract_window(p)
    if r is None:
        return None
    base, w = r
    base["site"] = SITES[p.parent.name]
    base["sequence"] = p.parent.name
    if w is None:
        return base
    mu_all, C, slices, G, alpha_w = w["mu_all"], w["C"], w["slices"], w["G"], w["alpha_w"]
    seed = zlib.crc32(base["name"].encode()) % 2 ** 31
    ref = sample_costs(mu_all, C, slices, G, N_REF, seed)
    E_mc, sd_mc = float(ref.mean()), float(ref.std(ddof=1))
    cvar_mc = float(np.sort(ref)[int(np.ceil(Q * len(ref))):].mean())
    base["mc"] = {"n_draws": N_REF, "seed": seed, "E_mc": E_mc, "sd_mc": sd_mc, "cvar_mc": cvar_mc}
    base["alpha_weighted_median"] = alpha_w
    base["n_nodes"] = len(slices)
    base["sigma_med_m"] = base.get("qc_sigma_med_m")
    E_c, sd_c = arm_clark(mu_all, C, slices, G)
    E_f, sd_f = arm_fosm(mu_all, C, slices, G)
    corr_bp = 1.0 / float(law(alpha_w, LAW_BASEPROD["a"], LAW_BASEPROD["b"]))
    arms = {"clark": (E_c, sd_c), "clark-corr-baseprod": (E_c, sd_c * corr_bp),
            "fosm": (E_f, sd_f), "hybrid": (E_c, sd_f)}
    base["arms"] = {}
    for a, (E, sd) in arms.items():
        base["arms"][a] = {"E": E, "sd": sd, "rel_err_E": abs(E / E_mc - 1), "rel_err_sd": abs(sd / sd_mc - 1),
                           "sd_ratio_to_mc": sd / sd_mc, "cvar_abs_err": abs(gaussian_cvar(E, sd, Q) - cvar_mc)}
    return base


def summarize(rows, include_flagged=False, overhang=None):
    """overhang: None = ignore the E1 flag (the 2026-09-13 reading); False = CLEAN windows only
    (E1 record present and not overhang-flagged); True = overhang-flagged windows only."""
    ok = [r for r in rows if r.get("scoreable") and "arms" in r and (include_flagged or not r["flagged"])
          and (overhang is None or r.get("overhang") is overhang)]
    flagged = [r for r in rows if r.get("flagged")]
    out = {"n_rows": len(rows), "n_scored": len(ok), "n_flagged": len(flagged), "include_flagged": include_flagged,
           "overhang_subset": overhang,
           "n_overhang_flagged": int(sum(r.get("overhang") is True for r in rows)),
           "n_no_e1_record": int(sum(r.get("overhang") is None for r in rows if r.get("scoreable") and "arms" in r)),
           "flags": {f: int(sum(f in r["qc_flags"] for r in flagged)) for f in ("coverage", "sigma")}}
    keb = [r for r in ok if r["site"] == "keble"]
    refit = fit_law([r["alpha_weighted_median"] for r in keb], [r["arms"]["clark"]["sd_ratio_to_mc"] for r in keb]) if len(keb) >= 5 else None
    out["law_baseprod"] = LAW_BASEPROD
    out["law_keble_refit"] = refit
    for site in ("keble", "blenheim", "christ-church", "virgin"):
        rs = [r for r in ok if (r["site"] == site or (site == "virgin" and r["site"] != "keble"))]
        if not rs:
            continue
        d = {"n_windows": len(rs), "alpha_w_median": float(np.median([r["alpha_weighted_median"] for r in rs])),
             "alpha_w_iqr": [float(np.quantile([r["alpha_weighted_median"] for r in rs], q)) for q in (0.25, 0.75)],
             "sigma_med_m_median": float(np.median([r["qc_sigma_med_m"] for r in rs if r.get("qc_sigma_med_m") is not None])),
             "overhang_gap_m_median": float(np.median([r["overhang_gap_m"] for r in rs if r.get("overhang_gap_m") is not None])) if any(r.get("overhang_gap_m") is not None for r in rs) else None,
             "arms": {}}
        for a in ("clark", "clark-corr-baseprod", "fosm", "hybrid"):
            d["arms"][a] = {k: float(np.median([r["arms"][a][k] for r in rs])) for k in ("rel_err_E", "rel_err_sd", "sd_ratio_to_mc", "cvar_abs_err")}
        if refit:
            ratio = np.array([r["arms"]["clark"]["sd_ratio_to_mc"] / law(r["alpha_weighted_median"], refit["a"], refit["b"]) for r in rs])
            d["arms"]["clark-corr-keble"] = {"sd_ratio_to_mc": float(np.median(ratio)), "rel_err_sd": float(np.median(np.abs(ratio - 1)))}
        # paired: corrected (baseprod law) sd error below raw fold's, and below fosm's
        d["paired"] = {"corr_bp_sd_err_below_raw": int(sum(r["arms"]["clark-corr-baseprod"]["rel_err_sd"] < r["arms"]["clark"]["rel_err_sd"] for r in rs)),
                       "corr_bp_sd_err_below_fosm": int(sum(r["arms"]["clark-corr-baseprod"]["rel_err_sd"] < r["arms"]["fosm"]["rel_err_sd"] for r in rs)),
                       "clark_E_err_below_fosm": int(sum(r["arms"]["clark"]["rel_err_E"] < r["arms"]["fosm"]["rel_err_E"] for r in rs)),
                       "corr_bp_cvar_err_below_fosm": int(sum(r["arms"]["clark-corr-baseprod"]["cvar_abs_err"] < r["arms"]["fosm"]["cvar_abs_err"] for r in rs))}
        out[site] = d
    return out


def print_summary(s):
    print(f"\nrows {s['n_rows']}, scored {s['n_scored']}, flagged {s['n_flagged']} {s['flags']}, overhang subset {s.get('overhang_subset')}, "
          f"overhang-flagged {s.get('n_overhang_flagged')}, no E1 record {s.get('n_no_e1_record')}")
    print("BASEPROD law", s["law_baseprod"], "| keble refit", s["law_keble_refit"])
    for site in ("keble", "blenheim", "christ-church", "virgin"):
        if site not in s:
            continue
        d = s[site]
        print(f"\n== {site}: {d['n_windows']} windows, alpha_w median {d['alpha_w_median']:.2f} (IQR {d['alpha_w_iqr'][0]:.2f}-{d['alpha_w_iqr'][1]:.2f}), sigma med {d['sigma_med_m_median']:.3f} m")
        for a, x in d["arms"].items():
            print(f"  {a:20s} " + " ".join(f"{k} {v:.4g}" for k, v in x.items()))
        print("  paired:", d["paired"])


def _job(p):
    r = one_window(p)
    if r is not None:
        print(f"[{time.strftime('%T')}] {r['sequence']}/{Path(r['name']).name} " + ("flagged " + ",".join(r["qc_flags"]) if r["flagged"] else "ok" if r.get("scoreable") else "unscoreable"), flush=True)
    return r


def main():
    global N_REF
    ap = argparse.ArgumentParser()
    ap.add_argument("spires_out_root")
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--n-ref", type=int, default=N_REF, help="referee draws per window (20k is the v2 default; "
                    "the Spires windows have ~12k candidates, so fewer draws keep the run tractable)")
    ap.add_argument("--overhang-json", nargs="*", default=[], help="E1 artifacts (risk_calibration*/e*_*.json) whose "
                    "per-window overhang_flag is joined by window name; windows absent from them count as no record")
    a = ap.parse_args()
    N_REF = a.n_ref
    root, out = Path(a.spires_out_root), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [p for seq in SITES for p in sorted((root / seq).glob("window_*.npz"))]
    # per-window persistence (2026-09-14): each finished row is appended to crosssite_rows.jsonl
    # as it completes, and a rerun skips the windows already there. A pool.map whose worker
    # dies (the first 2026-09-14 attempt lost a worker on the largest christ-church-02 windows
    # and hung with 105 of 111 done, nothing on disk) is what this guards against.
    rows_path = out / "crosssite_rows.jsonl"
    done = {}
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["name"]] = r
    jobs = [p for p in jobs if f"{p.parent.name}/{p.stem}" not in done]
    print(f"{len(jobs)} windows to score ({len(done)} already in {rows_path.name}), {a.workers} workers", flush=True)
    rows = list(done.values())
    with Pool(a.workers, maxtasksperchild=4) as pool, rows_path.open("a") as fh:
        for r in pool.imap_unordered(_job, jobs, chunksize=1):
            if r is None:
                continue
            fh.write(json.dumps(r) + "\n"); fh.flush()
            rows.append(r)
    order = {f"{p.parent.name}/{p.stem}": i for i, p in enumerate(p for seq in SITES for p in sorted((root / seq).glob("window_*.npz")))}
    rows.sort(key=lambda r: order.get(r["name"], 1 << 30))
    # join the E1 overhang flag by window name: True / False / None (no E1 record)
    e1 = {}
    for f in a.overhang_json:
        for r in json.loads(Path(f).read_text())["windows"]:
            e1.setdefault(r["window"], (bool(r.get("overhang_flag")), r.get("overhang_gap_m")))
    for r in rows:
        flag, gap = e1.get(r["name"], (None, None))
        r["overhang"], r["overhang_gap_m"] = flag, gap
        if flag:
            r["qc_flags"] = list(r.get("qc_flags", [])) + ["overhang"]
    (out / "crosssite_windows.json").write_text(json.dumps(rows))
    meta = {"written": time.strftime("%F %T"), "spires_out_root": str(root), "n_ref": N_REF,
            "overhang_json": [str(f) for f in a.overhang_json],
            "note": "post hoc; Oxford Spires E1 windows scored with the v2 attitude cost; BASEPROD law frozen (F1.json); "
                    "QC flags are BASEPROD's (sigma <= 0.10 m, observed fraction >= 0.70) plus the E1 overhang flag "
                    "(belief-to-ground gap > 1 m along the track, RISK_CAL_DATA_AUDIT.md R1) when --overhang-json is given"}
    s = summarize(rows); s["_meta"] = meta
    (out / "crosssite_summary.json").write_text(json.dumps(s, indent=1))
    print("\n##### UNFLAGGED WINDOWS (BASEPROD QC) #####"); print_summary(s)
    s2 = summarize(rows, include_flagged=True); s2["_meta"] = meta
    (out / "crosssite_summary_all.json").write_text(json.dumps(s2, indent=1))
    print("\n##### ALL SCOREABLE WINDOWS (flagged included) #####"); print_summary(s2)
    if a.overhang_json:
        s3 = summarize(rows, include_flagged=True, overhang=False); s3["_meta"] = meta
        (out / "crosssite_summary_clean.json").write_text(json.dumps(s3, indent=1))
        print("\n##### CLEAN WINDOWS (E1 record present, not overhang-flagged; BASEPROD sigma flag ignored) #####"); print_summary(s3)
        s4 = summarize(rows, include_flagged=True, overhang=True); s4["_meta"] = meta
        (out / "crosssite_summary_overhang.json").write_text(json.dumps(s4, indent=1))
        print("\n##### OVERHANG-FLAGGED WINDOWS #####"); print_summary(s4)
    print("wrote", out)


if __name__ == "__main__":
    main()
