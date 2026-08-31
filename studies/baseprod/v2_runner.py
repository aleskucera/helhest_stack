"""v2 campaign runner (PREREG_attitude_cost.md sections 7-8).

Deterministic, resumable, tmux-friendly. Stage markers under out/v2/state/;
any failed freeze condition writes STOPPED_AT and exits before EVAL contact.

    BASEPROD_ROOT=/local/kuceral4/baseprod PYTHONPATH=studies_v2 \
        python -m baseprod.v2_runner [--workers 8]

Stages: config -> dbuild -> dcheck (C1/C1b/C2/C3) -> f1 -> gate -> ebuild
        -> escore -> report.
The GPU wall benchmark (studies/bench/v2_att_wall.py, study repo) is
non-gating and runs separately.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from math import comb, exp
from pathlib import Path

import numpy as np

from . import constants as K
from .paths import DATA, DESIGN_TRAVERSES, HELDOUT_DIR, OUT, ROOT

V2OUT = OUT / "v2"
STATE = V2OUT / "state"
RAW = DATA / "raw"
GATE_DZDT = 0.3            # m/s (prereg section 5)
N_REF = 20_000
LAW = {"a": 0.110, "b": 1.0209}          # pooled design fit (F1 candidate)
FAN = {"K": 32, "T": 3.0, "v_max": 1.0, "w_max": 1.2, "n_ref": 8000,
       "att_gate_rad": 0.35}
BUILD = {"R_MAX": 3.0, "ALPHA": 1.0}
CONDS = ("foresight", "hindsight")


def log(msg, stage=""):
    print(f"[{time.strftime('%F %T')}][{stage}] {msg}", flush=True)


def done(stage):
    return (STATE / f"{stage}.done").exists()


def mark(stage):
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / f"{stage}.done").touch()


def stop(stage, why):
    V2OUT.mkdir(parents=True, exist_ok=True)
    (V2OUT / "STOPPED_AT").write_text(f"{stage}: {why}\n")
    log(f"STOP: {why}", stage)
    sys.exit(2)


def correction():
    a, b = LAW["a"], LAW["b"]
    return lambda al: 1.0 / (1.0 - a * exp(-b * al)) if al == al else 1.0


def sign_p(k, n):
    if n == 0:
        return 1.0
    return min(sum(comb(n, i) for i in range(min(k, n - k) + 1)) * 2 / 2 ** n, 1.0)


def med(xs):
    xs = sorted(xs)
    n = len(xs)
    return float("nan") if n == 0 else (xs[n // 2] if n % 2 else 0.5 * (xs[n//2-1] + xs[n//2]))


def heldout_names():
    man = json.loads((ROOT / "traverse_manifest.json").read_text())
    return [m["name"] for m in man if m["name"] not in DESIGN_TRAVERSES]


# ------------------------------- stages -------------------------------------

def stage_config():
    if done("config"):
        return
    try:
        h = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]),
                            "rev-parse", "HEAD"], capture_output=True, text=True,
                           timeout=10).stdout.strip()
    except Exception:
        h = "unversioned-deployment"
    V2OUT.mkdir(parents=True, exist_ok=True)
    (V2OUT / "pipeline_commit.json").write_text(json.dumps(
        {"commit": h, "deployed_from": str(Path(__file__).resolve())}))
    log(f"pipeline at {h}", "config")
    mark("config")


def stage_dbuild():
    if done("dbuild"):
        return
    from .v2_build import build_v2
    need = []
    for name in DESIGN_TRAVERSES:
        for cond in CONDS:
            from .paths import window_dir
            if not any(window_dir(name, f"{cond}_v2").glob("window_*.npz")):
                need.append(name)
                break
    if need:
        build_v2(sorted(set(need)))
    log("design windows present", "dbuild")
    mark("dbuild")


def _score_dir(tagdir: Path, n_ref: int, corr, fan_cfg=None):
    from spires.risk_calibration import Window
    from .v2_score import case_v2
    from .v2_fan import fan as run_fan
    rows = []
    for wp in sorted(tagdir.glob("*/window_*.npz")):
        r = case_v2(wp, n_ref=n_ref, seed=abs(hash(wp.stem)) % 2 ** 31, correction=corr)
        if r is None:
            continue
        if fan_cfg and r.get("scoreable") and not r["flagged"]:
            d = np.load(wp)
            win = Window(path=wp, mu=d["mu"].astype(np.float64),
                         sigma=d["sigma"].astype(np.float64),
                         raw_sd=d["raw_sd"].astype(np.float64),
                         meas_sd=d["meas_sd"].astype(np.float64), count=d["count"],
                         x0=float(d["xmin"]), y0=float(d["ymin"]), cell=float(d["cell"]),
                         sx=float(d["sx"]), sy=float(d["sy"]), gt_poses=d["gt_poses"])
            r["fan"] = run_fan(win, K=fan_cfg["K"], T=fan_cfg["T"],
                               v_max=fan_cfg["v_max"], w_max=fan_cfg["w_max"],
                               seed=17, n_ref=fan_cfg["n_ref"])
        rows.append(r)
        log(f"{r['name']} "
            + ("flagged" if r["flagged"] else
               (f"relE {r['arms']['clark-corr']['rel_err_E']:.1e}"
                if r.get("scoreable") else "unscoreable")), "score")
    return rows


def stage_dcheck():
    if done("dcheck"):
        return
    from .v2_moments import selftest
    from .v2_correction import fit_law
    from .paths import window_dir
    selftest(seed=0, n_mc=200_000)     # C1: quadratic formulas vs MC
    alphas, ratios = [], []
    for cond in CONDS:
        for name in DESIGN_TRAVERSES:
            rows = _score_dir(window_dir(name, f"{cond}_v2").parent / window_dir(name, f"{cond}_v2").name, 50_000, None)
            for r in rows:
                if not (r.get("scoreable") and not r["flagged"]):
                    continue
                cl = r["arms"]["clark"]
                if not (np.isfinite(cl["E"]) and np.isfinite(cl["sd"])):
                    stop("dcheck", f"C3 non-finite at {r['name']}")
                alphas.append(r["alpha_weighted_median"])
                ratios.append(cl["sd_ratio_to_mc"])
    refit = fit_law(alphas, ratios)
    (V2OUT / "dcheck_refit.json").write_text(json.dumps(
        {"n": len(alphas), "refit": {k: float(v) for k, v in refit.items()
                                     if not isinstance(v, str)}}))
    if abs(refit["a"] - LAW["a"]) > 0.02 or abs(refit["b"] - LAW["b"]) > 0.2:
        stop("dcheck", f"C2 law drift: refit a={refit['a']} b={refit['b']} vs F1 {LAW}")
    log(f"C1-C3 ok; refit a={refit['a']:.4f} b={refit['b']:.4f} (n={len(alphas)})", "dcheck")
    mark("dcheck")


def stage_f1():
    if done("f1"):
        return
    f1p = V2OUT / "F1.json"
    rec = {"law": LAW, "build": BUILD, "fan": FAN, "gate_dzdt_ms": GATE_DZDT,
           "n_ref": N_REF,
           "pipeline": json.loads((V2OUT / "pipeline_commit.json").read_text()),
           "written": time.strftime("%F %T")}
    if f1p.exists():
        prev = json.loads(f1p.read_text())
        for k in ("law", "build", "fan", "gate_dzdt_ms", "n_ref"):
            if prev.get(k) != rec[k]:
                stop("f1", f"F1 mismatch on {k}: {prev.get(k)} vs {rec[k]}")
    else:
        f1p.write_text(json.dumps(rec, indent=1))
    log("F1 record in place", "f1")
    mark("f1")


def stage_gate():
    if done("gate"):
        return
    from .register import npz_path
    kept, excluded, missing = [], [], []
    for name in heldout_names():
        try:
            z = np.load(npz_path(name))
        except FileNotFoundError:
            missing.append(name)
            continue
        t, base = z["t"].astype(np.float64), z["base"]
        dt = np.diff(t)
        ok = dt > 1e-3
        dzdt = np.abs(np.diff(base[:, 2])[ok] / dt[ok])
        p99 = float(np.percentile(dzdt, 99))
        (kept if p99 <= GATE_DZDT else excluded).append([name, round(p99, 4)])
    if missing:
        stop("gate", f"registration npz missing for {missing}")
    (V2OUT / "gate.json").write_text(json.dumps(
        {"threshold_ms": GATE_DZDT, "kept": kept, "excluded": excluded}, indent=1))
    log(f"kept {len(kept)}, excluded {[e[0] for e in excluded]}", "gate")
    mark("gate")


def stage_ebuild():
    if done("ebuild"):
        return
    from .v2_build import build_v2
    from .paths import window_dir
    kept = [k[0] for k in json.loads((V2OUT / "gate.json").read_text())["kept"]]
    HELDOUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in kept:
        link = HELDOUT_DIR / name
        src = RAW / name / name
        if not link.exists():
            if not src.exists():
                stop("ebuild", f"raw data missing for {name} at {src}")
            link.symlink_to(src)
        if all(any(window_dir(name, f"{c}_v2").glob("window_*.npz")) for c in CONDS):
            log(f"{name}: windows exist, skipping", "ebuild")
            continue
        build_v2([name])
    log("eval builds complete", "ebuild")
    mark("ebuild")


def stage_escore():
    if done("escore"):
        return
    from .paths import window_dir
    kept = [k[0] for k in json.loads((V2OUT / "gate.json").read_text())["kept"]]
    corr = correction()
    for cond in CONDS:
        outp = V2OUT / f"heldout_score_{cond}.json"
        if outp.exists():
            continue
        rows = []
        for name in kept:
            rows.extend(_score_dir(window_dir(name, f"{cond}_v2").parent
                                   / window_dir(name, f"{cond}_v2").name,
                                   N_REF, corr, fan_cfg=FAN))
        outp.write_text(json.dumps(rows, indent=1))
        log(f"{cond}: {len(rows)} windows scored", "escore")
    mark("escore")


def stage_report():
    if done("report"):
        return
    rep = {"criteria": {}, "tables": {}}
    for cond in CONDS:
        rows = [r for r in json.loads((V2OUT / f"heldout_score_{cond}.json").read_text())
                if r.get("scoreable") and not r["flagged"]]
        n = len(rows)
        a = lambda r, arm, k_: r["arms"][arm][k_]
        w1 = sum(a(r, "clark-corr", "rel_err_E") < a(r, "fosm", "rel_err_E") for r in rows)
        cvar_meds = {arm: med([a(r, arm, "cvar_abs_err") for r in rows])
                     for arm in rows[0]["arms"]} if rows else {}
        lowest = min(cvar_meds, key=cvar_meds.get) if cvar_meds else None
        w2 = sum(a(r, "clark-corr", "cvar_abs_err") < a(r, lowest, "cvar_abs_err")
                 for r in rows) if lowest and lowest != "clark-corr" else None
        w3 = sum(a(r, "clark-corr", "rel_err_E") < a(r, "mc-32", "rel_err_E")
                 and a(r, "clark-corr", "rel_err_sd") < a(r, "mc-32", "rel_err_sd")
                 for r in rows)
        rep["criteria"][cond] = {
            "n_unflagged": n,
            "V2-1": {"wins": w1, "n": n, "p": sign_p(w1, n),
                     "pass": (w1 > n / 2 and sign_p(w1, n) < 0.05)},
            "V2-2": {"lowest_median": lowest,
                     "clark_corr_median": cvar_meds.get("clark-corr"),
                     "tie_p_vs_lowest": (sign_p(w2, n) if w2 is not None else None),
                     "pass": (lowest == "clark-corr" or
                              (w2 is not None and sign_p(w2, n) >= 0.05))},
            "V2-3": {"wins": w3, "n": n, "p": sign_p(w3, n),
                     "pass": (w3 > n / 2 and sign_p(w3, n) < 0.05)},
        }
        rep["tables"][cond] = {
            "claim1": {arm: {"med_rel_err_E": med([a(r, arm, "rel_err_E") for r in rows]),
                             "med_rel_err_sd": med([a(r, arm, "rel_err_sd")
                                                    for r in rows if "rel_err_sd" in r["arms"][arm]])}
                       for arm in ("clark-corr", "clark", "fosm")},
            "claim2_cvar": {arm: {"median": cvar_meds.get(arm),
                                  "p95": (sorted(a(r, arm, "cvar_abs_err") for r in rows)
                                          [max(0, int(0.95 * n) - 1)] if n else None)}
                            for arm in rows[0]["arms"]} if rows else {},
            "deficit": {"clark_raw_sd_ratio_med": med([a(r, "clark", "sd_ratio_to_mc") for r in rows]),
                        "clark_corr_sd_ratio_med": med([a(r, "clark-corr", "sd_ratio_to_mc") for r in rows])},
            "fan": {"n_scoreable": sum(1 for r in rows if r.get("fan", {}).get("scoreable")),
                    "zero_regret_rate": {}},
        }
    (V2OUT / "REPORT_v2.json").write_text(json.dumps(rep, indent=1))
    lines = ["# v2 campaign report (machine-generated; verdicts below)\n"]
    for cond, c in rep["criteria"].items():
        lines.append(f"## {cond} (n={c['n_unflagged']})")
        for crit in ("V2-1", "V2-2", "V2-3"):
            lines.append(f"- {crit}: {'PASS' if c[crit]['pass'] else 'FAIL'} {c[crit]}")
    (V2OUT / "FINAL_REPORT_v2.md").write_text("\n".join(lines) + "\n")
    log("report written", "report")
    mark("report")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", default="report")
    a = ap.parse_args()
    order = ["config", "dbuild", "dcheck", "f1", "gate", "ebuild", "escore", "report"]
    for st in order:
        globals()[f"stage_{st}"]()
        if st == a.until:
            break
    log("runner finished", "main")


if __name__ == "__main__":
    main()
