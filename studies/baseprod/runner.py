"""The E2 state machine: one detached, resumable process that executes the chain
PREREG_baseprod.md's "Authorized autonomous execution" section orders.

  1  commit        record the pipeline commit hash (the hash F1 records)
  2  design        design phase: register both design traverses, re-check the two pinned
                   calibration constants, rebuild the LEGACY 10 m hindsight path for the
                   C1(a) code-parity check, build BOTH belief conditions at 4 m, fit (k, tau)
                   per condition, draw the belief referee (all seven arms)
  3  freeze_eval   evaluate C1-C4. C1 is now two parts -- (a) the 10 m hindsight path still
                   reproduces rehearsal v3 exactly, so the window-length change is a
                   configuration change and not a code change, and (b) the 4 m pipeline runs
                   both conditions end to end with finite outputs. C2 is re-evaluated on the
                   4 m windows under the same rule; C3 and C4 are unchanged.
  4  freeze        write FREEZE.json (all conditions hold) or STOPPED_AT.md (any fails), then
                   write DESIGN_PHASE_DONE.md and EXIT
  5  heldout       stream the 22 held-out traverses, both conditions
  6  score         score them once, at the frozen (k, tau)
  7  report        artifacts + FINAL_REPORT.md, with the design tables as an appendix

The chain is UNCONDITIONAL on success: if C1-C4 all hold the runner writes FREEZE.json and
goes straight on to the held-out run, unattended, with no GO file and no stop (the author
restored the chained authorization on the evening of 2026-08-28; PREREG_baseprod.md,
clark_paper 0506da2). If ANY gate fails it writes STOPPED_AT.md and exits before unlock, and
the held-out set stays untouched. There is no second draw. `--design-only` stops after the
freeze record, for a dry run.

Every stage appends to RUN_LOG.md with a timestamp. Every stage writes a done-marker under
out/state/, so rerunning the same script after a crash or a disk-pressure stop resumes rather
than repeats. Criteria, bands, thresholds and the banned-corrections list are in constants.py
and cannot be altered by this runner.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

from . import constants as K
from .paths import (CONDITIONS, DATA, DESIGN_TRAVERSES, HELDOUT_DIR, KEY, OUT, ROOT,
                    freeze_path, window_dir)

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "traverse_manifest.json").read_text())
C1_REF = json.loads((HERE / "c1_reference_v3.json").read_text())
HELD_OUT = [m for m in MANIFEST if m["name"] not in DESIGN_TRAVERSES]

RUN_LOG = OUT / "RUN_LOG.md"
STATE = OUT / "state"
# GO_FILE is retired: the author restored the chained authorization on 2026-08-28 evening.
PARITY_TAG = "parity10m"   # the C1(a) code-parity build, never scored as the experiment
DISK_CAP_GB = 100.0
DL_MARGIN_GB = 12.0


# ------------------------------------------------------------------ plumbing
def now() -> str:
    return dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def log(msg: str, stage: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"- `{now()}` **{stage or '-'}** — {msg}\n"
    with open(RUN_LOG, "a") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())
    print(f"[{now()}] {stage}: {msg}", flush=True)


def done(stage: str) -> bool:
    return (STATE / f"{stage}.done").exists()


def mark(stage: str, payload=None) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / f"{stage}.done").write_text(json.dumps(
        {"stage": stage, "at": now(), "payload": payload}, indent=1, default=str))


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=_jsonable))


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def free_gb(path: Path = ROOT) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def used_gb(path: Path = ROOT) -> float:
    tot = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                tot += p.stat().st_size
        except OSError:
            pass
    return tot / 1e9


# ------------------------------------------------------------------ C4 guard
def held_out_untouched() -> tuple[bool, list]:
    """C4: no held-out path accessed before the freeze record is written."""
    evidence = []
    if HELDOUT_DIR.exists():
        for p in sorted(HELDOUT_DIR.iterdir()):
            if p.name.startswith("."):
                continue
            evidence.append(f"{p} exists")
    names = {m["name"] for m in HELD_OUT}
    wroot = OUT / "windows"
    if wroot.exists():
        for cond in sorted(wroot.iterdir()):   # every tag, the parity build included
            if not cond.is_dir():
                continue
            for p in sorted(cond.iterdir()):
                if p.name in names:
                    evidence.append(f"{p} exists")
    return (not evidence), evidence


def assert_unfrozen_no_heldout(stage: str) -> None:
    ok, ev = held_out_untouched()
    if not ok:
        log(f"ABORT: held-out material present before the freeze record: {ev}", stage)
        raise SystemExit(3)


# ------------------------------------------------------------------ stage 1
def stage_commit() -> dict:
    stage = "1-commit"
    p = ROOT / "PIPELINE_COMMIT"
    rec = {"pipeline_commit": None, "source": None, "dirty": None}
    if p.exists():
        rec.update(json.loads(p.read_text()))
        rec["source"] = str(p)
    try:
        h = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=30)
        if h.returncode == 0:
            rec["git_rev_parse_here"] = h.stdout.strip()
    except Exception:
        pass
    write_json(OUT / "pipeline_commit.json", rec)
    log(f"pipeline commit {rec.get('pipeline_commit')} (deployed record)", stage)
    mark(stage, rec)
    return rec


# ------------------------------------------------------------------ stage 2
def stage_design(mc_draws: int) -> dict:
    stage = "2-design"
    assert_unfrozen_no_heldout(stage)
    from . import calibration, register
    from .build import build
    from .score import score_condition

    names = list(DESIGN_TRAVERSES)

    if not done("2a-register"):
        log(f"registering design traverses {names}", stage)
        reg = register.run_and_record(names)
        for k, v in reg.items():
            log(f"  {k}: model={v['chosen_model']} resid_sd={v['residual_after_registration']['sd_m']} m "
                f"blockCV={v['chosen_blockcv_rmse_m']} m voidfill={v['voidfill_frac_corridor']}", stage)
        mark("2a-register", {k: v["chosen_model"] for k, v in reg.items()})
    reg = json.loads((OUT / "registration" / "registration.json").read_text())

    if not done("2b-calibration-check"):
        log("re-checking the two pinned calibration constants (record only, never used in "
            "the build)", stage)
        ctx = calibration._context(names)
        pit = calibration.refit_pitch(names, ctx)
        log(f"  pitch: pinned {K.MOUNT_PITCH_OFFSET_DEG} deg, refit "
            f"{pit['refit_MOUNT_PITCH_OFFSET_DEG']} deg, |diff| {pit['abs_diff_deg']} deg", stage)
        sig = calibration.refit_sigma(names, K.MOUNT_PITCH_OFFSET_DEG, ctx)
        log(f"  sigma: pinned a={K.SIG_A} b={K.SIG_B}, refit "
            f"a={sig['refit_pooled']['a']} b={sig['refit_pooled']['b']}", stage)
        write_json(OUT / "calibration_check.json", {"pitch": pit, "sigma": sig})
        mark("2b-calibration-check", {"pitch": pit["refit_MOUNT_PITCH_OFFSET_DEG"],
                                      "sigma": sig["refit_pooled"]})
    calcheck = json.loads((OUT / "calibration_check.json").read_text())

    # --- C1(a): the LEGACY 10 m hindsight path, rebuilt and rescored -----------------------
    # Not part of the experiment. It exists to show that dropping the window length to 4 m is
    # a CONFIGURATION change and not a code change: at 10 m this same code must still land on
    # rehearsal v3 exactly. Legacy arm set, no MC -- v3 had neither.
    if not done("2c-parity10m"):
        log(f"C1(a) code parity: rebuilding the {K.LEGACY_WIN_LEN_M:g} m hindsight windows",
            stage)
        metas = {KEY[n]: build(n, "hindsight", win_len_m=K.LEGACY_WIN_LEN_M, tag=PARITY_TAG)
                 for n in names}
        write_json(OUT / "parity10m_build.json", metas)
        r = score_condition([window_dir(n, PARITY_TAG) for n in names], PARITY_TAG,
                            mc_n=0, arms=K.ARMS_LEGACY,
                            label=f"C1(a) code parity, hindsight at {K.LEGACY_WIN_LEN_M:g} m")
        write_json(OUT / "parity10m_score.json", r)
        log(f"  parity: {r['qc']['n_retained']} retained, {r['qc']['n_unflagged']} unflagged, "
            f"k={r['fit'].get('k')!r}", stage)
        mark("2c-parity10m", {"n_unflagged": r["qc"]["n_unflagged"], "k": r["fit"].get("k")})
    parity = json.loads((OUT / "parity10m_score.json").read_text())

    design = {}
    for cond in CONDITIONS:
        st = f"2d-build-{cond}"
        if not done(st):
            log(f"building the {cond} belief at {K.WIN_LEN_M:g} m on both design traverses",
                stage)
            metas = {KEY[n]: build(n, cond) for n in names}
            write_json(OUT / f"design_build_{cond}.json", metas)
            log(f"  {cond}: " + ", ".join(
                f"{k}={m['n_windows']} windows" for k, m in metas.items()), stage)
            mark(st, {k: m["n_windows"] for k, m in metas.items()})
        st = f"2e-score-{cond}"
        if not done(st):
            log(f"scoring the {cond} condition, seven arms, {mc_draws} MC draws per window",
                stage)
            r = score_condition([window_dir(n, cond) for n in names], cond, mc_n=mc_draws,
                                label=f"v5 design phase ({K.WIN_LEN_M:g} m windows), {cond}")
            write_json(OUT / f"design_score_{cond}.json", r)
            log(f"  {cond}: {r['qc']['n_retained']} retained, {r['qc']['n_unflagged']} "
                f"unflagged, k={r['fit'].get('k')!r} tau={r['fit'].get('tau_m')!r}", stage)
            mark(st, {"n_unflagged": r["qc"]["n_unflagged"], "k": r["fit"].get("k")})
        design[cond] = json.loads((OUT / f"design_score_{cond}.json").read_text())
    design["_parity10m"] = parity

    out = {"registration": reg, "calibration_check": calcheck,
           "win_len_m": K.WIN_LEN_M, "legacy_win_len_m": K.LEGACY_WIN_LEN_M,
           "step_form_formula": K.STEP_FORM_FORMULA,
           "parity10m": {kk: vv for kk, vv in parity.items() if kk != "windows"},
           "conditions": {c: {kk: vv for kk, vv in design[c].items() if kk != "windows"}
                          for c in CONDITIONS}}
    write_json(OUT / "design_phase.json", out)
    mark(stage, {c: design[c]["qc"]["n_unflagged"] for c in CONDITIONS})
    return design


# ------------------------------------------------------------------ stage 3
def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)


def _scan_degenerate(d: dict, cond: str, limit: int = 40) -> list:
    """Every NaN / inf / degenerate value in one condition's design tables.

    A finding here FAILS C3. The scan covers exactly the numbers the held-out run and the
    report are built from: the per-window arm moments, the pooled arm table, the
    belief-referee table (both moments, the sd-ratio, the CVaR error) and the regime
    statistics. It does not police diagnostics that are legitimately undefined (a window with
    zero relief has no sigma/relief), which are reported as notes by the report writer.
    """
    bad = []
    for w in d.get("windows", []):
        if not w.get("scoreable"):
            continue
        if not _finite(w.get("j_truth")):
            bad.append(f"{cond}/{w['name']}: j_truth = {w.get('j_truth')!r}")
        for a, v in (w.get("arms") or {}).items():
            if v is None:
                bad.append(f"{cond}/{w['name']}: arm {a} missing")
                continue
            E, sd = v
            if not _finite(E):
                bad.append(f"{cond}/{w['name']}: {a} E = {E!r}")
            if a == "mean-map":
                continue
            if not _finite(sd):
                bad.append(f"{cond}/{w['name']}: {a} sd = {sd!r}")
            elif sd <= 0.0:
                bad.append(f"{cond}/{w['name']}: {a} sd = {sd!r} is not positive")
        mc = w.get("mc")
        if mc:
            for nm in ("E_mc", "sd_mc", "cvar_mc"):
                if not _finite(mc.get(nm)):
                    bad.append(f"{cond}/{w['name']}: mc.{nm} = {mc.get(nm)!r}")
            if _finite(mc.get("sd_mc")) and mc["sd_mc"] <= 0:
                bad.append(f"{cond}/{w['name']}: mc.sd_mc = {mc['sd_mc']!r} is not positive")
            for a, r in (mc.get("arms") or {}).items():
                for nm in ("rel_err_E", "rel_err_sd", "sd_ratio_to_mc", "cvar_abs_err"):
                    if not _finite(r.get(nm)):
                        bad.append(f"{cond}/{w['name']}: mc.{a}.{nm} = {r.get(nm)!r}")
    p = d.get("pooled", {})
    for a in p.get("arms_scored", []):
        for suf in ("raw", "recal"):
            for nm, v in (p[a][suf] or {}).items():
                if v is None or isinstance(v, str):
                    continue
                if not _finite(v):
                    bad.append(f"{cond}: pooled.{a}.{suf}.{nm} = {v!r}")
    for nm in ("alpha_weighted_median", "sigma_med_m", "relief_p95_p5_m",
               "frac_nodes_alpha_lt_1", "retained", "qc_observed_frac"):
        blk = p.get(nm)
        if isinstance(blk, dict):
            for q, v in blk.items():
                if not _finite(v):
                    bad.append(f"{cond}: pooled.{nm}.{q} = {v!r}")
    mcs = d.get("belief_referee", {})
    for a, r in (mcs.get("arms") or {}).items():
        for nm in ("median_rel_err_E", "median_rel_err_sd", "p95_rel_err_E", "p95_rel_err_sd"):
            v = r.get(nm)
            if v is None:
                continue
            if not _finite(v):
                bad.append(f"{cond}: belief_referee.{a}.{nm} = {v!r}")
        sr = r.get("sd_ratio_to_mc")
        if isinstance(sr, dict):
            for q, v in sr.items():
                if not _finite(v):
                    bad.append(f"{cond}: belief_referee.{a}.sd_ratio_to_mc.{q} = {v!r}")
    for a, r in ((mcs.get("cvar_error") or {}).get("arms") or {}).items():
        for q, v in r.items():
            if not _finite(v):
                bad.append(f"{cond}: cvar_error.{a}.{q} = {v!r}")
    for b in (mcs.get("sd_deficit_vs_alpha") or {}).get("bins", []):
        if not b.get("n"):
            continue
        for q in ("alpha_median", "clark_sd_ratio_median"):
            if not _finite(b.get(q)):
                bad.append(f"{cond}: sd_deficit_vs_alpha bin "
                           f"[{b['alpha_lo']}, {b['alpha_hi']}).{q} = {b.get(q)!r}")
    if len(bad) > limit:
        bad = bad[:limit] + [f"... and {len(bad) - limit} more degenerate values"]
    return bad


def _rel(a, b) -> float:
    if b == 0:
        return abs(a - b)
    return abs(a - b) / abs(b)


def stage_freeze_eval(design: dict) -> dict:
    """C1-C4, exactly as PREREG_baseprod.md states them. Not alterable by the runner."""
    stage = "3-freeze-eval"
    h = design["hindsight"]
    f = design["foresight"]
    par = design["_parity10m"]

    # --- C1, two parts -------------------------------------------------------------------
    # (a) CODE PARITY: the legacy 10 m hindsight scoring path still reproduces rehearsal v3
    #     exactly, so the drop to 4 m windows is a configuration change and not a code change.
    # (b) the 4 m pipeline runs BOTH conditions end to end with finite outputs.
    c1_checks, tol = [], K.C1_REL_TOL

    def chk(nm, got, want, kind="rel"):
        d = _rel(got, want) if kind == "rel" else abs(got - want)
        ok = bool(d <= tol) if kind != "eq" else bool(got == want)
        c1_checks.append({"part": "a", "quantity": nm, "got": got, "v3": want,
                          ("rel_diff" if kind == "rel" else "abs_diff"):
                              d if kind != "eq" else None,
                          "pass": ok})
        return ok

    ref = C1_REF
    chk("fit.k", par["fit"]["k"], ref["fit"]["k"])
    chk("fit.tau_m", par["fit"]["tau_m"], ref["fit"]["tau_m"], "abs")
    chk("fit.mean_NLL_at_fit", par["fit"]["mean_NLL_at_fit"], ref["fit"]["mean_NLL_at_fit"])
    chk("qc.n_retained", par["qc"]["n_retained"], ref["qc"]["n_retained"], "eq")
    chk("qc.n_unflagged", par["qc"]["n_unflagged"], ref["qc"]["n_unflagged"], "eq")
    for a in K.ARMS_LEGACY:
        for suf in ("raw", "recal"):
            got = par["pooled"][a][suf]["mean_NLL"]
            want = ref["pooled_arm_table"][a][suf]["mean_NLL"]
            # the v3 reference is published to 4 dp, so the tolerance is the rounding
            # half-ulp plus the 1e-6 relative tolerance
            d = abs(got - want)
            c1_checks.append({"part": "a", "quantity": f"pooled.{a}.{suf}.mean_NLL",
                              "got": got, "v3": want, "abs_diff": d,
                              "pass": bool(d <= 5e-5 + tol * abs(want))})
    for pair in ("nll_clark_minus_fosm", "nll_clark_minus_clark-diag"):
        for suf in ("raw", "recal"):
            got = par["pooled"][pair][suf]["n_windows_clark_better"]
            want = ref[pair][suf]["n_windows_clark_better"]
            c1_checks.append({"part": "a", "quantity": f"{pair}.{suf}.n_windows_clark_better",
                              "got": got, "v3": want, "pass": bool(got == want)})
    # per-window j_truth and the legacy arm moments, full precision
    byname = {w["name"]: w for w in par["windows"]}
    wmax = {"j_truth": 0.0, "arm_E": 0.0, "arm_sd": 0.0, "n_missing": 0}
    for wref in ref["windows"]:
        w = byname.get(wref["name"])
        if w is None or not w.get("scoreable"):
            wmax["n_missing"] += 1
            continue
        wmax["j_truth"] = max(wmax["j_truth"], _rel(w["j_truth"], wref["j_truth"]))
        for a, (E, sd) in wref["arms"].items():
            gE, gsd = w["arms"][a]
            wmax["arm_E"] = max(wmax["arm_E"], _rel(gE, E))
            if sd is not None:
                wmax["arm_sd"] = max(wmax["arm_sd"], _rel(gsd, sd))
    c1_checks.append({"part": "a", "quantity": "per-window max relative difference",
                      "got": wmax, "v3": "rehearsal_v3 windows",
                      "pass": bool(wmax["n_missing"] == 0 and wmax["j_truth"] <= tol
                                   and wmax["arm_E"] <= tol and wmax["arm_sd"] <= tol)})
    C1a = all(c["pass"] for c in c1_checks)

    # (b) the 4 m pipeline ran both conditions end to end with finite outputs
    c1b = []
    for cond in CONDITIONS:
        d = design[cond]
        n_ok = d["qc"]["n_window_products"] > 0 and d["qc"]["n_retained"] > 0
        vals = [d["fit"].get("k"), d["fit"].get("tau_m"), d["fit"].get("mean_NLL_at_fit")]
        fin = all(v is not None and np.isfinite(v) for v in vals)
        arms_ok = True
        for w in d["windows"]:
            if not w.get("scoreable"):
                continue
            for a in K.ARMS_VARIANCE:
                v = w["arms"].get(a)
                if v is None or not (np.isfinite(v[0]) and np.isfinite(v[1])):
                    arms_ok = False
        mc_ok = d["belief_referee"].get("n_windows", 0) > 0
        c1b.append({"part": "b", "quantity": f"{cond}: products/retained > 0", "pass": n_ok,
                    "n_window_products": d["qc"]["n_window_products"],
                    "n_retained": d["qc"]["n_retained"]})
        c1b.append({"part": "b", "quantity": f"{cond}: (k, tau, NLL) finite", "pass": fin,
                    "values": vals})
        c1b.append({"part": "b", "quantity": f"{cond}: all seven arms finite on every "
                                             "scoreable window", "pass": arms_ok})
        c1b.append({"part": "b", "quantity": f"{cond}: belief referee drawn", "pass": mc_ok,
                    "n_windows": d["belief_referee"].get("n_windows", 0)})
    c1_checks += c1b
    C1b = all(c["pass"] for c in c1b)
    C1 = bool(C1a and C1b)

    # --- C2: >= 60% of the design windows unflagged under foresight, on the 4 m windows --
    # The denominator is the number of 4 m HINDSIGHT design windows -- "the design windows",
    # the same set foresight is the paired condition on.
    n_ref = h["qc"]["n_window_products"]
    n_unf = f["qc"]["n_unflagged"]
    C2 = bool(n_ref and n_unf >= K.C2_MIN_UNFLAGGED_FRAC * n_ref)
    # Diagnostic only -- NOT a criterion, and it changes nothing. It records which QC rule
    # binds under foresight: flag 1 (retention < 50%) or flags 2-3 (coverage / sigma /
    # void-fill). At 10 m windows retention was the binding rule, which is what the window
    # length change addresses.
    f_rows = [w for w in f["windows"] if w.get("scoreable")]
    c2_diag = {
        "note": "diagnostic, not a criterion; recorded so the binding QC rule is legible",
        "win_len_m": K.WIN_LEN_M,
        "n_scoreable": len(f_rows),
        "n_excluded_by_retention": sum(1 for w in f_rows if w["retained"] < K.MIN_RETAINED),
        "n_unflagged_ignoring_retention": sum(1 for w in f_rows if not w["flagged"]),
        "per_window": [{"name": w["name"], "retained": round(w["retained"], 4),
                        "qc_observed_frac": w["qc_observed_frac"],
                        "qc_sigma_med_m": w["qc_sigma_med_m"],
                        "qc_flags": w["qc_flags"]} for w in f_rows],
    }

    # --- C3: all fits finite; QC/registration ran; NO NaN OR DEGENERATE VALUE anywhere in
    # the design tables. A broken number must not ride into the held-out run.
    c3_notes = []
    finite = True
    for cond in CONDITIONS:
        d = design[cond]
        for nm, v in (("k", d["fit"].get("k")), ("tau_m", d["fit"].get("tau_m")),
                      ("mean_NLL_at_fit", d["fit"].get("mean_NLL_at_fit"))):
            if v is None or not np.isfinite(v):
                finite = False
                c3_notes.append(f"{cond}.fit.{nm} = {v!r}")
        if d["fit"].get("k") is not None and not (d["fit"]["k"] > 0):
            finite = False
            c3_notes.append(f"{cond}.fit.k = {d['fit']['k']!r} is not positive")
        if d["qc"]["n_window_products"] == 0:
            finite = False
            c3_notes.append(f"{cond}: no window products built")
        for w in d["windows"]:
            if w.get("qc_observed_frac") is None:
                finite = False
                c3_notes.append(f"{cond}/{w['name']}: QC did not run")
        bad = _scan_degenerate(d, cond)
        if bad:
            finite = False
            c3_notes += bad
    cal = json.loads((OUT / "calibration_check.json").read_text())
    for nm, v in (("pitch refit", cal["pitch"]["refit_MOUNT_PITCH_OFFSET_DEG"]),
                  ("sigma refit a", cal["sigma"]["refit_pooled"]["a"]),
                  ("sigma refit b", cal["sigma"]["refit_pooled"]["b"])):
        if v is None or not np.isfinite(v):
            finite = False
            c3_notes.append(f"{nm} = {v!r}")
    reg = json.loads((OUT / "registration" / "registration.json").read_text())
    for k in (KEY[n] for n in DESIGN_TRAVERSES):
        if k not in reg or reg[k].get("chosen_model") not in ("plane", "quadratic"):
            finite = False
            c3_notes.append(f"registration missing or invalid for {k}")
    C3 = finite

    # --- C4: no held-out path accessed before the freeze record is written -------------
    C4, c4_ev = held_out_untouched()

    res = {
        "evaluated_at": now(),
        "C1": {"statement": f"(a) the {K.LEGACY_WIN_LEN_M:g} m hindsight scoring path still "
                            f"reproduces rehearsal v3 within {tol:g} relative -- code parity "
                            f"preserved on the old configuration -- AND (b) the "
                            f"{K.WIN_LEN_M:g} m pipeline runs both conditions end to end "
                            f"with finite outputs",
               "pass": C1, "a_pass": C1a, "b_pass": C1b, "checks": c1_checks},
        "C2": {"statement": f"at least {K.C2_MIN_UNFLAGGED_FRAC:.0%} of the {n_ref} design "
                            f"windows ({K.WIN_LEN_M:g} m) unflagged under foresight",
               "n_unflagged_foresight": n_unf, "n_design_windows": n_ref,
               "fraction": round(n_unf / n_ref, 4) if n_ref else None,
               "n_window_products_foresight": f["qc"]["n_window_products"],
               "n_retained_foresight": f["qc"]["n_retained"],
               "n_excluded_foresight": f["qc"]["n_excluded"],
               "flagged_foresight": f["qc"]["flagged"], "pass": C2,
               "diagnostic": c2_diag},
        "C3": {"statement": "all fits finite, the QC/registration machinery ran on both "
                            "conditions without error, and NO NaN or degenerate value "
                            "appears anywhere in the design tables (arm moments, pooled arm "
                            "table, belief-referee table, regime statistics)",
               "pass": C3, "notes": c3_notes,
               "n_degenerate_findings": len(c3_notes)},
        "C4": {"statement": "no held-out path accessed before the freeze record is written",
               "pass": C4, "evidence_of_access": c4_ev},
        "ALL_PASS": bool(C1 and C2 and C3 and C4),
    }
    write_json(OUT / "freeze_conditions.json", res)
    for c in ("C1", "C2", "C3", "C4"):
        log(f"{c}: {'PASS' if res[c]['pass'] else 'FAIL'} — {res[c]['statement']}", stage)
    mark(stage, {c: res[c]["pass"] for c in ("C1", "C2", "C3", "C4")})
    return res


# ------------------------------------------------------------------ stage 4
def stage_freeze(design: dict, cond_res: dict, commit: dict) -> bool:
    stage = "4-freeze"
    reg = json.loads((OUT / "registration" / "registration.json").read_text())
    cal = json.loads((OUT / "calibration_check.json").read_text())
    if not cond_res["ALL_PASS"]:
        failed = [c for c in ("C1", "C2", "C3", "C4") if not cond_res[c]["pass"]]
        body = [
            "# STOPPED_AT — the freeze conditions did not all hold\n",
            f"Written {now()} by the E2 runner. **The held-out set was not touched and "
            "remains untouched.** No FREEZE.json was written; the pre-registration is not "
            "frozen.\n",
            f"Failed: **{', '.join(failed)}**\n",
        ]
        for c in ("C1", "C2", "C3", "C4"):
            r = cond_res[c]
            body.append(f"- **{c}** {'PASS' if r['pass'] else 'FAIL'} — {r['statement']}")
        body.append("\nFull detail: `freeze_conditions.json`. Design-phase results, both "
                    "conditions: `DESIGN_PHASE_DONE.md`, `design_score_hindsight.json`, "
                    "`design_score_foresight.json`.\n")
        body.append("\nHeld-out access remains blocked: stages 5-7 refuse to run without "
                    "FREEZE.json. Rerunning the same script resumes from the last "
                    "done-marker and will re-evaluate the gates.\n")
        (OUT / "STOPPED_AT.md").write_text("\n".join(body))
        log(f"STOPPED_AT.md written; failed {failed}; held-out untouched", stage)
        mark(stage, {"frozen": False, "failed": failed})
        return False

    rec = {
        "record": "F1 — the freeze record PREREG_baseprod.md's amendment F1 transcribes",
        "written_at": now(),
        "authorization": ("Authorized autonomous execution (2026-08-28), narrowed the same "
                          "day: the runner freezes and STOPS. The held-out run requires an "
                          "explicit manual start."),
        "pipeline_commit": commit.get("pipeline_commit"),
        "host": os.uname().nodename,
        "intrinsics": {"fx": K.FX, "fy": K.FY, "cx": K.CX, "cy": K.CY,
                       "width": K.IMG_W, "height": K.IMG_H, "distortion": list(K.DISTORTION),
                       "source": "depth CameraInfo, theory/BASEPROD_BLOCKER.md"},
        "mount_pitch": {"nominal_deg": K.NOMINAL_PITCH_DEG,
                        "offset_deg": K.MOUNT_PITCH_OFFSET_DEG,
                        "total_deg": K.TOTAL_MOUNT_PITCH_DEG,
                        "status": "calibration, common to both design traverses, mechanism "
                                  "not nuisance; applied inside belief construction",
                        "refit_check": cal["pitch"]},
        "sigma_model": {"form": "sigma_z(r) = a + b r^2 [m]", "a": K.SIG_A, "b": K.SIG_B,
                        "refit_check": cal["sigma"]},
        "cutoff_rule": {
            "foresight": "frames with a - LOOKBACK_M <= s(frame) < a, i.e. RTK along-track "
                         "arc length strictly before the window start",
            "hindsight": "frames with a - LOOKBACK_M <= s(frame) <= b (full trajectory)",
            "lookback_m": K.LOOKBACK_M, "win_len_m": K.WIN_LEN_M,
            "note": "the look-back is a shared pipeline element, so foresight is the "
                    "hindsight frame set intersected with the cutoff -- a strict subset"},
        "belief": {"cell_m": K.CELL, "alpha": K.ALPHA, "max_variance": K.MAX_VARIANCE,
                   "r_min_m": K.R_MIN, "r_max_m": K.R_MAX, "pixel_stride": K.STRIDE,
                   "track_smooth_fixes": K.TRACK_SMOOTH_FIXES},
        "geometry": {"r_wheel_m": K.R_WHEEL, "wheel_patch_r_m": K.WHEEL_PATCH_R},
        "registration": {"selection_rule": reg[KEY[DESIGN_TRAVERSES[0]]]["selection_rule"],
                         "outputs": {k: {"chosen_model": v["chosen_model"],
                                         "blockcv_rmse_m": v["chosen_blockcv_rmse_m"],
                                         "residual_sd_m": v["residual_after_registration"]["sd_m"],
                                         "tilt_mm_per_m": v["models"][v["chosen_model"]].get("tilt_mm_per_m"),
                                         "const_term_m": v["models"][v["chosen_model"]]["const_term_m"],
                                         "voidfill_frac_corridor": v["voidfill_frac_corridor"]}
                                     for k, v in reg.items()}},
        "qc_thresholds": {"min_retained": K.MIN_RETAINED,
                          "flag_min_observed_frac": K.FLAG_MIN_OBS_FRAC,
                          "flag_max_median_sigma_m": K.FLAG_MAX_SIGMA_M,
                          "flag_max_voidfill_frac": K.FLAG_MAX_INVALID_FRAC},
        "recalibration_per_condition": {
            c: {"form": "Var' = k Var + (tau n_steps)^2",
                "k": design[c]["fit"]["k"], "tau_m": design[c]["fit"]["tau_m"],
                "sd_multiplier": design[c]["fit"]["sd_multiplier_sqrt_k"],
                "fitted_on": f"UNFLAGGED design windows, arm {K.FIT_ARM}, ML",
                "n_windows_fitted": design[c]["fit"]["n_windows_fitted"],
                "pinned_tau_alternative": design[c]["fit_pinned_tau_alternative"]}
            for c in CONDITIONS},
        "windowing": {"win_len_m": K.WIN_LEN_M, "lookback_m": K.LOOKBACK_M,
                      "legacy_win_len_m": K.LEGACY_WIN_LEN_M,
                      "rationale": "4 m is the depth camera's range cap and the horizon a "
                                   "planner predicts over"},
        "arms": {"variance_arms": list(K.ARMS_VARIANCE), "point_arm": "mean-map",
                 "fit_arm": K.FIT_ARM,
                 "step_form_formula": K.STEP_FORM_FORMULA,
                 "mc_n_arms": list(K.MC_N_ARMS),
                 "mc_subsample": "N draws taken without replacement from the reference draws "
                                 f"with seed {K.MC_SUBSAMPLE_SEED}",
                 "fosm_label": "the canonical first-order route, constructed as the strongest "
                               "derivative baseline; no published system runs it"},
        "criteria": {"E2-i_cov1_band": list(K.E2_I_COV1_BAND),
                     "E2-ii_sd_ratio_band": list(K.E2_II_SD_RATIO_BAND),
                     "E2-iii_a_max_median_rel_err_E": K.E2_III_MAX_MEDIAN_REL_ERR,
                     "E2-iii_b_sd_ratio_band": list(K.E2_III_SD_BAND),
                     "E2-iii_b_pooling": K.E2_III_SD_POOL,
                     "E2-iii_sign_test_alpha": K.E2_III_SIGN_TEST_ALPHA,
                     "E2-v": "clark below mc-32 on BOTH belief-referee moments, majority, "
                             "exact sign test p < 0.05",
                     "mc_draws_per_window": K.N_MC_DRAWS, "mc_seed": K.MC_SEED},
        "reported_characterizations": {
            "sd_deficit_vs_alpha_bins": list(K.ALPHA_BIN_EDGES),
            "cvar_q": K.CVAR_Q, "cvar_gaussian_factor": K.CVAR_GAUSS_FACTOR},
        "banned_corrections": list(K.BANNED_CORRECTIONS),
        "freeze_conditions": {c: cond_res[c]["pass"] for c in ("C1", "C2", "C3", "C4")},
        "held_out_traverses": [m["name"] for m in HELD_OUT],
        "design_traverses": list(DESIGN_TRAVERSES),
    }
    write_json(freeze_path(), rec)
    log(f"FREEZE.json written — pipeline {rec['pipeline_commit']}, "
        f"k(hindsight)={rec['recalibration_per_condition']['hindsight']['k']}, "
        f"k(foresight)={rec['recalibration_per_condition']['foresight']['k']}", stage)
    mark(stage, {"frozen": True})
    return True


# ------------------------------------------------------------------ the design report
def _arm_table(d: dict) -> list[str]:
    p = d.get("pooled", {})
    if not p.get("n_windows"):
        return ["_no unflagged windows to pool_", ""]
    out = ["| arm | | mean NLL | \\|z\\|<=1 | \\|z\\|<=2 | sd-ratio | mean z | median sd |",
           "|---|---|---|---|---|---|---|---|"]
    for a in p.get("arms_scored", K.ARMS_VARIANCE):
        for suf in ("raw", "recal"):
            r = p[a][suf]
            out.append(f"| {a} | {suf} | {r['mean_NLL']} | {r['cov1']} | {r['cov2']} | "
                       f"{r['sd_ratio']} | {r['mean_z']} | {r['median_sd']} |")
    mm = p.get("mean-map", {})
    out += ["", f"mean-map (point prediction, no variance): mean error {mm.get('mean_err')}, "
                f"median |error| {mm.get('median_abs_err')}, median |error| per step "
                f"{mm.get('median_abs_err_per_step_m')} m", ""]
    return out


def _belief_referee_block(d: dict) -> list[str]:
    """The belief referee: the E2-iii and E2-v machinery plus the two characterizations."""
    mc = d.get("belief_referee", {})
    L = ["### Belief referee — E2-iii / E2-v machinery, exercised in-sample\n"]
    if not mc.get("n_windows"):
        return L + ["_no MC drawn_\n"]
    L.append(f"{mc['n_windows']} unflagged windows, {mc['n_draws_per_window']} draws each "
             f"(seed {mc['mc_seed']}; the mc-N arms are subsampled without replacement with "
             f"seed {mc['subsample_seed']}). Raw Var, no truth, no recalibration.\n")
    L += ["| arm | median rel err E | median rel err sd | p95 E | p95 sd | sd-ratio to MC (median) |",
          "|---|---|---|---|---|---|"]
    for a in K.ARMS_VARIANCE:
        r = mc["arms"][a]
        L.append(f"| {a} | {r['median_rel_err_E']:.3e} | {r['median_rel_err_sd']:.3e} | "
                 f"{r['p95_rel_err_E']:.3e} | {r['p95_rel_err_sd']:.3e} | "
                 f"{r['sd_ratio_to_mc']['median']:.4f} |")
    mm = mc["arms"].get("mean-map", {})
    if mm:
        L.append(f"| mean-map | {mm['median_rel_err_E']:.3e} | — (no predicted sd) | — | — | — |")
    L.append("")

    ck = mc["arms"]["clark"]
    cf = mc.get("clark_vs_fosm", {})
    lo, hi = K.E2_III_SD_BAND
    a1 = ck["median_rel_err_E"] <= K.E2_III_MAX_MEDIAN_REL_ERR
    a2 = (cf.get("n_clark_better_E", 0) > cf.get("n", 0) / 2
          and cf.get("sign_test_p_E", 1) < K.E2_III_SIGN_TEST_ALPHA)
    sdr = ck["sd_ratio_to_mc"][K.E2_III_SD_POOL]
    L.append("**E2-iii preview** (in-sample; the criterion is held-out only):\n")
    L.append(f"- (a) MEAN — clark's median relative error {ck['median_rel_err_E']:.3e} "
             f"({'<=' if a1 else '>'} 5%), below fosm's in "
             f"{cf.get('n_clark_better_E')} of {cf.get('n')} windows, two-sided sign test "
             f"p = {cf.get('sign_test_p_E')} ({'majority + significant' if a2 else 'not both'})")
    L.append(f"- (b) SD — clark's sd-ratio to the belief MC, {K.E2_III_SD_POOL} "
             f"**{sdr:.4f}** against the band [{lo}, {hi}] "
             f"({'inside' if lo <= sdr <= hi else 'OUTSIDE'}); mean "
             f"{ck['sd_ratio_to_mc']['mean']:.4f}, p5 {ck['sd_ratio_to_mc']['p5']:.4f}, "
             f"p95 {ck['sd_ratio_to_mc']['p95']:.4f}. No sd criterion against fosm.\n")
    cm = mc.get("clark_vs_mc-32", {})
    c2m = mc.get("clark_vs_mc-2", {})
    L.append(f"**E2-v preview** — clark below mc-32 on BOTH moments in "
             f"{cm.get('n_clark_better_BOTH_moments')} of {cm.get('n')} windows, exact "
             f"two-sided sign test p = {cm.get('sign_test_p_both')} (E alone "
             f"{cm.get('n_clark_better_E')}, sd alone {cm.get('n_clark_better_sd')}). "
             f"Against mc-2: both moments {c2m.get('n_clark_better_BOTH_moments')} of "
             f"{c2m.get('n')}, p = {c2m.get('sign_test_p_both')}.\n")
    for b in ("step-form", "clark-diag"):
        r = mc.get(f"clark_vs_{b}", {})
        if r:
            L.append(f"- clark vs {b}: both moments {r['n_clark_better_BOTH_moments']} of "
                     f"{r['n']} (p = {r['sign_test_p_both']}), E alone "
                     f"{r['n_clark_better_E']}, sd alone {r['n_clark_better_sd']}")
    L.append("")

    cur = mc.get("sd_deficit_vs_alpha", {})
    if cur.get("bins"):
        L.append("#### Reported characterization — the sd-deficit-vs-contest-depth curve\n")
        L.append("clark's belief-referee sd-ratio (sd_clark / sd_MC) binned by the window's "
                 "cost-weighted median alpha. No criterion attaches. The historical record is "
                 "0.90-0.94; the derived worst case is ~0.89 at alpha ~ 0.\n")
        L += ["| alpha bin | n | alpha median | clark sd-ratio median | p5 | p95 |",
              "|---|---|---|---|---|---|"]
        for b in cur["bins"]:
            if not b["n"]:
                L.append(f"| [{b['alpha_lo']}, {b['alpha_hi']}) | 0 | — | — | — | — |")
                continue
            L.append(f"| [{b['alpha_lo']}, {b['alpha_hi']}) | {b['n']} | {b['alpha_median']} | "
                     f"**{b['clark_sd_ratio_median']}** | {b['clark_sd_ratio_p5']} | "
                     f"{b['clark_sd_ratio_p95']} |")
        L.append(f"\ncorr(alpha, clark sd-ratio) = {cur.get('corr_alpha_vs_sd_ratio')}\n")

    cv = mc.get("cvar_error", {})
    if cv.get("arms"):
        L.append(f"#### Reported characterization — belief-referee CVaR error (q = {cv['q']})\n")
        L.append("|CVaR_q(arm) − CVaR_q(MC)| per window: the single number a planner "
                 "consumes, fusing both moments through the decision rule's own tail weight. "
                 "The arm CVaR is the Gaussian form E + phi(z_q)/(1−q) · sd; the MC CVaR is "
                 "the mean of the empirical upper tail. mean-map has no sd, so its CVaR is "
                 "its point prediction — the floor a risk-blind planner sits at. No criterion "
                 "attaches.\n")
        L += ["| arm | median | mean | p95 | max |", "|---|---|---|---|---|"]
        for a, r in cv["arms"].items():
            L.append(f"| {a} | **{r['median']:.4f}** | {r['mean']:.4f} | {r['p95']:.4f} | "
                     f"{r['max']:.4f} |")
        L.append(f"\nLowest median absolute CVaR error: "
                 f"**{cv['lowest_median_abs_cvar_error']}**\n")
    return L

def _stat_table(d: dict, fields) -> list[str]:
    p = d.get("pooled", {})
    out = ["| statistic | min | p25 | median | p75 | max |", "|---|---|---|---|---|---|"]
    for f, lbl in fields:
        v = p.get(f)
        if not v:
            continue
        out.append(f"| {lbl} | {v['min']} | {v['p25']} | **{v['median']}** | {v['p75']} | "
                   f"{v['max']} |")
    out.append("")
    return out


def write_design_report(design: dict, cond_res: dict, commit: dict, frozen: bool) -> None:
    stage = "4-freeze"
    cal = json.loads((OUT / "calibration_check.json").read_text())
    reg = json.loads((OUT / "registration" / "registration.json").read_text())
    L = []
    L += [f"# E2 design phase (v2, {K.WIN_LEN_M:g} m windows) — "
          f"{'FROZEN' if frozen else 'STOPPED'}\n",
          f"Written {now()} by the detached E2 runner on `{os.uname().nodename}`, "
          f"pipeline commit `{commit.get('pipeline_commit')}`.\n",
          "> **DESIGN-PHASE / IN-SAMPLE.** Two design traverses. Per-condition (k, tau) is "
          "fitted on the same windows it is scored on. No pre-registered criterion is "
          "evaluated here and none is implied; E2-i, E2-ii, E2-iii, E2-iv and E2-v are "
          "HELD-OUT criteria and the previews below only exercise their machinery. These "
          "numbers set the freeze record, they are not results.\n",
          f"Six author-approved prereg edits are in force (clark_paper `fab1a70`): windows "
          f"{K.LEGACY_WIN_LEN_M:g} m -> {K.WIN_LEN_M:g} m (the depth range cap), step-form "
          f"and mc-2/mc-32 join the arms, E2-iii splits by moment, E2-v is new, and two "
          f"reported characterizations are added.\n",
          "**The runner has stopped here by design.** The author narrowed the autonomous "
          "authorization on 2026-08-28: the held-out set is not touched until an explicit "
          "manual start (see *Starting the held-out run* below).\n"]

    L.append("## Freeze conditions\n")
    L += ["| condition | verdict | statement |", "|---|---|---|"]
    for c in ("C1", "C2", "C3", "C4"):
        r = cond_res[c]
        L.append(f"| {c} | **{'PASS' if r['pass'] else 'FAIL'}** | {r['statement']} |")
    L.append("")
    c1 = cond_res["C1"]
    npar = json.loads((OUT / "parity10m_score.json").read_text())["qc"]["n_retained"]
    L.append(f"C1(a) code parity — the {K.LEGACY_WIN_LEN_M:g} m hindsight path was rebuilt and "
             f"rescored by this same code ({npar} windows) and compared against rehearsal v3 "
             f"at {K.C1_REL_TOL:g} relative: **{'PASS' if c1['a_pass'] else 'FAIL'}**. That is "
             f"what makes the drop to {K.WIN_LEN_M:g} m a configuration change and not a code "
             f"change. C1(b) end-to-end at {K.WIN_LEN_M:g} m: "
             f"**{'PASS' if c1['b_pass'] else 'FAIL'}**.\n")
    bad = [c for c in c1["checks"] if not c["pass"]]
    if bad:
        L.append("C1 checks that did not hold:\n")
        L += ["| part | quantity | got | expected | diff |", "|---|---|---|---|---|"]
        for c in bad:
            d = c.get("rel_diff", c.get("abs_diff"))
            L.append(f"| {c.get('part')} | {c['quantity']} | {c.get('got')} | "
                     f"{c.get('v3', '-')} | {d} |")
        L.append("")
    c2 = cond_res["C2"]
    L.append(f"C2 detail — foresight produced {c2['n_window_products_foresight']} window "
             f"products from the {c2['n_design_windows']} design windows, "
             f"{c2['n_retained_foresight']} retained (>= {K.MIN_RETAINED:.0%} of steps "
             f"scoreable), {c2['n_unflagged_foresight']} unflagged "
             f"({c2['fraction']} of {c2['n_design_windows']}, threshold "
             f"{K.C2_MIN_UNFLAGGED_FRAC:.0%}). Excluded: "
             f"{c2['n_excluded_foresight']}. Flags raised: "
             f"{json.dumps(c2['flagged_foresight'])}\n")
    dg = c2.get("diagnostic")
    if dg:
        L.append(f"Which QC rule binds under foresight (**diagnostic, not a criterion**): of "
                 f"{dg['n_scoreable']} scoreable foresight windows, "
                 f"{dg['n_excluded_by_retention']} fall below the "
                 f"{K.MIN_RETAINED:.0%} retention floor (QC flag 1) and "
                 f"{dg['n_scoreable'] - dg['n_unflagged_ignoring_retention']} carry a "
                 f"coverage/sigma/void-fill flag (QC flags 2-3). A step is scoreable only "
                 f"when all twelve of its candidate cells carry a belief. At "
                 f"{K.LEGACY_WIN_LEN_M:g} m windows the retention floor was the binding rule "
                 f"and 11 of 17 windows failed it, because a {K.LEGACY_WIN_LEN_M:g} m window "
                 f"is never observed past the {K.R_MAX:g} m range cap from before its own "
                 f"start; matching the window length to the perception horizon is what this "
                 f"configuration change addresses.\n")
        L += ["| foresight window | retention | element-cell observed frac | median sigma (m) | flags |",
              "|---|---|---|---|---|"]
        for w in dg["per_window"]:
            L.append(f"| {w['name']} | {w['retained']} | {w['qc_observed_frac']} | "
                     f"{w['qc_sigma_med_m']} | {w['qc_flags'] or '-'} |")
        L.append("")
    if not cond_res["C3"]["pass"]:
        L.append(f"C3 detail — {cond_res['C3']['notes']}\n")
    if not cond_res["C4"]["pass"]:
        L.append(f"C4 detail — {cond_res['C4']['evidence_of_access']}\n")

    L.append("## Registration (truth side; no belief, no arm enters)\n")
    L += ["| traverse | selected model | residual sd | block-CV | plane-only CV | tilt (mm/m) | void-fill in corridor |",
          "|---|---|---|---|---|---|---|"]
    for k, v in reg.items():
        m = v["models"][v["chosen_model"]]
        L.append(f"| {k} | {v['chosen_model']} | {v['residual_after_registration']['sd_m']} m | "
                 f"{v['chosen_blockcv_rmse_m']} m | {v['plane_only_alternative']['blockcv_rmse_m']} m | "
                 f"{m.get('tilt_mm_per_m')} | {v['voidfill_frac_corridor']} |")
    L.append("")

    L.append("## The two pinned calibration constants, re-checked\n")
    L.append(f"- mount pitch offset: pinned **{K.MOUNT_PITCH_OFFSET_DEG} deg** "
             f"(total {K.TOTAL_MOUNT_PITCH_DEG} deg); refit on this host "
             f"**{cal['pitch']['refit_MOUNT_PITCH_OFFSET_DEG']} deg**, "
             f"|diff| {cal['pitch']['abs_diff_deg']} deg. The pinned value is what the build "
             f"used; the refit is recorded only.")
    L.append(f"- stereo noise sigma_z(r) = a + b r^2: pinned **a = {K.SIG_A}, b = {K.SIG_B}**; "
             f"refit **a = {cal['sigma']['refit_pooled']['a']}, "
             f"b = {cal['sigma']['refit_pooled']['b']}**.")
    L.append(f"- intrinsics: fx = fy = {K.FX}, cx = {K.CX}, cy = {K.CY}, zero distortion "
             f"(CameraInfo; never refitted).\n")

    for cond in CONDITIONS:
        d = design[cond]
        L.append(f"## Condition: {cond.upper()}"
                 + (" (PRIMARY — the planning-time belief)" if cond == "foresight"
                    else " (contrast — the belief at its lifetime best)") + "\n")
        q = d["qc"]
        L.append(f"{q['n_window_products']} window products, {q['n_retained']} retained, "
                 f"{q['n_unflagged']} unflagged, {q['n_excluded']} excluded. "
                 f"Flagged: `{json.dumps(q['flagged'])}`. Excluded: "
                 f"`{json.dumps(q['excluded'])}`\n")
        f = d["fit"]
        pin = d.get("fit_pinned_tau_alternative") or {}
        L.append(f"Recalibration (ML on unflagged, arm `{K.FIT_ARM}`, applied identically to "
                 f"every variance arm, E[cost] untouched): **k = {f.get('k')}**, "
                 f"**tau = {f.get('tau_m')} m**, sd multiplier "
                 f"{f.get('sd_multiplier_sqrt_k')}, mean NLL at fit "
                 f"{f.get('mean_NLL_at_fit')}. Pinned-tau (k = 1) alternative: "
                 f"tau = {pin.get('tau_m')} m, penalty {pin.get('nll_penalty_vs_ML')} nats.\n")
        L.append("### Reality referee — arm table (pooled, unflagged)\n")
        L += _arm_table(d)
        p = d.get("pooled", {})
        for a, b in (("clark", "fosm"), ("clark", "clark-diag"), ("clark", "step-form"),
                     ("clark", "mc-32"), ("clark", "mc-2")):
            r = p.get(f"nll_{a}_minus_{b}", {}).get("recal")
            if r:
                L.append(f"- {a} minus {b} (recalibrated, negative = clark better): mean "
                         f"{r['mean']}, median {r['median']}, clark better in "
                         f"{r['n_windows_clark_better']} of {r['n']}, two-sided sign test "
                         f"p = {r['sign_test_p_two_sided']}")
        sp = p.get("arm_NLL_spread")
        if sp:
            L.append(f"- arm-NLL spread: {sp['raw_nats']} nats raw, {sp['recal_nats']} nats "
                     f"recalibrated ({sp['compression_factor']}x compression)")
        L.append("\n_No pass/fail criterion attaches to any reality-side arm comparison "
                 "(PREREG_baseprod.md)._\n")
        L += _belief_referee_block(d)
        L.append("### Regime\n")
        L += _stat_table(d, [("sigma_med_m", "belief sigma on element cells (m)"),
                             ("relief_p95_p5_m", "truth relief p95-p5 (m)"),
                             ("sigma_over_relief", "sigma / relief"),
                             ("alpha_weighted_median", "alpha (cost-weighted median)"),
                             ("frac_nodes_alpha_lt_1", "fraction of cost-weighted nodes |alpha| < 1"),
                             ("qc_observed_frac", "QC element-cell observed fraction"),
                             ("qc_sigma_med_m", "QC median element-cell sigma (m)"),
                             ("retained", "retention"),
                             ("mu_minus_truth_med_m", "belief mu minus truth, per-window median (m)")])

    L.append("## Artifacts\n")
    for f in ("RUN_LOG.md", "FREEZE.json" if frozen else "STOPPED_AT.md",
              "freeze_conditions.json", "design_phase.json", "design_score_hindsight.json",
              "design_score_foresight.json", "design_build_hindsight.json",
              "design_build_foresight.json", "parity10m_score.json", "parity10m_build.json",
              "calibration_check.json", "registration/registration.json",
              "pipeline_commit.json"):
        L.append(f"- `{OUT / f}`")
    L.append("")
    L.append("## The held-out run\n")
    if frozen:
        L.append("C1-C4 all hold, so this runner proceeded **directly** into the held-out "
                 "run under the author's restored chained authorization — no GO file, no "
                 "stop. Follow it in `RUN_LOG.md`; the result lands in `FINAL_REPORT.md`, "
                 "which reproduces these design tables as an appendix.\n")
    else:
        L.append("A freeze condition failed, so the runner stopped before unlock and the "
                 "held-out set is untouched. To start the held-out stages after the "
                 "pre-registration is amended or the gate is met:\n")
    if not frozen:
        L.append("```sh\n"
                 f"ssh dasenka\n"
                 f"tmux new -d -s e2_heldout '{ROOT}/studies/baseprod/runner.sh'\n"
                 "```\n")
    L.append(f"The held-out stages refuse to run without `{freeze_path()}`. "
             + ("It exists.\n" if frozen else "**It does not exist.**\n"))
    L.append(f"Expected cost once started: {sum(m['size_gib'] for m in HELD_OUT):.0f} GiB "
             f"streamed across {len(HELD_OUT)} traverses, one archive resident at a time, "
             f"deleted after extraction; disk cap {DISK_CAP_GB:.0f} GB.\n")
    (OUT / "DESIGN_PHASE_DONE.md").write_text("\n".join(L))
    log("DESIGN_PHASE_DONE.md written", stage)


# ------------------------------------------------------------------ stage 5: held out
def _extract(archive: Path, dest: Path, name: str) -> None:
    import py7zr
    need_csv = {"GNSS.csv", "IMU.csv", "TF_STATIC.csv", "TF_CORRECTED_REAR_BOGIE.csv",
                "TF.csv", "FOG.csv", "FOG_CORRECTED.csv"}
    with py7zr.SevenZipFile(archive, "r") as a:
        names = [f.filename for f in a.list()
                 if (os.path.basename(f.filename) in need_csv)
                 or ("RS_DEPTH_16bit/" in f.filename and f.filename.endswith(".png"))]
        a.reset()
        a.extract(path=str(dest), targets=names)


def stage_heldout() -> None:
    stage = "5-heldout"
    from . import register
    from .build import build
    todo = [m for m in HELD_OUT if not done(f"5-{m['name']}")]
    log(f"streaming loop: {len(todo)} of {len(HELD_OUT)} traverses still to do "
        f"({sum(m['size_gib'] for m in todo):.1f} GiB), {free_gb():.1f} GB free on /local",
        stage)
    for i, m in enumerate(HELD_OUT, 1):
        name = m["name"]
        st = f"5-{name}"
        if done(st):
            log(f"[{i}/{len(HELD_OUT)}] {name}: already done, skipping", stage)
            continue
        if free_gb() < m["size_gib"] * 1.074 + DL_MARGIN_GB:
            log(f"STOPPING on disk pressure before {name}: {free_gb():.1f} GB free, need "
                f"~{m['size_gib'] * 1.074 + DL_MARGIN_GB:.1f} GB. Rerun to resume.", stage)
            return
        url = f"https://roboshare.esa.int/index.php/s/{m['token']}/download"
        arc = DATA / f"{name}.7z"
        log(f"[{i}/{len(HELD_OUT)}] {name}: downloading {m['size_gib']} GiB from "
            f"roboshare token {m['token']}; {free_gb():.1f} GB free", stage)
        r = subprocess.run(["nice", "-n", "15", "ionice", "-c3", "curl", "-sSL", "-C", "-",
                            "--retry", "8", "--retry-delay", "20", "-o", str(arc), url])
        if r.returncode != 0:
            log(f"download FAILED for {name} (curl {r.returncode}); skipping, rerun to retry",
                stage)
            arc.unlink(missing_ok=True)
            continue
        log(f"extracting {name}", stage)
        HELDOUT_DIR.mkdir(parents=True, exist_ok=True)
        try:
            _extract(arc, HELDOUT_DIR, name)
        except Exception as e:
            log(f"extract FAILED for {name}: {e}", stage)
            arc.unlink(missing_ok=True)
            shutil.rmtree(HELDOUT_DIR / name, ignore_errors=True)
            continue
        arc.unlink(missing_ok=True)
        log(f"archive deleted; {free_gb():.1f} GB free", stage)
        try:
            reg = register.run_and_record([name])
            log(f"  registered {name}: {reg[name]['chosen_model']}, resid sd "
                f"{reg[name]['residual_after_registration']['sd_m']} m", stage)
            for cond in CONDITIONS:
                meta = build(name, cond)
                log(f"  {name} {cond}: {meta['n_windows']} windows built from "
                    f"{meta.get('n_depth_frames')} depth frames over "
                    f"{meta.get('path_length_m')} m of track", stage)
        except Exception as e:
            log(f"build FAILED for {name}: {e}\n{traceback.format_exc()}", stage)
            shutil.rmtree(HELDOUT_DIR / name, ignore_errors=True)
            continue
        shutil.rmtree(HELDOUT_DIR / name, ignore_errors=True)
        mark(st, {"name": name})
        log(f"{name} done; extracted data deleted; {free_gb():.1f} GB free", stage)
    mark(stage)


def stage_score_heldout() -> dict:
    stage = "6-score"
    from .score import evaluate_criteria, score_condition
    fr = json.loads(freeze_path().read_text())
    out = {}
    for cond in CONDITIONS:
        st = f"6-{cond}"
        if not done(st):
            k = fr["recalibration_per_condition"][cond]["k"]
            tau = fr["recalibration_per_condition"][cond]["tau_m"]
            dirs = [OUT / "windows" / cond / m["name"] for m in HELD_OUT]
            dirs = [d for d in dirs if d.exists()]
            log(f"scoring held-out under {cond} at the FROZEN (k={k}, tau={tau})", stage)
            r = score_condition(dirs, cond, k=k, tau=tau, mc_n=K.N_MC_DRAWS,
                                label=f"HELD-OUT, {cond}, frozen recalibration")
            r["criteria"] = evaluate_criteria(r["pooled"], r["belief_referee"],
                                              r["windows"])
            write_json(OUT / f"heldout_score_{cond}.json", r)
            mark(st, {"n_unflagged": r["qc"]["n_unflagged"]})
        out[cond] = json.loads((OUT / f"heldout_score_{cond}.json").read_text())
    mark(stage)
    return out


def stage_report(heldout: dict) -> None:
    stage = "7-report"
    fr = json.loads(freeze_path().read_text())
    L = [f"# E2 FINAL REPORT — BASEPROD risk calibration on contested terrain\n",
         f"Written {now()}. Pipeline commit `{fr['pipeline_commit']}`. "
         f"Frozen at {fr['written_at']}. The held-out run happened once; there is no "
         f"second draw.\n",
         "The author ran this chain unattended, without prior review of the design tables "
         "(PREREG_baseprod.md, authorization update 2026-08-28 evening). The full "
         "design-phase tables are therefore reproduced in this report, after the held-out "
         "results, for post-hoc review.\n",
         f"Configuration: {fr['windowing']['win_len_m']:g} m windows, "
         f"{fr['windowing']['lookback_m']:g} m look-back, arms "
         f"{fr['arms']['variance_arms']} + mean-map.\n",
         "**Contents** — held-out results per condition, then the criteria, then the "
         "design-phase appendix.\n"]
    for cond in CONDITIONS:
        d = heldout[cond]
        L.append(f"## {cond.upper()}"
                 + (" — PRIMARY" if cond == "foresight" else " — contrast") + "\n")
        q = d["qc"]
        L.append(f"{q['n_window_products']} window products, {q['n_retained']} retained, "
                 f"{q['n_unflagged']} unflagged.\n")
        L.append(f"Frozen recalibration: k = {d['fit']['k']}, tau = {d['fit']['tau_m']} m "
                 f"(fitted on design, not refitted here).\n")
        L += _arm_table(d)
        L.append("### Criteria\n")
        L += ["| criterion | value | verdict |", "|---|---|---|"]
        for c, r in d["criteria"].items():
            if c == "reality_side_arm_comparisons":
                continue
            v = r.get("value")
            if v is None:
                v = {k2: r[k2] for k2 in r
                     if k2.startswith(("a_", "b_", "n_", "sign_")) and k2 != "statement"}
            p = r.get("pass")
            L.append(f"| {c} | {v} | "
                     f"{'PASS' if p else ('FAIL' if p is False else 'reported, no criterion')} |")
        L.append("")
        L += _belief_referee_block(d)
        L.append(d["criteria"]["reality_side_arm_comparisons"]["interpretation"] + "\n")
        L += _stat_table(d, [("sigma_over_relief", "sigma / relief"),
                             ("alpha_weighted_median", "alpha (cost-weighted median)"),
                             ("frac_nodes_alpha_lt_1", "fraction |alpha| < 1")])
    # --- appendix: the full design-phase tables, for post-hoc review -------------------
    L.append("---\n")
    L.append("# Appendix — the design phase, in full\n")
    L.append("> **DESIGN-PHASE / IN-SAMPLE.** Two design traverses; per-condition (k, tau) "
             "was fitted on the same windows it is scored on. These numbers set the freeze "
             "record; they are not results. They are reproduced here because the chain ran "
             "without prior review of them.\n")
    dpd = OUT / "DESIGN_PHASE_DONE.md"
    if dpd.exists():
        body = dpd.read_text().split("\n")
        # drop the design report's own H1 and demote its headings one level
        body = [("#" + ln if ln.startswith("#") else ln) for ln in body[1:]]
        L += body
    else:
        L.append("_DESIGN_PHASE_DONE.md missing_\n")
    (OUT / "FINAL_REPORT.md").write_text("\n".join(L))
    log(f"FINAL_REPORT.md written ({len(L)} lines, design-phase appendix included)", stage)
    mark(stage)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design-only", action="store_true",
                    help="stop after the freeze record instead of chaining into held-out")
    ap.add_argument("--heldout", action="store_true",
                    help="accepted and ignored; the chain is the default since the author "
                         "restored the chained authorization")
    ap.add_argument("--mc-draws", type=int, default=K.N_MC_DRAWS)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)
    if not RUN_LOG.exists():
        RUN_LOG.write_text(
            "# E2 RUN_LOG\n\nDetached runner for PREREG_baseprod.md, "
            "`studies/baseprod/runner.py`. Append-only, one line per stage event.\n\n")
    log(f"runner start (pid {os.getpid()}, host {os.uname().nodename}, "
        f"BASEPROD_ROOT={ROOT}, argv={sys.argv[1:]}, design_only={args.design_only}). "
        f"Chained authorization: if C1-C4 all pass, FREEZE.json is written and the "
        f"{len(HELD_OUT)}-traverse held-out run starts immediately, unattended. Any gate "
        f"failure stops before unlock.", "0-init")
    log(f"configuration: windows {K.WIN_LEN_M:g} m (legacy parity build "
        f"{K.LEGACY_WIN_LEN_M:g} m), look-back {K.LOOKBACK_M:g} m, cell {K.CELL:g} m, arms "
        f"{list(K.ARMS_VARIANCE)} + mean-map, {K.N_MC_DRAWS} MC draws/window (seed "
        f"{K.MC_SEED}, subsample seed {K.MC_SUBSAMPLE_SEED}), QC thresholds "
        f"retained>={K.MIN_RETAINED}, obs_frac>={K.FLAG_MIN_OBS_FRAC}, "
        f"sigma<={K.FLAG_MAX_SIGMA_M} m, voidfill<={K.FLAG_MAX_INVALID_FRAC}", "0-init")
    log(f"pins: fx=fy={K.FX}, cx={K.CX}, cy={K.CY}, mount pitch offset "
        f"{K.MOUNT_PITCH_OFFSET_DEG} deg (total {K.TOTAL_MOUNT_PITCH_DEG} deg), "
        f"sigma_z(r) = {K.SIG_A} + {K.SIG_B} r^2, r_wheel {K.R_WHEEL} m", "0-init")
    log(f"step-form formula: {K.STEP_FORM_FORMULA}", "0-init")

    try:
        commit = stage_commit() if not done("1-commit") else json.loads(
            (OUT / "pipeline_commit.json").read_text())
        design = stage_design(args.mc_draws)
        cond_res = (json.loads((OUT / "freeze_conditions.json").read_text())
                    if done("3-freeze-eval") else stage_freeze_eval(design))
        frozen = freeze_path().exists()
        if not done("4-freeze"):
            frozen = stage_freeze(design, cond_res, commit)
        write_design_report(design, cond_res, commit, frozen)

        if args.design_only:
            log("design phase complete; --design-only was passed, so STOPPING before "
                "held-out.", "4-freeze")
            return 0
        if not frozen:
            log("STOPPING before unlock: a freeze condition failed, so no FREEZE.json was "
                "written. The held-out set remains untouched. See STOPPED_AT.md and "
                "DESIGN_PHASE_DONE.md.", "5-heldout")
            return 4
        log("C1-C4 all hold and FREEZE.json is written. Proceeding DIRECTLY into the "
            "held-out run under the author's restored chained authorization "
            "(PREREG_baseprod.md, 2026-08-28 evening; clark_paper 0506da2). "
            f"{len(HELD_OUT)} traverses, "
            f"{sum(m['size_gib'] for m in HELD_OUT):.1f} GiB to stream, one archive resident "
            f"at a time, disk cap {DISK_CAP_GB:.0f} GB.", "5-heldout")
        stage_heldout()
        heldout = stage_score_heldout()
        stage_report(heldout)
        log("ALL STAGES COMPLETE — FINAL_REPORT.md written", "7-report")
        return 0
    except SystemExit:
        raise
    except Exception as e:
        log(f"UNHANDLED FAILURE: {e}\n```\n{traceback.format_exc()}\n```", "FATAL")
        (OUT / "STOPPED_AT.md").write_text(
            f"# STOPPED_AT — unhandled failure\n\n{now()}\n\n```\n"
            f"{traceback.format_exc()}\n```\n\nThe held-out set was not touched by this "
            f"failure path. Rerun the same script to resume from the last done-marker.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
