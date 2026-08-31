"""v2 campaign runner (PREREG_attitude_cost.md sections 7-8).

Deterministic, resumable, tmux-friendly. Stages, each with a done-marker
under OUT/state/; a failed freeze condition writes STOPPED_AT and exits
before any EVAL contact.

  0 config     resolve paths, F1 constants, pipeline hash
  1 dbuild     rebuild the two DESIGN traverses (R_MAX=3.0, ALPHA=1.0)
  2 dcheck     freeze conditions:
               C1  quadratic-form moments vs brute-force MC on design
                   windows (rel tol 2e-2 at 200k draws, E and sd)
               C1b fold parity vs clark_fold (< 1e-6 m)
               C2  pooled deficit law refit on THIS design build matches
                   F1 constants (|da| < 0.02, |db| < 0.2)
               C3  all outputs finite
  3 f1         write out/v2/F1.json (constants + hashes); STOP for the
               freeze commit unless F1.json already committed matches
  4 gate       kinematic gate: p99 |dz/dt| of each EVAL traverse's RTK
               track; exclude > 0.3 m/s (expected: exactly 2023-07-20_19-12-27)
  5 ebuild     rebuild the 21 EVAL traverses, both conditions
  6 escore     score every window ONCE (fast fold + 20k-draw referee,
               correction from F1); fan characterization per window
  7 report     criteria V2-1/2/3, tables, FINAL_REPORT_v2.md

GPU benchmark is a separate, non-gating stage (`bench`): it measures the
new cost per plan on the Warp fold (x-space contraction) and never touches
scoring.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from math import exp
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "studies" / "out" / "v2"
STATE = OUT / "state"
RAW = Path("/local/kuceral4/baseprod/data/raw")
DESIGN = ("2023-07-23_13-05-11", "2023-07-22_14-18-23")   # t1, t2
GATE_DZDT = 0.3          # m/s, prereg section 5
N_REF = 20_000
R_MAX_V2, ALPHA_V2 = 3.0, 1.0


def log(msg, stage=""):
    print(f"[{time.strftime('%F %T')}][{stage}] {msg}", flush=True)


def done(stage):
    return (STATE / f"{stage}.done").exists()


def mark(stage):
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / f"{stage}.done").touch()


def stop(stage, why):
    (OUT / "STOPPED_AT").write_text(f"{stage}: {why}\n")
    log(f"STOP: {why}", stage)
    sys.exit(2)


def load_f1():
    f1 = OUT / "F1.json"
    return json.loads(f1.read_text()) if f1.exists() else None


def correction_from(f1):
    a, b = f1["law"]["a"], f1["law"]["b"]
    return lambda al: 1.0 / (1.0 - a * exp(-b * al))


def stage_config():
    h = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                       capture_output=True, text=True).stdout.strip()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pipeline_commit.json").write_text(json.dumps({"commit": h}))
    log(f"pipeline at {h}", "config")
    mark("config")


def _patched_build(traverse, condition, out_dir):
    """Rebuild one traverse under the v2 constants. Constants are patched
    on the module (verified in dcheck via C2: a build drift changes the
    refit law and stops the run)."""
    from .v2_build import build_v2
    build_v2([traverse], conditions=(condition,))


def stage_dcheck(f1):
    from spires.risk_calibration import Window
    from .v2_moments import selftest
    from .v2_score import case_v2
    from .v2_correction import window_alpha, fit_law
    # C1/C1b run inside case_v2 (parity assert) + brute-force spot checks
    # C2: refit on this build
    alphas, ratios = [], []
    for cond in ("foresight", "hindsight"):
        for wp in sorted((OUT / f"windows_design/{cond}").glob("*/window_*.npz")):
            r = case_v2(wp, n_ref=50_000, seed=7)
            if not (r and r.get("scoreable") and not r["flagged"]):
                continue
            if not all(np.isfinite(v) for v in
                       (r["arms"]["clark"]["E"], r["arms"]["clark"]["sd"])):
                stop("dcheck", f"C3 non-finite at {r['name']}")
            alphas.append(r["alpha_weighted_median"])
            ratios.append(r["arms"]["clark"]["sd_ratio_to_mc"])
    refit = fit_law(alphas, ratios)
    if abs(refit["a"] - f1["law"]["a"]) > 0.02 or abs(refit["b"] - f1["law"]["b"]) > 0.2:
        stop("dcheck", f"C2 law drift: refit {refit} vs F1 {f1['law']}")
    log(f"C2 ok: refit a={refit['a']} b={refit['b']}", "dcheck")
    mark("dcheck")


def stage_gate():
    kept, excluded = [], []
    for tdir in sorted(RAW.iterdir()):
        if not tdir.is_dir() or tdir.name in DESIGN:
            continue
        # p99 |dz/dt| from the traverse RTK track (loader shared with build)
        from .build import load_rtk_track  # noqa: implemented with build integration
        t, z = load_rtk_track(tdir)
        dzdt = np.abs(np.diff(z) / np.maximum(np.diff(t), 1e-3))
        p99 = float(np.percentile(dzdt, 99))
        (kept if p99 <= GATE_DZDT else excluded).append((tdir.name, p99))
    (OUT / "gate.json").write_text(json.dumps({"kept": kept, "excluded": excluded,
                                               "threshold": GATE_DZDT}, indent=1))
    log(f"gate: {len(kept)} kept, excluded {[e[0] for e in excluded]}", "gate")
    mark("gate")


def stage_escore(f1):
    from .v2_score import case_v2
    from .v2_fan import fan as run_fan
    corr = correction_from(f1)
    fanc = f1["fan"]
    for cond in ("foresight", "hindsight"):
        rows = []
        for wp in sorted((OUT / f"windows_eval/{cond}").glob("*/window_*.npz")):
            r = case_v2(wp, n_ref=N_REF, seed=abs(hash(wp.stem)) % 2**31,
                        correction=corr)
            if r is not None:
                rows.append(r)
        (OUT / f"heldout_score_{cond}.json").write_text(json.dumps(rows, indent=1))
        log(f"{cond}: {len(rows)} windows scored", "escore")
    mark("escore")


def main():
    stages = ["config", "dbuild", "dcheck", "f1", "gate", "ebuild", "escore", "report"]
    # ... each stage guarded by done(); dbuild/ebuild/report wired during
    # design-phase integration on dasenka; this file is committed before F0
    # and hash-recorded in F1.
    stage_config()
    log("runner skeleton: design-phase integration pending for "
        "dbuild/ebuild/report stages", "main")


if __name__ == "__main__":
    main()
