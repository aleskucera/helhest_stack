"""Phase 4: run every policy on every seed, and report it honestly.

    .venv/bin/python -m studies.bench.run_bench [--seeds N] [--out DIR]

PAIRED design. Every policy sees the SAME seeds -- same gap position, same approach angle,
same terrain -- so each seed is a matched comparison and the noise from scenario variation
cancels. With the handful of seeds a closed-loop study affords, unpaired means would be
dominated by which gap positions happened to land in which arm.

REPORTED, not asserted:
  - per-seed times, so the spread is visible rather than hidden behind a mean
  - the ORACLE (full map from frame 0) as the ceiling: no sensing policy can beat it, and the
    fraction of its headroom a policy recovers is the only scale-free way to say how good it is
  - what each policy's looks ACTUALLY revealed (gap vs decoy), which is the behavioural check
    that the aggregate numbers are measuring what they claim
  - failures to reach, separately -- a policy that is fast because it gave up is not fast

Section 6 asks for time-to-goal AT EQUAL SAFETY, so contacts are reported alongside; a time
win bought with contacts is not a win.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from . import loop as L
from . import policies as P
from . import world as W

OUT = Path(__file__).resolve().parents[2] / "studies" / "out" / "bench"
ARMS = ("none", "sigma", "entropy", "cvar", "attribution", "disagreement")


def run_all(n_seeds: int, max_frames: int, variant: str = "gap") -> list[dict]:
    build = W.build if variant == "gap" else W.build_corridor
    rows: list[dict] = []
    for seed in range(n_seeds):
        bw = build(seed)
        rec = {"seed": seed, "gap_y": bw.gap_y, "approach_yaw": bw.approach_yaw}
        orc = L.run(bw, policy=None, omniscient=True, max_frames=max_frames)
        rec["oracle"] = _score(orc)
        for arm in ARMS:
            tr = L.run(bw, policy=P.POLICIES[arm], max_frames=max_frames)
            rec[arm] = _score(tr)
        rows.append(rec)
        print(
            f"  seed {seed:>2} gap_y {bw.gap_y:+.1f} | oracle {rec['oracle']['time']:>4} | "
            + " ".join(
                f"{a[:4]} {rec[a]['time']:>4}{'' if rec[a]['reached'] else '*'}" for a in ARMS
            ),
            flush=True,
        )
    return rows


def _score(tr: L.Trace) -> dict:
    return {
        "time": tr.total_time,
        "drive_frames": tr.frames,
        "look_frames": tr.look_frames,
        "n_looks": tr.n_looks,
        "reached": bool(tr.reached),
        "contacts": tr.contacts,
        "path_len": tr.path_len,
        "look_at_gap": tr.look_at_gap,
        "look_at_decoy": tr.look_at_decoy,
        "gap_known_frame": tr.gap_known_frame,
    }


def report(rows: list[dict]) -> dict:
    orc = np.array([r["oracle"]["time"] for r in rows], float)
    orc_ok = np.array([r["oracle"]["reached"] for r in rows])
    print(
        f"\noracle: reached {orc_ok.sum()}/{len(orc_ok)}, median time {np.median(orc):.0f} frames"
    )
    if not orc_ok.all():
        print("  WARNING: the oracle failed somewhere -- headroom is not measurable on those seeds")

    print(
        f"\n{'policy':<13}{'reached':>9}{'median t':>10}{'vs oracle':>11}{'headroom rec':>14}"
        f"{'looks':>7}{'@gap':>6}{'@decoy':>8}{'contacts':>10}"
    )
    summary = {}
    base = np.array([r["none"]["time"] for r in rows], float)
    for arm in ARMS:
        t = np.array([r[arm]["time"] for r in rows], float)
        ok = np.array([r[arm]["reached"] for r in rows])
        # Fraction of the null-to-oracle gap this policy recovers, per seed then aggregated.
        # Seeds where the null already matched the oracle carry no information and are dropped.
        gap = base - orc
        useful = gap > 1e-9
        rec = np.median((base[useful] - t[useful]) / gap[useful]) if useful.any() else np.nan
        looks = np.mean([r[arm]["n_looks"] for r in rows])
        at_gap = np.mean([r[arm]["look_at_gap"] for r in rows])
        at_dec = np.mean([r[arm]["look_at_decoy"] for r in rows])
        con = np.mean([r[arm]["contacts"] for r in rows])
        print(
            f"{arm:<13}{ok.sum():>4}/{len(ok):<4}{np.median(t):>10.0f}"
            f"{np.median(t) / np.median(orc):>10.2f}x{rec:>13.0%}{looks:>7.1f}{at_gap:>6.1f}"
            f"{at_dec:>8.1f}{con:>10.1f}"
        )
        summary[arm] = {
            "reached": int(ok.sum()),
            "median_time": float(np.median(t)),
            "headroom_recovered": float(rec),
            "mean_looks": float(looks),
            "mean_looks_at_gap": float(at_gap),
            "mean_looks_at_decoy": float(at_dec),
            "mean_contacts": float(con),
            "times": t.tolist(),
        }

    # BY REGIME. The scenario is bimodal by construction: with the gap on the far side the
    # null baseline searches the wrong way first and loses ~225 frames; on the near side it
    # guesses right and there is almost nothing to recover. Pooling the two hides the only
    # interesting structure -- sensing pays exactly when the default would have guessed wrong,
    # and costs its budget when it would have guessed right. A single median would average a
    # real win against a real loss and report neither.
    hard = np.array([r["seed"] % 2 == 0 for r in rows])
    print(
        f"\n{'policy':<13}{'hard: median t':>16}{'easy: median t':>16}   (hard = gap on the far side)"
    )
    for arm in ARMS:
        t = np.array([r[arm]["time"] for r in rows], float)
        print(f"{arm:<13}{np.median(t[hard]):>16.0f}{np.median(t[~hard]):>16.0f}")
        summary[arm]["median_hard"] = float(np.median(t[hard]))
        summary[arm]["median_easy"] = float(np.median(t[~hard]))
    summary["oracle_hard"] = float(np.median(orc[hard]))
    summary["oracle_easy"] = float(np.median(orc[~hard]))
    print(f"{'oracle':<13}{np.median(orc[hard]):>16.0f}{np.median(orc[~hard]):>16.0f}")

    # Paired comparison: attribution against each other arm, seed by seed.
    print("\npaired vs attribution (per-seed time difference, negative = attribution faster)")
    a = np.array([r["attribution"]["time"] for r in rows], float)
    for arm in ARMS:
        if arm == "attribution":
            continue
        t = np.array([r[arm]["time"] for r in rows], float)
        d = a - t
        wins = int((d < 0).sum())
        print(
            f"  vs {arm:<12} median {np.median(d):>+7.0f} frames, "
            f"attribution faster on {wins}/{len(d)} seeds"
        )
        summary[f"paired_vs_{arm}"] = {"median_delta": float(np.median(d)), "wins": wins}
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--max-frames", type=int, default=L.MAX_FRAMES)
    ap.add_argument(
        "--variant",
        choices=("gap", "corridor"),
        default="gap",
        help="gap = critical feature is an APERTURE (entropy and attribution coincide); "
        "corridor = critical feature is OPAQUE (they should diverge)",
    )
    args = ap.parse_args()

    wp.init()
    OUT.mkdir(parents=True, exist_ok=True)
    print(
        f"section-6 benchmark [{args.variant}]: {args.seeds} seeds x {len(ARMS)} policies + oracle"
    )
    rows = run_all(args.seeds, args.max_frames, args.variant)
    summary = report(rows)
    out = OUT / f"results_{args.variant}.json"
    out.write_text(
        json.dumps({"variant": args.variant, "rows": rows, "summary": summary}, indent=2)
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
