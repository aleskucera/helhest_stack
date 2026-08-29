"""Per-sequence end-window drift medians for the E1 windows.

The paper states the SLAM's end-window drift medians on the held-out sites
(the localization error under study; window_runner2 anchors each window at
its first GT pose and propagates VILENS relative motion, so the end-window
XY offset between the propagated track and GT is the drift accumulated
inside that window). This script pins those numbers to an artifact.

Input: the window products (window_*.npz: our_poses, gt_poses) under the
data root (default ~/data/oxford_spires/out). Reads only; writes
studies/out/risk_calibration_heldout/drift_medians.json.

Run:  python -m studies.spires.drift_medians
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "out" / "risk_calibration_heldout"
HELD_OUT_PREFIX = "virgin-"


def end_window_drifts_m(seq_dir: Path) -> list[float]:
    out = []
    for f in sorted(seq_dir.glob("window_*.npz")):
        d = np.load(f)
        po, pg = d["our_poses"][:, :3, 3], d["gt_poses"][:, :3, 3]
        n = min(len(po), len(pg))
        out.append(float(np.linalg.norm(po[n - 1, :2] - pg[n - 1, :2])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(Path.home() / "data/oxford_spires/out"))
    args = ap.parse_args()
    root = Path(args.data_root)

    result = {"data_root": str(root), "definition":
              "per window: XY distance between the final our_poses and gt_poses "
              "entries (drift accumulated from the first-GT-pose anchor); "
              "per sequence: median over its windows", "sequences": {}}
    for seq_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        drifts = end_window_drifts_m(seq_dir)
        if not drifts:
            continue
        result["sequences"][seq_dir.name] = {
            "held_out": seq_dir.name.startswith(HELD_OUT_PREFIX),
            "n_windows": len(drifts),
            "median_end_window_drift_m": round(float(np.median(drifts)), 4),
            "p90_end_window_drift_m": round(float(np.percentile(drifts, 90)), 4),
        }
    ho = [v["median_end_window_drift_m"] for v in result["sequences"].values()
          if v["held_out"]]
    result["held_out_median_range_m"] = [min(ho), max(ho)]
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "drift_medians.json"
    out.write_text(json.dumps(result, indent=1))
    for name, v in result["sequences"].items():
        print(f"{name:45s} {'HELD-OUT' if v['held_out'] else 'design  '} "
              f"n={v['n_windows']:3d} median {100*v['median_end_window_drift_m']:6.1f} cm")
    print("held-out median range [m]:", result["held_out_median_range_m"])
    print("wrote", out)


if __name__ == "__main__":
    main()
