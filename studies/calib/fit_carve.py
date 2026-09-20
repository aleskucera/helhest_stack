"""Re-fit the visibility-carve gates on the Odin dTOF bags.

`elevation_belief`'s carve defaults were ported from this project's spinning-LIDAR tuning,
where `max_range = 2.5 m` was fitted against 0.35 deg beams binned at 1.4 deg. The Odin sensor
is a 137 x 88 deg dToF returning ground from 0.36 m, so the gates are structurally right and
the numbers are not evidence.

Two measurements, both label-free:

  COST     cells the carve removes that a no-carve run keeps. Binned by the range the cell was
           observed at, because that is what `max_range` trades against -- the original
           over-carve took 37% of a map with 72% of the damage past 8 m.
  BENEFIT  the fusion residual the belief already accumulates. `resid_sq / count` is how far
           the map stood from each incoming measurement; carving stale geometry should shrink
           it, and a carve that only erodes will not.

The grid is fixed rather than rolling, deliberately: comparing the same cells between runs
isolates the carve from the window motion.

Points are height-cropped about the sensor first, as the deployed pipeline does (`z_max`,
_pipeline_common.py:186). Without it these indoor bags put the CEILING in the map -- measured
on ostrich4, 82% of cells sat 2 m or more above the floor and the median cell was at 3.57 m,
because keep-the-highest fusion takes the highest return in a column and upward rays reach the
roof. Any erosion figure computed against that denominator is a statement about ceiling.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import warp as wp

from calib import bagio
from elevation_belief import DriftRates
from elevation_belief import ElevationBelief
from elevation_belief import NoiseModel

CELL = 0.08  # the node's `resolution` default
# Fitted to these bags in studies/calib/RESULTS.md section 3: sigma ~ 2.1 cm at 0.8 m rising to
# ~18 cm at 5.8 m, closer to linear in range than to the quadratic such sensors usually get.
DTOF_NOISE = NoiseModel("linear", a=0.027, b=0.023)
# Height crop about the sensor [m]. The mount sits ~0.45 m above the floor
# (odin_elevation.params.yaml), so +1.0 keeps walls and torsos to ~1.45 m above ground while
# excluding the roof; -1.5 keeps any drop-off the robot could drive into.
Z_ABOVE = 1.0
Z_BELOW = 1.5


def crop(frame: bagio.Frame, z_above: float, z_below: float) -> np.ndarray:
    """Drop returns the wheels can never contact, before they reach the belief."""
    z = frame.points[:, 2]
    keep = (z < frame.sensor_xyz[2] + z_above) & (z > frame.sensor_xyz[2] - z_below)
    return frame.points[keep]


@dataclass
class CarveGates:
    max_range: float = 2.5
    margin: float = 0.05
    end_gap: float = 0.10
    persist: int = 8


def _build(odom: bagio.Odometry, pad: float) -> tuple[ElevationBelief, tuple[float, float]]:
    """A grid covering the whole traverse plus `pad`, so no cell ever scrolls out."""
    xmin = float(odom.xyz[:, 0].min() - pad)
    ymin = float(odom.xyz[:, 1].min() - pad)
    xmax = float(odom.xyz[:, 0].max() + pad)
    ymax = float(odom.xyz[:, 1].max() + pad)
    # Odin's pose is on-device SLAM, not the design-site handheld rig the shipped
    # rates were fitted to; see studies/calib/fit_drift.py and RESULTS.md section 4.
    belief = ElevationBelief(
        (xmin, xmax, ymin, ymax), CELL, noise=DTOF_NOISE, rates=DriftRates.odin_slam()
    )
    return belief, (xmin, ymin)


def run_bag(
    odom: bagio.Odometry,
    frames: list[bagio.Frame],
    gates: CarveGates | None,
    pad: float,
    z_above: float = Z_ABOVE,
    z_below: float = Z_BELOW,
) -> dict:
    """One pass over a bag. `gates=None` is the no-carve baseline."""
    belief, _ = _build(odom, pad)
    prev_t = frames[0].t
    for f in frames:
        belief.motion_update(
            max(f.t - prev_t, 0.0), (float(f.sensor_xyz[0]), float(f.sensor_xyz[1]))
        )
        prev_t = f.t
        pts = crop(f, z_above, z_below)
        if len(pts) == 0:
            continue
        if gates is not None:
            belief.carve(
                pts,
                f.sensor_xyz,
                max_range=gates.max_range,
                margin=gates.margin,
                end_gap=gates.end_gap,
                persist=gates.persist,
            )
        belief.measure_scan(pts, f.sensor_xyz, stamp=f.t)

    valid = belief.valid.numpy() > 0
    count = belief.n_upd.numpy().astype(np.float64)
    resid_sq = belief.resid_sq.numpy().astype(np.float64)
    rng = belief.range_sum.numpy().astype(np.float64) / np.maximum(count, 1.0)
    fused = valid & (count > 1)
    return {
        "valid": valid,
        "mean_range": rng,
        "n_valid": int(valid.sum()),
        "resid_rms": float(np.sqrt(resid_sq[fused].sum() / max(count[fused].sum(), 1.0))),
        "n_fused": int(fused.sum()),
    }


def _erosion(base: dict, run: dict) -> dict:
    """What the carve removed relative to the baseline, overall and by observation range."""
    lost = base["valid"] & ~run["valid"]
    n_base = max(base["n_valid"], 1)
    out = {
        "lost_frac": float(lost.sum() / n_base),
        "resid_rms": run["resid_rms"],
        "resid_delta": run["resid_rms"] - base["resid_rms"],
        "n_valid": run["n_valid"],
    }
    rng = base["mean_range"]
    for lo, hi in ((0.0, 2.0), (2.0, 5.0), (5.0, 8.0), (8.0, 1e9)):
        band = base["valid"] & (rng >= lo) & (rng < hi)
        n = max(int(band.sum()), 1)
        key = f"lost_{lo:g}_{hi:g}m" if hi < 1e9 else f"lost_over_{lo:g}m"
        out[key] = float((lost & band).sum() / n)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bags", nargs="+")
    ap.add_argument("--root", default="bags")
    ap.add_argument("--pad", type=float, default=16.0)
    ap.add_argument("--out", default="studies/out/calib")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cache = {}
    for b in args.bags:
        p = bagio.bag_path(b, args.root)
        odom = bagio.read_odometry(p)
        cache[b] = (odom, bagio.read_frames(p, odom))

    sweeps = {
        "max_range": [1.0, 2.5, 5.0, 10.0, 15.0],
        "persist": [2, 4, 8, 16],
        "margin": [0.02, 0.05, 0.10, 0.20],
    }
    report: dict = {"cell": CELL, "defaults": vars(CarveGates()), "baseline": {}, "sweeps": {}}

    baselines = {}
    for b in args.bags:
        odom, frames = cache[b]
        baselines[b] = run_bag(odom, frames, None, args.pad)
        report["baseline"][b] = {
            "n_valid": baselines[b]["n_valid"],
            "resid_rms": baselines[b]["resid_rms"],
        }
        print(
            f"baseline {b:11s} valid {baselines[b]['n_valid']:7d}  resid_rms {baselines[b]['resid_rms']:.4f} m"
        )

    for name, values in sweeps.items():
        report["sweeps"][name] = {}
        print(f"\n--- {name} " + "-" * 60)
        print(
            f"{'value':>8s} {'lost%':>7s} {'<2m':>7s} {'2-5m':>7s} {'5-8m':>7s} {'>8m':>7s} "
            f"{'resid_rms':>10s} {'d_resid':>9s}"
        )
        for v in values:
            gates = CarveGates(**{name: v})
            rows = []
            for b in args.bags:
                odom, frames = cache[b]
                rows.append(_erosion(baselines[b], run_bag(odom, frames, gates, args.pad)))
            agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            report["sweeps"][name][str(v)] = agg
            print(
                f"{v:>8} {agg['lost_frac']*100:6.2f}% {agg['lost_0_2m']*100:6.2f}% "
                f"{agg['lost_2_5m']*100:6.2f}% {agg['lost_5_8m']*100:6.2f}% "
                f"{agg['lost_over_8m']*100:6.2f}% {agg['resid_rms']:10.4f} {agg['resid_delta']:+9.4f}"
            )

    dst = os.path.join(args.out, "fit_carve.json")
    with open(dst, "w") as fh:
        json.dump(report, fh, indent=1)
    print("\n->", dst)


if __name__ == "__main__":
    main()
