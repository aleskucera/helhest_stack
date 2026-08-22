"""Window runner v2 (the amended prereg): their localization, their mapping.

Per window: anchor = the window's FIRST GT pose (the only place truth enters);
within-window motion = VILENS SLAM *relative* increments, whose real drift is the
localization error under study; belief = the Fankhauser RA-L 2018 model
(elevation_belief.py) with the frozen design-site drift rates. Nothing of our
own localization or mapping remains in the loop.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .elevation_belief import ElevationBelief
from .loaders import E_BODY_LIDAR
from .loaders import GtTrajectory
from .loaders import HesaiScanArchive

CELL = 0.10
WINDOW_S = 20.0
MARGIN_M = 45.0
MAX_RANGE_M = 40.0


def _homogeneous(R: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    return T


def run_sequence(seq_dir: Path, out_dir: Path, *, window_s: float = WINDOW_S) -> None:
    arch = HesaiScanArchive(str(seq_dir / "raw" / "lidar-clouds.zip"))
    gt = GtTrajectory(str(seq_dir / "trajectory" / "gt-tum.txt"))
    vil = GtTrajectory(str(seq_dir / "trajectory" / "vilens-slam-tum.txt"))
    out_dir.mkdir(parents=True, exist_ok=True)
    E = _homogeneous(E_BODY_LIDAR)

    t_lo = max(arch.stamps[0], gt.t[0], vil.t[0])
    t_hi = min(arch.stamps[-1], gt.t[-1], vil.t[-1])
    starts = np.arange(t_lo, t_hi - window_s * 0.5, window_s)

    drifts = []
    for w, t0 in enumerate(starts):
        t1 = min(t0 + window_s, t_hi)
        idx = np.where((arch.stamps >= t0) & (arch.stamps < t1))[0]
        if len(idx) < 5:
            continue

        ts0 = float(arch.stamps[idx[0]])
        W_T_B0 = gt.pose_at(ts0)  # the anchor: truth enters here and only here
        V0_inv = np.linalg.inv(vil.pose_at(ts0))

        # grid bounds from the GT path (clutter control, as in v1)
        path = np.array([gt.pose_at(float(arch.stamps[i]))[:3, 3] for i in idx])
        bounds = (
            path[:, 0].min() - MARGIN_M,
            path[:, 0].max() + MARGIN_M,
            path[:, 1].min() - MARGIN_M,
            path[:, 1].max() + MARGIN_M,
        )
        belief = ElevationBelief(bounds, CELL)

        our_poses, gt_poses = [], []
        t_prev = ts0
        for si in idx:
            ts = float(arch.stamps[si])
            scan = arch.read(int(si))
            # VILENS relative motion since the anchor, re-rooted at the GT anchor
            W_T_B = W_T_B0 @ (V0_inv @ vil.pose_at(ts))
            W_T_L = W_T_B @ E
            rng = np.linalg.norm(scan.xyz[:, :2], axis=1)
            keep = rng < MAX_RANGE_M
            pts = scan.xyz[keep] @ W_T_L[:3, :3].T + W_T_L[:3, 3]

            belief.motion_update(ts - t_prev, W_T_B[:2, 3])
            belief.measure_scan(pts.astype(np.float64), W_T_L[:3, 3])
            t_prev = ts
            our_poses.append(W_T_B)
            gt_poses.append(gt.pose_at(ts))

        maps = belief.readout()
        end_err = our_poses[-1][:3, 3] - gt_poses[-1][:3, 3]
        drift_xy = float(np.linalg.norm(end_err[:2]))
        drifts.append(drift_xy)

        np.savez_compressed(
            out_dir / f"window_{w:03d}.npz",
            mu=maps["mu"].astype(np.float32),
            sigma=maps["sigma"].astype(np.float32),
            raw_h=maps["raw_h"].astype(np.float32),
            raw_sd=maps["raw_sd"].astype(np.float32),
            meas_sd=maps["meas_sd"].astype(np.float32),
            count=maps["count"],
            xmin=bounds[0],
            ymin=bounds[2],
            cell=CELL,
            t0=t0,
            t1=t1,
            sx=maps["sx"],
            sy=maps["sy"],
            our_poses=np.stack(our_poses),
            gt_poses=np.stack(gt_poses),
        )
        print(
            f"window {w:03d}: {len(idx)} scans, vilens end drift {drift_xy*100:6.1f} cm, "
            f"fusion kernel sd ({maps['sx']*100:.0f}, {maps['sy']*100:.0f}) cm"
        )
    d = np.array(drifts)
    print(f"\n{len(d)} windows; vilens end-drift median {np.median(d)*100:.1f} cm, "
          f"p90 {np.percentile(d, 90)*100:.1f} cm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seq_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--window", type=float, default=WINDOW_S)
    args = ap.parse_args()
    run_sequence(Path(args.seq_dir), Path(args.out_dir), window_s=args.window)
