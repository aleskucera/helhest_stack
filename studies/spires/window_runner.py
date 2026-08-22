"""Build per-window beliefs from a Spires sequence with OUR odometry.

Per the FROZEN prereg (clark_paper/PREREG_oxford_spires.md): evaluation windows of
~20 s, each anchored to the ground-truth frame by its FIRST GT pose only -- the
ICP odometry drift accumulated inside the window stays inside the belief error,
which is exactly what the per-cell sigma must cover. Whole windows are never
re-fitted to the truth.

Belief per window: mu = per-cell MAX layer, sigma = per-cell std (Welford-grade
float64 accumulation in the heightmap builder), count. Saved as one .npz per
window plus our-vs-GT pose tracks for drift diagnostics.

Design-phase v1 scope (logged): no deskew and no gyro prior (walking-pace
handheld; the quicklook placed scans at 15 cm ground-band median even with
nearest-pose placement), scan-to-submap ICP with a rolling window of posed
scans as the target.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

from helhest.perception import IcpAligner
from helhest.perception import IcpConfig
from helhest.perception.heightmap.builder import HeightMapBuilder

from .loaders import E_BODY_LIDAR
from .loaders import GtTrajectory
from .loaders import HesaiScanArchive

CELL = 0.10
WINDOW_S = 20.0
SUBMAP_SCANS = 8
MARGIN_M = 45.0  # grid half-extent beyond the window's GT path bbox


@dataclass
class WindowResult:
    t0: float
    t1: float
    n_scans: int
    drift_xy_m: float  # our final in-window pose vs GT, translation
    drift_yaw_deg: float


def _homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _yaw_of(T: np.ndarray) -> float:
    return float(np.arctan2(T[1, 0], T[0, 0]))


def run_sequence(
    seq_dir: Path,
    out_dir: Path,
    *,
    window_s: float = WINDOW_S,
    device: str | None = None,
) -> list[WindowResult]:
    arch = HesaiScanArchive(str(seq_dir / "raw" / "lidar-clouds.zip"))
    traj = GtTrajectory(str(seq_dir / "trajectory" / "gt-tum.txt"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # target submap can hold several scans; voxel-thin it inside the aligner
    aligner = IcpAligner(
        IcpConfig(
            max_iters=12,
            max_correspondence_dist_m=0.5,
            voxel_size_m=0.12,
            voxel_target=True,
            max_points=600_000,
        ),
        device=device,
    )
    dev = aligner.device

    # windows over the overlap of scan stamps and trajectory time
    t_lo = max(arch.stamps[0], traj.t[0])
    t_hi = min(arch.stamps[-1], traj.t[-1])
    starts = np.arange(t_lo, t_hi - window_s * 0.5, window_s)

    results: list[WindowResult] = []
    E = _homogeneous(E_BODY_LIDAR, np.zeros(3))

    for w, t0 in enumerate(starts):
        t1 = min(t0 + window_s, t_hi)
        idx = np.where((arch.stamps >= t0) & (arch.stamps < t1))[0]
        if len(idx) < 5:
            continue

        # anchor: world_T_lidar at the window's first scan, from GT (the ONLY
        # place the truth enters; everything after is our odometry)
        W_T_B0 = traj.pose_at(float(arch.stamps[idx[0]]))
        W_T_L0 = W_T_B0 @ E

        L0_T_Li = np.eye(4)  # our in-window odometry chain (lidar frame)
        submap: list[wp.array] = []  # posed scans, window-local lidar0 frame
        world_pts: list[np.ndarray] = []
        our_poses: list[np.ndarray] = []
        gt_poses: list[np.ndarray] = []

        for k, si in enumerate(idx):
            scan = arch.read(int(si))
            src = wp.array(scan.xyz, dtype=wp.vec3, device=dev)
            if k > 0:
                target = _concat_device(submap, dev)
                res = aligner.align(src, target, init_pose=L0_T_Li)
                L0_T_Li = res.pose
            posed = _transform_np(scan.xyz, L0_T_Li)
            posed_wp = wp.array(posed, dtype=wp.vec3, device=dev)
            submap.append(posed_wp)
            if len(submap) > SUBMAP_SCANS:
                submap.pop(0)
            world_pts.append(_transform_np(posed, W_T_L0))
            our_poses.append(W_T_L0 @ L0_T_Li)
            gt_poses.append(traj.pose_at(float(arch.stamps[si])) @ E)

        P = np.concatenate(world_pts).astype(np.float32)
        # grid bounds from the GT path, not from the points: keeps far clutter out
        path = np.array([p[:3, 3] for p in gt_poses])
        xmin, ymin = path[:, 0].min() - MARGIN_M, path[:, 1].min() - MARGIN_M
        xmax, ymax = path[:, 0].max() + MARGIN_M, path[:, 1].max() + MARGIN_M
        builder = HeightMapBuilder(CELL, (xmin, xmax, ymin, ymax), device=dev)
        layers = builder.build(P)
        maps = layers.to_numpy()

        ours, gts = our_poses[-1], gt_poses[-1]
        drift_xy = float(np.linalg.norm((ours[:3, 3] - gts[:3, 3])[:2]))
        drift_yaw = float(np.degrees(abs(_yaw_of(ours) - _yaw_of(gts))))
        r = WindowResult(float(t0), float(t1), len(idx), drift_xy, drift_yaw)
        results.append(r)

        np.savez_compressed(
            out_dir / f"window_{w:03d}.npz",
            mu=maps["max"],
            sigma=maps["std"],
            count=maps["count"],
            mean=maps["mean"],
            xmin=xmin,
            ymin=ymin,
            cell=CELL,
            t0=t0,
            t1=t1,
            our_poses=np.stack(our_poses),
            gt_poses=np.stack(gt_poses),
        )
        print(
            f"window {w:03d}: {len(idx)} scans, end drift {drift_xy*100:.1f} cm / "
            f"{drift_yaw:.2f} deg"
        )
    return results


def _concat_device(arrays: list[wp.array], device) -> wp.array:
    if len(arrays) == 1:
        return arrays[0]
    total = sum(len(a) for a in arrays)
    out = wp.empty(total, dtype=wp.vec3, device=device)
    o = 0
    for a in arrays:
        wp.copy(out, a, dest_offset=o, count=len(a))
        o += len(a)
    return out


def _transform_np(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    return (pts @ T[:3, :3].T + T[:3, 3]).astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seq_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--window", type=float, default=WINDOW_S)
    args = ap.parse_args()
    res = run_sequence(Path(args.seq_dir), Path(args.out_dir), window_s=args.window)
    d = np.array([r.drift_xy_m for r in res])
    print(
        f"\n{len(res)} windows; end-drift xy median {np.median(d)*100:.1f} cm, "
        f"p90 {np.percentile(d, 90)*100:.1f} cm"
    )
