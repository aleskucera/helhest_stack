"""Window products: the paired FORESIGHT / HINDSIGHT beliefs on identical windows.

PREREG_baseprod.md, "Belief -- two paired conditions on the same windows":

  FORESIGHT (PRIMARY, the planning-time belief) -- for each window, fusion uses ONLY sensor
  frames acquired while the rover's RTK along-track arc length is before the window's start.
  Cells never observed under this cutoff carry no belief.

  HINDSIGHT (contrast) -- full-trajectory fusion, as in the design previews: the belief at
  its lifetime best.

  Both conditions share every other pipeline element.

Concretely, with a window spanning arc length [a, b) and LOOKBACK_M the shared look-back:

  hindsight   frames with   a - LOOKBACK_M <= s(frame) <= b
  foresight   frames with   a - LOOKBACK_M <= s(frame) <  a

The look-back is a shared pipeline element, so the foresight set is the hindsight set
intersected with the cutoff s < a -- a strict subset, never a superset. Applying the cutoff
without the shared look-back would let late windows fuse the entire preceding traverse and
would make "foresight" richer than "hindsight" for those windows, which is not the condition
the prereg defines.

Windows are WIN_LEN_M = 4 m long: the depth camera's range cap, and the horizon a planner
predicts over. The first design phase ran at 10 m and failed C2 for a structural reason -- a
10 m window is never observed past ~4 m from before its own start, so foresight retention
clustered at 0.40-0.55 and 11 of 17 windows fell below the retention floor.

At the LEGACY 10 m length the hindsight branch is byte-for-byte the rehearsal-v3 build
(`build_v3.py`) at the pinned constants. Freeze condition C1 exercises exactly that path, so
the window-length change is demonstrably a configuration change and not a code change.
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from . import constants as K
from .belief import StereoBelief
from .depth import (Poses, cam_transform, frame_paths, frame_time, uv_grid)
from .paths import KEY, OUT, window_dir
from .register import Corridor, npz_path, voidfill_mask


def smoothed_track(name: str):
    """Arc length along a SMOOTHED RTK track.

    Summing |diff| over the raw 4 Hz fixes counts the 1.4 cm position noise as travel (70+ m
    for a traverse whose true path length is 55.9 m). A centred rolling mean over 9 fixes
    (~2.2 s, ~15 cm of real motion) removes it.
    """
    z = np.load(npz_path(name))
    x, y, t = z["base"][:, 0], z["base"][:, 1], z["t"].astype(np.float64)
    k = np.ones(K.TRACK_SMOOTH_FIXES) / K.TRACK_SMOOTH_FIXES
    pad = K.TRACK_SMOOTH_FIXES // 2
    xs = np.convolve(np.pad(x, pad, mode="edge"), k, "valid")
    ys = np.convolve(np.pad(y, pad, mode="edge"), k, "valid")
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    return s, xs, ys, t, z


def frame_mask(fs: np.ndarray, a: float, b: float, condition: str) -> np.ndarray:
    lo = fs >= a - K.LOOKBACK_M
    if condition == "hindsight":
        return lo & (fs <= b)
    if condition == "foresight":
        return lo & (fs < a)
    raise ValueError(f"unknown belief condition {condition!r}")


def build(name: str, condition: str, win_len_m: float | None = None,
          tag: str | None = None) -> dict:
    """Build every window product of one traverse under one belief condition.

    `win_len_m` defaults to the pinned WIN_LEN_M (4 m, the perception horizon). The only other
    value it is ever given is LEGACY_WIN_LEN_M (10 m), for the C1 code-parity check against
    rehearsal v3; `tag` then routes those products into their own directory so they cannot be
    confused with the scored ones.
    """
    win_len = K.WIN_LEN_M if win_len_m is None else float(win_len_m)
    key = KEY.get(name, name)
    wdir = window_dir(name, tag or condition)
    wdir.mkdir(parents=True, exist_ok=True)

    s, sx_, sy_, t, z = smoothed_track(name)
    poses = Poses(name)
    T_cam = cam_transform(name, K.MOUNT_PITCH_OFFSET_DEG)
    ps = frame_paths(name)
    if not ps:
        return {"traverse": str(name), "key": key, "condition": condition,
                "win_len_m": win_len, "tag": tag or condition,
                "path_length_m": round(float(s[-1]), 2), "n_windows": 0,
                "windows": [], "note": "no RS_DEPTH_16bit frames"}
    fts = np.array([frame_time(p) for p in ps], np.float64)
    fs = np.interp(fts, t, s)
    fpos, fR = poses.at(fts)
    yaw_raw = z["yaw"]
    cor = Corridor(poses.x, poses.y, buffer=K.PAD_M + 1.0)
    fill = voidfill_mask(cor.a)
    u, v = uv_grid()
    wins, skipped = [], []

    for w in range(int(np.floor(s[-1] / win_len))):
        a, b = w * win_len, (w + 1) * win_len
        fm = frame_mask(fs, a, b, condition)
        if fm.sum() < K.MIN_FRAMES_PER_WINDOW:
            skipped.append({"w": w, "n_frames": int(fm.sum()),
                            "reason": f"< {K.MIN_FRAMES_PER_WINDOW} frames under {condition}"})
            continue
        idx = np.where(fm)[0]
        pos, R = fpos[idx], fR[idx]
        bounds = (pos[:, 0].min() - K.PAD_M, pos[:, 0].max() + K.PAD_M,
                  pos[:, 1].min() - K.PAD_M, pos[:, 1].max() + K.PAD_M)
        bel = StereoBelief(bounds, K.CELL, rates=None, max_variance=K.MAX_VARIANCE,
                           increase_height_alpha=K.ALPHA)
        npts = 0
        for j, i in enumerate(idx):
            d = (np.asarray(Image.open(ps[i]), np.uint16)[::K.STRIDE, ::K.STRIDE].ravel()
                 * K.DEPTH_SCALE)
            ok = (d > K.R_MIN) & (d < K.R_MAX)
            if not ok.any():
                continue
            dd, uu, vv = d[ok], u[ok], v[ok]
            po = np.stack([(uu - K.CX) * dd / K.FX, (vv - K.CY) * dd / K.FY, dd], 1)
            pw = (po @ T_cam[:3, :3].T + T_cam[:3, 3]) @ R[j].T + pos[j]
            bel.measure_scan(pw, T_cam[:3, 3] @ R[j].T + pos[j])
            npts += len(dd)
        o = bel.readout()
        ny, nx = o["mu"].shape

        # --- truth layers on the window lattice ------------------------------------------
        jj, ii = np.meshgrid(np.arange(nx), np.arange(ny))
        wx = bounds[0] + (jj + 0.5) * K.CELL
        wy = bounds[2] + (ii + 0.5) * K.CELL
        rr, cc = cor.rc(wx.ravel(), wy.ravel())
        half = int(round(K.CELL / cor.res / 2))
        dd_ = np.arange(-half, half + 1)
        DR, DC = np.meshgrid(dd_, dd_, indexing="ij")
        r_ = np.round(rr)[:, None] + DR.ravel()[None, :]
        c_ = np.round(cc)[:, None] + DC.ravel()[None, :]
        inb = ((r_ >= 0) & (r_ < cor.a.shape[0]) & (c_ >= 0) & (c_ < cor.a.shape[1]))
        r_ = np.clip(r_, 0, cor.a.shape[0] - 1).astype(np.int32)
        c_ = np.clip(c_, 0, cor.a.shape[1] - 1).astype(np.int32)
        vals = np.where(inb & ~fill[r_, c_], cor.a[r_, c_], np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            tmax, tmean = np.nanmax(vals, axis=1), np.nanmean(vals, axis=1)
        corr = poses.correction(wx.ravel(), wy.ravel()) - K.R_WHEEL
        truth_max = (tmax + corr).reshape(ny, nx)
        truth_mean = (tmean + corr).reshape(ny, nx)

        # --- gt poses: the resampled track through this window, as 4x4 -------------------
        sq = np.arange(a, b, K.CELL)
        px, py = np.interp(sq, s, sx_), np.interp(sq, s, sy_)
        pyaw = np.interp(sq, s, np.unwrap(yaw_raw))
        T = np.zeros((len(sq), 4, 4))
        T[:, 3, 3] = 1.0
        T[:, 2, 2] = 1.0
        T[:, 0, 0] = np.cos(pyaw); T[:, 0, 1] = -np.sin(pyaw)
        T[:, 1, 0] = np.sin(pyaw); T[:, 1, 1] = np.cos(pyaw)
        T[:, 0, 3], T[:, 1, 3] = px, py
        T[:, 2, 3] = np.interp(sq, s, z["base"][:, 2])

        np.savez_compressed(
            wdir / f"window_{w:03d}.npz",
            mu=o["mu"].astype(np.float32), sigma=o["sigma"].astype(np.float32),
            raw_sd=o["raw_sd"].astype(np.float32), meas_sd=o["meas_sd"].astype(np.float32),
            count=o["count"], xmin=bounds[0], ymin=bounds[2], cell=K.CELL,
            sx=o["sx"], sy=o["sy"], gt_poses=T,
            truth_max=truth_max.astype(np.float32), truth_mean=truth_mean.astype(np.float32))
        wins.append({"w": w, "s0": a, "s1": b, "grid": [int(ny), int(nx)],
                     "n_frames": int(len(idx)), "n_points": int(npts),
                     "map_observed_frac": round(float((o["count"] > 0).mean()), 4),
                     "truth_valid_frac": round(float(np.isfinite(truth_max).mean()), 4)})
        print(f"  [{tag or condition}] {key} w{w:02d} grid={ny}x{nx} frames={len(idx)} pts={npts} "
              f"obs={(o['count'] > 0).mean():.3f}", flush=True)

    meta = {"traverse": str(name), "key": key, "condition": condition,
            "tag": tag or condition, "win_len_m": win_len,
            "path_length_m": round(float(s[-1]), 2),
            "n_depth_frames": int(len(ps)),
            "cutoff_rule": ("frames with a - LOOKBACK_M <= s < a (arc length strictly before "
                            "the window start)") if condition == "foresight" else
                           "frames with a - LOOKBACK_M <= s <= b (full trajectory)",
            "lookback_m": K.LOOKBACK_M, "cell": K.CELL,
            "alpha": K.ALPHA, "max_variance": K.MAX_VARIANCE,
            "intrinsics": {"fx": K.FX, "fy": K.FY, "cx": K.CX, "cy": K.CY},
            "mount_pitch_offset_deg": K.MOUNT_PITCH_OFFSET_DEG,
            "sigma_model": {"a": K.SIG_A, "b": K.SIG_B},
            "n_windows": len(wins), "windows": wins, "skipped": skipped}
    d = OUT / "build_meta" / (tag or condition)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{key}.json").write_text(json.dumps(meta, indent=2))
    return meta
