"""The two design-phase calibration fits, re-run as CHECKS on the pinned constants.

Both quantities are already pinned in constants.py at their rehearsal-v3 values, and the
pinned values are what the build uses. These functions refit them from the design traverses
so FREEZE.json can record the refit beside the pin: a drift between code versions becomes
visible instead of silent. Neither refit ever enters a belief.

Mount pitch. A pitch error rotates every deprojected ray, displacing a point's height by
roughly delta * (horizontal distance to the camera) -- a bias LINEAR IN RANGE -- while a
mount-height or wheel-radius error is constant in range. The estimator is therefore the delta
at which the median residual against registered truth is FLAT in range, pooled over both
design traverses by point count, with the focal length pinned at CameraInfo so pitch is the
only free parameter.

Sigma model. sigma_z(r) = a + b r^2, robust sd binned by range, measured AT the fitted pitch
(a model fitted before the correction would re-import as variance the error the mechanism just
removed) and pooled over both design traverses so the belief carries no per-traverse tuning.
"""
from __future__ import annotations

import numpy as np

from . import constants as K
from .depth import Poses, cam_transform, residuals, sample_frames
from .register import Corridor, voidfill_mask

PITCH_BINS = np.array([0.6, 0.9, 1.2, 1.6, 2.2, 3.0])
PITCH_SCAN = (-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 1.0)
N_FRAMES = 60
SIG_BINS = np.array([0.35, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])


def _context(names):
    ctx = {}
    for n in names:
        poses = Poses(n)
        cor = Corridor(poses.x, poses.y, buffer=8.0)
        ctx[n] = {"poses": poses, "paths": sample_frames(n, N_FRAMES),
                  "cor": cor, "fill": voidfill_mask(cor.a)}
    return ctx


def _profile(dz, rng):
    """(bias intercept, bias slope in m per m of range) from binned medians."""
    med, ctr = [], []
    for lo, hi in zip(PITCH_BINS[:-1], PITCH_BINS[1:]):
        s = (rng >= lo) & (rng < hi)
        if s.sum() > 500:
            med.append(float(np.median(dz[s])))
            ctr.append(0.5 * (lo + hi))
    med, ctr = np.array(med), np.array(ctr)
    A = np.stack([np.ones_like(ctr), ctr], 1)
    b, *_ = np.linalg.lstsq(A, med, rcond=None)
    return float(b[0]), float(b[1]), med.tolist(), ctr.tolist()


def refit_pitch(names, ctx=None) -> dict:
    ctx = ctx or _context(names)
    scan = []
    for dp in PITCH_SCAN:
        row = {"dpitch_deg": dp, "per_traverse": {}}
        slopes, ns = [], []
        for n in names:
            c = ctx[n]
            dz, rng = residuals(c["paths"], c["poses"], cam_transform(n, dp),
                                c["cor"], c["fill"])
            b0, b1, med, ctr = _profile(dz, rng)
            row["per_traverse"][str(n)] = {
                "bias_at_0m_m": round(b0, 4), "bias_slope_m_per_m": round(b1, 4),
                "median_resid_m": round(float(np.median(dz)), 4), "n": int(len(dz))}
            slopes.append(b1)
            ns.append(len(dz))
        w = np.array(ns, float) / sum(ns)
        row["pooled_slope_m_per_m"] = round(float(np.dot(w, slopes)), 5)
        scan.append(row)
        print(f"  pitch scan dp={dp:+.1f} pooled_slope={row['pooled_slope_m_per_m']:+.5f}",
              flush=True)
    dps = np.array([r["dpitch_deg"] for r in scan])
    pss = np.array([r["pooled_slope_m_per_m"] for r in scan])
    o = np.argsort(pss)
    dp_star = float(np.interp(0.0, pss[o], dps[o]))
    final = {}
    for n in names:
        c = ctx[n]
        dz, rng = residuals(c["paths"], c["poses"], cam_transform(n, dp_star),
                            c["cor"], c["fill"])
        b0, b1, med, ctr = _profile(dz, rng)
        final[str(n)] = {"bias_slope_m_per_m": round(b1, 5), "bias_at_0m_m": round(b0, 4),
                         "median_resid_m": round(float(np.median(dz)), 4),
                         "robust_sd_m": round(float(
                             1.4826 * np.median(np.abs(dz - np.median(dz)))), 4),
                         "n_points": int(len(dz))}
    return {"estimator": ("delta at which the median residual against registered truth is flat "
                          "in range, pooled over both design traverses by point count; fx "
                          "pinned at CameraInfo; mount quaternion rebuilt as R_y(20 + delta)"),
            "refit_MOUNT_PITCH_OFFSET_DEG": round(dp_star, 4),
            "pinned_MOUNT_PITCH_OFFSET_DEG": K.MOUNT_PITCH_OFFSET_DEG,
            "abs_diff_deg": round(abs(dp_star - K.MOUNT_PITCH_OFFSET_DEG), 5),
            "per_traverse_at_refit": final, "scan": scan}


def refit_sigma(names, dpitch: float, ctx=None) -> dict:
    """sigma_z(r) = a + b r^2, robust sd binned by range, pooled by traverse."""
    ctx = ctx or _context(names)
    rr, ss, per = [], [], {}
    for n in names:
        c = ctx[n]
        dz, rng = residuals(c["paths"], c["poses"], cam_transform(n, dpitch),
                            c["cor"], c["fill"])
        prof = []
        for lo, hi in zip(SIG_BINS[:-1], SIG_BINS[1:]):
            m = (rng >= lo) & (rng < hi)
            if m.sum() < 200:
                continue
            v = dz[m]
            prof.append({"r": round(0.5 * (lo + hi), 3), "n": int(m.sum()),
                         "median_m": round(float(np.median(v)), 4),
                         "robust_sd_m": round(float(
                             1.4826 * np.median(np.abs(v - np.median(v)))), 4)})
        r = np.array([p["r"] for p in prof])
        s = np.array([p["robust_sd_m"] for p in prof])
        A = np.stack([np.ones_like(r), r**2], 1)
        b, *_ = np.linalg.lstsq(A, s, rcond=None)
        per[str(n)] = {"a": round(float(b[0]), 5), "b": round(float(b[1]), 5),
                       "profile": prof}
        rr.append(r)
        ss.append(s)
    r, s = np.concatenate(rr), np.concatenate(ss)
    A = np.stack([np.ones_like(r), r**2], 1)
    b, *_ = np.linalg.lstsq(A, s, rcond=None)
    a_hat, b_hat = round(float(b[0]), 5), round(float(b[1]), 5)
    return {"refit_pooled": {"a": a_hat, "b": b_hat},
            "pinned": {"a": K.SIG_A, "b": K.SIG_B},
            "abs_diff": {"a": round(abs(a_hat - K.SIG_A), 6),
                         "b": round(abs(b_hat - K.SIG_B), 6)},
            "per_traverse": per,
            "sigma_at": {f"{x:.1f}": round(float(b[0] + b[1] * x**2), 4)
                         for x in (0.75, 1.0, 1.5, 2.0, 3.0, 4.0)}}
