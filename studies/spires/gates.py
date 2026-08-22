"""Design-phase gate plumbing over window beliefs (FROZEN criteria in
clark_paper/PREREG_oxford_spires.md).

Gate C -- calibration: pooled per-cell z = (mu - truth)/sigma on cells where the
window belief and the TLS raster both exist. Pass bands (virgin): |z|<=1 coverage
in [0.58, 0.78], |z|<=2 in [0.88, 0.99], sd-ratio in [0.7, 1.4].

Gate J -- contact: footprints along the GT walk (spherical envelope, R = 0.35 m,
K = 37 at 0.10 m cells). Compares |Clark E[max] - truth| vs |max-of-means - truth|
(paired sign test) and checks contact-residual z-coverage against Clark's sd.

Design-phase v1 (logged): the fold is the DIAGONAL Clark (independent cells) --
the paper's covariance-carrying machinery enters with the full study; the
diagonal version is the plumbing check.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from .tls_raster import load_or_build

WHEEL_R = 0.35
FOOT_STEP_M = 0.5


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def clark_fold_diag(mu: np.ndarray, sig: np.ndarray) -> tuple[float, float]:
    """E and sd of max of independent Gaussians, pairwise moment-matched fold."""
    order = np.argsort(mu)[::-1]
    m = float(mu[order[0]])
    v = float(sig[order[0]] ** 2)
    for i in order[1:]:
        mi, vi = float(mu[i]), float(sig[i] ** 2)
        a = math.sqrt(max(v + vi, 1e-12))
        al = (m - mi) / a
        Phi, phi = _norm_cdf(al), _norm_pdf(al)
        e = m * Phi + mi * (1.0 - Phi) + a * phi
        e2 = (m * m + v) * Phi + (mi * mi + vi) * (1.0 - Phi) + (m + mi) * a * phi
        m, v = e, max(e2 - e * e, 1e-12)
    return m, math.sqrt(v)


def _sphere_element(cell: float, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_cells = int(math.ceil(radius / cell))
    dj, di = np.meshgrid(np.arange(-r_cells, r_cells + 1), np.arange(-r_cells, r_cells + 1))
    d = np.hypot(di, dj) * cell
    m = d <= radius
    kappa = np.sqrt(np.maximum(radius**2 - d[m] ** 2, 0.0)) - radius
    return di[m], dj[m], kappa


def _sign_test_p(wins: int, losses: int) -> float:
    """Two-sided sign test, normal approximation (n is in the hundreds here)."""
    n = wins + losses
    if n == 0:
        return 1.0
    z = (wins - n / 2.0) / math.sqrt(n / 4.0)
    return 2.0 * (1.0 - _norm_cdf(abs(z)))


def run_gates(window_dir: Path, site: str) -> dict:
    tls = load_or_build(site)
    Ht, tx0, ty0, tcell = tls["H"], tls["x0"], tls["y0"], tls["cell"]
    tny, tnx = Ht.shape

    z_all: list[np.ndarray] = []
    err_all: list[np.ndarray] = []
    j_clark: list[float] = []
    j_mean: list[float] = []
    j_z: list[float] = []
    win_diffs: list[list[float]] = []  # per-window paired |err| differences
    skipped = 0
    total_fp = 0

    di, dj, kappa = _sphere_element(0.10, WHEEL_R)

    for f in sorted(window_dir.glob("window_*.npz")):
        d = np.load(f)
        mu, sg, cnt = d["mu"], d["sigma"], d["count"]
        # independent/common split: the raw Kalman sd is the per-cell independent
        # part; whatever fusion added on top (drift-induced smear) is common-mode
        # across a 0.7 m footprint, because drift shifts neighborhoods coherently
        # the pure measurement sd is the independent part; raw_sd would still
        # carry the (common) q_z drift accumulation
        if "meas_sd" in d.files:
            sg_ind = d["meas_sd"]
        elif "raw_sd" in d.files:
            sg_ind = d["raw_sd"]
        else:
            sg_ind = np.zeros_like(sg)
        x0, y0, cell = float(d["xmin"]), float(d["ymin"]), float(d["cell"])
        ny, nx = mu.shape

        # --- Gate C: cell-wise z against the TLS raster (same 0.10 m lattice
        # convention, but different origins -> index by world coordinates)
        jj, ii = np.meshgrid(np.arange(nx), np.arange(ny))
        wx = x0 + (jj + 0.5) * cell
        wy = y0 + (ii + 0.5) * cell
        tj = ((wx - tx0) / tcell).astype(np.int64)
        ti = ((wy - ty0) / tcell).astype(np.int64)
        inside = (tj >= 0) & (tj < tnx) & (ti >= 0) & (ti < tny)
        gt = np.full((ny, nx), np.nan, np.float32)
        gt[inside] = Ht[ti[inside], tj[inside]]
        ok = np.isfinite(mu) & np.isfinite(sg) & (sg > 1e-4) & np.isfinite(gt) & (cnt >= 3)
        z_all.append(((mu - gt)[ok] / sg[ok]).ravel())
        err_all.append((mu - gt)[ok].ravel())

        # --- Gate J: footprints along the GT walk inside this window
        win_diffs.append([])
        gt_poses = d["gt_poses"]
        path = gt_poses[:, :3, 3]
        # subsample path to ~FOOT_STEP_M spacing
        keep = [0]
        for k in range(1, len(path)):
            if np.linalg.norm(path[k, :2] - path[keep[-1], :2]) >= FOOT_STEP_M:
                keep.append(k)
        for k in keep:
            cx, cy = path[k, 0], path[k, 1]
            j0 = int((cx - x0) / cell)
            i0 = int((cy - y0) / cell)
            fi, fj = i0 + di, j0 + dj
            if fi.min() < 0 or fj.min() < 0 or fi.max() >= ny or fj.max() >= nx:
                continue
            total_fp += 1
            bmu = mu[fi, fj] + kappa
            bsg = sg[fi, fj]
            tjf = ((x0 + (fj + 0.5) * cell - tx0) / tcell).astype(np.int64)
            tif = ((y0 + (fi + 0.5) * cell - ty0) / tcell).astype(np.int64)
            tin = (tjf >= 0) & (tjf < tnx) & (tif >= 0) & (tif < tny)
            tval = np.full(len(kappa), np.nan)
            tval[tin] = Ht[tif[tin], tjf[tin]]
            tval = tval + kappa
            good = np.isfinite(bmu) & np.isfinite(bsg) & (bsg > 1e-4) & np.isfinite(tval)
            # the contact needs (nearly) the whole footprint on both sides
            if good.sum() < 30:
                skipped += 1
                continue
            truth = float(np.nanmax(tval[good]))
            meanmap = float(np.nanmax(bmu[good]))
            # correlated fold: fold the INDEPENDENT parts; the common part adds
            # variance to the max but not expectation (max(h_i + c) = c + max h_i)
            si = sg_ind[fi, fj][good]
            si = np.where(np.isfinite(si), si, 0.0)
            common_var = float(np.median(np.maximum(bsg[good] ** 2 - si**2, 0.0)))
            e, sd_i = clark_fold_diag(bmu[good], np.maximum(si, 1e-3))
            sd = math.sqrt(sd_i**2 + common_var)
            j_clark.append(abs(e - truth))
            j_mean.append(abs(meanmap - truth))
            j_z.append((truth - e) / sd)
            win_diffs[-1].append(abs(meanmap - truth) - abs(e - truth))

    z = np.concatenate(z_all) if z_all else np.array([])
    err = np.concatenate(err_all) if err_all else np.array([])
    jc, jm, jz = np.array(j_clark), np.array(j_mean), np.array(j_z)
    wins = int(np.sum(jc < jm))
    losses = int(np.sum(jm < jc))
    # clustered test: footprints inside one window share that window's drift, so
    # the unit of independence is the window (one median difference each)
    wmed = [float(np.median(v)) for v in win_diffs if len(v) >= 3]
    w_wins = int(np.sum(np.array(wmed) > 0))
    w_losses = int(np.sum(np.array(wmed) < 0))

    out = {
        "C_cells": len(z),
        "C_cov1": float(np.mean(np.abs(z) <= 1)) if len(z) else np.nan,
        "C_cov2": float(np.mean(np.abs(z) <= 2)) if len(z) else np.nan,
        # median|z| of a calibrated N(0,1) is 0.6745, so this ratio is 1 when
        # sigma matches the realized error scale, >1 when sigma over-covers
        "C_sd_ratio": float(0.6745 / np.median(np.abs(z))) if len(z) else np.nan,
        "C_med_err_cm": float(np.median(np.abs(err)) * 100) if len(err) else np.nan,
        "J_footprints": len(jc),
        "J_skipped": skipped,
        "J_total": total_fp,
        "J_med_clark_cm": float(np.median(jc) * 100) if len(jc) else np.nan,
        "J_med_mean_cm": float(np.median(jm) * 100) if len(jm) else np.nan,
        "J_wins": wins,
        "J_losses": losses,
        "J_p": _sign_test_p(wins, losses),
        "J_windows": len(wmed),
        "J_win_wins": w_wins,
        "J_win_p": _sign_test_p(w_wins, w_losses),
        "J_cov1": float(np.mean(np.abs(jz) <= 1)) if len(jz) else np.nan,
        "J_cov2": float(np.mean(np.abs(jz) <= 2)) if len(jz) else np.nan,
    }
    np.savez_compressed(window_dir / "gate_records.npz", z=z, err=err, jc=jc, jm=jm, jz=jz)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("window_dir")
    ap.add_argument("site")
    args = ap.parse_args()
    r = run_gates(Path(args.window_dir), args.site)
    for k, v in r.items():
        print(f"{k:16s} {v}")
