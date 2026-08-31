"""Design-phase derivation of the deficit correction (PREREG section 3).

The fold's sd under the quadratic cost sits below the referee's by a
contest-depth-dependent factor. This module: (1) computes each design
window's cost-weighted median alpha (v1's definition, weights = the
attitude operator's column norms), (2) fits the two-parameter law
d(alpha) = 1 - a*exp(-b*alpha) to the measured sd ratios, (3) defines
correction(alpha) = 1/d(alpha), the multiplier clark-corr applies to the
fold's sd. Constants frozen in F1; derived on DESIGN data only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spires.risk_calibration import SIGMA_FLOOR, Window, build_nodes, resample_track
from spires.vehicle import PAD_CAP
from .score import cell_covariance, cell_variance_split
from .v2_score import _attitude_G


def window_alpha(p: Path) -> float | None:
    """Cost-weighted median contest depth of a window (v1 semantics,
    v2 weights)."""
    d = np.load(p)
    win = Window(path=p, mu=d["mu"].astype(np.float64), sigma=d["sigma"].astype(np.float64),
                 raw_sd=d["raw_sd"].astype(np.float64), meas_sd=d["meas_sd"].astype(np.float64),
                 count=d["count"], x0=float(d["xmin"]), y0=float(d["ymin"]),
                 cell=float(d["cell"]), sx=float(d["sx"]), sy=float(d["sy"]),
                 gt_poses=d["gt_poses"])
    track = resample_track(win.gt_poses)
    if len(track) == 0:
        return None
    cell_flat, caps, c_eff = build_nodes(win, track)
    t = len(track)
    mu_f, sg_f, cnt_f = win.mu.ravel(), win.sigma.ravel(), win.count.ravel()
    usable = (np.isfinite(mu_f) & np.isfinite(sg_f) & (sg_f > SIGMA_FLOOR) & (cnt_f > 0))
    node_ok = usable[cell_flat].all(axis=1).reshape(3 * t, 4).all(axis=1)
    step_ok = node_ok.reshape(3, t).all(axis=0)
    if not step_ok.any():
        return None
    sel = np.repeat(np.tile(step_ok, 3), 4)
    cell_flat, caps, c_eff = cell_flat[sel], caps[sel], c_eff[sel]
    u_cells, inv = np.unique(cell_flat.ravel(), return_inverse=True)
    u_idx = inv.reshape(cell_flat.shape)
    vi, vg, vl = cell_variance_split(win)
    C = cell_covariance(u_cells, win, vi, vg, vl)
    means = mu_f[u_cells][u_idx] + caps
    cov_self = C[u_idx[:, :, None], u_idx[:, None, :]]
    real = caps > PAD_CAP + 1.0
    m_masked = np.where(real, means, -np.inf)
    order = np.argsort(-m_masked, axis=1)
    i1, i2 = order[:, 0], order[:, 1]
    r = np.arange(means.shape[0])
    v1_, v2_ = cov_self[r, i1, i1], cov_self[r, i2, i2]
    c12 = cov_self[r, i1, i2]
    denom = np.sqrt(np.maximum(v1_ + v2_ - 2.0 * c12, 1e-18))
    alpha = np.abs(m_masked[r, i1] - m_masked[r, i2]) / denom
    G = _attitude_G(c_eff, int(step_ok.sum()))
    wcol = np.linalg.norm(G, axis=0)
    ok2 = np.isfinite(alpha) & real[r, i2]
    al, wgt = alpha[ok2], wcol[ok2]
    o = np.argsort(al)
    cw = np.cumsum(wgt[o]) / max(wgt.sum(), 1e-12)
    return float(np.interp(0.5, cw, al[o]))


def law(alpha, a, b):
    return 1.0 - a * np.exp(-b * np.asarray(alpha))


def fit_law(alphas, ratios):
    """Least squares over a coarse-to-fine grid (2 params; no scipy.optimize
    dependency drama)."""
    A, R = np.asarray(alphas), np.asarray(ratios)
    best = None
    for a in np.linspace(0.0, 0.4, 161):
        for b in np.linspace(0.05, 6.0, 240):
            r = R - law(A, a, b)
            sse = float(r @ r)
            if best is None or sse < best[0]:
                best = (sse, a, b)
    return {"a": round(best[1], 4), "b": round(best[2], 4), "sse": best[0],
            "form": "d(alpha) = 1 - a*exp(-b*alpha); correction = 1/d"}


def window_alpha_from_arrays(means, cov_self, caps, c_eff, G):
    """Same statistic as window_alpha, from the driver's arrays."""
    real = caps > -400.0  # PAD_CAP sentinel is far below any real cap
    m_masked = np.where(real, means, -np.inf)
    order = np.argsort(-m_masked, axis=1)
    i1, i2 = order[:, 0], order[:, 1]
    r = np.arange(means.shape[0])
    v1_, v2_ = cov_self[r, i1, i1], cov_self[r, i2, i2]
    c12 = cov_self[r, i1, i2]
    denom = np.sqrt(np.maximum(v1_ + v2_ - 2.0 * c12, 1e-18))
    alpha = np.abs(m_masked[r, i1] - m_masked[r, i2]) / denom
    ok2 = np.isfinite(alpha) & real[r, i2]
    wcol = np.linalg.norm(G, axis=0)
    al, wgt = alpha[ok2], wcol[ok2]
    if len(al) == 0:
        return float("nan")
    o = np.argsort(al)
    cw = np.cumsum(wgt[o]) / max(wgt.sum(), 1e-12)
    return float(np.interp(0.5, cw, al[o]))
