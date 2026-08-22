"""Faithful reimplementation of the Fankhauser RA-L 2018 elevation belief.

Per the prereg amendment (clark_paper/PREREG_oxford_spires.md): per-cell 1-D
Kalman fusion with a range-noise sensor model (their eqs. 5-6), the
keep-the-highest Mahalanobis multi-return rule, the motion update adding
relative-pose covariance to every cell (their eq. 20; diagonal var_x/var_y/var_h
with yaw lever-arm terms), and the III-D neighbor fusion moment-matched at
readout (law of total variance; uniform-kernel simplification documented below).

Host-side study harness, not robot-runtime code -- numpy is the right tool at
this dataset boundary (cf. the loaders); windows are ~1k x 1k cells.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIGMA_R = 0.02  # [m] a-priori Hesai range noise (fixed in the prereg, not tuned)
SIGMA_Z_MIN = 0.01  # [m] variance floor on a fresh cell's height
# audited against the elevation_mapping package (ElevationMap.hpp defaults):
MAHALANOBIS_GATE = 2.5  # their mahalanobisDistanceThreshold_
MIN_H_VARIANCE = 1.0e-4  # their minHorizontalVariance_ -- RESET on measurement
MULTI_HEIGHT_NOISE = 9.0e-6  # their multiHeightNoise_ (cross-scan conflict inflation)
MIN_VARIANCE = 9.0e-6  # their minVariance_ clamp
# their maxVariance_ default is 9e-4 ((3 cm)^2): the package CLAMPS vertical
# variance there, truncating any honestly-supplied pose covariance. Kept as a
# parameter so the design site can measure both the default and an unclamped run.
MAX_VARIANCE_DEFAULT = 9.0e-4


@dataclass
class DriftRates:
    """Random-walk rates [var/s], frozen from the design-site calibration."""

    q_x: float = 4.202e-3
    q_y: float = 1.061e-2
    q_z: float = 7.430e-3
    q_yaw: float = 3.651e-4


class ElevationBelief:
    """One window's belief grid; world-anchored at the window's first GT pose."""

    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        cell: float,
        rates: DriftRates | None = None,
        *,
        max_variance: float = MAX_VARIANCE_DEFAULT,
        increase_height_alpha: float = 0.0,
    ):
        # alpha follows the PAPER's keep-highest semantics (alpha=0: jump to the
        # higher surface); the package's default alpha=1 ignores it -- the
        # paper-vs-code discrepancy is documented in the prereg design log
        self.max_variance = max_variance
        self.alpha = increase_height_alpha
        self.xmin, self.xmax, self.ymin, self.ymax = bounds
        self.cell = cell
        self.rates = rates or DriftRates()
        self.nx = int(round((self.xmax - self.xmin) / cell))
        self.ny = int(round((self.ymax - self.ymin) / cell))
        shape = (self.ny, self.nx)
        self.h = np.full(shape, np.nan)
        self.var_h = np.full(shape, np.nan)
        # measurement-only variance: the Kalman arithmetic WITHOUT the q_z motion
        # inflation -- the truly independent per-cell part; the difference
        # var_h - var_meas is drift, common-mode across a footprint
        self.var_meas = np.full(shape, np.nan)
        self.var_x = np.zeros(shape)
        self.var_y = np.zeros(shape)
        self.n_upd = np.zeros(shape, np.int32)
        # cell-center world coordinates, for the yaw lever arms
        jj, ii = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        self._wx = self.xmin + (jj + 0.5) * cell
        self._wy = self.ymin + (ii + 0.5) * cell

    def motion_update(self, dt: float, robot_xy: np.ndarray) -> None:
        """Eq. 20, diagonalized: every cell's uncertainty grows with pose drift.

        Yaw drift moves a cell tangentially around the robot, so its lever arm
        feeds x-variance with the PERPENDICULAR offset (ry) and vice versa.
        """
        if dt <= 0:
            return
        r = self.rates
        rx = self._wx - robot_xy[0]
        ry = self._wy - robot_xy[1]
        qy_dt = r.q_yaw * dt
        self.var_x += r.q_x * dt + qy_dt * ry**2
        self.var_y += r.q_y * dt + qy_dt * rx**2
        # height variance grows only where a height exists to be wrong about
        m = np.isfinite(self.var_h)
        self.var_h[m] += r.q_z * dt

    def measure_scan(self, pts_world: np.ndarray, sensor_origin: np.ndarray) -> None:
        """Eqs. 5-6 + the keep-the-highest rule, one update per touched cell.

        Per-point height variance: range noise sigma_r acts along the ray, so
        only its z-component lands in the height (their eq. 5 with the sensor
        rotation term handled by the motion update instead -- documented in the
        prereg amendment).
        """
        d = pts_world - sensor_origin[None, :]
        dz = np.abs(d[:, 2]) / np.maximum(np.linalg.norm(d, axis=1), 1e-6)
        var_p = (SIGMA_R * dz) ** 2 + SIGMA_Z_MIN**2

        jx = ((pts_world[:, 0] - self.xmin) / self.cell).astype(np.int64)
        iy = ((pts_world[:, 1] - self.ymin) / self.cell).astype(np.int64)
        ok = (jx >= 0) & (jx < self.nx) & (iy >= 0) & (iy < self.ny)
        if not ok.any():
            return
        idx = iy[ok] * self.nx + jx[ok]
        z = pts_world[ok, 2]
        vp = var_p[ok]

        # per-scan per-cell HIGHEST point first: within one scan the package
        # ignores lower conflicts and (paper semantics, alpha=0) jumps to higher
        # ones, so the scan's highest point is the effective same-scan update
        order = np.lexsort((z, idx))
        idx_s, z_s, vp_s = idx[order], z[order], vp[order]
        last = np.append(idx_s[1:] != idx_s[:-1], True)
        cells, p_tilde, p_var = idx_s[last], z_s[last], vp_s[last]

        ci, cj = np.unravel_index(cells, (self.ny, self.nx))
        h0 = self.h[ci, cj]
        v0 = self.var_h[ci, cj]

        fresh = ~np.isfinite(h0)
        # their gate: |z - h| / sqrt(map variance) -- the MAP sd only (audited)
        d_maha = (p_tilde - h0) / np.sqrt(np.maximum(v0, 1e-12))
        conflict = ~fresh & (np.abs(d_maha) > MAHALANOBIS_GATE)
        # cross-scan conflict (their else-branch): elevation kept, variance
        # inflated by multiHeightNoise; paper-alpha semantics for HIGHER points:
        # blend toward the new surface with weight (1 - alpha)
        higher = conflict & (d_maha > 0)
        lower = conflict & (d_maha <= 0)
        fuse = ~fresh & ~conflict

        take = fresh
        self.h[ci[take], cj[take]] = p_tilde[take]
        self.var_h[ci[take], cj[take]] = p_var[take]
        self.var_meas[ci[take], cj[take]] = p_var[take]
        if higher.any():
            a = self.alpha
            self.h[ci[higher], cj[higher]] = a * h0[higher] + (1 - a) * p_tilde[higher]
            self.var_h[ci[higher], cj[higher]] = a * v0[higher] + (1 - a) * p_var[higher]
            self.var_meas[ci[higher], cj[higher]] = (
                a * self.var_meas[ci[higher], cj[higher]] + (1 - a) * p_var[higher]
            )
        if lower.any():
            self.var_h[ci[lower], cj[lower]] = v0[lower] + MULTI_HEIGHT_NOISE
        # Kalman fuse (their eq. 6)
        if fuse.any():
            hf, vf = h0[fuse], v0[fuse]
            pf, wf = p_tilde[fuse], p_var[fuse]
            self.h[ci[fuse], cj[fuse]] = (wf * hf + vf * pf) / (vf + wf)
            self.var_h[ci[fuse], cj[fuse]] = (vf * wf) / (vf + wf)
            vm = self.var_meas[ci[fuse], cj[fuse]]
            self.var_meas[ci[fuse], cj[fuse]] = (vm * wf) / (vm + wf)
        # variance clamps (their VarianceClampOperator)
        touched = ci[~lower], cj[~lower]
        self.var_h[touched] = np.clip(self.var_h[touched], MIN_VARIANCE, self.max_variance)
        # horizontal variances RESET on measurement (audited: init + fuse branches;
        # this is what keeps freshly-seen cells sharp in the package)
        rst = fresh | fuse | higher
        self.var_x[ci[rst], cj[rst]] = MIN_H_VARIANCE
        self.var_y[ci[rst], cj[rst]] = MIN_H_VARIANCE
        self.n_upd[ci[~lower], cj[~lower]] += 1

    def readout(self) -> dict[str, np.ndarray]:
        """III-D fusion, moment-matched: horizontal uncertainty becomes vertical
        variance through the neighbor mixture (law of total variance).

        Uniform-kernel simplification (documented in the prereg): var_x/var_y
        vary smoothly over one window, so a single median horizontal sd drives
        one shared neighbor kernel instead of a per-cell ellipse.
        """
        finite = np.isfinite(self.h)
        sx = float(np.sqrt(np.median(self.var_x[finite]))) if finite.any() else 0.0
        sy = float(np.sqrt(np.median(self.var_y[finite]))) if finite.any() else 0.0
        # their ellipse: 2.486 sigma (95% chi-square, 2 dof) + sqrt(2) * resolution
        rx = min(int(np.ceil((2.486 * sx + 1.42 * self.cell) / self.cell)), 25)
        ry = min(int(np.ceil((2.486 * sy + 1.42 * self.cell) / self.cell)), 25)

        w_sum = np.zeros_like(self.h)
        wh = np.zeros_like(self.h)
        wh2v = np.zeros_like(self.h)
        # zero-padded shifts (np.roll would wrap the far border into the fusion)
        pad = max(rx, ry)
        h0 = np.pad(np.where(finite, self.h, 0.0), pad)
        v0 = np.pad(np.where(finite, self.var_h, 0.0), pad)
        f0 = np.pad(finite.astype(float), pad)

        def shifted(a: np.ndarray, di: int, dj: int) -> np.ndarray:
            return a[pad + di : pad + di + self.ny, pad + dj : pad + dj + self.nx]

        from math import erf, sqrt

        def cdf(x: float, s: float) -> float:
            return 0.5 * (1.0 + erf(x / (s * sqrt(2.0))))

        half = self.cell / 2.0
        for di in range(-ry, ry + 1):
            for dj in range(-rx, rx + 1):
                # their weight: per-axis Gaussian probability MASS over the cell
                dx, dy = abs(dj) * self.cell, abs(di) * self.cell
                wx = cdf(dx + half, max(sx, 1e-3)) - cdf(dx - half, max(sx, 1e-3))
                wy = cdf(dy + half, max(sy, 1e-3)) - cdf(dy - half, max(sy, 1e-3))
                w = max(wx * wy, 1e-12)
                src_f = shifted(f0, di, dj)
                w_sum += w * src_f
                wh += w * src_f * shifted(h0, di, dj)
                wh2v += w * src_f * (shifted(v0, di, dj) + shifted(h0, di, dj) ** 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = wh / w_sum
            var = wh2v / w_sum - mu**2
        mu[w_sum <= 0] = np.nan
        sigma = np.sqrt(np.maximum(var, 0.0))
        sigma[w_sum <= 0] = np.nan
        return {
            "mu": mu,
            "sigma": sigma,
            "raw_h": self.h.copy(),
            "raw_sd": np.sqrt(self.var_h),
            "meas_sd": np.sqrt(self.var_meas),
            "count": self.n_upd.copy(),
            "sx": sx,
            "sy": sy,
        }
