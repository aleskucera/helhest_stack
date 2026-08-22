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
MAHALANOBIS_GATE = 2.0  # keep-highest rule threshold (paper cites Kleiner, no number)


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
    ):
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

        # per-scan per-cell HIGHEST point (their multi-return rule applied
        # within the scan): sort by (cell, z) and keep each cell's last row
        order = np.lexsort((z, idx))
        idx_s, z_s, vp_s = idx[order], z[order], vp[order]
        last = np.append(idx_s[1:] != idx_s[:-1], True)
        cells, p_tilde, p_var = idx_s[last], z_s[last], vp_s[last]

        ci, cj = np.unravel_index(cells, (self.ny, self.nx))
        h0 = self.h[ci, cj]
        v0 = self.var_h[ci, cj]

        fresh = ~np.isfinite(h0)
        d_maha = (p_tilde - h0) / np.sqrt(v0 + p_var)
        higher = ~fresh & (d_maha > MAHALANOBIS_GATE)  # new surface above: replace
        below = ~fresh & (d_maha < -MAHALANOBIS_GATE)  # stray low return: drop
        fuse = ~fresh & ~higher & ~below

        # initialize / replace
        take = fresh | higher
        self.h[ci[take], cj[take]] = p_tilde[take]
        self.var_h[ci[take], cj[take]] = p_var[take]
        self.var_meas[ci[take], cj[take]] = p_var[take]
        # Kalman fuse (their eq. 6)
        if fuse.any():
            hf, vf = h0[fuse], v0[fuse]
            pf, wf = p_tilde[fuse], p_var[fuse]
            self.h[ci[fuse], cj[fuse]] = (wf * hf + vf * pf) / (vf + wf)
            self.var_h[ci[fuse], cj[fuse]] = (vf * wf) / (vf + wf)
            vm = self.var_meas[ci[fuse], cj[fuse]]
            self.var_meas[ci[fuse], cj[fuse]] = (vm * wf) / (vm + wf)
        self.n_upd[ci[~below], cj[~below]] += 1

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
        rx = min(int(np.ceil(2.0 * sx / self.cell)), 8)
        ry = min(int(np.ceil(2.0 * sy / self.cell)), 8)

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

        for di in range(-ry, ry + 1):
            for dj in range(-rx, rx + 1):
                w = np.exp(
                    -0.5
                    * (
                        (dj * self.cell / max(sx, 1e-3)) ** 2
                        + (di * self.cell / max(sy, 1e-3)) ** 2
                    )
                )
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
