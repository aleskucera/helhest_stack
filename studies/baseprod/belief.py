"""Per-cell elevation belief from the D435i depth stream, multi-frame fused.

The filter is the spires study's `ElevationBelief` (Fankhauser RA-L 2018: per-cell 1-D Kalman
fusion, keep-the-highest Mahalanobis rule, eq.-20 motion update, III-D neighbour fusion
moment-matched at readout). It is IMPORTED from the spires study, not re-copied, so this study
cannot drift from the implementation E1 was run with.

Exactly one method is overridden -- `measure_scan`, and inside it exactly the two variance
lines. The spires model is a LIDAR one (a fixed 0.02 m range noise projected onto the vertical
through the ray's direction cosine); a stereo depth camera's error grows as r^2. The
replacement is the empirically fitted

    sigma_z(r) = SIG_A + SIG_B * r^2       [m]

with SIG_A, SIG_B frozen in constants.py (measured at the fitted mount pitch on the two design
traverses, pooled, so the belief carries no per-traverse tuning). Because the reference surface
is the registered DSM, whose own residual is ~0.045 m, this sigma is an UPPER bound on the
sensor's own noise -- recorded as a scope limit, not corrected for.

Two filter settings are frozen and both were chosen by measurement, not by default:
  alpha = 0            the paper's keep-the-highest semantics, which spires follows
  maxVariance LIFTED   the package's (3 cm)^2 default truncates every honest variance and
                       turns the multi-return gate into a ratchet toward the maximum of the
                       noise when the sensor's sd exceeds the map's (z-sd 3.77 vs 1.19).
"""
from __future__ import annotations

import numpy as np

from spires.elevation_belief import (MAHALANOBIS_GATE, MIN_H_VARIANCE, MIN_VARIANCE,
                                     MULTI_HEIGHT_NOISE, ElevationBelief)

from . import constants


class StereoBelief(ElevationBelief):
    """ElevationBelief with a stereo (range-squared) height-variance model."""

    def measure_scan(self, pts_world: np.ndarray, sensor_origin: np.ndarray) -> None:
        # ---- the ONLY change from the parent: var_p came from (SIGMA_R * dz)**2 +
        # SIGMA_Z_MIN**2 with dz the ray's vertical direction cosine; here it is the fitted
        # stereo model evaluated at slant range.
        r = np.linalg.norm(pts_world - sensor_origin[None, :], axis=1)
        var_p = (constants.SIG_A + constants.SIG_B * r**2) ** 2
        # ---- everything below is verbatim from the parent -------------------------------
        jx = ((pts_world[:, 0] - self.xmin) / self.cell).astype(np.int64)
        iy = ((pts_world[:, 1] - self.ymin) / self.cell).astype(np.int64)
        ok = (jx >= 0) & (jx < self.nx) & (iy >= 0) & (iy < self.ny)
        if not ok.any():
            return
        idx = iy[ok] * self.nx + jx[ok]
        z = pts_world[ok, 2]
        vp = var_p[ok]

        order = np.lexsort((z, idx))
        idx_s, z_s, vp_s = idx[order], z[order], vp[order]
        last = np.append(idx_s[1:] != idx_s[:-1], True)
        cells, p_tilde, p_var = idx_s[last], z_s[last], vp_s[last]

        ci, cj = np.unravel_index(cells, (self.ny, self.nx))
        h0 = self.h[ci, cj]
        v0 = self.var_h[ci, cj]

        fresh = ~np.isfinite(h0)
        d_maha = (p_tilde - h0) / np.sqrt(np.maximum(v0, 1e-12))
        conflict = ~fresh & (np.abs(d_maha) > MAHALANOBIS_GATE)
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
                a * self.var_meas[ci[higher], cj[higher]] + (1 - a) * p_var[higher])
        if lower.any():
            self.var_h[ci[lower], cj[lower]] = v0[lower] + MULTI_HEIGHT_NOISE
        if fuse.any():
            hf, vf = h0[fuse], v0[fuse]
            pf, wf = p_tilde[fuse], p_var[fuse]
            self.h[ci[fuse], cj[fuse]] = (wf * hf + vf * pf) / (vf + wf)
            self.var_h[ci[fuse], cj[fuse]] = (vf * wf) / (vf + wf)
            vm = self.var_meas[ci[fuse], cj[fuse]]
            self.var_meas[ci[fuse], cj[fuse]] = (vm * wf) / (vm + wf)
        touched = ci[~lower], cj[~lower]
        self.var_h[touched] = np.clip(self.var_h[touched], MIN_VARIANCE, self.max_variance)
        rst = fresh | fuse | higher
        self.var_x[ci[rst], cj[rst]] = MIN_H_VARIANCE
        self.var_y[ci[rst], cj[rst]] = MIN_H_VARIANCE
        self.n_upd[ci[~lower], cj[~lower]] += 1
