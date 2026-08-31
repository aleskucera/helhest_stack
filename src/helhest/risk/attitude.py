"""GPU estimator for the v2 attitude cost  L = sum_t (pitch_t^2 + roll_t^2).

Companion to `contact.py` (same node machinery, same rho1 separable belief
model, same register-resident Clark fold). Where the v1 scalar cost allowed
the variance to be one grid convolution, the attitude cost needs the joint
Gaussian of the stacked attitude vector x (2T rows): S_x = W C W'. That is
computed by direct pair contraction over folded nodes -- each attitude row
touches 12 nodes (3 wheels x 4 corners) of <= K candidates, and pairs of
rows further apart than the correlation table's reach contribute exactly
zero, so the work is banded.

The fold semantics match `studies/baseprod/v2_moments.fold_supports_fast`:
cross-node covariance by the lambda (SSTA) rule, Clark-exact variance on
the diagonal (carried as a per-node correction `dvar`), so the campaign's
frozen deficit correction applies unchanged.

Moments only (E[L], Var[L]); CVaR and the deficit correction are one
fused multiply-add on the host or in the MPPI cost kernel.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from .contact import MAX_K, VECF, VECI, _r1, k_nodes
from .settle import settle_map


@wp.kernel
def k_fold_att(
    belief: wp.array2d(dtype=wp.float32),
    sigma: wp.array2d(dtype=wp.float32),
    off_dy: wp.array2d(dtype=wp.int32),
    off_dx: wp.array2d(dtype=wp.int32),
    off_cap: wp.array2d(dtype=wp.float32),
    k_real: wp.array(dtype=wp.int32),
    rho1: wp.array(dtype=wp.float32),
    anchor_y: wp.array(dtype=wp.int32),
    anchor_x: wp.array(dtype=wp.int32),
    node_bin: wp.array(dtype=wp.int32),
    ny: int, nx: int,
    node_mean: wp.array(dtype=wp.float32),
    node_dvar: wp.array(dtype=wp.float32),
    cand_iy: wp.array2d(dtype=wp.int32),
    cand_ix: wp.array2d(dtype=wp.int32),
    cand_wsig: wp.array2d(dtype=wp.float32),
):
    """Per node: gather, sort, fold (contact.py's recursion verbatim), then
    write the folded mean, the lambda-weighted candidates, and the
    Clark-vs-lambda variance gap."""
    n = wp.tid()
    b = node_bin[n]
    ay = anchor_y[n]
    ax = anchor_x[n]
    kk = k_real[b]

    m = VECF(); sg = VECF(); ody = VECI(); odx = VECI(); ci = VECI(); cj = VECI()
    for j in range(kk):
        iy = wp.clamp(ay + off_dy[b, j], 0, ny - 1)
        ix = wp.clamp(ax + off_dx[b, j], 0, nx - 1)
        m[j] = belief[iy, ix] + off_cap[b, j]
        sg[j] = sigma[iy, ix]
        ody[j] = off_dy[b, j]; odx[j] = off_dx[b, j]
        ci[j] = iy; cj[j] = ix
    for a in range(1, kk):
        km = m[a]; ks = sg[a]; kdy = ody[a]; kdx = odx[a]; kci = ci[a]; kcj = cj[a]
        q = a - 1
        while q >= 0 and m[q] < km:
            m[q+1] = m[q]; sg[q+1] = sg[q]; ody[q+1] = ody[q]; odx[q+1] = odx[q]
            ci[q+1] = ci[q]; cj[q+1] = cj[q]
            q = q - 1
        m[q+1] = km; sg[q+1] = ks; ody[q+1] = kdy; odx[q+1] = kdx; ci[q+1] = kci; cj[q+1] = kcj

    base = m[0]
    for j in range(kk):
        m[j] = m[j] - base

    mean_run = m[0]
    var_run = sg[0] * sg[0]
    cov = VECF()
    for j in range(kk):
        cov[j] = _r1(rho1, ody[0] - ody[j]) * _r1(rho1, odx[0] - odx[j]) * sg[0] * sg[j]
    phi_s = VECF(); phin_s = VECF()
    for i in range(1, kk):
        v2 = sg[i] * sg[i]
        a2 = var_run + v2 - 2.0 * cov[i]
        if a2 < 0.0:
            a2 = 0.0
        aa = wp.sqrt(a2)
        alpha = float(0.0)
        if aa < 1.0e-9:
            if mean_run - m[i] >= 0.0:
                alpha = 1.0e6
            else:
                alpha = -1.0e6
        else:
            alpha = (mean_run - m[i]) / aa
        ph = 0.5 * (1.0 + wp.erf(alpha * 0.70710678))
        pn = 1.0 - ph
        pd = 0.39894228 * wp.exp(-0.5 * alpha * alpha)
        new_mean = mean_run * ph + m[i] * pn + aa * pd
        new_ex2 = ((mean_run * mean_run + var_run) * ph + (m[i] * m[i] + v2) * pn
                   + (mean_run + m[i]) * aa * pd)
        var_run = wp.max(new_ex2 - new_mean * new_mean, 0.0)
        for j in range(kk):
            cov[j] = (cov[j] * ph
                      + _r1(rho1, ody[i] - ody[j]) * _r1(rho1, odx[i] - odx[j])
                      * sg[i] * sg[j] * pn)
        phi_s[i-1] = ph; phin_s[i-1] = pn
        mean_run = new_mean

    wv = VECF()
    tail = float(1.0)
    for i in range(kk - 1, 0, -1):
        wv[i] = phin_s[i-1] * tail
        tail = tail * phi_s[i-1]
    wv[0] = tail

    # lambda C lambda (this node with itself) for the diagonal correction
    lcl = float(0.0)
    for a in range(kk):
        for bb in range(kk):
            lcl += (wv[a] * wv[bb] * sg[a] * sg[bb]
                    * _r1(rho1, ody[a] - ody[bb]) * _r1(rho1, odx[a] - odx[bb]))
    node_mean[n] = mean_run + base
    node_dvar[n] = var_run - lcl
    for j in range(kk):
        cand_iy[n, j] = ci[j]
        cand_ix[n, j] = cj[j]
        cand_wsig[n, j] = wv[j] * sg[j]
    for j in range(kk, MAX_K):
        cand_wsig[n, j] = 0.0
        cand_iy[n, j] = ci[0]
        cand_ix[n, j] = cj[0]


@wp.kernel
def k_att_cov(
    node_mean: wp.array(dtype=wp.float32),
    node_dvar: wp.array(dtype=wp.float32),
    cand_iy: wp.array2d(dtype=wp.int32),
    cand_ix: wp.array2d(dtype=wp.int32),
    cand_wsig: wp.array2d(dtype=wp.float32),
    node_wgt: wp.array(dtype=wp.float32),
    rho1: wp.array(dtype=wp.float32),
    row_w: wp.array2d(dtype=wp.float32),
    n_t: int, band_t: int, kk_max: int,
    S: wp.array3d(dtype=wp.float32),
    mx: wp.array2d(dtype=wp.float32),
):
    """One thread per (plan, tr, ts<=tr within band): computes the 12x12
    node-pair covariances of the step pair ONCE and scatters them into all
    four (pitch/roll x pitch/roll) entries of the 2x2 attitude block."""
    tid = wp.tid()
    nb_ = 2 * band_t + 1
    p = tid / (n_t * nb_)
    r_ = tid % (n_t * nb_)
    tr = r_ / nb_
    ts = tr - band_t + (r_ % nb_)
    if ts < 0 or ts > tr:
        return
    s00 = float(0.0)  # pitch-pitch
    s01 = float(0.0)  # pitch(tr)-roll(ts)
    s10 = float(0.0)  # roll(tr)-pitch(ts)
    s11 = float(0.0)  # roll-roll
    for wa in range(3):
        for ca in range(4):
            na = ((p * 3 + wa) * n_t + tr) * 4 + ca
            ga = node_wgt[na]
            if ga == 0.0:
                continue
            for wb in range(3):
                for cb in range(4):
                    nb2 = ((p * 3 + wb) * n_t + ts) * 4 + cb
                    gb = node_wgt[nb2]
                    if gb == 0.0:
                        continue
                    c = float(0.0)
                    if na == nb2:
                        c = node_dvar[na]
                    for j in range(kk_max):
                        wsa = cand_wsig[na, j]
                        if wsa == 0.0:
                            continue
                        ya = cand_iy[na, j]
                        xa = cand_ix[na, j]
                        for j2 in range(kk_max):
                            wsb = cand_wsig[nb2, j2]
                            if wsb == 0.0:
                                continue
                            c += (wsa * wsb * _r1(rho1, ya - cand_iy[nb2, j2])
                                  * _r1(rho1, xa - cand_ix[nb2, j2]))
                    g = ga * gb * c
                    s00 += row_w[0, wa] * row_w[0, wb] * g
                    s01 += row_w[0, wa] * row_w[1, wb] * g
                    s10 += row_w[1, wa] * row_w[0, wb] * g
                    s11 += row_w[1, wa] * row_w[1, wb] * g
    r0 = 2 * tr
    c0 = 2 * ts
    S[p, r0, c0] = s00
    S[p, r0, c0 + 1] = s01
    S[p, r0 + 1, c0] = s10
    S[p, r0 + 1, c0 + 1] = s11
    S[p, c0, r0] = s00
    S[p, c0 + 1, r0] = s01
    S[p, c0, r0 + 1] = s10
    S[p, c0 + 1, r0 + 1] = s11
    if tr == ts:
        mp = float(0.0)
        mr = float(0.0)
        for wa in range(3):
            for ca in range(4):
                na = ((p * 3 + wa) * n_t + tr) * 4 + ca
                mp += row_w[0, wa] * node_wgt[na] * node_mean[na]
                mr += row_w[1, wa] * node_wgt[na] * node_mean[na]
        mx[p, r0] = mp
        mx[p, r0 + 1] = mr


@wp.kernel
def k_att_finish(
    S: wp.array3d(dtype=wp.float32),
    mx: wp.array2d(dtype=wp.float32),
    n2: int,
    e_l: wp.array(dtype=wp.float32),
    var_l: wp.array(dtype=wp.float32),
):
    p = wp.tid()
    E = float(0.0)
    V = float(0.0)
    for r in range(n2):
        E += S[p, r, r] + mx[p, r] * mx[p, r]
        srm = float(0.0)
        for s in range(n2):
            V += 2.0 * S[p, r, s] * S[p, r, s]
            srm += S[p, r, s] * mx[p, s]
        V += 4.0 * mx[p, r] * srm
    e_l[p] = E
    var_l[p] = V


class WarpAttitudeEstimator:
    """E[L], Var[L] of the attitude cost per plan, device-resident."""

    def __init__(self, ny, nx, cell, origin_x, origin_y, rp, rho1, n_plans, n_t,
                 device="cuda:0", n_bins=None, band_m=None):
        from .contact import WarpContactEstimator
        # reuse the contact estimator's node/table construction wholesale
        self._c = WarpContactEstimator(ny, nx, cell, origin_x, origin_y, rp, rho1,
                                       n_plans, n_t, device=device,
                                       **({} if n_bins is None else {"n_bins": n_bins}))
        self.device = device
        self.n_plans, self.n_t = n_plans, n_t
        n_nodes = self._c.n_nodes
        R = settle_map(rp)[1:3, :].astype(np.float32)   # (pitch, roll) x wheel
        reach = (len(np.asarray(rho1)) * cell)
        self.band_t = int(np.ceil((reach + 2 * 0.5) / 0.10)) if band_m is None \
            else int(np.ceil(band_m / 0.10))
        with wp.ScopedDevice(device):
            self.row_w = wp.array(R, dtype=wp.float32)
            self.node_mean = wp.zeros(n_nodes, dtype=wp.float32)
            self.node_dvar = wp.zeros(n_nodes, dtype=wp.float32)
            self.cand_iy = wp.zeros((n_nodes, MAX_K), dtype=wp.int32)
            self.cand_ix = wp.zeros((n_nodes, MAX_K), dtype=wp.int32)
            self.cand_wsig = wp.zeros((n_nodes, MAX_K), dtype=wp.float32)
            self.node_wgt = wp.zeros(n_nodes, dtype=wp.float32)
            self.S = wp.zeros((n_plans, 2 * n_t, 2 * n_t), dtype=wp.float32)
            self.mx = wp.zeros((n_plans, 2 * n_t), dtype=wp.float32)
            self.e_l = wp.zeros(n_plans, dtype=wp.float32)
            self.var_l = wp.zeros(n_plans, dtype=wp.float32)
            ones = wp.array(np.ones(3, np.float32), dtype=wp.float32)
            self._ones_settle = ones

    def moments(self, belief, sigma, ctrl):
        c = self._c
        d = self.device
        # node pass with unit settle weights: node_ceff becomes the bare
        # bilinear corner weight
        wp.launch(k_nodes, dim=c.n_nodes, device=d,
                  inputs=[ctrl, c.wheel_xy, self._ones_settle, c.origin_x, c.origin_y,
                          c.cell, c.ny, c.nx, c.n_bins, c.n_t],
                  outputs=[c.anchor_y, c.anchor_x, c.node_bin, self.node_wgt])
        wp.launch(k_fold_att, dim=c.n_nodes, device=d,
                  inputs=[belief, sigma, c.off_dy, c.off_dx, c.off_cap, c.k_real,
                          c.rho1, c.anchor_y, c.anchor_x, c.node_bin, c.ny, c.nx],
                  outputs=[self.node_mean, self.node_dvar,
                           self.cand_iy, self.cand_ix, self.cand_wsig])
        self.S.zero_()
        n2 = 2 * self.n_t
        nb_ = 2 * self.band_t + 1
        wp.launch(k_att_cov, dim=self.n_plans * self.n_t * nb_, device=d,
                  inputs=[self.node_mean, self.node_dvar, self.cand_iy, self.cand_ix,
                          self.cand_wsig, self.node_wgt, c.rho1, self.row_w,
                          self.n_t, self.band_t, c.k_max],
                  outputs=[self.S, self.mx])
        wp.launch(k_att_finish, dim=self.n_plans, device=d,
                  inputs=[self.S, self.mx, n2],
                  outputs=[self.e_l, self.var_l])
        return self.e_l, self.var_l
