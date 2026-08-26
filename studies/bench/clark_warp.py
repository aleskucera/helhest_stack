"""The contact estimator as Warp GPU kernels: everything device-resident, no host sync.

WHY. The NumPy implementation is ~638 interpreter dispatches per plan on arrays of a few
thousand elements. Its arithmetic for 512 plans is ~300 MFLOP -- about 0.15 ms of GPU work --
while it takes ~215 ms. The method is not compute-bound; the prototype is dispatch-bound. This
module removes the interpreter from the loop.

WHAT IS PORTED. The convolution path (`clark_conv.plan_moments_conv`), not the universe-basis
one. That choice is forced: the universe path needs `np.unique` over the candidate cells to
build its basis, which is a sort with no good in-kernel analogue, while the convolution path
scatters onto a grid and needs no basis at all. The two are equivalent to 5e-7 relative on the
cylinder (`clark_blend_invariants.py`).

NO HOST SYNCHRONISATION. Every intermediate stays in device memory and every launch is
asynchronous:
  * the per-bin structuring-element tables are built ONCE at construction and uploaded;
  * the accumulation grid is the FULL map, so no bounds reduction (which would need a readback)
    is ever computed -- for a 90x90 map and 512 plans that is 17 MB, and the convolution cost is
    linear in it;
  * the fold, the scatter and E[J] are a single kernel, so no intermediate node array is
    written out;
  * the outputs are device arrays. Nothing calls `.numpy()`.

THE ONE APPROXIMATION THAT DIFFERS FROM NOTHING. `_erf` here is Abramowitz & Stegun 7.1.26,
byte-for-byte the same rational approximation `clark.py` uses -- deliberately NOT `wp.erf`, so
that a comparison against the NumPy path isolates float32-vs-float64 rather than mixing in a
different erf.

LIMITATION, stated rather than hidden: `MAX_K` is a compile-time constant sized for the
deployed cylinder (K = 5-7, padded to 7). The K = 37 sphere needs this module rebuilt with
MAX_K = 40; the kernels are otherwise unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

MAX_K = 8  # compile-time candidate bound; deployed cylinder is K = 5-7
VECF = wp.types.vector(length=MAX_K, dtype=wp.float32)
VECI = wp.types.vector(length=MAX_K, dtype=wp.int32)


@wp.func
def _erf(x: float) -> float:
    """Abramowitz & Stegun 7.1.26, max abs error 1.5e-7 -- the same one clark.py uses."""
    s = 1.0
    if x < 0.0:
        s = -1.0
    ax = wp.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((1.061405429 * t + -1.453152027) * t) + 1.421413741) * t + -0.284496736) * t
               + 0.254829592) * t * wp.exp(-ax * ax)
    return s * y


@wp.func
def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + _erf(x * 0.7071067811865476))


@wp.func
def _npdf(x: float) -> float:
    return wp.exp(-0.5 * x * x) * 0.3989422804014327


@wp.func
def _r1(rho1: wp.array(dtype=wp.float32), lag: int) -> float:
    a = wp.abs(lag)
    if a >= rho1.shape[0]:
        return 0.0
    return rho1[a]


@wp.kernel
def k_nodes(
    ctrl: wp.array3d(dtype=wp.float32),      # [T+1, P, 3]
    wheel_xy: wp.array2d(dtype=wp.float32),  # [3, 2]
    c_settle: wp.array(dtype=wp.float32),    # [3]
    origin_x: float, origin_y: float, cell: float,
    ny: int, nx: int, n_bins: int, n_t: int,
    anchor_y: wp.array(dtype=wp.int32),
    anchor_x: wp.array(dtype=wp.int32),
    node_bin: wp.array(dtype=wp.int32),
    node_ceff: wp.array(dtype=wp.float32),
):
    """One thread per (plan, wheel, timestep, stencil corner).

    Fuses the wheel-position rollout, the engine's bilinear stencil (`_locate`'s cell-CENTRE
    convention) and its yaw binning ([0, PI), `engine/step.py::yaw_bin`)."""
    n = wp.tid()
    corner = n % 4
    r = n / 4
    t = r % n_t
    r = r / n_t
    w = r % 3
    p = r / 3

    x = ctrl[t + 1, p, 0]
    y = ctrl[t + 1, p, 1]
    yaw = ctrl[t + 1, p, 2]
    cs = wp.cos(yaw)
    sn = wp.sin(yaw)
    wx = x + wheel_xy[w, 0] * cs - wheel_xy[w, 1] * sn
    wy = y + wheel_xy[w, 0] * sn + wheel_xy[w, 1] * cs

    fx = (wx - origin_x) / cell - 0.5
    fy = (wy - origin_y) / cell - 0.5
    xi = wp.clamp(int(wp.floor(fx)), 0, nx - 2)
    yi = wp.clamp(int(wp.floor(fy)), 0, ny - 2)
    tx = wp.clamp(fx - float(xi), 0.0, 1.0)
    ty = wp.clamp(fy - float(yi), 0.0, 1.0)

    dx = 0
    dy = 0
    wgt = (1.0 - tx) * (1.0 - ty)
    if corner == 1:
        dx = 1
        wgt = tx * (1.0 - ty)
    if corner == 2:
        dy = 1
        wgt = (1.0 - tx) * ty
    if corner == 3:
        dx = 1
        dy = 1
        wgt = tx * ty

    kb = int(wp.floor(yaw / (3.14159265358979 / float(n_bins)) + 0.5))
    b = ((kb % n_bins) + n_bins) % n_bins

    anchor_y[n] = yi + dy
    anchor_x[n] = xi + dx
    node_bin[n] = b
    node_ceff[n] = c_settle[w] * wgt


@wp.kernel
def k_fold_scatter(
    belief: wp.array2d(dtype=wp.float32),
    sigma: wp.array2d(dtype=wp.float32),
    off_dy: wp.array2d(dtype=wp.int32),      # [n_bins, K]
    off_dx: wp.array2d(dtype=wp.int32),
    off_cap: wp.array2d(dtype=wp.float32),
    k_real: wp.array(dtype=wp.int32),        # [n_bins] real candidates before padding
    rho1: wp.array(dtype=wp.float32),
    anchor_y: wp.array(dtype=wp.int32),
    anchor_x: wp.array(dtype=wp.int32),
    node_bin: wp.array(dtype=wp.int32),
    node_ceff: wp.array(dtype=wp.float32),
    ny: int, nx: int, n_t: int,
    e_j: wp.array(dtype=wp.float32),
    field: wp.array3d(dtype=wp.float32),     # [P, ny, nx]
):
    """One thread per node: gather, sort, Clark-fold, then scatter -- all in registers.

    Clark's fold is sequential in K but independent across nodes, so a thread per node is the
    natural shape: K is at most 7 here, the candidates live in registers, and nothing is written
    to global memory until the final scatter. E[J] and the sensitivity field are accumulated
    with atomics, so no per-node intermediate array exists at all."""
    n = wp.tid()
    p = n / (4 * 3 * n_t)
    b = node_bin[n]
    ay = anchor_y[n]
    ax = anchor_x[n]
    ce = node_ceff[n]
    kk = k_real[b]

    m = VECF()
    sg = VECF()
    ody = VECI()
    odx = VECI()
    ci = VECI()
    cj = VECI()

    for j in range(kk):
        iy = wp.clamp(ay + off_dy[b, j], 0, ny - 1)
        ix = wp.clamp(ax + off_dx[b, j], 0, nx - 1)
        m[j] = belief[iy, ix] + off_cap[b, j]
        sg[j] = sigma[iy, ix]
        ody[j] = off_dy[b, j]
        odx[j] = off_dx[b, j]
        ci[j] = iy
        cj[j] = ix

    # insertion sort, mean DESCENDING -- the order clark_conv.fold_weights uses
    for a in range(1, kk):
        km = m[a]
        ks = sg[a]
        kdy = ody[a]
        kdx = odx[a]
        kci = ci[a]
        kcj = cj[a]
        q = a - 1
        while q >= 0 and m[q] < km:
            m[q + 1] = m[q]
            sg[q + 1] = sg[q]
            ody[q + 1] = ody[q]
            odx[q + 1] = odx[q]
            ci[q + 1] = ci[q]
            cj[q + 1] = cj[q]
            q = q - 1
        m[q + 1] = km
        sg[q + 1] = ks
        ody[q + 1] = kdy
        odx[q + 1] = kdx
        ci[q + 1] = kci
        cj[q + 1] = kcj

    # SHIFT TO THE LARGEST MEAN before folding. The recursion forms
    # Var = E[X^2] - E[X]^2, and with contact heights around 20 m both terms are ~400 while
    # their difference is ~0.005 -- five digits of cancellation, and float32 has seven. The
    # fold is shift-EQUIVARIANT in the mean and shift-INVARIANT in the variance (a and alpha
    # depend only on differences), so folding about m[0] and adding it back at the end is
    # exact, and it moves the cancellation out of the float32 danger zone entirely.
    base = m[0]
    for j in range(kk):
        m[j] = m[j] - base

    mean_run = m[0]
    var_run = sg[0] * sg[0]
    cov = VECF()
    for j in range(kk):
        cov[j] = _r1(rho1, ody[0] - ody[j]) * _r1(rho1, odx[0] - odx[j]) * sg[0] * sg[j]

    phi_s = VECF()
    phin_s = VECF()
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
        ph = _ncdf(alpha)
        pn = 1.0 - ph
        pd = _npdf(alpha)
        new_mean = mean_run * ph + m[i] * pn + aa * pd
        new_ex2 = ((mean_run * mean_run + var_run) * ph + (m[i] * m[i] + v2) * pn
                   + (mean_run + m[i]) * aa * pd)
        var_run = wp.max(new_ex2 - new_mean * new_mean, 0.0)
        for j in range(kk):
            cov[j] = (cov[j] * ph
                      + _r1(rho1, ody[i] - ody[j]) * _r1(rho1, odx[i] - odx[j])
                      * sg[i] * sg[j] * pn)
        phi_s[i - 1] = ph
        phin_s[i - 1] = pn
        mean_run = new_mean

    # backward pass: the candidate weights, same recursion as fold_weights
    wv = VECF()
    tail = float(1.0)
    for i in range(kk - 1, 0, -1):
        wv[i] = phin_s[i - 1] * tail
        tail = tail * phi_s[i - 1]
    wv[0] = tail

    wp.atomic_add(e_j, p, ce * (mean_run + base))
    for j in range(kk):
        wp.atomic_add(field, p, ci[j], cj[j], ce * wv[j] * sg[j])


@wp.kernel
def k_conv_x(src: wp.array3d(dtype=wp.float32), ker: wp.array(dtype=wp.float32),
             dst: wp.array3d(dtype=wp.float32)):
    p, y, x = wp.tid()
    L = ker.shape[0] / 2
    acc = float(0.0)
    for t in range(ker.shape[0]):
        xx = x + t - L
        if xx >= 0 and xx < src.shape[2]:
            acc += src[p, y, xx] * ker[t]
    dst[p, y, x] = acc


@wp.kernel
def k_conv_y(src: wp.array3d(dtype=wp.float32), ker: wp.array(dtype=wp.float32),
             dst: wp.array3d(dtype=wp.float32)):
    p, y, x = wp.tid()
    L = ker.shape[0] / 2
    acc = float(0.0)
    for t in range(ker.shape[0]):
        yy = y + t - L
        if yy >= 0 and yy < src.shape[1]:
            acc += src[p, yy, x] * ker[t]
    dst[p, y, x] = acc


@wp.kernel
def k_quad(field: wp.array3d(dtype=wp.float32), tmp: wp.array3d(dtype=wp.float32),
           var_j: wp.array(dtype=wp.float32)):
    p, y, x = wp.tid()
    v = field[p, y, x] * tmp[p, y, x]
    if v != 0.0:
        wp.atomic_add(var_j, p, v)


@wp.kernel
def k_finish(e_j: wp.array(dtype=wp.float32), var_j: wp.array(dtype=wp.float32), const: float):
    p = wp.tid()
    e_j[p] = e_j[p] + const
    var_j[p] = wp.max(var_j[p], 0.0)


class WarpContactEstimator:
    """Device-resident estimator. Construct once per (map size, plan count, horizon)."""

    def __init__(self, ny, nx, cell, origin_x, origin_y, rp, rho1, n_plans, n_t,
                 device="cuda:0", n_bins=32, element="cylinder"):
        from .element import N_YAW_BINS, WHEEL_HALF_WIDTH, _cyl_table
        from ..adjoint.harness import DERIV_WZ, DERIV_WPITCH, DERIV_WROLL
        from .clark import settle_map

        self.device = device
        self.ny, self.nx, self.cell = ny, nx, cell
        self.origin_x, self.origin_y = origin_x, origin_y
        self.n_plans, self.n_t, self.n_bins = n_plans, n_t, n_bins
        self.n_nodes = n_plans * 3 * n_t * 4
        if element != "cylinder":
            raise ValueError("MAX_K is sized for the cylinder; see module docstring")

        # --- structuring-element tables, built ONCE on the host and uploaded ----------------
        dy = np.zeros((n_bins, MAX_K), np.int32)
        dx = np.zeros((n_bins, MAX_K), np.int32)
        cap = np.zeros((n_bins, MAX_K), np.float32)
        kr = np.zeros(n_bins, np.int32)
        for b in range(n_bins):
            a_, b_, c_ = _cyl_table(cell, rp.wheel_radius, WHEEL_HALF_WIDTH, b, n_bins)
            k = len(a_)
            if k > MAX_K:
                raise ValueError(f"K={k} exceeds MAX_K={MAX_K}")
            dy[b, :k], dx[b, :k], cap[b, :k], kr[b] = a_, b_, c_, k
        with wp.ScopedDevice(device):
            self.off_dy = wp.array(dy, dtype=wp.int32)
            self.off_dx = wp.array(dx, dtype=wp.int32)
            self.off_cap = wp.array(cap, dtype=wp.float32)
            self.k_real = wp.array(kr, dtype=wp.int32)
            self.rho1 = wp.array(np.asarray(rho1, np.float32), dtype=wp.float32)
            ker = np.concatenate([rho1[:0:-1], rho1]).astype(np.float32)
            self.ker = wp.array(ker, dtype=wp.float32)
            wxy = np.array([[0.0, rp.half_track], [0.0, -rp.half_track],
                            [-rp.rear_offset, 0.0]], np.float32)
            self.wheel_xy = wp.array(wxy, dtype=wp.float32)
            cw = (np.array([DERIV_WZ, DERIV_WPITCH, DERIV_WROLL]) @ settle_map(rp))
            self.c_settle = wp.array(cw.astype(np.float32), dtype=wp.float32)
            # --- persistent scratch: allocated once, reused every call ---------------------
            self.anchor_y = wp.zeros(self.n_nodes, dtype=wp.int32)
            self.anchor_x = wp.zeros(self.n_nodes, dtype=wp.int32)
            self.node_bin = wp.zeros(self.n_nodes, dtype=wp.int32)
            self.node_ceff = wp.zeros(self.n_nodes, dtype=wp.float32)
            self.field = wp.zeros((n_plans, ny, nx), dtype=wp.float32)
            self.tmp = wp.zeros((n_plans, ny, nx), dtype=wp.float32)
            self.tmp2 = wp.zeros((n_plans, ny, nx), dtype=wp.float32)
            self.e_j = wp.zeros(n_plans, dtype=wp.float32)
            self.var_j = wp.zeros(n_plans, dtype=wp.float32)
        self.const = float(n_t * DERIV_WZ * rp.wheel_radius)

    def moments(self, belief, sigma, ctrl):
        """E[J] and Var[J] for every plan. All arguments and both results are DEVICE arrays;
        this issues only asynchronous launches and never reads back."""
        d = self.device
        self.field.zero_()
        self.e_j.zero_()
        self.var_j.zero_()
        wp.launch(k_nodes, dim=self.n_nodes, device=d,
                  inputs=[ctrl, self.wheel_xy, self.c_settle, self.origin_x, self.origin_y,
                          self.cell, self.ny, self.nx, self.n_bins, self.n_t],
                  outputs=[self.anchor_y, self.anchor_x, self.node_bin, self.node_ceff])
        wp.launch(k_fold_scatter, dim=self.n_nodes, device=d,
                  inputs=[belief, sigma, self.off_dy, self.off_dx, self.off_cap, self.k_real,
                          self.rho1, self.anchor_y, self.anchor_x, self.node_bin,
                          self.node_ceff, self.ny, self.nx, self.n_t],
                  outputs=[self.e_j, self.field])
        g = (self.n_plans, self.ny, self.nx)
        wp.launch(k_conv_x, dim=g, device=d, inputs=[self.field, self.ker], outputs=[self.tmp])
        wp.launch(k_conv_y, dim=g, device=d, inputs=[self.tmp, self.ker], outputs=[self.tmp2])
        wp.launch(k_quad, dim=g, device=d, inputs=[self.field, self.tmp2], outputs=[self.var_j])
        wp.launch(k_finish, dim=self.n_plans, device=d,
                  inputs=[self.e_j, self.var_j, self.const])
        return self.e_j, self.var_j
