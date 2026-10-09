"""Batched path QPs on the GPU: one thread per QP, OSQP's ADMM on a banded structure.

The path QP (control/qp_path.py, spline_terms) always has the same shape:

    minimise   1/2 x^T P x + q^T x
    subject to lo[r] <= sum_t a[r, t] x[col[r] + t] <= hi[r]     t = 0 .. 3

x are the control points of a cubic B-spline, so P is banded (x[i] couples to x[i +- 3] at most)
and every constraint row touches 4 neighbouring control points. Every matrix ADMM needs is then
banded too -- K = P + sigma I + A^T diag(rho) A has bandwidth 3 -- and one thread can assemble and
factor it (O(n)) and solve with it (O(n)) per iteration in its own scratch rows. That is the point
of solving here: a batch of corridors (left or right of each obstacle, a few margins) costs one
launch, where OSQP on the host solves them one by one and reads every corridor back.

The iteration is OSQP's (Stellato et al., 2020): relaxed ADMM with alpha, a per-row rho (x1e3 on
equality rows), adaptive rho with a refactorisation when it moves by more than 5x, the same
termination test, the same primal-infeasibility certificate, and its polish -- an equality-
constrained solve on the active set ADMM found, kept only if it satisfies every bound. Not done:
OSQP's Ruiz scaling (spline_terms scales the cost and each row itself).

Float64 throughout. The problem is well-conditioned since the spline replaced one unknown per
sample, but the polish solve is stiff by construction (a penalty on the active rows).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import warp as wp

SOLVED = wp.constant(0)
MAX_ITER = wp.constant(1)
PRIMAL_INFEASIBLE = wp.constant(2)

BAND = 4  # P[i, i + d] stored for d = 0 .. 3; also the width of a constraint row

f64 = wp.float64


@dataclass(frozen=True)
class AdmmParams:
    rho: float = 0.1  # OSQP's default
    sigma: float = 1e-6
    alpha: float = 1.6  # over-relaxation
    # 1e-6 as qp_path's OSQP: at 1e-4 the active set is not settled and the polish misses it
    eps_abs: float = 1e-6
    eps_rel: float = 1e-6
    eps_pinf: float = 1e-4
    max_iter: int = 4000
    check_every: int = 25  # termination, infeasibility and rho are looked at this often
    adaptive_rho: bool = True
    polish: bool = True
    # the polish: penalty on active rows and proximal weight, in the scaled cost's units (its
    # largest diagonal is 1), and how far outside a bound its result may land and still be kept
    polish_penalty: float = 1e6
    polish_delta: float = 1e-9
    polish_tol: float = 1e-5


def bands_to_csc(p_band: np.ndarray, n: int) -> sp.csc_matrix:
    """The upper bands P[i, i + d] ([n, BAND]) back to a symmetric sparse P, for OSQP."""
    upper = sp.diags([p_band[: n - d, d] for d in range(BAND)], list(range(BAND)), shape=(n, n))
    return (upper + sp.triu(upper, 1).T).tocsc()


def rows_to_csc(a_coef: np.ndarray, a_col: np.ndarray, n: int) -> sp.csc_matrix:
    """Width-4 constraint rows ([M, BAND] coefficients from column a_col) to a sparse A."""
    m = len(a_col)
    rows = np.repeat(np.arange(m), BAND)
    cols = (a_col[:, None] + np.arange(BAND)).ravel()
    return sp.csc_matrix((a_coef.ravel(), (rows, cols)), shape=(m, n))


@wp.func
def _a_row(
    x: wp.array2d(dtype=wp.float64),
    a_coef: wp.array3d(dtype=wp.float64),
    a_col: wp.array2d(dtype=wp.int32),
    b: int,
    row: int,
) -> wp.float64:
    """(A x)[row]."""
    c = a_col[b, row]
    s = f64(0.0)
    for t in range(4):
        s = s + a_coef[b, row, t] * x[b, c + t]
    return s


@wp.func
def _at_scatter(
    v: wp.array2d(dtype=wp.float64),
    scale: wp.array2d(dtype=wp.float64),
    use_scale: int,
    sub: wp.array2d(dtype=wp.float64),
    use_sub: int,
    a_coef: wp.array3d(dtype=wp.float64),
    a_col: wp.array2d(dtype=wp.int32),
    b: int,
    m: int,
    out: wp.array2d(dtype=wp.float64),
):
    """out += A^T w with w = scale * v - sub (each part optional): rho z - y, y, or a y step."""
    for row in range(m):
        a = f64(v[b, row])
        if use_scale == 1:
            a = a * scale[b, row]
        if use_sub == 1:
            a = a - sub[b, row]
        c = a_col[b, row]
        for t in range(4):
            out[b, c + t] = out[b, c + t] + a_coef[b, row, t] * a


@wp.func
def _p_row(
    p_band: wp.array3d(dtype=wp.float64), x: wp.array2d(dtype=wp.float64), b: int, n: int, i: int
) -> wp.float64:
    """(P x)[i]."""
    s = f64(p_band[b, i, 0] * x[b, i])
    for d in range(1, 4):
        if i + d < n:
            s = s + p_band[b, i, d] * x[b, i + d]
        if i - d >= 0:
            s = s + p_band[b, i - d, d] * x[b, i - d]
    return s


@wp.func
def _factor(
    p_band: wp.array3d(dtype=wp.float64),
    a_coef: wp.array3d(dtype=wp.float64),
    a_col: wp.array2d(dtype=wp.int32),
    rho: wp.array2d(dtype=wp.float64),
    b: int,
    n: int,
    m: int,
    sigma: wp.float64,
    kb: wp.array3d(dtype=wp.float64),
    lb: wp.array3d(dtype=wp.float64),
):
    """Assemble K = P + sigma I + A^T diag(rho) A as upper bands kb[b, i, d] = K[i, i + d], then
    factor it, K = L L^T, stored as lb[b, i, d] = L[i, i - d] for d > 0 and 1 / L[i, i] at d = 0."""
    for i in range(n):
        for d in range(4):
            kb[b, i, d] = p_band[b, i, d]
        kb[b, i, 0] = kb[b, i, 0] + sigma
    for row in range(m):
        c = a_col[b, row]
        r = rho[b, row]
        for t1 in range(4):
            for t2 in range(t1, 4):
                kb[b, c + t1, t2 - t1] = (
                    kb[b, c + t1, t2 - t1] + r * a_coef[b, row, t1] * a_coef[b, row, t2]
                )
    for i in range(n):
        for j in range(wp.max(0, i - 3), i + 1):
            s = f64(kb[b, j, i - j])
            for k in range(wp.max(0, i - 3), j):
                s = s - lb[b, i, i - k] * lb[b, j, j - k]
            if i == j:
                # the reciprocal pivot: the solves then multiply. float64 division is slow on
                # consumer GPUs and the Jetson; this is the only one left per row and factor
                lb[b, i, 0] = f64(1.0) / wp.sqrt(wp.max(s, f64(1.0e-30)))
            else:
                lb[b, i, i - j] = s * lb[b, j, 0]


@wp.func
def _back_solve(lb: wp.array3d(dtype=wp.float64), w: wp.array2d(dtype=wp.float64), b: int, n: int):
    """w <- K^-1 w, in place."""
    for i in range(n):
        s = f64(w[b, i])
        for d in range(1, 4):
            if i - d >= 0:
                s = s - lb[b, i, d] * w[b, i - d]
        w[b, i] = s * lb[b, i, 0]
    for t in range(n):
        i = n - 1 - t
        s = f64(w[b, i])
        for d in range(1, 4):
            if i + d < n:
                s = s - lb[b, i + d, d] * w[b, i + d]
        w[b, i] = s * lb[b, i, 0]


@wp.func
def _polish(
    p_band: wp.array3d(dtype=wp.float64),
    q: wp.array2d(dtype=wp.float64),
    a_coef: wp.array3d(dtype=wp.float64),
    a_col: wp.array2d(dtype=wp.int32),
    lo: wp.array2d(dtype=wp.float64),
    hi: wp.array2d(dtype=wp.float64),
    b: int,
    n: int,
    m: int,
    penalty: wp.float64,
    delta: wp.float64,
    tol: wp.float64,
    kb: wp.array3d(dtype=wp.float64),
    lb: wp.array3d(dtype=wp.float64),
    rho: wp.array2d(dtype=wp.float64),
    z: wp.array2d(dtype=wp.float64),
    y: wp.array2d(dtype=wp.float64),
    bound: wp.array2d(dtype=wp.float64),
    w: wp.array2d(dtype=wp.float64),
    xb: wp.array2d(dtype=wp.float64),
    x: wp.array2d(dtype=wp.float64),
) -> int:
    """OSQP's polish, banded: guess the active set from ADMM's (z, y), solve the QP with those
    rows held at their bound (a stiff penalty plus a few proximal steps), keep the result if it
    satisfies every bound. ADMM alone stops where the cost is flat to its tolerance. 1 = kept."""
    for row in range(m):
        active = f64(0.0)
        value = f64(0.0)
        if hi[b, row] - lo[b, row] < f64(1.0e-9) or z[b, row] - lo[b, row] < -y[b, row]:
            active = f64(1.0)
            value = lo[b, row]
        elif hi[b, row] - z[b, row] < y[b, row]:
            active = f64(1.0)
            value = hi[b, row]
        bound[b, row] = value
        rho[b, row] = active * penalty
    _factor(p_band, a_coef, a_col, rho, b, n, m, delta, kb, lb)
    for i in range(n):
        xb[b, i] = x[b, i]
    for it in range(4):
        for i in range(n):
            w[b, i] = delta * x[b, i] - q[b, i]
        _at_scatter(bound, rho, 1, bound, 0, a_coef, a_col, b, m, w)
        _back_solve(lb, w, b, n)
        for i in range(n):
            x[b, i] = w[b, i]
    worst = f64(0.0)
    for row in range(m):
        ax = _a_row(x, a_coef, a_col, b, row)
        worst = wp.max(worst, wp.max(lo[b, row] - ax, ax - hi[b, row]))
    if worst > tol:
        for i in range(n):
            x[b, i] = xb[b, i]
        return 0
    return 1


@wp.kernel
def _admm_kernel(
    p_band: wp.array3d(dtype=wp.float64),  # [B, N, BAND] P[i, i + d]
    q: wp.array2d(dtype=wp.float64),  # [B, N]
    a_coef: wp.array3d(dtype=wp.float64),  # [B, M, BAND] row coefficients
    a_col: wp.array2d(dtype=wp.int32),  # [B, M] first column of each row
    lo: wp.array2d(dtype=wp.float64),  # [B, M]
    hi: wp.array2d(dtype=wp.float64),
    sizes: wp.array2d(dtype=wp.int32),  # [B, 2] (n, m) of each problem; n = 0 skips it
    rho0: wp.float64,
    sigma: wp.float64,
    alpha: wp.float64,
    eps_abs: wp.float64,
    eps_rel: wp.float64,
    eps_pinf: wp.float64,
    max_iter: int,
    check_every: int,
    adaptive: int,
    polish: int,
    polish_penalty: wp.float64,
    polish_delta: wp.float64,
    polish_tol: wp.float64,
    warm: int,
    kb: wp.array3d(dtype=wp.float64),  # scratch: K's bands
    lb: wp.array3d(dtype=wp.float64),  # scratch: its factor
    rho: wp.array2d(dtype=wp.float64),  # scratch: per-row rho
    rho_inv: wp.array2d(dtype=wp.float64),  # scratch: its reciprocal
    z: wp.array2d(dtype=wp.float64),  # kept between solves (warm start)
    y: wp.array2d(dtype=wp.float64),  # kept
    y_prev: wp.array2d(dtype=wp.float64),
    w: wp.array2d(dtype=wp.float64),  # scratch: x tilde
    aty: wp.array2d(dtype=wp.float64),  # scratch: A^T y
    xb: wp.array2d(dtype=wp.float64),  # scratch: ADMM's x, kept if the polish fails
    rho_last: wp.array(dtype=wp.float64),  # kept: each slot's rho at its last solve
    last_size: wp.array2d(dtype=wp.int32),  # kept: each slot's (n, m) at its last solve
    x: wp.array2d(dtype=wp.float64),  # out, and kept
    status: wp.array(dtype=wp.int32),  # out
    iters: wp.array(dtype=wp.int32),  # out
    polished: wp.array(dtype=wp.int32),  # out: 1 = the polish was kept
):
    b = wp.tid()
    n = sizes[b, 0]
    m = sizes[b, 1]
    polished[b] = 0
    if n < 4:
        status[b] = PRIMAL_INFEASIBLE
        iters[b] = 0
        return
    # warm: start from this slot's last solution and rho -- the previous frame's QP, which differs
    # by the robot's few centimetres of travel. Rows or unknowns it did not have start at zero.
    rho_s = f64(rho0)
    m_old = int(0)
    n_old = int(0)
    if warm == 1 and rho_last[b] > f64(0.0):
        rho_s = rho_last[b]
        n_old = last_size[b, 0]
        m_old = last_size[b, 1]
    for row in range(m):
        rho[b, row] = rho_s
        if hi[b, row] - lo[b, row] < f64(1.0e-9):
            rho[b, row] = rho_s * f64(1.0e3)  # an equality row, as OSQP
        rho_inv[b, row] = f64(1.0) / rho[b, row]
        if row >= m_old:
            z[b, row] = f64(0.0)
            y[b, row] = f64(0.0)
    for i in range(n_old, n):
        x[b, i] = f64(0.0)
    _factor(p_band, a_coef, a_col, rho, b, n, m, sigma, kb, lb)

    result = int(MAX_ITER)
    done = int(max_iter)
    for k in range(max_iter):
        check = (k + 1) % check_every == 0
        # x tilde = K^-1 (sigma x - q + A^T (rho z - y))
        for i in range(n):
            w[b, i] = sigma * x[b, i] - q[b, i]
        _at_scatter(z, rho, 1, y, 1, a_coef, a_col, b, m, w)
        _back_solve(lb, w, b, n)
        for row in range(m):
            zr = alpha * _a_row(w, a_coef, a_col, b, row) + (f64(1.0) - alpha) * z[b, row]
            zn = wp.clamp(zr + y[b, row] * rho_inv[b, row], lo[b, row], hi[b, row])
            if check:
                y_prev[b, row] = y[b, row]
            y[b, row] = y[b, row] + rho[b, row] * (zr - zn)
            z[b, row] = zn
        for i in range(n):
            x[b, i] = alpha * w[b, i] + (f64(1.0) - alpha) * x[b, i]
        if check:
            prim = f64(0.0)
            ax_n = f64(0.0)
            z_n = f64(0.0)
            for row in range(m):
                ax = _a_row(x, a_coef, a_col, b, row)
                prim = wp.max(prim, wp.abs(ax - z[b, row]))
                ax_n = wp.max(ax_n, wp.abs(ax))
                z_n = wp.max(z_n, wp.abs(z[b, row]))
            for i in range(n):
                aty[b, i] = f64(0.0)
            _at_scatter(y, rho, 0, y, 0, a_coef, a_col, b, m, aty)
            dual = f64(0.0)
            px_n = f64(0.0)
            aty_n = f64(0.0)
            q_n = f64(0.0)
            for i in range(n):
                px = _p_row(p_band, x, b, n, i)
                dual = wp.max(dual, wp.abs(px + q[b, i] + aty[b, i]))
                px_n = wp.max(px_n, wp.abs(px))
                aty_n = wp.max(aty_n, wp.abs(aty[b, i]))
                q_n = wp.max(q_n, wp.abs(q[b, i]))
            prim_scale = wp.max(ax_n, z_n)
            dual_scale = wp.max(wp.max(px_n, aty_n), q_n)
            if prim <= eps_abs + eps_rel * prim_scale and dual <= eps_abs + eps_rel * dual_scale:
                result = SOLVED
                done = k + 1
                break
            # primal infeasibility: a y step that A^T maps to ~0 and the bounds price negative
            dy_n = f64(0.0)
            support = f64(0.0)
            for row in range(m):
                dy = y[b, row] - y_prev[b, row]
                dy_n = wp.max(dy_n, wp.abs(dy))
                support = (
                    support + hi[b, row] * wp.max(dy, f64(0.0)) + lo[b, row] * wp.min(dy, f64(0.0))
                )
            if dy_n > f64(1.0e-10):
                for i in range(n):
                    aty[b, i] = f64(0.0)
                _at_scatter(y, rho, 0, y_prev, 1, a_coef, a_col, b, m, aty)
                atdy_n = f64(0.0)
                for i in range(n):
                    atdy_n = wp.max(atdy_n, wp.abs(aty[b, i]))
                if atdy_n <= eps_pinf * dy_n and support <= -eps_pinf * dy_n:
                    result = PRIMAL_INFEASIBLE
                    done = k + 1
                    break
            if adaptive == 1:
                num = prim / wp.max(prim_scale, f64(1.0e-10))
                den = dual / wp.max(dual_scale, f64(1.0e-10))
                rho_new = wp.clamp(
                    rho_s * wp.sqrt(num / wp.max(den, f64(1.0e-10))), f64(1.0e-6), f64(1.0e6)
                )
                if rho_new > f64(5.0) * rho_s or rho_new < f64(0.2) * rho_s:
                    ratio = rho_new / rho_s
                    for row in range(m):
                        rho[b, row] = rho[b, row] * ratio
                        rho_inv[b, row] = f64(1.0) / rho[b, row]
                    rho_s = rho_new
                    _factor(p_band, a_coef, a_col, rho, b, n, m, sigma, kb, lb)
    if result == SOLVED and polish == 1:
        # y_prev holds the active rows' bound values from here on
        polished[b] = _polish(
            p_band,
            q,
            a_coef,
            a_col,
            lo,
            hi,
            b,
            n,
            m,
            polish_penalty,
            polish_delta,
            polish_tol,
            kb,
            lb,
            rho,
            z,
            y,
            y_prev,
            w,
            xb,
            x,
        )
    status[b] = result
    iters[b] = done
    rho_last[b] = rho_s
    last_size[b, 0] = n
    last_size[b, 1] = m


class BatchQp:
    """Up to `batch` path QPs of up to `n_max` unknowns and `m_max` rows, solved in one launch.

    Inputs and outputs stay on the device: a caller that builds its problems on the device (the
    corridors already are) never touches the host."""

    def __init__(
        self,
        batch: int,
        n_max: int,
        m_max: int,
        params: AdmmParams = AdmmParams(),
        device: str | None = None,
    ) -> None:
        self.p = params
        self.batch, self.n_max, self.m_max = int(batch), int(n_max), int(m_max)
        self.device = device

        def zeros(*shape: int) -> wp.array:
            return wp.zeros(shape, dtype=wp.float64, device=device)

        self.kb = zeros(self.batch, self.n_max, BAND)
        self.lb = zeros(self.batch, self.n_max, BAND)
        self.rho = zeros(self.batch, self.m_max)
        self.rho_inv = zeros(self.batch, self.m_max)
        self.rho_last = wp.zeros(self.batch, dtype=wp.float64, device=device)
        self.last_size = wp.zeros((self.batch, 2), dtype=wp.int32, device=device)
        self.z = zeros(self.batch, self.m_max)
        self.y = zeros(self.batch, self.m_max)
        self.y_prev = zeros(self.batch, self.m_max)
        self.w = zeros(self.batch, self.n_max)
        self.aty = zeros(self.batch, self.n_max)
        self.xb = zeros(self.batch, self.n_max)
        self.x = zeros(self.batch, self.n_max)
        self.status = wp.zeros(self.batch, dtype=wp.int32, device=device)
        self.iters = wp.zeros(self.batch, dtype=wp.int32, device=device)
        self.polished = wp.zeros(self.batch, dtype=wp.int32, device=device)

    def solve(
        self,
        p_band: wp.array,
        q: wp.array,
        a_coef: wp.array,
        a_col: wp.array,
        lo: wp.array,
        hi: wp.array,
        sizes: wp.array,
        warm: bool = False,
    ) -> tuple[wp.array, wp.array, wp.array]:
        """[B, N, BAND], [B, N], [B, M, BAND], [B, M], [B, M], [B, M], [B, 2] (n, m)
        -> (x [B, N], status [B], iterations [B]). `warm` starts each slot from its own last
        solution: slot b should then hold the same corridor as last frame."""
        p = self.p
        wp.launch(
            _admm_kernel,
            dim=self.batch,
            inputs=[
                p_band,
                q,
                a_coef,
                a_col,
                lo,
                hi,
                sizes,
                float(p.rho),
                float(p.sigma),
                float(p.alpha),
                float(p.eps_abs),
                float(p.eps_rel),
                float(p.eps_pinf),
                int(p.max_iter),
                int(p.check_every),
                int(p.adaptive_rho),
                int(p.polish),
                float(p.polish_penalty),
                float(p.polish_delta),
                float(p.polish_tol),
                int(warm),
                self.kb,
                self.lb,
                self.rho,
                self.rho_inv,
                self.z,
                self.y,
                self.y_prev,
                self.w,
                self.aty,
                self.xb,
                self.rho_last,
                self.last_size,
            ],
            outputs=[self.x, self.status, self.iters, self.polished],
            device=self.device,
        )
        return self.x, self.status, self.iters
