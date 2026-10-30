"""Neural planner: MPPI's replan done by gradient descent through a learned terrain model.

The model (BundleSim, experiments/16_outcome_model, `terrain_model_196608_v5`) maps the robot's state, a 16 x 16
heightmap patch around it and a 1 s wheel-speed command to a mixture of 4 Gaussians over where the robot is one
second later. It was trained on Ostrich rollouts of Helhest Junior on random ridged terrain with 105 Nm wheels.
A plan is 6 such commands (6 s). Each re-plan:

  shoot   256 random plans (half uniform, half the warm start + noise), scored with 4 model particles each
  init    the warm start and the 7 best shots
  adam    20 Adam steps on each of the 8 plans, on the gradient of the mean plan loss over 16 particles
  pick    the plan with the lowest loss over 64 fresh particles

The plan loss is the cost-to-go at the end, its running mean, the footprint's wall clearance, P(tip), the tilt, the
effort and the command rate (BundleSim macro_terrain_worlds.plan_loss with the final config of graph_planner).

Everything runs on the device as one captured CUDA graph: Warp kernels for the features, the mixture head and the
loss with hand-written adjoints, and the MLP's GEMMs as Warp tile GEMMs (default) or through cuBLAS (torch). The kernels are BundleSim's warp_planner.py, whose loss and gradient match the torch rollout to float32
(cosine >= 0.9999999). The MLP runs in float32, the geometry and the loss in float64.

NeuralMppi subclasses MppiGpu so the node keeps every setter (set_lattice, set_reference, update_clearance, ...);
only replan() changes. It reads the frame's maps from them, resampled onto the model's 0.125 m grid in the MPPI
window's frame:
  H     the MPPI window's elevation (sim.elevation): the model's heightmap patches
  V     the cost-to-go (set_lattice), min over heading, + W_TRACK min(d, 1 m)^2 to the reference curve (set_reference)
  dist  the wall-distance map of the clearance cost (update_clearance): the footprint clearance term
Commands stay in the model's box [0, V_MAX]: no reverse, no spin in place except through the node's turn-first.

The weights file comes from BundleSim's export_stack_planner.py; it also carries the training config, which must
match the constants below.
"""

from __future__ import annotations

import contextlib

import numpy as np
import warp as wp

from ..engine.simulator import ForwardSimulator
from .mppi import CostParams
from .mppi import MppiGpu
from .mppi import SamplingConfig

torch = None  # imported by NeuralPlanner on first use, so the node does not pay for it under MPPI

# --- the model's training config (checked against the weights file on load) -----------------------------------------

N_SEG = 6  # macro steps per plan
SEG_T = 1.0  # [s] per macro step
D_U, D_F, D_X, N_COMP, HIDDEN = 2, 268, 17, 4, 512
I_UPMIN = 16  # outcome index of the smallest up_z during the step
PATCH_N = 16
MODEL_CELL = 0.125  # [m] the grid the planner's maps are resampled onto
WHEEL_RADIUS = 0.35  # [m]
WHEEL_MAX = 4.0  # [rad/s] the model's command box is [0, WHEEL_MAX]
SIGMA = 0.05  # the command noise the planner assumes, as a fraction of WHEEL_MAX
FRONT_WHEEL_Y = 0.365  # [m] left / right wheel offset from the chassis axis
# the footprint as three discs (x, y, radius) [m] in the body frame, one per wheel
DISCS = ((0.0, 0.365, 0.37), (0.0, -0.365, 0.37), (-0.75, 0.0, 0.37))
CONFIG = dict(
    K=N_COMP, d_u=D_U, d_f=D_F, d_x=D_X, hidden=HIDDEN, i_upmin=I_UPMIN, n_seg=N_SEG, seg_t=SEG_T,
    cell=MODEL_CELL, patch_n=PATCH_N, r_w=WHEEL_RADIUS, v_max=WHEEL_MAX, sigma=SIGMA, w_run=0.5, w_c=10.0,
    cmargin=0.3, w_tip=10.0, w_tilt=1.0, w_e=0.5, w_s=1.0, tilt_scale=30.0, grad_lr=0.05, ms_inits=8,
    ms_steps=20, ms_shoot=256,
)  # fmt: skip
W_TRACK = (
    2.0  # [1/m] reference-curve term: 0.5 m off the curve costs as much as 0.5 m of cost-to-go
)

# GEMM tile (particles, outputs, inner) and block sizes; small per-particle blocks spread a few hundred particles
# over many SMs
TM, TN, TK = 32, 64, 32
_Q = int(np.lcm(TN, TK))
DI = (D_U + 1 + D_F + _Q - 1) // _Q * _Q  # 271 inputs, padded
DO = N_COMP + 2 * N_COMP * D_X  # 140 outputs: logits, means, log-scales
DOP = (DO + _Q - 1) // _Q * _Q
GEMM_BLOCK = 128
BLOCK = 32

F64 = wp.float64
CELL = wp.constant(F64(MODEL_CELL))
R_W = wp.constant(F64(WHEEL_RADIUS))
V_MAX = wp.constant(F64(WHEEL_MAX))
LOGSIG3 = wp.constant(float(np.log(SIGMA) / 3.0))
W_RUN6 = wp.constant(F64(CONFIG["w_run"] / N_SEG))
W_C6 = wp.constant(F64(CONFIG["w_c"] / N_SEG))
CMARGIN = wp.constant(F64(CONFIG["cmargin"]))
W_TIP = wp.constant(F64(CONFIG["w_tip"]))
W_TILT = wp.constant(F64(CONFIG["w_tilt"]))
W_E = wp.constant(F64(CONFIG["w_e"]))
W_S = wp.constant(F64(CONFIG["w_s"]))
TSCALE = wp.constant(F64(CONFIG["tilt_scale"]))
NSEG = wp.constant(N_SEG)
PN = wp.constant(PATCH_N)
FRONT_Y = wp.constant(F64(FRONT_WHEEL_Y))
MODEL_CELL_F = wp.constant(MODEL_CELL)
SEG_T_F = wp.constant(SEG_T)
(D0X, D0Y, D0R), (D1X, D1Y, D1R), (D2X, D2Y, D2R) = (
    (wp.constant(F64(a)) for a in d) for d in DISCS
)


# --- tile GEMMs ------------------------------------------------------------------------------------------------------


@wp.func
def relu(x: float):
    return wp.max(x, 0.0)


@wp.func
def relu_mask(g: float, h: float):
    return wp.where(h > 0.0, g, 0.0)


def _gemm_kernels() -> dict:
    """Feature-major activations (features x particles): y = act(W x + b) (forward; W (N, K) the layer's weight,
    x (K, n), b (N, 1)) and y = (W^T x) * [h > 0] (backward; the left operand the transposed weight (N, K), x the
    output adjoint (K, n)), each with and without the activation / mask. Output tile (TN features, TM particles).
    """
    out = {}
    for act in (True, False):

        def fwd(
            w: wp.array2d[float], x: wp.array2d[float], b: wp.array2d[float], y: wp.array2d[float]
        ):
            i, j = wp.tid()
            acc = wp.tile_zeros(shape=(TN, TM), dtype=float)
            for k in range(w.shape[1] // TK):
                a = wp.tile_load(w, shape=(TN, TK), offset=(i * TN, k * TK))
                c = wp.tile_load(x, shape=(TK, TM), offset=(k * TK, j * TM))
                wp.tile_matmul(a, c, acc)
            o = acc + wp.tile_broadcast(
                wp.tile_load(b, shape=(TN, 1), offset=(i * TN, 0)), shape=(TN, TM)
            )
            if wp.static(act):
                o = wp.tile_map(relu, o)
            wp.tile_store(y, o, offset=(i * TN, j * TM))

        def bwd(
            w: wp.array2d[float], x: wp.array2d[float], h: wp.array2d[float], y: wp.array2d[float]
        ):
            i, j = wp.tid()
            acc = wp.tile_zeros(shape=(TN, TM), dtype=float)
            for k in range(w.shape[1] // TK):
                a = wp.tile_load(w, shape=(TN, TK), offset=(i * TN, k * TK))
                c = wp.tile_load(x, shape=(TK, TM), offset=(k * TK, j * TM))
                wp.tile_matmul(a, c, acc)
            if wp.static(act):
                acc = wp.tile_map(
                    relu_mask, acc, wp.tile_load(h, shape=(TN, TM), offset=(i * TN, j * TM))
                )
            wp.tile_store(y, acc, offset=(i * TN, j * TM))

        out[("fwd", act)] = wp.kernel(fwd, enable_backward=False, module="unique")
        out[("bwd", act)] = wp.kernel(bwd, enable_backward=False, module="unique")
    return out


GEMM = _gemm_kernels()


# --- geometry --------------------------------------------------------------------------------------------------------


@wp.func
def lookup(A: wp.array2d[F64], gx0: F64, gy0: F64, x: F64, y: F64):
    """Bilinear lookup of a model-grid field at (x, y) (clamped to the grid, nan -> 0)."""
    fx = (x - gx0) / CELL
    fy = (y - gy0) / CELL
    fx = wp.where(wp.isnan(fx), F64(0.0), fx)
    fy = wp.where(wp.isnan(fy), F64(0.0), fy)
    fx = wp.clamp(fx, F64(0.0), F64(A.shape[1] - 1) - F64(1e-6))
    fy = wp.clamp(fy, F64(0.0), F64(A.shape[0] - 1) - F64(1e-6))
    ix = int(wp.floor(fx))
    iy = int(wp.floor(fy))
    tx = fx - F64(ix)
    ty = fy - F64(iy)
    one = F64(1.0)
    return (one - ty) * ((one - tx) * A[iy, ix] + tx * A[iy, ix + 1]) + ty * (
        (one - tx) * A[iy + 1, ix] + tx * A[iy + 1, ix + 1]
    )


@wp.func
def minrot_ez(a: wp.vec3d):
    """The smallest rotation taking the unit vector a to ez."""
    one = F64(1.0)
    v = wp.cross(a, wp.vec3d(F64(0.0), F64(0.0), one))
    c = a[2]
    K = wp.skew(v)
    I = wp.identity(n=3, dtype=F64)  # noqa: E741
    R = I + K + (K * K) / wp.max(one + c, F64(1e-6))
    if c < F64(-0.9999):
        ax = wp.vec3d(one, F64(0.0), F64(0.0))
        if wp.abs(a[0]) >= F64(0.9):
            ax = wp.vec3d(F64(0.0), one, F64(0.0))
        alt = wp.cross(a, ax)
        alt = alt / wp.max(wp.length(alt), F64(1e-9))
        R = F64(2.0) * wp.outer(alt, alt) - I
    return R


@wp.kernel(enable_backward=False)
def init_state(
    S: wp.array2d[F64],
    ppe: int,
    M: int,
    P: wp.array2d[wp.vec3d],
    YAW: wp.array2d[F64],
    XF: wp.array3d[F64],
):
    """Step-0 state of every particle from its episode's state S (W, 16) (no gradient)."""
    i = wp.tid()
    e = (i // M) // ppe
    q = wp.vec4d(S[e, 3], S[e, 4], S[e, 5], S[e, 6])
    q = q / wp.max(wp.length(q), F64(1e-9))
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    one = F64(1.0)
    two = F64(2.0)
    R = wp.mat33d(
        one - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w),
        two * (x * y + z * w), one - two * (x * x + z * z), two * (y * z - x * w),
        two * (x * z - y * w), two * (y * z + x * w), one - two * (x * x + y * y),
    )  # fmt: skip
    u = wp.vec3d(R[2, 0], R[2, 1], R[2, 2])
    Rz = R * wp.transpose(minrot_ez(u))
    vb = wp.transpose(R) * wp.vec3d(S[e, 7], S[e, 8], S[e, 9])
    wb = wp.transpose(R) * wp.vec3d(S[e, 10], S[e, 11], S[e, 12])
    P[0, i] = wp.vec3d(S[e, 0], S[e, 1], S[e, 2])
    YAW[0, i] = wp.atan2(Rz[1, 0], Rz[0, 0])
    for j in range(3):
        XF[0, i, j] = u[j]
        XF[0, i, 3 + j] = vb[j]
        XF[0, i, 6 + j] = wb[j]
        XF[0, i, 9 + j] = S[e, 13 + j] / V_MAX


@wp.kernel
def feat(
    k: int,
    kb: int,
    M: int,
    P: wp.array2d[wp.vec3d],
    YAW: wp.array2d[F64],
    XF: wp.array3d[F64],
    U: wp.array3d[float],
    H: wp.array2d[F64],
    gx0: F64,
    gy0: F64,
    fmean: wp.array[F64],
    fstd: wp.array[F64],
    PXY: wp.array2d[F64],
    Z: wp.array3d[float],
):
    """MLP input of step k; thread (i, r): particle i, patch row r (r = 0 also writes the 15 leading inputs)."""
    i, r = wp.tid()
    if r == 0:
        pl = i // M
        Z[kb, 0, i] = U[k, pl, 0]
        Z[kb, 1, i] = U[k, pl, 1]
        Z[kb, 2, i] = LOGSIG3
        for j in range(12):
            Z[kb, 3 + j, i] = float(
                wp.clamp((XF[k, i, j] - fmean[j]) / fstd[j], F64(-8.0), F64(8.0))
            )
    p = P[k, i]
    c = wp.cos(YAW[k, i])
    s = wp.sin(YAW[k, i])
    zr = p[2] - R_W
    for b in range(PN):
        j = r * PN + b
        qx = PXY[j, 0]
        qy = PXY[j, 1]
        h = lookup(H, gx0, gy0, p[0] + c * qx - s * qy, p[1] + s * qx + c * qy) - zr
        Z[kb, 15 + j, i] = float(wp.clamp((h - fmean[12 + j]) / fstd[12 + j], F64(-8.0), F64(8.0)))


@wp.func
def lookup_grad(A: wp.array2d[F64], gx0: F64, gy0: F64, x: F64, y: F64):
    """(value, d/dx, d/dy) of lookup (zero gradient where the query is clamped to the grid)."""
    fx = (x - gx0) / CELL
    fy = (y - gy0) / CELL
    fx = wp.where(wp.isnan(fx), F64(0.0), fx)
    fy = wp.where(wp.isnan(fy), F64(0.0), fy)
    hx = F64(A.shape[1] - 1) - F64(1e-6)
    hy = F64(A.shape[0] - 1) - F64(1e-6)
    inx = wp.where((fx >= F64(0.0)) and (fx <= hx), F64(1.0), F64(0.0))
    iny = wp.where((fy >= F64(0.0)) and (fy <= hy), F64(1.0), F64(0.0))
    fx = wp.clamp(fx, F64(0.0), hx)
    fy = wp.clamp(fy, F64(0.0), hy)
    ix = int(wp.floor(fx))
    iy = int(wp.floor(fy))
    tx = fx - F64(ix)
    ty = fy - F64(iy)
    one = F64(1.0)
    a00 = A[iy, ix]
    a01 = A[iy, ix + 1]
    a10 = A[iy + 1, ix]
    a11 = A[iy + 1, ix + 1]
    v = (one - ty) * ((one - tx) * a00 + tx * a01) + ty * ((one - tx) * a10 + tx * a11)
    gx = ((one - ty) * (a01 - a00) + ty * (a11 - a10)) * inx / CELL
    gy = ((one - tx) * (a10 - a00) + tx * (a11 - a01)) * iny / CELL
    return wp.vec3d(v, gx, gy)


@wp.kernel(enable_backward=False)
def feat_adj(
    k: int,
    kb: int,
    M: int,
    P: wp.array2d[wp.vec3d],
    YAW: wp.array2d[F64],
    XF: wp.array3d[F64],
    H: wp.array2d[F64],
    gx0: F64,
    gy0: F64,
    fmean: wp.array[F64],
    fstd: wp.array[F64],
    PXY: wp.array2d[F64],
    aZ: wp.array3d[float],
    aP: wp.array2d[wp.vec3d],
    aYAW: wp.array2d[F64],
    aXF: wp.array3d[F64],
    aU: wp.array3d[float],
):
    """Hand-written adjoint of feat (same thread layout): aZ -> aU, aXF, aP, aYAW (accumulated)."""
    i, r = wp.tid()
    if r == 0:
        pl = i // M
        wp.atomic_add(aU, k, pl, 0, aZ[kb, 0, i])
        wp.atomic_add(aU, k, pl, 1, aZ[kb, 1, i])
        for j in range(12):
            zv = (XF[k, i, j] - fmean[j]) / fstd[j]
            if zv >= F64(-8.0) and zv <= F64(8.0):
                aXF[k, i, j] = aXF[k, i, j] + F64(aZ[kb, 3 + j, i]) / fstd[j]
    p = P[k, i]
    c = wp.cos(YAW[k, i])
    s = wp.sin(YAW[k, i])
    zr = p[2] - R_W
    gp = wp.vec3d(F64(0.0), F64(0.0), F64(0.0))
    gyaw = F64(0.0)
    for b in range(PN):
        j = r * PN + b
        qx = PXY[j, 0]
        qy = PXY[j, 1]
        hg = lookup_grad(H, gx0, gy0, p[0] + c * qx - s * qy, p[1] + s * qx + c * qy)
        zv = (hg[0] - zr - fmean[12 + j]) / fstd[12 + j]
        if zv >= F64(-8.0) and zv <= F64(8.0):
            g = F64(aZ[kb, 15 + j, i]) / fstd[12 + j]
            gp += wp.vec3d(g * hg[1], g * hg[2], -g)
            gyaw += g * (hg[1] * (-s * qx - c * qy) + hg[2] * (c * qx - s * qy))
    wp.atomic_add(aP, k, i, gp)
    wp.atomic_add(aYAW, k, i, gyaw)


@wp.func
def nan_to_num30(x: float):
    x = wp.where(wp.isnan(x), 0.0, x)
    return wp.where(wp.isinf(x), wp.clamp(x, -30.0, 30.0), x)


@wp.func
def tilt_cost(up: F64):
    t = wp.acos(wp.clamp(up, F64(-0.999), F64(0.999))) * F64(57.29577951308232) / TSCALE
    return t * t


@wp.func
def sample(O: wp.array3d[float], kb: int, i: int, comp: int, d: int, eps: float, xm: F64, xs: F64):
    """Outcome dimension d of component comp, reparameterized: clamp(mu + s eps, -8, 8) destandardized."""
    m = O[kb, 4 + comp * 17 + d, i]
    sd = wp.exp(wp.clamp(O[kb, 72 + comp * 17 + d, i], -4.0, 2.0))
    return F64(wp.clamp(m + sd * eps, -8.0, 8.0)) * xs + xm


@wp.func
def clearance(
    D: wp.array2d[F64],
    gx0: F64,
    gy0: F64,
    x: F64,
    y: F64,
    c: F64,
    s: F64,
    ox: F64,
    oy: F64,
    rad: F64,
):
    d = lookup(D, gx0, gy0, x + c * ox - s * oy, y + s * ox + c * oy)
    t = wp.max((rad + CMARGIN - d) / CMARGIN, F64(0.0))
    return t * t


@wp.kernel
def head(
    k: int,
    kb: int,
    O: wp.array3d[float],
    xm: wp.array[F64],
    xs: wp.array[F64],
    m_u: F64,
    s_u: F64,
    EPS: wp.array3d[float],
    RU: wp.array2d[float],
    P: wp.array2d[wp.vec3d],
    YAW: wp.array2d[F64],
    XF: wp.array3d[F64],
    VA: wp.array2d[F64],
    DA: wp.array2d[F64],
    gx0: F64,
    gy0: F64,
    PT: wp.array2d[F64],
    TL: wp.array2d[F64],
    LS: wp.array2d[F64],
):
    """Mixture terms of step k, sampled outcome, next state, step loss."""
    i = wp.tid()
    l0 = nan_to_num30(O[kb, 0, i])
    l1 = nan_to_num30(O[kb, 1, i])
    l2 = nan_to_num30(O[kb, 2, i])
    l3 = nan_to_num30(O[kb, 3, i])
    mx = wp.max(wp.max(l0, l1), wp.max(l2, l3))
    e0 = wp.exp(l0 - mx)
    e1 = wp.exp(l1 - mx)
    e2 = wp.exp(l2 - mx)
    e3 = wp.exp(l3 - mx)
    se = e0 + e1 + e2 + e3
    wv = wp.vec4d(F64(e0 / se), F64(e1 / se), F64(e2 / se), F64(e3 / se))
    pt = F64(0.0)
    tl = F64(0.0)
    sq3 = F64(1.7320508075688772)
    # P(tip) and the tilt term (3-point Gauss-Hermite) from every component's smallest up_z
    for c in range(4):
        mu = F64(O[kb, 4 + c * 17 + 16, i]) * s_u + m_u
        sd = F64(wp.exp(wp.clamp(O[kb, 72 + c * 17 + 16, i], -4.0, 2.0))) * s_u + F64(1e-6)
        pt += wv[c] * F64(0.5) * (F64(1.0) + wp.erf((F64(0.5) - mu) / sd / F64(1.4142135623730951)))
        tl += wv[c] * (
            tilt_cost(mu - sq3 * sd) / F64(6.0)
            + F64(2.0) / F64(3.0) * tilt_cost(mu)
            + tilt_cost(mu + sq3 * sd) / F64(6.0)
        )
    PT[k, i] = pt
    TL[k, i] = tl
    # the component by inverse CDF of one uniform draw
    r = F64(RU[k, i])
    cnt = int(0)
    cum = F64(0.0)
    for c in range(4):
        cum += wv[c]
        if cum <= r:
            cnt += 1
    comp = wp.min(cnt, 3)
    dx = sample(O, kb, i, comp, 0, EPS[k, i, 0], xm[0], xs[0])
    dy = sample(O, kb, i, comp, 1, EPS[k, i, 1], xm[1], xs[1])
    dz = sample(O, kb, i, comp, 2, EPS[k, i, 2], xm[2], xs[2])
    u = wp.vec3d(
        sample(O, kb, i, comp, 3, EPS[k, i, 3], xm[3], xs[3]),
        sample(O, kb, i, comp, 4, EPS[k, i, 4], xm[4], xs[4]),
        sample(O, kb, i, comp, 5, EPS[k, i, 5], xm[5], xs[5]),
    )
    dyaw = sample(O, kb, i, comp, 6, EPS[k, i, 6], xm[6], xs[6])
    yaw0 = YAW[k, i]
    c0 = wp.cos(yaw0)
    s0 = wp.sin(yaw0)
    p1 = P[k, i] + wp.vec3d(c0 * dx - s0 * dy, s0 * dx + c0 * dy, dz)
    yaw1 = yaw0 + dyaw
    u = u / wp.max(wp.length(u), F64(1e-9))
    P[k + 1, i] = p1
    YAW[k + 1, i] = yaw1
    for j in range(3):
        XF[k + 1, i, j] = u[j]
    for d in range(7, 16):  # body-frame velocities and wheel speeds / V_MAX: the outcome's own
        XF[k + 1, i, d - 4] = sample(O, kb, i, comp, d, EPS[k, i, d], xm[d], xs[d])
    c1 = wp.cos(yaw1)
    s1 = wp.sin(yaw1)
    cl = (
        clearance(DA, gx0, gy0, p1[0], p1[1], c1, s1, D0X, D0Y, D0R)
        + clearance(DA, gx0, gy0, p1[0], p1[1], c1, s1, D1X, D1Y, D1R)
        + clearance(DA, gx0, gy0, p1[0], p1[1], c1, s1, D2X, D2Y, D2R)
    )
    LS[k, i] = W_RUN6 * lookup(VA, gx0, gy0, p1[0], p1[1]) + W_C6 * cl


@wp.func
def tilt_cost_d(up: F64):
    """d tilt_cost / d up (0 where the clamp is active)."""
    if up < F64(-0.999) or up > F64(0.999):
        return F64(0.0)
    sc = F64(57.29577951308232) / TSCALE
    return F64(-2.0) * wp.acos(up) * sc * sc / wp.sqrt(F64(1.0) - up * up)


@wp.func
def clearance_adj(
    D: wp.array2d[F64],
    gx0: F64,
    gy0: F64,
    x: F64,
    y: F64,
    c: F64,
    s: F64,
    ox: F64,
    oy: F64,
    rad: F64,
    a: F64,
):
    """(d/dx, d/dy, d/dyaw) of a * clearance."""
    g = lookup_grad(D, gx0, gy0, x + c * ox - s * oy, y + s * ox + c * oy)
    t = (rad + CMARGIN - g[0]) / CMARGIN
    out = wp.vec3d(F64(0.0), F64(0.0), F64(0.0))
    if t > F64(0.0):
        gd = a * F64(-2.0) * t / CMARGIN  # d (a t^2) / d dist
        out = wp.vec3d(
            gd * g[1], gd * g[2], gd * (g[1] * (-s * ox - c * oy) + g[2] * (c * ox - s * oy))
        )
    return out


@wp.func
def sample_adj(
    O: wp.array3d[float],
    kb: int,
    i: int,
    comp: int,
    d: int,
    eps: float,
    xs: F64,
    a: F64,
    aO: wp.array3d[float],
):
    """Adjoint of sample: a = d loss / d x_d into aO (the component's mean and log-scale entries)."""
    m = O[kb, 4 + comp * 17 + d, i]
    ls = O[kb, 72 + comp * 17 + d, i]
    sd = wp.exp(wp.clamp(ls, -4.0, 2.0))
    z = m + sd * eps
    if z >= -8.0 and z <= 8.0:
        az = float(a * xs)
        aO[kb, 4 + comp * 17 + d, i] = az
        if ls >= -4.0 and ls <= 2.0:
            aO[kb, 72 + comp * 17 + d, i] = az * eps * sd


@wp.kernel(enable_backward=False)
def head_adj(
    k: int,
    kb: int,
    O: wp.array3d[float],
    xm: wp.array[F64],
    xs: wp.array[F64],
    m_u: F64,
    s_u: F64,
    EPS: wp.array3d[float],
    RU: wp.array2d[float],
    P: wp.array2d[wp.vec3d],
    YAW: wp.array2d[F64],
    VA: wp.array2d[F64],
    DA: wp.array2d[F64],
    gx0: F64,
    gy0: F64,
    aPT: wp.array2d[F64],
    aTL: wp.array2d[F64],
    aLS: wp.array2d[F64],
    aO: wp.array3d[float],
    aP: wp.array2d[wp.vec3d],
    aYAW: wp.array2d[F64],
    aXF: wp.array3d[F64],
):
    """Hand-written adjoint of head: (aPT, aTL, aLS of step k; aP, aYAW, aXF of step k + 1) -> aO of step k (the
    entries the step reads; aO is zeroed before, the rest stay zero) and aP, aYAW of step k (accumulated).
    """
    i = wp.tid()
    # forward recompute: mixture weights and component
    l0 = nan_to_num30(O[kb, 0, i])
    l1 = nan_to_num30(O[kb, 1, i])
    l2 = nan_to_num30(O[kb, 2, i])
    l3 = nan_to_num30(O[kb, 3, i])
    mx = wp.max(wp.max(l0, l1), wp.max(l2, l3))
    e0 = wp.exp(l0 - mx)
    e1 = wp.exp(l1 - mx)
    e2 = wp.exp(l2 - mx)
    e3 = wp.exp(l3 - mx)
    se = e0 + e1 + e2 + e3
    wv = wp.vec4d(F64(e0 / se), F64(e1 / se), F64(e2 / se), F64(e3 / se))
    r = F64(RU[k, i])
    cnt = int(0)
    cum = F64(0.0)
    for c in range(4):
        cum += wv[c]
        if cum <= r:
            cnt += 1
    comp = wp.min(cnt, 3)
    dx = sample(O, kb, i, comp, 0, EPS[k, i, 0], xm[0], xs[0])
    dy = sample(O, kb, i, comp, 1, EPS[k, i, 1], xm[1], xs[1])
    u = wp.vec3d(
        sample(O, kb, i, comp, 3, EPS[k, i, 3], xm[3], xs[3]),
        sample(O, kb, i, comp, 4, EPS[k, i, 4], xm[4], xs[4]),
        sample(O, kb, i, comp, 5, EPS[k, i, 5], xm[5], xs[5]),
    )
    yaw0 = YAW[k, i]
    c0 = wp.cos(yaw0)
    s0 = wp.sin(yaw0)
    p1 = P[k + 1, i]
    yaw1 = YAW[k + 1, i]
    c1 = wp.cos(yaw1)
    s1 = wp.sin(yaw1)
    # step loss LS = W_RUN6 V(p1) + W_C6 sum_discs clearance(p1, yaw1)
    als = aLS[k, i]
    gv = lookup_grad(VA, gx0, gy0, p1[0], p1[1])
    ga = als * W_C6
    gs = (
        clearance_adj(DA, gx0, gy0, p1[0], p1[1], c1, s1, D0X, D0Y, D0R, ga)
        + clearance_adj(DA, gx0, gy0, p1[0], p1[1], c1, s1, D1X, D1Y, D1R, ga)
        + clearance_adj(DA, gx0, gy0, p1[0], p1[1], c1, s1, D2X, D2Y, D2R, ga)
    )
    ap1 = aP[k + 1, i] + wp.vec3d(
        als * W_RUN6 * gv[1] + gs[0], als * W_RUN6 * gv[2] + gs[1], F64(0.0)
    )
    ayaw1 = aYAW[k + 1, i] + gs[2]
    # compose: p1 = p0 + Rz(yaw0) (dx, dy, dz), yaw1 = yaw0 + dyaw, XF[k+1] = (u / |u|, x_7..15)
    adx = c0 * ap1[0] + s0 * ap1[1]
    ady = -s0 * ap1[0] + c0 * ap1[1]
    aP[k, i] = aP[k, i] + ap1
    aYAW[k, i] = aYAW[k, i] + ayaw1 + ap1[0] * (-s0 * dx - c0 * dy) + ap1[1] * (c0 * dx - s0 * dy)
    au_hat = wp.vec3d(aXF[k + 1, i, 0], aXF[k + 1, i, 1], aXF[k + 1, i, 2])
    nu = wp.length(u)
    au = au_hat / F64(1e-9)
    if nu > F64(1e-9):
        uh = u / nu
        au = (au_hat - uh * wp.dot(uh, au_hat)) / nu
    sample_adj(O, kb, i, comp, 0, EPS[k, i, 0], xs[0], adx, aO)
    sample_adj(O, kb, i, comp, 1, EPS[k, i, 1], xs[1], ady, aO)
    sample_adj(O, kb, i, comp, 2, EPS[k, i, 2], xs[2], ap1[2], aO)
    sample_adj(O, kb, i, comp, 3, EPS[k, i, 3], xs[3], au[0], aO)
    sample_adj(O, kb, i, comp, 4, EPS[k, i, 4], xs[4], au[1], aO)
    sample_adj(O, kb, i, comp, 5, EPS[k, i, 5], xs[5], au[2], aO)
    sample_adj(O, kb, i, comp, 6, EPS[k, i, 6], xs[6], ayaw1, aO)
    for d in range(7, 16):
        sample_adj(O, kb, i, comp, d, EPS[k, i, d], xs[d], aXF[k + 1, i, d - 4], aO)
    # P(tip) and tilt of the step: through the weights and every component's min-up_z mean and scale
    apt = aPT[k, i]
    atl = aTL[k, i]
    sq3 = F64(1.7320508075688772)
    aw = wp.vec4d(F64(0.0), F64(0.0), F64(0.0), F64(0.0))
    for c in range(4):
        lsr = O[kb, 72 + c * 17 + 16, i]
        es = wp.exp(wp.clamp(lsr, -4.0, 2.0))
        mu = F64(O[kb, 4 + c * 17 + 16, i]) * s_u + m_u
        sd = F64(es) * s_u + F64(1e-6)
        z = (F64(0.5) - mu) / sd
        phi = wp.exp(F64(-0.5) * z * z) * F64(0.3989422804014327)
        Phi = F64(0.5) * (F64(1.0) + wp.erf(z / F64(1.4142135623730951)))
        T = (
            tilt_cost(mu - sq3 * sd) / F64(6.0)
            + F64(2.0) / F64(3.0) * tilt_cost(mu)
            + tilt_cost(mu + sq3 * sd) / F64(6.0)
        )
        tm = tilt_cost_d(mu - sq3 * sd)
        tp = tilt_cost_d(mu + sq3 * sd)
        dT_dmu = tm / F64(6.0) + F64(2.0) / F64(3.0) * tilt_cost_d(mu) + tp / F64(6.0)
        dT_dsd = sq3 * (tp - tm) / F64(6.0)
        amu = wv[c] * (apt * (-phi / sd) + atl * dT_dmu)
        asd = wv[c] * (apt * (-phi * z / sd) + atl * dT_dsd)
        aO[kb, 4 + c * 17 + 16, i] = float(amu * s_u)
        if lsr >= -4.0 and lsr <= 2.0:
            aO[kb, 72 + c * 17 + 16, i] = float(asd * s_u * F64(es))
        aw[c] = apt * Phi + atl * T
    sw = wp.dot(wv, aw)
    for c in range(4):
        lr = O[kb, c, i]
        aO[kb, c, i] = wp.where(wp.isnan(lr) or wp.isinf(lr), 0.0, float(wv[c] * (aw[c] - sw)))


@wp.kernel
def loss(
    M: int,
    ppe: int,
    U: wp.array3d[float],
    UP: wp.array2d[F64],
    P: wp.array2d[wp.vec3d],
    VA: wp.array2d[F64],
    gx0: F64,
    gy0: F64,
    PT: wp.array2d[F64],
    TL: wp.array2d[F64],
    LS: wp.array2d[F64],
    inv_n: F64,
    mode: int,
    LP: wp.array[F64],
    TOT: wp.array[F64],
    PL: wp.array[F64],
):
    """Plan loss of particle i; mode 0: TOT += L / n (the gradient objective), 1: PL[plan] += L / M."""
    i = wp.tid()
    pl = i // M
    e = pl // ppe
    pN = P[NSEG, i]
    L = lookup(VA, gx0, gy0, pN[0], pN[1])
    surv = F64(1.0)
    eff = F64(0.0)
    for k in range(NSEG):
        L += LS[k, i] + W_TILT * TL[k, i]
        surv *= F64(1.0) - PT[k, i]
        a = F64(U[k, pl, 0])
        b = F64(U[k, pl, 1])
        eff += a * a + b * b
    L += W_TIP * (F64(1.0) - surv) + W_E * eff / F64(2 * NSEG)
    # command rate, from the command last published (UP)
    ra = F64(U[0, pl, 0]) - UP[e, 0]
    rb = F64(U[0, pl, 1]) - UP[e, 1]
    rate = ra * ra + rb * rb
    for k in range(1, NSEG):
        ra = F64(U[k, pl, 0] - U[k - 1, pl, 0])
        rb = F64(U[k, pl, 1] - U[k - 1, pl, 1])
        rate += ra * ra + rb * rb
    L += W_S * rate
    LP[i] = L
    if mode == 0:
        wp.atomic_add(TOT, 0, wp.where(wp.isfinite(L), L, F64(0.0)) * inv_n)
    else:
        wp.atomic_add(PL, pl, wp.where(wp.isfinite(L), L / F64(M), F64(1e30)))


# --- planner bookkeeping ---------------------------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def noise(seed: wp.array[int], EPS: wp.array3d[float], RU: wp.array2d[float]):
    k, i = wp.tid()
    st_ = wp.rand_init(seed[0], k * EPS.shape[1] + i)
    for d in range(17):
        EPS[k, i, d] = wp.randn(st_)
    RU[k, i] = wp.randf(st_)


@wp.kernel(enable_backward=False)
def bump(c: wp.array[int]):
    c[0] = c[0] + 1


@wp.kernel(enable_backward=False)
def shoot_cands(seed: wp.array[int], U0: wp.array3d[float], K: int, C: wp.array3d[float]):
    """Shoot candidates: half uniform, half the warm start + 0.3 N(0, 1), clipped."""
    k, w, c = wp.tid()
    st_ = wp.rand_init(seed[0] + 7777, (k * U0.shape[1] + w) * K + c)
    for d in range(2):
        if c < K // 2:
            C[k, w * K + c, d] = wp.randf(st_)
        else:
            C[k, w * K + c, d] = wp.clamp(U0[k, w, d] + 0.3 * wp.randn(st_), 0.0, 1.0)


@wp.kernel(enable_backward=False)
def select_inits(
    PL: wp.array[F64],
    K: int,
    n0: int,
    C: wp.array3d[float],
    U0: wp.array3d[float],
    TAKEN: wp.array[int],
    UPL: wp.array3d[float],
):
    """Plans of episode w: the warm start, then the n0 - 1 best shoot candidates by mean model loss."""
    w = wp.tid()
    for k in range(NSEG):
        for d in range(2):
            UPL[k, w * n0, d] = U0[k, w, d]
    for j in range(n0 - 1):
        best = int(0)
        bv = F64(1e300)
        for c in range(K):
            v = PL[w * K + c]
            if TAKEN[w * K + c] == 0 and v < bv:
                bv = v
                best = c
        TAKEN[w * K + best] = 1
        for k in range(NSEG):
            for d in range(2):
                UPL[k, w * n0 + 1 + j, d] = C[k, w * K + best, d]


@wp.kernel(enable_backward=False)
def adam(
    g: wp.array3d[float],
    step: wp.array[int],
    lr: float,
    u: wp.array3d[float],
    m: wp.array3d[float],
    v: wp.array3d[float],
):
    """Adam (betas 0.9, 0.999, eps 1e-8) on the plans, grad nan -> 0, then clamp to [0, 1], nan -> 0.5."""
    k, p, d = wp.tid()
    gr = g[k, p, d]
    if not wp.isfinite(gr):
        gr = 0.0
    mm = 0.9 * m[k, p, d] + 0.1 * gr
    vv = 0.999 * v[k, p, d] + 0.001 * gr * gr
    m[k, p, d] = mm
    v[k, p, d] = vv
    t = float(step[0])
    bc1 = 1.0 - wp.pow(0.9, t)
    bc2 = 1.0 - wp.pow(0.999, t)
    x = u[k, p, d] - lr / bc1 * mm / (wp.sqrt(vv) / wp.sqrt(bc2) + 1e-8)
    x = wp.clamp(x, 0.0, 1.0)
    u[k, p, d] = wp.where(wp.isnan(x), 0.5, x)


@wp.kernel(enable_backward=False)
def pick_best(
    PL: wp.array[F64],
    n0: int,
    M: int,
    UPL: wp.array3d[float],
    P: wp.array2d[wp.vec3d],
    YAW: wp.array2d[F64],
    OUT: wp.array3d[float],
    KNOTS: wp.array2d[wp.vec3d],
):
    """The lowest-loss plan of episode w, and its mean path (x, y, yaw) over its M scoring particles."""
    w = wp.tid()
    best = int(0)
    bv = F64(1e300)
    for j in range(n0):
        if PL[w * n0 + j] < bv:
            bv = PL[w * n0 + j]
            best = j
    for k in range(NSEG):
        for d in range(2):
            OUT[k, w, d] = UPL[k, w * n0 + best, d]
    i0 = (w * n0 + best) * M
    for k in range(NSEG + 1):
        sx = F64(0.0)
        sy = F64(0.0)
        sc = F64(0.0)
        ss = F64(0.0)
        for m in range(M):
            p = P[k, i0 + m]
            sx += p[0]
            sy += p[1]
            sc += wp.cos(YAW[k, i0 + m])
            ss += wp.sin(YAW[k, i0 + m])
        KNOTS[k, w] = wp.vec3d(sx / F64(M), sy / F64(M), wp.atan2(ss, sc))


# --- the frame's maps onto the model grid ------------------------------------------------------------------------------


@wp.func
def bilinear_f32(A: wp.array2d[float], cell: float, ox: float, oy: float, x: float, y: float):
    """Bilinear lookup of a stack grid field (node (0, 0) at (ox, oy)), clamped to the grid."""
    fx = wp.clamp((x - ox) / cell, 0.0, float(A.shape[1] - 1) - 1e-6)
    fy = wp.clamp((y - oy) / cell, 0.0, float(A.shape[0] - 1) - 1e-6)
    ix = int(wp.floor(fx))
    iy = int(wp.floor(fy))
    tx = fx - float(ix)
    ty = fy - float(iy)
    return (1.0 - ty) * ((1.0 - tx) * A[iy, ix] + tx * A[iy, ix + 1]) + ty * (
        (1.0 - tx) * A[iy + 1, ix] + tx * A[iy + 1, ix + 1]
    )


@wp.kernel(enable_backward=False)
def resample_kernel(
    src: wp.array2d[float], cell: float, ox: float, oy: float, dst: wp.array2d[F64]
):
    r, c = wp.tid()
    dst[r, c] = F64(
        bilinear_f32(src, cell, ox, oy, float(c) * MODEL_CELL_F, float(r) * MODEL_CELL_F)
    )


@wp.kernel(enable_backward=False)
def value_kernel(
    V: wp.array3d[float],
    cell: float,
    ox: float,
    oy: float,
    ref_pts: wp.array[wp.vec2],
    ref_n: wp.array[wp.int32],
    use_ref: int,
    w_track: float,
    dst: wp.array2d[F64],
):
    """min over heading of the cost-to-go at model node (r, c), + w_track min(d, 1)^2 to the reference curve."""
    r, c = wp.tid()
    x = float(c) * MODEL_CELL_F
    y = float(r) * MODEL_CELL_F
    fx = wp.clamp((x - ox) / cell, 0.0, float(V.shape[1] - 1) - 1e-6)
    fy = wp.clamp((y - oy) / cell, 0.0, float(V.shape[0] - 1) - 1e-6)
    ix = int(wp.floor(fx))
    iy = int(wp.floor(fy))
    tx = fx - float(ix)
    ty = fy - float(iy)
    # bilinear in (x, y) of the heading minimum at the four corners, as the min over heading of the cell field
    v00 = V[iy, ix, 0]
    v01 = V[iy, ix + 1, 0]
    v10 = V[iy + 1, ix, 0]
    v11 = V[iy + 1, ix + 1, 0]
    for t in range(1, V.shape[2]):
        v00 = wp.min(v00, V[iy, ix, t])
        v01 = wp.min(v01, V[iy, ix + 1, t])
        v10 = wp.min(v10, V[iy + 1, ix, t])
        v11 = wp.min(v11, V[iy + 1, ix + 1, t])
    v = (1.0 - ty) * ((1.0 - tx) * v00 + tx * v01) + ty * ((1.0 - tx) * v10 + tx * v11)
    n = ref_n[0]
    if use_ref != 0 and n >= 3:
        d2 = float(1.0e30)
        for j in range(n):
            q = ref_pts[j]
            d2 = wp.min(d2, (q[0] - x) * (q[0] - x) + (q[1] - y) * (q[1] - y))
        d = wp.min(wp.sqrt(d2), 1.0)
        v += w_track * d * d
    dst[r, c] = F64(v)


@wp.kernel(enable_backward=False)
def ground_z_kernel(H: wp.array2d[F64], S: wp.array2d[F64]):
    """The chassis height as the model saw it at rest: the mean ground under the two front wheels + R_W.

    The heightmap and the odometry share the map frame, but base_link sits at a robot-specific height above the
    axle; the model's patch is heights relative to (z - R_W), so read z off the map instead.
    """
    q = wp.vec4d(S[0, 3], S[0, 4], S[0, 5], S[0, 6])
    yaw = wp.atan2(
        F64(2.0) * (q[3] * q[2] + q[0] * q[1]), F64(1.0) - F64(2.0) * (q[1] * q[1] + q[2] * q[2])
    )
    c = wp.cos(yaw)
    s = wp.sin(yaw)
    x = S[0, 0]
    y = S[0, 1]
    hl = lookup(H, F64(0.0), F64(0.0), x - s * FRONT_Y, y + c * FRONT_Y)
    hr = lookup(H, F64(0.0), F64(0.0), x + s * FRONT_Y, y - c * FRONT_Y)
    S[0, 2] = F64(0.5) * (hl + hr) + R_W


@wp.kernel(enable_backward=False)
def path_kernel(
    KNOTS: wp.array2d[wp.vec3d], yaw0: float, dt: float, controlled: wp.array2d[wp.vec3f]
):
    """The model's mean path at the rollouts' dt into rollout 0 of plan_sim.controlled (x, y, yaw)."""
    t = wp.tid()
    s = float(t) * dt / SEG_T_F
    i = wp.min(int(wp.floor(s)), N_SEG - 1)
    f = wp.min(s - float(i), 1.0)
    a = KNOTS[i, 0]
    b = KNOTS[i + 1, 0]
    # unwrap the knot headings onto yaw0
    ya = yaw0
    for j in range(i):
        ya += float(
            wp.atan2(
                wp.sin(KNOTS[j + 1, 0][2] - KNOTS[j, 0][2]),
                wp.cos(KNOTS[j + 1, 0][2] - KNOTS[j, 0][2]),
            )
        )
    db = float(wp.atan2(wp.sin(b[2] - a[2]), wp.cos(b[2] - a[2])))
    controlled[t, 0] = wp.vec3f(
        float((F64(1.0) - F64(f)) * a[0] + F64(f) * b[0]),
        float((F64(1.0) - F64(f)) * a[1] + F64(f) * b[1]),
        ya + f * db,
    )


# --- the rollout -----------------------------------------------------------------------------------------------------


class Net:
    """The frozen MLP as padded Warp arrays: w (out, in) the forward's left operand, wt (in, out) the backward's,
    b (out, 1)."""

    def __init__(self, z: np.lib.npyio.NpzFile) -> None:
        dims = [(DI, HIDDEN), (HIDDEN, HIDDEN), (HIDDEN, HIDDEN), (HIDDEN, DOP)]
        self.w: list[wp.array] = []
        self.wt: list[wp.array] = []
        self.b: list[wp.array] = []
        for layer, (din, dout) in enumerate(dims):
            w_src, b_src = z[f"w{layer}"], z[f"b{layer}"]
            W = np.zeros((dout, din), np.float32)
            W[: w_src.shape[0], : w_src.shape[1]] = w_src
            b = np.zeros((dout, 1), np.float32)
            b[: b_src.shape[0], 0] = b_src
            self.w.append(wp.array(W, dtype=float))
            self.wt.append(wp.array(np.ascontiguousarray(W.T), dtype=float))
            self.b.append(wp.array(b, dtype=float))


class Pass:
    """Buffers and launches of one batch of n particles (M per plan, ppe plans per episode); grad=True keeps every
    step's activations for the adjoint."""

    def __init__(
        self, plnr: NeuralPlanner, n: int, M: int, ppe: int, U: wp.array, grad: bool
    ) -> None:
        def z(*s: int, dt: type = float) -> wp.array:
            return wp.zeros(s, dtype=dt)

        nb = N_SEG if grad else 1
        assert n % TM == 0, f"{n} particles: not a multiple of the GEMM tile {TM}"
        self.p, self.n, self.M, self.ppe, self.U, self.grad, self.nb = plnr, n, M, ppe, U, grad, nb
        # features x particles
        self.Z, self.H1, self.H2, self.H3, self.O = (
            z(nb, DI, n), z(nb, HIDDEN, n), z(nb, HIDDEN, n), z(nb, HIDDEN, n), z(nb, DOP, n)
        )  # fmt: skip
        self.P = z(N_SEG + 1, n, dt=wp.vec3d)
        self.YAW = z(N_SEG + 1, n, dt=F64)
        self.XF = z(N_SEG + 1, n, 12, dt=F64)
        self.PT, self.TL, self.LS = z(N_SEG, n, dt=F64), z(N_SEG, n, dt=F64), z(N_SEG, n, dt=F64)
        self.EPS, self.RU = z(N_SEG, n, D_X), z(N_SEG, n)
        self.LP, self.TOT, self.PL = z(n, dt=F64), z(1, dt=F64), z(n // M, dt=F64)
        if grad:  # adjoints
            self.aZ, self.aH1, self.aH2, self.aH3, self.aO = (
                z(nb, DI, n), z(nb, HIDDEN, n), z(nb, HIDDEN, n), z(nb, HIDDEN, n), z(nb, DOP, n)
            )  # fmt: skip
            self.aP = z(N_SEG + 1, n, dt=wp.vec3d)
            self.aYAW = z(N_SEG + 1, n, dt=F64)
            self.aXF = z(N_SEG + 1, n, 12, dt=F64)
            self.aPT = z(N_SEG, n, dt=F64)
            self.aTL = z(N_SEG, n, dt=F64)
            self.aLS = z(N_SEG, n, dt=F64)
            self.aLP, self.aTOT = z(n, dt=F64), wp.array([1.0], dtype=F64)
            self.aU = wp.zeros(U.shape, dtype=float)

    def _feat_args(self, k: int, kb: int) -> list:
        p = self.p
        return [k, kb, self.M, self.P, self.YAW, self.XF, self.U, p.H, p.gx0, p.gy0, p.fmean, p.fstd, p.PXY, self.Z]  # fmt: skip

    def _head_args(self, k: int, kb: int) -> list:
        p = self.p
        return [k, kb, self.O, p.xm, p.xs, p.m_u, p.s_u, self.EPS, self.RU, self.P, self.YAW, self.XF, p.VA, p.DA,
                p.gx0, p.gy0, self.PT, self.TL, self.LS]  # fmt: skip

    def _loss_args(self, mode: int) -> list:
        p = self.p
        return [self.M, self.ppe, self.U, p.UP, self.P, p.VA, p.gx0, p.gy0, self.PT, self.TL, self.LS,
                F64(1.0 / self.n), mode, self.LP, self.TOT, self.PL]  # fmt: skip

    def _loss_adj(self) -> list:
        Z0 = F64(0.0)
        return [0, 0, self.aU, None, self.aP, None, Z0, Z0, self.aPT, self.aTL, self.aLS, Z0, 0, self.aLP,
                self.aTOT, None]  # fmt: skip

    def _gemm(self, w: wp.array, x: wp.array, b: wp.array, y: wp.array, act: bool) -> None:
        """y = act(w x + b)."""
        if self.p.cublas:
            yt = wp.to_torch(y)
            torch.addmm(wp.to_torch(b), wp.to_torch(w), wp.to_torch(x), out=yt)
            if act:
                yt.relu_()
        else:
            wp.launch_tiled(
                GEMM[("fwd", act)],
                dim=(y.shape[0] // TN, self.n // TM),
                inputs=[w, x, b, y],
                block_dim=GEMM_BLOCK,
            )

    def _gemm_b(self, wt: wp.array, x: wp.array, h: wp.array, y: wp.array, mask: bool) -> None:
        """y = (wt x) * [h > 0] (mask) or wt x."""
        if self.p.cublas:
            yt = wp.to_torch(y)
            torch.mm(wp.to_torch(wt), wp.to_torch(x), out=yt)
            if mask:
                torch.ops.aten.threshold_backward.grad_input(yt, wp.to_torch(h), 0.0, grad_input=yt)
        else:
            wp.launch_tiled(
                GEMM[("bwd", mask)],
                dim=(y.shape[0] // TN, self.n // TM),
                inputs=[wt, x, h, y],
                block_dim=GEMM_BLOCK,
            )

    def forward(self, mode: int, draw: bool = True) -> None:
        """Rollout + loss; draw: fresh noise from the planner's counter (else EPS / RU as set)."""
        p, n, net = self.p, self.n, self.p.net
        if draw:
            wp.launch(noise, dim=(N_SEG, n), inputs=[p.seed, self.EPS, self.RU])
            wp.launch(bump, dim=1, inputs=[p.seed])
        wp.launch(
            init_state,
            dim=n,
            inputs=[p.S, self.ppe, self.M, self.P, self.YAW, self.XF],
            block_dim=BLOCK,
        )
        for k in range(N_SEG):
            kb = k if self.grad else 0
            wp.launch(feat, dim=(n, PATCH_N), inputs=self._feat_args(k, kb), block_dim=BLOCK)
            self._gemm(net.w[0], self.Z[kb], net.b[0], self.H1[kb], True)
            self._gemm(net.w[1], self.H1[kb], net.b[1], self.H2[kb], True)
            self._gemm(net.w[2], self.H2[kb], net.b[2], self.H3[kb], True)
            self._gemm(net.w[3], self.H3[kb], net.b[3], self.O[kb], False)
            wp.launch(head, dim=n, inputs=self._head_args(k, kb), block_dim=BLOCK)
        if mode == 0:
            self.TOT.zero_()
        else:
            self.PL.zero_()
        wp.launch(loss, dim=n, inputs=self._loss_args(mode), block_dim=BLOCK)

    def backward(self) -> None:
        """d TOT / d U into aU (zeroed first)."""
        p, n, net = self.p, self.n, self.p.net
        for a in (self.aZ, self.aO, self.aP, self.aYAW, self.aXF, self.aPT, self.aTL, self.aLS, self.aLP, self.aU):  # fmt: skip
            a.zero_()
        wp.launch(
            loss,
            dim=n,
            inputs=self._loss_args(0),
            adj_inputs=self._loss_adj(),
            adjoint=True,
            block_dim=BLOCK,
        )
        for k in reversed(range(N_SEG)):
            wp.launch(
                head_adj,
                dim=n,
                inputs=[k, k, self.O, p.xm, p.xs, p.m_u, p.s_u, self.EPS, self.RU, self.P, self.YAW, p.VA,
                        p.DA, p.gx0, p.gy0, self.aPT, self.aTL, self.aLS, self.aO, self.aP, self.aYAW,
                        self.aXF],
                block_dim=BLOCK,
            )  # fmt: skip
            self._gemm_b(net.wt[3], self.aO[k], self.H3[k], self.aH3[k], True)
            self._gemm_b(net.wt[2], self.aH3[k], self.H2[k], self.aH2[k], True)
            self._gemm_b(net.wt[1], self.aH2[k], self.H1[k], self.aH1[k], True)
            self._gemm_b(net.wt[0], self.aH1[k], self.H1[k], self.aZ[k], False)
            wp.launch(
                feat_adj,
                dim=(n, PATCH_N),
                inputs=[k, k, self.M, self.P, self.YAW, self.XF, p.H, p.gx0, p.gy0, p.fmean, p.fstd, p.PXY,
                        self.aZ, self.aP, self.aYAW, self.aXF, self.aU],
                block_dim=BLOCK,
            )  # fmt: skip


def _import_torch() -> bool:
    """Import torch for the cuBLAS GEMMs; False when it is not installed."""
    global torch
    if torch is None:
        try:
            import torch as _torch
        except ImportError:
            return False
        torch = _torch
    return True


class NeuralPlanner:
    """The multistart gradient planner for one robot on fixed shapes, as one captured CUDA graph.

    Maps (H, VA, DA) are (n, n) float64 on the model grid with node (0, 0) at the window origin; write them in
    place between calls (the graph reads these buffers). Plans are in units of WHEEL_MAX, (N_SEG, 1, 2) = (wL, wR).
    """

    def __init__(
        self,
        weights: str,
        n_cells: int,
        n_init: int = 8,
        M: int = 16,
        steps: int = 20,
        K: int = 256,
        M_shoot: int = 4,
        M_final: int = 64,
        seed: int = 0,
        gemm: str = "tile",
    ) -> None:
        z = np.load(weights)
        for k, v in CONFIG.items():
            got = z["cfg_" + k].item()
            assert np.isclose(got, v), f"{weights}: {k} = {got}, the planner's kernels assume {v}"
        assert np.allclose(z["discs"], np.asarray(DISCS)), f"{weights}: footprint discs differ"
        assert gemm in ("auto", "cublas", "tile")
        self.cublas = gemm != "tile" and _import_torch()
        assert self.cublas or gemm != "cublas", "gemm='cublas' needs torch"
        self.n0, self.M, self.steps, self.K = n_init, M, steps, K
        self.net = Net(z)

        def f64(a: np.ndarray) -> wp.array:
            return wp.array(np.asarray(a, dtype=np.float64), dtype=F64)

        self.H = wp.zeros((n_cells, n_cells), dtype=F64)
        self.VA = wp.zeros((n_cells, n_cells), dtype=F64)
        self.DA = wp.full((n_cells, n_cells), 1.0e3, dtype=F64)
        self.gx0, self.gy0 = F64(0.0), F64(0.0)
        self.fmean, self.fstd = f64(z["f_mean"]), f64(z["f_std"])
        self.xm, self.xs = f64(z["x_mean"]), f64(z["x_std"])
        self.m_u, self.s_u = F64(z["x_mean"][I_UPMIN]), F64(z["x_std"][I_UPMIN])
        self.PXY = f64(z["patch_xy"])
        self.S, self.UP = wp.zeros((1, 16), dtype=F64), wp.zeros((1, 2), dtype=F64)
        self.U0 = wp.zeros((N_SEG, 1, 2), dtype=float)
        self.seed = wp.array([seed], dtype=int)
        self.C = wp.zeros((N_SEG, K, 2), dtype=float)
        self.UPL = wp.zeros((N_SEG, n_init, 2), dtype=float)
        self.m, self.v = wp.zeros_like(self.UPL), wp.zeros_like(self.UPL)
        self.step = wp.zeros(1, dtype=int)
        self.TAKEN = wp.zeros(K, dtype=int)
        self.OUT = wp.zeros((N_SEG, 1, 2), dtype=float)
        self.KNOTS = wp.zeros((N_SEG + 1, 1), dtype=wp.vec3d)
        self.shoot = Pass(self, K * M_shoot, M_shoot, K, self.C, grad=False)
        self.opt = Pass(self, n_init * M, M, n_init, self.UPL, grad=True)
        self.final = Pass(self, n_init * M_final, M_final, n_init, self.UPL, grad=False)
        self.M_final = M_final
        self.graph = None

    def _run(self) -> None:
        K, n0 = self.K, self.n0
        wp.launch(ground_z_kernel, dim=1, inputs=[self.H, self.S])
        wp.launch(shoot_cands, dim=(N_SEG, 1, K), inputs=[self.seed, self.U0, K, self.C])
        self.shoot.forward(1)
        self.TAKEN.zero_()
        wp.launch(
            select_inits,
            dim=1,
            inputs=[self.shoot.PL, K, n0, self.C, self.U0, self.TAKEN, self.UPL],
        )
        self.m.zero_()
        self.v.zero_()
        self.step.zero_()
        for _ in range(self.steps):
            self.opt.forward(0)
            self.opt.backward()
            wp.launch(bump, dim=1, inputs=[self.step])
            wp.launch(
                adam,
                dim=self.UPL.shape,
                inputs=[self.opt.aU, self.step, float(CONFIG["grad_lr"]), self.UPL, self.m, self.v],
            )
        self.final.forward(1)
        wp.launch(
            pick_best,
            dim=1,
            inputs=[self.final.PL, n0, self.M_final, self.UPL, self.final.P, self.final.YAW, self.OUT, self.KNOTS],  # fmt: skip
        )

    def _stream(self) -> contextlib.AbstractContextManager:
        """With cuBLAS GEMMs every Warp launch goes on torch's current stream (one stream to capture)."""
        if self.cublas:
            return wp.ScopedStream(wp.stream_from_torch(), sync_enter=False, sync_exit=False)
        return contextlib.nullcontext()

    def launch(self) -> None:
        """One re-plan from S, U0 and UP as set; the plan lands in OUT and its mean path in KNOTS (device)."""
        if self.cublas:
            wp.synchronize()  # the caller's Warp work on the maps is done before torch's stream reads them
            if self.graph is None:
                with self._stream():
                    self._run()  # compile the modules and create the cuBLAS handles outside the capture
                torch.cuda.synchronize()
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph), self._stream():
                    self._run()
            self.graph.replay()
            torch.cuda.synchronize()
        else:
            if self.graph is None:
                self._run()  # compile the modules outside the capture
                with wp.ScopedCapture() as cap:
                    self._run()
                self.graph = cap.graph
            wp.capture_launch(self.graph)


# --- the stack adapter -------------------------------------------------------------------------------------------------


def model_state(
    world_T_base: np.ndarray, origin_xy: tuple[float, float], twist: np.ndarray, wheels: np.ndarray
) -> np.ndarray:
    """The model's 16-d state from the node's measurements, in the MPPI window's frame.

    world_T_base the 4x4 base pose (map frame; roll and pitch matter, they are the up-vector feature),
    origin_xy the window origin in the map frame, twist (vx, vy, wz) in the base frame [m/s, rad/s],
    wheels (wL, wR, w_rear) [rad/s] (engine order). z is filled from the heightmap on the device.
    """
    R = np.asarray(world_T_base, np.float64)[:3, :3]
    # rotation matrix -> quaternion (x, y, z, w), the branch on the largest diagonal term
    tr = np.trace(R)
    if tr > 0.0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s])  # fmt: skip
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 1e-12))
        q = np.zeros(4)
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        q[3] = (R[k, j] - R[j, k]) / s
    v_world = R @ np.array([twist[0], twist[1], 0.0])
    w_world = R @ np.array([0.0, 0.0, twist[2]])
    S = np.zeros(16)
    S[0] = world_T_base[0, 3] - origin_xy[0]
    S[1] = world_T_base[1, 3] - origin_xy[1]
    S[3:7] = q
    S[7:10] = v_world
    S[10:13] = w_world
    S[13:16] = np.asarray(wheels, np.float64)
    return S


class NeuralMppi(MppiGpu):
    """MppiGpu with replan() done by NeuralPlanner; every setter the node calls is MppiGpu's own.

    Call set_robot_state() before every replan(). After replan() the nominal U [T, 2] is the plan's commands at
    the rollouts' dt (each 1 s command held), and rollout 0 of sim.controlled is the model's mean path, which the
    published path and the clearance governor read.
    """

    def __init__(
        self,
        sim: ForwardSimulator,
        cost: CostParams,
        sampling: SamplingConfig | None = None,
        n_theta: int = 16,
        weights: str = "",
        gemm: str = "tile",
        steps: int = 5,
        **kw,
    ) -> None:
        super().__init__(sim, cost, sampling=sampling, n_theta=n_theta, **kw)
        ny, nx = sim.elevation.shape
        cell = float(sim.grid.cell_size)
        self.n_cells = int(round(max(nx, ny) * cell / MODEL_CELL))
        with wp.ScopedDevice(self.device):
            self.neural = NeuralPlanner(weights, self.n_cells, steps=steps, gemm=gemm)
        self.u = np.full((N_SEG, 1, 2), 0.5)  # the warm start, in units of WHEEL_MAX
        self.S = np.zeros(16)
        self.S[6] = 1.0

    def set_robot_state(
        self,
        world_T_base: np.ndarray,
        origin_xy: tuple[float, float],
        twist: np.ndarray,
        wheels: np.ndarray,
    ) -> None:
        """This frame's state (model_state's arguments)."""
        self.S = model_state(world_T_base, origin_xy, twist, wheels)

    def _fields(self) -> None:
        """H, V and dist of this frame onto the model grid, in the planner's stable buffers."""
        p, n = self.neural, self.n_cells
        g = self.sim.grid
        wp.launch(
            resample_kernel,
            dim=(n, n),
            inputs=[self.sim.elevation, float(g.cell_size), float(g.origin_x), float(g.origin_y), p.H],  # fmt: skip
        )
        lg = self.lattice_grid
        use_ref = int(self.cw.track_lateral > 0.0 and W_TRACK > 0.0)
        wp.launch(
            value_kernel,
            dim=(n, n),
            inputs=[self.lattice_field, float(lg.cell_size), float(lg.origin_x), float(lg.origin_y),
                    self.ref_pts, self.ref_n, use_ref, W_TRACK, p.VA],
        )  # fmt: skip
        wp.launch(
            resample_kernel,
            dim=(n, n),
            inputs=[self.clear_field, float(g.cell_size), float(g.origin_x), float(g.origin_y), p.DA],  # fmt: skip
        )

    def replan(self, state: np.ndarray, goal_xy: tuple[float, float], n_refine: int) -> wp.array:
        p = self.neural
        with wp.ScopedDevice(self.device):
            self._fields()
            # warm start: the last plan shifted by one frame
            a = float(self.cw.dt) / SEG_T
            self.u[:-1] = (1.0 - a) * self.u[:-1] + a * self.u[1:]
            u_prev = np.clip(self.cmd_state.numpy()[0, :2].astype(np.float64) / WHEEL_MAX, 0.0, 1.0)
            p.S.assign(self.S[None])
            p.U0.assign(self.u.astype(np.float32))
            p.UP.assign(u_prev[None])
            p.launch()
            self.u = p.OUT.numpy().astype(np.float64)
            T = self.horizon
            k = np.minimum((np.arange(T) * float(self.cw.dt) / SEG_T).astype(int), N_SEG - 1)
            self.U.assign(np.ascontiguousarray(self.u[k, 0] * WHEEL_MAX, np.float32))
            yaw0 = float(np.arctan2(*self._yaw_sc()))
            wp.launch(
                path_kernel,
                dim=T + 1,
                inputs=[p.KNOTS, yaw0, float(self.cw.dt), self.sim.controlled],
            )
        return self.U

    def plan_particles(self) -> np.ndarray:
        """The chosen plan's scoring particles, (N_SEG + 1, M_final, 3) = (x, y, yaw) in the window frame, after
        replan(). A host readback: for recording, not for the control loop."""
        p = self.neural
        best = int(np.argmin(p.final.PL.numpy()))
        P = p.final.P.numpy()[:, best * p.M_final : (best + 1) * p.M_final]
        yaw = p.final.YAW.numpy()[:, best * p.M_final : (best + 1) * p.M_final]
        return np.concatenate([P[..., :2], yaw[..., None]], -1)

    def _yaw_sc(self) -> tuple[float, float]:
        """(sin, cos) of the state's heading, for unwrapping the path's headings."""
        x, y, z, w = self.S[3:7]
        return 2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)
