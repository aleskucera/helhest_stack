"""Device kinematics: the quasi-static settle + the monolithic forward step.

One Warp thread = one rollout. The 3x3 Newton settle runs in registers (numerical
Jacobian, fixed iters). Mirrors the numpy `placement`/`state` reference so that
stays the finite-diff oracle. Orientation: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).

Arg convention (Option 2): the differentiated grids (height, friction) are plain
`wp.array` kernel args; everything else rides in three structs --
`Grid` (terrain.py), `Robot` (robot.py), `Solver` (this module).

State across timesteps is two vec3s (avoids the length-6 spatial_vector type):
planar = (x, y, yaw) controlled DOF; tilt = (z, pitch, roll) derived DOF.
"""
from dataclasses import dataclass

import numpy as np
import warp as wp

from .robot import Robot
from .rotations import drot_x
from .rotations import drot_y
from .rotations import drot_z
from .rotations import euler_zyx
from .rotations import rot_x
from .rotations import rot_y
from .rotations import rot_z
from .terrain import _locate
from .terrain import Grid
from .terrain import sample_height
from .terrain import sample_height_grad
from .terrain import sample_normal

# Warp 1.13/1.14 ptxas MISCOMPILES this module's large combined `step` kernel at
# -O3 on CUDA: a register spill produces an invalid __local__ read (illegal
# memory access at runtime). Verified via compute-sanitizer + an -O level sweep
# (-O3 crashes; -O2/-O1/-O0 are correct). -O2 is correct and ~as fast as -O3, so
# pin this module to it. CPU is unaffected (defaults to -O2).
wp.set_module_options({"optimization_level": 2})


# --- settle/integration numerics: host params + the device-side `Solver` struct ---
@wp.struct
class Solver:
    """Device-side settle/integration numerics — the built form of SolverParams.
    All scalars -> safe as a struct.

    `k_turn` (the friction->alpha turning gain) rides along here as a non-diff
    scalar; promote it to a plain length-1 array only if d/dk is ever needed.
    """

    newton_iters: wp.int32
    max_step: wp.float32
    tilt_clamp: wp.float32
    dt: wp.float32
    k_turn: wp.float32


@dataclass
class SolverParams:  # settle/integration numerics — tuning, separate from the robot
    dt: float = 0.05
    newton_iters: int = 8
    max_step: float = 0.2
    tilt_clamp: float = 1.2
    k_turn: float = 2.0

    def build(self) -> Solver:
        s = Solver()
        s.newton_iters = self.newton_iters
        s.max_step = self.max_step
        s.tilt_clamp = self.tilt_clamp
        s.dt = self.dt
        s.k_turn = self.k_turn
        return s


@wp.func
def solve3(A: wp.mat33, b: wp.vec3):
    """Solve A x = b (3x3) via cofactors.

    Used instead of wp.inverse: Warp 1.13 miscompiles wp.inverse(mat33) on CUDA
    when two inverse call sites share a kernel (e.g. settle + normal_loads),
    causing illegal memory access. An explicit solve has a single code path.
    """
    a = A[0, 0]
    b1 = A[0, 1]
    c1 = A[0, 2]
    d = A[1, 0]
    e = A[1, 1]
    f = A[1, 2]
    g = A[2, 0]
    h = A[2, 1]
    i = A[2, 2]
    c00 = e * i - f * h
    c01 = -(d * i - f * g)
    c02 = d * h - e * g
    det = a * c00 + b1 * c01 + c1 * c02
    inv = 1.0 / det
    c10 = -(b1 * i - c1 * h)
    c11 = a * i - c1 * g
    c12 = -(a * h - b1 * g)
    c20 = b1 * f - c1 * e
    c21 = -(a * f - c1 * d)
    c22 = a * e - b1 * d
    return wp.vec3(
        (c00 * b[0] + c10 * b[1] + c20 * b[2]) * inv,
        (c01 * b[0] + c11 * b[1] + c21 * b[2]) * inv,
        (c02 * b[0] + c12 * b[1] + c22 * b[2]) * inv,
    )


@wp.func
def clearances(
    elevation: wp.array2d(dtype=wp.float32),
    g: Grid,
    robot: Robot,
    x: float,
    y: float,
    yaw: float,
    z: float,
    pitch: float,
    roll: float,
):
    """Signed wheel clearances c_i = hub_z - H_env(hub_xy) - R for the 3 wheels."""
    R = euler_zyx(yaw, pitch, roll)
    p = wp.vec3(x, y, z)
    h0 = p + R * robot.wheel_pos[0]
    h1 = p + R * robot.wheel_pos[1]
    h2 = p + R * robot.wheel_pos[2]
    c0 = h0[2] - sample_height(elevation, g, h0[0], h0[1]) - robot.wheel_radius
    c1 = h1[2] - sample_height(elevation, g, h1[0], h1[1]) - robot.wheel_radius
    c2 = h2[2] - sample_height(elevation, g, h2[0], h2[1]) - robot.wheel_radius
    return wp.vec3(c0, c1, c2)


@wp.func
def settle(
    elevation: wp.array2d(dtype=wp.float32),
    g: Grid,
    robot: Robot,
    sp: Solver,
    x: float,
    y: float,
    yaw: float,
    u_init: wp.vec3,
):
    """Solve (z, pitch, roll) with an ANALYTIC 3x3 Newton Jacobian.

    c_i = hub_iz - elevation(hub_ixy) - R, hub_i = (x,y,z) + Rz Ry Rx wheel_i.
    dc_i/dz = 1; dc_i/dpitch and dc_i/droll come from dR/dpitch, dR/droll applied
    to wheel_i, combined with the terrain gradient (gx,gy) at the hub. One
    euler + one value+grad sample per wheel per iter (vs 4 evals numerically).
    """
    Rz = rot_z(yaw)
    w0 = robot.wheel_pos[0]
    w1 = robot.wheel_pos[1]
    w2 = robot.wheel_pos[2]
    u = u_init
    for _ in range(sp.newton_iters):
        Ry = rot_y(u[1])
        Rx = rot_x(u[2])
        Rot = Rz * Ry * Rx
        dRp = Rz * drot_y(u[1]) * Rx  # d(Rot)/dpitch
        dRr = Rz * Ry * drot_x(u[2])  # d(Rot)/droll
        p = wp.vec3(x, y, u[0])
        hub0 = p + Rot * w0
        hub1 = p + Rot * w1
        hub2 = p + Rot * w2
        s0 = sample_height_grad(elevation, g, hub0[0], hub0[1])  # (h, gx, gy)
        s1 = sample_height_grad(elevation, g, hub1[0], hub1[1])
        s2 = sample_height_grad(elevation, g, hub2[0], hub2[1])
        c = wp.vec3(
            hub0[2] - s0[0] - robot.wheel_radius,
            hub1[2] - s1[0] - robot.wheel_radius,
            hub2[2] - s2[0] - robot.wheel_radius,
        )
        dp0 = dRp * w0
        dp1 = dRp * w1
        dp2 = dRp * w2
        dr0 = dRr * w0
        dr1 = dRr * w1
        dr2 = dRr * w2
        J = wp.mat33(
            1.0,
            dp0[2] - s0[1] * dp0[0] - s0[2] * dp0[1],
            dr0[2] - s0[1] * dr0[0] - s0[2] * dr0[1],
            1.0,
            dp1[2] - s1[1] * dp1[0] - s1[2] * dp1[1],
            dr1[2] - s1[1] * dr1[0] - s1[2] * dr1[1],
            1.0,
            dp2[2] - s2[1] * dp2[0] - s2[2] * dp2[1],
            dr2[2] - s2[1] * dr2[0] - s2[2] * dr2[1],
        )
        step = solve3(J, c)
        u = wp.vec3(
            u[0] - wp.clamp(step[0], -sp.max_step, sp.max_step),
            wp.clamp(
                u[1] - wp.clamp(step[1], -sp.max_step, sp.max_step), -sp.tilt_clamp, sp.tilt_clamp
            ),
            wp.clamp(
                u[2] - wp.clamp(step[2], -sp.max_step, sp.max_step), -sp.tilt_clamp, sp.tilt_clamp
            ),
        )
    return u


@wp.func
def _scatter_h(
    adj_elevation: wp.array2d(dtype=wp.float32), g: Grid, x: float, y: float, coef: float
):
    """Accumulate coef * (bilinear weights of (x,y)) into the elevation adjoint array.

    This is d(sample_height)/dH at (x,y): the same 4-node stencil sample_height
    reads (via the shared `_locate`), scattered with atomics (many output cells may
    hit the same node).
    """
    c = _locate(g, x, y)
    wp.atomic_add(adj_elevation, c.y_idx, c.x_idx, coef * (1.0 - c.frac_x) * (1.0 - c.frac_y))
    wp.atomic_add(adj_elevation, c.y_idx, c.x_idx + 1, coef * c.frac_x * (1.0 - c.frac_y))
    wp.atomic_add(adj_elevation, c.y_idx + 1, c.x_idx, coef * (1.0 - c.frac_x) * c.frac_y)
    wp.atomic_add(adj_elevation, c.y_idx + 1, c.x_idx + 1, coef * c.frac_x * c.frac_y)


@wp.func_grad(settle)
def adj_settle(
    elevation: wp.array2d(dtype=wp.float32),
    g: Grid,
    robot: Robot,
    sp: Solver,
    x: float,
    y: float,
    yaw: float,
    u_init: wp.vec3,
    adj_ret: wp.vec3,
):
    """Implicit (IFT) adjoint of the settle. adj_ret = cotangent on u*.

    With residual c(u*, x, y, yaw, elevation) = 0 and J = dc/du at u*:
        lambda = J^-T adj_ret
        adj_theta = -(dc/dtheta)^T lambda   for theta in {x, y, yaw, elevation}
    Everything analytic (same J as the forward + closed-form pose derivatives from
    the rotation derivatives and the terrain gradient). d/du_init = 0 (root
    independent of warm start).
    """
    u = settle(elevation, g, robot, sp, x, y, yaw, u_init)  # recompute converged u*
    Rz = rot_z(yaw)
    Ry = rot_y(u[1])
    Rx = rot_x(u[2])
    Rot = Rz * Ry * Rx
    dRp = Rz * drot_y(u[1]) * Rx
    dRr = Rz * Ry * drot_x(u[2])
    dRyaw = drot_z(yaw) * Ry * Rx
    w0 = robot.wheel_pos[0]
    w1 = robot.wheel_pos[1]
    w2 = robot.wheel_pos[2]
    p = wp.vec3(x, y, u[0])
    hub0 = p + Rot * w0
    hub1 = p + Rot * w1
    hub2 = p + Rot * w2
    s0 = sample_height_grad(elevation, g, hub0[0], hub0[1])  # (h, gx, gy)
    s1 = sample_height_grad(elevation, g, hub1[0], hub1[1])
    s2 = sample_height_grad(elevation, g, hub2[0], hub2[1])

    dp0 = dRp * w0
    dp1 = dRp * w1
    dp2 = dRp * w2
    dr0 = dRr * w0
    dr1 = dRr * w1
    dr2 = dRr * w2
    J = wp.mat33(
        1.0,
        dp0[2] - s0[1] * dp0[0] - s0[2] * dp0[1],
        dr0[2] - s0[1] * dr0[0] - s0[2] * dr0[1],
        1.0,
        dp1[2] - s1[1] * dp1[0] - s1[2] * dp1[1],
        dr1[2] - s1[1] * dr1[0] - s1[2] * dr1[1],
        1.0,
        dp2[2] - s2[1] * dp2[0] - s2[2] * dp2[1],
        dr2[2] - s2[1] * dr2[0] - s2[2] * dr2[1],
    )
    lam = solve3(wp.transpose(J), adj_ret)

    # pose adjoints: adj_x = sum gx_i lam_i, adj_y = sum gy_i lam_i (since dc_i/dx = -gx_i)
    wp.adjoint[x] += s0[1] * lam[0] + s1[1] * lam[1] + s2[1] * lam[2]
    wp.adjoint[y] += s0[2] * lam[0] + s1[2] * lam[1] + s2[2] * lam[2]
    dy0 = dRyaw * w0
    dy1 = dRyaw * w1
    dy2 = dRyaw * w2
    cw0 = dy0[2] - s0[1] * dy0[0] - s0[2] * dy0[1]
    cw1 = dy1[2] - s1[1] * dy1[0] - s1[2] * dy1[1]
    cw2 = dy2[2] - s2[1] * dy2[0] - s2[2] * dy2[1]
    wp.adjoint[yaw] += -(cw0 * lam[0] + cw1 * lam[1] + cw2 * lam[2])

    # elevation adjoint: adj_H[node] += lambda_i * (stencil of hub_i)  (per wheel)
    _scatter_h(wp.adjoint[elevation], g, hub0[0], hub0[1], lam[0])
    _scatter_h(wp.adjoint[elevation], g, hub1[0], hub1[1], lam[1])
    _scatter_h(wp.adjoint[elevation], g, hub2[0], hub2[1], lam[2])


@wp.func
def normal_loads(
    elevation: wp.array2d(dtype=wp.float32), g: Grid, robot: Robot, R: wp.mat33, p: wp.vec3
):
    """Quasi-static contact normal loads N_i from gravity (3x3 force/torque solve).

    Row 0: vertical force balance Sum N_i n_iz = m g.
    Rows 1-2: horizontal torque balance about the CoM. Returns N = vec3(N0,N1,N2).
    `elevation` is the wheel-envelope grid (same surface the contacts sit on).
    """
    com_world = p + R * robot.com
    hub0 = p + R * robot.wheel_pos[0]
    hub1 = p + R * robot.wheel_pos[1]
    hub2 = p + R * robot.wheel_pos[2]
    n0 = sample_normal(elevation, g, hub0[0], hub0[1])
    n1 = sample_normal(elevation, g, hub1[0], hub1[1])
    n2 = sample_normal(elevation, g, hub2[0], hub2[1])
    r0 = (hub0 - robot.wheel_radius * n0) - com_world
    r1 = (hub1 - robot.wheel_radius * n1) - com_world
    r2 = (hub2 - robot.wheel_radius * n2) - com_world
    m0 = wp.cross(r0, n0)
    m1 = wp.cross(r1, n1)
    m2 = wp.cross(r2, n2)
    A = wp.mat33(n0[2], n1[2], n2[2], m0[0], m1[0], m2[0], m0[1], m1[1], m2[1])
    b = wp.vec3(robot.mass * robot.gravity, 0.0, 0.0)
    return solve3(A, b)


@wp.func
def chassis_clearance(
    elevation: wp.array2d(dtype=wp.float32), g: Grid, robot: Robot, R: wp.mat33, p: wp.vec3
):
    """Min signed clearance of the chassis bottom-face points above RAW terrain.

    Negative == high-centered (belly penetrates). `elevation` is the raw heightmap.
    """
    cmin = float(1.0e9)
    for i in range(robot.n_chassis):
        w = p + R * robot.chassis_pts[i]
        c = w[2] - sample_height(elevation, g, w[0], w[1])
        cmin = wp.min(cmin, c)
    return cmin


# ----------------------------------------------------------------------------
# forward step + rollout
# ----------------------------------------------------------------------------
@wp.kernel
def init_state(
    elevation: wp.array2d(dtype=wp.float32),
    g: Grid,
    robot: Robot,
    sp: Solver,
    pose: wp.array(dtype=wp.vec3),  # [B] (x, y, yaw)
    planar: wp.array2d(dtype=wp.vec3),  # [T+1, B] -> writes row 0
    tilt: wp.array2d(dtype=wp.vec3),
):  # [T+1, B] -> writes row 0
    tid = wp.tid()
    pc = pose[tid]
    z0 = sample_height(elevation, g, pc[0], pc[1]) + robot.wheel_radius
    u = settle(elevation, g, robot, sp, pc[0], pc[1], pc[2], wp.vec3(z0, 0.0, 0.0))
    planar[0, tid] = pc
    tilt[0, tid] = u  # (z, pitch, roll)


@wp.kernel
def step(
    t: int,
    envelope: wp.array2d(dtype=wp.float32),
    elevation: wp.array2d(dtype=wp.float32),
    g: Grid,
    friction: wp.array2d(dtype=wp.float32),
    gmu: Grid,
    robot: Robot,
    sp: Solver,
    omega: wp.array2d(dtype=wp.vec3),  # [T, B] (wL, wR, w_rear)
    planar: wp.array2d(dtype=wp.vec3),  # [T+1, B] (x, y, yaw)
    tilt: wp.array2d(dtype=wp.vec3),  # [T+1, B] (z, pitch, roll)
    loads_out: wp.array2d(dtype=wp.vec3),  # [T, B] N_i of the NEW state
    turn_out: wp.array2d(dtype=wp.vec2),  # [T, B] (alpha, x_icr) used this step
    clear_out: wp.array2d(dtype=float),  # [T, B] belly clearance of the NEW state
    resid_out: wp.array2d(dtype=float),
):  # [T, B] settle residual (max|c|) of NEW state
    tid = wp.tid()
    pc = planar[t, tid]
    tc = tilt[t, tid]
    x = pc[0]
    y = pc[1]
    yaw = pc[2]
    R = euler_zyx(yaw, tc[1], tc[2])
    p = wp.vec3(x, y, tc[0])

    # --- turning params from the CURRENT pose (loads + friction at contacts) ---
    N = normal_loads(envelope, g, robot, R, p)
    w0v = robot.wheel_pos[0]
    w1v = robot.wheel_pos[1]
    w2v = robot.wheel_pos[2]
    h0 = p + R * w0v
    h1 = p + R * w1v
    h2 = p + R * w2v
    ct0 = h0 - robot.wheel_radius * sample_normal(envelope, g, h0[0], h0[1])
    ct1 = h1 - robot.wheel_radius * sample_normal(envelope, g, h1[0], h1[1])
    ct2 = h2 - robot.wheel_radius * sample_normal(envelope, g, h2[0], h2[1])
    mw0 = sample_height(friction, gmu, ct0[0], ct0[1]) * N[0]
    mw1 = sample_height(friction, gmu, ct1[0], ct1[1]) * N[1]
    mw2 = sample_height(friction, gmu, ct2[0], ct2[1]) * N[2]
    sw = mw0 + mw1 + mw2
    x_icr = (mw0 * w0v[0] + mw1 * w1v[0] + mw2 * w2v[0]) / sw
    alpha = 1.0 + sp.k_turn * sw / (robot.gravity * robot.mass)

    # --- predict: twist through the CURRENT orientation, Euler integrate ---
    om = omega[t, tid]
    vx = robot.wheel_radius * (om[0] + om[1]) / 2.0
    wz = robot.wheel_radius * (om[1] - om[0]) / (2.0 * robot.half_track * alpha)
    vy = -x_icr * wz
    vw = R * wp.vec3(vx, vy, 0.0)
    xn = x + vw[0] * sp.dt
    yn = y + vw[1] * sp.dt
    yawn = yaw + wz * sp.dt

    # --- project: settle the new pose (warm-started from current tilt) ---
    u = settle(envelope, g, robot, sp, xn, yn, yawn, tc)
    planar[t + 1, tid] = wp.vec3(xn, yn, yawn)
    tilt[t + 1, tid] = u

    Rn = euler_zyx(yawn, u[1], u[2])
    pn = wp.vec3(xn, yn, u[0])
    loads_out[t, tid] = normal_loads(envelope, g, robot, Rn, pn)
    turn_out[t, tid] = wp.vec2(alpha, x_icr)
    clear_out[t, tid] = chassis_clearance(elevation, g, robot, Rn, pn)
    cres = clearances(envelope, g, robot, xn, yn, yawn, u[0], u[1], u[2])
    resid_out[t, tid] = wp.max(wp.max(wp.abs(cres[0]), wp.abs(cres[1])), wp.abs(cres[2]))
