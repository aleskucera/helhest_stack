"""Option-B implicit gradient for the settle (single settle, d/dHenv).

The settle u* solves c(u*, Henv) = 0 (3 wheel clearances). The forward Newton
runs DETACHED (not on the tape). For the backward we use the implicit function
theorem: with J = dc/du at u*,

    du*/dHenv = -J^{-1} dc/dHenv
    => adj_Henv = -(dc/dHenv)^T lambda,   where  J^T lambda = adj_u

We hand-solve the 3x3 transpose system for lambda, then run the *residual* kernel
c(u*, Henv) on the tape with cotangent -lambda; Warp autodiff turns that into the
bilinear-stencil scatter into Henv.grad. No Newton, no max on the tape.
"""

import numpy as np
import warp as wp
from helhest.engine import clearances
from helhest.engine import Grid
from helhest.engine import Robot
from helhest.engine import sample_field
from helhest.engine import settle
from helhest.engine import Solver
from helhest.engine.envelope import _contact_kernel
from helhest.engine.envelope import _gather_kernel


def wheel_envelope(elevation, cell_size, wheel_radius, device="cpu"):
    """Verification-only: allocate scratch + run the two engine envelope passes
    (raw elevation -> dilated). Carries elevation.requires_grad so the backward tape
    routes d(loss)/d(raw elevation) to the contact cell."""
    ny, nx = elevation.shape
    env_radius = int(np.ceil(wheel_radius / cell_size))
    contact_iy = wp.zeros((ny, nx), dtype=wp.int32, device=device)
    contact_ix = wp.zeros((ny, nx), dtype=wp.int32, device=device)
    contact_cap = wp.zeros((ny, nx), dtype=wp.float32, device=device)
    envelope = wp.zeros(
        (ny, nx), dtype=wp.float32, device=device, requires_grad=elevation.requires_grad
    )
    wp.launch(
        _contact_kernel,
        dim=elevation.shape,
        inputs=[elevation, float(cell_size), float(wheel_radius), env_radius],
        outputs=[contact_iy, contact_ix, contact_cap],
        device=device,
    )
    wp.launch(
        _gather_kernel,
        dim=elevation.shape,
        inputs=[elevation, contact_iy, contact_ix, contact_cap],
        outputs=[envelope],
        device=device,
    )
    return envelope


@wp.kernel
def _settle_only(
    Henv: wp.array2d(dtype=wp.float32),
    g: Grid,
    robot: Robot,
    sp: Solver,
    pose: wp.array(dtype=wp.vec3),
    u_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    x = pose[tid][0]
    y = pose[tid][1]
    yaw = pose[tid][2]
    z0 = sample_field(Henv, g, x, y) + robot.wheel_radius
    u_out[tid] = settle(Henv, g, robot, sp, wp.vec3(x, y, yaw), wp.vec3(z0, 0.0, 0.0))


@wp.kernel
def _settle_jac(
    Henv: wp.array2d(dtype=wp.float32),
    g: Grid,
    robot: Robot,
    pose: wp.array(dtype=wp.vec3),
    u_star: wp.array(dtype=wp.vec3),
    eps: float,
    Jout: wp.array(dtype=wp.mat33),
):
    tid = wp.tid()
    x = pose[tid][0]
    y = pose[tid][1]
    yaw = pose[tid][2]
    u = u_star[tid]
    c = clearances(Henv, g, robot, x, y, yaw, u[0], u[1], u[2])
    jz = (clearances(Henv, g, robot, x, y, yaw, u[0] + eps, u[1], u[2]) - c) / eps
    jp = (clearances(Henv, g, robot, x, y, yaw, u[0], u[1] + eps, u[2]) - c) / eps
    jr = (clearances(Henv, g, robot, x, y, yaw, u[0], u[1], u[2] + eps) - c) / eps
    Jout[tid] = wp.mat33(jz[0], jp[0], jr[0], jz[1], jp[1], jr[1], jz[2], jp[2], jr[2])


@wp.kernel
def _solve_jt(
    Jin: wp.array(dtype=wp.mat33),
    adj_u: wp.array(dtype=wp.vec3),
    minus_lam: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    lam = wp.inverse(wp.transpose(Jin[tid])) * adj_u[tid]
    minus_lam[tid] = -lam


@wp.kernel
def _residual(
    Henv: wp.array2d(dtype=wp.float32),
    g: Grid,
    robot: Robot,
    pose: wp.array(dtype=wp.vec3),
    u_star: wp.array(dtype=wp.vec3),
    c: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    x = pose[tid][0]
    y = pose[tid][1]
    yaw = pose[tid][2]
    u = u_star[tid]
    c[tid] = clearances(Henv, g, robot, x, y, yaw, u[0], u[1], u[2])


def dsettle_dHenv(env_hm, poses, adj_u, params, jac_eps=1e-4, device="cpu"):
    """Implicit grad d(sum_p adj_u_p . u*_p)/dHenv. Returns (grad_Henv, u_star)."""
    from helhest.engine import GridParams, RobotParams

    elev = wp.array(
        np.ascontiguousarray(env_hm.H, np.float32),
        dtype=wp.float32,
        device=device,
        requires_grad=True,
    )
    g = GridParams(env_hm.nx, env_hm.ny, env_hm.cell, env_hm.x0, env_hm.y0).build()
    robot = RobotParams().build(device)
    sp = params.build()
    B = len(poses)
    pose = wp.array(np.asarray(poses, np.float32), dtype=wp.vec3, device=device)

    # forward settle (detached) + Jacobian + transpose solve
    u_star = wp.zeros(B, dtype=wp.vec3, device=device)
    wp.launch(_settle_only, B, inputs=[elev, g, robot, sp, pose, u_star], device=device)
    J = wp.zeros(B, dtype=wp.mat33, device=device)
    wp.launch(
        _settle_jac, B, inputs=[elev, g, robot, pose, u_star, float(jac_eps), J], device=device
    )
    adj = wp.array(np.asarray(adj_u, np.float32), dtype=wp.vec3, device=device)
    minus_lam = wp.zeros(B, dtype=wp.vec3, device=device)
    wp.launch(_solve_jt, B, inputs=[J, adj, minus_lam], device=device)

    # residual VJP on the tape with cotangent -lambda -> scatters into Henv.grad
    c = wp.zeros(B, dtype=wp.vec3, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(_residual, B, inputs=[elev, g, robot, pose, u_star], outputs=[c], device=device)
    tape.backward(grads={c: minus_lam})
    return elev.grad.numpy(), u_star.numpy()


def _selftest():
    """Implicit d/dHenv vs finite differences (numpy settle oracle), on the
    nonzero (contact) cells only."""
    from helhest import heightmap as hmmod
    from helhest.reference import placement
    from helhest.engine import SolverParams

    wp.init()
    params = SolverParams(newton_iters=12)
    cases = [
        ("ramp", hmmod.ramp_scene(), [(2.0, 0.0, 0.0), (3.0, 0.3, 0.2)]),
        ("box", hmmod.box_scene(), [(0.9, 0.0, 0.0)]),
    ]
    adj_template = np.array([0.3, 1.0, 0.5], np.float32)  # weights on (z, pitch, roll)
    worst = 0.0
    for name, scene, poses in cases:
        env = hmmod.wheel_envelope(scene, 0.35)
        adj_u = np.tile(adj_template, (len(poses), 1))
        g_imp, _ = dsettle_dHenv(env, poses, adj_u, params)

        # FD only on cells the implicit grad marks nonzero (the contact stencils)
        cells = list(zip(*np.where(np.abs(g_imp) > 1e-6)))
        eps = 1e-3
        err = 0.0
        for i, j in cells:
            gp = _fd_loss(env, poses, adj_u, i, j, +eps)
            gm = _fd_loss(env, poses, adj_u, i, j, -eps)
            g_fd = (gp - gm) / (2 * eps)
            err = max(err, abs(g_imp[i, j] - g_fd))
        worst = max(worst, err)
        print(
            f"  {name:4s}  {len(cells)} contact cells  max|g_imp-g_fd|={err:.2e}  "
            f"||g||={np.abs(g_imp).max():.3f}"
        )
    print(
        f"implicit settle d/dHenv vs FD  worst={worst:.2e}  "
        f"{'OK' if worst < 5e-2 else 'REVIEW'}"
    )


def _fd_loss(env, poses, adj_u, i, j, delta):
    from helhest import heightmap as hmmod
    from helhest.reference import placement

    Hp = env.H.copy()
    Hp[i, j] += delta
    hm = hmmod.Heightmap(Hp, (env.x0, env.y0), env.cell)
    total = 0.0
    for p, (x, y, yaw) in enumerate(poses):
        s = placement.settle(x, y, yaw, hm)
        u = np.array([s["z"], s["pitch"], s["roll"]])
        total += float(adj_u[p] @ u)
    return total


@wp.kernel
def _row_loss(
    controlled: wp.array2d(dtype=wp.vec3),
    derived: wp.array2d(dtype=wp.vec3),
    wpv: wp.vec3,
    wtv: wp.vec3,
    row: int,
    loss: wp.array(dtype=float),
):
    tid = wp.tid()
    wp.atomic_add(loss, 0, wp.dot(wpv, controlled[row, tid]) + wp.dot(wtv, derived[row, tid]))


def _gmeta(hm):
    from helhest.engine import Grid

    g = Grid()
    g.origin_x, g.origin_y, g.cell_size = float(hm.x0), float(hm.y0), float(hm.cell)
    g.cells_x, g.cells_y = int(hm.nx), int(hm.ny)
    return g


def _fwd(envH, rawH, muH, g, robot, sp, omega_np, init_pose, wpv, wtv, grad=False):
    """Forward init + T steps + loss on the FINAL state (B=1). If grad, taped
    backward -> (loss, gHenv, gHmu). T inferred from omega_np."""
    from helhest.engine import init_state_kernel, step_kernel

    dev = "cpu"
    T = omega_np.shape[0]
    Henv = wp.array(envH, dtype=wp.float32, device=dev, requires_grad=grad)
    Hraw = wp.array(rawH, dtype=wp.float32, device=dev)
    Hmu = wp.array(muH, dtype=wp.float32, device=dev, requires_grad=grad)
    omega = wp.array(omega_np, dtype=wp.vec3, device=dev)
    pose0 = wp.array(np.asarray([init_pose], np.float32), dtype=wp.vec3, device=dev)
    controlled = wp.zeros((T + 1, 1), dtype=wp.vec3, device=dev, requires_grad=grad)
    derived = wp.zeros((T + 1, 1), dtype=wp.vec3, device=dev, requires_grad=grad)
    cur_omega = wp.zeros((T + 1, 1), dtype=wp.vec3, device=dev)  # lagged omega (non-diff)
    twist = wp.zeros((T + 1, 1), dtype=wp.vec3, device=dev)  # momentum state (non-diff)
    mu_scale = wp.full(1, 1.0, dtype=float, device=dev)
    loads = wp.zeros((T, 1), dtype=wp.vec3, device=dev)
    turn = wp.zeros((T, 1), dtype=wp.vec2, device=dev)
    clear = wp.zeros((T, 1), dtype=float, device=dev)
    clear_soft = wp.zeros((T, 1), dtype=float, device=dev)
    resid = wp.zeros((T, 1), dtype=float, device=dev)
    loss = wp.zeros(1, dtype=float, device=dev, requires_grad=grad)

    def launches():
        wp.launch(
            init_state_kernel,
            1,
            inputs=[Henv, g, robot, sp, pose0],
            outputs=[controlled, derived],
            device=dev,
        )
        for t in range(T):
            wp.launch(
                step_kernel,
                1,
                inputs=[
                    Henv,
                    Hraw,
                    Hmu,
                    mu_scale,
                    g,
                    robot,
                    sp,
                    omega[t],
                    cur_omega[t],
                    controlled[t],
                    derived[t],
                    twist[t],
                ],
                outputs=[
                    cur_omega[t + 1],
                    controlled[t + 1],
                    derived[t + 1],
                    loads[t],
                    turn[t],
                    clear[t],
                    clear_soft[t],
                    resid[t],
                    twist[t + 1],
                ],
                device=dev,
            )
        wp.launch(
            _row_loss, 1, inputs=[controlled, derived, wpv, wtv, T], outputs=[loss], device=dev
        )

    if not grad:
        launches()
        return float(loss.numpy()[0])
    tape = wp.Tape()
    with tape:
        launches()
    tape.backward(loss=loss)
    return float(loss.numpy()[0]), Henv.grad.numpy(), Hmu.grad.numpy()


def _selftest_step_grad():
    """One full step on the tape: d(loss)/dHenv and d(loss)/dHmu vs finite diff."""
    from helhest import friction
    from helhest import heightmap as hmmod
    from helhest.engine import RobotParams, SolverParams

    wp.init()
    scene = hmmod.flat()
    env = hmmod.wheel_envelope(scene, 0.35)
    mu = friction.uniform(0.8)
    robot = RobotParams().build("cpu")
    sp = SolverParams(newton_iters=12, dt=0.05, k_turn=2.0).build()
    g = _gmeta(env)
    omega_np = np.array([[[1.0, 2.0, 1.5]]], np.float32)  # [T=1,B=1,3]: a turn
    wpv, wtv = wp.vec3(0.5, 0.3, 1.0), wp.vec3(0.2, 1.0, 0.5)
    init_pose = (0.0, 0.0, 0.0)

    envH = np.ascontiguousarray(env.H, np.float32)
    rawH = np.ascontiguousarray(scene.H, np.float32)
    muH = np.ascontiguousarray(mu.H, np.float32)

    _, gHenv, gHmu = _fwd(envH, rawH, muH, g, robot, sp, omega_np, init_pose, wpv, wtv, grad=True)

    eps = 1e-3
    err_e = _fd_grid(
        envH, rawH, muH, g, robot, sp, omega_np, init_pose, wpv, wtv, gHenv, "env", eps
    )
    err_m = _fd_grid(envH, rawH, muH, g, robot, sp, omega_np, init_pose, wpv, wtv, gHmu, "mu", eps)
    worst = max(err_e, err_m)
    print(
        f"  dHenv: {np.count_nonzero(np.abs(gHenv) > 1e-5)} cells ||g||={np.abs(gHenv).max():.3f}  "
        f"max|err|={err_e:.2e}"
    )
    print(
        f"  dHmu : {np.count_nonzero(np.abs(gHmu) > 1e-5)} cells ||g||={np.abs(gHmu).max():.3f}  "
        f"max|err|={err_m:.2e}"
    )
    print(
        f"step grad d/dHenv,d/dHmu vs FD  worst={worst:.2e}  {'OK' if worst < 5e-2 else 'REVIEW'}"
    )


def _fd_grid(envH, rawH, muH, g, robot, sp, omega_np, init_pose, wpv, wtv, g_an, which, eps):
    cells = list(zip(*np.where(np.abs(g_an) > 1e-5)))
    err = 0.0
    for i, j in cells:

        def loss_at(delta):
            e, r, m = envH.copy(), rawH.copy(), muH.copy()
            (e if which == "env" else m)[i, j] += delta
            return _fwd(e, r, m, g, robot, sp, omega_np, init_pose, wpv, wtv)

        g_fd = (loss_at(+eps) - loss_at(-eps)) / (2 * eps)
        err = max(err, abs(g_an[i, j] - g_fd))
    return err


def _fwd_h(rawH, muH, g, Rwheel, robot, sp, omega_np, init_pose, wpv, wtv, grad=False):
    """Like _fwd but the leaf is the RAW heightmap: Henv = wheel_envelope(rawH) is
    computed on the tape, so backward yields d(loss)/d(raw h)."""
    from helhest.engine import init_state_kernel, step_kernel

    dev = "cpu"
    T = omega_np.shape[0]
    Hraw = wp.array(rawH, dtype=wp.float32, device=dev, requires_grad=grad)
    Hmu = wp.array(muH, dtype=wp.float32, device=dev, requires_grad=grad)
    omega = wp.array(omega_np, dtype=wp.vec3, device=dev)
    pose0 = wp.array(np.asarray([init_pose], np.float32), dtype=wp.vec3, device=dev)
    controlled = wp.zeros((T + 1, 1), dtype=wp.vec3, device=dev, requires_grad=grad)
    derived = wp.zeros((T + 1, 1), dtype=wp.vec3, device=dev, requires_grad=grad)
    cur_omega = wp.zeros((T + 1, 1), dtype=wp.vec3, device=dev)  # lagged omega (non-diff)
    twist = wp.zeros((T + 1, 1), dtype=wp.vec3, device=dev)  # momentum state (non-diff)
    mu_scale = wp.full(1, 1.0, dtype=float, device=dev)
    loads = wp.zeros((T, 1), dtype=wp.vec3, device=dev)
    turn = wp.zeros((T, 1), dtype=wp.vec2, device=dev)
    clear = wp.zeros((T, 1), dtype=float, device=dev)
    clear_soft = wp.zeros((T, 1), dtype=float, device=dev)
    resid = wp.zeros((T, 1), dtype=float, device=dev)
    loss = wp.zeros(1, dtype=float, device=dev, requires_grad=grad)

    def launches():
        Henv = wheel_envelope(Hraw, g.cell_size, Rwheel, dev)  # raw h -> envelope, on the tape
        wp.launch(
            init_state_kernel,
            1,
            inputs=[Henv, g, robot, sp, pose0],
            outputs=[controlled, derived],
            device=dev,
        )
        for t in range(T):
            wp.launch(
                step_kernel,
                1,
                inputs=[
                    Henv,
                    Hraw,
                    Hmu,
                    mu_scale,
                    g,
                    robot,
                    sp,
                    omega[t],
                    cur_omega[t],
                    controlled[t],
                    derived[t],
                    twist[t],
                ],
                outputs=[
                    cur_omega[t + 1],
                    controlled[t + 1],
                    derived[t + 1],
                    loads[t],
                    turn[t],
                    clear[t],
                    clear_soft[t],
                    resid[t],
                    twist[t + 1],
                ],
                device=dev,
            )
        wp.launch(
            _row_loss, 1, inputs=[controlled, derived, wpv, wtv, T], outputs=[loss], device=dev
        )

    if not grad:
        launches()
        return float(loss.numpy()[0])
    tape = wp.Tape()
    with tape:
        launches()
    tape.backward(loss=loss)
    return float(loss.numpy()[0]), Hraw.grad.numpy(), Hmu.grad.numpy()


def _selftest_dh():
    """End-to-end d(loss)/d(raw h) through the envelope dilation + rollout vs FD."""
    from helhest import friction
    from helhest import heightmap as hmmod
    from helhest.engine import RobotParams, SolverParams

    wp.init()
    T, R = 8, 0.35
    scene = hmmod.ramp_scene()  # sloped: envelope arg-max routes uphill (non-trivial)
    mu = friction.uniform(0.8, xlim=(scene.x0, scene.x0 + (scene.nx - 1) * scene.cell))
    robot = RobotParams().build("cpu")
    sp = SolverParams(newton_iters=12, dt=0.05, k_turn=2.0).build()
    g = _gmeta(scene)
    omega_np = np.tile([1.0, 2.0, 1.5], (T, 1, 1)).astype(np.float32)
    wpv, wtv = wp.vec3(0.5, 0.3, 1.0), wp.vec3(0.2, 1.0, 0.5)
    init_pose = (1.5, 0.0, 0.0)  # on the slope

    rawH = np.ascontiguousarray(scene.H, np.float32)
    muH = np.ascontiguousarray(mu.H, np.float32)

    _, gH, gMu = _fwd_h(rawH, muH, g, R, robot, sp, omega_np, init_pose, wpv, wtv, grad=True)

    eps = 1e-3
    fwd = lambda rh, mh: _fwd_h(rh, mh, g, R, robot, sp, omega_np, init_pose, wpv, wtv)
    err_h = _fd_cells(rawH, muH, gH, "raw", eps, fwd)
    err_m = _fd_cells(rawH, muH, gMu, "mu", eps, fwd)
    worst = max(err_h, err_m)
    print(
        f"  d/d(raw h): {np.count_nonzero(np.abs(gH) > 1e-5)} cells "
        f"||g||={np.abs(gH).max():.3f}  max|err|={err_h:.2e}"
    )
    print(
        f"  d/dHmu    : {np.count_nonzero(np.abs(gMu) > 1e-5)} cells "
        f"||g||={np.abs(gMu).max():.3f}  max|err|={err_m:.2e}"
    )
    print(f"end-to-end d/d(raw h) vs FD  worst={worst:.2e}  {'OK' if worst < 5e-2 else 'REVIEW'}")


def _fd_cells(rawH, muH, g_an, which, eps, fwd):
    cells = list(zip(*np.where(np.abs(g_an) > 1e-5)))
    err = 0.0
    for i, j in cells:
        rp, mp = rawH.copy(), muH.copy()
        rm, mm = rawH.copy(), muH.copy()
        (rp if which == "raw" else mp)[i, j] += eps
        (rm if which == "raw" else mm)[i, j] -= eps
        g_fd = (fwd(rp, mp) - fwd(rm, mm)) / (2 * eps)
        err = max(err, abs(g_an[i, j] - g_fd))
    return err


def _fwd_batch(envH, rawH, muH, g, robot, sp, omega_np, poses, wpv, wtv, grad=False):
    """Batched (B>1) forward init + T steps + summed loss over all rollouts.
    omega_np: [T, B, 3]; poses: [B, 3]. Grads accumulate into shared Henv/Hmu."""
    from helhest.engine import init_state_kernel, step_kernel

    dev = "cpu"
    T, B = omega_np.shape[0], omega_np.shape[1]
    Henv = wp.array(envH, dtype=wp.float32, device=dev, requires_grad=grad)
    Hraw = wp.array(rawH, dtype=wp.float32, device=dev)
    Hmu = wp.array(muH, dtype=wp.float32, device=dev, requires_grad=grad)
    omega = wp.array(omega_np, dtype=wp.vec3, device=dev)
    pose0 = wp.array(np.asarray(poses, np.float32), dtype=wp.vec3, device=dev)
    controlled = wp.zeros((T + 1, B), dtype=wp.vec3, device=dev, requires_grad=grad)
    derived = wp.zeros((T + 1, B), dtype=wp.vec3, device=dev, requires_grad=grad)
    cur_omega = wp.zeros((T + 1, B), dtype=wp.vec3, device=dev)  # lagged omega (non-diff)
    twist = wp.zeros((T + 1, B), dtype=wp.vec3, device=dev)  # momentum state (non-diff)
    mu_scale = wp.full(B, 1.0, dtype=float, device=dev)
    loads = wp.zeros((T, B), dtype=wp.vec3, device=dev)
    turn = wp.zeros((T, B), dtype=wp.vec2, device=dev)
    clear = wp.zeros((T, B), dtype=float, device=dev)
    clear_soft = wp.zeros((T, B), dtype=float, device=dev)
    resid = wp.zeros((T, B), dtype=float, device=dev)
    loss = wp.zeros(1, dtype=float, device=dev, requires_grad=grad)

    def launches():
        wp.launch(
            init_state_kernel,
            B,
            inputs=[Henv, g, robot, sp, pose0],
            outputs=[controlled, derived],
            device=dev,
        )
        for t in range(T):
            wp.launch(
                step_kernel,
                B,
                inputs=[
                    Henv,
                    Hraw,
                    Hmu,
                    mu_scale,
                    g,
                    robot,
                    sp,
                    omega[t],
                    cur_omega[t],
                    controlled[t],
                    derived[t],
                    twist[t],
                ],
                outputs=[
                    cur_omega[t + 1],
                    controlled[t + 1],
                    derived[t + 1],
                    loads[t],
                    turn[t],
                    clear[t],
                    clear_soft[t],
                    resid[t],
                    twist[t + 1],
                ],
                device=dev,
            )
        wp.launch(
            _row_loss, B, inputs=[controlled, derived, wpv, wtv, T], outputs=[loss], device=dev
        )

    if not grad:
        launches()
        return float(loss.numpy()[0])
    tape = wp.Tape()
    with tape:
        launches()
    tape.backward(loss=loss)
    return float(loss.numpy()[0]), Henv.grad.numpy(), Hmu.grad.numpy()


def _selftest_batch():
    """Batched B rollouts == sum of B solo rollouts (forward loss + grads)."""
    from helhest import friction
    from helhest import heightmap as hmmod
    from helhest.engine import RobotParams, SolverParams

    wp.init()
    T, B = 5, 4
    scene = hmmod.flat()
    env = hmmod.wheel_envelope(scene, 0.35)
    mu = friction.uniform(0.8)
    robot = RobotParams().build("cpu")
    sp = SolverParams(newton_iters=12, dt=0.05, k_turn=2.0).build()
    g = _gmeta(env)
    wpv, wtv = wp.vec3(0.5, 0.3, 1.0), wp.vec3(0.2, 1.0, 0.5)

    # B distinct rollouts: different turn rates and start poses
    omega = np.stack([np.tile([1.0, 1.0 + 0.4 * b, 1.5], (T, 1)) for b in range(B)], axis=1)
    omega = omega.astype(np.float32)  # [T, B, 3]
    poses = np.array([[0.0, 0.3 * b, 0.1 * b] for b in range(B)], np.float32)

    envH = np.ascontiguousarray(env.H, np.float32)
    rawH = np.ascontiguousarray(scene.H, np.float32)
    muH = np.ascontiguousarray(mu.H, np.float32)

    lb, gHb, gMb = _fwd_batch(envH, rawH, muH, g, robot, sp, omega, poses, wpv, wtv, grad=True)

    ls, gHs, gMs = 0.0, np.zeros_like(gHb), np.zeros_like(gMb)
    for b in range(B):
        lo, gh, gm = _fwd_batch(
            envH,
            rawH,
            muH,
            g,
            robot,
            sp,
            omega[:, b : b + 1],
            poses[b : b + 1],
            wpv,
            wtv,
            grad=True,
        )
        ls += lo
        gHs += gh
        gMs += gm

    d_loss = abs(lb - ls)
    d_H = np.abs(gHb - gHs).max()
    d_M = np.abs(gMb - gMs).max()
    worst = max(d_loss, d_H, d_M)
    print(
        f"  B={B} T={T}  |loss_batch-sum_solo|={d_loss:.2e}  " f"dgHenv={d_H:.2e}  dgHmu={d_M:.2e}"
    )
    print(f"batch == sum-of-solo  worst={worst:.2e}  {'OK' if worst < 1e-3 else 'REVIEW'}")


def _selftest_bptt():
    """BPTT over a T-step rollout: d(loss on final state)/dHenv,dHmu vs finite diff."""
    from helhest import friction
    from helhest import heightmap as hmmod
    from helhest.engine import RobotParams, SolverParams

    wp.init()
    T = 8
    scene = hmmod.flat()
    env = hmmod.wheel_envelope(scene, 0.35)
    mu = friction.uniform(0.8)
    robot = RobotParams().build("cpu")
    sp = SolverParams(newton_iters=12, dt=0.05, k_turn=2.0).build()
    g = _gmeta(env)
    omega_np = np.tile([1.0, 2.0, 1.5], (T, 1, 1)).astype(np.float32)  # [T,1,3]: a turn
    wpv, wtv = wp.vec3(0.5, 0.3, 1.0), wp.vec3(0.2, 1.0, 0.5)
    init_pose = (0.0, 0.0, 0.0)

    envH = np.ascontiguousarray(env.H, np.float32)
    rawH = np.ascontiguousarray(scene.H, np.float32)
    muH = np.ascontiguousarray(mu.H, np.float32)

    _, gHenv, gHmu = _fwd(envH, rawH, muH, g, robot, sp, omega_np, init_pose, wpv, wtv, grad=True)

    eps = 1e-3
    err_e = _fd_grid(
        envH, rawH, muH, g, robot, sp, omega_np, init_pose, wpv, wtv, gHenv, "env", eps
    )
    err_m = _fd_grid(envH, rawH, muH, g, robot, sp, omega_np, init_pose, wpv, wtv, gHmu, "mu", eps)
    worst = max(err_e, err_m)
    print(
        f"  T={T}  dHenv: {np.count_nonzero(np.abs(gHenv) > 1e-5)} cells "
        f"||g||={np.abs(gHenv).max():.3f}  max|err|={err_e:.2e}"
    )
    print(
        f"  T={T}  dHmu : {np.count_nonzero(np.abs(gHmu) > 1e-5)} cells "
        f"||g||={np.abs(gHmu).max():.3f}  max|err|={err_m:.2e}"
    )
    print(f"BPTT d/dHenv,d/dHmu vs FD  worst={worst:.2e}  {'OK' if worst < 5e-2 else 'REVIEW'}")


# ---------------------------------------------------------------------------------------------
# the CYLINDER wheel: the same d(loss)/d(raw elevation) check, through the taped patch path
# ---------------------------------------------------------------------------------------------
# Everything above runs the SPHERE, whose envelope is one dilated grid. The cylinder element is
# yaw-dependent, so `DifferentiableSimulator` builds a local envelope patch per (rollout, timestep)
# in that step's yaw bin (`engine/envelope.py`). The derivative is taken exactly as before -- the
# arg-max frozen off the tape, the gather differentiated -- so the check that it is RIGHT is the
# same one: perturb a raw elevation cell, and compare the analytic gradient against a central
# difference of the true forward, which re-runs the arg-max and so includes contact switching.
#
# CUDA-only (`DifferentiableSimulator` is), and per-rollout rather than shared-terrain, so it
# cannot reuse `_fwd_h`'s CPU scaffolding. The rollouts differ only in START HEADING, which is what
# puts the "at several yaws" in this check: each one dilates with a different bin's element.


@wp.kernel
def _term_kernel(
    derived: wp.array2d(dtype=wp.vec3),  # [T+1, B] (z, pitch, roll)
    controlled: wp.array2d(dtype=wp.vec3),  # [T+1, B] (x, y, yaw)
    w_der: wp.vec3,
    w_ctrl: wp.vec3,
    n_steps: int,
    terms: wp.array(dtype=float),  # [B] this rollout's scalar -> written
):
    """Per-ROLLOUT scalar: the settle weighted over the whole horizon, plus the final pose.

    Per rollout, not one batch sum, for finite differences' sake: a perturbation of rollout b's
    terrain only moves term b, and differencing a number of size ~10 in float32 resolves ~1e-6
    where differencing the batch sum would resolve ~1e-5 and drown the gradient of a single cell.
    """
    b = wp.tid()
    acc = wp.dot(w_ctrl, controlled[n_steps, b])
    for t in range(n_steps + 1):
        acc += wp.dot(w_der, derived[t, b])
    terms[b] = acc


W_DER = wp.vec3(1.0, 0.7, 0.5)  # (z, pitch, roll): the study's settle-cost weights
W_CTRL = wp.vec3(0.3, 0.3, 0.2)


def _cyl_sim(scene, poses, omega, wheel_width, cell, origin, device="cuda:0"):
    """A `DifferentiableSimulator` on `scene` (broadcast to one slice per rollout) + its loss."""
    from helhest.engine import DifferentiableSimulator, GridParams, RobotParams, SolverParams

    ny, nx = scene.shape
    B, T = len(poses), omega.shape[0]
    sim = DifferentiableSimulator(
        RobotParams(wheel_width=wheel_width),
        SolverParams(),
        GridParams(nx, ny, cell, *origin),
        B,
        T,
        device=device,
    )
    with wp.ScopedDevice(device):
        sim.set_terrain(wp.array(np.repeat(scene[None], B, 0), dtype=wp.float32))
        sim.set_friction(wp.array(np.full((B, ny, nx), 0.7, np.float32), dtype=wp.float32))
        sim.terms = wp.zeros(B, dtype=float, requires_grad=True)
    sim.start_pose.assign(np.ascontiguousarray(poses, np.float32))
    sim.target_wheel_omega.assign(np.ascontiguousarray(omega, np.float32))
    return sim


@wp.kernel
def _sum_terms(terms: wp.array(dtype=float), loss: wp.array(dtype=float)):
    wp.atomic_add(loss, 0, terms[wp.tid()])


def _cyl_loss(sim):
    """Sum of the per-rollout terms -- the scalar the tape backpropagates."""
    loss = wp.zeros(1, dtype=float, device=sim.device, requires_grad=True)
    wp.launch(
        _term_kernel,
        sim.batch_size,
        inputs=[sim.derived, sim.controlled, W_DER, W_CTRL, sim.n_steps],
        outputs=[sim.terms],
        device=sim.device,
    )
    wp.launch(_sum_terms, sim.batch_size, inputs=[sim.terms], outputs=[loss], device=sim.device)
    return loss


def _cyl_scene(nx, ny, cell, origin, seed=20260826):
    rng = np.random.default_rng(seed)
    xs = origin[0] + cell * np.arange(nx)
    ys = origin[1] + cell * np.arange(ny)
    X, Y = np.meshgrid(xs, ys)
    h = 0.05 * X
    for _ in range(30):
        h += rng.uniform(-0.10, 0.22) * np.exp(
            -((X - rng.uniform(xs[0], xs[-1])) ** 2 + (Y - rng.uniform(ys[0], ys[-1])) ** 2)
            / (2.0 * rng.uniform(0.3, 0.9) ** 2)
        )
    return np.ascontiguousarray(h, np.float32)


def _cyl_fd_stats(wheel_width, batch, steps, eps, n_cells, device, seed=7):
    """Analytic vs central-difference d(loss)/d(raw elevation) on `n_cells` cells per rollout.

    Returns (stats dict, worst-cell probe). The forward is re-run for every perturbation, so the
    difference quotient sees the arg-max recomputed -- contact switching included, not assumed away.
    """
    cell, origin, nx, ny = 0.1, (-4.5, -4.5), 91, 91
    scene = _cyl_scene(nx, ny, cell, origin)
    yaws = np.linspace(0.0, np.pi, batch, endpoint=False) + 0.11
    poses = np.stack(
        [np.full(batch, -1.2), np.linspace(-0.8, 0.8, batch), yaws], axis=1
    ).astype(np.float32)
    rng = np.random.default_rng(seed)
    omega = rng.uniform(0.8, 2.0, (steps, batch, 3)).astype(np.float32)

    sim = _cyl_sim(scene, poses, omega, wheel_width, cell, origin, device)
    sim.rollout_taped(_cyl_loss)
    sim.backward()
    g_an = sim.elevation.grad.numpy().copy()
    stack = np.repeat(scene[None], batch, 0).astype(np.float32)

    def fd_at(b, iy, ix, step):
        pert = stack.copy()
        pert[b, iy, ix] += step
        sim.elevation.assign(np.ascontiguousarray(pert, np.float32))
        sim.rollout_taped(_cyl_loss)
        up = sim.terms.numpy()[b]
        pert[b, iy, ix] -= 2 * step
        sim.elevation.assign(np.ascontiguousarray(pert, np.float32))
        sim.rollout_taped(_cyl_loss)
        return (up - sim.terms.numpy()[b]) / (2 * step)

    err, worst = [], (0.0, None)
    for b in range(batch):
        mag = np.abs(g_an[b])
        cand = np.argwhere(mag > 0.05 * mag.max())
        pick = cand[rng.choice(len(cand), size=min(n_cells, len(cand)), replace=False)]
        for iy, ix in pick:
            fd = fd_at(b, iy, ix, eps)
            e = abs(g_an[b, iy, ix] - fd)
            err.append(e)
            if e > worst[0]:
                worst = (e, (b, int(iy), int(ix), float(g_an[b, iy, ix]), float(fd)))
    err = np.array(err)
    stats = {
        "n": len(err),
        "scale": float(np.abs(g_an).max()),
        "median": float(np.median(err)),
        "p90": float(np.percentile(err, 90)),
        "max": float(err.max()),
        "bins": sorted(int(v) for v in set(sim._patch_bin.numpy()[0]))
        if wheel_width is not None
        else [],
    }
    # the worst cell again at a fifth of the step: an active-set switch shrinks with eps, a wrong
    # chain rule does not
    b, iy, ix, ga, fd = worst[1]
    probe = (b, iy, ix, ga, fd, fd_at(b, iy, ix, eps / 5.0))
    del sim
    return stats, probe


def _selftest_cylinder_dh(batch=6, steps=12, eps=1e-4, n_cells=6, device="cuda:0"):
    """d(loss)/d(raw elevation) through the CYLINDER patch path vs FD, at `batch` start headings.

    Run against the SPHERE through the same scene, loss, poses and step, because that is the path
    whose gradient the rest of this file (and Section III) already validates: the claim being
    tested is not that the cylinder gradient is exact -- no frozen-arg-max gradient is, once the
    active set moves -- but that it is as right as the sphere's, which the patch construction must
    not degrade.

    MEASURED, choosing eps: the difference quotient is taken on a per-rollout term of size ~10 in
    float32, so it carries ~5e-3 of absolute noise at eps = 1e-4 and more below. Above it, the
    step starts to cross contact switches: at the worst cell of this scene the quotient runs
    1.91 (eps = 2e-3), 2.45 (5e-4), 3.4714 (1e-4) against an analytic 3.4705 -- converging on the
    analytic value as the step shrinks inside the validity radius, which is what an active-set
    switch looks like and a wrong chain rule does not.
    """
    wp.init()
    out = {}
    for tag, width in (("cylinder", 0.10), ("sphere", None)):
        stats, probe = _cyl_fd_stats(width, batch, steps, eps, n_cells, device)
        out[tag] = stats
        bins = f" yaw bins {stats['bins']}" if stats["bins"] else ""
        print(
            f"  {tag:8s} {stats['n']} cells{bins}  ||g||={stats['scale']:.3f}  "
            f"median={stats['median']:.2e}  p90={stats['p90']:.2e}  max={stats['max']:.2e}"
        )
        b, iy, ix, ga, fd, fd5 = probe
        print(
            f"  {tag:8s} worst cell b={b} ({iy},{ix}): analytic {ga:+.4f}  "
            f"fd@{eps:g} {fd:+.4f}  fd@{eps/5:g} {fd5:+.4f}"
        )
    cyl, sph = out["cylinder"], out["sphere"]
    ok = (
        cyl["median"] < 5e-3 * cyl["scale"]
        and cyl["p90"] < 2e-2 * cyl["scale"]
        and cyl["median"] < 4.0 * sph["median"]
    )
    print(
        f"cylinder d/d(raw h) vs FD  median={cyl['median']:.2e} "
        f"(sphere {sph['median']:.2e})  {'OK' if ok else 'REVIEW'}"
    )
    assert ok, "the cylinder patch gradient is worse against FD than the sphere's whole-map one"


if __name__ == "__main__":
    _selftest()
    _selftest_step_grad()
    _selftest_bptt()
    _selftest_dh()
    _selftest_batch()
    if wp.get_cuda_device_count() > 0:
        _selftest_cylinder_dh()
    else:
        print("cylinder d/d(raw h) vs FD  SKIPPED (no CUDA device)")
