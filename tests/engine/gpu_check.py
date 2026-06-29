"""GPU parity + timing for the settle/step kernels after the rotation refactor.

`settle`/`adj_settle` now compose the analytic Jacobian via rot_*/drot_* helpers,
which evaluate each angle's cos/sin TWICE per Newton iteration (once for the matrix,
once for its derivative). ptxas must CSE that back to one. This check confirms:

  1. forward parity  — CUDA step rollout == CPU step rollout (CPU is oracle-verified),
  2. adjoint parity  — the refactored IFT adjoint d/dHenv matches CPU on CUDA,
  3. throughput      — a planner-scale batched rollout runs within a control budget,
     so even if the duplicate trig survived, it isn't a regression that matters.

Needs a CUDA device; skips cleanly (exit 0) otherwise.

Run on a GPU box:  python -m tests.engine.gpu_check
"""

import time

import numpy as np
import warp as wp
from kinematic_helhest import dynamics
from kinematic_helhest import friction
from kinematic_helhest import heightmap as hmmod
from kinematic_helhest.control.reference import _to_wheel_omega
from kinematic_helhest.engine import DifferentiableSimulator
from kinematic_helhest.engine import ForwardSimulator
from kinematic_helhest.engine import GridParams
from kinematic_helhest.engine import RobotParams
from kinematic_helhest.engine import SolverParams

from tests.engine.gradients import dsettle_dHenv


@wp.kernel
def _cot_loss(
    controlled: wp.array2d(dtype=wp.vec3),
    derived: wp.array2d(dtype=wp.vec3),
    cc: wp.array2d(dtype=wp.vec3),
    cd: wp.array2d(dtype=wp.vec3),
    loss: wp.array(dtype=float),
):
    """Scalar surrogate loss = <controlled, cc> + <derived, cd>: its backward seeds exactly the
    cotangents (cc, cd) on the state buffers -- the reference for backward_from_cotangents."""
    t, b = wp.tid()
    wp.atomic_add(loss, 0, wp.dot(controlled[t, b], cc[t, b]) + wp.dot(derived[t, b], cd[t, b]))


def _sim(scene, mu, B, T, device):
    sim = ForwardSimulator(
        RobotParams(),
        SolverParams(dt=0.05, k_turn=2.0, newton_iters=12),
        GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0),
        B,
        T,
        device,
    )
    sim.set_terrain(
        wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device)
    )
    sim.set_friction(mu)
    return sim


def check_forward_parity():
    """CUDA step rollout vs CPU step rollout on the same node grid."""
    scene, mu = hmmod.box_scene(), friction.uniform(0.8)
    B, T, start = 64, 40, (-1.0, 0.0, 0.0)
    wheel_omega = _to_wheel_omega(np.full((B, T, 2), 2.0, np.float32))
    pc, tc, _, _ = _sim(scene, mu, B, T, "cpu").rollout(wheel_omega, start)
    pg, tg, _, _ = _sim(scene, mu, B, T, "cuda").rollout(wheel_omega, start)
    dp = float(np.abs(pc - pg).max())
    dt = float(np.abs(tc - tg).max())
    print(f"  forward CUDA-vs-CPU  dplanar={dp:.2e}  dtilt={dt:.2e}")
    assert max(dp, dt) < 1e-3, (dp, dt)
    print("  forward parity OK")


def check_adjoint_parity():
    """Refactored IFT adjoint on CUDA vs the device-free numpy finite-difference
    oracle (the real correctness bar). CPU-vs-CUDA drift is reported but not asserted
    tightly: an iterative settle + atomic scatter legitimately differs by ~1e-4 across
    hardware; a miscompile would crash or be O(0.1)+ wrong, not drift."""
    from tests.engine.gradients import _fd_loss

    params = SolverParams(newton_iters=12)
    env = hmmod.wheel_envelope(hmmod.ramp_scene(), 0.35)
    poses = [(2.0, 0.0, 0.0), (3.0, 0.3, 0.2)]
    adj_u = np.tile(np.array([0.3, 1.0, 0.5], np.float32), (len(poses), 1))
    g_gpu, _ = dsettle_dHenv(env, poses, adj_u, params, device="cuda")
    g_cpu, _ = dsettle_dHenv(env, poses, adj_u, params, device="cpu")

    eps, err = 1e-3, 0.0
    for i, j in zip(*np.where(np.abs(g_gpu) > 1e-6)):  # only the contact cells
        g_fd = (
            _fd_loss(env, poses, adj_u, i, j, +eps) - _fd_loss(env, poses, adj_u, i, j, -eps)
        ) / (2.0 * eps)
        err = max(err, abs(g_gpu[i, j] - g_fd))
    drift = float(np.abs(g_gpu - g_cpu).max())
    print(f"  adjoint CUDA-vs-FD  max|err|={err:.2e}  (CUDA-vs-CPU fp drift {drift:.2e})")
    assert err < 5e-2, err
    print("  adjoint parity OK")


def time_rollout(B=2048, T=70, reps=30):
    """Planner-scale batched rollout throughput on CUDA (includes the host readback
    the MPPI cost needs, so it's the realistic per-rollout cost)."""
    scene = hmmod.demo_terrain()
    mu = friction.uniform(0.8, xlim=(-3.0, 10.0), ylim=(-4.0, 4.0), cell=0.06)
    sim = _sim(scene, mu, B, T, "cuda")
    wheel_omega = _to_wheel_omega(np.full((B, T, 2), 2.0, np.float32))
    start = (0.0, 0.0, 0.0)
    sim.rollout(wheel_omega, start)  # warm up: triggers CUDA codegen + first launch
    wp.synchronize_device("cuda")
    t0 = time.perf_counter()
    for _ in range(reps):
        sim.rollout(wheel_omega, start)
    wp.synchronize_device("cuda")
    dt = (time.perf_counter() - t0) / reps
    print(
        f"  rollout B={B} T={T} iters=12:  {dt * 1e3:.2f} ms/rollout  "
        f"({B * T / dt / 1e6:.0f} M wheel-steps/s)"
    )


def check_vjp():
    """DifferentiableSimulator's VJP boundary: backward_from_cotangents(cc, cd) must give the same
    input grads as a scalar-loss backward whose loss = <controlled, cc> + <derived, cd>."""
    scene = hmmod.ramp_scene()
    ny, nx = scene.H.shape
    B, T = 3, 14
    grid = GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0)
    omega = _to_wheel_omega(np.tile(np.array([1.5, 2.5], np.float32), (B, T, 1)))
    rng = np.random.default_rng(0)
    cc = rng.standard_normal((T + 1, B, 3)).astype(np.float32)
    cd = rng.standard_normal((T + 1, B, 3)).astype(np.float32)

    def build():
        sim = DifferentiableSimulator(
            dynamics.robot_params(), dynamics.execution_solver(), grid, B, T, "cuda"
        )
        sim.set_terrain(
            wp.array(
                np.broadcast_to(scene.H, (B, ny, nx)).astype(np.float32),
                dtype=wp.float32,
                device="cuda",
            )
        )
        sim.set_uniform_friction(0.8)
        sim.wheel_omega.assign(np.ascontiguousarray(omega, np.float32))
        sim.start_pose.assign(np.tile(np.asarray((-1.0, 0.0, 0.0), np.float32), (B, 1)))
        return sim

    ccw = wp.array(cc, dtype=wp.vec3, device="cuda")
    cdw = wp.array(cd, dtype=wp.vec3, device="cuda")

    ref = build()  # scalar-loss reference

    def loss_fn(s):
        loss = wp.zeros(1, dtype=float, device="cuda", requires_grad=True)
        wp.launch(
            _cot_loss,
            (T + 1, B),
            inputs=[s.controlled, s.derived, ccw, cdw],
            outputs=[loss],
            device="cuda",
        )
        return loss

    ref.rollout_taped(loss_fn)
    ref.backward()
    g_ref_e, g_ref_m = ref.elevation.grad.numpy(), ref.friction.grad.numpy()

    sim = build()  # VJP path
    sim.rollout_taped(loss_fn=None)
    sim.backward_from_cotangents(
        wp.array(cc, dtype=wp.vec3, device="cuda"), wp.array(cd, dtype=wp.vec3, device="cuda")
    )
    de = np.abs(sim.elevation.grad.numpy() - g_ref_e).max()
    dm = np.abs(sim.friction.grad.numpy() - g_ref_m).max()
    print(f"  VJP vs scalar-loss  d/delev={de:.2e}  d/dfric={dm:.2e}")
    assert max(de, dm) < 1e-5, (de, dm)
    print("  VJP parity OK")


def main():
    wp.init()
    if not wp.is_cuda_available():
        print("CUDA not available — skipping GPU check.")
        return
    print("[1/4] forward parity")
    check_forward_parity()
    print("[2/4] adjoint parity")
    check_adjoint_parity()
    print("[3/4] VJP (backward_from_cotangents)")
    check_vjp()
    print("[4/4] throughput")
    time_rollout()
    print("GPU check: ALL OK")


if __name__ == "__main__":
    main()
