"""Wheel-envelope dilation (engine) vs the numpy oracle, forward + backward.

Run:  python -m tests.engine.envelope
"""

import numpy as np
import warp as wp

from helhest import heightmap as hmmod
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
def _weighted_sum(
    Henv: wp.array2d(dtype=wp.float32),
    W: wp.array2d(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
):
    iy, ix = wp.tid()
    wp.atomic_add(loss, 0, W[iy, ix] * Henv[iy, ix])


def selftest_forward():
    """Device dilation vs numpy heightmap.wheel_envelope on the real scenes."""
    wp.init()
    worst = 0.0
    for name, scene in [
        ("flat", hmmod.flat()),
        ("box", hmmod.box_scene()),
        ("ramp", hmmod.ramp_scene()),
    ]:
        R = 0.35
        ref = hmmod.wheel_envelope(scene, R).H
        H = wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device="cpu")
        Henv = wheel_envelope(H, scene.cell, R, "cpu").numpy()
        d = np.abs(Henv - ref).max()
        worst = max(worst, d)
        print(f"  {name:4s} grid={scene.H.shape}  max|dHenv|={d:.2e}")
    print(
        f"envelope forward device-vs-numpy worst={worst:.2e}  {'OK' if worst < 1e-4 else 'REVIEW'}"
    )


def selftest_backward():
    """Autodiff d(loss)/d(raw h) vs the numpy analytic subgradient (tie-immune),
    plus a coarse finite-difference sanity check (tie noise expected)."""
    wp.init()
    rng = np.random.default_rng(1)
    ny, nx, cell, R = 12, 12, 0.1, 0.35
    H0 = rng.uniform(0.0, 0.5, (ny, nx)).astype(np.float32)
    W = rng.uniform(-1.0, 1.0, (ny, nx)).astype(np.float32)
    Wd = wp.array(W, dtype=wp.float32, device="cpu")

    H = wp.array(H0, dtype=wp.float32, device="cpu", requires_grad=True)
    loss = wp.zeros(1, dtype=wp.float32, device="cpu", requires_grad=True)
    tape = wp.Tape()
    with tape:
        Henv = wheel_envelope(H, cell, R, "cpu")
        wp.launch(_weighted_sum, (ny, nx), inputs=[Henv, Wd], outputs=[loss], device="cpu")
    tape.backward(loss=loss)
    g_ad = H.grad.numpy()

    g_an = _numpy_subgrad(H0, W, cell, R)
    err = np.abs(g_ad - g_an).max()

    eps = 1e-3  # FD: expect tie-switch noise at a few cells
    g_fd = np.zeros_like(H0)
    for i in range(ny):
        for j in range(nx):
            g_fd[i, j] = (
                _loss_at(H0, W, cell, R, i, j, +eps) - _loss_at(H0, W, cell, R, i, j, -eps)
            ) / (2 * eps)
    fd_med = np.median(np.abs(g_ad - g_fd))

    print(
        f"  grid={ny}x{nx}  max|g_ad-g_analytic|={err:.2e}  "
        f"median|g_ad-g_fd|={fd_med:.2e}  ||g||={np.abs(g_an).max():.2f}"
    )
    print(f"envelope backward autodiff-vs-analytic  {'OK' if err < 1e-3 else 'REVIEW'}")


def _numpy_subgrad(H0, W, cell, R):
    """Analytic subgradient: route each output's weight to its arg-max contact cell
    (same first-wins convention as _argmax_kernel)."""
    ny, nx = H0.shape
    rad = int(np.ceil(R / cell))
    g = np.zeros_like(H0)
    for i in range(ny):
        for j in range(nx):
            best, sy, sx = -1e9, i, j
            for di in range(-rad, rad + 1):
                for dj in range(-rad, rad + 1):
                    d = np.hypot(di, dj) * cell
                    if d <= R:
                        cap = np.sqrt(R * R - d * d) - R
                        yy = min(max(i + di, 0), ny - 1)
                        xx = min(max(j + dj, 0), nx - 1)
                        v = H0[yy, xx] + cap
                        if v > best:
                            best, sy, sx = v, yy, xx
            g[sy, sx] += W[i, j]
    return g


def _loss_at(H0, W, cell, R, i, j, delta):
    Hp = H0.copy()
    Hp[i, j] += delta
    H = wp.array(Hp, dtype=wp.float32, device="cpu")
    Henv = wheel_envelope(H, cell, R, "cpu").numpy().astype(np.float64)
    return float((W.astype(np.float64) * Henv).sum())


if __name__ == "__main__":
    selftest_forward()
    selftest_backward()
