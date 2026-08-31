"""What does Clark propagation of the v2 attitude cost COST, against the
deterministic forward pass (one rollout-equivalent: bilinear element max on
the mean map -> settle rows -> sum of squared attitudes)?

    python -m studies.bench.v2_att_forward_bench
"""
from __future__ import annotations

import sys
import time

import numpy as np
import warp as wp

from helhest.engine import RobotParams
from helhest.risk.attitude import WarpAttitudeEstimator
from helhest.risk.contact import k_nodes
from helhest.risk.sigma import rho1_table
from helhest.risk.settle import settle_map


@wp.kernel
def k_forward(
    belief: wp.array2d(dtype=wp.float32),
    off_dy: wp.array2d(dtype=wp.int32), off_dx: wp.array2d(dtype=wp.int32),
    off_cap: wp.array2d(dtype=wp.float32), k_real: wp.array(dtype=wp.int32),
    anchor_y: wp.array(dtype=wp.int32), anchor_x: wp.array(dtype=wp.int32),
    node_bin: wp.array(dtype=wp.int32), node_wgt: wp.array(dtype=wp.float32),
    row_w: wp.array2d(dtype=wp.float32),
    ny: int, nx: int, n_t: int,
    x_rows: wp.array2d(dtype=wp.float32),
):
    n = wp.tid()
    b = node_bin[n]
    ay = anchor_y[n]
    ax = anchor_x[n]
    kk = k_real[b]
    m = float(-3.0e38)
    for j in range(kk):
        iy = wp.clamp(ay + off_dy[b, j], 0, ny - 1)
        ix = wp.clamp(ax + off_dx[b, j], 0, nx - 1)
        v = belief[iy, ix] + off_cap[b, j]
        if v > m:
            m = v
    r = n / 4
    t = r % n_t
    r = r / n_t
    w = r % 3
    p = r / 3
    g = node_wgt[n] * m
    wp.atomic_add(x_rows, p, 2 * t, row_w[0, w] * g)
    wp.atomic_add(x_rows, p, 2 * t + 1, row_w[1, w] * g)


@wp.kernel
def k_forward_finish(x_rows: wp.array2d(dtype=wp.float32), n2: int,
                     L: wp.array(dtype=wp.float32)):
    p = wp.tid()
    acc = float(0.0)
    for r in range(n2):
        acc += x_rows[p, r] * x_rows[p, r]
    L[p] = acc


def main():
    wp.init()
    rng = np.random.default_rng(3)
    ny = nx = 90
    cell = 0.10
    rp = RobotParams()
    rho1 = np.asarray(rho1_table(cell, 2.0), np.float64)
    belief = (np.cumsum(0.05 * rng.standard_normal((ny, nx)), axis=0) * 0.08).astype(np.float32)
    sigma = (0.02 + 0.02 * rng.random((ny, nx))).astype(np.float32)
    b = wp.array(belief, dtype=wp.float32, device="cuda:0")
    sg = wp.array(sigma, dtype=wp.float32, device="cuda:0")
    R = settle_map(rp)[1:3, :].astype(np.float32)

    for P, T in ((256, 40), (512, 40)):
        est = WarpAttitudeEstimator(ny, nx, cell, 0.0, 0.0, rp, rho1, P, T,
                                    device="cuda:0")
        c = est._c
        ctrl = np.zeros((T + 1, P, 3), np.float32)
        for p in range(P):
            yaw = -1.0 + p * 0.008
            x, y = 4.5, 4.5
            for t in range(T + 1):
                ctrl[t, p] = (x, y, yaw)
                x += 0.1 * np.cos(yaw)
                y += 0.1 * np.sin(yaw)
                yaw += 0.002 * np.sin(p)
        cw = wp.array(ctrl, dtype=wp.float32, device="cuda:0")
        with wp.ScopedDevice("cuda:0"):
            x_rows = wp.zeros((P, 2 * T), dtype=wp.float32)
            Lout = wp.zeros(P, dtype=wp.float32)
            row_w = wp.array(R, dtype=wp.float32)
            ones = wp.array(np.ones(3, np.float32), dtype=wp.float32)

        def forward():
            x_rows.zero_()
            wp.launch(k_nodes, dim=c.n_nodes, device="cuda:0",
                      inputs=[cw, c.wheel_xy, ones, c.origin_x, c.origin_y,
                              c.cell, c.ny, c.nx, c.n_bins, c.n_t],
                      outputs=[c.anchor_y, c.anchor_x, c.node_bin, est.node_wgt])
            wp.launch(k_forward, dim=c.n_nodes, device="cuda:0",
                      inputs=[b, c.off_dy, c.off_dx, c.off_cap, c.k_real,
                              c.anchor_y, c.anchor_x, c.node_bin, est.node_wgt,
                              row_w, c.ny, c.nx, c.n_t],
                      outputs=[x_rows])
            wp.launch(k_forward_finish, dim=P, device="cuda:0",
                      inputs=[x_rows, 2 * T], outputs=[Lout])

        forward()
        est.moments(b, sg, cw)
        wp.synchronize()
        t0 = time.time()
        for _ in range(50):
            forward()
        wp.synchronize()
        t_fwd = (time.time() - t0) / 50
        t0 = time.time()
        for _ in range(5):
            est.moments(b, sg, cw)
        wp.synchronize()
        t_clark = (time.time() - t0) / 5
        print(f"P={P} T={T}: forward {t_fwd*1e3:6.2f} ms ({t_fwd/P*1e6:5.1f} us/plan) | "
              f"clark {t_clark*1e3:6.1f} ms ({t_clark/P*1e6:5.0f} us/plan) | "
              f"propagation = {t_clark/t_fwd:.1f}x one forward pass")
        del est


if __name__ == "__main__":
    main()
