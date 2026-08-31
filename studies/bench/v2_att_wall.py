"""v2 wall: the GPU attitude estimator vs best-shape GPU sampling, same
device, same scene, deployed covariance constants (CORR_LEN=0.15, CELL=0.10).

Methodology of clark_warp_wall.py: common random numbers (D terrains drawn
once, every plan evaluated on each), noise generation timed and charged to
the sampler, equal-DRAW and equal-ACCURACY costs both reported (sampling
error falls as 1/sqrt(D); the estimator has a fixed floor measured here
against a large-D reference).

    PYTHONPATH=src python -m studies.bench.v2_att_wall
"""
from __future__ import annotations

import time

import numpy as np
import warp as wp

from helhest.engine import RobotParams
from helhest.risk.attitude import WarpAttitudeEstimator
from helhest.risk.contact import k_nodes
from helhest.risk.sigma import rho1_table
from helhest.risk.settle import settle_map

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from studies.adjoint.sigma import NoiseDraws  # noqa: E402
from studies.bench.v2_att_forward_bench import k_forward_finish  # noqa: E402

CELL, CORR_LEN = 0.10, 0.15
P, T, D_TIME, D_REF = 256, 40, 32, 2048


@wp.kernel
def k_forward_draw(
    fields: wp.array3d(dtype=wp.float32),    # [D, ny, nx]
    d: int,
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
        v = fields[d, iy, ix] + off_cap[b, j]
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


def main():
    wp.init()
    rng = np.random.default_rng(3)
    ny = nx = 90
    rp = RobotParams()
    rho1 = rho1_table(CORR_LEN, CELL)
    belief = (np.cumsum(0.05 * rng.standard_normal((ny, nx)), axis=0) * 0.08).astype(np.float32)
    sigma = (0.02 + 0.02 * rng.random((ny, nx))).astype(np.float32)
    R = settle_map(rp)[1:3, :].astype(np.float32)
    dev = "cuda:0"
    b = wp.array(belief, dtype=wp.float32, device=dev)
    sg = wp.array(sigma, dtype=wp.float32, device=dev)
    ctrl = np.zeros((T + 1, P, 3), np.float32)
    for p in range(P):
        yaw = -1.0 + p * 0.008
        x, y = 4.5, 4.5
        for t in range(T + 1):
            ctrl[t, p] = (x, y, yaw)
            x += 0.1 * np.cos(yaw)
            y += 0.1 * np.sin(yaw)
            yaw += 0.002 * np.sin(p)
    cw = wp.array(ctrl, dtype=wp.float32, device=dev)

    est = WarpAttitudeEstimator(ny, nx, CELL, 0.0, 0.0, rp, rho1, P, T, device=dev)
    c = est._c
    with wp.ScopedDevice(dev):
        row_w = wp.array(R, dtype=wp.float32)
        ones = wp.array(np.ones(3, np.float32), dtype=wp.float32)
        x_rows = wp.zeros((P, 2 * T), dtype=wp.float32)
        Ld = wp.zeros(P, dtype=wp.float32)

    # nodes once (shared by all draws; a real sampler would also do this once)
    wp.launch(k_nodes, dim=c.n_nodes, device=dev,
              inputs=[cw, c.wheel_xy, ones, c.origin_x, c.origin_y,
                      CELL, ny, nx, c.n_bins, c.n_t],
              outputs=[c.anchor_y, c.anchor_x, c.node_bin, est.node_wgt])

    def sample(D, seed, want_costs=False):
        nd = NoiseDraws((D, ny, nx), CELL, CORR_LEN, dev)
        with wp.ScopedDevice(dev):
            base = wp.array(np.zeros((D, ny, nx), np.float32))
            base3 = wp.array(np.ascontiguousarray(np.tile(belief, (D, 1, 1)), np.float32))
            out = wp.zeros((D, ny, nx), dtype=wp.float32)
        nd.perturb(base3, sg, 1.0, out, seed)
        costs = np.empty((D, P), np.float64) if want_costs else None
        acc = np.zeros(P)
        acc2 = np.zeros(P)
        for d in range(D):
            x_rows.zero_()
            wp.launch(k_forward_draw, dim=c.n_nodes, device=dev,
                      inputs=[out, d, c.off_dy, c.off_dx, c.off_cap, c.k_real,
                              c.anchor_y, c.anchor_x, c.node_bin, est.node_wgt,
                              row_w, ny, nx, c.n_t],
                      outputs=[x_rows])
            wp.launch(k_forward_finish, dim=P, device=dev,
                      inputs=[x_rows, 2 * T], outputs=[Ld])
            l = Ld.numpy()
            acc += l
            acc2 += l * l
            if want_costs:
                costs[d] = l
        mean = acc / D
        sd = np.sqrt(np.maximum(acc2 / D - mean * mean, 0) * D / (D - 1))
        return mean, sd, costs

    # ---- accuracy floor of the estimator, vs the large-D reference ----
    mean_ref, sd_ref, _ = sample(D_REF, 999)
    E, V = est.moments(b, sg, cw)
    wp.synchronize()
    E = E.numpy().copy()
    sdc = np.sqrt(np.maximum(V.numpy().copy(), 0))
    relE = np.median(np.abs(E / mean_ref - 1))
    relS = np.median(np.abs(sdc / sd_ref - 1))
    cv = np.median(sd_ref / mean_ref)
    d_eq_E = (cv / relE) ** 2
    d_eq_S = 1.0 / (2.0 * relS ** 2)
    d_eq = max(d_eq_E, d_eq_S)

    # ---- timings ----
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        est.moments(b, sg, cw)
        wp.synchronize()
        ts.append(time.perf_counter() - t0)
    t_clark = np.median(ts)

    def time_sampler(D):
        nd = NoiseDraws((D, ny, nx), CELL, CORR_LEN, dev)
        with wp.ScopedDevice(dev):
            base3 = wp.array(np.ascontiguousarray(np.tile(belief, (D, 1, 1)), np.float32))
            out = wp.zeros((D, ny, nx), dtype=wp.float32)
        nd.perturb(base3, sg, 1.0, out, 5)
        wp.synchronize()
        t0 = time.perf_counter()
        nd.perturb(base3, sg, 1.0, out, 6)
        for d in range(D):
            wp.launch(k_forward_draw, dim=c.n_nodes, device=dev,
                      inputs=[out, d, c.off_dy, c.off_dx, c.off_cap, c.k_real,
                              c.anchor_y, c.anchor_x, c.node_bin, est.node_wgt,
                              row_w, ny, nx, c.n_t],
                      outputs=[x_rows])
            wp.launch(k_forward_finish, dim=P, device=dev,
                      inputs=[x_rows, 2 * T], outputs=[Ld])
        wp.synchronize()
        return time.perf_counter() - t0

    t_s32 = time_sampler(D_TIME)
    d_eq_i = int(np.ceil(d_eq))
    t_seq = time_sampler(min(d_eq_i, 4096))

    print(f"estimator floor:   median relE {relE:.2e}   rel-sd {relS:.2e}")
    print(f"equal-accuracy D:  from E {d_eq_E:.0f}, from sd {d_eq_S:.0f} -> D_eq = {d_eq_i}")
    print(f"clark:             {t_clark*1e3:7.1f} ms  ({t_clark/P*1e6:6.0f} us/plan)")
    print(f"sampler D={D_TIME}:     {t_s32*1e3:7.1f} ms  ({t_s32/P*1e6:6.0f} us/plan)   clark/sampler = {t_clark/t_s32:.2f}")
    print(f"sampler D={min(d_eq_i,4096)}: {t_seq*1e3:9.1f} ms  ({t_seq/P*1e6:6.0f} us/plan)   clark/sampler = {t_clark/t_seq:.2f}")


if __name__ == "__main__":
    main()
