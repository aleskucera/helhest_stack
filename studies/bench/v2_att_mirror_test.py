"""Independent NumPy mirror of the GPU attitude estimator -- the correctness
contract for `helhest.risk.attitude.WarpAttitudeEstimator`.

Two checks, both of which must pass for any change to the estimator:

  1. FROZEN REFERENCE. E[L] of the first four plans of a fixed small scene,
     recorded when the kernels were first verified (commit c916db7) and never
     recomputed since. It catches a change of semantics that the mirror would
     follow (the mirror reads the fold's device intermediates, so a change
     INSIDE the fold moves both together).

  2. NUMPY MIRROR. The moments rebuilt on the host from the fold's device
     intermediates -- `node_mean`, `node_dvar`, `node_wgt`, the folded
     candidate cells and their lambda-weighted sigmas -- with the block
     structure written out in full and dense linear algebra:

         S = W R W' + D,   R[e,f] = rho1[|dy_ef|] * rho1[|dx_ef|]
         W[2t+i, e]       = row_w[i, wheel(e)] * node_wgt[e] * cand_wsig[e]
         D[2t+i, 2t+k]    = sum_nodes row_w[i,w] row_w[k,w] node_wgt^2 node_dvar
         E   = tr(S) + m'm
         Var = 2 tr(S^2) + 4 m'S m

     This shares nothing with the covariance kernel: no banding, no candidate
     packing, no wheel-pair culling, no step-pair decomposition. It is the
     definition the kernel is an optimisation of.

    PYTHONPATH=src python -m studies.bench.v2_att_mirror_test
"""
from __future__ import annotations

import sys

import numpy as np
import warp as wp

from helhest.engine import RobotParams
from helhest.risk.attitude import WarpAttitudeEstimator
from helhest.risk.settle import settle_map
from helhest.risk.sigma import rho1_table

# E[L] of plans 0..3 of `scene()`, recorded 2026-08-27 from the kernels verified
# against this mirror at 1e-6 (study repo commit c916db7). A change to the
# estimator that moves these is a change of semantics, not an optimisation.
E_FROZEN = np.array([0.12268846, 0.10915443, 0.12414471, 0.11225437])

NY = NX = 90
CELL = 0.10
P, T = 8, 30


def scene():
    """The fixed small scene the frozen reference was taken on."""
    rng = np.random.default_rng(3)
    belief = (np.cumsum(0.05 * rng.standard_normal((NY, NX)), axis=0) * 0.08).astype(np.float32)
    sigma = (0.02 + 0.02 * rng.random((NY, NX))).astype(np.float32)
    ctrl = np.zeros((T + 1, P, 3), np.float32)
    for p in range(P):
        yaw = -0.5 + p * 0.13
        x, y = 4.5, 4.5
        for t in range(T + 1):
            ctrl[t, p] = (x, y, yaw)
            x += 0.1 * np.cos(yaw)
            y += 0.1 * np.sin(yaw)
            yaw += 0.004 * (p - 3)
    return belief, sigma, ctrl


def mirror(est, rho1, rp):
    """E and Var per plan, from the estimator's device intermediates, in NumPy."""
    n_t = est.n_t
    rho1 = np.asarray(rho1, np.float64)
    L = len(rho1)
    row_w = est.row_w.numpy().astype(np.float64)           # [2, 3]
    wgt = est.node_wgt.numpy().astype(np.float64)          # [n_nodes]
    mean = est.node_mean.numpy().astype(np.float64)
    dvar = est.node_dvar.numpy().astype(np.float64)
    ciy = est.cand_iy.numpy().T          # the kernels hold these slot-major
    cix = est.cand_ix.numpy().T
    cws = est.cand_wsig.numpy().T.astype(np.float64)
    n2 = 2 * n_t
    E = np.zeros(est.n_plans)
    V = np.zeros(est.n_plans)

    def r1(lag):
        a = np.abs(lag)
        return np.where(a < L, rho1[np.minimum(a, L - 1)], 0.0)

    for p in range(est.n_plans):
        # every (wheel, step, corner, candidate) with a nonzero weight
        ys, xs, ws_, rows = [], [], [], []
        S_d = np.zeros((n2, n2))
        m = np.zeros(n2)
        for w in range(3):
            for t in range(n_t):
                for c in range(4):
                    n = ((p * 3 + w) * n_t + t) * 4 + c
                    g = wgt[n]
                    m[2 * t] += row_w[0, w] * g * mean[n]
                    m[2 * t + 1] += row_w[1, w] * g * mean[n]
                    for i in range(2):
                        for k in range(2):
                            S_d[2 * t + i, 2 * t + k] += (
                                row_w[i, w] * row_w[k, w] * g * g * dvar[n])
                    for j in range(cws.shape[1]):
                        if cws[n, j] == 0.0 or g == 0.0:
                            continue
                        ys.append(ciy[n, j])
                        xs.append(cix[n, j])
                        ws_.append(g * cws[n, j])
                        rows.append((t, w))
        ys = np.asarray(ys)
        xs = np.asarray(xs)
        ws_ = np.asarray(ws_)
        ne = len(ys)
        R = r1(ys[:, None] - ys[None, :]) * r1(xs[:, None] - xs[None, :])
        W = np.zeros((n2, ne))
        for e, (t, w) in enumerate(rows):
            W[2 * t, e] = row_w[0, w] * ws_[e]
            W[2 * t + 1, e] = row_w[1, w] * ws_[e]
        S = W @ R @ W.T + S_d
        E[p] = np.trace(S) + m @ m
        V[p] = 2.0 * np.sum(S * S) + 4.0 * (m @ S @ m)
    return E, V


def main():
    wp.init()
    dev = "cuda:0"
    rp = RobotParams()
    belief, sigma, ctrl = scene()
    ok = True
    # the third case is the perception node's 0.08 m cell: a longer rho1, K up to 13
    # rather than 7, a wider merged candidate list, and -- because the same plans run
    # off a 7.2 m map -- the border CLAMP path that the two 0.10 m cases never take.
    for tag, cell, rho1 in (("corr_len=0.15", CELL, rho1_table(0.15, CELL)),
                            ("degenerate", CELL, rho1_table(CELL, 2.0)),
                            ("cell=0.08, clamped", 0.08, rho1_table(0.15, 0.08))):
        rho1 = np.asarray(rho1, np.float64)
        est = WarpAttitudeEstimator(NY, NX, cell, 0.0, 0.0, rp, rho1, P, T, device=dev)
        b = wp.array(belief, dtype=wp.float32, device=dev)
        sg = wp.array(sigma, dtype=wp.float32, device=dev)
        cw = wp.array(ctrl, dtype=wp.float32, device=dev)
        Ed, Vd = est.moments(b, sg, cw)
        wp.synchronize()
        Eg, Vg = Ed.numpy().astype(np.float64), Vd.numpy().astype(np.float64)
        Em, Vm = mirror(est, rho1, rp)
        re = np.max(np.abs(Eg - Em) / np.maximum(np.abs(Em), 1e-12))
        rv = np.max(np.abs(Vg - Vm) / np.maximum(np.abs(Vm), 1e-12))
        print(f"[{tag}] rho1 len {len(rho1)}  band_t {est.band_t}  "
              f"merged list <= {est.u_max}  rho2 half {est.half}")
        print(f"    kernel E[:4] {np.array2string(Eg[:4], precision=8)}")
        print(f"    mirror E[:4] {np.array2string(Em[:4], precision=8)}")
        print(f"    max rel err: E {re:.2e}   Var {rv:.2e}")
        if not (re < 1e-5 and rv < 1e-5):
            ok = False
            print("    FAIL: mirror disagreement")
        if tag == "degenerate":
            rf = np.max(np.abs(Eg[:4] - E_FROZEN) / np.abs(E_FROZEN))
            print(f"    frozen reference: max rel err {rf:.2e}")
            if not rf < 1e-5:
                ok = False
                print(f"    FAIL: frozen reference is {E_FROZEN}")
        del est
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
