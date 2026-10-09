"""The batched GPU path-QP solver (control/qp_gpu.py) against OSQP, its oracle."""

from __future__ import annotations

import math

import numpy as np
import osqp
import scipy.sparse as sp
import warp as wp

from helhest.control.qp_gpu import AdmmParams
from helhest.control.qp_gpu import bands_to_csc
from helhest.control.qp_gpu import BatchQp
from helhest.control.qp_gpu import PRIMAL_INFEASIBLE
from helhest.control.qp_gpu import rows_to_csc
from helhest.control.qp_gpu import SOLVED
from helhest.control.qp_path import QpPathParams
from helhest.control.qp_path import spline_terms
from helhest.control.qp_path import SplineQp

DS = 0.15


def _problems(seed: int = 0) -> list[tuple[SplineQp, str]]:
    """Corridors like the robot's: free, an obstacle pushing the path aside, a bend in the route,
    a narrow lane, and one that jumps sides (no path can)."""
    rng = np.random.default_rng(seed)
    out = []
    for kind in ["box"] * 6 + ["bend"] * 6 + ["narrow"] * 3 + ["free"] * 3 + ["jump"] * 2:
        n = int(rng.integers(36, 55))
        s = np.arange(n) * DS
        kr = 0.4 * np.sin(s / 2.0) if kind == "bend" else np.zeros(n)
        lo, hi = np.full(n, -3.0), np.full(n, 3.0)
        if kind in ("box", "bend"):
            a, b = rng.integers(10, n // 2), rng.integers(4, 12)
            lo[a : a + b] = rng.uniform(0.6, 1.6)
        if kind == "narrow":
            lo[:], hi[:] = rng.uniform(-0.6, -0.2), rng.uniform(0.2, 0.6)
        if kind == "jump":
            lo[30:33], hi[30:33] = 2.0, 2.2
            lo[33:36], hi[33:36] = -2.2, -2.0
        e0, psi = rng.uniform(-0.3, 0.3), rng.uniform(-0.4, 0.4)
        lo[:3], hi[:3] = np.minimum(lo[:3], e0 - 0.05), np.maximum(hi[:3], e0 + 0.05)
        out.append((spline_terms(kr, lo, hi, e0, math.tan(psi), DS, QpPathParams()), kind))
    return out


def _gpu(problems: list[tuple[SplineQp, str]]) -> tuple[np.ndarray, np.ndarray]:
    b_ = len(problems)
    n_ = max(len(t.q) for t, _ in problems)
    m_ = max(len(t.lo) for t, _ in problems)
    pb, q = np.zeros((b_, n_, 4)), np.zeros((b_, n_))
    ac, acol = np.zeros((b_, m_, 4)), np.zeros((b_, m_), np.int32)
    lo, hi, sizes = np.zeros((b_, m_)), np.zeros((b_, m_)), np.zeros((b_, 2), np.int32)
    for b, (t, _) in enumerate(problems):
        n, m = len(t.q), len(t.lo)
        pb[b, :n], q[b, :n], ac[b, :m], acol[b, :m] = t.p_band, t.q, t.a_coef, t.a_col
        lo[b, :m], hi[b, :m], sizes[b] = t.lo, t.hi, (n, m)
    solver = BatchQp(b_, n_, m_, AdmmParams())
    x, status, _ = solver.solve(*(wp.array(a) for a in (pb, q, ac, acol, lo, hi, sizes)))
    return x.numpy(), status.numpy()


def _reference(t: SplineQp) -> osqp.OSQP:
    n = len(t.q)
    s = osqp.OSQP()
    s.setup(
        sp.triu(bands_to_csc(t.p_band, n), format="csc"),
        t.q,
        rows_to_csc(t.a_coef, t.a_col, n),
        t.lo,
        t.hi,
        verbose=False,
        eps_abs=1e-10,
        eps_rel=1e-10,
        max_iter=200000,
        polishing=True,
    )
    return s.solve()


def test_the_gpu_solver_finds_the_paths_osqp_does():
    problems = _problems()
    x, status = _gpu(problems)
    close = 0
    feasible = 0
    for b, (t, kind) in enumerate(problems):
        ref = _reference(t)
        if ref.info.status != "solved":
            continue
        feasible += 1
        assert status[b] == SOLVED, kind
        err = np.abs(t.offsets(x[b, : len(t.q)]) - t.offsets(ref.x)).max()
        assert err < 0.10, (kind, err)  # never far off ...
        close += err < 5e-3  # ... and nearly always the same path to 5 mm
    assert close >= feasible - 1


def test_a_corridor_that_jumps_sides_is_reported_infeasible():
    problems = [(t, k) for t, k in _problems() if k == "jump"]
    _, status = _gpu(problems)
    for (t, _), st in zip(problems, status):
        assert _reference(t).info.status == "primal infeasible"
        assert st == PRIMAL_INFEASIBLE
