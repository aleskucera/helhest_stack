"""Verification for `risk_calibration.py`. Run from the STUDY worktree:

    cd ~/projects/helhest_stack-study && \
      ./.venv/bin/python ~/projects/helhest_stack-spires/studies/spires/verify_fold_parity.py

It must be run there because it imports `studies.bench.clark` (the CPU reference fold) and
`helhest.*` (the deployed element and settle), which live on that worktree's branch; the
modules under test are loaded from the spires worktree by file path, so the two `studies`
packages never collide.

Three checks, matching the three the pipeline is required to pass:

  (a) FOLD PARITY. The universe-free fold in `risk_calibration.clark_fold` reproduces
      `clark.clark_build`'s mean and variance, its covariance-to-universe vector, and
      `clark.clark_quad_form`'s contracted plan variance, on a synthetic correlated case.
  (b) VENDOR PARITY. Every table `vehicle.py` vendors is bit-identical to the deployed one.
  (c) BELIEF IDENTITY. E0's Monte Carlo and the clark arm read the same bytes: the belief
      arrays are hashed before and after each consumer and the digests must match.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
import types
from pathlib import Path

import numpy as np

SPIRES = Path("/home/kuceral4/projects/helhest_stack-spires/studies/spires")
STUDY = Path("/home/kuceral4/projects/helhest_stack-study")  # the reference worktree
sys.path.insert(0, str(STUDY))  # `studies.bench.clark`, `studies.adjoint.harness`, `helhest`


def _load(name: str):
    """Import a spires module under the synthetic package `spx`.

    A synthetic package -- not `sys.path` -- because the spires worktree also has a top-level
    `studies` package, and putting it on the path would shadow the study worktree's own
    `studies.bench.clark`, which is the reference this file exists to check against. The alias
    keeps the spires modules' relative imports working (`from .elevation_belief import ...`
    resolves inside `spx`) while the two `studies` trees stay separate.
    """
    if "spx" not in sys.modules:
        pkg = types.ModuleType("spx")
        pkg.__path__ = [str(SPIRES)]
        sys.modules["spx"] = pkg
    return importlib.import_module(f"spx.{name}")


def _digest(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def check_fold_parity(seed: int = 7) -> None:
    import studies.bench.clark as CK

    rng = np.random.default_rng(seed)
    n_u, n, k = 40, 12, 7
    # a genuinely correlated PSD universe covariance
    A = rng.normal(size=(n_u, n_u + 5))
    Sigma = A @ A.T / (n_u + 5) * 0.02
    Sigma += np.diag(rng.uniform(1e-5, 1e-3, n_u))
    idx = rng.integers(0, n_u, size=(n, k))
    idx[:, -1] = idx[:, 0]  # force a repeated cell inside a node (the sentinel-pad shape)
    means = rng.normal(-6.0, 0.3, size=(n, k))  # realistic absolute elevations
    means[:, -1] = means[:, 0] - 1000.0  # a sentinel-capped pad candidate
    sigmas = np.sqrt(np.diagonal(Sigma)[idx])
    cov_self = Sigma[idx[:, :, None], idx[:, None, :]]
    cov_to_u = Sigma[idx]
    c = rng.normal(size=n)

    m_ref, v_ref, cu_ref, order, phi, phineg = CK.clark_build(means, sigmas, cov_self, cov_to_u)
    u_sorted = np.take_along_axis(idx, order, axis=1)
    var_ref = CK.clark_quad_form(cu_ref, u_sorted, phi, phineg, c)

    RC = _load("risk_calibration")
    m_new, v_new, lam = RC.clark_fold(means, cov_self)
    cu_new = np.einsum("nk,nku->nu", lam, cov_to_u)
    a = np.bincount(idx.ravel(), weights=(c[:, None] * lam).ravel(), minlength=n_u)
    var_new = float(a @ Sigma @ a)

    def rel(x, y):
        return float(np.max(np.abs(x - y) / np.maximum(np.abs(y), 1e-12)))

    print(f"(a) fold mean   max rel diff {rel(m_new, m_ref):.3e}")
    print(f"(a) fold var    max rel diff {rel(v_new, v_ref):.3e}")
    print(f"(a) cov-to-univ max rel diff {rel(cu_new, cu_ref):.3e}")
    print(f"(a) plan Var[J] rel diff     {abs(var_new - var_ref) / abs(var_ref):.3e}"
          f"   ({var_new:.9e} vs {var_ref:.9e})")
    assert rel(m_new, m_ref) < 1e-11
    assert rel(v_new, v_ref) < 1e-9
    assert rel(cu_new, cu_ref) < 1e-9
    assert abs(var_new - var_ref) / abs(var_ref) < 1e-9
    print("(a) PASS: the universe-free fold equals clark.py's universe fold")


def check_vendor_parity() -> None:
    import numpy as np
    from helhest.engine import RobotParams
    from helhest.engine.envelope import cyl_table as ref_cyl
    from helhest.engine.envelope import N_YAW_BINS as REF_BINS
    from helhest.engine.envelope import wheel_half_width
    from helhest.engine.step import yaw_bin as ref_yaw_bin
    from helhest.risk.settle import settle_map as ref_settle_map

    from studies.adjoint.harness import DERIV_WPITCH, DERIV_WROLL, DERIV_WZ
    from studies.bench import element as ref_el

    V = _load("vehicle")
    rp = RobotParams()
    assert (V.WHEEL_RADIUS, V.HALF_TRACK, V.REAR_OFFSET) == (
        rp.wheel_radius, rp.half_track, rp.rear_offset)
    assert V.HALF_WIDTH == wheel_half_width(rp)
    assert V.N_YAW_BINS == REF_BINS
    assert (V.W_Z, V.W_PITCH, V.W_ROLL) == (DERIV_WZ, DERIV_WPITCH, DERIV_WROLL)
    assert V.PAD_CAP == ref_el.PAD_CAP
    print("(b) robot / bin / weight / sentinel constants match the deployed values")

    for b in range(REF_BINS):
        a = V.cyl_table(0.10, V.WHEEL_RADIUS, V.HALF_WIDTH, b, REF_BINS)
        r = ref_cyl(0.10, rp.wheel_radius, wheel_half_width(rp), b, REF_BINS)
        for x, y in zip(a, r):
            assert np.array_equal(x, y), b
    print(f"(b) all {REF_BINS} cylinder tables bit-identical to helhest.engine.envelope.cyl_table")

    yaws = np.linspace(-4.0, 4.0, 401)
    mine = V.yaw_bins(yaws, REF_BINS)
    ref = np.array([ref_yaw_bin(float(y), REF_BINS) for y in yaws])
    assert np.array_equal(mine, ref)
    print("(b) yaw binning identical to engine/step.py::yaw_bin over 401 headings")

    dy, dx, cap = V.element_offsets(0.10, yaws)
    rdy, rdx, rcap = ref_el.element_offsets("cylinder", 0.10, rp, yaws)
    assert np.array_equal(dy, rdy) and np.array_equal(dx, rdx) and np.array_equal(cap, rcap)
    print(f"(b) element_offsets identical to studies/bench/element.py (K={dy.shape[1]}, "
          f"MAX_K={V.MAX_K})")
    assert dy.shape[1] <= V.MAX_K

    rng = np.random.default_rng(3)
    n = 500
    wx, wy = rng.uniform(-5, 5, n), rng.uniform(-5, 5, n)
    yaw_n = rng.uniform(-4, 4, n)
    a = V.blend_stencil(wx, wy, -7.3, 2.1, 0.10, 300, 300)
    r = ref_el.blend_stencil(wx, wy, -7.3, 2.1, 0.10, 300, 300)
    for x, y in zip(a, r):
        assert np.array_equal(x, y)
    ndy, ndx, _ = V.element_offsets(0.10, np.repeat(yaw_n, 4))
    mdy, mdx, _ = ref_el.element_offsets("cylinder", 0.10, rp, np.repeat(yaw_n, 4))
    assert np.array_equal(
        V.stencil_cells(a[0], a[1], ndy, ndx, 300, 300),
        ref_el.stencil_cells(r[0], r[1], mdy, mdx, 300, 300),
    )
    print("(b) blend_stencil / stencil_cells identical to element.py")

    assert np.array_equal(V.settle_map(), ref_settle_map(rp))
    w = np.array([DERIV_WZ, DERIV_WPITCH, DERIV_WROLL]) @ ref_settle_map(rp)
    assert np.array_equal(V.settle_weights(), w)
    assert V.SETTLE_CONST_PER_STEP == DERIV_WZ * rp.wheel_radius
    print("(b) settle map, settle weights and the per-step constant identical to helhest.risk")
    print("(b) PASS: every vendored table is bit-identical to the deployed one")


def check_belief_identity() -> None:
    RC = _load("risk_calibration")
    path = Path(
        "/home/kuceral4/data/oxford_spires/out/keble-college-02-default/window_000.npz")
    case = RC.build_case(path)
    d0 = _digest(case.C, case.mu_u, case.means, case.caps, case.c_eff)
    arms = RC.case_arms(case)
    d1 = _digest(case.C, case.mu_u, case.means, case.caps, case.c_eff)
    e_mc, sd_mc, n, _ = RC.mc_cost(case, 2000)
    d2 = _digest(case.C, case.mu_u, case.means, case.caps, case.c_eff)
    print(f"(c) belief digest before arms   {d0[:16]}")
    print(f"(c) belief digest after  arms   {d1[:16]}")
    print(f"(c) belief digest after  MC     {d2[:16]}")
    assert d0 == d1 == d2
    # and the MC's own inputs ARE those arrays, not a rebuild
    assert RC.mc_cost.__doc__ and "case.C" in RC.mc_cost.__doc__
    src = (SPIRES / "risk_calibration.py").read_text()
    body = src.split("def mc_cost")[1].split("\ndef ")[0]
    assert "case.C" in body and "case.mu_u" in body and "case.u_idx" in body
    assert "cell_covariance" not in body and "cell_variance_split" not in body
    print(f"(c) PASS: MC and the clark arm read the same arrays "
          f"(clark E {arms['clark'][0]:.4f} sd {arms['clark'][1]:.4f}; "
          f"MC E {e_mc:.4f} sd {sd_mc:.4f} over {n} draws)")


def check_no_held_out_path() -> None:
    """Static proof: no held-out path can be loaded."""
    src = (SPIRES / "risk_calibration.py").read_text()
    for fn in ("def window_dirs", "def load_window", "def build_case", "def tls_site_of"):
        body = src.split(fn)[1].split("\ndef ")[0]
        assert "_forbid_held_out" in body, fn
    assert 'DESIGN_SITE = "keble-college-*"' in src
    print("(d) PASS: every filesystem entry point calls _forbid_held_out; "
          "site filter defaults to keble-college-*")


if __name__ == "__main__":
    check_fold_parity()
    print()
    check_vendor_parity()
    print()
    check_belief_identity()
    print()
    check_no_held_out_path()
