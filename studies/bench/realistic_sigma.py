"""The ranking comparison re-run under the MEASURED correlation instead of the assumed one.

    .venv/bin/python -m studies.bench.realistic_sigma --gate         # check the generator first
    .venv/bin/python -m studies.bench.realistic_sigma --seeds 100

Criteria are pre-registered in `studies/PREREG_realistic_sigma.md` and were committed before
this ran. One variable moves: the correlation kernel, in the Monte-Carlo truth AND in every
estimator that consumes correlation. Scenes, plan families, the sigma field, the cost, the CVaR
level and the arms are all unchanged, so a difference in the result is attributable.

THE FAIRNESS POINT THAT DECIDES WHETHER THIS MEANS ANYTHING. It would be easy, and wrong, to
hand the new kernel to Clark and leave FOSM propagating the old one -- Clark would win by being
the only arm told the truth. FOSM is a quadratic form in the same covariance, so it gets the
measured kernel through exactly the same rank-M machinery. STEP-form, swept-area sigma and the
bracket consume sigma alone and are structurally unaffected.

THE GENERATOR. Terrain realisations are synthesised by FFT: white noise shaped by the square
root of the measured kernel's spectrum, then scaled per cell by the sigma field, which gives
covariance sigma_i sigma_j rho(i-j) -- the model the estimators assume, now with rho measured
rather than invented. `--gate` checks that what comes out has the variance and the
autocorrelation it was asked for, because a silently wrong generator would make every number
below meaningless.

DEVIATION FROM THE DEVICE-NATIVE RULE, DECLARED. The draws are synthesised on the host and
uploaded, rather than generated on device like `NoiseDraws` does. For 256 draws of 90x90 that is
8 MB per seed and it keeps the generator auditable against the measured kernel; it would be the
wrong choice inside the perception loop, and it is not used there.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from ..adjoint.harness import Harness
from .clark import _cost_settle
from .clark import _footprint_cells
from .clark import _settle_weights
from .clark import rho1_table
from .clark_conv import fold_weights
from .clark_conv import kernel_is_psd
from .clark_conv import separable_quadratic_multi
from .clark_conv import separable_terms
from .ranking import build_case
from .ranking import CELL
from .ranking import N_PLANS
from .ranking import OUT
from .risk import ALPHA
from .risk import empirical_cvar
from .risk import KAPPA
from .risk import N_DRAWS
from ..adjoint.harness import DERIV_WZ
from helhest.engine import RobotParams
from helhest.engine.envelope import wheel_offset_table

MEASURED_RHO = OUT / "rho_measured.npy"
RANK = 5  # PSD-projected rank-5 holds Var[J] to 0.57%; see clark_conv.separable_terms


def measured_kernel() -> np.ndarray:
    """The 2-D lag correlation measured by the sensing simulator, PSD-projected."""
    rho = np.load(MEASURED_RHO)
    terms = separable_terms(rho, RANK, enforce_psd=True)
    k = np.zeros_like(rho)
    for a, b in terms:
        k += np.outer(a, b)
    return k / k[k.shape[0] // 2, k.shape[1] // 2]


class KernelDraws:
    """Correlated terrain realisations with an arbitrary stationary kernel, by FFT synthesis."""

    def __init__(self, kernel: np.ndarray, shape: tuple[int, int]):
        ny, nx = shape
        emb = np.zeros((ny, nx))
        L = kernel.shape[0] // 2
        idx_y = (np.arange(-L, L + 1)) % ny
        idx_x = (np.arange(-L, L + 1)) % nx
        emb[np.ix_(idx_y, idx_x)] = kernel
        spec = np.fft.rfft2(emb).real
        self.amp = np.sqrt(np.maximum(spec, 0.0))
        self.shape = shape

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """`n` unit-variance fields with the kernel's correlation, [n, ny, nx]."""
        w = rng.standard_normal((n, *self.shape))
        f = np.fft.irfft2(np.fft.rfft2(w) * self.amp[None], s=self.shape)
        return f / f.std(axis=(1, 2), keepdims=True)


def gate(device: str) -> dict:
    """Does the generator produce what it was asked for? Nothing below means anything if not."""
    k = measured_kernel()
    chk = kernel_is_psd(k, separable_terms(k, RANK, enforce_psd=True))
    ny = nx = 90
    draws = KernelDraws(k, (ny, nx))
    f = draws.draw(256, np.random.default_rng(0))
    L = k.shape[0] // 2
    emp = np.zeros_like(k)
    for dy in range(-L, L + 1):
        for dx in range(-L, L + 1):
            a = f[:, L : ny - L, L : nx - L]
            b = f[:, L + dy : ny - L + dy, L + dx : nx - L + dx]
            emp[dy + L, dx + L] = float((a * b).mean())
    emp /= emp[L, L]
    err = np.abs(emp - k)
    return {
        "psd": chk["psd"], "min_spectrum_over_max": chk["min_over_max"],
        "field_std": float(f.std()),
        "max_abs_rho_error": float(err.max()),
        "median_abs_rho_error": float(np.median(err)),
        "passed": bool(chk["psd"] and err.max() < 0.05 and abs(f.std() - 1.0) < 0.05),
    }


def _clark_moments(
    belief: np.ndarray, sigma: np.ndarray, controlled: np.ndarray, rp: RobotParams,
    x0: float, y0: float, kernel: np.ndarray, terms: list, off: tuple,
) -> tuple[float, float]:
    """`clark_conv.plan_moments_conv` with a general (non-separable) kernel."""
    ny, nx = belief.shape
    bflat, sflat = belief.ravel(), sigma.ravel()
    off_dy, off_dx, off_cap, rho_kk = off
    wheel_xy = np.array([[0.0, rp.half_track], [0.0, -rp.half_track], [-rp.rear_offset, 0.0]])
    t_idx = np.arange(1, controlled.shape[0])
    n_t = len(t_idx)
    x, y, yaw = controlled[t_idx, 0], controlled[t_idx, 1], controlled[t_idx, 2]
    c, s = np.cos(yaw), np.sin(yaw)
    wx = np.stack([x + wheel_xy[w, 0] * c - wheel_xy[w, 1] * s for w in range(3)]).ravel()
    wy = np.stack([y + wheel_xy[w, 0] * s + wheel_xy[w, 1] * c for w in range(3)]).ravel()
    cells = _footprint_cells(wx, wy, off_dy, off_dx, x0, y0, CELL, ny, nx)
    means = bflat[cells] + off_cap[None, :]
    sig = sflat[cells]
    mean_n, w, order = fold_weights(means, sig, rho_kk)
    c_w = np.repeat(_settle_weights(rp), n_t)
    e_j = float(c_w @ mean_n) + n_t * DERIV_WZ * rp.wheel_radius
    cells_s = np.take_along_axis(cells, order, axis=1)
    iy, ix = cells_s // nx, cells_s % nx
    pad = kernel.shape[0]
    y_lo, x_lo = int(iy.min()) - pad, int(ix.min()) - pad
    patch = np.zeros((int(iy.max()) + pad + 1 - y_lo, int(ix.max()) + pad + 1 - x_lo))
    np.add.at(patch, (iy - y_lo, ix - x_lo), c_w[:, None] * w * sflat[cells_s])
    return e_j, max(separable_quadratic_multi(patch, terms), 0.0)


def _fosm_variance(grad: np.ndarray, sigma: np.ndarray, terms: list, kernel: np.ndarray) -> float:
    """g^T Sigma g under the SAME measured kernel -- FOSM is not handicapped by being given the
    old one while Clark gets the new one."""
    field = grad * sigma
    return max(separable_quadratic_multi(field, terms), 0.0)


def run_seed(seed: int, device: str, kernel: np.ndarray, terms: list) -> dict:
    scene, _t, _m, _o, sigma, poses, omega, _g = build_case(seed, "hybrid", "all")
    belief = scene.elevation.astype(np.float32)
    ny, nx = belief.shape
    rp = RobotParams()
    env_r = int(np.ceil(rp.wheel_radius / CELL))
    o_dy, o_dx, o_cap = wheel_offset_table(env_r, CELL, rp.wheel_radius)
    o_dy, o_dx = np.asarray(o_dy, np.int64), np.asarray(o_dx, np.int64)
    L = kernel.shape[0] // 2
    rho_kk = kernel[o_dy[:, None] - o_dy[None, :] + L, o_dx[:, None] - o_dx[None, :] + L]
    off = (o_dy, o_dx, o_cap, rho_kk)

    h = Harness(scene, poses, omega, device=device)
    grads, terms_c = h.adjoint(dilate=True, leaf="elevation")
    grad = grads[1]
    j_bel = _cost_settle(terms_c)
    controlled = h.sim.controlled.numpy()

    def forward_on(elev: np.ndarray) -> np.ndarray:
        stack = np.ascontiguousarray(np.tile(elev, (N_PLANS, 1, 1)), np.float32)
        with wp.ScopedDevice(h.device):
            h.sim.set_terrain(wp.array(stack, dtype=wp.float32))
        return _cost_settle(h.forward(dilate=True)).copy()

    j_hi = forward_on(belief + sigma.astype(np.float32))
    j_lo = forward_on(belief - sigma.astype(np.float32))

    e_c = np.empty(N_PLANS)
    sd_c = np.empty(N_PLANS)
    fosm = np.empty(N_PLANS)
    for k in range(N_PLANS):
        e_c[k], v = _clark_moments(
            belief, sigma, controlled[:, k, :], rp, scene.origin_x, scene.origin_y,
            kernel, terms, off,
        )
        sd_c[k] = np.sqrt(v)
        fosm[k] = np.sqrt(_fosm_variance(grad[k], sigma, terms, kernel))
    sig_t = np.array([
        sigma[
            np.clip(((controlled[1:, k, 1] - scene.origin_y) / CELL).astype(int), 0, ny - 1),
            np.clip(((controlled[1:, k, 0] - scene.origin_x) / CELL).astype(int), 0, nx - 1),
        ].sum()
        for k in range(N_PLANS)
    ])
    del h

    est = {
        "none": j_bel.copy(),
        "step": j_bel + KAPPA * sig_t,
        "fosm": j_bel + KAPPA * fosm,
        "bracket": np.maximum(j_bel, np.maximum(j_hi, j_lo)),
        "clark_mean": e_c.copy(),
        "clark_cvar": e_c + KAPPA * sd_c,
    }

    # --- Monte-Carlo truth, drawn from the MEASURED kernel ------------------------------------
    draws = KernelDraws(kernel, (ny, nx))
    fields = draws.draw(N_DRAWS, np.random.default_rng(900_000 + seed))
    poses_d = np.tile(poses[0], (N_DRAWS, 1)).astype(np.float32)
    omega_d = np.zeros((omega.shape[0], N_DRAWS, 3), np.float32)
    hd = Harness(scene, poses_d, omega_d, device=device)
    samples = np.empty((N_DRAWS, N_PLANS), np.float32)
    pert = (belief[None] + fields * sigma[None]).astype(np.float32)
    for k in range(N_PLANS):
        hd.sim.start_pose.assign(np.tile(poses[k], (N_DRAWS, 1)).astype(np.float32))
        hd.sim.target_wheel_omega.assign(
            np.ascontiguousarray(np.repeat(omega[:, k : k + 1, :], N_DRAWS, axis=1), np.float32)
        )
        with wp.ScopedDevice(hd.device):
            hd.sim.set_terrain(wp.array(np.ascontiguousarray(pert), dtype=wp.float32))
        samples[:, k] = _cost_settle(hd.forward(dilate=True))
    del hd

    mc_cvar = empirical_cvar(samples, ALPHA)
    best = int(np.argmin(mc_cvar))
    out = {"seed": seed, "arms": {}}
    for name, v in est.items():
        pick = int(np.argmin(v))
        out["arms"][name] = {
            "regret": float(mc_cvar[pick] - mc_cvar[best]),
            "picked_best": bool(pick == best),
        }
    out["calib"] = {
        "sd_clark": sd_c.tolist(), "mc_sd": samples.std(axis=0).tolist(),
        "e_clark": e_c.tolist(), "mc_mean": samples.mean(axis=0).tolist(),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args()
    wp.init()

    g = gate(args.device)
    print("=== generator gate ===")
    for k, v in g.items():
        print(f"  {k:26s} {v}")
    if not g["passed"]:
        print("GATE FAILED -- not running the comparison on a generator that is wrong.")
        return
    if args.gate:
        return

    kernel = measured_kernel()
    terms = separable_terms(kernel, RANK, enforce_psd=True)
    rows = []
    t0 = time.perf_counter()
    for seed in range(args.seeds):
        rows.append(run_seed(seed, args.device, kernel, terms))
        if (seed + 1) % 10 == 0:
            print(f"  {seed + 1}/{args.seeds} seeds, {(time.perf_counter() - t0) / 60:.1f} min")

    from math import comb

    def sign_p(d: np.ndarray) -> tuple[int, int, float]:
        nz = d[d != 0]
        n, kk = len(nz), int((nz > 0).sum())
        if n == 0:
            return 0, 0, 1.0
        tail = sum(comb(n, i) for i in range(min(kk, n - kk) + 1))
        return n, kk, min(1.0, 2.0 * tail / 2**n)

    arms = list(rows[0]["arms"].keys())
    regret = {a: float(np.mean([r["arms"][a]["regret"] for r in rows])) for a in arms}
    best_arm = min(regret, key=regret.get)
    tests = {}
    for other in ("none", "step", "bracket", "fosm"):
        d = np.array([
            r["arms"]["clark_cvar"]["regret"] - r["arms"][other]["regret"] for r in rows
        ])
        n, kk, p = sign_p(-d)
        tests[other] = {"mean_diff": float(d.mean()), "wins": kk, "n": n, "p": p}
    sd = np.concatenate([r["calib"]["sd_clark"] for r in rows])
    mc = np.concatenate([r["calib"]["mc_sd"] for r in rows])
    ratio = sd / np.maximum(mc, 1e-9)
    crit = {
        "i_clark_lowest_mean_regret": best_arm == "clark_cvar",
        "ii_beats_none_p05": tests["none"]["p"] < 0.05 and tests["none"]["mean_diff"] < 0,
        "iii_beats_step_p05": tests["step"]["p"] < 0.05 and tests["step"]["mean_diff"] < 0,
        "iv_sd_ratio_in_band": bool(0.7 <= float(np.median(ratio)) <= 1.4),
    }
    print("\n=== hybrid/all under the MEASURED kernel ===")
    for a, v in sorted(regret.items(), key=lambda kv: kv[1]):
        print(f"  {a:12s} {v:.4f}")
    for o, v in tests.items():
        print(f"  clark_cvar vs {o:8s} mean {v['mean_diff']:+.4f}  {v['wins']}/{v['n']}  "
              f"p={v['p']:.2e}")
    print(f"  pooled sd-ratio median {float(np.median(ratio)):.3f}")
    for k, v in crit.items():
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  VERDICT: {'PASSED' if all(crit.values()) else 'FAILED'}")
    path = OUT / "realistic_sigma.json"
    path.write_text(json.dumps(
        {"gate": g, "rows": rows, "mean_regret": regret, "tests": tests,
         "sd_ratio_median": float(np.median(ratio)), "criteria": crit,
         "passed": all(crit.values())}, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
