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

BELIEF_MODEL = OUT / "belief_model.npz"
RANK = 5  # PSD-projected rank-5 holds Var[J] to 0.57%; see clark_conv.separable_terms


class BeliefModel:
    """Sigma = a rank-3 plane plus a compact stationary kernel, fitted to simulated sensing.

    The first attempt used one stationary kernel and its generator gate failed: the measured
    correlation never decays inside the window, because a scan's pose error shifts and TILTS
    every point in it together, and no stationary kernel can express that. Splitting it off
    leaves a residual that decorrelates within three cells. Measured split: 30% plane, 70%
    stationary.

    Marginal per-cell variance is preserved at sigma_p^2 -- the pre-registration fixes the sigma
    field and moves only the correlation -- so the two parts carry `share` and `1 - share` of it:

        e(p) = sigma_p [ sqrt(share)/sd_P (a + b xc_p + c yc_p) + sqrt(1-share) s(p) ]
    """

    def __init__(self, path: Path = BELIEF_MODEL):
        d = np.load(path)
        self.plane_cov = d["plane_cov"]
        self.share = float(d["plane_share"])
        self.rho_s = d["rho_stationary"]
        self.xc, self.yc = float(d["x_center"]), float(d["y_center"])
        self.terms = separable_terms(self.rho_s, RANK, enforce_psd=True)

    def basis(self, ny: int, nx: int, x0: float, y0: float) -> np.ndarray:
        gy, gx = np.mgrid[0:ny, 0:nx]
        return np.stack([
            np.ones((ny, nx)), gx * CELL + x0 - self.xc, gy * CELL + y0 - self.yc
        ])

    def sd_plane(self, ny: int, nx: int, x0: float, y0: float) -> float:
        """RMS of the plane over the map, the constant that normalizes it to unit variance."""
        b = self.basis(ny, nx, x0, y0).reshape(3, -1)
        return float(np.sqrt(np.mean(np.einsum("in,ij,jn->n", b, self.plane_cov, b))))

    def rho_kk(self, off_dy: np.ndarray, off_dx: np.ndarray) -> np.ndarray:
        """Correlation between a node's own candidates. The plane is smooth over a 0.9 m
        footprint, so its contribution there is `share` to within a fraction of a percent --
        stated as an approximation rather than hidden."""
        L = self.rho_s.shape[0] // 2
        dy = off_dy[:, None] - off_dy[None, :]
        dx = off_dx[:, None] - off_dx[None, :]
        return self.share + (1.0 - self.share) * self.rho_s[dy + L, dx + L]

    def variance_of(
        self, field: np.ndarray, ny: int, nx: int, x0: float, y0: float
    ) -> float:
        """Var of a linear functional of the cells, EXACT in both parts: a 3x3 form for the
        plane, the rank-M convolution for the stationary residual."""
        b = self.basis(ny, nx, x0, y0)
        u = np.array([float((field * b[i]).sum()) for i in range(3)])
        sdp = self.sd_plane(ny, nx, x0, y0)
        v_plane = self.share / sdp**2 * float(u @ self.plane_cov @ u)
        v_stat = (1.0 - self.share) * separable_quadratic_multi(field, self.terms)
        return max(v_plane + v_stat, 0.0)


class TwoPartDraws:
    """Unit-variance realisations of the two-part model: a random plane plus a stationary field
    synthesised by FFT. The stationary kernel decays inside three cells, so the zero-padding
    that broke the single-kernel generator is harmless here."""

    def __init__(self, model: BeliefModel, shape: tuple[int, int], x0: float, y0: float):
        ny, nx = shape
        self.m = model
        self.shape = shape
        emb = np.zeros((ny, nx))
        L = model.rho_s.shape[0] // 2
        k = np.zeros_like(model.rho_s)
        for a, b in model.terms:
            k += np.outer(a, b)
        k /= k[L, L]
        emb[np.ix_(np.arange(-L, L + 1) % ny, np.arange(-L, L + 1) % nx)] = k
        self.amp = np.sqrt(np.maximum(np.fft.rfft2(emb).real, 0.0))
        self.basis = model.basis(ny, nx, x0, y0)
        self.sdp = model.sd_plane(ny, nx, x0, y0)
        self.chol = np.linalg.cholesky(model.plane_cov + 1e-15 * np.eye(3))

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        w = rng.standard_normal((n, *self.shape))
        s = np.fft.irfft2(np.fft.rfft2(w) * self.amp[None], s=self.shape)
        s /= s.std(axis=(1, 2), keepdims=True)
        c = rng.standard_normal((n, 3)) @ self.chol.T
        plane = np.einsum("nc,cyx->nyx", c, self.basis) / self.sdp
        return np.sqrt(self.m.share) * plane + np.sqrt(1.0 - self.m.share) * s


def gate(device: str) -> dict:
    """End-to-end: does the variance the estimator PREDICTS for a linear functional match the
    variance the generator actually produces for it?

    Checking that the synthesised kernel matches the target would only test the generator. This
    tests the generator and the estimator against each other, which is the property every number
    below depends on -- and it is what the single-kernel attempt failed.
    """
    m = BeliefModel()
    ny = nx = 90
    x0 = y0 = 0.0
    draws = TwoPartDraws(m, (ny, nx), x0, y0)
    fields = draws.draw(2048, np.random.default_rng(0))
    rng = np.random.default_rng(3)
    errs = []
    for _ in range(24):
        g = np.zeros((ny, nx))
        iy = rng.integers(8, ny - 8, 200)
        ix = rng.integers(8, nx - 8, 200)
        np.add.at(g, (iy, ix), rng.normal(0, 1, 200))
        emp = float(np.var(np.einsum("nyx,yx->n", fields, g)))
        pred = m.variance_of(g, ny, nx, x0, y0)
        errs.append(abs(pred - emp) / max(emp, 1e-12))
    # and the split the model claims
    b = m.basis(ny, nx, x0, y0).reshape(3, -1)
    proj = np.linalg.lstsq(b.T, fields.reshape(len(fields), -1).T, rcond=None)[0]
    plane = (b.T @ proj).T.reshape(fields.shape)
    share_emp = float(np.var(plane) / np.var(fields))
    return {
        "field_std": float(fields.std()),
        "plane_share_model": m.share,
        "plane_share_realized": share_emp,
        "median_var_error": float(np.median(errs)),
        "max_var_error": float(np.max(errs)),
        "passed": bool(
            abs(fields.std() - 1.0) < 0.05
            and abs(share_emp - m.share) < 0.05
            and np.median(errs) < 0.05
        ),
    }


def _clark_moments(
    belief: np.ndarray, sigma: np.ndarray, controlled: np.ndarray, rp: RobotParams,
    x0: float, y0: float, model: BeliefModel, off: tuple,
) -> tuple[float, float]:
    """`clark_conv.plan_moments_conv` under the two-part belief."""
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
    field = np.zeros((ny, nx))
    np.add.at(field, (cells_s // nx, cells_s % nx), c_w[:, None] * w * sflat[cells_s])
    return e_j, model.variance_of(field, ny, nx, x0, y0)


def _fosm_variance(
    grad: np.ndarray, sigma: np.ndarray, model: BeliefModel, x0: float, y0: float
) -> float:
    """g^T Sigma g under the SAME two-part belief -- FOSM is not handicapped by being given the
    old kernel while Clark gets the new one."""
    ny, nx = sigma.shape
    return model.variance_of(grad * sigma, ny, nx, x0, y0)


def run_seed(seed: int, device: str, model: BeliefModel,
             family: str = "hybrid", noise: str = "all") -> dict:
    scene, _t, _m, _o, sigma, poses, omega, _g = build_case(seed, family, noise)
    belief = scene.elevation.astype(np.float32)
    ny, nx = belief.shape
    rp = RobotParams()
    env_r = int(np.ceil(rp.wheel_radius / CELL))
    o_dy, o_dx, o_cap = wheel_offset_table(env_r, CELL, rp.wheel_radius)
    o_dy, o_dx = np.asarray(o_dy, np.int64), np.asarray(o_dx, np.int64)
    off = (o_dy, o_dx, o_cap, model.rho_kk(o_dy, o_dx))

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
            belief, sigma, controlled[:, k, :], rp, scene.origin_x, scene.origin_y, model, off,
        )
        sd_c[k] = np.sqrt(v)
        fosm[k] = np.sqrt(
            _fosm_variance(grad[k], sigma, model, scene.origin_x, scene.origin_y)
        )
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
    draws = TwoPartDraws(model, (ny, nx), scene.origin_x, scene.origin_y)
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

    out_samples = samples.copy()
    mc_cvar = empirical_cvar(samples, ALPHA)
    best = int(np.argmin(mc_cvar))
    out = {"seed": seed, "arms": {}}
    for name, v in est.items():
        pick = int(np.argmin(v))
        out["arms"][name] = {
            "regret": float(mc_cvar[pick] - mc_cvar[best]),
            "picked_best": bool(pick == best),
        }
    out["_samples"] = out_samples
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
    ap.add_argument("--family", default="hybrid")
    ap.add_argument("--noise", default="all")
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

    model = BeliefModel()
    rows = []
    t0 = time.perf_counter()
    for seed in range(args.seeds):
        rows.append(run_seed(seed, args.device, model, args.family, args.noise))
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
    # matched-budget curve under the SAME measured belief, so the paper does not quote a
    # sample-count from one noise model beside regrets from another
    clark_r = float(np.mean([r["arms"]["clark_cvar"]["regret"] for r in rows]))
    curve = []
    # Score the nd-draw pick against HELD-OUT draws. Scoring it against a "truth" built from
    # the same N_DRAWS pool (old code: truth = empirical_cvar(sm), pick from sm[:nd]) let the
    # pick grade its own homework -- at nd == N_DRAWS the pick pool and the truth pool were
    # literally the same array, so regret was identically 0 there by construction, and every
    # nd < N_DRAWS was biased down by the same overlap. Splitting N_DRAWS in half keeps the
    # pick and the truth disjoint at every nd, at the cost of halving the top of the curve's
    # n-axis (an independent fresh truth block would restore N_DRAWS but doubles the MC
    # forward-pass cost of every seed, which this fix does not spend).
    half = N_DRAWS // 2
    for nd in (4, 8, 16, 32, 64, half):
        reg = []
        for r in rows:
            sm = r["_samples"]
            sm_pick, sm_truth = sm[:half], sm[half:]
            true_c = empirical_cvar(sm_truth, ALPHA)
            pick = int(np.argmin(empirical_cvar(sm_pick[:nd], ALPHA)))
            reg.append(float(true_c[pick] - true_c[int(np.argmin(true_c))]))
        curve.append({"n": nd, "mean_regret": float(np.mean(reg))})
    n_star = next((c["n"] for c in curve if c["mean_regret"] <= clark_r), None)
    print(f"  matched budget: MC needs N = {n_star} draws to match clark_cvar "
          f"({clark_r:.4f}); curve " + ", ".join(f"{c['n']}:{c['mean_regret']:.3f}" for c in curve))
    for r in rows:
        r.pop("_samples")
    sd = np.concatenate([r["calib"]["sd_clark"] for r in rows])
    mc = np.concatenate([r["calib"]["mc_sd"] for r in rows])
    ratio = sd / np.maximum(mc, 1e-9)
    crit = {
        "i_clark_lowest_mean_regret": best_arm == "clark_cvar",
        "ii_beats_none_p05": tests["none"]["p"] < 0.05 and tests["none"]["mean_diff"] < 0,
        "iii_beats_step_p05": tests["step"]["p"] < 0.05 and tests["step"]["mean_diff"] < 0,
        "iv_sd_ratio_in_band": bool(0.7 <= float(np.median(ratio)) <= 1.4),
    }
    print(f"\n=== {args.family}/{args.noise} under the MEASURED belief ===")
    for a, v in sorted(regret.items(), key=lambda kv: kv[1]):
        print(f"  {a:12s} {v:.4f}")
    for o, v in tests.items():
        print(f"  clark_cvar vs {o:8s} mean {v['mean_diff']:+.4f}  {v['wins']}/{v['n']}  "
              f"p={v['p']:.2e}")
    print(f"  pooled sd-ratio median {float(np.median(ratio)):.3f}")
    for k, v in crit.items():
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  VERDICT: {'PASSED' if all(crit.values()) else 'FAILED'}")
    path = OUT / f"realistic_sigma_{args.family}_{args.noise}.json"
    path.write_text(json.dumps(
        {"gate": g, "rows": rows, "mean_regret": regret, "tests": tests,
         "budget_curve": curve, "n_star": n_star,
         "sd_ratio_median": float(np.median(ratio)), "criteria": crit,
         "passed": all(crit.values())}, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
