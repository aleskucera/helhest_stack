"""The two untried risk estimators: the BUNDLED adjoint, and uncertainty BRACKETING.

    .venv/bin/python -m studies.bench.bundled --seeds 100

`risk.py` established that the point adjoint cannot rank plans by risk, with a mechanism:
the gradient is evaluated at the one belief map, whose flat inpainted regions starve it of
support (the catch-22) and whose contact arg-max makes it valid over millimetres while sigma
is centimetres (the validity radius). Both mechanisms are properties of differentiating AT A
SINGLE POINT — so the two estimators that do not are tested here, against the identical
protocol, truth, baselines and seeds:

  bundled_pt  J(belief) + kappa * sqrt(Var_FOSM(g_bar)) with g_bar the MEAN ADJOINT over
              M terrain draws from the belief (Suh-Pang-Tedrake randomized smoothing, applied
              to the terrain). Each draw has real relief on inpainted cells, so support
              spreads onto them; the average is the gradient of the sigma-smoothed cost, so
              its validity scale is sigma by construction. Isolates the smoothing effect —
              same formula as `fosm`, only the gradient changes.
  bundled     mean_m J_m + the same tail: the full use of the bundle (its M forwards already
              estimate E[J], which the belief cost misses by the Jensen bias).
  bracket     elementwise max of J(belief), J(belief + sigma), J(belief - sigma): the
              pessimistic two-rollout bracket of IMPROVEMENTS.md section 9. No gradient, no
              draw — the coarsest neighbourhood evaluation that is honest at sigma scale.

And the control that decides whether the bundle EARNS its backward passes:

  mc_small    the empirical CVaR of the SAME M draws the bundle rolled forward — identical
              terrains, identical compute minus every backward pass. If this matches or beats
              `bundled`, the gradient work is dead weight at equal budget; the bundle's only
              defensible product is then the per-cell field g_bar (a sensing quantity), not
              the risk scalar.

Everything else — none / step / fosm / mc — is exactly `risk.py`'s ladder, recomputed here so
all arms share seeds and draws and every comparison is paired.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import warp as wp

from ..adjoint.harness import Harness
from ..adjoint.harness import N_TERMS
from ..adjoint.harness import TERM_NAMES
from ..adjoint.sigma import _gauss_kernel
from ..adjoint.sigma import fosm_variance
from ..adjoint.sigma import NoiseDraws
from .ranking import _cost
from .ranking import build_case
from .ranking import CELL
from .ranking import COST_TERMS
from .ranking import kendall_tau
from .ranking import N_PLANS
from .ranking import OUT
from .ranking import sign_test
from .risk import _footprint_sigma
from .risk import ALPHA
from .risk import CORR_LEN
from .risk import empirical_cvar
from .risk import KAPPA
from .risk import N_DRAWS

M_BUNDLE = 16  # draws in the bundle; mc_small uses the SAME draws so the budget is matched
BRACKET_C = 1.0  # bracket at belief +- c * sigma


def _correlated_field(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """One unit-marginal-variance correlated field, the same model `NoiseDraws` samples:
    white noise, separable Gaussian blur at CORR_LEN, renormalised. Host-side because the
    bundle needs the SAME field for all K plan slices (every plan sees one world per draw),
    which the per-slice-independent device generator cannot produce."""
    w, radius = _gauss_kernel(CORR_LEN, CELL)
    f = rng.standard_normal(shape)
    if radius > 0:
        pad = np.pad(f, radius, mode="edge")
        f = np.apply_along_axis(lambda m: np.convolve(m, w, mode="valid"), 1, pad)
        f = np.apply_along_axis(lambda m: np.convolve(m, w, mode="valid"), 0, f)
        f /= (w**2).sum()  # two blur axes each scale the std by sqrt(sum(w^2))
    return f


def _weighted_adjoint(h: Harness, elev2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """d(cost)/d(elevation) [K, ny, nx] and the costs [K], in ONE backward pass: the cotangent
    on `terms` carries the COST_TERMS weights instead of `Harness.adjoint`'s per-term one-hots.
    """
    stack = np.ascontiguousarray(np.tile(elev2d, (N_PLANS, 1, 1)), np.float32)
    with wp.ScopedDevice(h.device):
        h.sim.set_terrain(wp.array(stack, dtype=wp.float32))
        wp.copy(h.sim.friction, h._fric0)
    tape = wp.Tape()
    h._rollout(dilate=True, tape=tape)
    costs = _cost(h.terms.numpy()).copy()
    seed_np = np.zeros((N_TERMS, N_PLANS), np.float32)
    for name, weight in COST_TERMS.items():
        seed_np[TERM_NAMES.index(name)] = weight
    seed = wp.zeros_like(h.terms)
    seed.assign(seed_np)
    tape.backward(grads={h.terms: seed})
    grad = h.sim.elevation.grad.numpy().copy()
    tape.zero()
    return grad, costs


def run_seed(seed: int, family: str, noise: str, m_bundle: int) -> dict:
    scene, _truth, _meas, _obs, sigma, poses, omega, grid = build_case(seed, family, noise)
    belief = scene.elevation.astype(np.float32)

    h = Harness(scene, poses, omega, device="cuda")

    # --- point adjoint (the refuted baseline, recomputed so every comparison is paired) ---
    grads, terms = h.adjoint(dilate=True, leaf="elevation")
    grad_pt = sum(w * grads[TERM_NAMES.index(k)] for k, w in COST_TERMS.items())
    j_bel = _cost(terms)
    traj = h.sim.controlled.numpy()[:, :, :2].copy()
    sig_t = _footprint_sigma(traj, sigma, grid)

    if seed == 0:  # gate: the weighted single-backward must reproduce the per-term adjoint
        g1, c1 = _weighted_adjoint(h, belief)
        rel = np.abs(g1 - grad_pt).max() / max(np.abs(grad_pt).max(), 1e-12)
        assert rel < 1e-4, f"weighted-cotangent adjoint disagrees with per-term: {rel:.2e}"
        assert np.allclose(c1, j_bel, atol=1e-4)

    # --- bracket: two extra forwards, no gradient ------------------------------------
    def _forward_on(elev2d: np.ndarray) -> np.ndarray:
        stack = np.ascontiguousarray(np.tile(elev2d, (N_PLANS, 1, 1)), np.float32)
        with wp.ScopedDevice(h.device):
            h.sim.set_terrain(wp.array(stack, dtype=wp.float32))
        return _cost(h.forward(dilate=True)).copy()

    j_hi = _forward_on(belief + BRACKET_C * sigma.astype(np.float32))
    j_lo = _forward_on(belief - BRACKET_C * sigma.astype(np.float32))

    # --- the bundle: M draws, one weighted backward each ------------------------------
    rng = np.random.default_rng(700_000 + seed)  # independent of the truth draws
    g_sum = np.zeros_like(grad_pt)
    j_draws = np.empty((m_bundle, N_PLANS), np.float32)
    for m in range(m_bundle):
        field = _correlated_field(rng, belief.shape)
        g_m, j_draws[m] = _weighted_adjoint(h, belief + (field * sigma).astype(np.float32))
        g_sum += g_m
    g_bar = g_sum / m_bundle
    del h

    def _fosm_sd(g: np.ndarray) -> np.ndarray:
        return np.sqrt(
            [max(fosm_variance(g[k], sigma, CELL, CORR_LEN), 0.0) for k in range(N_PLANS)]
        )

    sd_pt, sd_bar = _fosm_sd(grad_pt), _fosm_sd(g_bar)
    est = {
        "none": j_bel.copy(),
        "step": j_bel + KAPPA * sig_t.sum(axis=0),
        "fosm": j_bel + KAPPA * sd_pt,
        "bracket": np.maximum(j_bel, np.maximum(j_hi, j_lo)),
        "bundled_pt": j_bel + KAPPA * sd_bar,
        "bundled": j_draws.mean(axis=0) + KAPPA * sd_bar,
        "mc_small": empirical_cvar(j_draws, ALPHA),
    }

    # --- Monte-Carlo truth: identical protocol to risk.py (common random numbers) -----
    ny, nx = belief.shape
    poses_d = np.tile(poses[0], (N_DRAWS, 1)).astype(np.float32)
    omega_d = np.zeros((omega.shape[0], N_DRAWS, 3), np.float32)
    hd = Harness(scene, poses_d, omega_d, device="cuda")
    draws = NoiseDraws((N_DRAWS, ny, nx), CELL, CORR_LEN, hd.device)
    with wp.ScopedDevice(hd.device):
        base = wp.array(np.ascontiguousarray(np.tile(belief, (N_DRAWS, 1, 1)), np.float32))
        sig_dev = wp.array(np.ascontiguousarray(sigma, np.float32), dtype=wp.float32)
    samples = np.empty((N_DRAWS, N_PLANS), np.float32)
    for k in range(N_PLANS):
        hd.sim.start_pose.assign(np.tile(poses[k], (N_DRAWS, 1)).astype(np.float32))
        hd.sim.target_wheel_omega.assign(
            np.ascontiguousarray(np.repeat(omega[:, k : k + 1, :], N_DRAWS, axis=1), np.float32)
        )
        draws.perturb(base, sig_dev, 1.0, hd.sim.elevation, 900_000 + seed)
        samples[:, k] = _cost(hd.forward(dilate=True))
    del hd

    mc_cvar = empirical_cvar(samples, ALPHA)
    best = int(np.argmin(mc_cvar))
    true_risk = mc_cvar - j_bel

    out = {"seed": seed, "mc_cvar_spread": float(mc_cvar.max() - mc_cvar.min()), "arms": {}}
    for name, v in est.items():
        pick = int(np.argmin(v))
        out["arms"][name] = {
            "regret": float(mc_cvar[pick] - mc_cvar[best]),
            "tau": kendall_tau(v, mc_cvar),
            "picked_best": bool(pick == best),
            "cvar_err": float(np.mean(np.abs(v - mc_cvar))),
        }
    out["arms"]["mc"] = {"regret": 0.0, "tau": 1.0, "picked_best": True, "cvar_err": 0.0}
    # the risk TERMS alone, ranked against the true risk (units cancel under tau)
    out["surrogates"] = {
        "point_fosm_sd": kendall_tau(sd_pt, true_risk),
        "bundled_fosm_sd": kendall_tau(sd_bar, true_risk),
        "bracket_up": kendall_tau(np.maximum(j_hi, j_lo) - j_bel, true_risk),
        "step_sd": kendall_tau(sig_t.sum(axis=0), true_risk),
        "mc_small_risk": kendall_tau(est["mc_small"] - j_bel, true_risk),
    }
    # does bundling fix the catch-22? mass of |g| on high-sigma cells, point vs bundled
    hi_sig = sigma > 0.5 * sigma.max()
    for tag, g in (("point", grad_pt), ("bundled", g_bar)):
        a = np.abs(g).sum(axis=0)
        out[f"gradmass_hisig_{tag}"] = float(a[hi_sig].sum() / max(a.sum(), 1e-12))
    return out


ARM_ORDER = ("none", "step", "fosm", "bracket", "bundled_pt", "bundled", "mc_small", "mc")


def report(rows: list[dict], m_bundle: int) -> None:
    n = len(rows)
    print(f"\nn={n} seeds, bundle M={m_bundle} draws, truth {N_DRAWS} draws, CVaR alpha={ALPHA}")
    print(
        f"true CVaR spread across plans: {np.mean([r['mc_cvar_spread'] for r in rows]):.2f}\n"
    )
    print("DECISION QUALITY -- pick the argmin, pay its true CVaR")
    print(f"{'estimator':<12}{'regret':>9}{'picked best':>13}{'tau vs truth':>14}{'|cvar err|':>12}")
    for a in ARM_ORDER:
        rg = np.mean([r["arms"][a]["regret"] for r in rows])
        pb = np.mean([r["arms"][a]["picked_best"] for r in rows])
        tt = np.mean([r["arms"][a]["tau"] for r in rows])
        ce = np.mean([r["arms"][a]["cvar_err"] for r in rows])
        print(f"{a:<12}{rg:>9.3f}{pb:>12.0%}{tt:>+14.3f}{ce:>12.3f}")

    print("\nrisk TERM alone vs true risk (mc_cvar - J(belief)), Kendall tau")
    for k in ("step_sd", "point_fosm_sd", "bundled_fosm_sd", "bracket_up", "mc_small_risk"):
        print(f"  {k:<16}{np.mean([r['surrogates'][k] for r in rows]):>+8.3f}")

    print("\nfraction of |gradient| mass on high-sigma cells (the catch-22, measured)")
    for tag in ("point", "bundled"):
        print(f"  {tag:<8}{np.mean([r[f'gradmass_hisig_{tag}'] for r in rows]):>8.3f}")

    print("\npaired regret differences (negative = first arm better)")
    pairs = (
        ("bundled_pt", "fosm"),  # does smoothing the gradient fix the point FOSM?
        ("bundled", "mc_small"),  # does the gradient earn its backwards at equal draws?
        ("bundled", "step"),
        ("bracket", "none"),
        ("bracket", "step"),
        ("mc_small", "step"),
    )
    for a, b in pairs:
        d = np.array([r["arms"][a]["regret"] - r["arms"][b]["regret"] for r in rows])
        k, w, p = sign_test(-d)
        print(f"  {a:<10} vs {b:<10} mean {d.mean():>+7.3f}   better on {w:>3}/{k:<3}   p={p:.2e}")
    print("\npaired tau differences (positive = first arm better)")
    for a, b in pairs:
        d = np.array([r["arms"][a]["tau"] - r["arms"][b]["tau"] for r in rows])
        k, w, p = sign_test(d)
        print(f"  {a:<10} vs {b:<10} mean {d.mean():>+7.3f}   better on {w:>3}/{k:<3}   p={p:.2e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--family", default="hybrid")
    ap.add_argument("--noise", default="all")
    ap.add_argument("--draws", type=int, default=M_BUNDLE)
    a = ap.parse_args()

    wp.init()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(a.seeds):
        rows.append(run_seed(seed, a.family, a.noise, a.draws))
        if (seed + 1) % 10 == 0:
            print(f"  {seed + 1}/{a.seeds} seeds", flush=True)
    report(rows, a.draws)
    path = OUT / f"bundled_{a.family}_{a.noise}.json"
    path.write_text(json.dumps({"family": a.family, "noise": a.noise, "rows": rows}, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
