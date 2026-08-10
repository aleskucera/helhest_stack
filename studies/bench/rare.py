"""Rare-event flip probability: does a gradient-guided attack beat derivative-free subset
simulation, at matched compute, when P(flip) is small (1e-2 .. 1e-4)?

    .venv/bin/python -m studies.bench.rare

THE QUESTION. A planner commits to one of K well-separated candidate routes on an uncertain
elevation map. studies/CLAIMS.md's own repeated finding (this project's) is that a gradient
and a sampling-based estimator TIE at cm-scale perturbations -- the derivative earns nothing
extra there. The one place a derivative is SUPPOSED to matter and sampling is SUPPOSED to
struggle is the rare-event regime: when P(flip) is small enough that naive Monte-Carlo needs
huge N to see even one flip, while FORM/importance-sampling need only find the boundary once
and reason about the local geometry around it. This is the final, pre-registered test of that
one theoretically-distinct niche. If subset simulation ties or wins anyway, that is the result,
stated plainly (see the SUCCESS CRITERION below and its verdict in `main`'s printout).

WHERE THE MACHINERY COMES FROM.
  - studies/bench/certify.py: the iterated, multi-restart, softmin-over-rivals attack, and its
    noise geometry (`_blur`, `_softmin_direction`, `_bracket_gap`). Its own `_run_attack` /
    `_attack_with_restarts` / `_bisect_flip` / `_weighted_adjoint` / `_evaluate` are NOT reused
    directly: they close over the module-level `N_PLANS = 16` (ranking.py's fan/hybrid family),
    but this study prunes the candidate set to K = 4 -- reusing them naively would silently tile
    every candidate elevation to batch 16 and crash (or worse, silently mismatch) against a
    batch-4 harness. So the low-level, N_PLANS-free pieces are imported, and the pieces that
    depend on batch size are copied here with `4` in the name, adapted to `harness.batch_size`.
  - studies/bench/ranking.py: `build_case(seed, "fan", "sensor")` for the terrain, sigma, and
    the 16-corridor fan. PLAN_GROUPS["fan"] is 16 singleton groups (every plan its own corridor,
    not 4 groups of 4) -- so K = 4 is obtained by keeping the plans ranked {1st, 5th, 9th, 13th}
    by BELIEF cost, exactly as the top-level instructions specify for this case.
  - studies/adjoint/sigma.py `NoiseDraws`: the device-resident correlated noise generator used
    for every Monte-Carlo draw (quick screen, reference truth, mc_naive). Its "gain" argument is
    always called with 1.0 here; the per-case gain is baked into `sigma_case = gain * sigma_base`
    ONCE, so every downstream piece (the attack's noise geometry, NoiseDraws, and the exact
    latent construction below) shares one sigma field and "gain" never needs to be threaded
    through twice. This matches the top-level framing exactly ("the gain knob scales sigma
    globally").

THE EXACT WHITENED LATENT (for `attack_is`, method 3). NoiseDraws' generative model is LINEAR:
draw an iid white field z ~ N(0, I) (one float per cell, `wp.randn`), blur it with a fixed
separable Gaussian W (`_gauss_kernel`), rescale by 1 / sum(w^2) per axis-pair to restore unit
per-cell marginal variance (`NoiseDraws.renorm`, replicated here as `_RENORM`), then scale by
sigma: delta_h = sigma * _RENORM * blur2D(z). Because this is linear in z, z IS the model's own
whitened Gaussian latent -- not an approximation of one. The attack's accumulated step field is,
by the SAME construction (`_attack_direction`: step = sigma * blur(u), ||u|| = 1), exactly
`sigma * _RENORM * blur2D((eta / _RENORM) * u)` for a single step of size eta along direction u
-- so tracking `z_star = sum_t (eta_t / _RENORM) * (+-u_t)` over every accepted step (plus each
restart's own pre-blur white sample, scaled by RESTART_SIGMA, plus the bisection's final
fraction) reproduces the attack's ENTIRE physical excursion exactly, in the SAME coordinates
NoiseDraws samples in. That is what lets `attack_is` compute importance weights exactly
(w_i = N(z_i; 0, I) / N(z_i; z_star, I) = exp(0.5||z_star||^2 - z_i . z_star)) rather than by a
Monte-Carlo approximation of the weight itself.

FORWARD-PASS ACCOUNTING. One "forward pass" = one candidate elevation map evaluated for a
flip/no-flip decision against all K = 4 candidates -- matching certify.py's own convention
(`_evaluate`'s single kernel launch already returns all K costs, because the K=4 harness's
batch axis IS the plan axis). For the MC-heavy methods (mc_naive, the IS draws, subset's raw and
repopulated batches), the SAME convention is used even though the *implementation* loops over
the 4 plans sequentially with the draws on the batch axis (more efficient for large D): one
"forward pass" is charged per DRAW, not per (draw, plan) pair, because evaluating a candidate's
decision against all 4 rivals is the unit of information a flip check needs, whichever way it
is computed on the GPU.

BATCH SIZE. The task suggests 4096-draw reference batches; this machine's GPU (RTX A500, 4 GiB)
is shared with another job and had ~2.2 GiB free when this ran, so `DRAW_BATCH = 1024` is used
instead (more batches, same total draws, ~3x the memory headroom) -- noted here so the deviation
from the brief is not silent.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from math import erf
from math import sqrt
from pathlib import Path

import numpy as np
import warp as wp

from ..adjoint.harness import Harness
from ..adjoint.harness import N_TERMS
from ..adjoint.harness import TERM_NAMES
from ..adjoint.sigma import _gauss_kernel
from ..adjoint.sigma import NoiseDraws
from .certify import _blur
from .certify import _bracket_gap
from .certify import _softmin_direction
from .certify import BISECT_ITERS
from .certify import ETA0
from .certify import ETA_FLOOR
from .certify import MAX_STEPS
from .certify import N_RESTARTS
from .certify import RESTART_SIGMA
from .certify import SOFTMIN_TEMP_FRAC
from .ranking import _cost
from .ranking import build_case
from .ranking import CELL
from .ranking import COST_TERMS
from .ranking import kendall_tau
from .ranking import OUT
from .risk import CORR_LEN

# --- study configuration -------------------------------------------------------------------
K = 4  # pruned candidate-route count (see module docstring: {1st, 5th, 9th, 13th} by belief cost)
FAMILY = "fan"
NOISE = "sensor"
SEEDS = tuple(range(10))
GAINS = (1.0, 0.5, 0.3)
SELECT_RANKS = (0, 4, 8, 12)  # indices into the belief-cost-sorted 16-plan fan

DRAW_BATCH = 1024  # see module docstring: reduced from the suggested 4096 for shared-GPU memory
REF_MIN_FLIPS = 50
REF_MAX_DRAWS = 200_000
LOGP_LO, LOGP_HI = -4.0, np.log10(0.2)  # target span for case selection
P_DISCARD_HI = 0.3  # not rare -- discarded from the testbed
SELECT_TARGET_N = 20

K_CAP_RARE = 4.0  # certify.py censors the attack at 3 sigma; Phi(-3) = 1.3e-3 cannot even reach
# this study's target floor of 1e-4, while Phi(-4) = 3.2e-5 can, so the cap is extended to 4.

MC_NAIVE_DRAWS = 1500  # method 1's whole budget
FORM_FORWARD_BUDGET = 1000  # method 2/3's shared attack, budget-capped per the top-level brief
IS_DRAWS = 650  # method 3's importance-sampling draws, on top of the shared attack
SUBSET_N = 500  # method 4: samples per level
SUBSET_P0 = 0.2  # conditional-level probability
SUBSET_N_SEED = round(SUBSET_P0 * SUBSET_N)  # 100 -- kept as next level's chain starts
SUBSET_MCMC_ROUNDS = (SUBSET_N - SUBSET_N_SEED) // SUBSET_N_SEED  # 4 new samples per chain
SUBSET_LEVELS_MAX = 3  # raw level 0 + up to 2 conditional levels, within the ~1500-forward budget
SUBSET_SIGMA_PROP = 0.5  # pCN-crawl step size: z' = sqrt(1-p^2) z + p xi, exactly N(0,I)-invariant

FLIP_SCREEN_BUDGET = 300  # small attack budget used only to decide "is this case measurable"

# --- the exact whitened latent --------------------------------------------------------------
_BLUR_W, _BLUR_RADIUS = _gauss_kernel(CORR_LEN, CELL)
_RENORM = 1.0 / (_BLUR_W**2).sum()  # matches NoiseDraws.renorm exactly (same kernel, same axes)


def _edge_blur_matrix(n: int) -> np.ndarray:
    """Dense (n, n) matrix form of certify.py's `_blur` for ONE axis (edge-clamped separable
    Gaussian). `_blur` pads by `radius` with edge values then runs a 'valid' 1-D convolution,
    which is exactly a linear map row -> row' with row'_i = sum_k w[k+r] * row[clip(i+k, 0, n-1)]
    -- built once here so a WHOLE BATCH of latent fields (subset-sim, attack_is: hundreds of
    them) can be blurred by one matrix multiply instead of one np.convolve call per field.
    """
    w, r = _BLUR_W, _BLUR_RADIUS
    b = np.zeros((n, n))
    for i in range(n):
        for k in range(-r, r + 1):
            j = min(max(i + k, 0), n - 1)
            b[i, j] += w[k + r]
    return b


def _blur_batch(z: np.ndarray, by: np.ndarray, bx: np.ndarray) -> np.ndarray:
    """blur2D for a batch [D, ny, nx] of latent fields: x-axis first, then y-axis, matching
    `_blur`'s own order exactly (verified against it at module load, see `_self_check`)."""
    out = np.einsum("dyx,ux->dyu", z, bx)
    out = np.einsum("vy,dyx->dvx", by, out)
    return out


def _self_check() -> None:
    """`_blur_batch` must reproduce certify.py's own `_blur` bit-close on a single slice --
    the entire IS/subset latent accounting depends on this matrix form being the SAME linear
    operator, not merely 'a similar blur'."""
    rng = np.random.default_rng(0)
    n = 37  # deliberately not the study's 90, so a transposed or mis-axed matrix would show up
    field = rng.standard_normal((n, n))
    by = bx = _edge_blur_matrix(n)
    got = _blur_batch(field[None], by, bx)[0]
    want = _blur(field)
    err = np.abs(got - want).max()
    assert err < 1e-9, f"_blur_batch disagrees with certify._blur: max err {err:.2e}"


_self_check()


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


# --- K=4 harness primitives (adapted from bundled.py / ranking.py, generic in batch size) ---
def _evaluate4(h: Harness, elev2d: np.ndarray) -> np.ndarray:
    """Cost of every plan [K] on one terrain -- ranking.py's `_evaluate`, but tiling by
    `h.batch_size` (= K = 4 here) instead of the module's fixed N_PLANS = 16."""
    stack = np.ascontiguousarray(np.tile(elev2d, (h.batch_size, 1, 1)), np.float32)
    with wp.ScopedDevice(h.device):
        h.sim.set_terrain(wp.array(stack, dtype=wp.float32))
    return _cost(h.forward(dilate=True)).copy()


def _weighted_adjoint4(h: Harness, elev2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """d(cost)/d(elevation) [K, ny, nx] and costs [K] in one backward pass -- bundled.py's
    `_weighted_adjoint`, tiling by `h.batch_size` instead of the fixed N_PLANS."""
    stack = np.ascontiguousarray(np.tile(elev2d, (h.batch_size, 1, 1)), np.float32)
    with wp.ScopedDevice(h.device):
        h.sim.set_terrain(wp.array(stack, dtype=wp.float32))
        wp.copy(h.sim.friction, h._fric0)
    tape = wp.Tape()
    h._rollout(dilate=True, tape=tape)
    costs = _cost(h.terms.numpy()).copy()
    seed_np = np.zeros((N_TERMS, h.batch_size), np.float32)
    for name, weight in COST_TERMS.items():
        seed_np[TERM_NAMES.index(name)] = weight
    seed = wp.zeros_like(h.terms)
    seed.assign(seed_np)
    tape.backward(grads={h.terms: seed})
    grad = h.sim.elevation.grad.numpy().copy()
    tape.zero()
    return grad, costs


def _gen_restart_field(
    rng: np.random.Generator, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """One restart's pre-blur white sample AND its correlated field -- bundled.py's
    `_correlated_field` only returns the latter, but the attack_is latent needs the FORMER too
    (the raw iid sample IS the whitened-space coordinate a restart lives at), so this is
    reimplemented rather than imported."""
    f_raw = rng.standard_normal(shape)
    field = _RENORM * _blur(f_raw)
    return f_raw, field


def _bisect_flip4(
    h4: Harness, h_lo: np.ndarray, h_hi: np.ndarray, a: int, k_lo: float, k_hi: float
) -> tuple[float, float, int]:
    """certify.py's `_bisect_flip`, on the K=4 harness, ALSO returning the converged fraction
    `hi` -- needed to scale the last step's contribution to the exact latent `z_star`."""
    lo, hi = 0.0, 1.0
    n_forward = 0
    for _ in range(BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        h_mid = (h_lo + mid * (h_hi - h_lo)).astype(np.float32)
        c_mid = _evaluate4(h4, h_mid)
        n_forward += 1
        if int(np.argmin(c_mid)) != a:
            hi = mid
        else:
            lo = mid
    return k_lo + hi * (k_hi - k_lo), hi, n_forward


def _censored4(
    k_acc: float,
    z_vec: np.ndarray,
    h_t: np.ndarray,
    n_forward: int,
    n_backward: int,
    stalled: bool,
) -> dict:
    return {
        "k": K_CAP_RARE,
        "flipped": False,
        "stalled": stalled,
        "z": z_vec,
        "h_boundary": h_t,
        "n_forward": n_forward,
        "n_backward": n_backward,
    }


def _run_attack4(
    h4: Harness,
    h0: np.ndarray,
    k0: float,
    a: int,
    sigma: np.ndarray,
    temp: float,
    grad_h0: np.ndarray,
    costs_h0: np.ndarray,
    z0: np.ndarray,
    forward_budget: int,
) -> dict:
    """certify.py's `_run_attack`, on the K=4 harness, additionally accumulating the exact
    whitened latent `z_vec` over every accepted step (see module docstring) and stopping early
    once `forward_budget` forward passes have been spent (the top-level brief's ~1000-forward
    cap on the guided-descent phase, budget-capped so `attack_is`'s IS draws still fit)."""
    h_t, k_acc, z_vec = h0, k0, z0.copy()
    grad_t, costs_t = grad_h0, costs_h0
    n_forward = n_backward = 0
    for _ in range(MAX_STEPS):
        if k_acc >= K_CAP_RARE or n_forward >= forward_budget:
            break
        g = _softmin_direction(costs_t, grad_t, a, temp)
        blurred = _blur(sigma * g)
        norm = float(np.linalg.norm(blurred))
        if norm < 1e-12:
            return _censored4(k_acc, z_vec, h_t, n_forward, n_backward, stalled=True)
        u = blurred / norm
        step_field = sigma * _blur(u)
        margin_t = _bracket_gap(costs_t, a)
        eta = ETA0
        accepted = False
        cand = costs_cand = margin_cand = sign = None
        while eta >= ETA_FLOOR and n_forward < forward_budget:
            eta_use = min(eta, K_CAP_RARE - k_acc)
            cand_minus = (h_t - eta_use * step_field).astype(np.float32)
            cand_plus = (h_t + eta_use * step_field).astype(np.float32)
            c_minus, c_plus = _evaluate4(h4, cand_minus), _evaluate4(h4, cand_plus)
            n_forward += 2
            m_minus, m_plus = _bracket_gap(c_minus, a), _bracket_gap(c_plus, a)
            if m_minus <= m_plus:
                cand, costs_cand, margin_cand, sign = cand_minus, c_minus, m_minus, -1.0
            else:
                cand, costs_cand, margin_cand, sign = cand_plus, c_plus, m_plus, 1.0
            if margin_cand < margin_t:
                accepted = True
                break
            eta /= 2.0
        if not accepted:
            return _censored4(k_acc, z_vec, h_t, n_forward, n_backward, stalled=True)
        if margin_cand < 0.0:
            k_tight, frac, nf = _bisect_flip4(h4, h_t, cand, a, k_acc, k_acc + eta_use)
            n_forward += nf
            z_final = z_vec + frac * sign * (eta_use / _RENORM) * u
            h_boundary = (h_t + frac * sign * eta_use * step_field).astype(np.float32)
            return {
                "k": min(k_tight, K_CAP_RARE),
                "flipped": True,
                "stalled": False,
                "z": z_final,
                "h_boundary": h_boundary,
                "n_forward": n_forward,
                "n_backward": n_backward,
            }
        z_vec = z_vec + sign * (eta_use / _RENORM) * u
        h_t, k_acc = cand, k_acc + eta_use
        grad_t, costs_t = _weighted_adjoint4(h4, h_t)
        n_backward += 1
    return _censored4(k_acc, z_vec, h_t, n_forward, n_backward, stalled=False)


def _attack_with_restarts4(
    h4: Harness,
    belief: np.ndarray,
    sigma: np.ndarray,
    a: int,
    grad0: np.ndarray,
    costs0: np.ndarray,
    rng: np.random.Generator,
    forward_budget: int,
) -> dict:
    """certify.py's `_attack_with_restarts` on the K=4 harness, budget-capped and latent-tracking
    (see `_run_attack4`)."""
    rivals_cost_spread = float(np.ptp(np.delete(costs0, a)))
    temp = max(SOFTMIN_TEMP_FRAC * rivals_cost_spread, 1e-3)
    zero_z = np.zeros_like(belief)

    results: list[dict] = []
    starts = [(belief, 0.0, grad0, costs0, zero_z)]
    n_forward_used = 0
    for _ in range(N_RESTARTS):
        if n_forward_used >= forward_budget:
            break
        f_raw, field = _gen_restart_field(rng, belief.shape)
        h0 = (belief + RESTART_SIGMA * sigma * field).astype(np.float32)
        grad_h0, costs_h0 = _weighted_adjoint4(h4, h0)
        n_backward_restart = 1
        if _bracket_gap(costs_h0, a) < 0.0:  # already flipped by the random start alone
            k_tight, frac, nf = _bisect_flip4(h4, belief, h0, a, 0.0, RESTART_SIGMA)
            n_forward_used += nf
            results.append(
                {
                    "k": min(k_tight, K_CAP_RARE),
                    "flipped": True,
                    "stalled": False,
                    "z": frac * RESTART_SIGMA * f_raw,
                    "h_boundary": (belief + frac * RESTART_SIGMA * sigma * field).astype(
                        np.float32
                    ),
                    "n_forward": nf,
                    "n_backward": n_backward_restart,
                }
            )
            continue
        starts.append((h0, RESTART_SIGMA, grad_h0, costs_h0, RESTART_SIGMA * f_raw))
        results.append(
            {
                "k": K_CAP_RARE,
                "flipped": False,
                "stalled": False,
                "z": RESTART_SIGMA * f_raw,
                "h_boundary": h0,
                "n_forward": 0,
                "n_backward": n_backward_restart,
            }
        )

    run_results: list[dict] = []  # one final outcome per start (excludes the pre-run placeholders
    # added above), so the aggregate stall check below isn't fooled by a placeholder's k=K_CAP_RARE
    # tying with -- and, by min()'s stability, masking -- a genuinely stalled real run
    for h0, k0, grad_h0, costs_h0, z0 in starts:
        remaining = max(forward_budget - n_forward_used, 1)
        r = _run_attack4(h4, h0, k0, a, sigma, temp, grad_h0, costs_h0, z0, remaining)
        n_forward_used += r["n_forward"]
        results.append(r)
        run_results.append(r)

    best = min(results, key=lambda r: r["k"])
    # a single restart's degenerate gradient (norm < 1e-12) says nothing about the others -- the
    # whole attack only counts as a stall (its k is a budget artifact, not a boundary estimate)
    # if EVERY restart's run stalled and none flipped; a run that merely exhausted its forward
    # budget while still making progress is certify.py's own right-censored-at-cap convention.
    stalled_overall = (not best["flipped"]) and all(r["stalled"] for r in run_results)
    return {
        "k": best["k"],
        "flipped": best["flipped"],
        "stalled": stalled_overall,
        "z": best["z"],
        "h_boundary": best["h_boundary"],
        "n_forward": sum(r["n_forward"] for r in results),
        "n_backward": sum(r["n_backward"] for r in results),
    }


# --- per-seed case construction --------------------------------------------------------------
@dataclass
class SeedState:
    seed: int
    scene: object
    sigma_base: np.ndarray
    belief: np.ndarray
    poses4: np.ndarray
    omega4: np.ndarray
    a: int
    grid: tuple[np.ndarray, np.ndarray]


def build_seed_state(seed: int) -> SeedState:
    """The K=4 pruned case for one seed: build the full 16-corridor fan, rank by BELIEF cost,
    keep {1st, 5th, 9th, 13th} (PLAN_GROUPS["fan"] is 16 singleton groups, not 4 groups of 4 --
    see module docstring), and identify the belief-winner `a` among the pruned four."""
    scene, _truth, _measured, _observed, sigma, poses, omega, grid = build_case(seed, FAMILY, NOISE)
    belief = scene.elevation.astype(np.float32)

    h16 = Harness(scene, poses, omega, device="cuda")
    cost16 = _cost(h16.forward(dilate=True))
    del h16
    order = np.argsort(cost16)
    sel = order[list(SELECT_RANKS)]
    poses4 = poses[sel]
    omega4 = omega[:, sel, :]

    h4 = Harness(scene, poses4, omega4, device="cuda")
    costs4 = _evaluate4(h4, belief)
    a = int(np.argmin(costs4))
    del h4

    return SeedState(seed, scene, sigma, belief, poses4, omega4, a, grid)


def make_draw_harness(state: SeedState, batch: int) -> Harness:
    """A K=4-plan harness with `batch` DRAWS on the batch axis instead of plans -- the
    risk.py/certify.py ground-truth pattern: all K plans share start pose (0,0,0), so one
    harness per DRAW-batch size serves every plan via a per-plan omega reassignment + forward."""
    poses_d = np.tile(state.poses4[0], (batch, 1)).astype(np.float32)
    omega_d = np.zeros((state.omega4.shape[0], batch, 3), np.float32)
    return Harness(state.scene, poses_d, omega_d, device="cuda")


def _mc_costs(
    hd: Harness, draws: NoiseDraws, base_dev, sigma_dev, seed_int: int, omega4: np.ndarray
) -> np.ndarray:
    """[D, K] costs from D correlated draws (COMMON RANDOM NUMBERS across the K plans, exactly
    certify.py's ground-truth loop) -- `gain` is always 1.0 here since it is baked into `sigma`
    (`sigma_case = gain * sigma_base`, see module docstring)."""
    d = base_dev.shape[0]
    k = omega4.shape[1]
    samples = np.empty((d, k), np.float32)
    for p in range(k):
        hd.sim.target_wheel_omega.assign(
            np.ascontiguousarray(np.repeat(omega4[:, p : p + 1, :], d, axis=1), np.float32)
        )
        draws.perturb(base_dev, sigma_dev, 1.0, hd.sim.elevation, seed_int)
        samples[:, p] = _cost(hd.forward(dilate=True))
    return samples


def _costs_for_elev_batch(hd: Harness, omega4: np.ndarray, elev_batch: np.ndarray) -> np.ndarray:
    """[D, K] costs for a batch of D CALLER-SUPPLIED terrains (subset-sim's MCMC candidates,
    attack_is's mean-shifted draws) -- same pattern as `_mc_costs` but the terrain is set once,
    not drawn by `NoiseDraws`, since neither sampler is the zero-mean correlated model."""
    d = elev_batch.shape[0]
    k = omega4.shape[1]
    with wp.ScopedDevice(hd.device):
        hd.sim.set_terrain(wp.array(np.ascontiguousarray(elev_batch, np.float32)))
        wp.copy(hd.sim.friction, hd._fric0)
    samples = np.empty((d, k), np.float32)
    for p in range(k):
        hd.sim.target_wheel_omega.assign(
            np.ascontiguousarray(np.repeat(omega4[:, p : p + 1, :], d, axis=1), np.float32)
        )
        samples[:, p] = _cost(hd.forward(dilate=True))
    return samples


# --- screening: quick MC + existence check, to pick ~20 (seed, gain) cases ------------------
def screen_seed(state: SeedState) -> list[dict]:
    """Quick (1024-draw) P(flip) for every gain at this seed, plus -- for any gain that shows
    ZERO flips -- a small existence-check attack (does k <= 4 reach a flip at all)."""
    rows = []
    hd = make_draw_harness(state, DRAW_BATCH)
    draws = NoiseDraws((DRAW_BATCH, *state.belief.shape), CELL, CORR_LEN, hd.device)
    with wp.ScopedDevice(hd.device):
        base_np = np.ascontiguousarray(np.tile(state.belief, (DRAW_BATCH, 1, 1)), np.float32)
        base_dev = wp.array(base_np)
    for gi, gain in enumerate(GAINS):
        sigma_case = (gain * state.sigma_base).astype(np.float32)
        with wp.ScopedDevice(hd.device):
            sigma_dev = wp.array(sigma_case, dtype=wp.float32)
        seed_int = 1_000_000 + state.seed * 10 + gi
        samples = _mc_costs(hd, draws, base_dev, sigma_dev, seed_int, state.omega4)
        winners = np.argmin(samples, axis=1)
        n_flips = int((winners != state.a).sum())
        rows.append(
            {"seed": state.seed, "gain": gain, "p_quick": n_flips / DRAW_BATCH, "n_flips": n_flips}
        )
    del hd, draws, base_dev

    for row in rows:
        if row["n_flips"] > 0:
            continue
        h4 = Harness(state.scene, state.poses4, state.omega4, device="cuda")
        sigma_case = (row["gain"] * state.sigma_base).astype(np.float32)
        grad0, costs0 = _weighted_adjoint4(h4, state.belief)
        rng = np.random.default_rng(2_000_000 + state.seed * 10 + GAINS.index(row["gain"]))
        attack = _attack_with_restarts4(
            h4, state.belief, sigma_case, state.a, grad0, costs0, rng, FLIP_SCREEN_BUDGET
        )
        row["screen_attack_flipped"] = bool(attack["flipped"])
        row["screen_attack_k"] = float(attack["k"])
        del h4
    return rows


def select_cases(screen_rows: list[dict]) -> list[dict]:
    """Discard P > 0.3 (not rare) and P == 0-with-no-attack-flip-within-k=4 (too rare to
    reference within budget), then keep ~SELECT_TARGET_N spanning log10 P as evenly as
    the surviving candidates allow."""
    eligible = []
    for row in screen_rows:
        if row["p_quick"] > P_DISCARD_HI:
            continue
        if row["n_flips"] == 0 and not row.get("screen_attack_flipped", False):
            continue
        if row["p_quick"] > 0:
            proxy_logp = np.log10(row["p_quick"])
        else:  # zero-flip quick MC, but the attack found a boundary: use Phi(-k) as a proxy
            proxy_logp = np.log10(max(_norm_cdf(-row["screen_attack_k"]), 1e-12))
        eligible.append({**row, "proxy_logp": float(proxy_logp)})

    if len(eligible) <= SELECT_TARGET_N:
        return eligible
    eligible.sort(key=lambda r: r["proxy_logp"])
    targets = np.linspace(LOGP_LO, LOGP_HI, SELECT_TARGET_N)
    proxies = np.array([r["proxy_logp"] for r in eligible])
    chosen_idx = sorted({int(np.argmin(np.abs(proxies - t))) for t in targets})
    return [eligible[i] for i in chosen_idx]


# --- reference truth: adaptive brute-force MC -----------------------------------------------
def reference_truth(state: SeedState, sigma_case: np.ndarray, case_id: int) -> dict:
    hd = make_draw_harness(state, DRAW_BATCH)
    draws = NoiseDraws((DRAW_BATCH, *state.belief.shape), CELL, CORR_LEN, hd.device)
    with wp.ScopedDevice(hd.device):
        base_np = np.ascontiguousarray(np.tile(state.belief, (DRAW_BATCH, 1, 1)), np.float32)
        base_dev = wp.array(base_np)
        sigma_dev = wp.array(sigma_case, dtype=wp.float32)
    flips = draws_used = 0
    batch_idx = 0
    while flips < REF_MIN_FLIPS and draws_used < REF_MAX_DRAWS:
        seed_int = 9_000_000 + case_id * 1000 + batch_idx
        samples = _mc_costs(hd, draws, base_dev, sigma_dev, seed_int, state.omega4)
        winners = np.argmin(samples, axis=1)
        flips += int((winners != state.a).sum())
        draws_used += DRAW_BATCH
        batch_idx += 1
    del hd, draws, base_dev
    return {
        "p_ref": flips / draws_used,
        "flips": flips,
        "draws": draws_used,
        "uncertain": flips < 10,
    }


# --- method 1: naive Monte-Carlo --------------------------------------------------------------
def method_mc_naive(state: SeedState, sigma_case: np.ndarray, case_id: int) -> dict:
    t0 = time.time()
    hd = make_draw_harness(state, MC_NAIVE_DRAWS)
    draws = NoiseDraws((MC_NAIVE_DRAWS, *state.belief.shape), CELL, CORR_LEN, hd.device)
    with wp.ScopedDevice(hd.device):
        base_np = np.ascontiguousarray(np.tile(state.belief, (MC_NAIVE_DRAWS, 1, 1)), np.float32)
        base_dev = wp.array(base_np)
        sigma_dev = wp.array(sigma_case, dtype=wp.float32)
    samples = _mc_costs(hd, draws, base_dev, sigma_dev, 5_000_000 + case_id, state.omega4)
    del hd, draws, base_dev
    winners = np.argmin(samples, axis=1)
    n_flips = int((winners != state.a).sum())
    return {
        "p_hat": n_flips / MC_NAIVE_DRAWS,
        "n_forward": MC_NAIVE_DRAWS,
        "wall_s": time.time() - t0,
    }


# --- methods 2+3: the shared attack, then FORM and attack-seeded IS -------------------------
def method_form_and_is(state: SeedState, sigma_case: np.ndarray, case_id: int) -> dict:
    t0 = time.time()
    h4 = Harness(state.scene, state.poses4, state.omega4, device="cuda")
    grad0, costs0 = _weighted_adjoint4(h4, state.belief)
    rng = np.random.default_rng(5_000_000 + case_id * 7 + 1)
    attack = _attack_with_restarts4(
        h4, state.belief, sigma_case, state.a, grad0, costs0, rng, FORM_FORWARD_BUDGET
    )
    del h4
    t_attack = time.time() - t0

    form = {
        "p_hat": _norm_cdf(-attack["k"]),
        "k_star": attack["k"],
        "n_forward": attack["n_forward"],
        "wall_s": t_attack,
        "stalled": attack["stalled"],  # see BUG 1 fix: k=K_CAP_RARE from a stall is a budget
        # artifact, not a boundary estimate -- analyse() must censor it, not score it as finite
    }

    if not attack["flipped"]:
        # no boundary found within budget: attack_is has no mean-shift point to draw around.
        attack_is = {
            "p_hat": 0.0,
            "n_forward": attack["n_forward"],
            "wall_s": t_attack,
            "censored_no_boundary": True,
        }
        return {"form": form, "attack_is": attack_is}

    z_star = attack["z"]
    ny, nx = state.belief.shape
    by = bx = _edge_blur_matrix(ny) if ny == nx else None
    if by is None:  # ny != nx would need separate matrices; not the case for this study's grid
        by, bx = _edge_blur_matrix(ny), _edge_blur_matrix(nx)

    t1 = time.time()
    rng_is = np.random.default_rng(5_000_000 + case_id * 7 + 2)
    xi = rng_is.standard_normal((IS_DRAWS, ny, nx))
    z_i = z_star[None] + xi
    elev_batch = (
        state.belief[None] + sigma_case[None] * _RENORM * _blur_batch(z_i, by, bx)
    ).astype(np.float32)
    hd = make_draw_harness(state, IS_DRAWS)
    samples = _costs_for_elev_batch(hd, state.omega4, elev_batch)
    del hd
    winners = np.argmin(samples, axis=1)
    flip = winners != state.a
    z_norm2 = float(np.sum(z_star**2))
    zi_dot = np.einsum("dij,ij->d", z_i, z_star)
    # exact IS weight: N(z; 0, I) / N(z; z_star, I) = exp(0.5||z_star||^2 - z . z_star) -- see
    # module docstring for the derivation and why `z_i` is the model's OWN whitened latent.
    w = np.exp(0.5 * z_norm2 - zi_dot)
    p_hat_is = float(np.mean(w * flip))
    attack_is = {
        "p_hat": p_hat_is,
        "n_forward": attack["n_forward"] + IS_DRAWS,
        "wall_s": t_attack + (time.time() - t1),
        "ess": float((w.sum() ** 2) / max((w**2).sum(), 1e-12)),  # effective sample size
        "mean_weight": float(w.mean()),
    }
    return {"form": form, "attack_is": attack_is}


# --- method 4: subset simulation (derivative-free) -------------------------------------------
def method_subset(state: SeedState, sigma_case: np.ndarray, case_id: int) -> dict:
    t0 = time.time()
    ny, nx = state.belief.shape
    by = bx = _edge_blur_matrix(ny)
    rng = np.random.default_rng(6_000_000 + case_id)
    n_forward = 0

    def _margins(z_batch: np.ndarray, hd_local: Harness) -> np.ndarray:
        nonlocal n_forward
        elev_batch = (
            state.belief[None] + sigma_case[None] * _RENORM * _blur_batch(z_batch, by, bx)
        ).astype(np.float32)
        samples = _costs_for_elev_batch(hd_local, state.omega4, elev_batch)
        n_forward += z_batch.shape[0]
        return np.array([_bracket_gap(samples[i], state.a) for i in range(z_batch.shape[0])])

    hd0 = make_draw_harness(state, SUBSET_N)
    z_pop = rng.standard_normal((SUBSET_N, ny, nx))
    g_pop = _margins(z_pop, hd0)
    del hd0

    hd_mcmc = make_draw_harness(state, SUBSET_N_SEED)
    n_levels_established = 0
    t_established = float(np.quantile(g_pop, SUBSET_P0))
    while t_established > 0.0 and n_levels_established < SUBSET_LEVELS_MAX - 1:
        order = np.argsort(g_pop)[:SUBSET_N_SEED]  # the p0-quantile-closest seeds
        z_seed, g_seed = z_pop[order].copy(), g_pop[order].copy()
        z_chain, g_chain = z_seed.copy(), g_seed.copy()
        new_z, new_g = [z_seed], [g_seed]
        for _ in range(SUBSET_MCMC_ROUNDS):
            rho = np.sqrt(1.0 - SUBSET_SIGMA_PROP**2)
            xi = rng.standard_normal(z_chain.shape)
            z_prop = rho * z_chain + SUBSET_SIGMA_PROP * xi  # N(0,I)-invariant crawl proposal
            g_prop = _margins(z_prop, hd_mcmc)
            accept = g_prop < t_established  # stay within the conditioning failure region
            z_chain = np.where(accept[:, None, None], z_prop, z_chain)
            g_chain = np.where(accept, g_prop, g_chain)
            new_z.append(z_chain.copy())
            new_g.append(g_chain.copy())
        z_pop, g_pop = np.concatenate(new_z, axis=0), np.concatenate(new_g, axis=0)
        n_levels_established += 1
        t_next = float(np.quantile(g_pop, SUBSET_P0))
        if t_next <= 0.0:
            t_established = t_next
            break
        t_established = t_next
    del hd_mcmc

    p_final = float((g_pop < 0.0).mean())
    p_hat = (SUBSET_P0**n_levels_established) * p_final
    return {
        "p_hat": p_hat,
        "n_levels": n_levels_established,
        "p_final": p_final,
        "n_forward": n_forward,
        "wall_s": time.time() - t0,
    }


# --- per-case pipeline -----------------------------------------------------------------------
def run_case(state: SeedState, gain: float, case_id: int) -> dict:
    sigma_case = (gain * state.sigma_base).astype(np.float32)
    ref = reference_truth(state, sigma_case, case_id)
    mc_naive = method_mc_naive(state, sigma_case, case_id)
    fi = method_form_and_is(state, sigma_case, case_id)
    subset = method_subset(state, sigma_case, case_id)
    return {
        "seed": state.seed,
        "gain": gain,
        "a": state.a,
        "reference": ref,
        "mc_naive": mc_naive,
        "form": fi["form"],
        "attack_is": fi["attack_is"],
        "subset": subset,
    }


# --- analysis / reporting ---------------------------------------------------------------------
METHODS = ("mc_naive", "form", "attack_is", "subset")


def _err(p_hat: float, p_ref: float, stalled: bool) -> float:
    if stalled or p_hat <= 0.0:
        # a stalled attack's k is a budget artifact, not a boundary estimate (see BUG 1 in the
        # module's fix history) -- censor it exactly like the existing p_hat<=0 (no-boundary) path
        return float("inf")
    return abs(np.log10(p_hat) - np.log10(p_ref))


def analyse(rows: list[dict]) -> dict:
    valid = [r for r in rows if not r["reference"]["uncertain"]]
    p_ref = np.array([r["reference"]["p_ref"] for r in valid])
    out: dict = {"n_cases": len(rows), "n_valid": len(valid)}
    for method in METHODS:
        p_hat = np.array([r[method]["p_hat"] for r in valid])
        # `.get` tolerates records written before this flag existed (see BUG 1 fix note above)
        stalled = [bool(r[method].get("stalled", False)) for r in valid]
        errs = np.array([_err(h, t, s) for h, t, s in zip(p_hat, p_ref, stalled)])
        finite = errs[np.isfinite(errs)]
        lo_mask = p_ref < 1e-2
        hi_mask = ~lo_mask
        out[method] = {
            "median_err": float(np.median(finite)) if len(finite) else float("inf"),
            "max_err": float(np.max(finite)) if len(finite) else float("inf"),
            "median_err_lo": float(np.median(errs[lo_mask][np.isfinite(errs[lo_mask])]))
            if lo_mask.any() and np.isfinite(errs[lo_mask]).any()
            else float("inf"),
            "median_err_hi": float(np.median(errs[hi_mask][np.isfinite(errs[hi_mask])]))
            if hi_mask.any() and np.isfinite(errs[hi_mask]).any()
            else float("inf"),
            "frac_within_0.5dex": float(np.mean(errs <= 0.5)) if len(errs) else float("nan"),
            "n_censored": int(np.sum(~np.isfinite(errs))),
            "kendall_tau": kendall_tau(p_hat, p_ref) if len(p_hat) > 1 else float("nan"),
            "mean_forward": float(np.mean([r[method]["n_forward"] for r in valid]))
            if valid
            else float("nan"),
            "mean_wall_s": float(np.mean([r[method]["wall_s"] for r in valid]))
            if valid
            else float("nan"),
        }
    return out


def report(rows: list[dict], analysis: dict) -> None:
    print(
        f"\n=== rare.py: n_cases={analysis['n_cases']}  "
        f"n_valid_reference={analysis['n_valid']} ===\n"
    )
    print(
        f"{'seed':>5}{'gain':>6}{'p_ref':>10}{'mc_naive':>10}{'form':>10}"
        f"{'attack_is':>10}{'subset':>10}"
    )
    for r in rows:
        tag = "*" if r["reference"]["uncertain"] else " "
        print(
            f"{r['seed']:>5}{r['gain']:>6.2f}{r['reference']['p_ref']:>9.2e}{tag}"
            f"{r['mc_naive']['p_hat']:>10.2e}{r['form']['p_hat']:>10.2e}"
            f"{r['attack_is']['p_hat']:>10.2e}{r['subset']['p_hat']:>10.2e}"
        )
    print("  (* = reference uncertain, <10 flips at the 2e5-draw cap -- excluded from scoring)\n")

    print(f"{'method':<12}{'med err':>9}{'max err':>9}{'med err<1e-2':>14}{'within 0.5dex':>15}"
          f"{'tau':>7}{'mean fwd':>10}{'mean s':>8}")
    for m in METHODS:
        a = analysis[m]
        print(
            f"{m:<12}{a['median_err']:>9.3f}{a['max_err']:>9.3f}{a['median_err_lo']:>14.3f}"
            f"{a['frac_within_0.5dex']:>15.0%}{a['kendall_tau']:>+7.3f}"
            f"{a['mean_forward']:>10.0f}{a['mean_wall_s']:>8.2f}"
        )

    lo = {m: analysis[m]["median_err_lo"] for m in METHODS}
    best_attack = min(lo["form"], lo["attack_is"])
    best_attack_name = "form" if lo["form"] <= lo["attack_is"] else "attack_is"
    print(
        f"\nSUCCESS CRITERION: best-of(form, attack_is) median err <= 0.5 dex in the P<1e-2 "
        f"regime, AND matches/beats subset there.\n"
        f"  best attack method = {best_attack_name}, median err (P<1e-2) = {best_attack:.3f}\n"
        f"  subset,             median err (P<1e-2) = {lo['subset']:.3f}\n"
        f"  mc_naive,            median err (P<1e-2) = {lo['mc_naive']:.3f}\n"
    )
    criterion_a = best_attack <= 0.5
    criterion_b = best_attack <= lo["subset"]
    if criterion_a and criterion_b:
        print("VERDICT: the gradient-guided family's rare-event niche HOLDS.")
    else:
        print("VERDICT: sampling wins again -- the gradient family's last niche closes.")


def main() -> None:
    wp.init()
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("--- screening: quick (1024-draw) MC across 10 seeds x 3 gains ---", flush=True)
    screen_rows: list[dict] = []
    for seed in SEEDS:
        state = build_seed_state(seed)
        screen_rows.extend(screen_seed(state))
        summary = ", ".join(f"g={r['gain']}:p={r['p_quick']:.4f}" for r in screen_rows[-3:])
        print(f"  seed {seed}: {summary}", flush=True)
    t_screen = time.time() - t_start
    print(f"[timing] screening took {t_screen:.1f}s", flush=True)

    selected = select_cases(screen_rows)
    print(f"\nselected {len(selected)} cases:", flush=True)
    for r in selected:
        print(f"  seed={r['seed']} gain={r['gain']} proxy_logP={r['proxy_logp']:.2f}", flush=True)

    rows: list[dict] = []
    seed_states = {s: None for s in {r["seed"] for r in selected}}
    t_ref_start = time.time()
    for i, case in enumerate(selected):
        if seed_states[case["seed"]] is None:
            seed_states[case["seed"]] = build_seed_state(case["seed"])
        state = seed_states[case["seed"]]
        row = run_case(state, case["gain"], case_id=i)
        rows.append(row)
        elapsed = time.time() - t_ref_start
        print(
            f"  case {i + 1}/{len(selected)} (seed={case['seed']}, gain={case['gain']}) "
            f"p_ref={row['reference']['p_ref']:.2e} ({row['reference']['draws']} draws) "
            f"[{elapsed:.0f}s elapsed]",
            flush=True,
        )
        if i == 1:
            est_total = elapsed / 2 * len(selected)
            print(
                f"[timing] 2 cases in {elapsed:.1f}s -> est. {est_total / 60:.1f} min for "
                f"{len(selected)} cases",
                flush=True,
            )
            if (time.time() - t_start) + est_total > 90 * 60 and len(selected) > 12:
                print(
                    "[timing] projected total exceeds 90 min -- truncating to 12 cases",
                    flush=True,
                )
                # in-place: `selected = selected[:12]` would rebind the name to a NEW list the
                # running `for i, case in enumerate(selected)` iterator (bound to the OLD list
                # object) can't see, so the loop would keep running past the cap -- `del` mutates
                # the same object the iterator already holds, so it actually stops at 12.
                del selected[12:]

    analysis = analyse(rows)
    report(rows, analysis)
    path = OUT / "rare.json"
    path.write_text(
        json.dumps(
            {
                "family": FAMILY,
                "noise": NOISE,
                "k": K,
                "screen_rows": screen_rows,
                "selected": selected,
                "rows": rows,
                "analysis": analysis,
                "wall_s_total": time.time() - t_start,
            },
            indent=2,
        )
    )
    print(f"\nwrote {path}")
    print(f"[timing] total wall time {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
