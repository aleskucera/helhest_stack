"""Map-error robustness certificates: does an ITERATED, forward-verified attack on the belief
map predict the true probability that sampled map errors flip a planner's decision, better
than cheap baselines?

    .venv/bin/python -m studies.bench.certify --seeds 100

THE IDEA UNDER TEST. A planner commits to the best of K candidate plans on an uncertain
elevation map. `k_star` is the MINIMAL map perturbation (in units of the map's own noise
model, sigma.py) that flips the decision -- a rival overtakes the winner. If `k_star` predicts
the true P(flip) under sampled map errors better than much cheaper baselines, it is a
principled commit/abstain certificate; if not, the idea dies.

WHY ITERATED, NOT ONE-SHOT. studies/CLAIMS.md's C2 (the validity-radius finding, reproduced
across 12/12 studies in this repo): one-shot gradient extrapolation across cm-scale map
perturbations is REFUTED -- the contact dilation's arg-max is a Danskin subgradient, exact
only within the arg-max's OWN margin (millimetres), while sigma is centimetres. A one-shot
FOSM z-score (`k_lin` below, the project's OWN earlier estimator) extrapolates that gradient
straight past its validity radius and is included here to test that it is badly calibrated,
as the project's theory predicts. `k_star` instead takes ~0.1-sigma steps, RE-SOLVES the
contact arg-max and RE-EVALUATES the cost by forward pass at every step (`Harness.forward`'s
off-tape argmax via `_evaluate`), and only trusts the gradient to pick the direction of the
NEXT small step -- exactly the regime the validity-radius finding says a derivative is good
for. The gradient only guides; forward evaluation decides, every step, including the sign.

THE ATTACK'S NOISE GEOMETRY. sigma.py's noise model draws delta_h = gain * sigma ⊙ W(white),
W a separable Gaussian blur at CORR_LEN. Among unit-norm white-noise directions, the one that
maximises a LINEAR functional g . delta_h is white* ∝ W^T(sigma ⊙ g) (W is symmetric, so the
same blur implements both W and W^T) -- this is `_attack_direction` below, reused at every
step with g recomputed at the CURRENT point. Whether stepping along it actually helps is
never assumed; both signs are tried and scored by forward evaluation, and if neither improves
the margin the step is halved -- a direct enactment of "the gradient only guides."
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from ..adjoint.harness import Harness
from ..adjoint.sigma import _gauss_kernel
from ..adjoint.sigma import fosm_variance
from ..adjoint.sigma import NoiseDraws
from .bundled import _correlated_field
from .bundled import _weighted_adjoint
from .ranking import _cost
from .ranking import _evaluate
from .ranking import build_case
from .ranking import CELL
from .ranking import kendall_tau
from .ranking import N_PLANS
from .ranking import OUT
from .risk import _footprint_sigma
from .risk import CORR_LEN
from .risk import N_DRAWS

TOP_K_RIVALS = 3  # rivals considered for k_lin ONLY (the one-shot z-score, unchanged by the
# rival-coverage fix below); the attack itself now covers ALL rivals via softmin, see below
MAX_STEPS = 60  # per-attack step budget
ETA0 = 0.1  # [sigma-units] nominal step size
ETA_FLOOR = 1e-3  # below this the attack is declared stalled
K_CAP = 3.0  # [sigma-units] "certified robust to 3 sigma" censoring point
BISECT_ITERS = 8  # refines the flip boundary to ~eta0 / 2**8 sigma-units
BRACKET_C = 1.0  # bracket predictor evaluated at belief +- BRACKET_C * sigma
FLIP_P_THRESH = 0.1  # P(flip) threshold for the binary AUC classification
N_BOOT = 1000  # bootstrap resamples for the paired-tau-difference CI

# RIVAL COVERAGE FIX. The original attack targeted only the top-3 belief-cheapest rivals, but
# the ground truth counts a flip to ANY of the other 15 -- a seed can be falsely "certified
# robust" if the cheapest path to a flip runs through rival #4-15, never attacked at all. Fixed
# by attacking the SOFTMIN margin over ALL non-winner rivals rather than looping per rival: one
# combined direction per step, softmax-weighted toward whichever rivals are CURRENTLY closest
# (which can shift mid-attack, unlike a fixed single-b target) -- chosen over "attack each of
# the 15 rivals separately" because it converges to the same exact flip check (a hard argmin
# over all K costs, still verified by forward pass every step) at roughly 1/5 the backward
# passes, since one trajectory implicitly covers every rival instead of 15 independent ones.
SOFTMIN_TEMP_FRAC = 0.05  # softmin temperature, as a fraction of the belief rival-cost spread

# MULTI-START FIX. A single-basin guided descent from the belief can get stuck in a local
# margin-minimum that never reaches a flip within budget, while a DIFFERENT starting point
# reaches one easily -- a between-basin failure, not a within-basin (curvature) one, so it is
# fixed by restarting from different points, NOT by adding second-order machinery (the
# dilation's Hessian is piecewise-zero off the contact arg-max boundary, FINDINGS.md 3.2 --
# there is no local curvature to exploit here; the failure is which basin you start in).
N_RESTARTS = 4  # random restarts IN ADDITION to the plain belief start
RESTART_SIGMA = 0.5  # [sigma-units] magnitude of each restart's initial random perturbation


# --- the attack ---------------------------------------------------------------------------
def _blur(field: np.ndarray) -> np.ndarray:
    """Apply the SAME separable Gaussian W the noise model draws with (sigma.py). The kernel
    is symmetric, so this one function implements both W and W^T in `_attack_direction`."""
    w, radius = _gauss_kernel(CORR_LEN, CELL)
    if radius == 0:
        return field.copy()
    pad = np.pad(field, radius, mode="edge")
    out = np.apply_along_axis(lambda m: np.convolve(m, w, mode="valid"), 1, pad)
    out = np.apply_along_axis(lambda m: np.convolve(m, w, mode="valid"), 0, out)
    return out


def _attack_direction(g: np.ndarray, sigma: np.ndarray) -> np.ndarray | None:
    """The noise-geometry step field sigma ⊙ W(u), u = normalise(W^T(sigma ⊙ g)) -- the
    direction that, among the noise model's own unit-white-noise directions, maximises the
    LINEAR response of margin `g` per step. Returns None where the gradient has no support at
    all (blurred field ~0), the one case the attack cannot proceed from.
    """
    blurred = _blur(sigma * g)
    norm = float(np.linalg.norm(blurred))
    if norm < 1e-12:
        return None
    u = blurred / norm
    return sigma * _blur(u)


def _bisect_flip(
    harness: Harness, h_lo: np.ndarray, h_hi: np.ndarray, a: int, k_lo: float, k_hi: float
) -> tuple[float, int]:
    """Tighten the flip point along the last accepted step by bisection. `h_lo` still has
    argmin == a; `h_hi` does not (both already forward-verified by the caller)."""
    lo, hi = 0.0, 1.0
    n_forward = 0
    for _ in range(BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        h_mid = (h_lo + mid * (h_hi - h_lo)).astype(np.float32)
        c_mid = _evaluate(harness, h_mid)
        n_forward += 1
        if int(np.argmin(c_mid)) != a:
            hi = mid
        else:
            lo = mid
    return k_lo + hi * (k_hi - k_lo), n_forward


def _softmin_direction(costs: np.ndarray, grad: np.ndarray, a: int, temp: float) -> np.ndarray:
    """g = grad[a] - softmax-weighted average of the OTHER 15 plans' gradients, weight favouring
    whichever rivals are cheapest right now. temp -> 0 recovers grad[a] - grad[argmin rival]
    exactly; temp is only used to pick a DIRECTION -- margins, acceptance and the flip test
    below are always the EXACT hard min over all K costs from a real forward pass, never the
    softmin. This is what gives one attack trajectory coverage of every rival: whichever one is
    closest gets most of the weight, and that can shift from step to step.
    """
    idx = np.array([k for k in range(len(costs)) if k != a])
    rc = costs[idx]
    w = np.exp(-(rc - rc.min()) / temp)
    w /= w.sum()
    grad_soft = np.tensordot(w, grad[idx], axes=(0, 0))
    return grad[a] - grad_soft


def _bracket_gap(costs: np.ndarray, a: int) -> float:
    """min_{k != a}(cost_k) - cost_a, i.e. the margin recomputed fresh at this map. Used as the
    EXACT (non-soft) margin for step acceptance and the flip test."""
    rest = np.delete(costs, a)
    return float(rest.min() - costs[a])


def _run_attack(
    harness: Harness,
    h0: np.ndarray,
    k0: float,
    a: int,
    sigma: np.ndarray,
    temp: float,
    grad_h0: np.ndarray,
    costs_h0: np.ndarray,
) -> dict:
    """One guided-descent trajectory starting from (h0, k0) -- k0 is the sigma-budget ALREADY
    spent getting to h0 (0 for the plain belief start, RESTART_SIGMA for a random restart), so
    every restart's reported k is an honest total cost, not just the guided-descent part.
    `grad_h0`/`costs_h0` are the adjoint already evaluated at h0 (the caller has them either
    way, so no redundant pass here).
    """
    h_t, k_acc = h0, k0
    grad_t, costs_t = grad_h0, costs_h0
    n_forward = 0
    n_backward = 0
    for _ in range(MAX_STEPS):
        if k_acc >= K_CAP:
            break
        g = _softmin_direction(costs_t, grad_t, a, temp)
        step_field = _attack_direction(g, sigma)
        if step_field is None:
            return _censored(k_acc, n_forward, n_backward, stalled=True)
        margin_t = _bracket_gap(costs_t, a)
        eta = ETA0
        accepted = False
        cand = costs_cand = margin_cand = None
        while eta >= ETA_FLOOR:
            eta_use = min(eta, K_CAP - k_acc)
            # Try both signs and let the forward evaluation pick -- the direction maximises
            # the LINEAR response, but the true forward is non-smooth (contact re-solved), so
            # which sign actually decreases the margin is not assumed, only checked.
            cand_minus = (h_t - eta_use * step_field).astype(np.float32)
            cand_plus = (h_t + eta_use * step_field).astype(np.float32)
            c_minus, c_plus = _evaluate(harness, cand_minus), _evaluate(harness, cand_plus)
            n_forward += 2
            m_minus, m_plus = _bracket_gap(c_minus, a), _bracket_gap(c_plus, a)
            if m_minus <= m_plus:
                cand, costs_cand, margin_cand = cand_minus, c_minus, m_minus
            else:
                cand, costs_cand, margin_cand = cand_plus, c_plus, m_plus
            if margin_cand < margin_t:
                accepted = True
                break
            eta /= 2.0
        if not accepted:
            return _censored(k_acc, n_forward, n_backward, stalled=True)
        if margin_cand < 0.0:  # any rival overtook a -- verified by the exact hard min above
            k_tight, nf = _bisect_flip(harness, h_t, cand, a, k_acc, k_acc + eta_use)
            return {
                "k": min(k_tight, K_CAP),
                "flipped": True,
                "stalled": False,
                "n_forward": n_forward + nf,
                "n_backward": n_backward,
            }
        h_t, k_acc = cand, k_acc + eta_use
        grad_t, costs_t = _weighted_adjoint(harness, h_t)
        n_backward += 1
    return _censored(k_acc, n_forward, n_backward, stalled=False)


def _censored(k_acc: float, n_forward: int, n_backward: int, stalled: bool) -> dict:
    return {
        "k": K_CAP,
        "flipped": False,
        "stalled": stalled,
        "n_forward": n_forward,
        "n_backward": n_backward,
    }


def _attack_with_restarts(
    harness: Harness,
    belief: np.ndarray,
    sigma: np.ndarray,
    a: int,
    grad0: np.ndarray,
    costs0: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """The plain belief start plus N_RESTARTS random ones; keeps the SMALLEST flip distance.

    A restart's h0 = belief + RESTART_SIGMA * sigma * (correlated field), which may already
    have flipped before any guided step is taken -- that is checked and, if so, bisected along
    the belief->h0 segment for a tight k, exactly like a normal accepted step.
    """
    rivals_cost_spread = float(np.ptp(np.delete(costs0, a)))
    temp = max(SOFTMIN_TEMP_FRAC * rivals_cost_spread, 1e-3)

    # Total compute cost is the SUM across every restart attempted (that is what is actually
    # paid); only the reported k/flipped come from whichever restart did best.
    results: list[dict] = []
    starts = [(belief, 0.0, grad0, costs0)]
    for _ in range(N_RESTARTS):
        field = _correlated_field(rng, belief.shape)
        h0 = (belief + RESTART_SIGMA * sigma * field).astype(np.float32)
        grad_h0, costs_h0 = _weighted_adjoint(harness, h0)
        n_backward_restart = 1
        if _bracket_gap(costs_h0, a) < 0.0:  # already flipped by the random start alone
            k_tight, nf = _bisect_flip(harness, belief, h0, a, 0.0, RESTART_SIGMA)
            results.append(
                {"k": min(k_tight, K_CAP), "flipped": True, "stalled": False,
                 "n_forward": nf, "n_backward": n_backward_restart}
            )
            continue
        starts.append((h0, RESTART_SIGMA, grad_h0, costs_h0))
        results.append(  # the initial-point adjoint call itself, charged even if the guided
            {"k": K_CAP, "flipped": False, "stalled": False,  # descent below does the rest
             "n_forward": 0, "n_backward": n_backward_restart}
        )

    for h0, k0, grad_h0, costs_h0 in starts:
        results.append(_run_attack(harness, h0, k0, a, sigma, temp, grad_h0, costs_h0))

    best = min(results, key=lambda r: r["k"])
    return {
        "k": best["k"],
        "flipped": best["flipped"],
        "stalled": best["stalled"],
        "n_forward": sum(r["n_forward"] for r in results),
        "n_backward": sum(r["n_backward"] for r in results),
    }


# --- per-seed pipeline ---------------------------------------------------------------------
def run_seed(seed: int, family: str = "hybrid", noise: str = "all") -> dict:
    scene, _truth, _measured, _observed, sigma, poses, omega, grid = build_case(
        seed, family, noise
    )
    belief = scene.elevation.astype(np.float32)

    h = Harness(scene, poses, omega, device="cuda")
    grad0, costs0 = _weighted_adjoint(h, belief)
    a = int(np.argmin(costs0))
    order = np.argsort(costs0)
    second = int(order[1])
    gap = float(costs0[second] - costs0[a])
    rivals_klin = [int(k) for k in order if k != a][:TOP_K_RIVALS]  # k_lin only, unchanged

    # --- gap_over_stepsd: cheapest scale-normalised variant, no map info beyond `step`'s own
    traj = h.sim.controlled.numpy()[:, :, :2].copy()
    sig_t = _footprint_sigma(traj, sigma, grid)
    step_risk = sig_t.sum(axis=0)
    stepsd_scale = abs(float(step_risk[second] - step_risk[a]))
    gap_over_stepsd = gap / max(stepsd_scale, 1e-6)

    # --- bracket: two extra forwards, no gradient ---------------------------------------
    j_hi = _evaluate(h, (belief + BRACKET_C * sigma).astype(np.float32))
    j_lo = _evaluate(h, (belief - BRACKET_C * sigma).astype(np.float32))
    bracket = min(_bracket_gap(j_hi, a), _bracket_gap(j_lo, a))

    # --- k_lin: one-shot FOSM z-score of the margin vs the best of the top rivals -------
    z_scores = []
    for b in rivals_klin:
        margin = float(costs0[b] - costs0[a])
        g_ab = grad0[a] - grad0[b]
        sd = float(np.sqrt(max(fosm_variance(g_ab, sigma, CELL, CORR_LEN), 0.0)))
        z_scores.append(margin / max(sd, 1e-9))
    k_lin = float(min(z_scores)) if z_scores else float("nan")

    # --- k_star: iterated, forward-verified attack -- ALL rivals via softmin, N_RESTARTS+1
    # starting points, smallest flip distance kept. See the two FIX comments above N_RESTARTS
    # and SOFTMIN_TEMP_FRAC for why (rival coverage + between-basin, not within-basin, failure).
    restart_rng = np.random.default_rng(800_000 + seed)
    attack = _attack_with_restarts(h, belief, sigma, a, grad0, costs0, restart_rng)
    k_star = float(attack["k"])
    any_flip = bool(attack["flipped"])
    n_forward = attack["n_forward"]
    n_backward = attack["n_backward"]

    # --- ground truth P(flip): 256 correlated draws, risk.py's MC pattern --------------
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
        # COMMON RANDOM NUMBERS: every plan sees the identical draw set, so flips are read off
        # a paired comparison rather than two independently noisy estimates.
        draws.perturb(base, sig_dev, 1.0, hd.sim.elevation, 900_000 + seed)
        samples[:, k] = _cost(hd.forward(dilate=True))
    del hd, h

    winners = np.argmin(samples, axis=1)
    flip_mask = winners != a
    p_flip = float(flip_mask.mean())
    regret_flip = (
        float(np.mean(samples[flip_mask, a] - samples[flip_mask, winners[flip_mask]]))
        if flip_mask.any()
        else 0.0
    )

    return {
        "seed": seed,
        "a": a,
        "gap": gap,
        "gap_over_stepsd": gap_over_stepsd,
        "bracket": bracket,
        "k_lin": k_lin,
        "k_star": k_star,
        "k_star_any_flip": any_flip,
        "k_star_stalled": bool(attack["stalled"]),
        "n_forward": n_forward,
        "n_backward": n_backward,
        "p_flip": p_flip,
        "regret_flip": regret_flip,
    }


# --- analysis -------------------------------------------------------------------------------
def _pred_tau(predictor: np.ndarray, p_flip: np.ndarray) -> float:
    """Sign-corrected Kendall tau: positive means LARGER predictor -> LOWER P(flip), which is
    the direction every predictor here is defined to have (gap, bracket, k_lin, k_star all
    read 'large = safe')."""
    return -kendall_tau(predictor, p_flip)


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    """P(score_pos > score_neg), ties at 0.5 -- Mann-Whitney AUC without a scipy/sklearn dep.
    O(n^2), fine at n ~ 100."""
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = sum((p > neg).sum() + 0.5 * (p == neg).sum() for p in pos)
    return float(wins / (len(pos) * len(neg)))


def _bootstrap_tau_diff(
    pred_a: np.ndarray, pred_b: np.ndarray, target: np.ndarray, n_boot: int = N_BOOT
) -> tuple[float, float, float]:
    """Bootstrap the seeds (not the predictors) to get a CI on tau(pred_a) - tau(pred_b)."""
    rng = np.random.default_rng(0)
    n = len(target)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = _pred_tau(pred_a[idx], target[idx]) - _pred_tau(pred_b[idx], target[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


PREDICTOR_NAMES = ("gap", "gap_over_stepsd", "bracket", "k_lin", "k_star")


def analyse(rows: list[dict], bootstrap: bool = True) -> dict:
    p_flip = np.array([r["p_flip"] for r in rows])
    preds = {name: np.array([r[name] for r in rows]) for name in PREDICTOR_NAMES}
    label = p_flip > FLIP_P_THRESH

    table = {}
    for name, v in preds.items():
        table[name] = {
            "tau": _pred_tau(v, p_flip),
            "auc": _auc(-v, label),  # larger predictor => safer => lower flip score
        }

    out: dict = {"n": len(rows), "predictor_table": table}

    if bootstrap:
        ci = {}
        for name in ("bracket", "k_lin"):
            ci[f"k_star_minus_{name}"] = _bootstrap_tau_diff(preds["k_star"], preds[name], p_flip)
        out["bootstrap_ci"] = ci
    else:
        out["raw_tau_diff"] = {
            "k_star_minus_gap": table["k_star"]["tau"] - table["gap"]["tau"],
            "k_star_minus_bracket": table["k_star"]["tau"] - table["bracket"]["tau"],
        }

    # calibration: mean P(flip) within k_star bins
    bins = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0)]
    calib = {}
    ks = preds["k_star"]
    for lo, hi in bins:
        m = (ks >= lo) & (ks < hi)
        calib[f"[{lo},{hi})"] = {
            "n": int(m.sum()),
            "mean_p_flip": float(p_flip[m].mean()) if m.any() else float("nan"),
        }
    censored = ks >= K_CAP
    calib["censored(=3.0)"] = {
        "n": int(censored.sum()),
        "mean_p_flip": float(p_flip[censored].mean()) if censored.any() else float("nan"),
    }
    out["calibration"] = calib

    # attack diagnostics -- now one combined (softmin, multi-start) attack per seed, so these
    # are per-SEED rates rather than per-rival-attack rates as in the single-rival version.
    n_seeds = len(rows)
    n_flipped = sum(r["k_star_any_flip"] for r in rows)
    n_stalled = sum(r["k_star_stalled"] for r in rows)
    out["attack_diagnostics"] = {
        "n_seed_any_flip": int(n_flipped),
        "n_seeds": n_seeds,
        "n_restarts_per_seed": N_RESTARTS + 1,
        "seed_flip_rate": n_flipped / n_seeds if n_seeds else float("nan"),
        "seed_stall_rate": n_stalled / n_seeds if n_seeds else float("nan"),
        "seed_censored_rate": float(censored.mean()),
        "mean_forwards_per_seed": float(np.mean([r["n_forward"] for r in rows])),
        "mean_backwards_per_seed": float(np.mean([r["n_backward"] for r in rows])),
    }
    return out


def report(rows: list[dict], analysis: dict) -> None:
    n = analysis["n"]
    print(f"\n=== certify.py: n={n} seeds ===")
    print(f"\nmean P(flip) = {np.mean([r['p_flip'] for r in rows]):.3f}   "
          f"mean regret|flip = {np.mean([r['regret_flip'] for r in rows]):.3f}\n")

    print(f"{'predictor':<18}{'tau vs P_flip':>16}{'AUC (P_flip>' + f'{FLIP_P_THRESH})':>18}")
    for name, v in analysis["predictor_table"].items():
        print(f"{name:<18}{v['tau']:>+16.3f}{v['auc']:>18.3f}")

    if "bootstrap_ci" in analysis:
        print("\npaired tau differences, bootstrapped 95% CI over seeds:")
        for k, (mean, lo, hi) in analysis["bootstrap_ci"].items():
            print(f"  {k:<24}{mean:>+8.3f}   CI [{lo:+.3f}, {hi:+.3f}]")
    else:
        print("\nraw (unbootstrapped) paired tau differences:")
        for k, v in analysis["raw_tau_diff"].items():
            print(f"  {k:<24}{v:>+8.3f}")

    print("\ncalibration: mean P(flip) within k_star bins")
    for k, v in analysis["calibration"].items():
        print(f"  k_star in {k:<16} n={v['n']:>4}   mean P(flip) = {v['mean_p_flip']:.3f}")

    d = analysis["attack_diagnostics"]
    print("\nattack diagnostics "
          f"({d['n_restarts_per_seed']} starts/seed: belief + {N_RESTARTS} random restarts)")
    print(f"  seeds with a flip found:      {d['n_seed_any_flip']}/{d['n_seeds']}")
    print(f"  seed censored (k_star=3.0):    {d['seed_censored_rate']:.0%}")
    print(f"  seed flip rate:                {d['seed_flip_rate']:.0%}")
    print(f"  seed stall rate:                {d['seed_stall_rate']:.0%}")
    print(f"  mean forwards/backwards per seed: {d['mean_forwards_per_seed']:.0f} / "
          f"{d['mean_backwards_per_seed']:.0f}")

    kb = analysis["predictor_table"]["k_star"]["tau"]
    tb = analysis["predictor_table"]["gap"]["tau"]
    tc = analysis["predictor_table"]["bracket"]["tau"]
    print("\nKILL CRITERION: k_star dies unless tau(k_star) - tau(gap) >= 0.10 AND "
          "tau(k_star) - tau(bracket) >= 0.10")
    print(f"  tau(k_star)={kb:+.3f}  tau(gap)={tb:+.3f}  tau(bracket)={tc:+.3f}")
    print(f"  delta vs gap = {kb - tb:+.3f}   delta vs bracket = {kb - tc:+.3f}")
    verdict = "LIVES" if (kb - tb >= 0.10 and kb - tc >= 0.10) else "DIES"
    print(f"  VERDICT: certificate {verdict}")

    kl = analysis["predictor_table"]["k_lin"]["tau"]
    print(f"\niteration-vs-one-shot: tau(k_star)={kb:+.3f} vs tau(k_lin)={kl:+.3f}   "
          f"delta={kb - kl:+.3f}  ({'iteration helps' if kb > kl else 'iteration does NOT help'})")


def main(n_seeds: int = 100, family: str = "hybrid", noise: str = "all") -> None:
    wp.init()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    t_start = time.time()
    for seed in range(n_seeds):
        rows.append(run_seed(seed, family, noise))
        if seed == 4:
            elapsed = time.time() - t_start
            est_total = elapsed / 5 * n_seeds
            print(
                f"[timing] 5 seeds in {elapsed:.1f}s -> est. {est_total / 60:.1f} min "
                f"for {n_seeds} seeds",
                flush=True,
            )
        if (seed + 1) % 5 == 0:
            print(f"  {seed + 1}/{n_seeds} seeds  ({time.time() - t_start:.0f}s elapsed)",
                  flush=True)
    bootstrap = n_seeds >= 30  # meaningless with fewer seeds than that
    analysis = analyse(rows, bootstrap=bootstrap)
    report(rows, analysis)
    path = OUT / "certify.json"
    path.write_text(
        json.dumps({"family": family, "noise": noise, "rows": rows, "analysis": analysis},
                    indent=2)
    )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--family", default="hybrid")
    ap.add_argument("--noise", default="all")
    a = ap.parse_args()
    main(a.seeds, a.family, a.noise)
