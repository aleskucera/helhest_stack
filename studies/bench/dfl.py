"""Decision-focused training of a terrain inpainter -- kill-test.

    .venv/bin/python -m studies.bench.dfl              # stage 1 gate, then stage 2
    .venv/bin/python -m studies.bench.dfl --stage1-only

Perception models for terrain (UNRealNet, MonoForce) train an inpainter with a likelihood/MSE
loss. The idea under test: train it instead with a DECISION loss -- the error in the downstream
plan choice, backpropagated THROUGH the differentiable contact simulator via `_weighted_adjoint`.
CLAIMS.md's own finding is the reason this could fail even in principle: the point adjoint's
per-cell gradient has a mm validity radius against a cm sigma, and bundled adjoints have heavy
tails (single draws at 5e10). Training is the one regime where averaging over many steps might
still make the gradient useful -- that is exactly what this module measures, not assumes.

The model is deliberately tiny and honest: one shared linear map from six hand features to a
predicted height, trained with hand-written gradients and a hand-written Adam step -- no torch,
no autodiff library on the model side. The only autodiff in this file is the simulator's own
(via `bundled._weighted_adjoint`), which is the whole point: the decision loss's gradient has to
survive one real trip through the contact solver's adjoint to reach the model weights.

Map assembly, used identically for training AND evaluation, for every one of the five methods
compared: TRUTH on observed cells, the method's own fill on unobserved cells. This isolates the
unobserved-cell-filling method from the separate (and here irrelevant) question of sensor/pose
noise on cells actually seen -- the thing this kill-test is about.

A 30-train/25-test pilot (2026-08-06) ran first to de-risk the plumbing before committing to the
full budget: stage 1 passed, and the pilot's own held-out numbers pointed the right direction
(decision/cost-space beat MSE on regret, though not significantly at n=25). The measured cost per
`_weighted_adjoint` call turned out far below the ~0.1-0.3s budgeted (~0.04s, shared GPU), so the
constants below are the FULL pre-registered scale, not the pilot's.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import warp as wp

from ..adjoint.harness import Harness
from . import noise as noise_mod
from .bundled import _weighted_adjoint
from .ranking import _evaluate
from .ranking import build_case
from .ranking import CELL
from .ranking import kendall_tau
from .ranking import N_PLANS
from .ranking import OUT
from .ranking import sign_test

FAMILY = "hybrid"  # the family that actually discriminates (ranking.py); realistic noise below
NOISE = "all"  # occlusion + sensor + localisation combined -- the paper's primary regime

TRAIN_SEEDS = tuple(range(80))
TEST_SEEDS = tuple(range(100, 180))  # 80 virgin seeds: more power than the pre-registered 50, cheap

PLANE_FIT_RADIUS_M = 1.0
MEAN_HEIGHT_RADIUS_M = 2.0
FEATURE_NAMES = ("bias", "dist_unobs", "planefit", "nearest_h", "mean_h_2m", "sigma")
N_FEATURES = len(FEATURE_NAMES)

N_STEPS = 400  # full budget: optimizer steps per condition (scope says 300-500)
MINIBATCH = 8
LR_TUNE_STEPS = 15  # short run used only to pick a learning rate, per condition
LR_CANDIDATES = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
SOFTMIN_TEMP_FRAC = 0.3  # decision loss: softmin temperature as a fraction of the true cost spread
ADAM_BETA1, ADAM_BETA2, ADAM_EPS = 0.9, 0.999, 1e-8


# --- hand features -----------------------------------------------------------------------------


def _disc_offsets(radius_cells: float) -> list[tuple[int, int]]:
    """Integer (dy, dx) offsets inside a disc of the given radius, in cells."""
    r = int(np.ceil(radius_cells))
    return [
        (dy, dx)
        for dy in range(-r, r + 1)
        for dx in range(-r, r + 1)
        if dy * dy + dx * dx <= radius_cells**2
    ]


def _plane_fit(observed: np.ndarray, belief: np.ndarray, cell: float, radius_m: float) -> np.ndarray:
    """Per-cell plane extrapolation from observed cells within `radius_m`.

    Fits z = a*dx + b*dy + c in coordinates RELATIVE to the query cell, so the plane's value AT
    the query cell is just the intercept c -- no separate evaluation step. The offsets are
    translation-invariant (the same disc at every cell), so each moment of the normal equations
    is one disc-sum computed with `np.roll`, exactly the trick `ranking._disc_pool` uses for the
    contact-cap pooling. Cells with fewer than 3 observed neighbours in range are left at 0, per
    spec ("if none in range use 0").
    """
    offsets = _disc_offsets(radius_m / cell)
    obs_f = observed.astype(np.float64)
    n = sx = sy = sxx = syy = sxy = sz = sxz = syz = np.zeros_like(belief)
    for dy, dx in offsets:
        # np.roll(a, -dy, 0)[iy] == a[iy + dy]: shift the source field so index iy reads the
        # cell (iy + dy, ix + dx) -- the offset sample relative to query cell (iy, ix).
        o = np.roll(np.roll(obs_f, -dy, axis=0), -dx, axis=1)
        h = np.roll(np.roll(belief, -dy, axis=0), -dx, axis=1)
        x, y = dx * cell, dy * cell
        n = n + o
        sx = sx + o * x
        sy = sy + o * y
        sxx = sxx + o * x * x
        syy = syy + o * y * y
        sxy = sxy + o * x * y
        sz = sz + o * h
        sxz = sxz + o * x * h
        syz = syz + o * y * h
    shape = belief.shape
    mat = np.stack(
        [np.stack([n, sx, sy], -1), np.stack([sx, sxx, sxy], -1), np.stack([sy, sxy, syy], -1)], -2
    )  # [ny, nx, 3, 3]
    rhs = np.stack([sz, sxz, syz], -1)  # [ny, nx, 3]
    enough = n >= 3.0  # underdetermined below 3 points
    mat_reg = mat + 1e-9 * np.eye(3)  # tiny ridge: guards a near-collinear window
    out = np.zeros(shape)
    # An explicit trailing size-1 dim on the RHS is required: this numpy treats (N,3),(N,3) as
    # ONE (3,3)-vs-(N,3) matrix solve rather than a BATCH of N vector solves without it.
    sol = np.linalg.solve(mat_reg[enough], rhs[enough][..., None])[..., 0]
    out[enough] = sol[:, 0]  # intercept c = plane value AT the query cell
    return out


def _mean_height_window(
    observed: np.ndarray, belief: np.ndarray, cell: float, radius_m: float
) -> np.ndarray:
    """Mean OBSERVED height within `radius_m`; 0 where nothing observed in range."""
    offsets = _disc_offsets(radius_m / cell)
    obs_f = observed.astype(np.float64)
    n = s = np.zeros_like(belief)
    for dy, dx in offsets:
        o = np.roll(np.roll(obs_f, -dy, axis=0), -dx, axis=1)
        h = np.roll(np.roll(belief, -dy, axis=0), -dx, axis=1)
        n, s = n + o, s + o * h
    out = np.zeros_like(belief)
    have = n > 0
    out[have] = s[have] / n[have]
    return out


def _nearest_observed_height(observed: np.ndarray, belief: np.ndarray, cell: float) -> np.ndarray:
    """Height of the nearest observed cell, by exact brute-force Euclidean nearest-neighbour.

    The grid is small (<=~8k cells) and this runs once per seed (cached) -- cheaper to write
    correctly this way than a second Felzenszwalb-style distance-transform pass with index
    tracking (no scipy in this venv; see noise.py's own comment on the same constraint).
    """
    obs_idx = np.argwhere(observed)
    out = np.zeros_like(belief)
    if obs_idx.size == 0:
        return out
    obs_h = belief[obs_idx[:, 0], obs_idx[:, 1]]
    all_idx = np.argwhere(np.ones_like(belief, bool))
    chunk = 2000  # bounds the [chunk, n_observed] pairwise-distance block's memory
    for start in range(0, all_idx.shape[0], chunk):
        block = all_idx[start : start + chunk]
        d2 = ((block[:, None, :] - obs_idx[None, :, :]) ** 2).sum(-1)
        out[block[:, 0], block[:, 1]] = obs_h[d2.argmin(axis=1)]
    return out


def build_features(observed: np.ndarray, belief: np.ndarray, sigma: np.ndarray, cell: float) -> np.ndarray:
    """[ny, nx, N_FEATURES]: bias, dist-to-observed, plane-fit, nearest-obs height, mean height
    within 2m, sigma -- exactly the six features specified for the linear inpainter."""
    dist = noise_mod._distance_to_observed(observed, cell)
    planefit = _plane_fit(observed, belief, cell, PLANE_FIT_RADIUS_M)
    nearest_h = _nearest_observed_height(observed, belief, cell)
    mean_h = _mean_height_window(observed, belief, cell, MEAN_HEIGHT_RADIUS_M)
    bias = np.ones_like(belief)
    return np.stack([bias, dist, planefit, nearest_h, mean_h, sigma], axis=-1).astype(np.float64)


# --- per-seed data, cached (scene/harness/features are fixed given a seed) ---------------------


@dataclass
class SeedData:
    seed: int
    harness: Harness
    truth: np.ndarray  # [ny, nx] float64
    belief: np.ndarray  # [ny, nx] float64, what the robot has actually sensed (0 unobserved)
    observed: np.ndarray  # [ny, nx] bool
    unobs: np.ndarray  # [ny, nx] bool, ~observed
    phi: np.ndarray  # [ny, nx, N_FEATURES]
    j_true: np.ndarray  # [N_PLANS] cost of every plan evaluated on ground truth


def build_seed(seed: int) -> SeedData:
    scene, truth, _measured, observed, sigma, poses, omega, _grid = build_case(seed, FAMILY, NOISE)
    harness = Harness(scene, poses, omega, device="cuda")
    belief = scene.elevation.astype(np.float64)
    phi = build_features(observed, belief, sigma, CELL)
    truth64 = truth.astype(np.float64)
    j_true = _evaluate(harness, truth.astype(np.float32))
    return SeedData(seed, harness, truth64, belief, observed, ~observed, phi, j_true)


def _assemble(seed: SeedData, fill: np.ndarray) -> np.ndarray:
    """Truth on observed cells, `fill` on unobserved -- the one map-assembly rule shared by
    training and evaluation for every method compared."""
    return np.where(seed.observed, seed.truth, fill).astype(np.float32)


def zero_fill(seed: SeedData) -> np.ndarray:
    """Baseline: current pipeline behaviour -- unobserved cells stay at the (zero) belief value."""
    return np.zeros_like(seed.truth)


def plane_fill(seed: SeedData) -> np.ndarray:
    """Baseline: feature 3 (the plane-fit extrapolation) as the prediction, nothing learned."""
    return seed.phi[..., FEATURE_NAMES.index("planefit")]


def model_fill(seed: SeedData, w: np.ndarray) -> np.ndarray:
    return seed.phi @ w


# --- hand-written Adam ---------------------------------------------------------------------------


class Adam:
    def __init__(self, n: int, lr: float) -> None:
        self.lr = lr
        self.m = np.zeros(n)
        self.v = np.zeros(n)
        self.t = 0

    def step(self, w: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = ADAM_BETA1 * self.m + (1 - ADAM_BETA1) * grad
        self.v = ADAM_BETA2 * self.v + (1 - ADAM_BETA2) * grad**2
        m_hat = self.m / (1 - ADAM_BETA1**self.t)
        v_hat = self.v / (1 - ADAM_BETA2**self.t)
        return w - self.lr * m_hat / (np.sqrt(v_hat) + ADAM_EPS)


# --- the three loss/gradient conditions ----------------------------------------------------------
# Each returns (loss, grad_w) or (loss, grad_w, per_seed_grad_norms). `grad_w` is the mean over
# the minibatch of dLoss_seed/dw, computed by hand: the model is linear (dh_hat/dw = phi), so the
# chain rule only needs dLoss/dh_hat per cell, contracted against phi.


def mse_loss_grad(seeds: list[SeedData], w: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """No simulator: mean over unobserved cells of (h_hat - truth)^2.

    Returns per-seed grad norms too (3-tuple), matching the other two conditions' shape, so
    `train`/the stage-1 report treat all three uniformly.
    """
    losses, grads, gnorms = [], [], []
    for s in seeds:
        h_hat = s.phi @ w
        resid = (h_hat - s.truth)[s.unobs]
        n = resid.size
        losses.append(float(np.mean(resid**2)))
        g = (2.0 / n) * (s.phi[s.unobs].T @ resid)
        grads.append(g)
        gnorms.append(float(np.linalg.norm(g)))
    return float(np.mean(losses)), np.mean(grads, axis=0), np.asarray(gnorms)


def _through_adjoint(
    seeds: list[SeedData], w: np.ndarray, dloss_dcost_fn
) -> tuple[float, np.ndarray, np.ndarray, list[float]]:
    """Shared machinery for cost-space and decision losses: assemble h_hat into the map, get
    (dJ_k/d(map), J_k) from ONE `_weighted_adjoint` call, then let `dloss_dcost_fn` supply
    dLoss/dJ_k; the rest of the chain (map -> w) is identical for both losses."""
    losses, grads, gnorms = [], [], []
    for s in seeds:
        elev = _assemble(s, s.phi @ w)
        g_map, costs = _weighted_adjoint(s.harness, elev)  # [K, ny, nx], [K]
        loss, dloss_dcost = dloss_dcost_fn(s, costs)
        dloss_dmap = np.tensordot(dloss_dcost, g_map, axes=(0, 0))  # [ny, nx]
        dloss_dmap = np.where(s.unobs, dloss_dmap, 0.0)  # observed cells: d(assembled)/dw = 0
        g_w = np.tensordot(dloss_dmap, s.phi, axes=([0, 1], [0, 1]))  # [N_FEATURES]
        losses.append(loss)
        grads.append(g_w)
        gnorms.append(float(np.linalg.norm(g_w)))
    return float(np.mean(losses)), np.mean(grads, axis=0), np.asarray(gnorms), losses


def cost_space_loss_grad(seeds: list[SeedData], w: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """mean_k (J_hat_k - J_true_k)^2, backpropagated through the contact adjoint into w."""

    def _dl(s: SeedData, costs: np.ndarray) -> tuple[float, np.ndarray]:
        diff = costs - s.j_true
        return float(np.mean(diff**2)), (2.0 / N_PLANS) * diff

    loss, grad, gnorms, _ = _through_adjoint(seeds, w, _dl)
    return loss, grad, gnorms


def decision_loss_grad(seeds: list[SeedData], w: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """SPO+-flavoured expected regret under a soft plan choice made on the PREDICTED map:
    L = sum_k softmin(J_hat)_k * J_true_k - min_k J_true_k. Softmin temperature is a fraction of
    the seed's own true-cost spread, so it is dimensionally consistent across seeds."""

    def _dl(s: SeedData, j_hat: np.ndarray) -> tuple[float, np.ndarray]:
        spread = max(float(s.j_true.max() - s.j_true.min()), 1e-3)
        temp = SOFTMIN_TEMP_FRAC * spread
        z = -j_hat / temp
        z = z - z.max()
        p = np.exp(z)
        p = p / p.sum()
        e_true = float((p * s.j_true).sum())
        loss = e_true - float(s.j_true.min())
        # d/dJ_hat_j of sum_k p_k J_true_k, through the softmax: -(p_j/T) * (J_true_j - E_p[J_true])
        dloss_djhat = -(p / temp) * (s.j_true - e_true)
        return loss, dloss_djhat

    loss, grad, gnorms, _ = _through_adjoint(seeds, w, _dl)
    return loss, grad, gnorms


# --- training loop, shared by all three conditions -----------------------------------------------


def tune_lr(loss_fn, seeds: list[SeedData], minibatch: int, rng: np.random.Generator) -> float:
    """Short runs at each candidate LR, scored on final loss over the SAME tuning seeds -- train
    loss only, per spec; held-out seeds never touch this."""
    best_lr, best_loss = LR_CANDIDATES[0], np.inf
    for lr in LR_CANDIDATES:
        w = np.zeros(N_FEATURES)
        opt = Adam(N_FEATURES, lr)
        ok = True
        for _ in range(LR_TUNE_STEPS):
            idx = rng.choice(len(seeds), size=min(minibatch, len(seeds)), replace=False)
            out = loss_fn([seeds[i] for i in idx], w)
            loss, grad = out[0], out[1]
            if not (np.isfinite(loss) and np.all(np.isfinite(grad))):
                ok = False
                break
            gnorm = np.linalg.norm(grad)
            if gnorm > 50.0:  # coarse safety clip during tuning only, not the reported one
                grad = grad * (50.0 / gnorm)
            w = opt.step(w, grad)
        final_loss = loss_fn(seeds, w)[0] if ok else np.inf
        if np.isfinite(final_loss) and final_loss < best_loss:
            best_loss, best_lr = final_loss, lr
    return best_lr


def train(
    loss_fn,
    train_data: list[SeedData],
    lr: float,
    n_steps: int,
    minibatch: int,
    clip_norm: float,
    rng: np.random.Generator,
) -> dict:
    w = np.zeros(N_FEATURES)
    opt = Adam(N_FEATURES, lr)
    history, gnorm_all, n_skipped, n_clipped = [], [], 0, 0
    for _ in range(n_steps):
        idx = rng.choice(len(train_data), size=minibatch, replace=False)
        batch = [train_data[i] for i in idx]
        out = loss_fn(batch, w)
        loss, grad = out[0], out[1]
        if len(out) > 2:
            gnorm_all.extend(np.asarray(out[2]).tolist())
        if not (np.isfinite(loss) and np.all(np.isfinite(grad))):
            n_skipped += 1
            history.append(history[-1] if history else float("nan"))
            continue
        gnorm = float(np.linalg.norm(grad))
        if gnorm > clip_norm:
            grad = grad * (clip_norm / gnorm)
            n_clipped += 1
        w = opt.step(w, grad)
        history.append(loss)
    return {
        "w": w.tolist(),
        "history": history,
        "gnorm_all": gnorm_all,
        "n_skipped": n_skipped,
        "n_clipped": n_clipped,
        "lr": lr,
    }


# --- stage 1: plumbing gate -----------------------------------------------------------------------


def _fd_check(seed_data: SeedData, loss_fn, eps: float) -> dict:
    """Central finite-difference check of the assembled dLoss/dw against `loss_fn`'s analytic
    gradient, at one seed. `loss_fn` must be one of the `_through_adjoint`-based conditions (its
    scalar loss alone is cheap: a forward through `_evaluate`, no backward)."""
    rng = np.random.default_rng(0)
    w0 = 0.02 * rng.standard_normal(N_FEATURES)  # small, nonzero: a zero map degenerates features

    def loss_only(w: np.ndarray) -> float:
        elev = _assemble(seed_data, seed_data.phi @ w)
        costs = _evaluate(seed_data.harness, elev)
        if loss_fn is mse_loss_grad:
            return float(np.mean((seed_data.phi @ w - seed_data.truth)[seed_data.unobs] ** 2))
        if loss_fn is cost_space_loss_grad:
            return float(np.mean((costs - seed_data.j_true) ** 2))
        spread = max(float(seed_data.j_true.max() - seed_data.j_true.min()), 1e-3)
        temp = SOFTMIN_TEMP_FRAC * spread
        z = -costs / temp
        z = z - z.max()
        p = np.exp(z)
        p = p / p.sum()
        return float((p * seed_data.j_true).sum() - seed_data.j_true.min())

    out = loss_fn([seed_data], w0)
    analytic = np.asarray(out[1])
    fd = np.zeros(N_FEATURES)
    for j in range(N_FEATURES):
        wp_, wm = w0.copy(), w0.copy()
        wp_[j] += eps
        wm[j] -= eps
        fd[j] = (loss_only(wp_) - loss_only(wm)) / (2 * eps)
    scale = max(np.abs(analytic).max(), np.abs(fd).max(), 1e-8)
    rel_err = np.abs(fd - analytic) / scale
    return {
        "analytic": analytic.tolist(),
        "finite_diff": fd.tolist(),
        "max_rel_err": float(rel_err.max()),
        "eps": eps,
    }


def stage1_gate(train_data: list[SeedData]) -> dict:
    s0 = train_data[0]
    out: dict = {"seed": s0.seed, "fd_checks": {}}
    print("\n=== STAGE 1: plumbing gate ===")
    for name, fn, eps in (
        ("mse", mse_loss_grad, 1e-4),
        # eps is a sweet spot, not a free choice: at 3e-3 the uniform (bias-feature) height
        # shift it induces is already at the contact solver's documented mm validity radius
        # (CLAIMS.md) and the check picks up real curvature, not a bug (rel err ~1-9%,
        # measured); at <=3e-5 float32 noise in the forward dominates instead. 1e-4 sits
        # between both failure modes (rel err <1% here, swept 3e-5..3e-3).
        ("cost_space", cost_space_loss_grad, 1e-4),
        ("decision", decision_loss_grad, 1e-4),
    ):
        chk = _fd_check(s0, fn, eps)
        out["fd_checks"][name] = chk
        print(f"  {name:<12} max rel err {chk['max_rel_err']:.3e}  (eps={eps:g})")

    # (b) one-seed training: does the loss go down, and how does the gradient behave?
    print("\n  one-seed training (no clip), 60 steps:")
    rng = np.random.default_rng(1)
    out["one_seed"] = {}
    for name, fn in (
        ("mse", mse_loss_grad),
        ("cost_space", cost_space_loss_grad),
        ("decision", decision_loss_grad),
    ):
        res = train(fn, [s0], lr=0.03, n_steps=60, minibatch=1, clip_norm=1e9, rng=rng)
        gnorms = np.asarray(res["gnorm_all"]) if res["gnorm_all"] else np.array([np.nan])
        finite_hist = [h for h in res["history"] if np.isfinite(h)]
        improved = len(finite_hist) >= 2 and finite_hist[-1] < finite_hist[0]
        out["one_seed"][name] = {
            "loss0": finite_hist[0] if finite_hist else None,
            "loss_final": finite_hist[-1] if finite_hist else None,
            "improved": bool(improved),
            "n_skipped": res["n_skipped"],
            "gnorm_median": float(np.nanmedian(gnorms)),
            "gnorm_max": float(np.nanmax(gnorms)),
        }
        print(
            f"    {name:<12} loss {out['one_seed'][name]['loss0']:.4g} -> "
            f"{out['one_seed'][name]['loss_final']:.4g}  "
            f"improved={improved}  skipped={res['n_skipped']}/60  "
            f"gnorm median/max {out['one_seed'][name]['gnorm_median']:.3g}/"
            f"{out['one_seed'][name]['gnorm_max']:.3g}"
        )
    return out


# --- stage 2: the kill-test ------------------------------------------------------------------------


def height_rmse(seed: SeedData, fill: np.ndarray) -> float:
    return float(np.sqrt(np.mean((fill[seed.unobs] - seed.truth[seed.unobs]) ** 2)))


def evaluate_method(test_data: list[SeedData], fill_fn) -> dict:
    """Regret, Kendall tau, and unobserved-cell height RMSE, one held-out seed at a time."""
    regrets, taus, rmses = [], [], []
    for s in test_data:
        fill = fill_fn(s)
        elev = _assemble(s, fill)
        j_hat = _evaluate(s.harness, elev)
        pick = int(np.argmin(j_hat))
        regrets.append(float(s.j_true[pick] - s.j_true.min()))
        taus.append(kendall_tau(j_hat, s.j_true))
        rmses.append(height_rmse(s, fill))
    return {"regret": regrets, "tau": taus, "rmse": rmses}


# --- decision-loss diagnostic ---------------------------------------------------------------------
# The pilot's decision loss barely moved (1.882 -> 1.8 over 120 steps) yet the trained model won
# on held-out regret. Two candidate explanations, both checked here: (1) the softmin temperature
# is too high relative to the true cost spread, so the soft plan choice stays close to uniform
# throughout training -- the scalar loss (an expectation under a near-uniform distribution) is
# then nearly insensitive to h_hat even while its GRADIENT still points somewhere useful; (2) the
# useful signal is actually the cost-space-like component baked into the same chain rule (dJ_hat/dw
# through the identical `_weighted_adjoint` call) -- i.e. decision training works, when it works,
# for the same underlying reason cost-space training does, not because of the softmin machinery.


def softmin_entropy_and_temp(seeds: list[SeedData], w: np.ndarray) -> tuple[float, float]:
    """Mean softmin weight entropy (nats) and mean temperature used, over `seeds`, at `w`."""
    ents, temps = [], []
    for s in seeds:
        elev = _assemble(s, s.phi @ w)
        j_hat = _evaluate(s.harness, elev)
        spread = max(float(s.j_true.max() - s.j_true.min()), 1e-3)
        temp = SOFTMIN_TEMP_FRAC * spread
        z = -j_hat / temp
        z = z - z.max()
        p = np.exp(z)
        p = p / p.sum()
        ents.append(float(-(p * np.log(p + 1e-12)).sum()))
        temps.append(temp)
    return float(np.mean(ents)), float(np.mean(temps))


def cost_space_alignment(seeds: list[SeedData], w: np.ndarray) -> float:
    """mean_k (J_hat_k - J_true_k)^2 at `w`, forward-only -- the exact quantity `cost_space_loss_grad`
    minimizes, evaluated here on DECISION-trained weights as a diagnostic: does decision training
    reduce it too, even while the decision loss itself barely moves?"""
    losses = []
    for s in seeds:
        elev = _assemble(s, s.phi @ w)
        j_hat = _evaluate(s.harness, elev)
        losses.append(float(np.mean((j_hat - s.j_true) ** 2)))
    return float(np.mean(losses))


def diagnose_decision(seeds: list[SeedData], w_decision: np.ndarray, w_cost_space: np.ndarray) -> dict:
    w0 = np.zeros(N_FEATURES)
    ent0, temp = softmin_entropy_and_temp(seeds, w0)
    ent1, _ = softmin_entropy_and_temp(seeds, w_decision)
    cs0 = cost_space_alignment(seeds, w0)
    cs1 = cost_space_alignment(seeds, w_decision)
    cos_sim = float(
        np.dot(w_decision, w_cost_space)
        / (np.linalg.norm(w_decision) * np.linalg.norm(w_cost_space) + 1e-12)
    )
    return {
        "n_seeds": len(seeds),
        "softmin_temp_mean": temp,
        "entropy_max_possible": float(np.log(N_PLANS)),
        "entropy_at_init": ent0,
        "entropy_at_decision_trained_w": ent1,
        "cost_space_like_loss_at_init": cs0,
        "cost_space_like_loss_at_decision_trained_w": cs1,
        "cosine_sim_decision_vs_cost_space_weights": cos_sim,
    }


def main(stage1_only: bool = False) -> None:
    t_start = time.time()
    wp.init()
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"building {len(TRAIN_SEEDS)} train + {len(TEST_SEEDS)} test seeds "
          f"(family={FAMILY}, noise={NOISE}) ...")
    train_data = [build_seed(s) for s in TRAIN_SEEDS]
    test_data = [build_seed(s) for s in TEST_SEEDS]
    print(f"  done in {time.time() - t_start:.1f}s")

    report: dict = {"family": FAMILY, "noise": NOISE, "n_train": len(train_data), "n_test": len(test_data)}
    report["stage1"] = stage1_gate(train_data)

    # Decide the clip norm from stage 1's own gradient-norm stats: clip well above the median so
    # normal steps pass through untouched, but well below the max so a heavy-tailed outlier (the
    # failure mode CLAIMS.md predicts for this exact gradient) cannot dominate one Adam update.
    all_medians = [
        report["stage1"]["one_seed"][k]["gnorm_median"] for k in ("cost_space", "decision")
    ]
    clip_norm = float(10.0 * max(all_medians)) if all_medians else 1.0
    report["clip_norm"] = clip_norm
    print(f"\ngradient clip norm set to {clip_norm:.4g} (10x the larger of the two median grad norms)")

    if stage1_only:
        (OUT / "dfl_stage1.json").write_text(json.dumps(report, indent=2))
        print(f"\nwrote {OUT / 'dfl_stage1.json'} -- stopping (--stage1-only)")
        return

    print("\n=== STAGE 2: kill-test ===")
    rng = np.random.default_rng(2)
    report["training"] = {}
    trained: dict[str, np.ndarray] = {}
    for name, fn in (
        ("mse", mse_loss_grad),
        ("cost_space", cost_space_loss_grad),
        ("decision", decision_loss_grad),
    ):
        t0 = time.time()
        lr = tune_lr(fn, train_data, MINIBATCH, rng)
        res = train(fn, train_data, lr=lr, n_steps=N_STEPS, minibatch=MINIBATCH, clip_norm=clip_norm, rng=rng)
        dt = time.time() - t0
        finite_hist = [h for h in res["history"] if np.isfinite(h)]
        gnorms = np.asarray(res["gnorm_all"]) if res["gnorm_all"] else np.array([np.nan])
        trained[name] = np.asarray(res["w"])
        report["training"][name] = {
            "lr": lr,
            "loss0": finite_hist[0] if finite_hist else None,
            "loss_final": finite_hist[-1] if finite_hist else None,
            "n_skipped": res["n_skipped"],
            "n_clipped": res["n_clipped"],
            "gnorm_median": float(np.nanmedian(gnorms)),
            "gnorm_max": float(np.nanmax(gnorms)),
            "weights": res["w"],
            "history": res["history"],
            "seconds": dt,
        }
        print(
            f"  {name:<12} lr={lr:<6g} loss {finite_hist[0]:.4g} -> {finite_hist[-1]:.4g}  "
            f"skipped {res['n_skipped']}/{N_STEPS}  clipped {res['n_clipped']}/{N_STEPS}  "
            f"({dt:.1f}s)"
        )

    print("\nevaluating on held-out seeds ...")
    methods = {
        "zero_fill": zero_fill,
        "plane_fit": plane_fill,
        "mse": lambda s: model_fill(s, trained["mse"]),
        "cost_space": lambda s: model_fill(s, trained["cost_space"]),
        "decision": lambda s: model_fill(s, trained["decision"]),
    }
    eval_out = {name: evaluate_method(test_data, fn) for name, fn in methods.items()}
    report["eval"] = eval_out

    print(f"\n{'method':<12}{'regret':>10}{'tau':>10}{'height RMSE':>14}")
    for name in methods:
        r = eval_out[name]
        print(
            f"{name:<12}{np.mean(r['regret']):>10.4f}{np.mean(r['tau']):>+10.3f}"
            f"{np.mean(r['rmse']):>14.4f}"
        )

    zf_to_mse = float(np.mean(eval_out["zero_fill"]["regret"]) - np.mean(eval_out["mse"]["regret"]))
    report["zero_fill_to_mse_improvement"] = zf_to_mse
    print(f"\nzero-fill -> MSE regret improvement: {zf_to_mse:+.4f}")

    report["paired"] = {}
    print(f"\npaired per-seed results on regret, {len(test_data)} virgin test seeds "
          f"(b or c vs a = mse; positive = b/c better):")
    for name in ("cost_space", "decision"):
        per_seed_diff = np.asarray(eval_out["mse"]["regret"]) - np.asarray(eval_out[name]["regret"])
        n, k, p = sign_test(per_seed_diff)
        mean_improve = float(per_seed_diff.mean())
        frac_of_zf = mean_improve / zf_to_mse if abs(zf_to_mse) > 1e-9 else float("nan")
        report["paired"][name] = {
            "per_seed_diff": per_seed_diff.tolist(),  # mse_regret - name_regret, one per test seed
            "mean_improvement": mean_improve,
            "median_improvement": float(np.median(per_seed_diff)),
            "std_improvement": float(per_seed_diff.std()),
            "n_nonzero": n,
            "wins": k,
            "p": p,
            "frac_of_zerofill_to_mse": frac_of_zf,
        }
        print(
            f"  {name:<12} vs mse: mean {mean_improve:+.4f}  median {np.median(per_seed_diff):+.4f}  "
            f"std {per_seed_diff.std():.4f}  wins {k}/{n} (ties dropped)  p={p:.3g}  "
            f"({frac_of_zf:+.1%} of the zero-fill->MSE gap)"
        )

    print("\n=== decision-loss anomaly: why did the scalar loss barely move? ===")
    diag = diagnose_decision(train_data[:20], trained["decision"], trained["cost_space"])
    report["decision_diagnostic"] = diag
    print(
        f"  softmin temperature (mean, {diag['n_seeds']} train seeds): {diag['softmin_temp_mean']:.4g}"
        f"   max possible entropy ln({N_PLANS}) = {diag['entropy_max_possible']:.3f}"
    )
    print(
        f"  softmin entropy    at init: {diag['entropy_at_init']:.3f}   "
        f"at decision-trained w: {diag['entropy_at_decision_trained_w']:.3f}"
    )
    print(
        f"  cost-space-like MSE(J_hat,J_true)  at init: {diag['cost_space_like_loss_at_init']:.4g}   "
        f"at decision-trained w: {diag['cost_space_like_loss_at_decision_trained_w']:.4g}"
    )
    print(
        f"  cosine similarity, decision-w vs cost_space-w: "
        f"{diag['cosine_sim_decision_vs_cost_space_weights']:+.3f}"
    )

    report["seconds_total"] = time.time() - t_start
    (OUT / "dfl_full.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT / 'dfl_full.json'}  ({report['seconds_total']:.1f}s total)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage1-only", action="store_true")
    a = ap.parse_args()
    main(a.stage1_only)
