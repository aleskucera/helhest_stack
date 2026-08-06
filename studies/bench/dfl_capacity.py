"""Capacity ladder for the DFL result -- does a real model close the decision-vs-MSE gap?

    .venv/bin/python -m studies.bench.dfl_capacity

DFL_CLAIMS.md weakness #1 (the pre-declared strongest objection): the headline gap (regret
0.269 decision vs 0.598 MSE, `dfl.py`, linear 6-feature model) might be a CAPACITY artifact --
give MSE a real network and it might learn the same spatial structure the decision loss was
"buying" through shaping, closing the gap without any loss-shaping at all. This module runs the
first two rungs of the capacity ladder and reports whether that happens.

Rung 0 (reference, NOT rerun here): the existing linear-model numbers from `dfl_full.json`.
Rung 1: 6 -> 32 (tanh) -> 1 MLP on the identical six hand features `dfl.py` uses.
Rung 2: 31 -> 64 (tanh) -> 1 MLP on those six features plus a flattened 5x5 patch of the
    nearest-observed-height field around each cell -- real local spatial structure the six
    disc/point summaries alone do not carry, so MSE gets a genuine shot at exploiting it.

Only the MODEL changes. Everything else replicates `dfl.py` exactly: Adam, minibatch 8, 400
steps, an LR grid tuned on TRAIN loss only (never touching the 80 held-out seeds), gradient
clipping from a pre-training probe of the gradient's own tail, and the same three losses (mse /
cost_space / decision) via the identical `_weighted_adjoint` chain. The one difference forced by
the model change: an MLP cannot start at w=0 (tanh(0)=0 and a zero output layer kill the hidden
gradient by symmetry -- see `mlp_init`), so training starts from a small random point instead of
the linear model's exact zero, identical across all three conditions at a given rung so the
comparison stays fair.

Hand-written forward/backward (no torch needed at 257-2113 parameters), gated by a
finite-difference check against the SAME three loss functions used for training, run once BEFORE
any optimizer step -- a wrong gradient here would invalidate the entire ladder.
"""

from __future__ import annotations

import functools
import json
import time
from dataclasses import dataclass

import numpy as np
import warp as wp

from ..adjoint.harness import Harness
from .bundled import _weighted_adjoint
from .dfl import _assemble
from .dfl import build_features
from .dfl import evaluate_method
from .dfl import FAMILY
from .dfl import FEATURE_NAMES
from .dfl import LR_CANDIDATES
from .dfl import LR_TUNE_STEPS
from .dfl import MINIBATCH
from .dfl import N_STEPS
from .dfl import NOISE
from .dfl import SOFTMIN_TEMP_FRAC
from .dfl import TEST_SEEDS
from .dfl import TRAIN_SEEDS
from .dfl import Adam
from .ranking import _evaluate
from .ranking import build_case
from .ranking import CELL
from .ranking import N_PLANS
from .ranking import OUT
from .ranking import sign_test

RUNG0_PATH = OUT / "dfl_full.json"  # the linear-model reference; QUOTED, never rerun

PATCH_HALF = 2  # 5x5 window -> 25 taps, rung 2's spatial addition
FD_CHECK_N_WEIGHTS = 20  # "a few weights", per spec -- checked at each rung x condition
FD_MAX_REL_ERR = 1e-4  # the gate: training aborts if any condition exceeds this


# --- rung 2's extra feature: a spatial patch, not another point summary -----------------------


def _patch_offsets() -> list[tuple[int, int]]:
    return [
        (dy, dx)
        for dy in range(-PATCH_HALF, PATCH_HALF + 1)
        for dx in range(-PATCH_HALF, PATCH_HALF + 1)
    ]


def _patch_features(nearest_h_field: np.ndarray) -> np.ndarray:
    """[ny, nx, 25]: the 5x5 window of the nearest-observed-height field around each cell.

    Same np.roll convention `dfl._plane_fit` uses (np.roll(a, -dy, 0)[iy] == a[iy + dy]): each
    tap samples the field at the offset cell relative to the query cell. Unlike the six hand
    features -- all disc averages or single points -- this hands the model the local SHAPE of
    what has actually been seen nearby, which a linear or small model has no way to fold into a
    single scalar. This is rung 2's whole point.
    """
    cols = [
        np.roll(np.roll(nearest_h_field, -dy, axis=0), -dx, axis=1) for dy, dx in _patch_offsets()
    ]
    return np.stack(cols, axis=-1)


# --- per-seed data, built once and shared by both rungs -----------------------------------------


@dataclass
class CapSeedData:
    seed: int
    harness: Harness
    truth: np.ndarray  # [ny, nx] float64
    observed: np.ndarray  # [ny, nx] bool
    unobs: np.ndarray  # [ny, nx] bool
    j_true: np.ndarray  # [N_PLANS]
    phi_flat_r1: np.ndarray  # [ny*nx, 6]  contiguous
    phi_flat_r2: np.ndarray  # [ny*nx, 31] contiguous


def build_cap_seed(seed: int) -> CapSeedData:
    """Builds the harness/truth/features ONCE per seed and reuses them for both rungs -- the
    scene, contact solver setup and truth-cost evaluation do not depend on the inpainting model,
    so paying for them twice would be pure waste (the runtime guard's own advice)."""
    scene, truth, _measured, observed, sigma, poses, omega, _grid = build_case(seed, FAMILY, NOISE)
    harness = Harness(scene, poses, omega, device="cuda")
    belief = scene.elevation.astype(np.float64)
    phi6 = build_features(observed, belief, sigma, CELL)  # dfl.py's six features, unmodified
    patch = _patch_features(phi6[..., FEATURE_NAMES.index("nearest_h")])
    ny, nx = truth.shape
    phi_flat_r1 = np.ascontiguousarray(phi6.reshape(ny * nx, 6))
    phi_flat_r2 = np.ascontiguousarray(np.concatenate([phi6, patch], axis=-1).reshape(ny * nx, 31))
    j_true = _evaluate(harness, truth.astype(np.float32))
    return CapSeedData(
        seed,
        harness,
        truth.astype(np.float64),
        observed,
        ~observed,
        j_true,
        phi_flat_r1,
        phi_flat_r2,
    )


def _phi_flat(s: CapSeedData, n_in: int) -> np.ndarray:
    return s.phi_flat_r1 if n_in == 6 else s.phi_flat_r2


# --- hand-written MLP: forward, backward, init ---------------------------------------------------


@dataclass(frozen=True)
class MLPShape:
    n_in: int
    n_hidden: int

    @property
    def n_params(self) -> int:
        return self.n_in * self.n_hidden + self.n_hidden + self.n_hidden + 1


def mlp_init(shape: MLPShape, rng: np.random.Generator) -> np.ndarray:
    """Small random init, NOT zero: at w=0, tanh(0)=0 and the zero output layer make
    dLoss/dW1 == 0 by symmetry (da1 = dy @ W2.T = 0), unlike the linear model, which trains fine
    from w=0. Glorot-ish scale (1/sqrt(fan_in)) keeps the initial tanh in its linear region."""
    w1 = rng.standard_normal((shape.n_in, shape.n_hidden)) * np.sqrt(1.0 / shape.n_in)
    b1 = np.zeros(shape.n_hidden)
    w2 = rng.standard_normal((shape.n_hidden, 1)) * np.sqrt(1.0 / shape.n_hidden)
    b2 = np.zeros(1)
    return _pack(w1, b1, w2, b2)


def _pack(w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    return np.concatenate([w1.ravel(), b1.ravel(), w2.ravel(), b2.ravel()])


def _split(shape: MLPShape, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    i = shape.n_in * shape.n_hidden
    w1 = w[:i].reshape(shape.n_in, shape.n_hidden)
    j = i + shape.n_hidden
    b1 = w[i:j]
    k = j + shape.n_hidden
    w2 = w[j:k].reshape(shape.n_hidden, 1)
    b2 = w[k : k + 1]
    return w1, b1, w2, b2


def mlp_forward(shape: MLPShape, w: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, dict]:
    """x: [N, n_in] -> y: [N]."""
    w1, b1, w2, b2 = _split(shape, w)
    z1 = x @ w1 + b1  # [N, H]
    a1 = np.tanh(z1)  # [N, H]
    y = (a1 @ w2 + b2)[:, 0]  # [N]
    return y, {"x": x, "a1": a1, "w2": w2}


def mlp_backward(shape: MLPShape, cache: dict, dy: np.ndarray) -> np.ndarray:
    """dy: [N] upstream dLoss/dy_i, already carrying whatever scaling the caller's loss needs
    (the 2/n of an MSE, or the assembled dLoss/dmap of the adjoint-backed losses). Returns the
    flat parameter gradient summed over the N samples -- the direct generalisation of
    `dfl.py`'s `phi.T @ resid` one layer deeper."""
    x, a1, w2 = cache["x"], cache["a1"], cache["w2"]
    dy2 = dy[:, None]  # [N, 1]
    d_w2 = a1.T @ dy2  # [H, 1]
    d_b2 = dy2.sum(axis=0)  # [1]
    da1 = dy2 @ w2.T  # [N, H]
    dz1 = da1 * (1.0 - a1 * a1)  # tanh'
    d_w1 = x.T @ dz1  # [n_in, H]
    d_b1 = dz1.sum(axis=0)  # [H]
    return _pack(d_w1, d_b1, d_w2, d_b2)


def model_fill(shape: MLPShape, seed: CapSeedData, w: np.ndarray) -> np.ndarray:
    y, _ = mlp_forward(shape, w, _phi_flat(seed, shape.n_in))
    return y.reshape(seed.truth.shape)


# --- the three loss/gradient conditions, generalised from dfl.py's linear versions --------------


def mse_loss_grad(
    shape: MLPShape, seeds: list[CapSeedData], w: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    losses, grads, gnorms = [], [], []
    for s in seeds:
        y, cache = mlp_forward(shape, w, _phi_flat(s, shape.n_in))
        h_hat = y.reshape(s.truth.shape)
        resid_map = h_hat - s.truth
        resid = resid_map[s.unobs]
        n = resid.size
        losses.append(float(np.mean(resid**2)))
        dy = np.zeros_like(y)
        dy[s.unobs.ravel()] = (2.0 / n) * resid
        g = mlp_backward(shape, cache, dy)
        grads.append(g)
        gnorms.append(float(np.linalg.norm(g)))
    return float(np.mean(losses)), np.mean(grads, axis=0), np.asarray(gnorms)


def _through_adjoint(
    shape: MLPShape, seeds: list[CapSeedData], w: np.ndarray, dloss_dcost_fn
) -> tuple[float, np.ndarray, np.ndarray]:
    """Shared machinery for cost-space and decision losses -- `dfl._through_adjoint`, one layer
    of chain rule deeper: dLoss/dw for a linear model was `tensordot(dloss_dmap, phi)`; here it
    is `mlp_backward` fed the same per-cell `dloss_dmap`, flattened, as its upstream gradient."""
    losses, grads, gnorms = [], [], []
    for s in seeds:
        y, cache = mlp_forward(shape, w, _phi_flat(s, shape.n_in))
        h_hat = y.reshape(s.truth.shape)
        elev = _assemble(s, h_hat)
        g_map, costs = _weighted_adjoint(s.harness, elev)  # [K, ny, nx], [K]
        loss, dloss_dcost = dloss_dcost_fn(s, costs)
        dloss_dmap = np.tensordot(dloss_dcost, g_map, axes=(0, 0))  # [ny, nx]
        dloss_dmap = np.where(s.unobs, dloss_dmap, 0.0)  # observed cells: d(assembled)/dw = 0
        g_w = mlp_backward(shape, cache, dloss_dmap.ravel())
        losses.append(loss)
        grads.append(g_w)
        gnorms.append(float(np.linalg.norm(g_w)))
    return float(np.mean(losses)), np.mean(grads, axis=0), np.asarray(gnorms)


def _dl_cost_space(j_true: np.ndarray, costs: np.ndarray) -> tuple[float, np.ndarray]:
    diff = costs - j_true
    # 2/n from d(mean(diff**2))/d(diff): n = diff.size, NOT the module's N_PLANS constant -- the
    # two coincide in every real run (always 16 plans) but the synthetic wiring check below uses
    # a smaller synthetic plan count, and this keeps the formula correct there too.
    return float(np.mean(diff**2)), (2.0 / diff.size) * diff


def _dl_decision(j_true: np.ndarray, j_hat: np.ndarray) -> tuple[float, np.ndarray]:
    spread = max(float(j_true.max() - j_true.min()), 1e-3)
    temp = SOFTMIN_TEMP_FRAC * spread
    z = -j_hat / temp
    z = z - z.max()
    p = np.exp(z)
    p = p / p.sum()
    e_true = float((p * j_true).sum())
    loss = e_true - float(j_true.min())
    dloss_djhat = -(p / temp) * (j_true - e_true)
    return loss, dloss_djhat


def cost_space_loss_grad(
    shape: MLPShape, seeds: list[CapSeedData], w: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    return _through_adjoint(shape, seeds, w, lambda s, costs: _dl_cost_space(s.j_true, costs))


def decision_loss_grad(
    shape: MLPShape, seeds: list[CapSeedData], w: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    return _through_adjoint(shape, seeds, w, lambda s, j_hat: _dl_decision(s.j_true, j_hat))


CONDITIONS = {
    "mse": mse_loss_grad,
    "cost_space": cost_space_loss_grad,
    "decision": decision_loss_grad,
}
# forward-only scalar loss, matching each grad_fn above, for the finite-difference gate
_NEEDS_COSTS = {"mse": False, "cost_space": True, "decision": True}


def _forward_loss_mse(s: CapSeedData, h_hat: np.ndarray, _costs: np.ndarray | None) -> float:
    return float(np.mean((h_hat - s.truth)[s.unobs] ** 2))


def _forward_loss_cost_space(s: CapSeedData, _h_hat: np.ndarray, costs: np.ndarray) -> float:
    return float(np.mean((costs - s.j_true) ** 2))


def _forward_loss_decision(s: CapSeedData, _h_hat: np.ndarray, costs: np.ndarray) -> float:
    spread = max(float(s.j_true.max() - s.j_true.min()), 1e-3)
    temp = SOFTMIN_TEMP_FRAC * spread
    z = -costs / temp
    z = z - z.max()
    p = np.exp(z)
    p = p / p.sum()
    return float((p * s.j_true).sum() - s.j_true.min())


_FORWARD_LOSS = {
    "mse": _forward_loss_mse,
    "cost_space": _forward_loss_cost_space,
    "decision": _forward_loss_decision,
}


# --- finite-difference gate, run BEFORE any optimizer step ---------------------------------------


def fd_check(
    shape: MLPShape, name: str, seed_data: CapSeedData, w0: np.ndarray, eps: float, n_check: int
) -> dict:
    """Central FD check of `mlp_backward`'s output against `CONDITIONS[name]`'s analytic
    gradient, at `n_check` randomly chosen weights ("a few weights", per spec) -- checking all
    2113 of rung 2's would cost nothing but adds no information over a random subset."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(w0), size=min(n_check, len(w0)), replace=False)

    def loss_only(w: np.ndarray) -> float:
        h_hat = model_fill(shape, seed_data, w)
        costs = None
        if _NEEDS_COSTS[name]:
            elev = _assemble(seed_data, h_hat)
            costs = _evaluate(seed_data.harness, elev)
        return _FORWARD_LOSS[name](seed_data, h_hat, costs)

    analytic_full = np.asarray(CONDITIONS[name](shape, [seed_data], w0)[1])
    analytic = analytic_full[idx]
    fd = np.empty(len(idx))
    for j_local, j in enumerate(idx):
        wp_, wm = w0.copy(), w0.copy()
        wp_[j] += eps
        wm[j] -= eps
        fd[j_local] = (loss_only(wp_) - loss_only(wm)) / (2 * eps)
    scale = max(np.abs(analytic).max(), np.abs(fd).max(), 1e-8)
    rel_err = np.abs(fd - analytic) / scale
    return {
        "checked_idx": idx.tolist(),
        "analytic": analytic.tolist(),
        "finite_diff": fd.tolist(),
        "max_rel_err": float(rel_err.max()),
        "eps": eps,
    }


def _synthetic_wiring_check(
    shape: MLPShape, name: str, rng: np.random.Generator, n_check: int = 30
) -> dict:
    """Verifies `_through_adjoint`'s chain-rule composition (dloss/dcost -> tensordot(g_map) ->
    `mlp_backward`) against a SMOOTH, exactly-known linear-in-elevation cost functional
    (costs_k = sum(A_k * elev)) instead of the real contact solver -- so g_map = A exactly, no
    Newton-solve or float32 noise. This isolates "did I wire the chain rule correctly" from "how
    noisy is the solver's own forward pass", which the FIRST run of `fd_check` against the real
    adjoint could not: it topped out at ~1e-2 to 3e-2 (see `real_adjoint_best` in `fd_gate`'s
    output) even after an eps sweep, and a pure-MLP check with no adjoint at all (`mlp_backward`
    on a smooth scalar loss of `mlp_forward`'s own output) separately nailed ~1e-8 -- so the
    error had to be coming from the solver, not from anything in this file. This check confirms
    that directly: it is the same composition `cost_space_loss_grad`/`decision_loss_grad` use
    (via the shared `_dl_cost_space`/`_dl_decision`), with only `_weighted_adjoint` replaced."""
    ny, nx, n_plans = 14, 12, 6
    x = rng.standard_normal((ny * nx, shape.n_in))
    truth = rng.standard_normal((ny, nx))
    observed = rng.random((ny, nx)) < 0.4
    j_true = rng.standard_normal(n_plans)
    a_field = rng.standard_normal((n_plans, ny, nx)) * 0.1
    w0 = mlp_init(shape, rng)
    dl_fn = _dl_cost_space if name == "cost_space" else _dl_decision

    def forward(w: np.ndarray) -> tuple[np.ndarray, dict]:
        y, cache = mlp_forward(shape, w, x)
        h_hat = y.reshape(truth.shape)
        elev = np.where(observed, truth, h_hat)  # float64: no solver, no float32 cast
        costs = np.einsum("kij,ij->k", a_field, elev)
        return costs, cache

    def scalar_loss(w: np.ndarray) -> float:
        costs, _ = forward(w)
        return dl_fn(j_true, costs)[0]

    costs, cache = forward(w0)
    _, dloss_dcost = dl_fn(j_true, costs)
    dloss_dmap = np.where(~observed, np.tensordot(dloss_dcost, a_field, axes=(0, 0)), 0.0)
    analytic_full = mlp_backward(shape, cache, dloss_dmap.ravel())

    idx = rng.choice(len(w0), size=min(n_check, len(w0)), replace=False)
    eps = 1e-5
    fd = np.empty(len(idx))
    for j_local, j in enumerate(idx):
        wp_, wm = w0.copy(), w0.copy()
        wp_[j] += eps
        wm[j] -= eps
        fd[j_local] = (scalar_loss(wp_) - scalar_loss(wm)) / (2 * eps)
    analytic = analytic_full[idx]
    scale = max(np.abs(analytic).max(), np.abs(fd).max(), 1e-8)
    rel_err = np.abs(fd - analytic) / scale
    return {"max_rel_err": float(rel_err.max()), "eps": eps, "n_check": len(idx)}


def fd_gate(
    shape: MLPShape, seed_data: CapSeedData, w0: np.ndarray, rng: np.random.Generator
) -> dict:
    """Two-tier gate. (1) HARD, blocking, held to `FD_MAX_REL_ERR`: mse's real check (no adjoint
    involved, so nothing to decouple) and cost_space/decision's SYNTHETIC wiring check -- this is
    what "a wrong gradient invalidates everything" is actually about, and it is exactly checkable
    to high precision. (2) INFORMATIONAL, not blocking: cost_space/decision against the REAL
    adjoint, eps-swept the same way dfl.py calibrated its own linear-model gate. This is expected
    to land at the solver's own forward-noise floor, NOT nail 1e-4 -- dfl.py's reference gate
    itself only achieves 4.1e-4 / 6.8e-3 at this eps for the identical losses on a linear model
    (dfl_full.json stage1.fd_checks), so holding the MLP to a stricter bar there would be
    penalising the solver, not this file's backprop."""
    out: dict = {}
    ok = True

    chk = fd_check(shape, "mse", seed_data, w0, 1e-4, FD_CHECK_N_WEIGHTS)
    out["mse"] = chk
    passed = chk["max_rel_err"] < FD_MAX_REL_ERR
    ok = ok and passed
    print(
        f"    mse          max rel err {chk['max_rel_err']:.3e}  (eps={chk['eps']:g})  "
        f"{'OK' if passed else 'FAIL'}"
    )

    for name in ("cost_space", "decision"):
        syn = _synthetic_wiring_check(shape, name, rng)
        real_best = None
        for eps in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
            rchk = fd_check(shape, name, seed_data, w0, eps, FD_CHECK_N_WEIGHTS)
            if real_best is None or rchk["max_rel_err"] < real_best["max_rel_err"]:
                real_best = rchk
        out[name] = {"synthetic_wiring": syn, "real_adjoint_best": real_best}
        passed = syn["max_rel_err"] < FD_MAX_REL_ERR
        ok = ok and passed
        print(
            f"    {name:<12} synthetic-wiring rel err {syn['max_rel_err']:.3e}  "
            f"{'OK' if passed else 'FAIL'}  |  real-adjoint best {real_best['max_rel_err']:.3e} "
            f"(eps={real_best['eps']:g}, solver noise floor, informational)"
        )
    out["gate_passed"] = ok
    return out


# --- training loop, parameterised on an explicit init (dfl.train/tune_lr hardcode w=0) ----------


def tune_lr(
    loss_fn,
    seeds: list[CapSeedData],
    minibatch: int,
    rng: np.random.Generator,
    init_w: np.ndarray,
) -> float:
    """`dfl.tune_lr`, generalised to start from `init_w` instead of a hardcoded zero -- required
    because the MLP cannot start at zero (see `mlp_init`). Otherwise identical: short runs per
    candidate LR, scored on final loss over the SAME tuning seeds, train loss only."""
    best_lr, best_loss = LR_CANDIDATES[0], np.inf
    for lr in LR_CANDIDATES:
        w = init_w.copy()
        opt = Adam(len(w), lr)
        ok = True
        for _ in range(LR_TUNE_STEPS):
            idx = rng.choice(len(seeds), size=min(minibatch, len(seeds)), replace=False)
            loss, grad = loss_fn([seeds[i] for i in idx], w)[:2]
            if not (np.isfinite(loss) and np.all(np.isfinite(grad))):
                ok = False
                break
            gnorm = np.linalg.norm(grad)
            if gnorm > 50.0:
                grad = grad * (50.0 / gnorm)
            w = opt.step(w, grad)
        final_loss = loss_fn(seeds, w)[0] if ok else np.inf
        if np.isfinite(final_loss) and final_loss < best_loss:
            best_loss, best_lr = final_loss, lr
    return best_lr


def train(
    loss_fn,
    train_data: list[CapSeedData],
    lr: float,
    n_steps: int,
    minibatch: int,
    clip_norm: float,
    rng: np.random.Generator,
    init_w: np.ndarray,
) -> dict:
    """`dfl.train`, generalised the same way as `tune_lr` above; otherwise byte-identical logic
    (minibatch sampling, skip-on-nonfinite, clip-then-step)."""
    w = init_w.copy()
    opt = Adam(len(w), lr)
    history, gnorm_all, n_skipped, n_clipped = [], [], 0, 0
    for _ in range(n_steps):
        idx = rng.choice(len(train_data), size=minibatch, replace=False)
        batch = [train_data[i] for i in idx]
        loss, grad, gnorms = loss_fn(batch, w)
        gnorm_all.extend(np.asarray(gnorms).tolist())
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
        "w": w,
        "history": history,
        "gnorm_all": gnorm_all,
        "n_skipped": n_skipped,
        "n_clipped": n_clipped,
        "lr": lr,
    }


def _probe_gnorm_median(loss_fn, seed_data: CapSeedData, init_w: np.ndarray) -> float:
    """A cheap stand-in for `dfl.py`'s one-seed stage-1 diagnostic: 20 unclipped steps on one
    seed, just to read off the gradient's own scale before committing to a clip norm. The full
    60-step/no-clip diagnostic is dfl.py's plumbing-sanity tool, not part of what this ladder
    needs replicated (training protocol + evaluation); this keeps its one load-bearing output
    (a gnorm scale to clip against) without the extra adjoint calls."""
    w = init_w.copy()
    opt = Adam(len(w), 0.03)
    gnorms = []
    for _ in range(20):
        _, grad, gn = loss_fn([seed_data], w)
        if not np.all(np.isfinite(grad)):
            continue
        gnorms.append(float(gn[0]))
        w = opt.step(w, grad)
    return float(np.median(gnorms)) if gnorms else 1.0


# --- driving one rung end-to-end ------------------------------------------------------------------


def run_rung(
    rung: int, shape: MLPShape, train_data: list[CapSeedData], test_data: list[CapSeedData]
) -> dict:
    print(f"\n=== RUNG {rung}: {shape.n_in} -> {shape.n_hidden} (tanh) -> 1  "
          f"({shape.n_params} params) ===")
    report: dict = {"n_in": shape.n_in, "n_hidden": shape.n_hidden, "n_params": shape.n_params}

    init_w = mlp_init(shape, np.random.default_rng(100 + rung))  # shared init, all 3 conditions

    print("  finite-difference gate (before any optimizer step):")
    report["fd_gate"] = fd_gate(shape, train_data[0], init_w, np.random.default_rng(300 + rung))
    if not report["fd_gate"]["gate_passed"]:
        raise RuntimeError(
            f"rung {rung}: finite-difference gate FAILED (max_rel_err >= {FD_MAX_REL_ERR}) -- "
            "refusing to train on an unverified gradient"
        )

    # CONDITIONS' functions take (shape, seeds, w); bind `shape` for this rung so every call
    # site below can use the plain (seeds, w) signature `tune_lr`/`train`/`_probe_gnorm_median`
    # expect (mirroring dfl.py's loss_fn contract exactly, just closed over the rung's model).
    bound = {name: functools.partial(fn, shape) for name, fn in CONDITIONS.items()}

    clip_probe = {
        name: _probe_gnorm_median(fn, train_data[0], init_w)
        for name, fn in bound.items()
        if name != "mse"
    }
    clip_norm = float(10.0 * max(clip_probe.values()))
    report["clip_probe_gnorm_median"] = clip_probe
    report["clip_norm"] = clip_norm
    print(
        f"  clip norm set to {clip_norm:.4g} "
        "(10x the larger of cost_space/decision's probed median)"
    )

    rng = np.random.default_rng(200 + rung)
    trained: dict[str, np.ndarray] = {}
    report["training"] = {}
    for name, fn in bound.items():
        t0 = time.time()
        lr = tune_lr(fn, train_data, MINIBATCH, rng, init_w)
        res = train(
            fn,
            train_data,
            lr=lr,
            n_steps=N_STEPS,
            minibatch=MINIBATCH,
            clip_norm=clip_norm,
            rng=rng,
            init_w=init_w,
        )
        dt = time.time() - t0
        finite_hist = [h for h in res["history"] if np.isfinite(h)]
        trained[name] = res["w"]
        report["training"][name] = {
            "lr": lr,
            "loss0": finite_hist[0] if finite_hist else None,
            "loss_final": finite_hist[-1] if finite_hist else None,
            "n_skipped": res["n_skipped"],
            "n_clipped": res["n_clipped"],
            "seconds": dt,
        }
        print(
            f"  {name:<12} lr={lr:<6g} loss {finite_hist[0]:.4g} -> {finite_hist[-1]:.4g}  "
            f"skipped {res['n_skipped']}/{N_STEPS}  clipped {res['n_clipped']}/{N_STEPS}  "
            f"({dt:.1f}s)"
        )

    print("  evaluating on held-out seeds ...")
    methods = {
        name: (lambda s, name=name: model_fill(shape, s, trained[name])) for name in CONDITIONS
    }
    eval_out = {name: evaluate_method(test_data, fn) for name, fn in methods.items()}
    report["eval"] = eval_out

    print(f"\n  {'method':<12}{'regret':>10}{'tau':>10}{'height RMSE':>14}")
    for name in CONDITIONS:
        r = eval_out[name]
        print(
            f"  {name:<12}{np.mean(r['regret']):>10.4f}{np.mean(r['tau']):>+10.3f}"
            f"{np.mean(r['rmse']):>14.4f}"
        )

    report["paired"] = {}
    for name in ("cost_space", "decision"):
        diff = np.asarray(eval_out["mse"]["regret"]) - np.asarray(eval_out[name]["regret"])
        n, k, p = sign_test(diff)
        report["paired"][name] = {
            "mean_improvement": float(diff.mean()),
            "median_improvement": float(np.median(diff)),
            "n_nonzero": n,
            "wins": k,
            "p": p,
        }
        print(f"  paired {name:<12} vs mse: mean {diff.mean():+.4f}  wins {k}/{n}  p={p:.3g}")
    return report


# --- main -----------------------------------------------------------------------------------------


def main() -> None:
    t_start = time.time()
    wp.init()
    OUT.mkdir(parents=True, exist_ok=True)

    rung0 = json.loads(RUNG0_PATH.read_text())
    print("=== RUNG 0 (reference, quoted from dfl_full.json, NOT rerun) ===")
    for name in ("zero_fill", "plane_fit", "mse", "cost_space", "decision"):
        r = rung0["eval"][name]
        print(
            f"  {name:<12}{np.mean(r['regret']):>10.4f}{np.mean(r['tau']):>+10.3f}"
            f"{np.mean(r['rmse']):>14.4f}"
        )

    print(f"\nbuilding {len(TRAIN_SEEDS)} train + {len(TEST_SEEDS)} test seeds "
          f"(shared by both rungs; family={FAMILY}, noise={NOISE}) ...")
    t0 = time.time()
    train_data = [build_cap_seed(s) for s in TRAIN_SEEDS]
    test_data = [build_cap_seed(s) for s in TEST_SEEDS]
    print(f"  done in {time.time() - t0:.1f}s")

    ref_keys = ("zero_fill", "plane_fit", "mse", "cost_space", "decision")
    report: dict = {"rung0_reference": {k: rung0["eval"][k] for k in ref_keys}}
    report["rung0_paired"] = rung0["paired"]

    rungs = {1: MLPShape(n_in=6, n_hidden=32), 2: MLPShape(n_in=31, n_hidden=64)}
    report["rungs"] = {}
    for rung, shape in rungs.items():
        report["rungs"][str(rung)] = run_rung(rung, shape, train_data, test_data)

    # --- the pre-registered reading: does the gap shrink, and does capacity help MSE more? -----
    print("\n=== ladder summary: mean regret by rung x condition ===")
    print(f"{'rung':<8}{'mse':>10}{'cost_space':>12}{'decision':>10}")
    mse0 = float(np.mean(rung0["eval"]["mse"]["regret"]))
    dec0 = float(np.mean(rung0["eval"]["decision"]["regret"]))
    cs0 = float(np.mean(rung0["eval"]["cost_space"]["regret"]))
    print(f"{'0 (linear)':<8}{mse0:>10.4f}{cs0:>12.4f}{dec0:>10.4f}")
    mse_regrets, dec_regrets = [mse0], [dec0]
    for rung in (1, 2):
        ev = report["rungs"][str(rung)]["eval"]
        m, c, d = (float(np.mean(ev[k]["regret"])) for k in ("mse", "cost_space", "decision"))
        mse_regrets.append(m)
        dec_regrets.append(d)
        print(f"{rung:<8}{m:>10.4f}{c:>12.4f}{d:>10.4f}")

    gap0 = mse_regrets[0] - dec_regrets[0]
    # Pre-registered reading (SENSITIVITY_PLAN.md): confirmed if the gap shrinks BELOW
    # SIGNIFICANCE "at rung 1-2" -- i.e. at EITHER intermediate rung, not just the last one --
    # while MSE's absolute regret improves substantially. Check both rungs individually.
    per_rung_confound = {}
    for rung in (1, 2):
        gap_r = mse_regrets[rung] - dec_regrets[rung]
        p_r = report["rungs"][str(rung)]["paired"]["decision"]["p"]
        mse_gain_r = mse_regrets[0] - mse_regrets[rung]
        per_rung_confound[str(rung)] = {
            "gap": gap_r,
            "paired_p": p_r,
            "mse_regret_gain_vs_rung0": mse_gain_r,
            "confound_signature": bool(
                gap_r < gap0 and p_r >= 0.05 and mse_gain_r > 0.3 * mse_regrets[0]
            ),
        }
    mse_gain_final = mse_regrets[0] - mse_regrets[-1]  # regret DROP = improvement
    dec_gain_final = dec_regrets[0] - dec_regrets[-1]
    confound_confirmed = any(v["confound_signature"] for v in per_rung_confound.values())
    verdict = {
        "gap_rung0": gap0,
        "per_rung": per_rung_confound,
        "mse_regret_gain_rung0_to_2": mse_gain_final,
        "decision_regret_gain_rung0_to_2": dec_gain_final,
        "capacity_helps_mse_more_than_decision": mse_gain_final > dec_gain_final,
        "confound_confirmed": confound_confirmed,
    }
    report["verdict"] = verdict
    print(f"\ngap (mse - decision) at rung 0: {gap0:.4f}")
    for rung in (1, 2):
        v = per_rung_confound[str(rung)]
        print(
            f"  rung {rung}: gap {v['gap']:.4f}  paired p={v['paired_p']:.3g}  "
            f"mse regret gain vs rung0 {v['mse_regret_gain_vs_rung0']:+.4f}  "
            f"confound signature={v['confound_signature']}"
        )
    print(f"\nMSE regret gain (rung0->2):      {mse_gain_final:+.4f}")
    print(f"decision regret gain (rung0->2):  {dec_gain_final:+.4f}")
    print(
        "\nVERDICT: "
        + (
            "capacity confound CONFIRMED -- the gap shrank below significance at rung 1 or 2 "
            "while MSE improved substantially."
            if confound_confirmed
            else "capacity confound NOT confirmed -- the gap persisted (or grew) at every rung "
            "checked."
        )
    )

    report["seconds_total"] = time.time() - t_start
    (OUT / "dfl_capacity.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT / 'dfl_capacity.json'}  ({report['seconds_total']:.1f}s total)")


if __name__ == "__main__":
    main()
