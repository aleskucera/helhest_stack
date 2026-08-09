"""Can the adjoint run through a CYLINDER contact, and is it right?

    .venv/bin/python -m studies.adjoint.cylinder_adjoint

`DifferentiableSimulator` refuses `wheel_width`, and the commit that added the cylinder envelope
says why: the taped path would need a [B, n_yaw, ny, nx] envelope stack threaded through the
settle, because a cylinder is not yaw-invariant and the production pipeline pre-dilates the whole
grid for every heading. That is an engineering cost, and it was read as a blocker.

It is not one, because of how the existing tape is already built. In `harness._rollout` the
arg-max is computed OFF the tape and only the gather is differentiated:

    sim._contact()               off tape -> frozen source index per cell (best_k)
    with tape: _launches()       on tape  -> envelope[c] = elevation[best_k[c]] + cap[c]

So the contact element never appears in the derivative. It decides WHICH indices are frozen;
the differentiable part is the same linear gather either way. That means the cylinder adjoint
can be assembled from parts that already exist:

    d(cost)/d(raw)  =  d(cost)/d(envelope)  x  d(envelope)/d(raw)

The first factor is what the harness already returns with `dilate=False` (identity contact, the
envelope IS the leaf). The second is a scatter through the cylinder's own arg-max, which is a
permutation with a constant offset -- no engine change, no stack, no new kernel.

WHAT THIS DOES AND DOES NOT SHOW. It runs on a straight plan, where the heading is constant and
one dilated slice covers the whole rollout, so it sidesteps the stack rather than implementing
it. That is enough to answer the question that matters -- is the cylinder adjoint numerically
sound, or does the thinner element make it worse? -- because the failure the paper documents
(the validity radius) is about the distance to a contact-set change, and a cylinder has FEWER
candidates competing for the arg-max than a disk, which should move that distance.

Verified against finite differences of the TRUE forward: perturb a raw cell, re-dilate with the
cylinder element, re-run the rollout. That differences the real thing including contact
switching, so it catches a wrong chain rule and a frozen arg-max alike.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import warp as wp

from ..bench.ranking import build_case
from ..bench.ranking import CELL
from ..bench.ranking import OUT
from .harness import Harness
from .harness import TERM_NAMES
from helhest.engine import RobotParams

SETTLE = TERM_NAMES.index("settle")
HALF_WIDTH = 0.05  # [m] half the measured tread


def element(kind: str, cell: float, radius: float, yaw: float) -> tuple[np.ndarray, ...]:
    """(dy, dx, cap) for the disk or for the cylinder at heading `yaw`."""
    r = int(math.ceil(math.hypot(radius, HALF_WIDTH) / cell))
    dy, dx = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
    wx, wy = dx * cell, dy * cell
    if kind == "sphere":
        d2 = wx**2 + wy**2
        keep = d2 <= radius**2
        cap = np.sqrt(np.maximum(radius**2 - d2[keep], 0.0)) - radius
    else:
        c, s = math.cos(yaw), math.sin(yaw)
        along = wx * c + wy * s
        across = -wx * s + wy * c
        keep = (np.abs(along) <= radius) & (np.abs(across) <= HALF_WIDTH)
        cap = np.sqrt(np.maximum(radius**2 - along[keep] ** 2, 0.0)) - radius
    return dy[keep], dx[keep], cap


def dilate(raw: np.ndarray, dy: np.ndarray, dx: np.ndarray, cap: np.ndarray) -> tuple:
    """Envelope and the arg-max source index, exactly the operation `_contact` performs.

    env[p] = max_k raw[p + off_k] + cap_k, and `src[p]` is the flat index of the winning cell --
    which is the whole of d(env)/d(raw): one 1.0 from each envelope cell to its own winner.
    """
    ny, nx = raw.shape
    iy, ix = np.mgrid[0:ny, 0:nx]
    cand = np.empty((len(dy), ny, nx))
    src = np.empty((len(dy), ny, nx), np.int64)
    for k, (ddy, ddx, cc) in enumerate(zip(dy, dx, cap)):
        yy = np.clip(iy + ddy, 0, ny - 1)
        xx = np.clip(ix + ddx, 0, nx - 1)
        cand[k] = raw[yy, xx] + cc
        src[k] = yy * nx + xx
    best = cand.argmax(axis=0)
    env = np.take_along_axis(cand, best[None], 0)[0]
    return env.astype(np.float32), np.take_along_axis(src, best[None], 0)[0]


def chain(grad_env: np.ndarray, src: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """d(cost)/d(raw) from d(cost)/d(envelope): scatter each envelope cell's gradient onto the
    raw cell that won its max. Every envelope cell contributes to exactly one raw cell, so this
    is the transpose of the frozen gather -- the same thing the engine does internally."""
    out = np.zeros(shape[0] * shape[1])
    np.add.at(out, src.ravel(), grad_env.ravel())
    return out.reshape(shape)


def straight_plans(
    scene, n_plans: int, n_steps: int, speed: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Constant-heading plans, so ONE dilated slice covers the whole rollout.

    Anchored to the SCENE's own origin: this patch runs x in [-1, 8], y in [-4.5, 4.5], and a
    plan started outside it clamps its whole footprint onto the border, where six cells carry
    every gradient and a finite-difference check means nothing.
    """
    poses = np.zeros((n_plans, 3), np.float32)
    poses[:, 0] = scene.origin_x + 1.5
    poses[:, 1] = scene.origin_y + 4.5 + np.linspace(-1.2, 1.2, n_plans)
    omega = np.full((n_steps, n_plans, 3), speed, np.float32)
    return poses, omega


def run(seed: int, device: str, n_cells: int, eps: float) -> dict:
    scene, *_rest = build_case(seed, "hybrid", "all")
    raw = np.asarray(scene.elevation, np.float64)
    rp = RobotParams()
    poses, omega = straight_plans(scene, 4, 30)
    out = {}
    for kind in ("sphere", "cylinder"):
        dy, dx, cap = element(kind, CELL, rp.wheel_radius, yaw=0.0)
        env, src = dilate(raw, dy, dx, cap)

        h = Harness(scene, poses, omega, device=device)
        with wp.ScopedDevice(device):
            h._env0 = wp.array(np.tile(env, (len(poses), 1, 1)), dtype=wp.float32)
        h._env_np = env
        # dilate=False: identity contact, so the ENVELOPE is the differentiation leaf
        grads_env, terms = h.adjoint(dilate=False, leaf="elevation")
        g_env = grads_env[SETTLE]
        g_raw = np.stack([chain(g_env[b], src, raw.shape) for b in range(len(poses))])

        # finite differences of the TRUE forward: perturb raw, RE-DILATE, re-run
        rng = np.random.default_rng(seed)
        mag = np.abs(g_raw).sum(axis=0)
        # sample the whole support, not just its top decile: a check that only visits the
        # loudest cells cannot see the quiet ones being wrong
        cand = np.argwhere(mag > 0.02 * mag.max())
        pick = cand[rng.choice(len(cand), size=min(n_cells, len(cand)), replace=False)]
        fd, ad = [], []
        for (cy, cx) in pick:
            vals = []
            for sgn in (+1, -1):
                pert = raw.copy()
                pert[cy, cx] += sgn * eps
                e2, _ = dilate(pert, dy, dx, cap)
                # `_rollout` does NOT reload the terrain, so overwriting `_env0` alone leaves
                # the previous envelope on the device and differences nothing.
                with wp.ScopedDevice(device):
                    h.sim.set_terrain(
                        wp.array(np.tile(e2, (len(poses), 1, 1)).astype(np.float32),
                                 dtype=wp.float32)
                    )
                vals.append(h.forward(dilate=False)[SETTLE].copy())
            fd.append((vals[0] - vals[1]) / (2 * eps))
            ad.append(g_raw[:, cy, cx])
        del h
        fd, ad = np.array(fd), np.array(ad)
        scale = max(np.abs(fd).max(), 1e-12)
        rel = np.abs(ad - fd) / scale
        out[kind] = {
            "K": int(len(dy)),
            "n_cells": int(len(pick)),
            "median_rel_err": float(np.median(rel)),
            "p90_rel_err": float(np.percentile(rel, 90)),
            "max_rel_err": float(rel.max()),
            "frac_within_1pct": float((rel < 0.01).mean()),
            "grad_scale": float(np.abs(fd).max()),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cells", type=int, default=40)
    ap.add_argument("--eps", type=float, default=1e-3)
    args = ap.parse_args()
    wp.init()
    res = run(args.seed, args.device, args.cells, args.eps)
    print(f"=== adjoint vs finite differences at eps = {args.eps*1000:.0f} mm ===")
    print(f"{'element':>10}{'K':>5}{'cells':>7}{'median':>10}{'p90':>10}{'max':>10}{'<1%':>8}")
    for k, v in res.items():
        print(f"{k:>10}{v['K']:>5}{v['n_cells']:>7}{v['median_rel_err']:>10.2e}"
              f"{v['p90_rel_err']:>10.2e}{v['max_rel_err']:>10.2e}{v['frac_within_1pct']:>8.0%}")
    (OUT / "cylinder_adjoint.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT / 'cylinder_adjoint.json'}")


if __name__ == "__main__":
    main()
