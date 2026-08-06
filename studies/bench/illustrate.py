"""Show what each sensing policy actually LOOKS AT, on one real seed of the experiment.

    .venv/bin/python -m studies.bench.illustrate --seed 7

The three policies differ only in how they score a map cell, and the scores are easier to
believe once seen. Everything here is computed by the same code path `ranking.py` uses -- the
same belief, the same taped adjoint, the same budget -- so the pictures are the experiment, not
an illustration drawn to match it.

  ENTROPY        score = sigma^2. Reveal where the map is least certain. Knows nothing about
                 where the robot intends to go.
  GEOMETRY       score = closeness to the nearest candidate plan's path. Reveal the ground you
                 are about to drive over. Knows the plans, but nothing about physics.
  ADJOINT        score = Var_k(dJ_k/dh_i) * sigma_i^2. Reveal the cells whose height the
                 candidate plans' costs DISAGREE about most. Needs a backward pass through the
                 settle for every plan.
  ORACLE         not a policy: it is allowed to see the actual error, and marks the ceiling.
"""

from __future__ import annotations

import argparse

import numpy as np
import warp as wp

from ..adjoint.harness import Harness
from ..adjoint.harness import TERM_NAMES
from .ranking import _evaluate
from .ranking import _plan_distance
from .ranking import build_case
from .ranking import COST_TERMS
from .ranking import OUT
from .ranking import POLICIES

PANELS = ("entropy", "swath", "disagreement", "oracle")
TITLE = {
    "entropy": "ENTROPY:  look where the map is least certain\n(score = sigma^2)",
    "swath": "GEOMETRY:  look where you are about to drive\n(score = closeness to a plan)",
    "disagreement": "ADJOINT:  look where the plans' costs disagree\n(score = Var_k(dJ/dh) * sigma^2)",
    "oracle": "ORACLE (not a policy):  look where the map\nis actually wrong AND it matters",
}


def build(seed: int, family: str, noise: str, budget: int):
    scene, truth, measured, observed, sigma, poses, omega, grid = build_case(seed, family, noise)
    harness = Harness(scene, poses, omega, device="cuda")
    belief = scene.elevation.astype(np.float32)

    grads, _ = harness.adjoint(dilate=True, leaf="elevation")
    grad = sum(w * grads[TERM_NAMES.index(k)] for k, w in COST_TERMS.items())
    dist = _plan_distance(harness, grid)
    traj = harness.sim.controlled.numpy()[:, :, :2].copy()
    j_bel = _evaluate(harness, belief)

    rng = np.random.default_rng(10_000 + seed)
    ctx = {
        "grad": grad,
        "sigma": sigma,
        "believed": j_bel,
        "dist": dist,
        "truth": truth,
        "measured": measured,
        "belief": belief,
        "env_cells": harness.sim.env_radius / 0.10,
        "rng": rng,
    }
    picks, scores = {}, {}
    for name in PANELS:
        score = np.asarray(POLICIES[name](ctx), float)
        flat = score.ravel().copy()
        flat[observed.ravel()] = -np.inf
        perm = rng.permutation(flat.size)
        order = perm[np.argsort(-flat[perm], kind="stable")][:budget]
        picks[name] = np.stack(np.unravel_index(order, sigma.shape), axis=1)
        scores[name] = np.where(observed, np.nan, score)
    del harness
    return truth, belief, observed, sigma, traj, picks, scores, grid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--family", default="hybrid")
    ap.add_argument("--noise", default="clean")
    ap.add_argument("--budget", type=int, default=100)
    a = ap.parse_args()

    wp.init()
    truth, belief, observed, _, traj, picks, scores, grid = build(
        a.seed, a.family, a.noise, a.budget
    )
    XX, YY = grid
    ext = [XX.min(), XX.max(), YY.min(), YY.max()]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(21.0, 10.5))
    axes = axes.ravel()

    def draw_plans(ax, lw=1.0, alpha=0.85):
        for k in range(traj.shape[1]):
            ax.plot(traj[:, k, 0], traj[:, k, 1], color="#111111", lw=lw, alpha=alpha)

    def frame(ax, title):
        ax.set_title(title, fontsize=11)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    # (1) the world: true terrain, the candidate plans, and what has been seen
    ax = axes[0]
    ax.imshow(truth, origin="lower", extent=ext, cmap="terrain", vmin=-0.45, vmax=0.45)
    ax.contour(XX, YY, observed.astype(float), levels=[0.5], colors="white", linewidths=2.0)
    draw_plans(ax, lw=1.2)
    ax.plot(0, 0, "o", color="red", ms=9, mec="k")
    frame(
        ax, "THE TRUTH (unknown to the robot)\nblack = the 16 candidate plans, white = already seen"
    )

    # (2) what the robot actually believes
    ax = axes[1]
    ax.imshow(belief, origin="lower", extent=ext, cmap="terrain", vmin=-0.45, vmax=0.45)
    draw_plans(ax, lw=1.2)
    ax.plot(0, 0, "o", color="red", ms=9, mec="k")
    frame(ax, "THE BELIEF: everything unseen is guessed FLAT\nthis is what the plans are ranked on")

    # (3) the error the sensing has to find
    ax = axes[2]
    err = np.abs(truth - belief)
    im = ax.imshow(err, origin="lower", extent=ext, cmap="magma", vmin=0, vmax=0.5)
    draw_plans(ax, lw=1.0, alpha=0.55)
    frame(ax, "HOW WRONG THE BELIEF IS\nbut only the part under the plans can matter")
    fig.colorbar(im, ax=ax, fraction=0.046, label="|truth - belief| [m]")

    # (4-6 + ) each policy's score and its picks
    axes[3].axis("off")  # the top row is the setup; the bottom row is the four policies
    for ax, name in zip(axes[4:], PANELS):
        s = scores[name]
        hi = np.nanpercentile(s, 99.5)
        ax.imshow(
            np.clip(s / (hi if hi > 0 else 1.0), 0, 1) ** 0.45,
            origin="lower",
            extent=ext,
            cmap="viridis",
        )
        draw_plans(ax, lw=0.8, alpha=0.5)
        p = picks[name]
        ax.plot(
            XX[p[:, 0], p[:, 1]],
            YY[p[:, 0], p[:, 1]],
            "s",
            color="red",
            ms=2.6,
            mec="none",
            alpha=0.95,
        )
        frame(ax, TITLE[name])

    fig.suptitle(
        f"What each sensing policy looks at  --  seed {a.seed}, {a.family} plans, "
        f"{a.noise} belief, {a.budget} cells revealed (red)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUT / f"methods_seed{a.seed}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")

    print("\nwhere each policy's cells landed:")
    for name in PANELS:
        p = picks[name]
        d = np.array(
            [
                min(
                    np.hypot(traj[:, k, 0] - XX[iy, ix], traj[:, k, 1] - YY[iy, ix]).min()
                    for k in range(traj.shape[1])
                )
                for iy, ix in p
            ]
        )
        print(f"  {name:<13} median distance off the nearest plan: {np.median(d):5.2f} m")


if __name__ == "__main__":
    main()
