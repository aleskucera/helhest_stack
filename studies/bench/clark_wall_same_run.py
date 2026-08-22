"""The one wall-clock comparison the paper is allowed to quote: batched fold vs GPU MC-256,
both timed in THIS process, per seed, on the same scenes.

    .venv/bin/python -m studies.bench.clark_wall_same_run --seeds 10

WHY THIS EXISTS. clark_fast.py's docstring already warns that a fold-vs-sampling comparison
must never straddle two machine states, and re-measures the MC side in its own run -- but the
fold it times is the UNBATCHED lean fold (13.3 ms/plan), while the deployment case is the
batched fold over the whole candidate set (clark_conv.py, ~1.8 ms/plan), which had never been
timed against the Monte-Carlo in one run. The paper's "faster than sampling" sentence was
therefore built from two different runs of two different scripts. This script closes that gap:
same process, same seeds, same scenes -- `plan_moments_conv_batch` (sphere element) against a
256-draw batched GPU rollout, both reported per plan over the same 16-plan candidate set.

The MC side is timed exactly as clark_fast.rollout_cost does it (5 repeats of a warmed
Harness.forward on a 256-wide batch), but per seed on that seed's own scene.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import warp as wp

from ..adjoint.harness import Harness
from .clark import rho1_table
from .clark_conv import plan_moments_conv_batch
from .ranking import build_case
from .ranking import CELL
from .ranking import N_PLANS
from .ranking import OUT
from .risk import CORR_LEN
from helhest.engine import RobotParams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    wp.init()

    rp = RobotParams()
    rho1 = rho1_table(CORR_LEN, CELL)
    rows = []
    for seed in range(args.seeds):
        scene, _, _, _, sigma, poses, omega, _ = build_case(seed, "hybrid", "all")
        belief = scene.elevation.astype(np.float32)
        h = Harness(scene, poses, omega, device=args.device)
        h.forward(dilate=True)
        controlled = h.sim.controlled.numpy()
        del h
        geo = (scene.origin_x, scene.origin_y, CELL)

        # warm once (JIT/caches), then time the batched fold over the candidate set
        plan_moments_conv_batch(belief, sigma, controlled, rp, *geo, rho1)
        t0 = time.perf_counter()
        plan_moments_conv_batch(belief, sigma, controlled, rp, *geo, rho1)
        fold_s = (time.perf_counter() - t0) / N_PLANS

        # MC-256 on the SAME scene, timed as clark_fast.rollout_cost does
        poses_d = np.tile(poses[0], (256, 1)).astype(np.float32)
        omega_d = np.zeros((omega.shape[0], 256, 3), np.float32)
        hm = Harness(scene, poses_d, omega_d, device=args.device)
        hm.forward(dilate=True)  # warm
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            hm.forward(dilate=True)
        wp.synchronize()
        mc_s = (time.perf_counter() - t0) / 5
        del hm

        rows.append({"seed": seed, "fold_batch_s_per_plan": fold_s, "mc_256_s_per_plan": mc_s})
        print(
            f"  seed {seed:2d}: batched fold {fold_s*1e3:5.2f} ms/plan   "
            f"MC-256 {mc_s*1e3:5.2f} ms/plan   ({mc_s/fold_s:.2f}x)"
        )

    out = {
        "element": "sphere",
        "rows": rows,
        "median_fold_batch_ms": float(np.median([r["fold_batch_s_per_plan"] for r in rows]) * 1e3),
        "median_mc_256_ms": float(np.median([r["mc_256_s_per_plan"] for r in rows]) * 1e3),
        "median_ratio_mc_over_fold": float(
            np.median([r["mc_256_s_per_plan"] / r["fold_batch_s_per_plan"] for r in rows])
        ),
    }
    print(
        f"\n  medians: fold {out['median_fold_batch_ms']:.2f} ms/plan, "
        f"MC-256 {out['median_mc_256_ms']:.2f} ms/plan, "
        f"ratio {out['median_ratio_mc_over_fold']:.2f}x"
    )
    path = OUT / "clark_wall_same_run.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
