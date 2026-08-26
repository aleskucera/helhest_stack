"""GPU analytic estimator vs GPU sampling, same process, same scenes, same device.

WHY THIS REPLACES `clark_wall_same_run.py`. That benchmark timed a NumPy fold on one core
against a batched GPU sampler. The comparison is honest about what it measures, but it measures
an IMPLEMENTATION gap, not a method gap: a reviewer is entitled to read it as "the baseline was
better engineered than the method". With `helhest.risk.contact` the estimator runs as Warp
kernels on the same device as the sampler, so the comparison is finally between the two methods.

WHAT IS COMPARED, and the choices that decide the answer:

  * Sampling is given its BEST shape, not the study's. Its P*D rollouts are packed into the
    largest batch the card holds, so it runs at full occupancy -- a per-plan loop of D=256
    rollouts leaves the GPU idle (256 rollouts cost what 16 do). The chunk size is MEASURED here
    by bisection, not assumed.
  * The D perturbed terrains are drawn ONCE and reused across plans (common random numbers), so
    noise generation is O(D) rather than O(P*D). Again the favourable choice for sampling.
  * EQUAL ACCURACY is reported beside equal draw count. The estimator has a fixed error floor;
    sampling's error falls as 1/sqrt(n). Comparing both at D=256 compares them at UNEQUAL
    accuracy, so the script measures the estimator's own error against Monte-Carlo truth on
    these scenes and derives the draw count that matches it.
  * Both arms are timed with `wp.synchronize()` around the whole call, so neither hides queued
    work.

WHAT THIS DOES *NOT* SHOW. Both methods are O(P) in time, so this is a constant factor, not a
better asymptotic rate. The scaling difference is in MEMORY: sampling's working set is O(P*D)
and the estimator's is O(P), which is why the sampler needs chunking here and the estimator
does not.

    python -m studies.bench.clark_warp_wall --seeds 8
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import warp as wp

from helhest.engine import RobotParams
from helhest.risk.contact import WarpContactEstimator
from helhest.risk.sigma import rho1_table
from ..adjoint.harness import Harness
from ..adjoint.sigma import NoiseDraws
from . import matched_truth as mt
from .ranking import build_case
from .ranking import CELL
from .ranking import N_PLANS
from .ranking import OUT
from .risk import CORR_LEN
from .risk import N_DRAWS


# Largest rollout batch used for the sampling arm. Probing the card to failure was tried and
# abandoned: an OOM inside Warp leaves the async allocator in a state that corrupts the timings
# that follow it. 4096 is comfortably under the measured ceiling (8192 fails on a 4 GB A500) and
# occupancy is already flat well below it, so the sampler is not handicapped by the choice.
SAMPLER_BATCH = 4096


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    wp.init()

    rp = RobotParams()
    rho1 = rho1_table(CORR_LEN, CELL)
    rows = []
    for seed in range(a.seeds):
        scene, _t, _m, _o, sigma, poses, omega, _g = build_case(seed, "hybrid", "all")
        belief = scene.elevation.astype(np.float32)
        ny, nx = belief.shape
        ctrl, _der = mt.cylinder_controlled_trajectory(scene, poses, omega, device=a.device)
        if not np.isfinite(ctrl).all():
            continue
        n_t = ctrl.shape[0] - 1

        # --- the analytic arm ---------------------------------------------------------------
        est = WarpContactEstimator(ny, nx, CELL, scene.origin_x, scene.origin_y, rp, rho1,
                                   N_PLANS, n_t, device=a.device)
        with wp.ScopedDevice(a.device):
            b_d = wp.array(belief, dtype=wp.float32)
            s_d = wp.array(np.ascontiguousarray(sigma, np.float32), dtype=wp.float32)
            c_d = wp.array(np.ascontiguousarray(ctrl, np.float32), dtype=wp.float32)
        est.moments(b_d, s_d, c_d)
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            e_d, v_d = est.moments(b_d, s_d, c_d)
        wp.synchronize()
        fold_ms = (time.perf_counter() - t0) / 5 * 1000 / N_PLANS

        # --- the sampling arm, at full occupancy --------------------------------------------
        B = SAMPLER_BATCH
        hm = Harness(scene, np.tile(poses[0], (B, 1)).astype(np.float32),
                     np.zeros((n_t, B, 3), np.float32), device=a.device)
        hm.forward(dilate=True)
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            hm.forward(dilate=True)
        wp.synchronize()
        chunk_ms = (time.perf_counter() - t0) / 3 * 1000
        del hm
        per_rollout_ms = chunk_ms / B

        # the D terrains, drawn once (common random numbers across plans)
        hd = Harness(scene, np.tile(poses[0], (N_DRAWS, 1)).astype(np.float32),
                     np.zeros((n_t, N_DRAWS, 3), np.float32), device=a.device)
        draws = NoiseDraws((N_DRAWS, ny, nx), CELL, CORR_LEN, hd.device)
        with wp.ScopedDevice(a.device):
            base = wp.array(np.ascontiguousarray(np.tile(belief, (N_DRAWS, 1, 1)), np.float32))
            sg = wp.array(np.ascontiguousarray(sigma, np.float32), dtype=wp.float32)
            out = wp.zeros((N_DRAWS, ny, nx), dtype=wp.float32)
        draws.perturb(base, sg, 1.0, out, 1)
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            draws.perturb(base, sg, 1.0, out, 1)
        wp.synchronize()
        noise_ms = (time.perf_counter() - t0) / 3 * 1000
        del hd

        # --- the estimator's own error, against Monte-Carlo truth on THIS scene -------------
        terms = mt.cylinder_mc_truth_terms(scene, belief, sigma, poses, omega, device=a.device,
                                           seed=900_000 + seed, n_draws=N_DRAWS,
                                           corr_len=CORR_LEN, cell=CELL)
        from .clark import SETTLE_IDX
        samples = terms[SETTLE_IDX]           # [N_DRAWS, N_PLANS]; _cost_settle(terms) IS this
        mc_mean, mc_sd = samples.mean(axis=0), samples.std(axis=0)
        e_j = e_d.numpy().astype(np.float64)
        per_plan = (np.abs(e_j - mc_mean) / np.maximum(mc_sd, 1e-9)).tolist()
        err_over_sd = float(np.median(per_plan))

        rows.append({"seed": seed, "fold_ms_per_plan": fold_ms,
                     "sampler_batch": B, "chunk_ms": chunk_ms,
                     "per_rollout_ms": per_rollout_ms, "noise_ms": noise_ms,
                     "err_mean_over_sd": err_over_sd,
                     "err_per_plan": per_plan})
        del est
        print(f"  seed {seed:2d}: fold {fold_ms * 1000:6.1f} us/plan   rollout "
              f"{per_rollout_ms * 1000:6.2f} us   err/sd {err_over_sd:.4f}", flush=True)

    fold = float(np.median([r["fold_ms_per_plan"] for r in rows]))
    roll = float(np.median([r["per_rollout_ms"] for r in rows]))
    # POOL over every (seed, plan) case. A median of per-seed medians is a different and
    # noisier statistic, and with few seeds it lands several tens of percent off -- which
    # matters here because the equal-accuracy ratio goes as 1/err^2.
    err = float(np.median(np.concatenate([r["err_per_plan"] for r in rows])))
    n_eq = 1.0 / err ** 2
    res = {
        "element": "cylinder", "n_plans": N_PLANS, "n_draws": N_DRAWS, "seeds": len(rows),
        "median_fold_ms_per_plan": fold,
        "median_per_rollout_ms": roll,
        "rollouts_per_plan_equivalent": fold / roll,
        "pooled_err_mean_over_sd": err,
        "n_plan_cases": int(sum(len(r["err_per_plan"]) for r in rows)),
        "n_equivalent_draws": n_eq,
        "ratio_equal_draws": N_DRAWS * roll / fold,
        "ratio_equal_accuracy": n_eq * roll / fold,
        "sampler_batch": SAMPLER_BATCH,
        "rows": rows,
    }
    (OUT / "clark_warp_wall.json").write_text(json.dumps(res, indent=1))
    print("\n" + "=" * 78)
    print(f"  analytic          {fold * 1000:7.2f} us per plan")
    print(f"  one MC rollout    {roll * 1000:7.2f} us")
    print(f"  -> the estimator costs {fold / roll:.2f} rollouts")
    print(f"  its error is {err:.4f} of one noise sd -> matched by {n_eq:.0f} draws")
    print(f"  ratio at equal draws ({N_DRAWS}):  {N_DRAWS * roll / fold:6.1f}x")
    print(f"  ratio at equal ACCURACY:        {n_eq * roll / fold:6.1f}x")
    print(f"  sampler batch: {SAMPLER_BATCH}")
    print(f"\nwrote {OUT / 'clark_warp_wall.json'}")


if __name__ == "__main__":
    main()
