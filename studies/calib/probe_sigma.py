"""Probe B: the per-cell sigma the mapping node has to publish, and its correlation length.

Two numbers the planner needs and the current mapper does not provide
(PROBABILISTIC_PLANNING_PLAN.md section 5):

  sigma(range)  each sweep is one independent estimate of a cell's surface, so the spread
                ACROSS sweeps is the per-cell uncertainty -- sensor noise plus registration,
                with within-cell terrain variation removed by using each sweep's own estimate.

  L             the spatial correlation length. Section 3.2 needs it for the clearance cross
                term, where dropping it forces the independent-cells answer and overestimates
                sigma_clear badly on a drift-dominated map. It is measured from the residual of
                one sweep against the consensus: a registration error shifts a whole patch
                together, so it shows up as correlation that decays with cell separation.

Both passes run on device. The second needs the consensus map, so the frames are replayed.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import warp as wp

from calib import bagio
from calib.mapbuild import MapAccumulator

N_RANGE_BINS = 12
RANGE_BIN_M = 0.5
N_LAG_BINS = 24  # in cells; at 0.08 m that reaches 1.9 m, past the robot's own footprint


@wp.kernel
def sigma_by_range_kernel(
    map_sum: wp.array2d(dtype=wp.float32),
    map_sumsq: wp.array2d(dtype=wp.float32),
    map_nobs: wp.array2d(dtype=wp.float32),
    map_rng: wp.array2d(dtype=wp.float32),
    min_obs: wp.float32,
    bin_m: wp.float32,
    var_sum: wp.array(dtype=wp.float32),
    var_cnt: wp.array(dtype=wp.float32),
):
    """Accumulate each cell's across-sweep variance into a bin keyed by its mean observation range."""
    r, c = wp.tid()
    n = map_nobs[r, c]
    if n < min_obs:
        return
    mean = map_sum[r, c] / n
    var = map_sumsq[r, c] / n - mean * mean
    if var < 0.0:  # catastrophic cancellation on a near-constant cell
        var = 0.0
    var = var * n / (n - 1.0)  # unbiased
    b = int((map_rng[r, c] / n) / bin_m)
    if b < 0 or b >= var_sum.shape[0]:
        return
    wp.atomic_add(var_sum, b, var)
    wp.atomic_add(var_cnt, b, 1.0)


@wp.kernel
def frame_residual_kernel(
    frame_max: wp.array2d(dtype=wp.float32),
    frame_sum: wp.array2d(dtype=wp.float32),
    frame_cnt: wp.array2d(dtype=wp.float32),
    map_sum: wp.array2d(dtype=wp.float32),
    map_nobs: wp.array2d(dtype=wp.float32),
    min_points: wp.float32,
    min_obs: wp.float32,
    use_max: wp.int32,
    resid: wp.array2d(dtype=wp.float32),
    valid: wp.array2d(dtype=wp.float32),
):
    """This sweep's estimate of each cell minus the consensus. Invalid cells carry zero weight."""
    r, c = wp.tid()
    resid[r, c] = 0.0
    valid[r, c] = 0.0
    n = frame_cnt[r, c]
    k = map_nobs[r, c]
    if n < min_points or k < min_obs:
        return
    if use_max != 0:
        h = frame_max[r, c]
    else:
        h = frame_sum[r, c] / n
    resid[r, c] = h - map_sum[r, c] / k
    valid[r, c] = 1.0


@wp.kernel
def lag_correlation_kernel(
    resid: wp.array2d(dtype=wp.float32),
    valid: wp.array2d(dtype=wp.float32),
    prod_sum: wp.array(dtype=wp.float32),
    prod_cnt: wp.array(dtype=wp.float32),
):
    """Accumulate E[r(x) r(x+d)] into bins of lag |d| in cells, over axis-aligned offsets.

    Lag 0 gives E[r^2], so the ratio of each bin to bin 0 is the correlation coefficient. Only
    +x and +y offsets are walked; the field is symmetric, so the negative directions add nothing
    but work.
    """
    r, c = wp.tid()
    if valid[r, c] < 0.5:
        return
    a = resid[r, c]
    ny = resid.shape[0]
    nx = resid.shape[1]
    for d in range(prod_sum.shape[0]):
        if c + d < nx and valid[r, c + d] > 0.5:
            wp.atomic_add(prod_sum, d, a * resid[r, c + d])
            wp.atomic_add(prod_cnt, d, 1.0)
        if d > 0 and r + d < ny and valid[r + d, c] > 0.5:
            wp.atomic_add(prod_sum, d, a * resid[r + d, c])
            wp.atomic_add(prod_cnt, d, 1.0)


def run(bag: str, stat: str = "mean", min_obs: int = 4, root: str = "bags") -> dict:
    path = bagio.bag_path(bag, root)
    odom = bagio.read_odometry(path)
    frames = bagio.read_frames(path, odom)
    if not frames:
        raise RuntimeError(f"{bag}: no usable frames")

    cell = 0.08
    pad = 16.0  # wide, unlike Probe A: this probe wants the far field, where sigma is largest
    xmin = float(odom.xyz[:, 0].min() - pad)
    ymin = float(odom.xyz[:, 1].min() - pad)
    nx = int(np.ceil((odom.xyz[:, 0].max() + pad - xmin) / cell))
    ny = int(np.ceil((odom.xyz[:, 1].max() + pad - ymin) / cell))
    device = wp.get_device()

    acc = MapAccumulator(ny, nx, cell, xmin, ymin, device=device, stat=stat)
    for f in frames:
        acc.add_frame(f.points, f.sensor_xyz)

    var_sum = wp.zeros(N_RANGE_BINS, dtype=wp.float32, device=device)
    var_cnt = wp.zeros(N_RANGE_BINS, dtype=wp.float32, device=device)
    wp.launch(
        sigma_by_range_kernel,
        dim=(ny, nx),
        inputs=[acc.map_sum, acc.map_sumsq, acc.map_nobs, acc.map_rng, float(min_obs), RANGE_BIN_M],
        outputs=[var_sum, var_cnt],
        device=device,
    )
    vs, vc = var_sum.numpy(), var_cnt.numpy()
    sigma_range = [
        {
            "range_m": (b + 0.5) * RANGE_BIN_M,
            "n_cells": int(vc[b]),
            "sigma_m": float(np.sqrt(vs[b] / vc[b])) if vc[b] > 0 else None,
        }
        for b in range(N_RANGE_BINS)
    ]

    resid = wp.zeros((ny, nx), dtype=wp.float32, device=device)
    valid = wp.zeros((ny, nx), dtype=wp.float32, device=device)
    prod_sum = wp.zeros(N_LAG_BINS, dtype=wp.float32, device=device)
    prod_cnt = wp.zeros(N_LAG_BINS, dtype=wp.float32, device=device)
    for f in frames:
        acc.add_frame(f.points, f.sensor_xyz)  # repopulates frame_* for this sweep
        wp.launch(
            frame_residual_kernel,
            dim=(ny, nx),
            inputs=[
                acc.frame_max,
                acc.frame_sum,
                acc.frame_cnt,
                acc.map_sum,
                acc.map_nobs,
                acc.min_points,
                float(min_obs),
                acc.use_max,
            ],
            outputs=[resid, valid],
            device=device,
        )
        wp.launch(
            lag_correlation_kernel,
            dim=(ny, nx),
            inputs=[resid, valid],
            outputs=[prod_sum, prod_cnt],
            device=device,
        )
    ps, pc = prod_sum.numpy(), prod_cnt.numpy()
    cov = np.where(pc > 0, ps / np.maximum(pc, 1.0), np.nan)
    corr = cov / cov[0] if cov[0] > 0 else cov * np.nan
    lags = [
        {"lag_m": d * cell, "n_pairs": int(pc[d]), "corr": float(corr[d])}
        for d in range(N_LAG_BINS)
    ]
    # Correlation length: the lag at which the correlation first falls below 1/e.
    below = [d for d in range(1, N_LAG_BINS) if np.isfinite(corr[d]) and corr[d] < np.exp(-1.0)]
    L = below[0] * cell if below else None

    return {
        "bag": bag,
        "stat": stat,
        "min_obs": min_obs,
        "n_frames": len(frames),
        "grid": [ny, nx],
        "sigma_by_range": sigma_range,
        "lag_correlation": lags,
        "L_1_over_e_m": L,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bags", nargs="+")
    ap.add_argument("--stat", default="mean", choices=("max", "mean"))
    ap.add_argument("--min-obs", type=int, default=4)
    ap.add_argument("--out", default="studies/out/calib")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results = []
    for b in args.bags:
        r = run(b, args.stat, args.min_obs)
        results.append(r)
        sr = [x for x in r["sigma_by_range"] if x["sigma_m"] is not None and x["n_cells"] > 50]
        head = "  ".join(f"{x['range_m']:.1f}m:{x['sigma_m'] * 100:.1f}cm" for x in sr[:6])
        print(f"{b:11s} L={r['L_1_over_e_m']}  sigma {head}")
    dst = os.path.join(args.out, f"probe_sigma_{args.stat}.json")
    with open(dst, "w") as fh:
        json.dump(results, fh, indent=1)
    print("->", dst)


if __name__ == "__main__":
    main()
