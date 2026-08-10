"""Fit and write `belief_model.npz` + `rho_measured.npy` -- the measured-belief foundation every
`realistic_sigma.py` / `plot_paper.py` number downstream of HANDOFF.md 10.3 depends on.

    .venv/bin/python -m studies.sensing.fit_belief_model --smoke --realizations 8
    .venv/bin/python -m studies.sensing.fit_belief_model --write               # full-scale refit

These two artifacts were fitted ad hoc in a past session and never got a committed producer
(`RERUNS.md` "Known still-unfixed"). This is that producer, rebuilt from the spec in
HANDOFF.md 10.3 items 5-6 and reverse-engineered from the two places that read the files:
`realistic_sigma.BeliefModel` (every key/shape) and `plot_paper.fig_*` (the raw `rho_measured.npy`
load). Nothing here invents new fitting math -- the plane/residual split mirrors
`belief_sweep.fit`, the autocorrelation reuses `lidar_belief.autocorrelation_2d`, and the
rank-M PSD projection stays where it already lives, applied by the CONSUMER
(`BeliefModel.__init__` calls `clark_conv.separable_terms` on load) -- this script only has to
hand it a valid stationary kernel.

THE MODEL BEING FIT. Sigma = a rank-3 plane (a scan's pose error shifts and TILTS every point in
it together, so it never decorrelates -- no stationary kernel can express that) plus a compact
stationary kernel for what is left. Per HANDOFF 10.3.6: plane 30% of the (spatial) variance,
stationary 70%; sigma_offset ~1.4 cm, tilt sigma ~0.005 rad/m; residual decorrelating within
~3 cells (rho +0.022 @ 1 m, -0.004 @ 2 m). This script's job is to reproduce that fit from the
FIXED `lidar_belief.py` (body-frame tilt, cell-center `_ground`) so the refit is directionally
comparable but not identically numbered -- see RERUNS.md "Fix 4".

CALIBRATION SCENARIO. One fractal terrain (beta=1.8, the default), one straight 40-pose traverse
-- the same scenario `lidar_belief.run` and `belief_sweep.fit`'s "1.8"/"straight" row use, and
the one HANDOFF's caveat on the original fit names explicitly ("fitted to ONE terrain and ONE
straight traverse"). `belief_sweep.py` shows the fit moves only modestly across terrain/trajectory
axes, so this one scenario is deliberate, not an oversight.

SAFETY. `--write` is required to touch the committed `studies/out/bench/` artifacts; without it
(the default, and always under `--smoke`) output goes to `--out-dir` (a scratch directory) so a
test run can never clobber the committed fit. Both files are written to a temp name in the target
directory and `os.replace`d into place, so a crash mid-write cannot corrupt the committed ones
either.
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import warp as wp

from ..adjoint.generalise import fractal_terrain
from ..bench.ranking import CELL
from ..bench.ranking import OUT
from ..bench.realistic_sigma import BeliefModel
from .belief_sweep import trajectory
from .lidar_belief import autocorrelation_2d
from .lidar_belief import LidarSim
from .lidar_belief import NoiseParams
from .lidar_belief import simulate_belief
from helhest.perception.heightmap import HeightMapBuilder

N = 90  # grid cells per side of the calibration scene, matching lidar_belief.run/belief_sweep.fit
FULL_REALIZATIONS_DEFAULT = 32  # matches lidar_belief.py's own --realizations default
SMOKE_REALIZATIONS_DEFAULT = 8
MAX_LAG_CELLS = 25  # 2.5 m half-window: residual decorrelates within ~3 cells (HANDOFF 10.3.6),
# this leaves margin while still covering the 1 m / 2 m points HANDOFF reports

# HANDOFF.md 10.3.6's numbers, printed alongside the refit for comparison -- NOT the target of
# an assertion, since the `_ground`/tilt-frame fix (RERUNS.md fix 4) is expected to move them.
HANDOFF_SIGMA_OFFSET_CM = 1.4
HANDOFF_TILT_RAD_PER_M = 0.005
HANDOFF_PLANE_SHARE = 0.30
HANDOFF_RHO_1M = 0.022
HANDOFF_RHO_2M = -0.004


def simulate_realizations(
    seed: int, realizations: int, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drive the calibration scenario through the simulated sensor `realizations` times.

    The ray-marching and rasterization run on-device inside `simulate_belief`/`HeightMapBuilder`
    (Warp kernels, reused as-is). What comes back here is the small [R, 90, 90] map ensemble --
    the plane/kernel regression below is ordinary host-side analysis of that ensemble, the same
    trade `realistic_sigma.py` and `belief_sweep.py` declare for themselves.
    """
    truth = fractal_terrain(N, N, CELL, seed=seed)
    traj = trajectory("straight")
    sim = LidarSim(truth, 0.0, 0.0, CELL, device)
    builder = HeightMapBuilder(CELL, (0.0, N * CELL, 0.0, N * CELL), device=wp.get_device(device))
    p = NoiseParams()
    maps, counts = [], []
    for r in range(realizations):
        m, c = simulate_belief(sim, traj, p, seed * 1000 + r, builder)
        maps.append(m)
        counts.append(c)
    maps, counts = np.stack(maps), np.stack(counts)
    obs = np.isfinite(maps).all(axis=0) & (counts > 0).all(axis=0)
    return maps, counts, obs


def fit_plane(
    maps: np.ndarray, obs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Per-scan shift+tilt regression against the ensemble mean, mirroring `belief_sweep.fit`.

    A scan's pose error shifts and tilts every point in it together, so this 3-term plane
    (offset, x-tilt, y-tilt) fit to each realization's deviation from the mean IS the pose-error
    structure the model is meant to isolate. Returns the per-realization coefficients [R, 3],
    the residual fields [R, ny, nx] (zero outside `obs`), the basis recentring (x_center,
    y_center), and the plane's share of the spatial variance.
    """
    ny, nx = obs.shape
    mean_map = np.nanmean(maps, axis=0)
    gy, gx = np.mgrid[0:ny, 0:nx]
    x_center = float(gx[obs].mean() * CELL)
    y_center = float(gy[obs].mean() * CELL)
    basis = np.c_[np.ones(int(obs.sum())), (gx * CELL - x_center)[obs], (gy * CELL - y_center)[obs]]
    coefs, residuals, plane_var, resid_var = [], [], [], []
    for m in maps:
        e = (m - mean_map)[obs]
        c3, *_ = np.linalg.lstsq(basis, e, rcond=None)
        plane_vals = basis @ c3
        resid_vals = e - plane_vals
        coefs.append(c3)
        f = np.zeros((ny, nx))
        f[obs] = resid_vals
        residuals.append(f)
        plane_var.append(np.var(plane_vals))
        resid_var.append(np.var(resid_vals))
    share = float(np.mean(plane_var) / (np.mean(plane_var) + np.mean(resid_var)))
    return np.array(coefs), np.stack(residuals), x_center, y_center, share


def fit_stationary_kernel(
    residuals: np.ndarray, maps: np.ndarray, mean_map: np.ndarray, obs: np.ndarray, max_lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """The residual's spatial autocorrelation (`rho_stationary`, plane removed) and the raw
    per-realization anomaly's (`rho_measured`, plane still in it -- what `plot_paper` calls
    "measured, total"), each averaged over realizations for stability. `belief_sweep.fit`'s quick
    check uses only one realization; this is the committed fit, so it uses all of them.
    """
    anomaly = np.where(obs[None], maps - mean_map[None], 0.0)
    rho_stationary = np.mean([autocorrelation_2d(f, obs, max_lag) for f in residuals], axis=0)
    rho_measured = np.mean([autocorrelation_2d(a, obs, max_lag) for a in anomaly], axis=0)
    return rho_stationary, rho_measured


def fit(seed: int, realizations: int, device: str) -> dict[str, object]:
    """Run the calibration scenario and fit Sigma = rank-3 plane + compact stationary kernel."""
    maps, _counts, obs = simulate_realizations(seed, realizations, device)
    if obs.sum() < 500:
        raise RuntimeError(
            f"only {int(obs.sum())} observed cells -- calibration scene is under-covered"
        )
    mean_map = np.nanmean(maps, axis=0)
    coefs, residuals, x_center, y_center, share = fit_plane(maps, obs)
    rho_stationary, rho_measured = fit_stationary_kernel(
        residuals, maps, mean_map, obs, MAX_LAG_CELLS
    )
    plane_cov = np.cov(coefs.T)
    return {
        "maps": maps,
        "obs": obs,
        "mean_map": mean_map,
        "plane_cov": plane_cov,
        "plane_share": share,
        "rho_stationary": rho_stationary,
        "rho_measured": rho_measured,
        "x_center": x_center,
        "y_center": y_center,
    }


def print_summary(result: dict[str, object], realizations: int) -> None:
    plane_cov = result["plane_cov"]
    rho_s = result["rho_stationary"]
    L = rho_s.shape[0] // 2
    n1, n2 = round(1.0 / CELL), round(2.0 / CELL)
    sigma_offset_cm = math.sqrt(plane_cov[0, 0]) * 100
    tilt_x = math.sqrt(plane_cov[1, 1])
    tilt_y = math.sqrt(plane_cov[2, 2])
    share = result["plane_share"]
    rho_1m = rho_s[L, L + n1]
    rho_2m = rho_s[L, L + n2]
    print(f"\n=== fitted belief model ({realizations} realizations) ===")
    print(f"{'':22s}{'fitted':>12s}{'HANDOFF 10.3.6':>18s}")
    print(f"{'sigma_offset [cm]':22s}{sigma_offset_cm:12.2f}{HANDOFF_SIGMA_OFFSET_CM:18.2f}")
    print(f"{'tilt_x sigma [rad/m]':22s}{tilt_x:12.4f}{HANDOFF_TILT_RAD_PER_M:18.4f}")
    print(f"{'tilt_y sigma [rad/m]':22s}{tilt_y:12.4f}{HANDOFF_TILT_RAD_PER_M:18.4f}")
    print(f"{'plane share':22s}{share:12.2f}{HANDOFF_PLANE_SHARE:18.2f}")
    print(f"{'stationary share':22s}{1.0 - share:12.2f}{1.0 - HANDOFF_PLANE_SHARE:18.2f}")
    print(f"{'residual rho @ 1 m':22s}{rho_1m:12.3f}{HANDOFF_RHO_1M:18.3f}")
    print(f"{'residual rho @ 2 m':22s}{rho_2m:12.3f}{HANDOFF_RHO_2M:18.3f}")


def _write_atomic_npz(path: Path, **arrays: np.ndarray | float) -> None:
    tmp = path.with_name(f".{path.stem}.tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _write_atomic_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(f".{path.stem}.tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def verify(
    npz_path: Path, npy_path: Path, result: dict[str, object], tol: float
) -> dict[str, object]:
    """Load both artifacts back through their exact consumers and check the model's predicted
    variance of a simple linear functional against the EMPIRICAL variance across the maps just
    simulated -- not the model's own synthetic generator (that would only test self-consistency,
    which is what the first, retracted, single-kernel fit passed and was still wrong).

    Reported error is noisy at smoke scale (`realizations` draws give a relative std on the
    empirical variance of roughly sqrt(2/(realizations-1)) -- 53% at n=8), so `tol` scales with
    `realizations` rather than using one fixed threshold for both smoke and full-scale runs.
    """
    model = BeliefModel(npz_path)  # raises if any key/shape realistic_sigma.py needs is wrong
    raw = np.load(npz_path)
    for key in ("plane_cov", "plane_share", "rho_stationary", "x_center", "y_center"):
        if key not in raw:
            raise KeyError(f"{key!r} missing from {npz_path} -- BeliefModel would not load this")
    rho_measured = np.load(npy_path)  # the exact load plot_paper.fig_realistic_belief does

    maps, obs, mean_map = result["maps"], result["obs"], result["mean_map"]
    # `mean_map` is NaN where every realization missed a cell; zero it under `obs` masking below
    # rather than at the source, since `fit_stationary_kernel` needs the NaN-free `anomaly` it
    # already built the same way (`np.where(obs, ..., 0.0)`) -- 0 * NaN is NaN, not 0, so the
    # multiply-by-g below must happen AFTER masking, not before.
    diffs = np.where(obs[None], maps - np.nan_to_num(mean_map, nan=0.0)[None], 0.0)
    # `BeliefModel.variance_of` predicts Var of a functional of the UNIT-sigma correlation field
    # -- every caller (`_clark_moments`, `_fosm_variance`) folds the scene's own per-cell sigma
    # into the field it passes. belief_model.npz carries no sigma (by design -- the pre-reg fixes
    # sigma and moves only the correlation), so the check must fold in THIS scenario's empirical
    # per-cell sigma the same way, or it compares two different physical scales.
    sigma_emp = np.nan_to_num(np.nanstd(maps, axis=0), nan=0.0)
    ny, nx = obs.shape
    rng = np.random.default_rng(1)
    errs = []
    for _ in range(24):
        g = np.zeros((ny, nx))
        iy = rng.integers(8, ny - 8, 200)
        ix = rng.integers(8, nx - 8, 200)
        np.add.at(g, (iy, ix), rng.normal(0, 1, 200))
        g *= obs  # a functional of never-observed cells is not something the sensing measured
        pred = model.variance_of(g * sigma_emp, ny, nx, 0.0, 0.0)
        emp = float(np.var([float((g * d).sum()) for d in diffs]))
        errs.append(abs(pred - emp) / max(emp, 1e-12))
    median_err = float(np.median(errs))
    return {
        "median_var_rel_error": median_err,
        "max_var_rel_error": float(np.max(errs)),
        "tolerance": tol,
        "passed": median_err < tol,
        "rho_measured_shape": rho_measured.shape,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--realizations",
        type=int,
        default=None,
        help=f"default {FULL_REALIZATIONS_DEFAULT} ({SMOKE_REALIZATIONS_DEFAULT} with --smoke)",
    )
    ap.add_argument("--seed", type=int, default=0, help="terrain + sensing seed")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="few realizations; always writes to --out-dir, never the committed artifacts",
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="write studies/out/bench/{belief_model.npz,rho_measured.npy}",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output dir for a non-committing run (default: a system temp dir)",
    )
    args = ap.parse_args()
    if args.smoke and args.write:
        ap.error(
            "--smoke and --write are mutually exclusive -- smoke fits must never overwrite "
            "the committed belief_model.npz / rho_measured.npy"
        )

    realizations = args.realizations
    if realizations is None:
        realizations = SMOKE_REALIZATIONS_DEFAULT if args.smoke else FULL_REALIZATIONS_DEFAULT
    out_dir = (
        OUT if args.write else (args.out_dir or Path(tempfile.gettempdir()) / "fit_belief_model")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    wp.init()
    result = fit(args.seed, realizations, args.device)
    print_summary(result, realizations)

    npz_path, npy_path = out_dir / "belief_model.npz", out_dir / "rho_measured.npy"
    _write_atomic_npz(
        npz_path,
        plane_cov=result["plane_cov"],
        plane_share=result["plane_share"],
        rho_stationary=result["rho_stationary"],
        x_center=result["x_center"],
        y_center=result["y_center"],
    )
    _write_atomic_npy(npy_path, result["rho_measured"])
    print(f"\nwrote {npz_path}")
    print(f"wrote {npy_path}")

    tol = max(0.05, 3.0 * math.sqrt(2.0 / max(realizations - 1, 1)))
    check = verify(npz_path, npy_path, result, tol)
    print("\n=== end-to-end check: model-predicted vs. empirical Var of a linear functional ===")
    print(
        f"  median rel error {check['median_var_rel_error']:.1%}  "
        f"max {check['max_var_rel_error']:.1%}  (tolerance {tol:.1%} @ n={realizations})"
    )
    print("  PASSED" if check["passed"] else "  FAILED")
    if not check["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
