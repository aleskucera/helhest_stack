"""Persist the separable-kernel accuracy numbers 04_method.tex quotes -- measured, not invented.

    .venv/bin/python -m studies.bench.separable_report              # writes separable_report.json

`04_method.tex` (~lines 144-155) claims: the measured correlation is not separable (worst-case
error 0.48), a single separable term gets Var[J] wrong by 60%, and a PSD-projected rank-5
expansion holds it to 0.57%. Those numbers came from an ad hoc HANDOFF.md 10.3.5 session and were
never written to a committed artifact -- this script is that producer, and `belief_model.npz` /
`rho_measured.npy` were regenerated on 2026-08-10 under fixed sensing code, so the numbers WILL
move from the ones currently in the paper.

TWO DIFFERENT "measured kernel" OBJECTS, ON PURPOSE. `rho_measured.npy` is the raw 2-D
autocorrelation of a scan's height error -- plane (never decays) and stationary residual both
still in it -- so it is what "worst-case separability error" means: how far is the RAW
measurement from any single separable term. `belief_model.npz`'s `rho_stationary` is the plane
already removed (`fit_belief_model.fit_plane`), which is what `realistic_sigma.BeliefModel`
and its `RANK = 5` constant actually consume downstream (04_method's eq:cov treats the plane
as exact and only the stationary part needs a separable approximation). So (a) below reports
the raw kernel's fit quality, and (b) -- the Var[J] numbers the paper actually quotes -- uses
`rho_stationary`, matching what `BeliefModel.variance_of` does in production.

PROTOCOL FOR (b), reproducing HANDOFF 10.3.5's measurement. `plan_moments_conv(...,
return_patch=True)` gives back the scattered field G each real plan corridor's Var[J] convolves
-- reusing the SAME `hybrid`/`all` scenes and Harness the rest of the study ranks plans on
(`ranking.build_case`, `cuda:0`), not a synthetic field. Ground truth is `_dense_quadratic`,
<G, K*G> for the FULL rho_stationary lag table via one 2-D FFT convolution -- no rank reduction,
no assumed profile. Three things are checked against it per plan: the PRODUCTION single term
(the assumed `rho1_table(CORR_LEN, CELL)`, exactly what `plan_moments_conv`'s own returned Var
already used), the SVD-best single term of the measured kernel (`separable_terms(rho_s, 1)`,
Bochner-PSD-projected same as the rest), and rank M in {2, 3, 5} likewise PSD-projected.
"""

from __future__ import annotations

import json

import numpy as np
import warp as wp

from ..adjoint.harness import Harness
from .clark import rho1_table
from .clark_conv import kernel_is_psd
from .clark_conv import plan_moments_conv
from .clark_conv import separable_quadratic_multi
from .clark_conv import separable_terms
from .ranking import build_case
from .ranking import CELL
from .ranking import N_PLANS
from .ranking import OUT
from .risk import CORR_LEN
from helhest.engine import RobotParams

RANKS = (1, 2, 3, 5)  # 5 matches realistic_sigma.RANK -- the one the rest of the paper deploys
N_SEEDS = 8  # matches clark_conv.run()'s own default: 8 scenes x 16 plans = 128 corridors
BIG_LAG_THRESHOLD = 0.05  # lidar_belief.py's own separability-error convention, reused so the
# two numbers are computed the same way


def _rank1_fit_error(kernel2d: np.ndarray) -> float:
    """Worst-case |kernel - best rank-1 separable fit| where the kernel itself is not already
    negligible (mirrors `lidar_belief.py`'s own `separability_max_abs_err`, applied here to the
    COMMITTED artifact instead of a fresh simulator run)."""
    (a, b) = separable_terms(kernel2d, 1, enforce_psd=False)[0]
    fit = np.outer(a, b)
    big = np.abs(kernel2d) > BIG_LAG_THRESHOLD
    return float(np.abs(kernel2d - fit)[big].max()) if big.any() else float("nan")


def _dense_quadratic(field: np.ndarray, kernel2d: np.ndarray) -> float:
    """<G, K * G> for the FULL (non-separable, non-truncated) kernel table, by one 2-D FFT
    convolution -- the "no approximation" ground truth the separable/rank-M forms below are
    checked against. A stationary autocorrelation is always symmetric under (dy,dx) -> (-dy,-dx)
    (verified to 1e-16 on the committed `rho_stationary`), so using `kernel2d` as-is, the same
    way `_separable_quadratic` uses its 1-D symmetric kernel, reproduces the 'same'-mode
    correlation exactly."""
    ny, nx = field.shape
    ky, kx = kernel2d.shape
    fy, fx = ny + ky - 1, nx + kx - 1
    conv_full = np.fft.irfft2(
        np.fft.rfft2(field, s=(fy, fx)) * np.fft.rfft2(kernel2d, s=(fy, fx)), s=(fy, fx)
    )
    y0, x0 = (ky - 1) // 2, (kx - 1) // 2
    same = conv_full[y0 : y0 + ny, x0 : x0 + nx]
    return float((field * same).sum())


def _pad_to_at_least(field: np.ndarray, min_shape: tuple[int, int]) -> np.ndarray:
    """Zero-pad `field` so both axes are at least `min_shape` -- some plan corridors are narrower
    than the measured kernel's 2.5 m window (51 cells), and `np.convolve(..., mode="same")`
    silently returns a kernel-length output (not field-length) when the kernel is the longer
    array, which would corrupt the elementwise `field * conv` step downstream. G is zero outside
    its true support, so padding it with more zero cells changes nothing physical."""
    pads = [(max(0, m - n) // 2, max(0, m - n) - max(0, m - n) // 2)
            for n, m in zip(field.shape, min_shape)]
    return np.pad(field, pads) if any(p != (0, 0) for p in pads) else field


def _plan_patches(
    n_seeds: int, device: str, rho1: np.ndarray
) -> tuple[list[np.ndarray], np.ndarray]:
    """G-fields for every plan of `n_seeds` `hybrid`/`all` scenes -- the actual corridors the
    estimator ranks, not an invented field -- plus the Var[J] `plan_moments_conv` itself already
    computes for each one under the PRODUCTION assumed kernel (its ordinary return value, not
    recomputed here). The within-footprint fold (candidate-to-candidate correlation inside one
    node's K cells) still uses the assumed `rho1`, matching production; only the plan-level Var
    convolution this script performs afterwards, on the patches returned here, varies the kernel.
    """
    rp = RobotParams()
    patches, var_assumed = [], []
    for seed in range(n_seeds):
        scene, _, _, _, sigma, poses, omega, _ = build_case(seed, "hybrid", "all")
        belief = scene.elevation.astype(np.float32)
        h = Harness(scene, poses, omega, device=device)
        h.forward(dilate=True)
        controlled = h.sim.controlled.numpy()
        del h
        geo = (scene.origin_x, scene.origin_y, CELL)
        for k in range(N_PLANS):
            _, var_j, patch, _, _ = plan_moments_conv(
                belief, sigma, controlled[:, k, :], rp, *geo, rho1, return_patch=True
            )
            patches.append(patch)
            var_assumed.append(var_j)
    return patches, np.array(var_assumed)


def run(device: str, n_seeds: int) -> dict:
    rho_measured = np.load(OUT / "rho_measured.npy")
    belief_model = np.load(OUT / "belief_model.npz")
    rho_s = belief_model["rho_stationary"]

    # --- (a) single-term separability error, on the RAW measured kernel --------------------
    sep_err_raw = _rank1_fit_error(rho_measured)
    sep_err_stationary = _rank1_fit_error(rho_s)  # reported for context, not the paper's number

    # --- (b) Var[J] error of a linear functional, on real plan corridors --------------------
    rho1_assumed = rho1_table(CORR_LEN, CELL)
    patches, var_assumed = _plan_patches(n_seeds, device, rho1_assumed)
    patches = [_pad_to_at_least(g, rho_s.shape) for g in patches]

    var_true = np.array([_dense_quadratic(g, rho_s) for g in patches])
    err_assumed = np.abs(var_assumed - var_true) / np.maximum(var_true, 1e-12)

    rank_errors: dict[int, np.ndarray] = {}
    psd_check: dict[int, dict] = {}
    for m in RANKS:
        terms = separable_terms(rho_s, m, enforce_psd=True)
        var_m = np.array([separable_quadratic_multi(g, terms) for g in patches])
        rank_errors[m] = np.abs(var_m - var_true) / np.maximum(var_true, 1e-12)
        raw_terms = separable_terms(rho_s, m, enforce_psd=False)
        psd_check[m] = {
            "before_projection": kernel_is_psd(rho_s, raw_terms),
            "after_projection": kernel_is_psd(rho_s, terms),
        }

    out = {
        "n_seeds": n_seeds,
        "n_plans": len(patches),
        "separability": {
            "raw_measured_kernel_worst_case_abs_err": sep_err_raw,
            "stationary_kernel_worst_case_abs_err": sep_err_stationary,
            "threshold": BIG_LAG_THRESHOLD,
            "note": "worst-case |kernel - best rank-1 separable fit| over lags where the "
            "kernel itself exceeds `threshold`; raw = rho_measured.npy (plane still in it), "
            "stationary = belief_model.npz rho_stationary (plane removed, context only)",
        },
        "var_error": {
            "assumed_single_term": {
                "median": float(np.median(err_assumed)),
                "mean": float(np.mean(err_assumed)),
                "max": float(np.max(err_assumed)),
                "note": "the PRODUCTION assumption rho1_table(CORR_LEN, CELL) outer rho1_table "
                "-- what plan_moments_conv's own Var[J] already uses -- checked against the "
                "dense (no-approximation) quadratic form under the measured rho_stationary",
            },
            "rank_m_psd_projected": {
                str(m): {
                    "median": float(np.median(rank_errors[m])),
                    "mean": float(np.mean(rank_errors[m])),
                    "max": float(np.max(rank_errors[m])),
                }
                for m in RANKS
            },
        },
        "psd_projection": {
            str(m): {
                "min_spectrum_before": psd_check[m]["before_projection"]["min_spectrum"],
                "min_spectrum_after": psd_check[m]["after_projection"]["min_spectrum"],
                "psd_after": psd_check[m]["after_projection"]["psd"],
            }
            for m in RANKS
        },
    }
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()
    wp.init()
    print("=== separable-kernel accuracy (measured rho_stationary, real plan corridors) ===")
    out = run(args.device, args.seeds)

    sep = out["separability"]
    print(f"\n  single-term separability, worst-case |err| (raw measured kernel): "
          f"{sep['raw_measured_kernel_worst_case_abs_err']:.3f}")
    print(f"  single-term separability, worst-case |err| (stationary kernel, context): "
          f"{sep['stationary_kernel_worst_case_abs_err']:.3f}")

    a = out["var_error"]["assumed_single_term"]
    print(f"\n  Var[J] error, assumed single term (production rho1_table): "
          f"median {a['median']:.1%}  max {a['max']:.1%}  (n={out['n_plans']} plans)")
    for m in RANKS:
        r = out["var_error"]["rank_m_psd_projected"][str(m)]
        psd = out["psd_projection"][str(m)]
        print(f"  Var[J] error, rank {m} PSD-projected:                 "
              f"median {r['median']:.2%}  max {r['max']:.2%}   "
              f"min spectrum {psd['min_spectrum_before']:.1e} -> {psd['min_spectrum_after']:.1e}")

    path = OUT / "separable_report.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
