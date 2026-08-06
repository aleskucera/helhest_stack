"""Curb-approach direction under position uncertainty — the decisive scenario test.

The scenario (posed as the motivating case for gradient-weighted uncertainty): a curb whose
position along the approach axis is uncertain (sigma_s ~ 5 cm, localisation-scale). Candidate
plans cross it at increasing obliquity. Physically, head-on is safer: a curb taken sideways
converts the step into ROLL, and the robot's envelope is asymmetric (max_roll 15 deg vs
max_pitch_up 25 deg; roll_cost_weight 1.0 vs pitch 0.5 — "attack slopes head-on").

The question is NOT whether head-on is safer — it is WHERE that preference comes from and
WHICH estimator can see it. Decomposition, per candidate angle:

  J(belief)          cost on the mean map alone (no uncertainty term at all)
  E[J], sd[J], CVaR  truth: the ensemble of sharp curbs at shifted positions
  |dJ/ds| * sigma    first-order FOSM through the ONE correlated mode (the shift) — the most
                     generous gradient treatment possible: smooth belief, exact correlation,
                     no per-cell diagonal approximation
  0.5 J'' sigma^2    second-order predicted mean shift (Jensen gap)
  sum_t sigma_t      the free per-timestep exposure heuristic (RMS over wheels)

Stencils at 1 mm / 1 cm / 5 cm separate "the local derivative" from "a secret two-point
sample": if the mm and cm stencils disagree, the function is not locally polynomial at the
scale sigma lives at, and no derivative evaluated AT the belief can summarise the ensemble.

The batch dimension carries the ensemble: slice b = one terrain hypothesis (sharp curb at
shift s_b, or the belief / stencil maps), identical start pose and control across slices.
One launch per candidate angle therefore yields truth + belief + all stencils together.
"""
from __future__ import annotations

import numpy as np

from helhest.engine import RobotParams

from .harness import Harness
from .scene import Scene

# --- geometry -----------------------------------------------------------------------------
CELL = 0.05
XLIM = (-1.5, 5.5)
YLIM = (-4.5, 4.5)
CURB_X = 1.0  # nominal edge position; the edge line is x = CURB_X (normal along +x)
CURB_H = 0.12
MU = 0.6

SIGMA_S = 0.05  # [m] std of the edge position along its normal
# rasterisation smoothing: a cell averages the step over its footprint, so even a "sharp"
# curb carries cell-scale blur; the belief adds it in quadrature to the position spread
SIGMA_B = float(np.sqrt(SIGMA_S**2 + CELL**2 / 12.0))

N_SHIFTS = 53  # odd, so s = 0 is on the grid
SHIFT_SPAN = 3.0  # +- 3 sigma of edge-position hypotheses

ANGLES_DEG = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0)
APPROACH_DIST = 1.0  # [m] start this far (along the heading) before the nominal edge
WHEEL_OMEGA = 2.0  # [rad/s] -> v = 0.7 m/s, straight drive
N_STEPS = 60
DT = 0.1

# stencil half-widths for dJ/ds and J'' on the belief; 1 mm is "the honest local derivative",
# 5 cm ( = sigma) is secretly a two-point sample of the ensemble
STENCILS = (0.001, 0.01, 0.05)
CVAR_ALPHA = 0.25


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    from math import erf
    from math import sqrt

    return 0.5 * (1.0 + np.vectorize(erf)(np.asarray(z) / sqrt(2.0)))


def _shift_grid() -> tuple[np.ndarray, np.ndarray]:
    """Edge-position hypotheses and their normalised Gaussian weights."""
    s = np.linspace(-SHIFT_SPAN * SIGMA_S, SHIFT_SPAN * SIGMA_S, N_SHIFTS)
    w = np.exp(-0.5 * (s / SIGMA_S) ** 2)
    return s, w / w.sum()


def _sharp_profile(xc: np.ndarray, edge: float) -> np.ndarray:
    """Area-weighted rasterised step: what a cell-averaging mapper records of a sharp curb."""
    return CURB_H * np.clip((xc - edge) / CELL + 0.5, 0.0, 1.0)


def _belief_profile(xc: np.ndarray, edge: float) -> np.ndarray:
    """Mean map under the position ensemble: Gaussian-convolved step (smooth by construction,
    so the mm-scale stencil differentiates a genuinely smooth function — the gradient's best
    case)."""
    return CURB_H * _norm_cdf((xc - edge) / SIGMA_B)


def _sigma_profile(xc: np.ndarray) -> np.ndarray:
    """Per-cell std of the ensemble: h is CURB_H w.p. p and 0 w.p. 1-p."""
    p = _norm_cdf((xc - CURB_X) / SIGMA_B)
    return CURB_H * np.sqrt(p * (1.0 - p))


def _terrain_stack(xc: np.ndarray, ny: int) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    """[B, ny, nx] slices: the shift ensemble, then belief + stencil maps. Returns the stack,
    a name->slice index map for the non-ensemble slices, and the shift weights."""
    shifts, weights = _shift_grid()
    profiles = [_sharp_profile(xc, CURB_X + s) for s in shifts]
    idx: dict[str, int] = {}
    for eps in (0.0,) + tuple(e for pair in ((h, -h) for h in STENCILS) for e in pair):
        idx[f"belief{eps:+.3f}"] = len(profiles)
        profiles.append(_belief_profile(xc, CURB_X + eps))
    stack = np.repeat(np.stack(profiles)[:, None, :], ny, axis=1)
    return np.ascontiguousarray(stack, np.float32), idx, weights


def _candidates() -> np.ndarray:
    """Start poses [n_angles, 3]; every path crosses the nominal edge at (CURB_X, 0)."""
    poses = []
    for deg in ANGLES_DEG:
        th = np.radians(deg)
        poses.append([CURB_X - APPROACH_DIST * np.cos(th), -APPROACH_DIST * np.sin(th), th])
    return np.array(poses, np.float32)


def _tilt_cost(derived: np.ndarray, rp: RobotParams) -> np.ndarray:
    """[T+1, B, 3] -> [B]: the production-shaped graded tilt cost. Pitch is normalised by the
    sign-matched limit (nose-up = NEGATIVE pitch -> max_pitch_up)."""
    pitch, roll = derived[1:, :, 1], derived[1:, :, 2]
    pitch_lim = np.where(pitch < 0.0, rp.max_pitch_up, rp.max_pitch_down)
    per_step = rp.roll_cost_weight * (roll / rp.max_roll) ** 2 + rp.pitch_cost_weight * (
        pitch / pitch_lim
    ) ** 2
    return per_step.sum(axis=0)


def _sigma_exposure(controlled: np.ndarray, rp: RobotParams, xs: np.ndarray) -> float:
    """sum_t RMS-over-wheels of sigma at the wheel contacts, along the belief trajectory.
    The field is 1-D in x, so sampling is a 1-D interpolation at each wheel's world x."""
    sigma_x = _sigma_profile(xs)
    wheels = np.array([[0.0, rp.half_track], [0.0, -rp.half_track], [-rp.rear_offset, 0.0]])
    total = 0.0
    for x, y, yaw in controlled[1:]:
        c, s = np.cos(yaw), np.sin(yaw)
        wx = x + wheels[:, 0] * c - wheels[:, 1] * s
        total += float(np.sqrt(np.mean(np.interp(wx, xs, sigma_x) ** 2)))
    return total


def _weighted_cvar(costs: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    """Mean of the worst alpha-tail under the ensemble weights."""
    order = np.argsort(costs)[::-1]
    c, w = costs[order], weights[order]
    cum = np.cumsum(w)
    take = cum <= alpha
    take[np.searchsorted(cum, alpha)] = True  # partial last slice, kept whole (grid is fine)
    return float(np.sum(c[take] * w[take]) / np.sum(w[take]))


def _kendall(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    num = 0
    for i in range(n):
        for j in range(i + 1, n):
            num += int(np.sign((a[i] - a[j]) * (b[i] - b[j])))
    return num / (n * (n - 1) / 2)


def main() -> None:
    nx = int(round((XLIM[1] - XLIM[0]) / CELL)) + 1
    ny = int(round((YLIM[1] - YLIM[0]) / CELL)) + 1
    xc = XLIM[0] + (np.arange(nx) + 0.5) * CELL
    stack, idx, weights = _terrain_stack(xc, ny)
    batch = stack.shape[0]
    n_ens = N_SHIFTS

    rp = RobotParams()
    # the Scene handed to the Harness only fixes the grid + friction; every launch overwrites
    # the elevation slices with the stack above
    scene = Scene(
        elevation=np.zeros((ny, nx)),
        friction=np.full((ny, nx), MU),
        region=np.zeros((ny, nx), np.int8),
        cell=CELL,
        origin_x=XLIM[0],
        origin_y=YLIM[0],
    )
    poses = _candidates()
    omega = np.full((N_STEPS, batch, 3), WHEEL_OMEGA, np.float32)
    harness = Harness(scene, np.tile(poses[0], (batch, 1)), omega, dt=DT)

    import warp as wp

    stack_dev = wp.array(stack, dtype=wp.float32, device=harness.device)

    rows = []
    for pose in poses:
        harness.sim.start_pose.assign(np.ascontiguousarray(np.tile(pose, (batch, 1)), np.float32))
        wp.copy(harness.sim.elevation, stack_dev)
        harness._rollout(dilate=True)
        derived = harness.sim.derived.numpy()
        controlled = harness.sim.controlled.numpy()
        costs = _tilt_cost(derived, rp)

        ens = costs[:n_ens]
        e_j = float(np.sum(ens * weights))
        sd_j = float(np.sqrt(np.sum(weights * (ens - e_j) ** 2)))
        j0 = float(costs[idx["belief+0.000"]])

        row = {
            "E[J]": e_j,
            "sd[J]": sd_j,
            "CVaR": _weighted_cvar(ens, weights, CVAR_ALPHA),
            "J(belief)": j0,
            "sigma_sum": _sigma_exposure(controlled[:, idx["belief+0.000"]], rp, xc),
            # worst-tail roll, as a fraction of the tip limit: the interpretable safety number
            "peak_roll": float(
                np.max(np.abs(derived[1:, :n_ens, 2])) / rp.max_roll
            ),
        }
        for h in STENCILS:
            jp = float(costs[idx[f"belief{h:+.3f}"]])
            jm = float(costs[idx[f"belief{-h:+.3f}"]])
            slope = (jp - jm) / (2 * h)
            row[f"slope@{h * 1e3:.0f}mm"] = slope
            row[f"fosm_sd@{h * 1e3:.0f}mm"] = abs(slope) * SIGMA_S
            row[f"jensen@{h * 1e3:.0f}mm"] = 0.5 * ((jp - 2 * j0 + jm) / h**2) * SIGMA_S**2
        rows.append(row)
        # J(s) curve for the figure
        row["curve"] = ens.copy()

    shifts, _ = _shift_grid()
    _report(rows, shifts)
    _figure(rows, shifts)


def _report(rows: list[dict], shifts: np.ndarray) -> None:
    keys = [
        "E[J]",
        "CVaR",
        "sd[J]",
        "J(belief)",
        "sigma_sum",
        "peak_roll",
        "fosm_sd@1mm",
        "fosm_sd@10mm",
        "fosm_sd@50mm",
        "jensen@10mm",
        "jensen@50mm",
    ]
    print(f"{'angle':>6} " + " ".join(f"{k:>12}" for k in keys))
    for deg, row in zip(ANGLES_DEG, rows):
        print(f"{deg:6.0f} " + " ".join(f"{row[k]:12.3f}" for k in keys))
    print("\ntrue Jensen gap E[J]-J(belief) per angle:")
    print("  " + " ".join(f"{r['E[J]'] - r['J(belief)']:+.3f}" for r in rows))

    # the sharp curb at the nominal position is the middle ensemble slice: what a mode map
    # (no smearing, no uncertainty encoded at all) would score each candidate
    mid = len(shifts) // 2
    print("\nJ(sharp curb at s=0) per angle:")
    print("  " + " ".join(f"{r['curve'][mid]:.3f}" for r in rows))

    truth = np.array([r["E[J]"] for r in rows])
    print("\nKendall tau of each score against truth E[J] over the candidate set:")
    for k in ("J(belief)", "sigma_sum", "sd[J]", "fosm_sd@1mm", "fosm_sd@50mm"):
        score = np.array([r[k] for r in rows])
        print(f"  {k:>12}: {_kendall(score, truth):+.2f}")
    # the sharpest question: can each estimator rank the UNCERTAINTY INCREMENT (which
    # candidate is hurt most by not knowing where the curb is)?
    inc = np.array([r["E[J]"] - r["J(belief)"] for r in rows])
    print("\nKendall tau against the true increment E[J]-J(belief):")
    for k in ("sigma_sum", "fosm_sd@1mm", "fosm_sd@50mm", "jensen@10mm", "jensen@50mm"):
        score = np.array([r[k] for r in rows])
        print(f"  {k:>12}: {_kendall(score, inc):+.2f}")


def _figure(rows: list[dict], shifts: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    for ax, deg, row in zip(axes.ravel(), ANGLES_DEG, rows):
        ax.plot(shifts, row["curve"], "k.-", lw=1, ms=3, label="J(curb at s)")
        j0 = row["J(belief)"]
        ax.axhline(j0, color="tab:blue", lw=1, label="J(belief)")
        ax.axhline(row["E[J]"], color="tab:red", lw=1, label="E[J] truth")
        ax.plot(shifts, j0 + row["slope@1mm"] * shifts, "g--", lw=1, label="1st-order @1mm")
        ax.set_title(f"approach {deg:.0f} deg")
        ax.set_xlabel("edge shift s [m]")
    axes[0, 0].set_ylabel("tilt cost J")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Curb crossing under position uncertainty: the ensemble J(s) vs what the belief-point "
        "derivative sees"
    )
    fig.tight_layout()
    out = "studies/out/curb_direction.png"
    fig.savefig(out, dpi=130)
    print(f"\nfigure: {out}")


if __name__ == "__main__":
    main()
