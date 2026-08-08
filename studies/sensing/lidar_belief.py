"""Where does the belief's sigma actually come from? Simulate the sensing instead of assuming it.

    .venv/bin/python -m studies.sensing.lidar_belief --realizations 32

The risk study hands the planner a belief (mean, sigma) drawn from a HAND-BUILT noise field:
sigma grows with frontier distance and relief, correlation is imposed as a separable kernel with
a 0.15 m length scale, and the Monte-Carlo "truth" then samples from that same model. Two things
are wrong with that. The sigma is invented, and the evaluation is circular -- every estimator is
graded on how well it predicts a distribution we wrote down.

This module removes both. It generates a true terrain, flies a real beam pattern over it with a
ray-march against the height field, corrupts the returns the way a lidar actually errs (range
noise and dropout that grow at grazing incidence), moves the sensor with a DRIFTING pose error,
and rasterizes the surviving points with the robot's own `HeightMapBuilder`. The belief is then
whatever that pipeline produced, and the truth is the terrain we generated. Nothing is assumed
twice.

WHAT COMES OUT THAT CANNOT BE PUT IN.

  sigma        emerges from geometry -- range, incidence, hit count, occlusion -- rather than
               from a frontier-distance heuristic.
  rho          emerges too, and this is the interesting one. Pose error is SHARED by every point
               in a scan, so it makes the map wrong coherently over metres; the beam footprint
               spanning cells makes it wrong coherently over centimetres. A per-cell variance
               cannot express either. We measure the resulting 2-D autocorrelation and test
               whether it is separable, which is what `clark.rho_lookup` assumes.
  occlusion    is exact: a cell is unobserved because no beam reached it, not because a heuristic
               said so.
  the tails    are measurable. Clark moment-matches Gaussians; grazing incidence and occlusion
               edges are where that assumption would break, and the skew/kurtosis of the measured
               error says whether it does.

WHAT IS STILL MODELLED, AND SHOULD BE CALLED THAT. The range-noise coefficients are placeholders
in the right regime, not calibrated for a specific unit: a centimetre-scale base, a few mm per
metre of range, and an incidence term that reaches ~10 cm near 80 deg, which is the order
reported for a VLP-16. Vegetation, multi-echo, wet surfaces and dust are NOT modelled at all,
and they are exactly where real maps break -- so this replaces "invented sigma" with "sigma from
a simulated sensor", not with ground truth. The pose-error random walk should be replaced by
statistics measured from the stack's own ICP on real bags; that is the cheapest next upgrade.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

from ..adjoint.generalise import fractal_terrain
from ..bench.clark import rho1_table
from ..bench.risk import CORR_LEN
from ..bench.ranking import CELL
from ..bench.ranking import OUT
from helhest.perception.heightmap import HeightMapBuilder

# --- sensor description ------------------------------------------------------------------------
# An Ouster-like spinning stack: 64 rings over a +/-22.5 deg elevation fan, 1024 azimuths.
N_AZIMUTH = 1024
N_ELEVATION = 64
EL_MIN_DEG, EL_MAX_DEG = -22.5, 22.5
MAX_RANGE = 25.0  # [m]
SENSOR_HEIGHT = 0.55  # [m] above the contact plane, roughly where the puck sits


@dataclass
class NoiseParams:
    """Range-noise and dropout model. Placeholders in the right regime -- see module docstring."""

    sigma_base: float = 0.010  # [m] noise floor
    sigma_per_m: float = 0.002  # [m per m] growth with range
    sigma_incidence: float = 0.020  # [m] scale of the grazing-incidence term
    dropout_base: float = 0.02
    dropout_incidence: float = 0.60
    pose_step_xy: float = 0.010  # [m] per-scan random-walk step of the localization error
    pose_step_yaw: float = 0.0035  # [rad] per-scan random-walk step (~0.2 deg)
    pose_step_z: float = 0.004  # [m] vertical drift -- shifts every height in a scan together
    pose_step_pitch: float = 0.0020  # [rad] attitude drift; tilts the scan, so the height error
    # it induces GROWS WITH RANGE. This is the dominant source of long-range error correlation,
    # and a per-cell variance model has no way to express it.


@wp.func
def _sample_height(
    truth: wp.array2d(dtype=wp.float32), x0: float, y0: float, inv_cell: float, nx: int, ny: int,
    px: float, py: float,
) -> float:
    """Bilinear height at world (px, py); cell-center convention, clamped at the border."""
    fx = (px - x0) * inv_cell - 0.5
    fy = (py - y0) * inv_cell - 0.5
    ix = int(wp.floor(fx))
    iy = int(wp.floor(fy))
    ix = wp.clamp(ix, 0, nx - 2)
    iy = wp.clamp(iy, 0, ny - 2)
    tx = wp.clamp(fx - float(ix), 0.0, 1.0)
    ty = wp.clamp(fy - float(iy), 0.0, 1.0)
    h00 = truth[iy, ix]
    h10 = truth[iy, ix + 1]
    h01 = truth[iy + 1, ix]
    h11 = truth[iy + 1, ix + 1]
    return (
        h00 * (1.0 - tx) * (1.0 - ty)
        + h10 * tx * (1.0 - ty)
        + h01 * (1.0 - tx) * ty
        + h11 * tx * ty
    )


@wp.kernel
def _cast_kernel(
    truth: wp.array2d(dtype=wp.float32),
    x0: float, y0: float, inv_cell: float, nx: int, ny: int,
    ox: float, oy: float, oz: float, yaw: float,
    az: wp.array(dtype=wp.float32),
    el: wp.array(dtype=wp.float32),
    n_az: int,
    step: float,
    max_range: float,
    sigma_base: float, sigma_per_m: float, sigma_incidence: float,
    dropout_base: float, dropout_incidence: float,
    seed: int,
    hits: wp.array(dtype=wp.vec3),
    valid: wp.array(dtype=wp.int32),
):
    """One thread per beam: march the ray until it goes under the surface, refine the crossing,
    then corrupt the RANGE (not the height) and decide whether the return survives.

    Perturbing along the beam is the whole point of ray-marching: at normal incidence a range
    error lands in z, and at grazing incidence the same error mostly slides the point sideways.
    An additive per-cell height noise cannot represent that difference, and grazing incidence is
    where the map is worst.
    """
    tid = wp.tid()
    i_el = tid / n_az
    i_az = tid - i_el * n_az
    a = az[i_az] + yaw
    e = el[i_el]
    dx = wp.cos(e) * wp.cos(a)
    dy = wp.cos(e) * wp.sin(a)
    dz = wp.sin(e)

    valid[tid] = 0
    if dz > -1.0e-6 and oz > 0.0:
        # pointing up or level: only hits if the terrain rises into it, still worth marching
        pass

    t = step
    prev_gap = oz - _sample_height(truth, x0, y0, inv_cell, nx, ny, ox, oy)
    while t < max_range:
        px = ox + dx * t
        py = oy + dy * t
        pz = oz + dz * t
        if px < x0 or py < y0:
            return
        if px > x0 + float(nx) / inv_cell or py > y0 + float(ny) / inv_cell:
            return
        gap = pz - _sample_height(truth, x0, y0, inv_cell, nx, ny, px, py)
        if gap <= 0.0:
            # linear refinement of the crossing between t-step and t
            frac = prev_gap / wp.max(prev_gap - gap, 1.0e-9)
            t_hit = (t - step) + step * frac
            hx = ox + dx * t_hit
            hy = oy + dy * t_hit
            hz = oz + dz * t_hit

            # surface normal from the height field, for the incidence angle
            d = 1.0 / inv_cell
            gx = (
                _sample_height(truth, x0, y0, inv_cell, nx, ny, hx + d, hy)
                - _sample_height(truth, x0, y0, inv_cell, nx, ny, hx - d, hy)
            ) / (2.0 * d)
            gy = (
                _sample_height(truth, x0, y0, inv_cell, nx, ny, hx, hy + d)
                - _sample_height(truth, x0, y0, inv_cell, nx, ny, hx, hy - d)
            ) / (2.0 * d)
            nrm = wp.normalize(wp.vec3(-gx, -gy, 1.0))
            cos_inc = wp.abs(wp.dot(nrm, wp.vec3(-dx, -dy, -dz)))
            cos_inc = wp.clamp(cos_inc, 0.02, 1.0)

            state = wp.rand_init(seed, tid)
            sigma_r = (
                sigma_base
                + sigma_per_m * t_hit
                + sigma_incidence * (1.0 / cos_inc - 1.0)
            )
            p_drop = dropout_base + dropout_incidence * wp.pow(1.0 - cos_inc, 3.0)
            if wp.randf(state) < p_drop:
                return
            r_noisy = t_hit + sigma_r * wp.randn(state)
            hits[tid] = wp.vec3(ox + dx * r_noisy, oy + dy * r_noisy, oz + dz * r_noisy)
            valid[tid] = 1
            return
        prev_gap = gap
        t += step


@wp.kernel
def _compact_kernel(
    hits: wp.array(dtype=wp.vec3), valid: wp.array(dtype=wp.int32),
    counter: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if valid[tid] == 1:
        idx = wp.atomic_add(counter, 0, 1)
        if idx < out.shape[0]:
            out[idx] = hits[tid]


class LidarSim:
    """Casts the beam pattern against a true height field, on device."""

    def __init__(self, truth: np.ndarray, x0: float, y0: float, cell: float, device: str):
        self.ny, self.nx = truth.shape
        self.x0, self.y0, self.cell = x0, y0, cell
        self.device = device
        n_beams = N_AZIMUTH * N_ELEVATION
        with wp.ScopedDevice(device):
            self.truth = wp.array(np.ascontiguousarray(truth, np.float32), dtype=wp.float32)
            self.az = wp.array(
                np.linspace(0.0, 2.0 * np.pi, N_AZIMUTH, endpoint=False, dtype=np.float32),
                dtype=wp.float32,
            )
            self.el = wp.array(
                np.radians(np.linspace(EL_MIN_DEG, EL_MAX_DEG, N_ELEVATION, dtype=np.float32)),
                dtype=wp.float32,
            )
            self._hits = wp.zeros(n_beams, dtype=wp.vec3)
            self._valid = wp.zeros(n_beams, dtype=wp.int32)
            self._out = wp.zeros(n_beams, dtype=wp.vec3)
            self._counter = wp.zeros(1, dtype=wp.int32)

    def scan(self, pose: tuple[float, float, float], p: NoiseParams, seed: int) -> wp.array:
        """One scan from (x, y, yaw). Returns the surviving hits as a device vec3 array."""
        x, y, yaw = pose
        z = self._ground(x, y) + SENSOR_HEIGHT
        self._valid.zero_()
        self._counter.zero_()
        with wp.ScopedDevice(self.device):
            wp.launch(
                _cast_kernel,
                dim=N_AZIMUTH * N_ELEVATION,
                inputs=[
                    self.truth, self.x0, self.y0, 1.0 / self.cell, self.nx, self.ny,
                    x, y, z, yaw, self.az, self.el, N_AZIMUTH,
                    0.5 * self.cell, MAX_RANGE,
                    p.sigma_base, p.sigma_per_m, p.sigma_incidence,
                    p.dropout_base, p.dropout_incidence, seed,
                ],
                outputs=[self._hits, self._valid],
            )
            wp.launch(
                _compact_kernel,
                dim=N_AZIMUTH * N_ELEVATION,
                inputs=[self._hits, self._valid, self._counter],
                outputs=[self._out],
            )
        n = int(self._counter.numpy()[0])
        return self._out[:n]

    def _ground(self, x: float, y: float) -> float:
        iy = int(np.clip(round((y - self.y0) / self.cell), 0, self.ny - 1))
        ix = int(np.clip(round((x - self.x0) / self.cell), 0, self.nx - 1))
        return float(self.truth.numpy()[iy, ix])


def simulate_belief(
    sim: LidarSim, traj: np.ndarray, p: NoiseParams, seed: int, builder: HeightMapBuilder,
) -> tuple[np.ndarray, np.ndarray]:
    """Drive `traj`, accumulate every scan's hits, rasterize with the robot's own builder.

    The localization error is a RANDOM WALK over the trajectory, not per-scan white noise: that
    is what makes the map wrong coherently rather than independently, and it is the part a
    per-cell variance model structurally cannot represent.
    """
    rng = np.random.default_rng(seed)
    steps = np.array([p.pose_step_xy, p.pose_step_xy, p.pose_step_z, p.pose_step_yaw,
                      p.pose_step_pitch, p.pose_step_pitch])
    drift = np.zeros(6)  # dx, dy, dz, dyaw, dpitch, droll
    clouds = []
    for k, (x, y, yaw) in enumerate(traj):
        drift = drift + rng.normal(0.0, steps)
        pts = sim.scan((float(x), float(y), float(yaw)), p, seed * 7919 + k)
        if len(pts) == 0:
            continue
        arr = pts.numpy()
        # The map is built in the ESTIMATED frame, so the drift is a rigid error on the WHOLE
        # scan: every point in it moves together. The attitude part matters most -- a pitch
        # error lifts far returns more than near ones, which is what correlates the map error
        # over metres rather than over cells.
        dxp, dyp = arr[:, 0] - x, arr[:, 1] - y
        c, sn = np.cos(drift[3]), np.sin(drift[3])
        rx = c * dxp - sn * dyp + x + drift[0]
        ry = sn * dxp + c * dyp + y + drift[1]
        rz = arr[:, 2] + drift[2] - drift[4] * dxp + drift[5] * dyp
        clouds.append(np.stack([rx, ry, rz], axis=1))
    cloud = np.concatenate(clouds, axis=0).astype(np.float32)
    layers = builder.build(cloud)
    out = layers.to_numpy()
    return out["mean"], out["count"]


def autocorrelation_2d(err: np.ndarray, mask: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalized spatial autocorrelation of the error field over observed cells, for every lag
    in [-max_lag, max_lag]^2. Computed on the masked, mean-removed field so that unobserved
    cells contribute nothing rather than contributing a zero."""
    e = np.where(mask, err - err[mask].mean(), 0.0)
    m = mask.astype(np.float64)
    size = (err.shape[0] + 2 * max_lag, err.shape[1] + 2 * max_lag)
    fe = np.fft.rfft2(e, s=size)
    fm = np.fft.rfft2(m, s=size)
    num = np.fft.irfft2(fe * np.conj(fe), s=size)
    den = np.fft.irfft2(fm * np.conj(fm), s=size)
    lags = np.arange(-max_lag, max_lag + 1)
    idx = np.mod(lags, size[0])[:, None], np.mod(lags, size[1])[None, :]
    cov = num[idx] / np.maximum(den[idx], 1.0)
    return cov / cov[max_lag, max_lag]


def run(seed: int, realizations: int, device: str, p: NoiseParams) -> tuple[dict, dict]:
    ny = nx = 90
    x0 = y0 = 0.0
    truth = fractal_terrain(ny, nx, CELL, seed=seed)
    # a straight traverse across the middle of the patch, 40 poses like the study's rollouts
    xs = np.linspace(1.0, 8.0, 40)
    traj = np.stack([xs, np.full_like(xs, 4.5), np.zeros_like(xs)], axis=1)

    sim = LidarSim(truth, x0, y0, CELL, device)
    builder = HeightMapBuilder(CELL, (x0, x0 + nx * CELL, y0, y0 + ny * CELL), device=wp.get_device(device))

    maps, counts = [], []
    for r in range(realizations):
        m, c = simulate_belief(sim, traj, p, seed * 1000 + r, builder)
        maps.append(m)
        counts.append(c)
    maps = np.stack(maps)
    counts = np.stack(counts)

    observed = np.isfinite(maps).all(axis=0) & (counts > 0).all(axis=0)
    mean_map = np.where(observed, np.nanmean(maps, axis=0), np.nan)
    sigma_emp = np.where(observed, np.nanstd(maps, axis=0), np.nan)
    bias = np.where(observed, mean_map - truth, np.nan)

    # TWO different error fields, and conflating them would be a mistake. Sigma in the paper
    # describes the belief's RANDOM spread, so that is what its correlation must be compared
    # against. The systematic bias is a separate object: the belief does not know about it, and
    # no covariance model can absorb it.
    err_total = np.where(observed, maps[0] - truth, 0.0)
    err_random = np.where(observed, maps[0] - mean_map, 0.0)
    max_lag = 40   # 4 m: the random part turned out to reach much further than 12 cells
    rho_tot = autocorrelation_2d(err_total, observed, max_lag)
    rho_rnd = autocorrelation_2d(err_random, observed, max_lag)
    mid = rho_rnd.shape[0] // 2
    rho_y, rho_x = rho_rnd[:, mid], rho_rnd[mid, :]
    outer = np.outer(rho_y, rho_x)
    big = np.abs(rho_rnd) > 0.05
    sep_err = float(np.abs(rho_rnd - outer)[big].max()) if big.any() else float("nan")

    def _len_scale(profile: np.ndarray, centered: bool = True) -> float:
        """Distance in metres at which the profile first falls below 1/e."""
        half = profile[mid:] if centered else profile
        below = np.where(half < np.exp(-1.0))[0]
        return float(below[0] * CELL) if len(below) else float("nan")

    # the study's own kernel, measured with the SAME 1/e rule so the comparison is like-for-like
    assumed = _len_scale(rho1_table(CORR_LEN, CELL), centered=False)

    # Tail statistics on the RANDOM part only, normalized by a pooled sigma rather than each
    # cell's own few-sample estimate -- dividing by a noisy per-cell sigma manufactures a heavy
    # tail all by itself.
    resid = (maps - mean_map[None]) / max(float(np.nanmedian(sigma_emp)), 1e-9)
    z = resid[:, observed].ravel()
    return {
        "seed": seed, "realizations": realizations,
        "observed_frac": float(observed.mean()),
        "sigma_median_m": float(np.nanmedian(sigma_emp)),
        "sigma_p90_m": float(np.nanpercentile(sigma_emp[observed], 90)),
        "bias_median_m": float(np.nanmedian(bias)),
        "bias_p90_abs_m": float(np.nanpercentile(np.abs(bias[observed]), 90)),
        "corr_len_y_m": _len_scale(rho_y),
        "corr_len_x_m": _len_scale(rho_x),
        "corr_len_total_x_m": _len_scale(rho_tot[mid, :]),
        "corr_len_assumed_m": assumed,
        "separability_max_abs_err": sep_err,
        # How much of the RANDOM variation is just the whole map moving up or down together?
        # A rigid per-realization offset is a real localization mode, it is perfectly correlated
        # at every lag, and it is the one structure a per-cell sigma cannot express at all.
        "global_offset_share": float(
            np.var([np.nanmean((maps[r] - mean_map)[observed]) for r in range(realizations)])
            / max(float(np.nanmean(sigma_emp[observed] ** 2)), 1e-12)
        ),
        "bias_share_of_mse": float(
            np.nanmean(bias[observed] ** 2)
            / max(np.nanmean(bias[observed] ** 2) + np.nanmean(sigma_emp[observed] ** 2), 1e-12)
        ),
        "z_skew": float(((z - z.mean()) ** 3).mean() / max(z.std() ** 3, 1e-12)),
        "z_kurtosis_excess": float(((z - z.mean()) ** 4).mean() / max(z.std() ** 4, 1e-12) - 3.0),
        "rho_profile_x": rho_x[mid:].tolist(),
        "rho_profile_y": rho_y[mid:].tolist(),
    }, {"sigma": sigma_emp, "err_random": err_random, "observed": observed, "traj": traj}


def figure(res: dict, sigma: np.ndarray, err: np.ndarray, observed: np.ndarray,
           traj: np.ndarray, path: Path) -> None:
    """Three panels: what sigma looks like, how far the error correlates, and how heavy its
    tail is. Everything measured, nothing assumed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.6,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.0, 1.95))

    im = a.imshow(np.where(observed, sigma, np.nan) * 100, origin="lower",
                  extent=(0, sigma.shape[1] * CELL, 0, sigma.shape[0] * CELL),
                  cmap="viridis", vmin=0, vmax=8)
    a.plot(traj[:, 0], traj[:, 1], color="#cb181d", lw=1.2)
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.03, label="$\\sigma$ [cm]")
    a.set_title("(a) $\\sigma$ from simulated sensing", loc="left")
    a.set_xlabel("$x$ [m]")
    a.set_ylabel("$y$ [m]")

    lag = np.arange(len(res["rho_profile_x"])) * CELL
    b.plot(lag, res["rho_profile_x"], "-o", ms=2.5, color="#cb181d", label="measured, along travel")
    b.plot(lag, res["rho_profile_y"], "-o", ms=2.5, color="#2171b5", label="measured, across")
    assumed = rho1_table(CORR_LEN, CELL)
    b.plot(np.arange(len(assumed)) * CELL, assumed, "k--", label="assumed by the study")
    b.axhline(np.exp(-1.0), color="#969696", lw=0.8, ls=":")
    b.set_xlim(0, 2.5)
    b.set_xlabel("lag [m]")
    b.set_ylabel(r"$\rho$")
    b.legend(frameon=False)
    b.set_title("(b) correlation of the random error", loc="left")

    z = err[observed] / np.nanstd(err[observed])
    c.hist(z, bins=90, density=True, color="#c6dbef", edgecolor="none")
    g = np.linspace(-6, 6, 200)
    c.plot(g, np.exp(-0.5 * g**2) / np.sqrt(2 * np.pi), "k--", label="Gaussian")
    c.set_yscale("log")
    c.set_ylim(1e-4, 1.0)
    c.set_xlim(-6, 6)
    c.set_xlabel("standardized map error")
    c.set_ylabel("density")
    c.legend(frameon=False)
    c.set_title(f"(c) tails: excess kurtosis {res['z_kurtosis_excess']:+.1f}", loc="left")

    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--realizations", type=int, default=32)
    args = ap.parse_args()
    wp.init()
    res, extras = run(args.seed, args.realizations, args.device, NoiseParams())
    print("=== belief produced by simulated sensing ===")
    print(f"  observed cells              {res['observed_frac']:.1%}")
    print(f"  sigma  median / p90         {res['sigma_median_m']*100:.2f} / "
          f"{res['sigma_p90_m']*100:.2f} cm")
    print(f"  |bias| median / p90         {abs(res['bias_median_m'])*100:.2f} / "
          f"{res['bias_p90_abs_m']*100:.2f} cm")
    print(f"  corr length, random part    {res['corr_len_x_m']*100:.0f} / "
          f"{res['corr_len_y_m']*100:.0f} cm (x/y)   vs {res['corr_len_assumed_m']*100:.0f} cm"
          " assumed, same 1/e rule")
    print(f"  corr length, total error    {res['corr_len_total_x_m']*100:.0f} cm")
    print(f"  bias share of total MSE     {res['bias_share_of_mse']:.0%}")
    print(f"  rigid global offset share   {res['global_offset_share']:.0%} of the random variance")
    print(f"  separability max |error|    {res['separability_max_abs_err']:.3f}"
          "   (0 = rho(dy,dx) = rho1(dy) rho1(dx) exactly)")
    print(f"  standardized error skew     {res['z_skew']:+.2f}")
    print(f"  standardized excess kurt.   {res['z_kurtosis_excess']:+.2f}   (0 = Gaussian)")
    path = Path(OUT) / "lidar_belief.json"
    path.write_text(json.dumps(res, indent=1))
    print(f"wrote {path}")
    figure(res, extras["sigma"], extras["err_random"], extras["observed"], extras["traj"],
           Path(OUT) / "lidar_belief.png")


if __name__ == "__main__":
    main()
