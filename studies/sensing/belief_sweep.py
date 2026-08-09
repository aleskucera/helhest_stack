"""Is the belief model a property of the sensing, or of the one scene we fitted it on?

    .venv/bin/python -m studies.sensing.belief_sweep --realizations 12

`lidar_belief.py` fitted Sigma = rank-3 plane + compact stationary kernel to ONE fractal terrain
driven in ONE straight line, and the paper now rests on that fit: plane share 0.30, tilt sigma
~0.005 rad/m, residual decaying within three cells. If those numbers are properties of that
scenario rather than of the sensing, the paper's foundation is a coincidence.

This refits the model across three axes that should each move it for a different reason:

  terrain CHARACTER (beta)   more sub-grid roughness feeds the sensor-side variance, which
                             should LOWER the plane's share without touching the plane itself.
  terrain DRAW (seed)        pure sampling: whatever varies here is noise in the fit, and it
                             sets the resolution at which the other two axes can be read.
  TRAJECTORY shape           the tilt is fixed in the sensor frame, so a curving path rotates
                             it in the world and should smear the anisotropy that a straight
                             run produces. A stationary kernel fitted to a curving traverse and
                             one fitted to a straight one need not agree.
  terrain SLOPE              a ramp couples horizontal drift into height error in proportion to
                             the gradient, so it should raise the plane's share.

Reported per configuration: the plane's share of the random variance, the offset and tilt
magnitudes, and how quickly the stationary residual decorrelates. The question is not whether
these are identical -- they will not be -- but whether they move enough to change what the
paper claims.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import warp as wp

from ..adjoint.generalise import fractal_terrain
from ..bench.ranking import CELL
from ..bench.ranking import OUT
from .lidar_belief import autocorrelation_2d
from .lidar_belief import LidarSim
from .lidar_belief import NoiseParams
from .lidar_belief import simulate_belief
from helhest.perception.heightmap import HeightMapBuilder


def trajectory(kind: str, n: int = 40) -> np.ndarray:
    """Straight, curving, or a right-angle turn -- all covering a similar distance."""
    if kind == "straight":
        xs = np.linspace(1.0, 8.0, n)
        return np.stack([xs, np.full_like(xs, 4.5), np.zeros_like(xs)], axis=1)
    if kind == "arc":
        s = np.linspace(0.0, 7.0, n)
        kappa = 0.22
        yaw = kappa * s
        x = 1.0 + np.cumsum(np.cos(yaw)) * (s[1] - s[0])
        y = 2.0 + np.cumsum(np.sin(yaw)) * (s[1] - s[0])
        return np.stack([x, y, yaw], axis=1)
    if kind == "turn":
        half = n // 2
        x = np.concatenate([np.linspace(1.5, 4.5, half), np.full(n - half, 4.5)])
        y = np.concatenate([np.full(half, 2.0), np.linspace(2.0, 7.0, n - half)])
        yaw = np.concatenate([np.zeros(half), np.full(n - half, np.pi / 2)])
        return np.stack([x, y, yaw], axis=1)
    raise ValueError(kind)


def surface(kind: str, seed: int, n: int) -> np.ndarray:
    """Fractal terrain of a given spectrum, optionally on a ramp."""
    if kind.startswith("ramp"):
        beta = float(kind.split("_")[1])
        base = fractal_terrain(n, n, CELL, seed=seed, beta=beta)
        gy, _gx = np.mgrid[0:n, 0:n]
        return base + np.tan(np.radians(10.0)) * gy * CELL  # a 10 deg slope across the patch
    return fractal_terrain(n, n, CELL, seed=seed, beta=float(kind))


def fit(surf: np.ndarray, traj: np.ndarray, reps: int, device: str) -> dict:
    ny, nx = surf.shape
    sim = LidarSim(surf, 0.0, 0.0, CELL, device)
    builder = HeightMapBuilder(
        CELL, (0.0, nx * CELL, 0.0, ny * CELL), device=wp.get_device(device)
    )
    p = NoiseParams()
    maps, cnts = [], []
    for r in range(reps):
        m, c = simulate_belief(sim, traj, p, 5000 + r, builder)
        maps.append(m)
        cnts.append(c)
    maps, cnts = np.stack(maps), np.stack(cnts)
    obs = np.isfinite(maps).all(0) & (cnts > 0).all(0)
    if obs.sum() < 500:
        return {"observed": float(obs.mean()), "usable": False}
    mean_map = np.nanmean(maps, 0)
    gy, gx = np.mgrid[0:ny, 0:nx]
    xc = gx * CELL - gx[obs].mean() * CELL
    yc = gy * CELL - gy[obs].mean() * CELL
    A = np.c_[np.ones(obs.sum()), xc[obs], yc[obs]]
    coefs, res = [], []
    for r in range(reps):
        e = (maps[r] - mean_map)[obs]
        c3, *_ = np.linalg.lstsq(A, e, rcond=None)
        coefs.append(c3)
        f = np.zeros((ny, nx))
        f[obs] = e - A @ c3
        res.append(f)
    coefs = np.array(coefs)
    vp = float(np.mean([np.var(A @ c3) for c3 in coefs]))
    vr = float(np.mean([np.var(f[obs]) for f in res]))
    rho = autocorrelation_2d(res[0], obs, 12)
    prof = rho[12, 12:] / rho[12, 12]
    below = np.where(prof < np.exp(-1.0))[0]
    return {
        "usable": True,
        "observed": float(obs.mean()),
        "plane_share": vp / (vp + vr),
        "sd_offset_cm": float(np.sqrt(np.cov(coefs.T)[0, 0]) * 100),
        "sd_tilt_x_mrad_per_m": float(np.sqrt(np.cov(coefs.T)[1, 1]) * 1000),
        "sd_tilt_y_mrad_per_m": float(np.sqrt(np.cov(coefs.T)[2, 2]) * 1000),
        "sigma_median_cm": float(np.nanmedian(np.nanstd(maps, 0)[obs]) * 100),
        "residual_1e_m": float(below[0] * CELL) if len(below) else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--realizations", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    wp.init()
    n = 90
    rows = []
    print(f"{'terrain':>10}{'traj':>10}{'seed':>5}{'obs':>6}{'plane':>8}{'offs cm':>9}"
          f"{'tilt x':>8}{'tilt y':>8}{'sigma cm':>10}{'resid 1/e':>11}")
    for kind in ("1.8", "2.4", "3.0", "ramp_2.4"):
        for tkind in ("straight", "arc", "turn"):
            for seed in range(args.seeds):
                r = fit(surface(kind, seed, n), trajectory(tkind), args.realizations, args.device)
                r.update({"terrain": kind, "trajectory": tkind, "seed": seed})
                rows.append(r)
                if not r["usable"]:
                    print(f"{kind:>10}{tkind:>10}{seed:>5}{r['observed']:>6.0%}   too little coverage")
                    continue
                print(f"{kind:>10}{tkind:>10}{seed:>5}{r['observed']:>6.0%}"
                      f"{r['plane_share']:>8.3f}{r['sd_offset_cm']:>9.2f}"
                      f"{r['sd_tilt_x_mrad_per_m']:>8.2f}{r['sd_tilt_y_mrad_per_m']:>8.2f}"
                      f"{r['sigma_median_cm']:>10.2f}{r['residual_1e_m']:>11.2f}")
    ok = [r for r in rows if r.get("usable")]
    agg = {
        k: {"median": float(np.median([r[k] for r in ok])),
            "min": float(np.min([r[k] for r in ok])),
            "max": float(np.max([r[k] for r in ok]))}
        for k in ("plane_share", "sd_offset_cm", "sd_tilt_x_mrad_per_m",
                  "sigma_median_cm", "residual_1e_m")
    }
    print("\n  across all configurations       median      min      max")
    for k, v in agg.items():
        print(f"    {k:26s} {v['median']:8.3f} {v['min']:8.3f} {v['max']:8.3f}")
    (OUT / "belief_sweep.json").write_text(json.dumps({"rows": rows, "aggregate": agg}, indent=1))
    print(f"\nwrote {OUT / 'belief_sweep.json'}")


if __name__ == "__main__":
    main()
