"""Probe A: does the map predict the attitude the robot actually experiences?

For every sweep the belief is folded in causally, then the robot's pose `lead` seconds LATER
is settled on that belief and the predicted (pitch, roll) compared against the SLAM pose's.
The map therefore always predicts ground it has not yet driven on, which is the foresight
condition the planner works under.

The residual's spread is sigma_pitch / sigma_roll directly -- the quantity the z-margin field
divides by (PROBABILISTIC_PLANNING_PLAN.md section 3) -- so this needs no sigma model of its
own to be useful. Attitude is compared, never z: the engine's body origin and
`odin1_base_link` differ by a fixed mount offset, which cancels in pitch/roll but not height.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field

import numpy as np
import warp as wp

from calib import bagio
from calib.mapbuild import MapAccumulator
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams
from helhest.heightmap import Heightmap
from helhest.perception.heightmap.postprocess import gaussian_smooth

# The settle reads the dilated envelope at the three wheel centres, so the terrain that must be
# observed is a disk of one wheel radius about each -- not a bounding box over the whole robot.
# Gating on the box rejected 104 of 280 frames on ostrich4, most of them for ground behind the
# robot that a forward-facing sensor can never have seen.
WHEEL_LOCAL_XY = ((0.0, 0.365), (0.0, -0.365), (-0.75, 0.0))  # front-L, front-R, rear
WHEEL_RADIUS_M = 0.35

# `odin1_base_link` is NOT the engine's body origin. The engine puts the origin at the FRONT
# AXLE (RobotParams.build), while base_link sits at the Odin device; the node itself records the
# gap as unresolved ("Plan is device-centered (offset from the real base_link) until a mount TF
# exists", ros/odin/odin_elevation.params.yaml). Left at zero the front wheels land on the
# sensor origin, whose ground is inside the near blind zone -- ground returns start at 0.36 m --
# so their contact disks are unobservable by construction. `mount_dx` is that offset along the
# robot's own +x, fitted in `sweep_mount_offset`.


@dataclass
class Gates:
    """Quasi-static validity gates. The settle is a static solve, so dynamic attitude has to go."""

    max_speed: float = 0.6  # [m/s]
    max_accel: float = 0.8  # [m/s^2]
    max_yaw_rate: float = 0.6  # [rad/s]
    min_obs: int = 2  # frames that must have seen every footprint cell
    max_resid: float = 1e-2  # settle residual gate [m], the engine's own validity bound
    min_separation: float = 0.15  # [m] between accepted probes; a parked robot is one sample,
    # not the fifty near-identical ones it would otherwise contribute


@dataclass
class ProbeResult:
    bag: str
    lead_s: float
    mount_dx: float = 0.0
    n_frames: int = 0
    n_probes: int = 0
    rejected: dict = field(default_factory=dict)
    pitch: dict = field(default_factory=dict)
    roll: dict = field(default_factory=dict)
    coverage: dict = field(default_factory=dict)
    samples: list = field(default_factory=list)


def _wheel_coverage(
    acc,
    x: float,
    y: float,
    yaw: float,
    xmin: float,
    ymin: float,
    cell: float,
    nx: int,
    ny: int,
    wheel_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-wheel (min observation count, mean observation range) over each wheel's contact disk.

    The engine's body origin is the FRONT AXLE (`RobotParams.build`: front wheels at (0, +-b),
    rear at (-rear_offset, 0)), so the local offsets are taken about that, not about a centroid.
    """
    nobs = acc.map_nobs.numpy()
    rng_sum = acc.map_rng.numpy()
    ca, sa = np.cos(yaw), np.sin(yaw)
    obs = np.zeros(len(WHEEL_LOCAL_XY))
    rngs = np.zeros(len(WHEEL_LOCAL_XY))
    for w, (lx, ly) in enumerate(WHEEL_LOCAL_XY):
        wx = x + ca * lx - sa * ly
        wy = y + sa * lx + ca * ly
        c = int((wx - xmin) / cell)
        r = int((wy - ymin) / cell)
        r0, r1 = max(0, r - wheel_cells), min(ny, r + wheel_cells + 1)
        c0, c1 = max(0, c - wheel_cells), min(nx, c + wheel_cells + 1)
        if r1 <= r0 or c1 <= c0:
            return np.zeros(1), np.zeros(1)
        patch = nobs[r0:r1, c0:c1]
        obs[w] = patch.min()
        rngs[w] = (rng_sum[r0:r1, c0:c1] / np.maximum(patch, 1.0)).mean()
    return obs, rngs


def _stats(resid: np.ndarray) -> dict:
    """Bias, spread and tails of a residual population, in degrees."""
    d = np.degrees(resid)
    return {
        "n": int(d.size),
        "bias_deg": float(np.mean(d)),
        "sd_deg": float(np.std(d, ddof=1)) if d.size > 1 else float("nan"),
        "sd_debiased_deg": float(np.std(d - np.mean(d), ddof=1)) if d.size > 1 else float("nan"),
        "mad_deg": float(np.median(np.abs(d - np.median(d)))),
        "p50_abs_deg": float(np.percentile(np.abs(d), 50)),
        "p95_abs_deg": float(np.percentile(np.abs(d), 95)),
        "max_abs_deg": float(np.max(np.abs(d))),
    }


def run(
    bag: str,
    lead_s: float,
    gates: Gates,
    root: str = "bags",
    keep_samples: bool = False,
    mount_dx: float = 0.0,
    cached: tuple | None = None,
    stat: str = "max",
    smooth_sigma_m: float = 0.0,
):
    if cached is not None:
        odom, frames = cached
    else:
        path = bagio.bag_path(bag, root)
        odom = bagio.read_odometry(path)
        frames = bagio.read_frames(path, odom)
    if not frames:
        raise RuntimeError(f"{bag}: no usable frames")

    # Derivatives of the SLAM track, for the quasi-static gates.
    speed = np.linalg.norm(np.gradient(odom.xyz[:, :2], odom.t, axis=0), axis=1)
    accel = np.gradient(speed, odom.t)
    yaw_unwrapped = np.unwrap(odom.rpy[:, 2])
    yaw_rate = np.gradient(yaw_unwrapped, odom.t)

    # Grid: the driven path plus a margin. Cells beyond it are never settled on, so the
    # envelope stays small enough to rebuild every frame.
    cell = 0.08  # the node's `resolution` default
    pad = 3.0
    xmin = float(odom.xyz[:, 0].min() - pad)
    ymin = float(odom.xyz[:, 1].min() - pad)
    nx = int(np.ceil((odom.xyz[:, 0].max() + pad - xmin) / cell))
    ny = int(np.ceil((odom.xyz[:, 1].max() + pad - ymin) / cell))

    grid_params = GridParams(cells_x=nx, cells_y=ny, cell_size=cell, origin_x=xmin, origin_y=ymin)
    robot_params = RobotParams()
    solver_params = SolverParams()
    device = wp.get_device()

    acc = MapAccumulator(ny, nx, cell, xmin, ymin, device=device, stat=stat)
    sim = ForwardSimulator(
        robot_params=robot_params,
        solver_params=solver_params,
        grid_params=grid_params,
        batch_size=1,
        n_steps=1,
        device=device,
    )
    sim.target_wheel_omega.zero_()
    sim.set_friction(Heightmap(np.full((ny, nx), 0.8, np.float32), (xmin, ymin), cell))

    wheel_cells = int(np.ceil(WHEEL_RADIUS_M / cell))
    rej = {
        "speed": 0,
        "accel": 0,
        "yaw_rate": 0,
        "coverage": 0,
        "resid": 0,
        "no_pose": 0,
        "duplicate": 0,
    }
    rows: list[dict] = []
    last_xy: np.ndarray | None = None

    for f in frames:
        acc.add_frame(f.points, f.sensor_xyz)

        t_probe = f.t + lead_s
        if t_probe < odom.t[0] or t_probe > odom.t[-1]:
            rej["no_pose"] += 1
            continue
        j = int(np.argmin(np.abs(odom.t - t_probe)))
        if speed[j] > gates.max_speed:
            rej["speed"] += 1
            continue
        if abs(accel[j]) > gates.max_accel:
            rej["accel"] += 1
            continue
        if abs(yaw_rate[j]) > gates.max_yaw_rate:
            rej["yaw_rate"] += 1
            continue

        yaw = float(odom.rpy[j, 2])
        # base_link -> engine body origin (front axle), along the robot's own +x.
        x = float(odom.xyz[j, 0]) + mount_dx * np.cos(yaw)
        y = float(odom.xyz[j, 1]) + mount_dx * np.sin(yaw)
        if last_xy is not None and np.hypot(x - last_xy[0], y - last_xy[1]) < gates.min_separation:
            rej["duplicate"] += 1
            continue
        obs, rngs = _wheel_coverage(acc, x, y, yaw, xmin, ymin, cell, nx, ny, wheel_cells)
        if obs.min() < gates.min_obs:
            rej["coverage"] += 1
            continue

        meas = acc.measured.numpy()
        elev = acc.elevation(fill_z=float(np.median(acc.elev.numpy()[meas > 0.5])))
        if smooth_sigma_m > 0.0:
            # The deployed node blurs the heightmap (`smooth_sigma`, metres) before anything
            # downstream sees it. That averages over many cells, which attacks exactly the
            # estimator variance the max-vs-mean gap measures -- so the gap has to be re-checked
            # WITH it before any claim about the deployed pipeline is made.
            elev = gaussian_smooth(elev, sigma=smooth_sigma_m / cell)
        sim.set_terrain(elev)
        sim.start_pose.assign(np.array([[x, y, yaw]], np.float32))
        sim.rollout_launch()
        derived = sim.derived.numpy()[0, 0]
        resid = float(sim.residual.numpy()[0, 0])
        if abs(resid) > gates.max_resid:
            rej["resid"] += 1
            continue

        last_xy = np.array([x, y])
        rows.append(
            {
                "t": float(f.t),
                "x": x,
                "y": y,
                "yaw": yaw,
                "pitch_pred": float(derived[1]),
                "roll_pred": float(derived[2]),
                "pitch_meas": float(odom.rpy[j, 1]),
                "roll_meas": float(odom.rpy[j, 0]),
                "speed": float(speed[j]),
                "resid": resid,
                "obs_min": float(obs.min()),
                "obs_mean": float(obs.mean()),
                "range_mean": float(np.mean(rngs)),
            }
        )

    out = ProbeResult(bag=bag, lead_s=lead_s, n_frames=len(frames), n_probes=len(rows))
    out.mount_dx = mount_dx
    out.rejected = rej
    if rows:
        dp = np.array([r["pitch_meas"] - r["pitch_pred"] for r in rows])
        dr = np.array([r["roll_meas"] - r["roll_pred"] for r in rows])
        out.pitch = _stats(dp)
        out.roll = _stats(dr)
        out.coverage = {
            "obs_mean": float(np.mean([r["obs_mean"] for r in rows])),
            "range_mean_m": float(np.mean([r["range_mean"] for r in rows])),
            "speed_mean": float(np.mean([r["speed"] for r in rows])),
        }
        if keep_samples:
            out.samples = rows
    return out


def _pool(results: list[ProbeResult]) -> dict:
    """Pool every bag's probes into one population, and add the pred-vs-meas correlations.

    The correlation is the sign/convention check: a high positive value says the settle and the
    SLAM attitude agree about which way the robot is leaning, which no amount of residual
    spread would reveal on its own.
    """
    rows = [s for r in results for s in r.samples]
    if not rows:
        return {}
    pp = np.array([s["pitch_pred"] for s in rows])
    pm = np.array([s["pitch_meas"] for s in rows])
    rp = np.array([s["roll_pred"] for s in rows])
    rm = np.array([s["roll_meas"] for s in rows])
    out = {
        "n_bags": len(results),
        "pitch": _stats(pm - pp),
        "roll": _stats(rm - rp),
        "corr_pitch": float(np.corrcoef(pp, pm)[0, 1]),
        "corr_roll": float(np.corrcoef(rp, rm)[0, 1]),
        "pitch_meas_sd_deg": float(np.degrees(pm).std(ddof=1)),
        "roll_meas_sd_deg": float(np.degrees(rm).std(ddof=1)),
        "range_mean_m": float(np.mean([s["range_mean"] for s in rows])),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bags", nargs="+")
    ap.add_argument("--lead", type=float, nargs="+", default=[2.0], help="foresight [s]")
    ap.add_argument("--root", default="bags")
    ap.add_argument("--out", default="studies/out/calib")
    ap.add_argument("--mount-dx", type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cache = {}
    for b in args.bags:
        path = bagio.bag_path(b, args.root)
        odom = bagio.read_odometry(path)
        cache[b] = (odom, bagio.read_frames(path, odom))

    report = {"mount_dx": args.mount_dx, "leads": {}}
    for lead in args.lead:
        results = []
        for b in args.bags:
            res = run(b, lead, Gates(), args.root, True, args.mount_dx, cache[b])
            results.append(res)
            p_, r_ = res.pitch or {}, res.roll or {}
            print(
                f"  lead {lead:4.1f}  {b:11s} probes {res.n_probes:3d}/{res.n_frames:3d}  "
                f"pitch {p_.get('bias_deg', float('nan')):+6.2f} +- {p_.get('sd_deg', float('nan')):5.2f}  "
                f"roll {r_.get('bias_deg', float('nan')):+6.2f} +- {r_.get('sd_deg', float('nan')):5.2f}"
            )
        pooled = _pool(results)
        report["leads"][f"{lead:g}"] = {
            "pooled": pooled,
            "per_bag": [asdict(r) for r in results],
        }
        if pooled:
            print(
                f"POOLED lead {lead:4.1f}  n={pooled['pitch']['n']:4d}  "
                f"pitch {pooled['pitch']['bias_deg']:+.2f} +- {pooled['pitch']['sd_deg']:.2f} deg "
                f"(corr {pooled['corr_pitch']:+.2f}, meas sd {pooled['pitch_meas_sd_deg']:.2f})  "
                f"roll {pooled['roll']['bias_deg']:+.2f} +- {pooled['roll']['sd_deg']:.2f} deg "
                f"(corr {pooled['corr_roll']:+.2f}, meas sd {pooled['roll_meas_sd_deg']:.2f})"
            )
    dst = os.path.join(args.out, "probe_attitude.json")
    with open(dst, "w") as fh:
        json.dump(report, fh, indent=1)
    print("->", dst)


if __name__ == "__main__":
    main()
