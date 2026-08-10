"""Run the estimator on a belief the REAL perception stack produced, from a recorded bag.

    .venv/bin/python -m studies.sensing.bag_belief --bag bags/outdoor_experiment_new2

Every number in the risk study comes from a synthetic belief: a smooth sigma field, no holes, no
NaNs, no occlusion ring, no blind fill. A real map has all of those, and this project's own
history says that is where things break -- the phantom-plateau bug came from unobserved cells
filling at 0.0 while the ground sat at -1.29 m. So before any of this goes near the robot, the
estimator should meet a real map.

This is a ROBUSTNESS test, not an accuracy test. There is no ground truth here. What it asks is:
does the estimator survive real map pathology, what does it do at unobserved cells, and does it
still run in the time budget.

NO ROS IS REQUIRED. The bag is read with `rosbags`, the cloud is deserialized from its raw
buffer, and the sensor mount -- which is NOT in this bag's tf_static -- is recovered by fitting
a ground plane to the near field rather than guessed. Poses come from /odom_2d, so the
accumulated map carries the real odometry drift, which is the point.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from ..bench.clark import rho1_table
from ..bench.ranking import CELL
from ..bench.ranking import OUT
from ..bench.risk import CORR_LEN
from ..bench.clark_conv import plan_moments_conv
from helhest.engine import RobotParams
from helhest.perception.heightmap import HeightMapBuilder

WINDOW = 9.0  # [m] square map window, matching the study's patch size


def read_bag(path: Path, n_clouds: int, stride: int) -> tuple[list[np.ndarray], np.ndarray]:
    """`n_clouds` point clouds (every `stride`-th) and the odom pose nearest each in time."""
    from rosbags.highlevel import AnyReader

    clouds, stamps = [], []
    odom_t, odom_p = [], []
    with AnyReader([path]) as reader:
        conns = [c for c in reader.connections if c.topic in ("/ouster/points", "/odom_2d")]
        seen = 0
        for conn, t, raw in reader.messages(connections=conns):
            if conn.topic == "/odom_2d":
                m = reader.deserialize(raw, conn.msgtype)
                q = m.pose.pose.orientation
                yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
                odom_t.append(t)
                odom_p.append([m.pose.pose.position.x, m.pose.pose.position.y, yaw])
            else:
                seen += 1
                if seen % stride or len(clouds) >= n_clouds:
                    continue
                m = reader.deserialize(raw, conn.msgtype)
                buf = np.frombuffer(m.data, dtype=np.uint8).reshape(-1, m.point_step)
                xyz = buf[:, :12].copy().view(np.float32).reshape(-1, 3)
                good = np.isfinite(xyz).all(axis=1) & (np.abs(xyz).sum(axis=1) > 1e-6)
                clouds.append(xyz[good].astype(np.float64))
                stamps.append(t)
    odom_t = np.array(odom_t)
    odom_p = np.array(odom_p)
    # searchsorted gives the first odom AT-OR-AFTER the cloud stamp; compare it against the one
    # just before to pick whichever is actually nearest in time (the docstring's promise).
    after = np.clip(np.searchsorted(odom_t, np.array(stamps)), 0, len(odom_t) - 1)
    before = np.clip(after - 1, 0, len(odom_t) - 1)
    idx = np.where(
        np.abs(odom_t[before] - np.array(stamps)) <= np.abs(odom_t[after] - np.array(stamps)),
        before,
        after,
    )
    return clouds, odom_p[idx]


def level_from_ground(cloud: np.ndarray, r_min: float = 2.0, r_max: float = 8.0) -> np.ndarray:
    """Recover the sensor mount by fitting the ground plane, since the bag has no TF for it.

    Iteratively trimmed least squares on the near field: fit z = a x + b y + c, keep the points
    below the fit (ground rather than structure), refit. Returns the 4x4 that levels the cloud
    and puts the ground plane at z = 0."""
    r = np.linalg.norm(cloud[:, :2], axis=1)
    pts = cloud[(r > r_min) & (r < r_max)]
    keep = np.ones(len(pts), bool)
    for _ in range(6):
        a = np.c_[pts[keep, 0], pts[keep, 1], np.ones(keep.sum())]
        coef, *_ = np.linalg.lstsq(a, pts[keep, 2], rcond=None)
        resid = pts[:, 2] - (coef[0] * pts[:, 0] + coef[1] * pts[:, 1] + coef[2])
        keep = resid < np.percentile(resid, 60)  # ground is the lower envelope
    nrm = np.array([-coef[0], -coef[1], 1.0])
    nrm /= np.linalg.norm(nrm)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(nrm, z)
    s, c = np.linalg.norm(v), float(np.dot(nrm, z))
    rot = np.eye(3)
    if s > 1e-9:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rot = np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)
    tf = np.eye(4)
    tf[:3, :3] = rot
    tf[2, 3] = -coef[2] * c
    return tf


def build_real_belief(
    clouds: list[np.ndarray], poses: np.ndarray, device: str, self_filter: float = 1.0
) -> dict:
    """Accumulate the scans into one grid with the robot's own rasterizer."""
    tf = level_from_ground(clouds[0])
    ctr = poses[:, :2].mean(axis=0)
    pts_world = []
    for cloud, (px, py, yaw) in zip(clouds, poses):
        r = np.linalg.norm(cloud[:, :2], axis=1)
        cloud = cloud[(r > self_filter) & (r < 25.0)]
        p = (tf[:3, :3] @ cloud.T).T + tf[:3, 3]
        c, s = np.cos(yaw), np.sin(yaw)
        pts_world.append(
            np.stack([c * p[:, 0] - s * p[:, 1] + px, s * p[:, 0] + c * p[:, 1] + py, p[:, 2]], 1)
        )
    pts = np.concatenate(pts_world)
    x0, y0 = ctr[0] - WINDOW / 2, ctr[1] - WINDOW / 2
    n = int(WINDOW / CELL)
    inside = (
        (pts[:, 0] >= x0) & (pts[:, 0] < x0 + WINDOW)
        & (pts[:, 1] >= y0) & (pts[:, 1] < y0 + WINDOW)
    )
    pts = pts[inside]

    builder = HeightMapBuilder(CELL, (x0, x0 + WINDOW, y0, y0 + WINDOW), device=wp.get_device(device))
    layers = builder.build(np.ascontiguousarray(pts, np.float32))
    out = layers.to_numpy()

    ix = np.clip(((pts[:, 0] - x0) / CELL).astype(int), 0, n - 1)
    iy = np.clip(((pts[:, 1] - y0) / CELL).astype(int), 0, n - 1)
    s1 = np.zeros((n, n))
    s2 = np.zeros((n, n))
    cnt = np.zeros((n, n))
    np.add.at(s1, (iy, ix), pts[:, 2])
    np.add.at(s2, (iy, ix), pts[:, 2] ** 2)
    np.add.at(cnt, (iy, ix), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / cnt
        var = np.maximum(s2 / cnt - mean**2, 0.0)
    return {
        "mean": mean, "count": cnt, "within_var": var, "layers": out,
        "x0": x0, "y0": y0, "n": n, "center": ctr, "n_points": len(pts),
    }


def arcs(n_plans: int, n_steps: int, length: float) -> np.ndarray:
    """A fan of constant-curvature candidates from the origin, shaped like the planner's."""
    out = np.zeros((n_steps + 1, n_plans, 3))
    for k, kappa in enumerate(np.linspace(-0.6, 0.6, n_plans)):
        s = np.linspace(0.0, length, n_steps + 1)
        yaw = kappa * s
        out[:, k, 0] = np.cumsum(np.cos(yaw)) * (s[1] - s[0])
        out[:, k, 1] = np.cumsum(np.sin(yaw)) * (s[1] - s[0])
        out[:, k, 2] = yaw
    return out


def run(bag: Path, n_clouds: int, stride: int, device: str) -> dict:
    rp = RobotParams()
    rho1 = rho1_table(CORR_LEN, CELL)
    clouds, poses = read_bag(bag, n_clouds, stride)
    b = build_real_belief(clouds, poses, device)
    n, mean, cnt = b["n"], b["mean"], b["count"]
    observed = cnt > 0
    ground = float(np.nanmedian(mean[observed]))

    # sigma: standard error of the cell mean, floored, and large where nothing was seen
    with np.errstate(invalid="ignore", divide="ignore"):
        sig_obs = np.sqrt(b["within_var"] / np.maximum(cnt, 1.0))
    sigma_base = np.where(observed, np.maximum(sig_obs, 0.01), 0.25)

    # THE FILL POLICY IS THE EXPERIMENT. Unobserved cells are 100% of the difference between a
    # synthetic belief and a real one, and this project has already shipped a bug where they
    # filled at 0.0 while the ground sat over a metre below.
    fills = {
        "ground-referenced": np.where(observed, mean, ground),
        "zero-filled (the old bug)": np.where(observed, mean, 0.0),
        "raw (NaN left in)": mean.copy(),
    }

    plans = arcs(16, 40, 4.0)
    start = b["center"]
    ctl = plans.copy()
    ctl[:, :, 0] += start[0] - 2.0
    ctl[:, :, 1] += start[1]

    results = {}
    for name, belief in fills.items():
        e_all, sd_all, bad = [], [], 0
        t0 = time.perf_counter()
        for k in range(plans.shape[1]):
            try:
                e, v = plan_moments_conv(
                    belief.astype(np.float32), sigma_base, ctl[:, k, :], rp,
                    b["x0"], b["y0"], CELL, rho1,
                )
            except Exception:  # noqa: BLE001 -- the point is to find out THAT it breaks
                bad += 1
                continue
            if not (np.isfinite(e) and np.isfinite(v)):
                bad += 1
                continue  # keep e=inf out of the spread stats; it still counts toward `bad`
            e_all.append(e)
            sd_all.append(np.sqrt(max(v, 0.0)))
        ms = (time.perf_counter() - t0) / plans.shape[1] * 1e3
        e_all = np.array(e_all)
        results[name] = {
            "ms_per_plan": ms,
            "n_nonfinite_or_raised": bad,
            "E_spread": float(np.nanmax(e_all) - np.nanmin(e_all)) if len(e_all) else float("nan"),
            "E_median": float(np.nanmedian(e_all)) if len(e_all) else float("nan"),
            "sd_median": float(np.nanmedian(sd_all)) if len(sd_all) else float("nan"),
        }
    return {
        "bag": bag.name, "n_clouds": len(clouds), "n_points": int(b["n_points"]),
        "grid": f"{n}x{n} @ {CELL} m",
        "observed_frac": float(observed.mean()),
        "median_hits_per_observed_cell": float(np.median(cnt[observed])),
        "ground_level_m": ground,
        "sigma_observed_median_cm": float(np.median(sigma_base[observed]) * 100),
        "policies": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default="bags/outdoor_experiment_new2")
    ap.add_argument("--clouds", type=int, default=40)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    wp.init()
    res = run(Path(args.bag), args.clouds, args.stride, args.device)
    print(f"=== {res['bag']}: {res['n_clouds']} scans, {res['n_points']} points in the window ===")
    print(f"  grid {res['grid']}, observed {res['observed_frac']:.0%}, "
          f"{res['median_hits_per_observed_cell']:.0f} hits/cell, ground at "
          f"{res['ground_level_m']:+.2f} m")
    print(f"  sigma of observed cells: median {res['sigma_observed_median_cm']:.2f} cm")
    print(f"\n  {'unobserved-cell policy':28s}{'ms/plan':>9}{'broken':>8}{'E median':>10}"
          f"{'E spread':>10}{'sd median':>11}")
    for k, v in res["policies"].items():
        print(f"  {k:28s}{v['ms_per_plan']:9.2f}{v['n_nonfinite_or_raised']:8d}"
              f"{v['E_median']:10.2f}{v['E_spread']:10.2f}{v['sd_median']:11.3f}")
    path = OUT / "bag_belief.json"
    path.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
