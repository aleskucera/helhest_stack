"""Does the SPHERE or the CYLINDER wheel-contact footprint (commit 621a4a0) better predict the
real robot's attitude?

    .venv/bin/python -m studies.bench.footprint_fidelity

NO ROS. Follows `studies.sensing.bag_belief`'s approach: the bag is read with `rosbags`, the
Ouster mount is recovered by fitting a ground plane to the near field (there is no TF for it in
these bags), and the map is built on `/odom_2d` poses with nearest-in-time matching. Unlike
`bag_belief`'s single 9 m window centered on the trajectory mean -- fine for a short loop, wrong
for `stromovka1`'s 35x30 m excursion -- this script builds a small LOCAL patch per query pose from
only the temporally-nearby scans, which also sidesteps global odom-drift blur (a known issue in
this project's map-quality history; see `icp_compare_maps_caveat` in the project memory).

THE HEADLINE RESULT IS A CODE-PATH FINDING, not a terrain-dependent one: `settle()` (step.py) and
the envelope dilation (`envelope.py`) both take `robot.wheel_radius` but never read
`robot.wheel_half_width` -- grep confirms it appears nowhere in their bodies or anything they call.
`wheel_half_width` is read in exactly one place, `normal_loads` (step.py:411-425), which computes
the LOAD SPLIT across wheels, not the settled (z, pitch, roll). So at a FIXED (x, y, yaw) -- which
is what this script evaluates, one query pose at a time, no rollout -- `settle(wheel_half_width=0)`
and `settle(wheel_half_width=0.05)` are the same computation graph and return bit-identical output
on every terrain, not just a flat one. This was confirmed directly (a lateral-tilt synthetic plane,
16.7 deg roll) before running any real data: both conditions returned identical float32 arrays.
The per-pose loop below still calls settle() under both conditions and reports the diff, so that
fact is measured here rather than assumed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import warp as wp

from ..sensing.bag_belief import level_from_ground
from helhest.engine import Grid
from helhest.engine import GridParams
from helhest.engine import Robot
from helhest.engine import RobotParams
from helhest.engine.envelope import _contact_kernel
from helhest.engine.envelope import _gather_kernel
from helhest.engine.step import clearances
from helhest.engine.step import settle
from helhest.engine.step import Solver
from helhest.engine.step import SolverParams

BAG = Path("/home/kuceral4/projects/helhest_stack/bags/stromovka1")
OUT = Path(__file__).resolve().parents[2] / "studies" / "out" / "bench" / "footprint_fidelity.json"

CELL = 0.10  # [m] matches the production perception resolution (studies.bench.ranking.CELL)
LOCAL_WINDOW = 3.0  # [m] square local map patch built fresh around each query pose
CLOUD_STRIDE = 6  # load every 6th /ouster/points message (~1.7 Hz of the ~10 Hz stream)
N_CLOUDS_MAX = 400  # caps loaded clouds (~400 covers the full 232 s trajectory at this stride)
QUERY_STRIDE = 3  # evaluate settle at every 3rd LOADED cloud's pose (~130 query poses)
NEIGHBOR_CLOUDS = 10  # +/- this many LOADED clouds contribute points to one query's local patch
SELF_FILTER = 1.0  # [m] drop points closer than this (robot self-returns), matches bag_belief
MAX_RANGE = 20.0  # [m] matches bag_belief's default
CYLINDER_HALF_WIDTH = 0.05  # [m] ruler-measured half-tread (commit 621a4a0's message)


def read_bag_with_imu(
    path: Path, n_clouds: int, stride: int
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mirrors `bag_belief.read_bag`, extended to also return cloud stamps and the nearest-in-time
    IMU (roll, pitch) from `/imu/data`'s fused orientation. `bag_belief.read_bag` discards the
    per-cloud stamp (it only needs the odom match), so this cannot reuse it as-is -- mirrored
    instead of imported, per the assignment's fallback.

    `/ouster/imu` carries `orientation_covariance[0] == -1` (checked directly against this bag) --
    ROS's "no orientation" sentinel, rates only. `/imu/data`'s AHRS orientation is used instead;
    its yaw is known-broken (project memory: imu_orientation_yaw_broken) but roll/pitch are
    gravity-referenced and are exactly what this script needs -- no gyro integration involved, so
    the 600 dps gyro-spike gate (a deskew/prior concern) does not apply here.

    Returns (clouds, cloud_stamps [N], odom_poses [N,3] (x,y,yaw), imu_roll [N], imu_pitch [N]).
    """
    from rosbags.highlevel import AnyReader

    clouds, stamps = [], []
    odom_t, odom_p = [], []
    imu_t, imu_rp = [], []
    with AnyReader([path]) as reader:
        conns = [
            c
            for c in reader.connections
            if c.topic in ("/ouster/points", "/odom_2d", "/imu/data")
        ]
        seen = 0
        for conn, t, raw in reader.messages(connections=conns):
            if conn.topic == "/odom_2d":
                m = reader.deserialize(raw, conn.msgtype)
                q = m.pose.pose.orientation
                yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
                odom_t.append(t)
                odom_p.append([m.pose.pose.position.x, m.pose.pose.position.y, yaw])
            elif conn.topic == "/imu/data":
                m = reader.deserialize(raw, conn.msgtype)
                q = m.orientation
                sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
                cosr_cosp = 1 - 2 * (q.x**2 + q.y**2)
                roll = np.arctan2(sinr_cosp, cosr_cosp)
                sinp = np.clip(2 * (q.w * q.y - q.z * q.x), -1.0, 1.0)
                pitch = np.arcsin(sinp)
                imu_t.append(t)
                imu_rp.append([roll, pitch])
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
    imu_t = np.array(imu_t)
    imu_rp = np.array(imu_rp)
    stamps = np.array(stamps)

    def nearest(query_t: np.ndarray, ref_t: np.ndarray) -> np.ndarray:
        """Index into `ref_t` nearest each `query_t`, same before/after tie-break as bag_belief."""
        after = np.clip(np.searchsorted(ref_t, query_t), 0, len(ref_t) - 1)
        before = np.clip(after - 1, 0, len(ref_t) - 1)
        return np.where(
            np.abs(ref_t[before] - query_t) <= np.abs(ref_t[after] - query_t), before, after
        )

    odom_idx = nearest(stamps, odom_t)
    imu_idx = nearest(stamps, imu_t)
    return clouds, stamps, odom_p[odom_idx], imu_rp[imu_idx, 0], imu_rp[imu_idx, 1]


def local_elevation(
    clouds: list[np.ndarray],
    poses: np.ndarray,
    tf: np.ndarray,
    lo: int,
    hi: int,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, float, float, int]:
    """Rasterize clouds[lo:hi] (transformed to world via `tf` + each cloud's own odom pose) into a
    `LOCAL_WINDOW`-square, `CELL`-resolution mean-height grid centered at (cx, cy). Cells with no
    points are NaN -- NOT filled at 0 or the local median -- so a wheel that lands on one is
    visible downstream (settle's bilinear sample propagates the NaN; see `run`'s exclusion pass).
    Mirrors `bag_belief.build_real_belief`'s transform + `np.add.at` binning, re-centered per pose
    instead of once on the trajectory mean.
    """
    n = int(round(LOCAL_WINDOW / CELL))
    x0, y0 = cx - LOCAL_WINDOW / 2, cy - LOCAL_WINDOW / 2
    pts_world = []
    for cloud, (px, py, yaw) in zip(clouds[lo:hi], poses[lo:hi]):
        r = np.linalg.norm(cloud[:, :2], axis=1)
        cloud = cloud[(r > SELF_FILTER) & (r < MAX_RANGE)]
        p = (tf[:3, :3] @ cloud.T).T + tf[:3, 3]
        c, s = np.cos(yaw), np.sin(yaw)
        pts_world.append(
            np.stack([c * p[:, 0] - s * p[:, 1] + px, s * p[:, 0] + c * p[:, 1] + py, p[:, 2]], 1)
        )
    pts = np.concatenate(pts_world) if pts_world else np.zeros((0, 3))
    inside = (
        (pts[:, 0] >= x0) & (pts[:, 0] < x0 + LOCAL_WINDOW)
        & (pts[:, 1] >= y0) & (pts[:, 1] < y0 + LOCAL_WINDOW)
    )
    pts = pts[inside]
    ix = np.clip(((pts[:, 0] - x0) / CELL).astype(int), 0, n - 1)
    iy = np.clip(((pts[:, 1] - y0) / CELL).astype(int), 0, n - 1)
    s1 = np.zeros((n, n))
    cnt = np.zeros((n, n))
    np.add.at(s1, (iy, ix), pts[:, 2])
    np.add.at(cnt, (iy, ix), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, s1 / np.maximum(cnt, 1.0), np.nan)
    observed_frac = float((cnt > 0).mean())
    return mean.astype(np.float32), x0, y0, observed_frac


@wp.kernel
def _settle_and_resid_kernel(
    envelope: wp.array2d(dtype=wp.float32),
    grid: Grid,
    robot: Robot,
    solver: Solver,
    controlled: wp.array(dtype=wp.vec3),
    derived_init: wp.array(dtype=wp.vec3),
    derived_out: wp.array(dtype=wp.vec3),
    resid_out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    d = settle(envelope, grid, robot, solver, controlled[tid], derived_init[tid])
    derived_out[tid] = d
    c = clearances(
        envelope, grid, robot, controlled[tid][0], controlled[tid][1], controlled[tid][2],
        d[0], d[1], d[2],
    )
    resid_out[tid] = wp.max(wp.max(wp.abs(c[0]), wp.abs(c[1])), wp.abs(c[2]))


def run(device: str) -> dict:
    wp.init()
    clouds, stamps, poses, imu_roll, imu_pitch = read_bag_with_imu(BAG, N_CLOUDS_MAX, CLOUD_STRIDE)
    tf = level_from_ground(clouds[0])

    n = int(round(LOCAL_WINDOW / CELL))
    env_radius = int(np.ceil(RobotParams().wheel_radius / CELL))
    solver = SolverParams(newton_iters=20, atol=1e-8).build()
    robots = {
        # Post-merge (bbc98cf) the knob is wheel_width, and it switches the ENVELOPE too, not
        # only the load moment arm -- the original finding ("bit-identical settle") no longer
        # holds by construction. Sphere-vs-cylinder attitude is now a real comparison, still
        # blocked on bag quality (see the module docstring's negative result).
        "sphere": RobotParams(wheel_width=None).build(device=device),
        "cylinder": RobotParams(wheel_width=2 * CYLINDER_HALF_WIDTH).build(device=device),
    }

    with wp.ScopedDevice(device):
        elevation = wp.zeros((n, n), dtype=wp.float32)
        contact_iy = wp.zeros((n, n), dtype=wp.int32)
        contact_ix = wp.zeros((n, n), dtype=wp.int32)
        cap = wp.zeros((n, n), dtype=wp.float32)
        envelope = wp.zeros((n, n), dtype=wp.float32)
        controlled = wp.zeros(1, dtype=wp.vec3)
        derived_init = wp.zeros(1, dtype=wp.vec3)
        derived_out = wp.zeros(1, dtype=wp.vec3)
        resid_out = wp.zeros(1, dtype=wp.float32)

    query_idx = list(range(0, len(clouds), QUERY_STRIDE))
    rows = []
    n_excluded_unobserved = 0
    n_excluded_resid = 0
    for qi in query_idx:
        x, y, yaw = poses[qi]
        lo, hi = max(0, qi - NEIGHBOR_CLOUDS), min(len(clouds), qi + NEIGHBOR_CLOUDS + 1)
        local_elev, x0, y0, observed_frac = local_elevation(clouds, poses, tf, lo, hi, x, y)
        grid = GridParams(n, n, CELL, x0, y0).build()

        elevation.assign(local_elev)
        wp.launch(_contact_kernel, dim=(n, n),
                  inputs=[elevation, CELL, RobotParams().wheel_radius, env_radius],
                  outputs=[contact_iy, contact_ix, cap], device=device)
        wp.launch(_gather_kernel, dim=(n, n),
                  inputs=[elevation, contact_iy, contact_ix, cap],
                  outputs=[envelope], device=device)

        ground_guess = np.nanmedian(local_elev) if np.isfinite(local_elev).any() else 0.0
        z0 = float(ground_guess) + RobotParams().wheel_radius
        controlled.assign(np.array([[x, y, yaw]], np.float32))
        derived_init.assign(np.array([[z0, 0.0, 0.0]], np.float32))

        row = {
            "i": int(qi), "t": float(stamps[qi]), "x": float(x), "y": float(y), "yaw": float(yaw),
            "imu_roll": float(imu_roll[qi]), "imu_pitch": float(imu_pitch[qi]),
            "observed_frac": observed_frac,
        }
        finite_ok = True
        resid_ok = True
        for name, robot in robots.items():
            wp.launch(_settle_and_resid_kernel, 1,
                      inputs=[envelope, grid, robot, solver, controlled, derived_init],
                      outputs=[derived_out, resid_out], device=device)
            d = derived_out.numpy()[0]
            resid = float(resid_out.numpy()[0])
            row[f"{name}_z"] = float(d[0])
            row[f"{name}_pitch"] = float(d[1])
            row[f"{name}_roll"] = float(d[2])
            row[f"{name}_resid"] = resid
            if not np.all(np.isfinite(d)):
                finite_ok = False
            # `resid > tol` is silently False for NaN (IEEE754), which would let a diverged
            # settle (pinned at the tilt clamp, e.g. from too few observed cells) through
            # uncaught -- require finiteness explicitly.
            if not (np.isfinite(resid) and resid <= RobotParams().resid_tol):
                resid_ok = False
        if not finite_ok:
            n_excluded_unobserved += 1
            continue
        if not resid_ok:
            n_excluded_resid += 1
            continue
        rows.append(row)

    return {
        "rows": rows,
        "n_queries_attempted": len(query_idx),
        "n_excluded_wheel_on_unobserved": n_excluded_unobserved,
        "n_excluded_resid": n_excluded_resid,
        "n_clouds_loaded": len(clouds),
        "config": {
            "bag": BAG.name, "cell": CELL, "local_window": LOCAL_WINDOW,
            "cloud_stride": CLOUD_STRIDE, "n_clouds_max": N_CLOUDS_MAX,
            "query_stride": QUERY_STRIDE, "neighbor_clouds": NEIGHBOR_CLOUDS,
            "self_filter": SELF_FILTER, "max_range": MAX_RANGE,
            "cylinder_half_width": CYLINDER_HALF_WIDTH,
        },
    }


def _two_sided_binom_test(n_pos: int, n_trials: int, p: float = 0.5) -> float:
    """Exact two-sided sign-test p-value: sum P(k) over all k at least as extreme (as-or-less
    likely) as the observed n_pos, under Binomial(n_trials, p). No scipy in this venv."""
    if n_trials == 0:
        return 1.0
    pmf = [math.comb(n_trials, k) * p**k * (1 - p) ** (n_trials - k) for k in range(n_trials + 1)]
    observed = pmf[n_pos]
    return float(sum(prob for prob in pmf if prob <= observed + 1e-12))


def _align_axes(
    pred_pitch: np.ndarray, pred_roll: np.ndarray, meas_roll: np.ndarray, meas_pitch: np.ndarray
) -> dict:
    """Correlate {pred_pitch, pred_roll} against {meas_roll, meas_pitch} (both signs), and pick
    the pred->measured axis assignment + sign that maximizes total |correlation|. Returns the
    chosen alignment plus the full evidence matrix so the choice can be audited."""
    pairs = {
        "pred_pitch:meas_roll": np.corrcoef(pred_pitch, meas_roll)[0, 1],
        "pred_pitch:meas_pitch": np.corrcoef(pred_pitch, meas_pitch)[0, 1],
        "pred_roll:meas_roll": np.corrcoef(pred_roll, meas_roll)[0, 1],
        "pred_roll:meas_pitch": np.corrcoef(pred_roll, meas_pitch)[0, 1],
    }
    identity = abs(pairs["pred_pitch:meas_pitch"]) + abs(pairs["pred_roll:meas_roll"])
    swap = abs(pairs["pred_pitch:meas_roll"]) + abs(pairs["pred_roll:meas_pitch"])
    if identity >= swap:
        pitch_match, roll_match = "meas_pitch", "meas_roll"
        pitch_corr, roll_corr = pairs["pred_pitch:meas_pitch"], pairs["pred_roll:meas_roll"]
    else:
        pitch_match, roll_match = "meas_roll", "meas_pitch"
        pitch_corr, roll_corr = pairs["pred_pitch:meas_roll"], pairs["pred_roll:meas_pitch"]
    return {
        "correlation_matrix": pairs,
        "pred_pitch_matches": pitch_match, "pred_pitch_corr": float(pitch_corr),
        "pred_pitch_sign": float(np.sign(pitch_corr) or 1.0),
        "pred_roll_matches": roll_match, "pred_roll_corr": float(roll_corr),
        "pred_roll_sign": float(np.sign(roll_corr) or 1.0),
    }


def analyze(result: dict) -> dict:
    rows = result["rows"]
    n = len(rows)
    out: dict = {"n_poses": n}
    if n == 0:
        out["sanity_gate"] = "FAILED: zero poses survived exclusion -- nothing to analyze"
        return out

    imu_roll = np.array([r["imu_roll"] for r in rows])
    imu_pitch = np.array([r["imu_pitch"] for r in rows])
    # sphere and cylinder settle predictions are bit-identical (see module docstring); pick sphere
    # as THE model under test and separately confirm the equality numerically below.
    pred_pitch = np.array([r["sphere_pitch"] for r in rows])
    pred_roll = np.array([r["sphere_roll"] for r in rows])

    align = _align_axes(pred_pitch, pred_roll, imu_roll, imu_pitch)
    out["axis_alignment"] = align

    best_axis_corr = max(abs(align["pred_pitch_corr"]), abs(align["pred_roll_corr"]))
    if best_axis_corr < 0.5:
        out["sanity_gate"] = (
            f"FAILED: best-axis correlation {best_axis_corr:.3f} < 0.5 -- the settled attitude "
            "does not track the measured one. Diagnosis needed (map quality / pose alignment / "
            "settle misuse) before trusting any error metric below."
        )
        return out
    out["sanity_gate"] = f"PASSED: best-axis correlation {best_axis_corr:.3f} >= 0.5"

    meas_for_pitch = imu_pitch if align["pred_pitch_matches"] == "meas_pitch" else imu_roll
    meas_for_roll = imu_roll if align["pred_roll_matches"] == "meas_roll" else imu_pitch
    pitch_aligned = align["pred_pitch_sign"] * pred_pitch
    roll_aligned = align["pred_roll_sign"] * pred_roll

    pitch_offset = float(np.mean(meas_for_pitch - pitch_aligned))
    roll_offset = float(np.mean(meas_for_roll - roll_aligned))
    out["offsets_removed_rad"] = {"pitch": pitch_offset, "roll": roll_offset}
    out["offsets_removed_deg"] = {
        "pitch": float(np.degrees(pitch_offset)), "roll": float(np.degrees(roll_offset))
    }

    def metrics_for(condition: str) -> dict:
        p_pitch = align["pred_pitch_sign"] * np.array([r[f"{condition}_pitch"] for r in rows])
        p_roll = align["pred_roll_sign"] * np.array([r[f"{condition}_roll"] for r in rows])
        err_pitch = np.degrees((p_pitch + pitch_offset) - meas_for_pitch)
        err_roll = np.degrees((p_roll + roll_offset) - meas_for_roll)
        return {
            "pitch_rms_deg": float(np.sqrt(np.mean(err_pitch**2))),
            "pitch_median_abs_deg": float(np.median(np.abs(err_pitch))),
            "roll_rms_deg": float(np.sqrt(np.mean(err_roll**2))),
            "roll_median_abs_deg": float(np.median(np.abs(err_roll))),
            "_err_pitch_deg": err_pitch, "_err_roll_deg": err_roll,
        }

    sphere_m = metrics_for("sphere")
    cyl_m = metrics_for("cylinder")
    out["overall"] = {
        "sphere": {k: v for k, v in sphere_m.items() if not k.startswith("_")},
        "cylinder": {k: v for k, v in cyl_m.items() if not k.startswith("_")},
    }
    out["sphere_cylinder_bit_identical"] = bool(
        np.array_equal(sphere_m["_err_pitch_deg"], cyl_m["_err_pitch_deg"])
        and np.array_equal(sphere_m["_err_roll_deg"], cyl_m["_err_roll_deg"])
    )

    # stratify by measured |lateral tilt| terciles -- the pre-registered prediction's own axis
    meas_abs_roll = np.abs(meas_for_roll)
    edges = np.percentile(meas_abs_roll, [100 / 3, 200 / 3])
    tercile_labels = np.digitize(meas_abs_roll, edges)  # 0=low,1=mid,2=high
    out["by_tilt_tercile"] = {}
    for k, name in enumerate(("low", "mid", "high")):
        mask = tercile_labels == k
        if mask.sum() == 0:
            continue
        out["by_tilt_tercile"][name] = {
            "n": int(mask.sum()),
            "meas_abs_roll_deg_range": [
                float(np.degrees(meas_abs_roll[mask].min())),
                float(np.degrees(meas_abs_roll[mask].max())),
            ],
            "sphere_roll_rms_deg": float(np.sqrt(np.mean(sphere_m["_err_roll_deg"][mask] ** 2))),
            "cylinder_roll_rms_deg": float(np.sqrt(np.mean(cyl_m["_err_roll_deg"][mask] ** 2))),
            "sphere_pitch_rms_deg": float(np.sqrt(np.mean(sphere_m["_err_pitch_deg"][mask] ** 2))),
            "cylinder_pitch_rms_deg": float(np.sqrt(np.mean(cyl_m["_err_pitch_deg"][mask] ** 2))),
        }

    # paired sign test on |err_sphere| - |err_cylinder|, combined roll+pitch magnitude per pose
    combined_sphere = np.hypot(sphere_m["_err_pitch_deg"], sphere_m["_err_roll_deg"])
    combined_cyl = np.hypot(cyl_m["_err_pitch_deg"], cyl_m["_err_roll_deg"])
    diff = combined_sphere - combined_cyl
    n_pos, n_neg, n_tie = int((diff > 0).sum()), int((diff < 0).sum()), int((diff == 0).sum())
    p_value = _two_sided_binom_test(n_pos, n_pos + n_neg)
    out["paired_sign_test"] = {
        "n_sphere_worse": n_pos, "n_cylinder_worse": n_neg, "n_tied": n_tie, "p_value": p_value,
        "note": "n_tied == n_poses is expected: settle() never reads wheel_half_width (see module "
        "docstring); this is a code-path fact, not a terrain-dependent measurement.",
    }
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    result = run(args.device)
    analysis = analyze(result)
    payload = {**result, "analysis": {k: v for k, v in analysis.items()}}
    # strip the large per-error arrays stashed under keys the caller doesn't need to persist
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, default=lambda o: o.tolist()))

    print(f"=== footprint_fidelity: {result['config']['bag']} ===")
    print(f"  {len(result['rows'])}/{result['n_queries_attempted']} query poses kept "
          f"({result['n_excluded_wheel_on_unobserved']} excluded: wheel over unobserved/NaN cells; "
          f"{result['n_excluded_resid']} excluded: settle residual above tol)")
    print(f"  sanity gate: {analysis.get('sanity_gate')}")
    if "axis_alignment" in analysis:
        a = analysis["axis_alignment"]
        print(f"  axis alignment: pred_pitch -> {a['pred_pitch_matches']} "
              f"(r={a['pred_pitch_corr']:+.2f}), pred_roll -> {a['pred_roll_matches']} "
              f"(r={a['pred_roll_corr']:+.2f})")
    if "overall" in analysis:
        print(f"  offsets removed [deg]: {analysis['offsets_removed_deg']}")
        print(f"  bit-identical sphere vs cylinder: {analysis['sphere_cylinder_bit_identical']}")
        print(
            f"\n  {'':10s}{'roll RMS':>10}{'roll med|e|':>13}{'pitch RMS':>11}{'pitch med|e|':>14}"
        )
        for cond in ("sphere", "cylinder"):
            m = analysis["overall"][cond]
            print(f"  {cond:10s}{m['roll_rms_deg']:10.2f}{m['roll_median_abs_deg']:13.2f}"
                  f"{m['pitch_rms_deg']:11.2f}{m['pitch_median_abs_deg']:14.2f}")
        print("\n  by measured |roll| tercile (roll RMS deg, sphere vs cylinder):")
        for name, t in analysis["by_tilt_tercile"].items():
            print(f"    {name:5s} n={t['n']:4d} range={t['meas_abs_roll_deg_range'][0]:.1f}"
                  f"-{t['meas_abs_roll_deg_range'][1]:.1f} deg   "
                  f"sphere={t['sphere_roll_rms_deg']:.2f}  "
                  f"cylinder={t['cylinder_roll_rms_deg']:.2f}")
        s = analysis["paired_sign_test"]
        print(f"\n  paired sign test: sphere worse={s['n_sphere_worse']} "
              f"cylinder worse={s['n_cylinder_worse']} tied={s['n_tied']} p={s['p_value']:.3f}")
        print(f"  {s['note']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
