"""Friction saturation along the trajectories the robot ACTUALLY drove.

Run:  python scripts/saturation_from_bags.py bags/out_experiment_goal_unreachable0 [more bags]

Screening test for whether a better friction model could change anything (IMPROVEMENTS.md
section 10). The certificate needs only tilt, body twist and mu, and all three are already in the
bag -- no terrain, no perception, no simulator:

    demand_long = m g sin(pitch)
    demand_lat  = m v psi_dot + m g cos(pitch) sin(roll)
    saturation  = hypot(demand_long, demand_lat) / (mu m g cos(pitch) cos(roll))

Tilt comes from the IMU's fused orientation, cross-checked against the low-passed accelerometer
(vehicle acceleration averages out over seconds, gravity does not); the disagreement between the
two is printed so a bad orientation is obvious. Only roll and pitch are taken from it -- the fused
YAW on this robot is wrong-signed and is not used here. mu is unknown per terrain, so it is swept.

READ THE RESULT AS A LOWER BOUND. These are trajectories the robot drove and survived, so they are
biased toward the feasible; the planner's candidate set contains worse. "Saturation is low even
here" is therefore a valid stop signal, while "saturation is low" alone does not prove that
candidate trajectories never bind.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

MASS, GRAVITY = 106.2, 9.81
IMU_TOPICS = ("/imu/data", "/odin1/imu")
ODOM_TOPICS = ("/odom_2d", "/odin1/odometry")
MAX_GYRO = np.radians(600.0)  # /imu/data carries single-sample glitch spikes; gate them
MU_SWEEP = (0.2, 0.3, 0.4, 0.6, 0.8)


def _quat_roll_pitch(x: float, y: float, z: float, w: float) -> tuple[float, float]:
    """Roll and pitch from a quaternion. Yaw is deliberately ignored (unreliable on this robot)."""
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return float(roll), float(pitch)


def _load(bag: Path) -> dict[str, np.ndarray]:
    out: dict[str, list] = {k: [] for k in ("t", "roll", "pitch", "acc", "wz", "ot", "v")}
    with AnyReader([bag]) as reader:
        wanted = set(IMU_TOPICS) | set(ODOM_TOPICS)
        conns = [c for c in reader.connections if c.topic in wanted]
        for conn, stamp, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic in IMU_TOPICS:
                q = msg.orientation
                roll, pitch = _quat_roll_pitch(q.x, q.y, q.z, q.w)
                acc, gyro = msg.linear_acceleration, msg.angular_velocity
                out["t"].append(stamp * 1e-9)
                out["roll"].append(roll)
                out["pitch"].append(pitch)
                out["acc"].append([acc.x, acc.y, acc.z])
                out["wz"].append(gyro.z)
            else:
                out["ot"].append(stamp * 1e-9)
                out["v"].append(msg.twist.twist.linear.x)
    return {k: np.asarray(v, float) for k, v in out.items()}


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window < 2:
        return x
    kernel = np.ones(window) / window
    if x.ndim == 1:
        return np.convolve(x, kernel, "same")
    return np.stack([np.convolve(x[:, i], kernel, "same") for i in range(x.shape[1])], axis=1)


def analyse(bag: Path) -> None:
    data = _load(bag)
    if data["t"].size == 0:
        print(f"{bag.name}: no IMU topic among {IMU_TOPICS}")
        return
    hz = 1.0 / np.median(np.diff(data["t"]))
    roll, pitch = data["roll"], data["pitch"]

    # tilt from the gravity direction, as an independent check on the fused orientation
    acc = _smooth(data["acc"], max(int(2.0 * hz), 1))
    norm = np.linalg.norm(acc, axis=1)
    good = norm > 1e-6
    acc_pitch = np.arcsin(np.clip(acc[good, 0] / norm[good], -1.0, 1.0))
    acc_roll = -np.arcsin(np.clip(acc[good, 1] / norm[good], -1.0, 1.0))
    disagreement = np.degrees(np.hypot(roll[good] - acc_roll, pitch[good] - acc_pitch))

    print(
        f"=== {bag.name}   {len(data['t'])} imu samples @ {hz:.0f} Hz, "
        f"{data['t'].max() - data['t'].min():.0f} s"
    )
    print(
        f"  tilt: |pitch| max {np.degrees(np.abs(pitch)).max():5.1f} deg, "
        f"|roll| max {np.degrees(np.abs(roll)).max():5.1f} deg; "
        f"fused-vs-accelerometer median disagreement {np.median(disagreement):.1f} deg"
    )

    yaw_rate = np.clip(data["wz"], -MAX_GYRO, MAX_GYRO)
    has_odom = data["ot"].size > 10
    speed = np.interp(data["t"], data["ot"], data["v"]) if has_odom else np.zeros_like(data["t"])
    if not has_odom:
        print("  no odometry topic -- centripetal term unavailable, gravity terms only")
    else:
        print(f"  speed: median {np.median(np.abs(speed)):.2f}, max {np.abs(speed).max():.2f} m/s")

    demand_long = MASS * GRAVITY * np.sin(pitch)
    demand_lat = MASS * speed * yaw_rate + MASS * GRAVITY * np.cos(pitch) * np.sin(roll)
    demand = np.hypot(demand_long, demand_lat)
    support = MASS * GRAVITY * np.cos(pitch) * np.cos(roll)
    print(f"  {'mu':>5} {'median':>8} {'p95':>8} {'p99':>8} {'max':>8}   {'>0.5':>7} {'>1.0':>7}")
    for mu in MU_SWEEP:
        sat = demand / (mu * support)
        print(
            f"  {mu:5.2f} {np.median(sat):8.3f} {np.percentile(sat, 95):8.3f} "
            f"{np.percentile(sat, 99):8.3f} {sat.max():8.3f}   "
            f"{100 * (sat > 0.5).mean():6.2f}% {100 * (sat > 1.0).mean():6.2f}%"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    for name in sys.argv[1:]:
        analyse(Path(name))
