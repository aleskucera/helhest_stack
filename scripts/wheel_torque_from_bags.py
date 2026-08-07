"""Calibrate /joint_states.effort to Nm at the wheel, and bound the drivetrain torque envelope.

Run:  python scripts/wheel_torque_from_bags.py bags/out_experiment_goal_unreachable0 [more bags]

The LLC publishes an `effort` field on /joint_states in an undocumented raw unit. Two
independent estimators pin it, both from Newton's law along the body x axis:

    sum_i tau_i / wheel_radius  =  m * a_x  (+ rolling resistance, absorbed by the fit offset)

  IMU  -- a_x is the accelerometer's SPECIFIC force, which already contains the gravity
          component, so the fit is valid on grades as well as on the flat.
  ODOM -- a_x is d/dt of the wheel-kinematic forward speed. No accelerometer involved, so
          agreement between the two is a real cross-check rather than a restatement.

Only straight-line, moving frames are used: turning adds skid scrub and a lever arm that this
one-axis balance does not model.

CAVEAT ON BAG AGE. Bags older than 2026-07-14 give a NEGATIVE correlation here (the IMU was
remounted), and bags older than 2026-07-27 have the /cmd_joints units bug, which corrupts the
odometry estimator specifically. Use post-2026-07-27 bags; the script prints the correlation so a
bad one is obvious.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

JOINTS = ["left_wheel_j", "right_wheel_j", "rear_wheel_j"]
MASS = 106.2  # [kg] robot total (model.py / RobotParams)
WHEEL_RADIUS = 0.35  # [m]
STRAIGHT_YAW_RATE = 0.15  # [rad/s] above this the frame is turning, so it is dropped
MOVING_OMEGA = 0.3  # [rad/s] wheel speed below which the frame is not driving


def _load(bag: Path) -> dict[str, np.ndarray]:
    """Pull effort/velocity, IMU and odometry out of one bag, as parallel arrays."""
    out: dict[str, list] = {k: [] for k in ("t", "eff", "vel", "imu_t", "ax", "wz", "od_t", "od_v")}
    with AnyReader([bag]) as reader:
        topics = {"/joint_states", "/imu/data", "/odom_2d"}
        conns = [c for c in reader.connections if c.topic in topics]
        order = None
        for conn, stamp, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            t = stamp * 1e-9
            if conn.topic == "/joint_states" and len(msg.effort) >= 3:
                if order is None:
                    order = [list(msg.name).index(j) for j in JOINTS]
                out["t"].append(t)
                out["eff"].append([msg.effort[i] for i in order])
                out["vel"].append([msg.velocity[i] for i in order])
            elif conn.topic == "/imu/data":
                out["imu_t"].append(t)
                out["ax"].append(msg.linear_acceleration.x)
                out["wz"].append(msg.angular_velocity.z)
            elif conn.topic == "/odom_2d":
                out["od_t"].append(t)
                out["od_v"].append(msg.twist.twist.linear.x)
    return {k: np.asarray(v, float) for k, v in out.items()}


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(x, np.ones(window) / window, "same") if window > 1 else x


def _fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares y = slope*x + offset, plus the correlation."""
    A = np.stack([x, np.ones_like(x)], axis=1)
    (slope, offset), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(slope), float(offset), float(np.corrcoef(x, y)[0, 1])


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window < 2 or len(x) < window:
        return x
    cumulative = np.cumsum(np.insert(x, 0, 0.0, axis=0), axis=0)
    return (cumulative[window:] - cumulative[:-window]) / window


def analyse(bag: Path) -> None:
    data = _load(bag)
    if data["t"].size == 0 or data["imu_t"].size == 0:
        print(f"{bag.name}: no /joint_states effort or no /imu/data")
        return
    finite = np.isfinite(data["eff"]).all(1)
    t, eff, vel = data["t"][finite], data["eff"][finite], data["vel"][finite]
    dt = float(np.median(np.diff(t)))
    effort_sum = _smooth(eff.sum(1), max(int(0.2 / dt), 1))

    ax = np.interp(t, data["imu_t"], _smooth(data["ax"], 20))
    yaw_rate = np.interp(t, data["imu_t"], _smooth(data["wz"], 20))
    straight = (np.abs(yaw_rate) < STRAIGHT_YAW_RATE) & (np.abs(vel).max(1) > MOVING_OMEGA)
    if straight.sum() < 100:
        print(f"{bag.name}: only {straight.sum()} straight-line frames, skipping")
        return

    imu_slope, imu_offset, imu_corr = _fit(MASS * ax[straight] * WHEEL_RADIUS, effort_sum[straight])
    print(f"=== {bag.name}   {straight.sum()} straight frames of {len(t)}")
    print(
        f"  IMU   1 Nm = {imu_slope:6.2f} raw  (corr {imu_corr:+.3f}, offset {imu_offset:5.0f} raw "
        f"= {imu_offset / imu_slope:5.1f} Nm of rolling resistance)"
    )
    if data["od_t"].size > 50:
        speed = _smooth(np.interp(t, data["od_t"], data["od_v"]), max(int(0.2 / dt), 1))
        od_slope, _, od_corr = _fit(
            MASS * np.gradient(speed, t)[straight] * WHEEL_RADIUS, effort_sum[straight]
        )
        print(f"  ODOM  1 Nm = {od_slope:6.2f} raw  (corr {od_corr:+.3f})")

    if imu_corr < 0.5:
        print("  correlation too low to convert -- wrong bag era? (see the module docstring)")
        return
    torque = np.abs(eff) / imu_slope
    sustained = _rolling_mean(torque, max(int(1.0 / dt), 1))
    print(
        f"  {'wheel':14s} {'p50':>7} {'p95':>7} {'p99':>7} {'max':>7}   {'1s p99':>7} {'1s max':>7}"
    )
    for i, joint in enumerate(JOINTS):
        a = torque[:, i]
        print(
            f"  {joint:14s} {np.percentile(a, 50):7.1f} {np.percentile(a, 95):7.1f} "
            f"{np.percentile(a, 99):7.1f} {a.max():7.1f}   "
            f"{np.percentile(sustained[:, i], 99):7.1f} {sustained[:, i].max():7.1f}   [Nm]"
        )
    near_max = (torque > 0.95 * torque.max(0)).mean(0) * 100
    print(f"  fraction of samples within 5% of each wheel's peak: {near_max.round(2)} %")
    print("  (a plateau there would BE the envelope; without one these are lower bounds)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    for name in sys.argv[1:]:
        analyse(Path(name))
