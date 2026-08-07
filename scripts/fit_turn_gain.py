"""Fit the turning model (alpha, and hence k_turn / plan_turn_boost) against real bags.

Run:  python scripts/fit_turn_gain.py bags/out_odin0 bags/in_speed_odin0

The engine turns by

    psi_dot = wheel_radius (wR - wL) / (2 half_track alpha),   alpha = 1 + k_turn mu

so alpha is whatever makes the model's yaw rate match the measured one. This fits it by least
squares through the origin against a gyro, which is independent of the wheels.

TWO DIFFERENT ALPHAS, and which one you get depends on the bag:

  from COMMANDED wheel speeds -- what the planner needs, because it predicts the robot's response
      to a command. It absorbs the drivetrain's own differential loss, so it is not a terrain
      property and it moves if the LLC is retuned.
  from MEASURED wheel speeds (/joint_states) -- the vehicle model proper, separating what the
      wheels did from what the body did.

The Odin bags carry no /joint_states, so only the first is available there. Where both exist the
script prints both and their ratio, which IS the drivetrain realization factor.
"""

from __future__ import annotations

import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

WHEEL_RADIUS, HALF_TRACK = 0.35, 0.365
JOINTS = ("left_wheel_j", "right_wheel_j", "rear_wheel_j")
IMU_TOPICS = ("/imu/data", "/odin1/imu", "/ouster/imu")
ODOM_TOPICS = ("/odom_2d", "/odin1/odometry")
MIN_DIFFERENTIAL = 0.5  # [rad/s] commanded (wR - wL) below this is noise, not a turn
MAX_GYRO = np.radians(600.0)
# Before commit f056dcc (2026-07-27) the LLC read /cmd_joints through a x22.5/2pi scaling, so
# commanded wheel speeds in older bags are NOT wheel rad/s and any alpha fitted from them is wrong
# by that factor. Refuse rather than silently mis-fit; multiply by 2pi/22.5 first if you must.
CMD_UNITS_FIX = datetime(2026, 7, 27, tzinfo=timezone.utc).timestamp()


def _load(bag: Path) -> dict:
    out: dict[str, list] = {k: [] for k in ("gt", "gyro", "ct", "cmd", "st", "meas", "ot", "v")}
    with AnyReader([bag]) as reader:
        started = reader.start_time * 1e-9
        if started < CMD_UNITS_FIX:
            when = datetime.fromtimestamp(started, timezone.utc).date()
            raise SystemExit(
                f"{bag.name}: recorded {when}, before the 2026-07-27 /cmd_joints units fix -- "
                "its commanded wheel speeds are not rad/s and alpha would be wrong. Refusing."
            )
        topics = set(IMU_TOPICS) | set(ODOM_TOPICS) | {"/cmd_joints", "/joint_states"}
        conns = [c for c in reader.connections if c.topic in topics]
        imu_topic = next((t for t in IMU_TOPICS if any(c.topic == t for c in conns)), None)
        for conn, stamp, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            t = stamp * 1e-9
            if conn.topic == imu_topic:
                out["gt"].append(t)
                g = msg.angular_velocity
                out["gyro"].append([g.x, g.y, g.z])
            elif conn.topic in ODOM_TOPICS:
                out["ot"].append(t)
                out["v"].append(msg.twist.twist.linear.x)
            elif conn.topic in ("/cmd_joints", "/joint_states") and len(msg.velocity) >= 3:
                idx = [list(msg.name).index(j) for j in JOINTS]
                key = "ct" if conn.topic == "/cmd_joints" else "st"
                out[key].append(t)
                out["cmd" if key == "ct" else "meas"].append([msg.velocity[i] for i in idx])
    return {k: np.asarray(v, float) for k, v in out.items()}


def _yaw_axis(gyro: np.ndarray, reference: np.ndarray) -> tuple[int, float]:
    """Pick the gyro axis that actually carries yaw (this fleet has both z-up and -y mountings)."""
    scores = [np.corrcoef(gyro[:, i], reference)[0, 1] for i in range(3)]
    best = int(np.argmax(np.abs(scores)))
    return best, float(scores[best])


def _lag_seconds(command: np.ndarray, response: np.ndarray, dt: float, max_lag: float = 1.0):
    """Delay of `response` behind `command`, by maximising correlation over integer shifts."""
    best_shift, best_corr = 0, -2.0
    for shift in range(int(max_lag / dt)):
        end = len(command) - shift
        corr = float(np.corrcoef(command[:end], response[shift:])[0, 1])
        if corr > best_corr:
            best_shift, best_corr = shift, corr
    return best_shift * dt


def _fit_alpha(diff_omega: np.ndarray, yaw_rate: np.ndarray) -> tuple[float, float, int]:
    """alpha from psi_dot = R dw / (2 b alpha), by least squares through the origin."""
    model_rate = WHEEL_RADIUS * diff_omega / (2.0 * HALF_TRACK)
    keep = np.abs(diff_omega) > MIN_DIFFERENTIAL
    if keep.sum() < 50:
        return float("nan"), float("nan"), int(keep.sum())
    x, y = model_rate[keep], yaw_rate[keep]
    slope = float(x @ y / (x @ x))  # y = slope * x, slope = 1 / alpha
    corr = float(np.corrcoef(x, y)[0, 1])
    return (1.0 / slope if slope else float("nan")), corr, int(keep.sum())


def analyse(bag: Path) -> None:
    d = _load(bag)
    if d["gt"].size == 0 or d["ct"].size == 0:
        print(f"{bag.name}: needs an IMU and /cmd_joints")
        return
    dt = float(np.median(np.diff(d["gt"])))
    odom_rate = np.interp(d["gt"], d["ot"], d["v"]) if d["ot"].size > 10 else np.zeros_like(d["gt"])
    cmd = np.stack([np.interp(d["gt"], d["ct"], d["cmd"][:, i]) for i in range(3)], 1)
    # the commanded differential is the cleanest reference for which gyro axis carries yaw
    axis, score = _yaw_axis(d["gyro"], cmd[:, 1] - cmd[:, 0])
    yaw_rate = np.clip(d["gyro"][:, axis], -MAX_GYRO, MAX_GYRO)
    if score < 0:
        yaw_rate = -yaw_rate

    diff_cmd = cmd[:, 1] - cmd[:, 0]
    lag = _lag_seconds(diff_cmd, yaw_rate, dt)
    shift = int(round(lag / dt))
    if shift:
        diff_cmd = diff_cmd[: len(diff_cmd) - shift]
        fwd_cmd = (cmd[: len(cmd) - shift, 0] + cmd[: len(cmd) - shift, 1]) / 2.0
        yaw_rate = yaw_rate[shift:]
        odom_rate = odom_rate[shift:]
    else:
        fwd_cmd = (cmd[:, 0] + cmd[:, 1]) / 2.0

    alpha, corr, n = _fit_alpha(diff_cmd, yaw_rate)
    print(
        f"=== {bag.name}   gyro axis {'xyz'[axis]}{'(negated)' if score < 0 else ''}, "
        f"command lag {lag * 1000:.0f} ms"
    )
    print(f"  alpha from COMMANDED wheels: {alpha:5.2f}  (corr {corr:+.3f}, n={n})")
    if d["st"].size > 10:
        meas = np.stack([np.interp(d["gt"], d["st"], d["meas"][:, i]) for i in range(3)], 1)
        meas = meas[shift:] if shift else meas
        alpha_m, corr_m, n_m = _fit_alpha(meas[:, 1] - meas[:, 0], yaw_rate)
        print(f"  alpha from MEASURED wheels:  {alpha_m:5.2f}  (corr {corr_m:+.3f}, n={n_m})")
        print(f"  drivetrain realization of the commanded differential: {alpha_m / alpha:.2f}")
    if d["ot"].size > 10:
        model_v = WHEEL_RADIUS * fwd_cmd
        keep = np.abs(model_v) > 0.2
        if keep.sum() > 50:
            gain = float(model_v[keep] @ odom_rate[keep] / (model_v[keep] @ model_v[keep]))
            circular = (
                " (WHEEL-derived odometry -- circular, not evidence)" if d["st"].size > 10 else ""
            )
            print(f"  forward gain (measured / commanded): {gain:.3f}{circular}")
    print(
        f"  -> k_turn = (alpha - 1) / plan_friction = {(alpha - 1) / 0.8:.2f} at plan_friction 0.8"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    for name in sys.argv[1:]:
        analyse(Path(name))
