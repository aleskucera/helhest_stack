"""Fit the shear traction model's L/K from bags -- no calibration drive required.

Run:  python scripts/fit_traction.py bags/out_experiment_goal_unreachable0 [more bags]

The shear model predicts the body twist directly from WHEEL SPEEDS, so the calibration compares
two contemporaneous sensor readings -- measured wheel speed (/joint_states) against measured yaw
rate (a gyro) -- with no command in the loop. That matters: it means no command-transport delay
enters, and no steady-state manoeuvre is needed to fit the parameter. On flat ground the model is
also mu-independent (every contact force scales with mu, so the balance is homogeneous), which
removes the other unknown.

QUASI-STATIC FILTERING IS THE POINT. Pooled over all turning samples both models sit at RMS ~0.16
rad/s and are indistinguishable: the residual is dominated by yaw-inertia transients that no
quasi-static model can capture. Restricting to samples whose wheel speeds have been steady for
`STEADY_WINDOW` seconds drops the RMS about fourfold and only then does the traction model matter.
Measured there: legacy 0.0456, shear 0.0393-0.0396 at L/K 8-15 -- a real but modest 14% (21% on
median), with L/K landing where the soil literature puts it.

Needs /joint_states, so it runs only on post-2026-07-27 Ouster bags.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from rosbags.highlevel import AnyReader

JOINTS = ("left_wheel_j", "right_wheel_j", "rear_wheel_j")
WHEEL_RADIUS, HALF_TRACK, REAR_OFFSET = 0.35, 0.365, 0.75
MASS, GRAVITY = 106.2, 9.81
WHEELS = np.array([[0.0, HALF_TRACK], [0.0, -HALF_TRACK], [-REAR_OFFSET, 0.0]])
LOADS = MASS * GRAVITY * np.array([0.368, 0.368, 0.264])  # flat-ground barycentric weights
MU = 0.8  # cancels on flat ground; only the load RATIOS matter
PATCH = 0.075  # contact patch radius [m], from the geometry
STEADY_WINDOW = 0.4  # [s] wheel speeds must be flat this long for a sample to count
STEADY_TOL = 0.5  # [rad/s] peak-to-peak allowed inside that window
K_TURN = 1.0  # the legacy baseline this is compared against


def _load(bag: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(wheel speeds, gyro yaw rate, wheel-speed peak-to-peak) on the /joint_states clock."""
    stamps, wheels, gyro_t, gyro = [], [], [], []
    with AnyReader([bag]) as reader:
        topics = {"/joint_states", "/ouster/imu", "/imu/data", "/odin1/imu"}
        conns = [c for c in reader.connections if c.topic in topics]
        imu_topic = next(
            (t for t in ("/ouster/imu", "/imu/data", "/odin1/imu") if any(c.topic == t for c in conns)),
            "/imu/data",
        )
        for conn, stamp, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == "/joint_states":
                if len(msg.velocity) < 3:
                    continue
                idx = [list(msg.name).index(j) for j in JOINTS]
                stamps.append(stamp * 1e-9)
                wheels.append([msg.velocity[i] for i in idx])
            elif conn.topic == imu_topic:
                gyro_t.append(stamp * 1e-9)
                # the Ouster IMU carries yaw on negated y; /imu/data carries it on z
                g = msg.angular_velocity
                gyro.append(-g.y if imu_topic == "/ouster/imu" else g.z)
    stamps, wheels = np.asarray(stamps), np.asarray(wheels)
    if stamps.size == 0:
        return stamps, np.empty(0), np.empty(0)
    yaw_rate = np.interp(stamps, np.asarray(gyro_t), np.asarray(gyro))
    dt = float(np.median(np.diff(stamps)))
    window = max(int(STEADY_WINDOW / dt), 3)
    spread = np.zeros(len(stamps))
    for k in range(3):
        view = sliding_window_view(wheels[:, k], window)
        column = np.zeros(len(stamps))
        column[window - 1 :] = view.max(1) - view.min(1)
        spread = np.maximum(spread, column)
    return wheels, yaw_rate, spread


def _residual(twist: np.ndarray, omega: np.ndarray, lk: float, patch: float) -> np.ndarray:
    out = np.zeros_like(twist)
    for k, (x, y) in enumerate(WHEELS):
        slip_x = twist[:, 0] - twist[:, 2] * y - WHEEL_RADIUS * omega[:, k]
        slip_y = twist[:, 1] + twist[:, 2] * x
        slip_spin = patch * twist[:, 2]
        norm = np.sqrt(slip_x**2 + slip_y**2 + slip_spin**2) + 1e-12
        rolling = np.abs(WHEEL_RADIUS * omega[:, k])
        lam = np.where(rolling > 1e-6, np.maximum(norm / np.maximum(rolling, 1e-9) * lk, 1e-9), 1e6)
        mobilised = 1.0 - (1.0 - np.exp(-np.minimum(lam, 50.0))) / lam
        scale = MU * LOADS[k] * mobilised / norm
        fx, fy = -scale * slip_x, -scale * slip_y
        out[:, 0] += fx
        out[:, 1] += fy
        out[:, 2] += x * fy - y * fx - scale * patch * slip_spin
    out[:, 0] -= MASS * (-twist[:, 1] * twist[:, 2])
    out[:, 1] -= MASS * (twist[:, 0] * twist[:, 2])
    return out


def _solve(omega: np.ndarray, lk: float, patch: float, iters: int = 40) -> np.ndarray:
    """Batched damped Newton over every sample at once."""
    twist = np.stack(
        [
            WHEEL_RADIUS * (omega[:, 0] + omega[:, 1]) / 2.0,
            np.zeros(len(omega)),
            WHEEL_RADIUS * (omega[:, 1] - omega[:, 0]) / (2.0 * HALF_TRACK * 2.0),
        ],
        axis=1,
    )
    for _ in range(iters):
        r = _residual(twist, omega, lk, patch)
        jac = np.zeros((len(omega), 3, 3))
        for k in range(3):
            probe = twist.copy()
            probe[:, k] += 1e-6
            jac[:, :, k] = (_residual(probe, omega, lk, patch) - r) / 1e-6
        try:
            step = np.clip(np.linalg.solve(jac, r[:, :, None])[:, :, 0], -1.0, 1.0)
        except np.linalg.LinAlgError:
            break
        best, best_norm, scale = twist.copy(), np.linalg.norm(r, axis=1), 1.0
        for _trial in range(4):
            candidate = twist - scale * step
            norm = np.linalg.norm(_residual(candidate, omega, lk, patch), axis=1)
            better = norm < best_norm
            best[better] = candidate[better]
            best_norm[better] = norm[better]
            scale *= 0.5
        twist = best
    return twist


def analyse(bags: list[Path]) -> None:
    omega, measured, spread = [], [], []
    for bag in bags:
        w, y, s = _load(bag)
        if w.size == 0:
            print(f"{bag.name}: no /joint_states -- skipped")
            continue
        turning = (np.abs(w).max(1) > 0.5) & (np.abs(w[:, 1] - w[:, 0]) > 0.3)
        omega.append(w[turning])
        measured.append(y[turning])
        spread.append(s[turning])
    if not omega:
        return
    omega = np.concatenate(omega)
    measured = np.concatenate(measured)
    spread = np.concatenate(spread)
    kinematic = WHEEL_RADIUS * (omega[:, 1] - omega[:, 0]) / (2.0 * HALF_TRACK)

    for label, mask in (
        ("all turning samples", np.ones(len(omega), bool)),
        (f"quasi-static (ptp < {STEADY_TOL} over {STEADY_WINDOW} s)", spread < STEADY_TOL),
    ):
        idx = np.where(mask)[0][::2]
        if len(idx) < 80:
            print(f"=== {label}: only {len(idx)} samples, skipped\n")
            continue
        om, truth, kin = omega[idx], measured[idx], kinematic[idx]
        print(f"=== {label}   n={len(idx)}")
        legacy = kin / (1.0 + K_TURN * MU)
        print(
            f"   {f'legacy k_turn={K_TURN}':>22} RMS {np.sqrt(np.mean((legacy - truth) ** 2)):.4f}"
            f"  median {np.median(np.abs(legacy - truth)):.4f}"
        )
        for lk in (2.0, 5.0, 8.0, 15.0, 30.0):
            pred = _solve(om, lk, PATCH)[:, 2]
            print(
                f"   {f'shear L/K={lk:.0f}':>22} RMS {np.sqrt(np.mean((pred - truth) ** 2)):.4f}"
                f"  median {np.median(np.abs(pred - truth)):.4f}"
            )
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    analyse([Path(a) for a in sys.argv[1:]])
