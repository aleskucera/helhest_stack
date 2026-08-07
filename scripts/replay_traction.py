"""Replay a bag through the engine on MEASURED wheel speeds and score the traction models.

Run:  python scripts/replay_traction.py bags/out_experiment_goal_unreachable0 [more bags]

`fit_traction.py` scores a model as an instantaneous map from wheel speeds to yaw rate, which is
only fair where the body is quasi-static -- it had to discard 93% of the samples. A model WITH
momentum cannot be scored that way at all, because its prediction depends on the twist it carried
in. So this drives the real engine step by step at the bag's own rate, feeding it the measured
wheel speeds, and compares the yaw-rate TRAJECTORY it produces against the gyro. Every sample
counts, and the three models are run through the same simulator:

    legacy            alpha = 1 + k_turn mu, commanded speed always achieved
    shear             the quasi-static shear force balance
    shear + momentum  the same balance with implicit body inertia

Flat terrain is assumed, which these bags support (tilt stays under 5.4 deg) and which also makes
the comparison insensitive to mu, since the balance is mu-homogeneous on the flat.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import warp as wp
from rosbags.highlevel import AnyReader

from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams

JOINTS = ("left_wheel_j", "right_wheel_j", "rear_wheel_j")
CELL, CELLS = 0.1, 81
ORIGIN = (-4.0, -4.0)
MU = 0.8
SHEAR_LK = 8.0
STEADY_WINDOW, STEADY_TOL = 0.4, 0.5


def _load(bag: Path):
    """Measured wheel speeds, gyro yaw rate, and a quasi-static mask, on the /joint_states clock."""
    stamps, wheels, gyro_t, gyro = [], [], [], []
    with AnyReader([bag]) as reader:
        topics = {"/joint_states", "/ouster/imu"}
        conns = [c for c in reader.connections if c.topic in topics]
        for conn, stamp, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == "/joint_states":
                if len(msg.velocity) < 3:
                    continue
                idx = [list(msg.name).index(j) for j in JOINTS]
                stamps.append(stamp * 1e-9)
                wheels.append([msg.velocity[i] for i in idx])
            else:
                gyro_t.append(stamp * 1e-9)
                gyro.append(-msg.angular_velocity.y)  # the Ouster IMU carries yaw on negated y
    stamps, wheels = np.asarray(stamps), np.asarray(wheels)
    yaw = np.interp(stamps, np.asarray(gyro_t), np.asarray(gyro))
    dt = float(np.median(np.diff(stamps)))
    window = max(int(STEADY_WINDOW / dt), 3)
    spread = np.zeros(len(stamps))
    from numpy.lib.stride_tricks import sliding_window_view

    for k in range(3):
        view = sliding_window_view(wheels[:, k], window)
        column = np.zeros(len(stamps))
        column[window - 1 :] = view.max(1) - view.min(1)
        spread = np.maximum(spread, column)
    return wheels, yaw, spread, dt


def _replay(wheels: np.ndarray, dt: float, shear_lk: float, momentum: bool) -> np.ndarray:
    """Drive the engine one step per sample and return its predicted yaw rate."""
    device = "cuda" if wp.get_cuda_device_count() > 0 else "cpu"
    steps = len(wheels)
    sim = ForwardSimulator(
        RobotParams(),
        SolverParams(
            dt=dt, shear_lk=shear_lk, shear_iters=8, body_momentum=momentum, newton_iters=6
        ),
        GridParams(CELLS, CELLS, CELL, *ORIGIN),
        1,
        steps,
        device=device,
    )
    sim.set_uniform_friction(MU)
    sim.set_terrain(wp.zeros((CELLS, CELLS), dtype=wp.float32, device=device))
    sim.rollout(np.ascontiguousarray(wheels[:, None, :], np.float32), (0.0, 0.0, 0.0))
    return sim.twist.numpy()[1:, 0, 2]


def analyse(bags: list[Path]) -> None:
    wheels, truth, spread, dt = [], [], [], None
    for bag in bags:
        w, y, s, d = _load(bag)
        if w.size == 0:
            continue
        wheels.append(w)
        truth.append(y)
        spread.append(s)
        dt = d
    if not wheels:
        print("no usable bags")
        return

    configs = (
        ("legacy", 0.0, False),
        ("shear", SHEAR_LK, False),
        ("shear + momentum", SHEAR_LK, True),
    )
    scores: dict[str, list] = {name: [[], []] for name, _, _ in configs}
    for w, y, s in zip(wheels, truth, spread):
        turning = (np.abs(w).max(1) > 0.5) & (np.abs(w[:, 1] - w[:, 0]) > 0.3)
        quasi = turning & (s < STEADY_TOL)
        for name, lk, mom in configs:
            pred = _replay(w, dt, lk, mom)
            scores[name][0].append((pred[turning] - y[turning]))
            scores[name][1].append((pred[quasi] - y[quasi]))

    print(f"predicted vs gyro yaw rate, {sum(len(w) for w in wheels)} samples @ {1 / dt:.0f} Hz")
    print(f"{'model':>18} {'RMS all':>9} {'med all':>9} {'RMS q-s':>9} {'med q-s':>9}")
    for name, _, _ in configs:
        allx = np.concatenate(scores[name][0])
        qs = np.concatenate(scores[name][1])
        print(
            f"{name:>18} {np.sqrt(np.mean(allx**2)):9.4f} {np.median(np.abs(allx)):9.4f} "
            f"{np.sqrt(np.mean(qs**2)):9.4f} {np.median(np.abs(qs)):9.4f}"
        )
    print("\n('all' = every turning sample; 'q-s' = the quasi-static subset fit_traction.py uses)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    wp.init()
    analyse([Path(a) for a in sys.argv[1:]])
