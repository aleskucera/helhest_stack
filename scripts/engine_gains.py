"""The engine's forward gain and turn gain, measured exactly as `chrono_vehicle_model.py` does.

    python scripts/engine_gains.py --out /tmp/engine_gains.json

Same protocol, same commanded wheel speeds, flat ground: drive at constant command, take the
steady second half, and report

    forward gain  = distance travelled / (R * mean(wL, wR) * elapsed)
    alpha         = ideal differential-drive yaw rate / realised yaw rate

so the three-way table -- bags, engine, Chrono -- is like for like. The bag targets are 0.906-0.925
and 2.20 (TIER1_REPORT.md section 3).

Both traction models are run. The legacy one is analytic here and worth stating up front: forward
gain is 1.000 BY CONSTRUCTION (the commanded speed is always achieved) and alpha is
1 + k_turn * mu, so at the planner's k_turn = 0.6 and mu = 0.8 it is 1.48 regardless of manoeuvre.
Neither is a fit to anything; that is the point of measuring them.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import warp as wp

from helhest import dynamics
from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams
from helhest.heightmap import _grid
from helhest.heightmap import Heightmap

CELL, EXTENT, MU = 0.1, 30.0, 0.8
DT, T = 0.01, 600  # 6 s at 100 Hz, matching the Chrono run


def flat() -> Heightmap:
    XX, _ = _grid((-EXTENT, EXTENT), (-EXTENT, EXTENT), CELL)
    return Heightmap(np.zeros_like(XX).astype(np.float32), (-EXTENT, -EXTENT), CELL)


def drive(omega: tuple[float, float, float], shear: bool, device: str) -> np.ndarray:
    scene = flat()
    kw = dict(shear_lk=12.0, body_momentum=True, rolling_resistance=0.09) if shear else {}
    sp = SolverParams(
        dt=DT, k_turn=dynamics.K_TURN, newton_iters=6, atol=1e-4,
        tau_motor=dynamics.MOTOR_TAU, **kw,
    )
    sim = ForwardSimulator(
        RobotParams(), sp,
        GridParams(scene.nx, scene.ny, scene.cell, scene.x0, scene.y0), 1, T, device,
    )
    sim.set_terrain(
        wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device)
    )
    sim.set_uniform_friction(MU)
    U = np.tile(np.asarray(omega, np.float32), (T, 1, 1))
    controlled, *_ = sim.rollout(np.ascontiguousarray(U, np.float32), (0.0, 0.0, 0.0))
    del sim
    return controlled[:, 0, :].astype(np.float64)


def gains(traj: np.ndarray, omega: tuple[float, float, float]) -> tuple[float, float]:
    half = traj[len(traj) // 2:]
    elapsed = (len(half) - 1) * DT
    d = float(np.hypot(half[-1, 0] - half[0, 0], half[-1, 1] - half[0, 1]))
    v_cmd = RobotParams().wheel_radius * 0.5 * (omega[0] + omega[1])
    fwd = d / (v_cmd * elapsed) if v_cmd > 0 else float("nan")
    wz = float(np.unwrap(half[:, 2])[-1] - np.unwrap(half[:, 2])[0]) / elapsed
    ideal = RobotParams().wheel_radius * (omega[1] - omega[0]) / (2.0 * RobotParams().half_track)
    alpha = float("nan") if abs(wz) < 1e-6 else ideal / wz
    return fwd, alpha


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    wp.init()
    out = {}
    for label, shear in (("legacy", False), ("shear+momentum", True)):
        rec = {"forward": [], "turn": []}
        print(f"=== engine, {label} ===")
        for w in (1.0, 2.0, 3.0, 4.0):
            f, _a = gains(drive((w, w, w), shear, args.device), (w, w, w))
            rec["forward"].append({"omega": w, "gain": f})
            print(f"  forward  omega {w:.1f}  ->  gain {f:.4f}")
        for wl, wr in ((1.0, 3.0), (1.5, 3.5), (0.5, 3.5), (2.0, 4.0)):
            om = (wl, wr, 0.5 * (wl + wr))
            _f, a = gains(drive(om, shear, args.device), om)
            rec["turn"].append({"wl": wl, "wr": wr, "alpha": a})
            print(f"  turn     wL {wl:.1f} wR {wr:.1f}  ->  alpha {a:.4f}")
        out[label] = rec
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
