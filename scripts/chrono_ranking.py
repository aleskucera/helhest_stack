"""Score a fixed candidate population in Chrono -- the Level-3 reference IMPROVEMENTS.md wants.

    scripts/chrono_env.sh scripts/chrono_ranking.py --n 150 --out /tmp/chrono_rank.npz

IMPROVEMENTS.md section 10 asks whether the cheap quasi-static model ORDERS candidate trajectories
the way an expensive one does; it was never run because no expensive reference existed. Chrono is
now that reference.

Flat rigid ground on purpose. It removes the two unvalidated things -- soil parameters and the
terrain-dependent cost terms -- so what remains is a clean test of the dynamics: same controls,
same actuator lag, same geometry, and the only difference is quasi-static kinematics against a
full contact solve.

`k_turn` must be matched on the engine side before the ranking means anything (see
scripts/engine_ranking.py). Otherwise the two disagree mostly because our alpha is a fitted
constant of 1.48 while Chrono's emerges at 3.17, and that known gap would swamp everything else.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np
import pychrono as chrono

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import chrono_vehicle_model as M  # noqa: E402

DT_PLAN = 0.1  # the planner's step; the candidate holds each command for one of these
DT_SIM = 5.0e-3  # Chrono's integration step


def candidates(n: int, horizon: int, seed: int, wmin=0.0, wmax=4.0, nominal=1.5) -> np.ndarray:
    """[T, N, 2] wheel-speed schedules, the same shape MPPI samples: a knot spline plus jitter."""
    rng = np.random.default_rng(seed)
    n_wide = int(0.25 * n)
    pos = np.linspace(0.0, 3.0, horizon)
    lo = np.clip(np.floor(pos).astype(int), 0, 3)
    hi = np.clip(lo + 1, 0, 3)
    f = (pos - lo)[:, None, None]
    u = np.empty((horizon, n, 2))
    kw = rng.uniform(wmin, wmax, (4, n_wide, 2))
    u[:, :n_wide] = (1 - f) * kw[lo] + f * kw[hi]  # f is (T,1,1): broadcasts, do not slice
    kn = rng.normal(0.0, 1.0, (4, n - n_wide, 2))
    u[:, n_wide:] = nominal + (1 - f) * kn[lo] + f * kn[hi]
    u[:, n_wide:] += rng.normal(0.0, 0.5, (horizon, n - n_wide, 1))
    u[:, 0] = nominal
    return np.clip(u, wmin, wmax)


def run_one(schedule: np.ndarray) -> np.ndarray:
    """One candidate: [T, 2] commanded (wL, wR) held for DT_PLAN each. Returns [T+1, 3] poses."""
    chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.0005)
    chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.0005)
    sys_ = chrono.ChSystemNSC()
    sys_.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
    sys_.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    mat = chrono.ChContactMaterialNSC()
    mat.SetFriction(M.MU)
    mat.SetRestitution(0.0)
    mat.SetRollingFriction(0.0)  # the NSC rolling constraint locks the yaw; applied as torque below

    chassis, wheels, motors = M.build_robot(sys_, mat, "fixed")
    M.rigid_terrain(sys_, mat)
    solver = chrono.ChSolverBB()
    solver.SetMaxIterations(2000)
    solver.SetTolerance(1e-10)
    sys_.SetSolver(solver)

    realized = {k: 0.0 for k in ("left", "right", "rear")}
    steps_per_plan = int(round(DT_PLAN / DT_SIM))
    out = [[0.0, 0.0, 0.0]]
    for t in range(schedule.shape[0]):
        cmd = {
            "left": float(schedule[t, 0]),
            "right": float(schedule[t, 1]),
            "rear": 0.5 * float(schedule[t, 0] + schedule[t, 1]),
        }
        for _ in range(steps_per_plan):
            blend = DT_SIM / (DT_SIM + M.MOTOR_TAU)
            for k in cmd:
                realized[k] += blend * (cmd[k] - realized[k])
                motors[k].SetConstant(realized[k])
            for w in wheels.values():
                w.EmptyAccumulators()
                nf = abs(float(w.GetContactForce().z))
                spin = float(w.GetAngVelLocal().y)
                if nf > 1.0 and abs(spin) > 1e-6:
                    tq = -math.copysign(M.ROLLING_RESISTANCE * nf * M.WHEEL_RADIUS, spin)
                    w.AccumulateTorque(chrono.ChVector3d(0.0, tq, 0.0), True)
            sys_.DoStepDynamics(DT_SIM)
        f = chassis.GetFrameRefToAbs()
        ex = f.TransformDirectionLocalToParent(chrono.ChVector3d(1, 0, 0))
        out.append([f.GetPos().x, f.GetPos().y, math.atan2(ex.y, ex.x)])
    return np.array(out, np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    U = candidates(args.n, args.horizon, args.seed)
    trajs = np.zeros((args.n, args.horizon + 1, 3), np.float32)
    for i in range(args.n):
        trajs[i] = run_one(U[:, i, :])
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{args.n} candidates")
    np.savez_compressed(args.out, U=U.astype(np.float32), trajs=trajs,
                        horizon=np.int32(args.horizon), seed=np.int32(args.seed))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
