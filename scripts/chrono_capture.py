"""Run scenarios and dump the state needed to render them: poses, wheel tracks, deformed soil.

    scripts/chrono_env.sh scripts/chrono_capture.py --out /tmp/chrono_capture.npz

Runs in Chrono's environment, which has no matplotlib, so it only WRITES arrays;
`scripts/chrono_render.py` draws them in the project venv. For SCM runs the terrain height field
is sampled on a grid over the driven region at intervals, so the ruts can be shown forming.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import pychrono as chrono

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import chrono_vehicle_model as M  # noqa: E402

GRID_X = (-1.0, 6.0)
GRID_Y = (-3.5, 3.5)
GRID_STEP = 0.05


def sample_soil(terrain) -> np.ndarray:
    xs = np.arange(GRID_X[0], GRID_X[1], GRID_STEP)
    ys = np.arange(GRID_Y[0], GRID_Y[1], GRID_STEP)
    out = np.zeros((len(ys), len(xs)), np.float32)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            out[j, i] = terrain.GetHeight(chrono.ChVector3d(float(x), float(y), 1.0))
    return out


def run(terrain_kind: str, omega, t_end: float, rear: str, snapshots: int, dt: float = 2.0e-3):
    """A copy of chrono_vehicle_model.drive that also records wheel poses and soil snapshots."""
    chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.0005)
    chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.0005)
    sys_ = chrono.ChSystemNSC()
    sys_.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
    sys_.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    mat = chrono.ChContactMaterialNSC()
    mat.SetFriction(M.MU)
    mat.SetRestitution(0.0)
    mat.SetRollingFriction(0.0)  # see the model: this constraint locks the yaw

    chassis, wheels, motors = M.build_robot(sys_, mat, rear)
    if terrain_kind == "rigid":
        terrain = M.rigid_terrain(sys_, mat)
    else:
        terrain = M.scm_terrain(sys_)
        for w in wheels.values():
            terrain.AddMovingPatch(
                w, chrono.ChVector3d(0, 0, 0),
                chrono.ChVector3d(0.5, 4.0 * M.HALF_WIDTH, 2.0 * M.WHEEL_RADIUS),
            )
    solver = chrono.ChSolverBB()
    solver.SetMaxIterations(2000)
    solver.SetTolerance(1e-10)
    sys_.SetSolver(solver)

    cmd = dict(zip(("left", "right", "rear"), omega))
    realized = {k: 0.0 for k in cmd}
    n = int(t_end / dt)
    snap_every = max(n // snapshots, 1)
    poses, tracks, soils, times = [], [], [], []
    for i in range(n):
        blend = dt / (dt + M.MOTOR_TAU)
        for k in cmd:
            realized[k] += blend * (cmd[k] - realized[k])
            motors[k].SetConstant(realized[k])
        for k, w in wheels.items():
            w.EmptyAccumulators()
            nf = abs(float(w.GetContactForce().z))
            spin = float(w.GetAngVelLocal().y)
            if nf > 1.0 and abs(spin) > 1e-6:
                t_roll = -math.copysign(M.ROLLING_RESISTANCE * nf * M.WHEEL_RADIUS, spin)
                w.AccumulateTorque(chrono.ChVector3d(0.0, t_roll, 0.0), True)
        if terrain_kind == "scm":
            terrain.Advance(dt)
        sys_.DoStepDynamics(dt)

        if i % 10 == 0:
            f = chassis.GetFrameRefToAbs()
            ex = f.TransformDirectionLocalToParent(chrono.ChVector3d(1, 0, 0))
            poses.append([f.GetPos().x, f.GetPos().y, f.GetPos().z, math.atan2(ex.y, ex.x)])
            tracks.append([[w.GetPos().x, w.GetPos().y, w.GetPos().z] for w in wheels.values()])
        if i % snap_every == 0 and terrain_kind == "scm":
            soils.append(sample_soil(terrain))
            times.append(i * dt)
    if terrain_kind == "scm":
        soils.append(sample_soil(terrain))
        times.append(n * dt)
    return (
        np.array(poses, np.float32),
        np.array(tracks, np.float32),
        np.array(soils, np.float32) if soils else np.zeros((0, 1, 1), np.float32),
        np.array(times, np.float32),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--t-end", type=float, default=6.0)
    ap.add_argument("--snapshots", type=int, default=40)
    args = ap.parse_args()

    blobs = {
        "grid_x": np.arange(GRID_X[0], GRID_X[1], GRID_STEP).astype(np.float32),
        "grid_y": np.arange(GRID_Y[0], GRID_Y[1], GRID_STEP).astype(np.float32),
        "wheel_pos": np.array(list(M.WHEELS.values()), np.float32),
        "wheel_radius": np.float32(M.WHEEL_RADIUS),
        "half_width": np.float32(M.HALF_WIDTH),
    }
    cases = [
        ("scm_fixed_turn", "scm", (1.0, 3.0, 2.0), "fixed"),
        ("scm_caster_turn", "scm", (1.0, 3.0, 2.0), "caster"),
        ("scm_fixed_straight", "scm", (2.0, 2.0, 2.0), "fixed"),
        ("rigid_fixed_turn", "rigid", (1.0, 3.0, 2.0), "fixed"),
        ("rigid_caster_turn", "rigid", (1.0, 3.0, 2.0), "caster"),
    ]
    for name, terr, om, rear in cases:
        poses, tracks, soils, times = run(terr, om, args.t_end, rear, args.snapshots)
        blobs[f"{name}/poses"] = poses
        blobs[f"{name}/tracks"] = tracks
        blobs[f"{name}/soils"] = soils
        blobs[f"{name}/times"] = times
        print(f"  {name:>20}: {len(poses)} poses, {len(soils)} soil snapshots, "
              f"end ({poses[-1][0]:+.2f}, {poses[-1][1]:+.2f}) yaw {math.degrees(poses[-1][3]):+.0f} deg")
    np.savez_compressed(args.out, **blobs)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
