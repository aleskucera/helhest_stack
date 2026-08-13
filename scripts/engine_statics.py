"""The engine's answer to the same statics question `chrono_statics.py` asks.

    python scripts/engine_statics.py --out /tmp/engine_statics.json

Settles the robot on a tilted plane and reports `(z, pitch, roll)` and the three normal loads,
over the same tilt sweep. Runs in the project venv; the two scripts share nothing but the JSON,
because Chrono lives in its own conda environment. `scripts/chrono_compare.py` joins them.

The plane is built as an exact heightfield and dilated by the engine's own wheel envelope, so the
comparison includes the envelope, not just the settle: on a plane of slope s the sphere envelope
lifts by R(sqrt(1+s^2) - 1) analytically, which is itself a check on the dilation.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import warp as wp

from helhest import heightmap as hmmod
from helhest.engine import Grid
from helhest.engine import GridParams
from helhest.engine import Robot
from helhest.engine import RobotParams
from helhest.engine import Solver
from helhest.engine import SolverParams
from helhest.engine.rotations import euler_zyx
from helhest.engine.step import normal_loads
from helhest.engine.step import settle
from helhest.heightmap import _grid
from helhest.heightmap import Heightmap

CELL, EXTENT = 0.02, 6.0  # fine cell: the comparison should not be limited by rasterisation


@wp.kernel
def statics_probe(
    envelope: wp.array2d(dtype=wp.float32),
    grid: Grid,
    robot: Robot,
    solver: Solver,
    pose: wp.array(dtype=wp.vec3),  # (x, y, yaw)
    derived_out: wp.array(dtype=wp.vec3),  # (z, pitch, roll)
    loads_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    d = settle(envelope, grid, robot, solver, pose[tid], wp.vec3(0.4, 0.0, 0.0))
    derived_out[tid] = d
    R = euler_zyx(pose[tid][2], d[1], d[2])  # the engine's own convention, not a quaternion
    loads_out[tid] = normal_loads(
        envelope, grid, robot, R, wp.vec3(pose[tid][0], pose[tid][1], d[0])
    )


def tilted_plane(pitch_deg: float, roll_deg: float) -> Heightmap:
    """A plane whose surface normal matches a ground tilted by (pitch about +y, roll about +x).

    Height field of that plane: with the ground's own normal n, h(x, y) = -(n_x x + n_y y)/n_z.
    """
    p, r = math.radians(pitch_deg), math.radians(roll_deg)
    # ground normal after Ry(pitch) then Rx(roll) applied to +z, matching chrono_statics.build()
    n = np.array(
        [
            math.sin(p) * math.cos(r),
            -math.sin(r),
            math.cos(p) * math.cos(r),
        ]
    )
    XX, YY = _grid((-EXTENT, EXTENT), (-EXTENT, EXTENT), CELL)
    H = -(n[0] * XX + n[1] * YY) / n[2]
    return Heightmap(H.astype(np.float32), (-EXTENT, -EXTENT), CELL)


def measure(pitch_deg: float, roll_deg: float, rp: RobotParams, sp: SolverParams) -> dict:
    scene = tilted_plane(pitch_deg, roll_deg)
    env_hm = hmmod.wheel_envelope(scene, rp.wheel_radius)
    env = wp.array(np.ascontiguousarray(env_hm.H, np.float32), dtype=wp.float32, device="cpu")
    grid = GridParams(env_hm.nx, env_hm.ny, env_hm.cell, env_hm.x0, env_hm.y0).build()
    pose = wp.array(np.array([[0.0, 0.0, 0.0]], np.float32), dtype=wp.vec3, device="cpu")
    d_out = wp.zeros(1, dtype=wp.vec3, device="cpu")
    l_out = wp.zeros(1, dtype=wp.vec3, device="cpu")
    wp.launch(
        statics_probe,
        1,
        inputs=[env, grid, rp.build("cpu"), sp.build(), pose],
        outputs=[d_out, l_out],
        device="cpu",
    )
    d = np.asarray(d_out.numpy()[0], np.float64)
    loads = np.asarray(l_out.numpy()[0], np.float64) / (rp.mass * rp.gravity)
    return {
        "z": float(d[0]),
        "pitch_deg": float(math.degrees(d[1])),
        "roll_deg": float(math.degrees(d[2])),
        "loads": {"left": float(loads[0]), "right": float(loads[1]), "rear": float(loads[2])},
        "sum_loads": float(loads.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    wp.init()
    rp = RobotParams()
    sp = SolverParams(newton_iters=40, atol=1e-10)  # statics: converge hard, speed is irrelevant
    rows = []
    print(f"{'tilt':>18}{'pitch':>8}{'roll':>8}{'z':>8}"
          f"{'N_left':>9}{'N_right':>9}{'N_rear':>9}{'sum':>8}")
    for axis in ("pitch", "roll"):
        for deg in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0):
            r = measure(deg if axis == "pitch" else 0.0, deg if axis == "roll" else 0.0, rp, sp)
            r["axis"], r["tilt_deg"] = axis, deg
            rows.append(r)
            print(f"{f'{axis} {deg:.0f} deg':>18}{r['pitch_deg']:>8.2f}{r['roll_deg']:>8.2f}"
                  f"{r['z']:>8.4f}{r['loads']['left']:>9.4f}{r['loads']['right']:>9.4f}"
                  f"{r['loads']['rear']:>9.4f}{r['sum_loads']:>8.4f}")
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
