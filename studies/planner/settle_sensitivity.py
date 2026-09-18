"""Plan section 6.2: does the settle's clamping mask the sensitivity sigma has to travel through?

The z-margin design rests on `J^-1`, the settle's own analytic Jacobian, carrying map
uncertainty into attitude uncertainty (PROBABILISTIC_PLANNING_PLAN.md section 3.1). But
`settle()` runs a DAMPED Newton -- `solver.max_step` caps each iteration and `solver.tilt_clamp`
bounds the angles every pass -- so the question is whether the attitude response to a terrain
perturbation is the true sensitivity or a clipped one, especially at a contested contact where
two cells nearly tie for a wheel's support.

Measured by finite difference against the closed form. With wheels at (0, +b), (0, -b) and
(-l, 0) and small angles, the contact heights are

    e1 = z + roll*b,  e2 = z - roll*b,  e3 = z + pitch*l

so  roll = (e1 - e2) / 2b  and  pitch = (e3 - (e1 + e2)/2) / l, giving
d(roll)/d(e1) = 1/(2b) and d(pitch)/d(e3) = 1/l. Those are the numbers a perturbation under a
wheel must reproduce if `J^-1` is sound.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import warp as wp

from helhest.engine import ForwardSimulator
from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams
from helhest.heightmap import Heightmap

CELL = 0.08
N = 161  # grid side; ~12.8 m, ample for one robot


def _sim(grid: GridParams, robot: RobotParams, solver: SolverParams):
    sim = ForwardSimulator(
        robot_params=robot, solver_params=solver, grid_params=grid, batch_size=1, n_steps=1
    )
    sim.target_wheel_omega.zero_()
    sim.set_friction(
        Heightmap(
            np.full((grid.cells_y, grid.cells_x), 0.8, np.float32),
            (grid.origin_x, grid.origin_y),
            CELL,
        )
    )
    return sim


def settle_at(
    sim, elev: np.ndarray, x: float, y: float, yaw: float
) -> tuple[float, float, float, float]:
    """(z, pitch, roll, residual) for one pose on one terrain."""
    sim.set_terrain(wp.array(elev, dtype=wp.float32))
    sim.start_pose.assign(np.array([[x, y, yaw]], np.float32))
    sim.rollout_launch()
    d = sim.derived.numpy()[0, 0]
    return float(d[0]), float(d[1]), float(d[2]), float(sim.residual.numpy()[0, 0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="studies/out/planner")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    robot = RobotParams()
    b, l = robot.half_track, robot.rear_offset
    grid = GridParams(
        cells_x=N, cells_y=N, cell_size=CELL, origin_x=-N * CELL / 2, origin_y=-N * CELL / 2
    )
    sim = _sim(grid, robot, SolverParams())

    flat = np.zeros((N, N), np.float32)
    x = y = yaw = 0.0
    z0, p0, r0, res0 = settle_at(sim, flat, x, y, yaw)
    print(
        f"flat ground: z {z0:+.4f} pitch {np.degrees(p0):+.3f} roll {np.degrees(r0):+.3f} "
        f"residual {res0:.2e}"
    )

    report = {
        "cell": CELL,
        "half_track": b,
        "rear_offset": l,
        "closed_form": {"droll_de1_rad_per_m": 1.0 / (2 * b), "dpitch_de3_rad_per_m": 1.0 / l},
    }

    def cell_of(wx, wy):
        return int((wy - grid.origin_y) / CELL), int((wx - grid.origin_x) / CELL)

    # --- 1. sensitivity under each wheel, against the closed form -------------------------
    wheels = {
        "front-L": (0.0, +b, "roll"),
        "front-R": (0.0, -b, "roll"),
        "rear": (-l, 0.0, "pitch"),
    }
    print(
        f"\n{'wheel':9s} {'delta_m':>8s} {'d(roll)/dh':>11s} {'d(pitch)/dh':>12s} {'residual':>10s}"
    )
    sens = {}
    for name, (wx, wy, axis) in wheels.items():
        rows = []
        for delta in (0.002, 0.005, 0.01, 0.02, 0.05):
            e = flat.copy()
            r_, c_ = cell_of(wx, wy)
            e[r_ - 2 : r_ + 3, c_ - 2 : c_ + 3] = delta  # a patch, so the dilation sees it whole
            _, p1, r1, res1 = settle_at(sim, e, x, y, yaw)
            dr, dp = (r1 - r0) / delta, (p1 - p0) / delta
            rows.append({"delta": delta, "droll": dr, "dpitch": dp, "residual": res1})
            print(f"{name:9s} {delta:8.3f} {dr:11.3f} {dp:12.3f} {res1:10.2e}")
        sens[name] = rows
    report["per_wheel"] = sens
    print(f"  closed form: d(roll)/de1 = {1/(2*b):.3f}   d(pitch)/de3 = {1/l:.3f} rad/m")

    # --- 2. a contested contact: two cells competing for one wheel's support ---------------
    print("\ncontested contact -- two cells under the front-left wheel, lead swept through zero:")
    print(
        f"{'lead_m':>9s} {'roll_deg':>9s} {'d(roll)/dh_A':>13s} {'d(roll)/dh_B':>13s} {'residual':>10s}"
    )
    contest = []
    # The cylinder envelope is `wheel_width` wide (0.10 m) and 2*wheel_radius long (0.70 m), so
    # at yaw = 0 the front-left wheel's footprint is x in [-0.35, 0.35], y in [0.315, 0.415].
    # Contesting cells must sit INSIDE that: a first attempt placed them +-0.10 m in y, outside
    # the 0.10 m width, and the perturbation did nothing at all.
    wx, wy = 0.0, +b
    ra, ca = cell_of(wx - 0.20, wy)
    rb, cb = cell_of(wx + 0.20, wy)
    for lead in (-0.04, -0.02, -0.005, 0.0, 0.005, 0.02, 0.04):
        e = flat.copy()
        e[ra, ca] = 0.05 + lead  # cell A
        e[rb, cb] = 0.05  # cell B
        _, _, r_base, res_b = settle_at(sim, e, x, y, yaw)
        eps = 1e-3
        ea = e.copy()
        ea[ra, ca] += eps
        eb = e.copy()
        eb[rb, cb] += eps
        _, _, r_a, _ = settle_at(sim, ea, x, y, yaw)
        _, _, r_b, _ = settle_at(sim, eb, x, y, yaw)
        da, db = (r_a - r_base) / eps, (r_b - r_base) / eps
        contest.append({"lead": lead, "roll": r_base, "dA": da, "dB": db, "residual": res_b})
        print(f"{lead:9.3f} {np.degrees(r_base):9.3f} {da:13.3f} {db:13.3f} {res_b:10.2e}")
    report["contested"] = contest

    # --- 3. does anything actually clamp? --------------------------------------------------
    sp = SolverParams()
    print(
        f"\nsolver caps: tilt_clamp {np.degrees(sp.tilt_clamp):.0f} deg, "
        f"max_step (z {sp.max_step[0]} m, pitch {np.degrees(sp.max_step[1]):.0f} deg, "
        f"roll {np.degrees(sp.max_step[2]):.0f} deg), newton_iters {sp.newton_iters}"
    )
    worst = 0.0
    for slope_deg in (5, 10, 15, 20, 25, 30):
        e = np.tile(
            np.linspace(-1, 1, N, dtype=np.float32)
            * (N * CELL / 2)
            * np.tan(np.radians(slope_deg)),
            (N, 1),
        )
        _, p1, r1, res1 = settle_at(sim, np.ascontiguousarray(e.T), x, y, yaw)
        worst = max(worst, abs(res1))
        print(
            f"   {slope_deg:2d} deg cross-slope -> roll {np.degrees(r1):+7.3f} deg  residual {res1:.2e}"
        )
    report["max_residual_on_slopes"] = worst

    dst = os.path.join(args.out, "settle_sensitivity.json")
    with open(dst, "w") as fh:
        json.dump(report, fh, indent=1)
    print("\n->", dst)


if __name__ == "__main__":
    main()
