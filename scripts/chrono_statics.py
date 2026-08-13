"""Settle the robot on a tilted rigid plane in Project Chrono, and report pose + normal loads.

    ~/.local/opt/chrono-env/bin/python scripts/chrono_statics.py --out /tmp/chrono_statics.json

Runs in Chrono's own conda environment (`pychrono` has no pip distribution and this build has no
`pychrono.vehicle`, so no SCM -- rigid contact only). It therefore imports NO helhest code and
writes a plain JSON; `scripts/chrono_compare.py` reads that back in the project venv and puts it
next to the engine's answer. See PREREG_chrono.md for what this is supposed to show.

The robot is one rigid body with three cylinder collision shapes, which is what the engine models:
no suspension, no articulation, wheels rigidly fixed to the chassis. `ChSystemNSC` is the
complementarity solver, so the friction cone is real rather than a penalty approximation -- the
whole point being that Chrono carries the tangential contact reaction the engine's normals-only
balance leaves out.
"""

from __future__ import annotations

import argparse
import json
import math

import pychrono as chrono

MASS = 106.2  # [kg]
COM = (-0.198, 0.0, 0.0)  # [m] body frame, behind the front axle
WHEEL_RADIUS = 0.35
HALF_WIDTH = 0.05  # half the 0.10 m tread
HALF_TRACK = 0.365
REAR_OFFSET = 0.75
MU = 0.8
WHEELS = {
    "left": (0.0, HALF_TRACK, 0.0),
    "right": (0.0, -HALF_TRACK, 0.0),
    "rear": (-REAR_OFFSET, 0.0, 0.0),
}


class ContactByWheel(chrono.ReportContactCallback):
    """Sum each contact's normal force onto the nearest wheel.

    Chrono reports per contact POINT, and a cylinder on a plane generally produces more than one,
    so the per-wheel load is a sum. `plane_normal` is the direction to project onto: the reported
    force is in the contact's own frame, whose x axis is the contact normal.
    """

    def __init__(self, wheel_world):
        super().__init__()
        self.wheel_world = wheel_world
        self.loads = {k: 0.0 for k in wheel_world}
        self.points = {k: 0 for k in wheel_world}

    def OnReportContact(self, pA, pB, plane_coord, distance, eff_radius, react_forces, torques,
                        modA, modB):
        # react_forces[0] is the normal component in the contact frame (x = normal)
        fn = float(react_forces.x)
        if fn == 0.0:
            return True
        mid = ((pA.x + pB.x) * 0.5, (pA.y + pB.y) * 0.5, (pA.z + pB.z) * 0.5)
        best, bestd = None, 1e30
        for name, w in self.wheel_world.items():
            d = (mid[0] - w[0]) ** 2 + (mid[1] - w[1]) ** 2 + (mid[2] - w[2]) ** 2
            if d < bestd:
                best, bestd = name, d
        self.loads[best] += fn
        self.points[best] += 1
        return True


def build(pitch_deg: float, roll_deg: float, shape: str):
    """A system with a plane tilted by (pitch, roll) and the robot resting on it.

    The PLANE is tilted and the robot starts level above it, rather than the robot being tilted:
    that way the settled orientation is an output of the solve, comparable with the engine's.
    """
    sys = chrono.ChSystemNSC()
    sys.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
    sys.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)

    mat = chrono.ChContactMaterialNSC()
    mat.SetFriction(MU)
    mat.SetRestitution(0.0)

    # tilt as an intrinsic rotation: pitch about +y, then roll about +x
    q = chrono.QuatFromAngleY(math.radians(pitch_deg)) * chrono.QuatFromAngleX(
        math.radians(roll_deg)
    )
    # The box rotates about its OWN centre, so placing it at (0, 0, -0.5) and tilting leaves its
    # top face off the world origin by 0.5 * n -- 4.7 cm at 25 deg, which reads as a pose
    # disagreement that does not exist. Push the centre half a thickness down its own normal so
    # the contact plane passes through the origin, as the engine's heightfield does.
    ground = chrono.ChBodyEasyBox(40, 40, 1.0, 1000, True, True, mat)
    n_g = q.Rotate(chrono.ChVector3d(0, 0, 1))
    ground.SetPos(chrono.ChVector3d(-0.5 * n_g.x, -0.5 * n_g.y, -0.5 * n_g.z))
    ground.SetRot(q)
    ground.SetFixed(True)
    sys.Add(ground)

    body = chrono.ChBodyAuxRef()  # reference frame at the wheel origin, COG offset from it
    body.SetMass(MASS)
    body.SetFrameCOMToRef(chrono.ChFramed(chrono.ChVector3d(*COM), chrono.QUNIT))
    # statics only depends on mass and CoM; inertia set to a plausible box so the transient settles
    body.SetInertiaXX(chrono.ChVector3d(8.0, 12.0, 10.135))
    # SetPos on ChBodyAuxRef moves the COG, not the reference frame -- placing the robot with it
    # leaves the reference offset by the CoM lever and its z is then not the engine's z at all.
    body.SetFrameRefToAbs(
        chrono.ChFramed(chrono.ChVector3d(0, 0, WHEEL_RADIUS + 0.002), q)
    )
    body.EnableCollision(True)

    for _name, (wx, wy, wz) in WHEELS.items():
        if shape == "sphere":  # the engine's DEFAULT contact, and a clean single Bullet point
            sh = chrono.ChCollisionShapeSphere(mat, WHEEL_RADIUS)
            fr = chrono.ChFramed(chrono.ChVector3d(wx, wy, wz), chrono.QUNIT)
        else:  # cylinder axis is body +y; the shape's own axis is its frame z
            sh = chrono.ChCollisionShapeCylinder(mat, WHEEL_RADIUS, 2.0 * HALF_WIDTH)
            fr = chrono.ChFramed(chrono.ChVector3d(wx, wy, wz), chrono.QuatFromAngleX(math.pi / 2))
        body.AddCollisionShape(sh, fr)
    sys.Add(body)

    solver = chrono.ChSolverBB()
    solver.SetMaxIterations(2000)
    solver.SetTolerance(1e-10)
    sys.SetSolver(solver)
    return sys, body


def settle(sys, body, wheel_world_fn, dt: float = 1.0e-3, t_end: float = 3.0,
           avg_s: float = 0.5):
    """Integrate to rest and TIME-AVERAGE the contact loads over the last `avg_s`.

    No artificial damping. An earlier version scaled the body velocities down each step to force a
    rest state, which corrupts every reported force: the solver reports the impulse needed to
    absorb the velocity change that was just injected, so the flat plane came out asymmetric and
    the load sum wandered between 0.57 and 1.07 of m g. Left alone the body settles by itself
    (restitution 0), and averaging suppresses the point-count flicker Bullet shows on a cylinder.
    """
    n, navg = int(t_end / dt), int(avg_s / dt)
    acc = {k: 0.0 for k in WHEELS}
    pts = {k: 0 for k in WHEELS}
    cnt = 0
    for i in range(n):
        sys.DoStepDynamics(dt)
        if i >= n - navg:
            cb = ContactByWheel(wheel_world_fn())
            sys.GetContactContainer().ReportAllContacts(cb)
            for k in WHEELS:
                acc[k] += cb.loads[k]
                pts[k] += cb.points[k]
            cnt += 1
    loads = {k: acc[k] / cnt for k in WHEELS}
    return loads, {k: pts[k] / cnt for k in WHEELS}, float(body.GetLinVel().Length())


def measure(pitch_deg: float, roll_deg: float, shape: str) -> dict:
    sys, body = build(pitch_deg, roll_deg, shape)

    def wheel_world():
        r = body.GetFrameRefToAbs()
        out = {}
        for name, (wx, wy, wz) in WHEELS.items():
            p = r.TransformPointLocalToParent(chrono.ChVector3d(wx, wy, wz))
            out[name] = (p.x, p.y, p.z)
        return out

    loads, pts, v = settle(sys, body, wheel_world)
    ref = body.GetFrameRefToAbs()
    # Chrono's RotToCardanAnglesXYZ is intrinsic X-Y-Z; the engine uses Rz(yaw)Ry(pitch)Rx(roll),
    # so read the tilt off the body's own axes instead of trusting an Euler convention to match
    ez = ref.TransformDirectionLocalToParent(chrono.ChVector3d(0, 0, 1))
    ex = ref.TransformDirectionLocalToParent(chrono.ChVector3d(1, 0, 0))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -ex.z))))
    # Rz Ry Rx applied to +z gives ez.y = -sin(roll), so the sign here must be negated to read
    # the engine's roll convention rather than its mirror image
    roll = math.degrees(math.atan2(-ez.y, ez.z))
    weight = MASS * 9.81
    return {
        "pitch_deg": pitch,
        "roll_deg": roll,
        "z": float(ref.GetPos().z),
        "ref_xy": [float(ref.GetPos().x), float(ref.GetPos().y)],
        "loads": {k: loads[k] / weight for k in WHEELS},
        "contact_points": pts,
        "sum_loads": sum(loads.values()) / weight,
        "residual_vel": v,
        "shape": shape,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shape", default="sphere", choices=("sphere", "cylinder"))
    args = ap.parse_args()
    rows = []
    print(f"{'tilt':>18}{'pitch':>8}{'roll':>8}{'z':>8}"
          f"{'N_left':>9}{'N_right':>9}{'N_rear':>9}{'sum':>8}{'rest':>10}")
    for axis in ("pitch", "roll"):
        for deg in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0):
            r = measure(deg if axis == "pitch" else 0.0,
                        deg if axis == "roll" else 0.0, args.shape)
            r["axis"], r["tilt_deg"] = axis, deg
            rows.append(r)
            print(f"{f'{axis} {deg:.0f} deg':>18}{r['pitch_deg']:>8.2f}{r['roll_deg']:>8.2f}"
                  f"{r['z']:>8.4f}{r['loads']['left']:>9.4f}{r['loads']['right']:>9.4f}"
                  f"{r['loads']['rear']:>9.4f}{r['sum_loads']:>8.4f}"
                  f"{r['residual_vel']:>10.1e}")
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
