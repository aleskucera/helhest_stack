"""The helhest robot in Project Chrono: chassis, three driven wheels, rigid or SCM terrain.

    <chrono-env>/bin/python scripts/chrono_vehicle_model.py --terrain rigid --out /tmp/veh.json

Built to the spec in PREREG_chrono_vehicle.md. Every parameter is tagged MODEL (from RobotParams),
BAG (measured on the robot, TIER1_REPORT.md section 3) or ASSUMED, and the two predictions being
scored are the turn gain alpha = 2.20 and the forward gain 0.906-0.925 -- the pair our own traction
fit could not satisfy simultaneously.

NOT a ChWheeledVehicle. Those templates are built around suspensions, steering and a driveline,
and this robot has none: three wheels bolted to the chassis, skid-steered. The idiomatic Chrono
shape for that is the ROVER pattern (chrono_models' Viper and Curiosity are chassis + wheel bodies
+ revolute motors over SCMTerrain, and are not ChWheeledVehicles either). The vehicle module is
still needed because SCMTerrain lives in it.

Wheels are 48-gon convex-hull prisms rather than ChCollisionShapeCylinder. Chrono's default
collision detection returns a SINGLE contact point for a cylinder on a plane, which is degenerate
for what is physically a line contact -- it made a flat plane read asymmetric and transferred load
the wrong way at 10 deg of bank (PREREG_chrono.md addendum). The hull gives a real manifold and
reproduces the sphere's loads exactly on the flat.
"""

from __future__ import annotations

import argparse
import json
import math

import pychrono as chrono
import pychrono.vehicle as veh

# --- MODEL: straight out of RobotParams -------------------------------------------------------
MASS = 106.2  # [kg] total
COM = (-0.198, 0.0, 0.0)  # [m] body frame
WHEEL_RADIUS = 0.35
HALF_WIDTH = 0.05  # half the ruler-measured 0.10 m tread
HALF_TRACK = 0.365
REAR_OFFSET = 0.75
YAW_INERTIA = 10.135  # [kg m^2] whole-robot, about the CoM -- a CROSS-CHECK, not an input here
# The engine's mass table in full, so the Chrono model inherits the same geometry rather than a
# guess: two chassis boxes (centre_x, mass, length, width, height) plus three 5.5 kg wheels. The
# chassis body below carries only the BOXES -- Chrono composes the wheels' contribution itself.
CHASSIS_BOXES = ((-0.13, 78.8375, 0.48, 0.56, 0.20), (-0.61, 10.8625, 0.48, 0.24, 0.20))
CHASSIS_MASS = 89.7
CHASSIS_COM_X = -0.188127
CHASSIS_INERTIA = (2.4114, 4.2209, 6.0343)  # about the chassis's own CoM
WHEEL_MASS = 5.5
WHEEL_I_AXIS = 0.336875  # 1/2 m R^2
WHEEL_I_DIAM = 0.173021  # 1/12 m (3R^2 + h^2) -- matches the engine's table exactly
WHEELS = {
    "left": (0.0, HALF_TRACK, 0.0),
    "right": (0.0, -HALF_TRACK, 0.0),
    "rear": (-REAR_OFFSET, 0.0, 0.0),
}

# --- BAG: measured on the robot ---------------------------------------------------------------
MOTOR_TAU = 0.19  # [s] first-order actuator lag, joint fit over 34 setpoint steps
TORQUE_LIMIT = 105.0  # [Nm] per wheel; a LOWER bound, nothing ever saturated
ROLLING_RESISTANCE = 0.09  # fraction of normal load; Chrono wants a torque arm, so x R below
JANOSI_K = 0.0125  # [m] shear modulus, from L/K = 12 at contact length L ~ 0.15 m

# --- ASSUMED: flagged, and nothing may rest on these ------------------------------------------
MU = 0.8  # the planner's plan_friction default
HULL_FACETS = 48
CASTER_TRAIL = 0.08  # [m] swivel axis ahead of the contact patch; UNMEASURED


def hull_points(radius: float, half_width: float, n: int) -> "chrono.vector_ChVector3d":
    """A prism CIRCUMSCRIBING the true cylinder, so the flat faces are tangent to it and the
    resting height is exact rather than up to 1.7 mm low."""
    rr = radius / math.cos(math.pi / n)
    pts = chrono.vector_ChVector3d()
    for i in range(n):
        a = 2.0 * math.pi * i / n
        for s in (-half_width, half_width):
            pts.push_back(chrono.ChVector3d(rr * math.cos(a), s, rr * math.sin(a)))
    return pts


def build_robot(sys, mat, rear: str = "caster"):
    """Chassis + three driven wheels. Returns (chassis, wheels, motor functions).

    `rear` decides how the trailing wheel is MOUNTED, which turns out to decide whether this
    vehicle can turn at all:

      fixed   its axle is parallel to the front pair. Then the vehicle is kinematically unable to
              yaw without skidding that wheel sideways on a 0.75 m lever, and Chrono says it
              essentially does not yaw (alpha ~ 24 against a measured 2.20).
      caster  the hub swivels about a vertical axis, so the wheel trails the motion. This is what
              docs/motion_model_pipeline.md implies by calling the rear wheel "trailing" and
              "kinematically redundant" -- a fixed axle would not be redundant, it would pin the
              instantaneous centre onto the rear axle line.
    """
    chassis = chrono.ChBodyAuxRef()
    chassis.SetMass(CHASSIS_MASS)
    # the reference frame is the wheel-hub plane origin, as in the engine; the CoM sits behind it
    chassis.SetFrameCOMToRef(
        chrono.ChFramed(chrono.ChVector3d(CHASSIS_COM_X, 0.0, 0.0), chrono.QUNIT)
    )
    chassis.SetInertiaXX(chrono.ChVector3d(*CHASSIS_INERTIA))
    chassis.SetFrameRefToAbs(
        chrono.ChFramed(chrono.ChVector3d(0, 0, WHEEL_RADIUS), chrono.QUNIT)
    )
    chassis.EnableCollision(False)  # the belly is not in contact in any case tested here
    sys.Add(chassis)

    wheels, motors = {}, {}
    for name, (wx, wy, wz) in WHEELS.items():
        parent = chassis
        if name == "rear" and rear == "caster":
            # a light swivel link between chassis and wheel, free about the vertical hub axis
            link = chrono.ChBody()
            link.SetMass(0.5)
            link.SetInertiaXX(chrono.ChVector3d(0.01, 0.01, 0.01))
            link.SetPos(chrono.ChVector3d(wx + CASTER_TRAIL, wy, wz + WHEEL_RADIUS))
            link.EnableCollision(False)
            sys.Add(link)
            # The swivel axis must sit AHEAD of the contact patch. With zero trail the caster has
            # no self-aligning moment at all -- it stays wherever it started and behaves exactly
            # like a fixed axle, which is what a first version of this measured.
            swivel = chrono.ChLinkLockRevolute()
            swivel.Initialize(
                link, chassis,
                chrono.ChFramed(
                    chrono.ChVector3d(wx + CASTER_TRAIL, wy, wz + WHEEL_RADIUS), chrono.QUNIT
                ),
            )
            sys.Add(swivel)
            parent = link
        w = chrono.ChBody()
        w.SetMass(WHEEL_MASS)
        w.SetInertiaXX(chrono.ChVector3d(WHEEL_I_DIAM, WHEEL_I_AXIS, WHEEL_I_DIAM))
        w.SetPos(chrono.ChVector3d(wx, wy, wz + WHEEL_RADIUS))
        w.EnableCollision(True)
        hull = hull_points(WHEEL_RADIUS, HALF_WIDTH, HULL_FACETS)
        w.AddCollisionShape(
            chrono.ChCollisionShapeConvexHull(mat, hull),
            chrono.ChFramed(chrono.ChVector3d(0, 0, 0), chrono.QUNIT),
        )
        sys.Add(w)
        # Speed-controlled revolute, because the engine commands wheel SPEED too. The motor spins
        # about its frame's Z, and Rx(+90) would map that onto body -y -- positive omega would
        # then drive the robot BACKWARDS. Rx(-90) puts it on body +y, which is forward.
        m = chrono.ChLinkMotorRotationSpeed()
        m.Initialize(
            w, parent,
            chrono.ChFramed(chrono.ChVector3d(wx, wy, wz + WHEEL_RADIUS),
                            chrono.QuatFromAngleX(-math.pi / 2)),
        )
        # ONE function object per motor, mutated in place. Handing the motor a NEW ChFunctionConst
        # every step disturbs the angle it integrates internally from the speed function.
        fn = chrono.ChFunctionConst(0.0)
        m.SetSpeedFunction(fn)
        sys.Add(m)
        wheels[name], motors[name] = w, fn
    return chassis, wheels, motors


def rigid_terrain(sys, mat):
    ground = chrono.ChBodyEasyBox(60, 60, 1.0, 1000, True, True, mat)
    ground.SetPos(chrono.ChVector3d(0, 0, -0.5))
    ground.SetFixed(True)
    sys.Add(ground)
    return ground


def scm_terrain(sys):
    """Deformable soil. Only the Janosi parameter is ours; the Bekker set is a named soil held
    fixed, because nothing measured on this robot constrains pressure-sinkage."""
    terrain = veh.SCMTerrain(sys)
    # Bekker set taken verbatim from Chrono's own demo_VEH_SCMTerrain_RigidTire, and held fixed:
    # nothing measured on this robot constrains pressure-sinkage, so inventing values would just
    # move the answer. Only Janosi is ours.
    terrain.SetSoilParameters(
        0.2e6,  # Bekker Kphi [Pa/m^n]
        0.0,  # Bekker Kc
        1.1,  # Bekker n exponent
        0.0,  # Mohr cohesion [Pa]
        30.0,  # Mohr friction angle [deg]
        JANOSI_K,  # BAG: from our own L/K fit -- the one parameter this comparison is about
        4e7,  # elastic stiffness [Pa/m], must exceed Kphi
        3e4,  # damping [Pa s/m]
    )
    # SCM's default reference frame is already ISO / Z-up, which is ours, so no rotation here
    terrain.SetPlane(chrono.ChCoordsysd(chrono.ChVector3d(0, 0, 0), chrono.QUNIT))
    terrain.Initialize(24.0, 12.0, 0.04)
    return terrain


def drive(terrain_kind: str, omega: tuple[float, float, float], t_end: float,
          dt: float = 2.0e-3, rear: str = "caster") -> dict:
    """Run one commanded-wheel-speed manoeuvre and return the chassis trajectory."""
    # a tight collision envelope: the default leaves the robot riding ~1.4 mm high, which is
    # noise on a rigid plane but would be read as sinkage against SCM
    chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.0005)
    chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.0005)
    sys = chrono.ChSystemNSC()
    sys.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
    sys.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    mat = chrono.ChContactMaterialNSC()
    mat.SetFriction(MU)
    mat.SetRestitution(0.0)
    # Chrono's rolling friction is a torque per unit normal force; the engine's mu_roll is a
    # force fraction, so the arm is mu_roll * R
    mat.SetRollingFriction(ROLLING_RESISTANCE * WHEEL_RADIUS)

    chassis, wheels, motors = build_robot(sys, mat, rear)
    if terrain_kind == "rigid":
        terrain = rigid_terrain(sys, mat)
    else:
        terrain = scm_terrain(sys)
        # restrict the active soil region to a box that follows each wheel, as Chrono's demos do
        for w in wheels.values():
            terrain.AddMovingPatch(
                w, chrono.ChVector3d(0, 0, 0),
                chrono.ChVector3d(0.5, 4.0 * HALF_WIDTH, 2.0 * WHEEL_RADIUS),
            )

    solver = chrono.ChSolverBB()
    solver.SetMaxIterations(2000)
    solver.SetTolerance(1e-10)
    sys.SetSolver(solver)

    cmd = dict(zip(("left", "right", "rear"), omega))
    realized = {k: 0.0 for k in cmd}
    traj = []
    n = int(t_end / dt)
    for i in range(n):
        # BAG: the measured 0.19 s first-order actuator lag, applied exactly as the engine does,
        # so the spin-up is set by the measurement rather than by the assumed wheel inertia
        blend = dt / (dt + MOTOR_TAU)
        for k in cmd:
            realized[k] += blend * (cmd[k] - realized[k])
            motors[k].SetConstant(realized[k])
        if terrain_kind == "scm":
            terrain.Advance(dt)
        sys.DoStepDynamics(dt)
        if i % 10 == 0:
            f = chassis.GetFrameRefToAbs()
            ez = f.TransformDirectionLocalToParent(chrono.ChVector3d(1, 0, 0))
            traj.append(
                {
                    "t": (i + 1) * dt,
                    "x": float(f.GetPos().x),
                    "y": float(f.GetPos().y),
                    "z": float(f.GetPos().z),
                    "yaw": math.atan2(ez.y, ez.x),
                    "wz": float(chassis.GetAngVelParent().z),
                }
            )
    return {"omega": list(omega), "terrain": terrain_kind, "traj": traj}


def forward_gain(run: dict) -> float:
    """Distance travelled over commanded wheel distance, over the steady second half."""
    tr = run["traj"]
    half = tr[len(tr) // 2:]
    d = math.hypot(half[-1]["x"] - half[0]["x"], half[-1]["y"] - half[0]["y"])
    v_cmd = WHEEL_RADIUS * 0.5 * (run["omega"][0] + run["omega"][1])
    return d / (v_cmd * (half[-1]["t"] - half[0]["t"]))


def turn_gain(run: dict) -> float:
    """alpha = ideal differential-drive yaw rate / realised yaw rate, over the steady half."""
    tr = run["traj"]
    half = tr[len(tr) // 2:]
    wz = sum(p["wz"] for p in half) / len(half)
    ideal = WHEEL_RADIUS * (run["omega"][1] - run["omega"][0]) / (2.0 * HALF_TRACK)
    return float("nan") if abs(wz) < 1e-6 else ideal / wz


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--terrain", default="rigid", choices=("rigid", "scm"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--t-end", type=float, default=6.0)
    args = ap.parse_args()

    out = {"terrain": args.terrain, "forward": [], "turn": []}
    print(f"=== forward gain (target 0.906-0.925 from the bags) -- {args.terrain} terrain ===")
    for w in (1.0, 2.0, 3.0, 4.0):
        r = drive(args.terrain, (w, w, w), args.t_end)
        g = forward_gain(r)
        out["forward"].append({"omega": w, "gain": g})
        print(f"  omega {w:.1f} rad/s  ->  gain {g:.4f}")

    print(f"\n=== turn gain alpha (target 2.20 from two IMUs) -- {args.terrain} terrain ===")
    for wl, wr in ((1.0, 3.0), (1.5, 3.5), (0.5, 3.5), (2.0, 4.0)):
        r = drive(args.terrain, (wl, wr, 0.5 * (wl + wr)), args.t_end)
        a = turn_gain(r)
        out["turn"].append({"wl": wl, "wr": wr, "alpha": a})
        print(f"  wL {wl:.1f} wR {wr:.1f}  ->  alpha {a:.4f}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
