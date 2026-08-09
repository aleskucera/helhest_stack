# Pre-registration — the Chrono vehicle model, and the two numbers it has to reproduce

Written before the model was built. Follows PREREG_chrono.md, whose rigid statics gate passed
(geometry to 0.1 mm, flat loads to 3e-4 m g) and which found the engine's slope loads wrong by
0.249 m g.

## Why this model is not a `ChWheeledVehicle`

Chrono::Vehicle's wheeled templates are built around suspensions, steering and a driveline. This
robot has none of those: three wheels rigidly mounted to the chassis, skid-steered, no springs, no
steer axis. Wrapping it in the vehicle templates would add scaffolding that models nothing and
then has to be neutralised.

The idiomatic Chrono construction for exactly this shape is the ROVER pattern -- `chrono_models`'
Viper and Curiosity are chassis + wheel bodies + revolute motors, driven over `SCMTerrain`, and
they are not `ChWheeledVehicle`s either. So: chassis body, three wheel bodies, three revolute
motors, `SCMTerrain` from the vehicle module for deformable soil and a rigid plane for the
control case. The vehicle module is still required -- SCM lives in it -- which is why it is in
the build.

## Parameters, and where each one comes from

Three provenance classes, kept separate on purpose: MODEL (from `RobotParams`, exact), BAG
(measured on the robot, recorded in TIER1_REPORT.md section 3), ASSUMED (neither -- flagged, and
no conclusion may rest on one).

| quantity | value | provenance |
|---|---|---|
| total mass | 106.2 kg | MODEL |
| CoM | (-0.198, 0, 0) m | MODEL |
| yaw inertia I_zz | 10.135 kg m^2 | MODEL (from the mass table) |
| wheel radius / tread | 0.35 / 0.10 m | MODEL (tread ruler-measured, 1b8b0b4) |
| wheel positions | front (0, +/-0.365), rear (-0.75, 0) | MODEL |
| roll/pitch inertia | 8 / 12 kg m^2 | **ASSUMED** — statics is insensitive; transients are not |
| wheel mass / inertia | 5 kg, 0.31 kg m^2 | **ASSUMED** — see the note on tau below |
| actuator lag tau | 0.19 s | BAG (joint fit, 34 step responses) |
| per-wheel torque limit | 105 Nm | BAG (lower bound; nothing ever saturated) |
| rolling resistance | 0.09 x weight | BAG (torque-calibration offset) |
| Janosi shear modulus K | 0.0125 m | BAG, via L/K = 12 at contact length L ~ 0.15 m |
| Coulomb friction mu | 0.8 | ASSUMED (the planner's `plan_friction` default) |

Wheel inertia is ASSUMED but should not matter much: the commanded wheel speed is put through the
measured 0.19 s first-order lag before it reaches the motor, exactly as the engine does, so the
spin-up transient is set by the measurement rather than by the guessed inertia.

Chrono's rolling friction is a torque per unit normal force, so the engine's force-fraction
`mu_roll = 0.09` maps to `SetRollingFriction(0.09 * R) = 0.0315 m`.

The remaining SCM soil parameters -- Bekker Kphi, Kc, n, Mohr cohesion and friction angle -- are
NOT determined by anything measured here. They will be taken from a single named soil in Chrono's
own SCM examples and held fixed. Only the Janosi parameter is pinned by our own fit, and that is
the one the comparison is about.

## The two predictions

The point of this model is not to reproduce our engine. It is to be an INDEPENDENT instrument for
the one thing our engine could not do: fit both channels at once. TIER1_REPORT.md section 2 records
the conflict -- a single isotropic `(L/K, mu_roll)` gives either the measured turn gain or the
measured forward gain, never both. Chrono is not party to that fit, so it can arbitrate.

1. **Turn gain.** Driven with a wheel differential on flat ground, the model's realised yaw rate
   over the ideal differential-drive yaw rate should come out at `alpha ~ 2.20`, the value two
   IMUs agree on to 1% (`fit_turn_gain.py`). Bar: within **20%**, i.e. 1.76 to 2.64. Our engine
   only reaches this by SETTING `k_turn`; in Chrono it is emergent, so agreement would be real
   corroboration and disagreement would say the skid-steer yaw loss is not what we think.

2. **Forward gain.** Driven straight, distance travelled over commanded wheel distance should be
   **0.906 to 0.925**. Bar: within **0.05**. The engine's legacy model gives exactly 1.000 by
   construction and the shear model needs rolling resistance to get near 0.91.

The interesting outcome is a SPLIT: if Chrono reproduces one and not the other with a single
parameter set, that localises the conflict to a mechanism rather than to a fitting artefact.

## What is blocked, and what is not

The robot at 192.168.18.5 is unreachable today, so no raw bag can be replayed. Every parameter
above is a bag-DERIVED number already recorded, so the model can be built and both predictions
scored without the raw data. What cannot be done until the robot is up is trajectory-level
scoring -- driving Chrono on the measured wheel speeds and comparing yaw rate against the gyro,
which is what `scripts/replay_traction.py` does for the engine. That stays open.

## What would change as a result

- Both pass -> Chrono, with our own L/K, reproduces both channels the engine could not fit
  together. That makes the conflict an artefact of the isotropic `|slip|` treatment, which is the
  suspect the report already names, and it tells us what to fix rather than that something is
  wrong.
- Turn passes, forward fails (or vice versa) -> the conflict is real physics and localised.
- Both fail -> the soil parameterisation, not our fit, is doing the work; report and stop, because
  nothing downstream would be trustworthy.

---

## Result, written after the run: forward channel PASSES, turn channel is NOT TRUSTWORTHY

**Chrono 9.0.1 is built from source with the vehicle module and SCM** (`scripts/build_chrono.sh`,
`scripts/chrono_env.sh`). Three non-obvious blockers are recorded in the build script.

**Two ASSUMED parameters became MODEL.** The engine's mass table carries box extents, so the
chassis is now built from the real geometry -- two boxes (78.8375 kg at x=-0.13, 0.48x0.56x0.20;
10.8625 kg at x=-0.61, 0.48x0.24x0.20) giving mass 89.7 kg, CoM_x -0.188127 and inertia
(2.4114, 4.2209, 6.0343) about its own CoM -- and the wheels are the table's 5.5 kg. Cross-check:
the wheel's diametral inertia computes to 0.173021, which is the engine's table value exactly.

**Forward gain: PASS as a model check, and it is not the bag number.** On rigid ground the model
returns 1.0016 -- pure rolling, no slip, and the 0.16% residual is the circumscribed 48-gon's
+0.21% radius. That is the correct answer for RIGID ground and it says the drivetrain, the lag and
the contact are wired right. It cannot be compared against the bags' 0.906-0.925, which is a soil
number; that comparison needs the SCM run.

Getting there took two real bugs, both worth recording: `QuatFromAngleX(+pi/2)` puts the motor
axis on body -y, so positive omega drove the robot BACKWARDS; and handing the motor a NEW
`ChFunctionConst` every step disturbs the angle it integrates internally from the speed function,
which alone accounted for a 3.6% forward-gain error. One function object, mutated in place.

**Turn gain: NOT REPRODUCED, and the model is not trustworthy here.** alpha should be 2.20.
Measured across the variants:

| rear mount | mu | alpha |
|---|---|---|
| fixed axle | 0.8 | 24.2 |
| caster, zero trail | 0.8 | 57.6 |
| caster, 8 cm trail | 0.8 | 29.9 |
| caster, 8 cm trail | 0.4 | **0.86** |
| caster, 15 cm trail | 0.8 | 31.7 |

A physical model does not jump 24 -> 57 -> 30 -> 0.86 -> 32 under small parameter changes. This is
bimodal -- essentially locked, or free -- so no value here is evidence about the real robot, and
tuning until one of them reads 2.20 would be fitting noise.

Two things WERE learned on the way, and they survive the above:

- **A fixed-axle rear wheel cannot turn this vehicle**, and that is kinematics rather than a
  solver artifact: a rear wheel whose axle is parallel to the front pair pins the instantaneous
  centre onto its own axle line. It also explains the engine's own wording --
  docs/motion_model_pipeline.md calls the rear wheel "trailing" and "kinematically redundant",
  and a fixed axle would be neither. Whether the hardware has a swivel is a ten-second question
  for someone standing next to the robot, and it should be answered before any of this is retried.
- **A zero-trail caster is not a caster.** With the swivel axis through the contact patch there is
  no self-aligning moment, and it behaves exactly like a fixed axle. Trail must be measured too.

**The most likely cause of the bimodality, and the next thing to try:** the wheels are driven by
`ChLinkMotorRotationSpeed`, which imposes the commanded spin as a HARD constraint. Three hard
speed constraints on one rigid body over rigid contacts is over-determined -- the system can only
resolve it by slipping, and whether it slips or locks is exactly the kind of knife-edge that
produces this. The fix is to drive the wheels with TORQUE motors under a speed controller, capped
at the measured 105 Nm, which is also closer to the real drivetrain. That is the first thing to do
before trusting any turn number, and before running SCM at all.
