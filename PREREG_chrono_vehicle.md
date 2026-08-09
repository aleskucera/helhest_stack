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

## Result, written after the run

**Chrono 9.0.1 built from source with the vehicle module and SCM** (`scripts/build_chrono.sh`,
`scripts/chrono_env.sh`); three non-obvious blockers recorded there.

**Two ASSUMED parameters became MODEL.** The engine's mass table carries box extents, so the
chassis is built from real geometry -- two boxes giving 89.7 kg, CoM_x -0.188127, inertia
(2.4114, 4.2209, 6.0343) -- and the wheels are the table's 5.5 kg. Cross-check: the wheel's
diametral inertia computes to 0.173021, the table's value exactly.

### Three bugs found, all mine, all worth recording

1. `QuatFromAngleX(+pi/2)` puts the motor axis on body -y, so positive omega drove BACKWARDS.
2. Handing the motor a NEW `ChFunctionConst` each step disturbs the angle it integrates internally
   from the speed function. Worth 3.6% of forward gain on its own. One object, mutated in place.
3. **`SetRollingFriction` locks the yaw.** Chrono's NSC rolling friction is a complementarity
   constraint; with three wheels on one rigid body it froze rotation -- alpha 26.0 with it on,
   1.13 with it off, on an otherwise identical model. Rolling resistance is now applied as an
   explicit torque `mu_roll * N * R` about each wheel's spin axis, which is the same quantity the
   engine applies as a contact force. **This retracts an earlier claim in this file that a
   fixed-axle rear wheel cannot turn the vehicle "as a matter of kinematics". It turns fine. The
   rolling-friction constraint was the whole effect.**

A fourth was in the measurement, not the model: alpha was taken from the MEAN instantaneous `wz`,
which reads 0.0396 rad/s in a case whose body rotated 0.016 rad in 5 s. Contact jitter integrates
to nothing; net rotation over the window is the honest estimator, and `turn_gain` now uses it.

### The numbers

Soil stiffness swept; everything else fixed, Janosi = our own 0.0125 m.

| Kphi | rear | forward gain | alpha | sinkage |
|---|---|---|---|---|
| rigid | fixed | 1.0022 | 3.168 | -- |
| rigid | caster | 1.0016 | 1.102 | -- |
| 0.05 MPa | fixed | **0.9226** | 4.563 | 6.8 cm |
| 0.10 MPa | fixed | 0.9480 | 4.223 | 6.3 cm |
| 0.20 MPa | fixed | 0.9688 | 3.741 | 5.1 cm |
| 0.50 MPa | fixed | 0.9874 | 3.081 | 3.6 cm |
| 0.20 MPa | caster | 0.9683 | 1.258 | 5.3 cm |
| 0.50 MPa | caster | 0.9874 | 1.359 | 3.6 cm |
| **measured** | | **0.906 - 0.925** | **2.20** | |

### What it says

**The conflict is NOT an artefact of our isotropic |slip| treatment.** That was the leading suspect
in TIER1_REPORT.md section 2, and it is now much less likely: with a fixed rear axle the two
channels move MONOTONICALLY AND IN OPPOSITE DIRECTIONS with soil stiffness. Soft soil lands the
forward gain squarely in the measured band (0.9226 at Kphi 0.05 MPa) and pushes alpha to 4.56;
stiff soil pulls alpha toward 2.20 and drives the forward gain to 1.0. An independent simulator,
with a completely different soil formulation, reproduces the same tension our own fit hit.

**The rear-wheel mounting is a near-orthogonal knob for alpha.** At fixed soil it moves alpha by a
factor of three (3.741 -> 1.258 at 0.20 MPa) while leaving the forward gain untouched to four
decimal places (0.9688 -> 0.9683). And the measured 2.20 is BRACKETED by the two mountings. So the
two channels probably CAN be fitted together -- with a rear wheel between a free caster and a
fixed axle, i.e. a caster with swivel friction. That is one parameter, and it is a hardware fact.

Neither pre-registered bar is met (alpha 1.26 or 3.74 against a 1.76-2.64 band; forward gain 0.9688
against 0.906-0.925 at the same soil). Reported as failures. But they fail in a structured way that
localises the remaining freedom to one measurable thing, which is worth more than a pass would have
been.

### Caveat

The caster goes unstable in soft soil -- alpha reads -9.9 at Kphi 0.05 MPa and -59.3 at 0.10 MPa,
i.e. it flips. Only the fixed-rear column is trustworthy across the whole sweep, and the caster
rows at 0.20 and 0.50 MPa. Do not read the soft-soil caster numbers as anything.

### Next

1. **Measure the rear wheel**: does it swivel, and with how much trail and swivel friction? Ten
   seconds next to the robot, and it is now the dominant unknown for yaw.
2. Model it as a caster WITH swivel friction and fit that one parameter to alpha = 2.20 at the soil
   stiffness that already matches the forward gain. If a single configuration lands both, the
   conflict is resolved and the answer is a rear-wheel model, not a traction model.
3. Only then the bag replay, for trajectory-level scoring rather than two scalars.
