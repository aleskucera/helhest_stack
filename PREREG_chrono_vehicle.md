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
