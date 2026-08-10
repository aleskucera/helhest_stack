# Engine work — session report

Two stacked branches off `main` (`de96100`). Neither is merged.

```
main
 └── engine/tier1-certificates          HISTORICAL -- do not work here (see below)
      └── engine/exact-arc-integration  <- THE WORKING BRANCH; everything is here
```

**Work on `engine/exact-arc-integration`.** `tier1-certificates` is an ANCESTOR of it, not an
alternative: an intermediate point in the same history. It was originally described here as the
smaller, safer merge candidate, and that is no longer true. What made it "safe" was that its Tier-1
additions were all default-on and inert -- and those certificates were later REMOVED (§1.1), after
Chrono showed the tip-over margin could not work and measurement showed the other two cost 40% of
the rollout for outputs nothing read. So `tier1-certificates` is now the one branch that still
carries code the project has deliberately deleted, and it has none of the load-transfer fix, the
Chrono validation, or the recording tooling. It is pushed for history and nothing else.

Everything on the working branch is additive or opt-in, with two stated exceptions: the exact-arc
integrator (§2) and the `normal_loads` tangential reaction (§1.2), both of which change existing
outputs on purpose and are validated against Chrono.

---

## 1. Tier 1 — complete

The four commissioned items, in the order IMPROVEMENTS.md recommends. Each verified against a
closed-form answer rather than eyeballed. `tests/engine/golden.py` pins every simulator output for
a seeded batch and passed at every commit on this branch.

| item | state |
|---|---|
| §3 tip-over margin | built, verified, **since REMOVED** — see §1.1 |
| §1 friction saturation | built, crossed 1.0 at `tan θ = μ` to 1.2e-6, **since REMOVED** — see §1.1 |
| §2 torque / stall | built, boundary independent of μ to 7e-7, **since REMOVED** — see §1.1 |
| §4 cylinder envelope | `wheel_width`, 32 yaw bins, +1.7% rollout cost |

**Finding: `min N_i` is not a tip-over test on this robot.** `normal_loads` balances vertical force
and horizontal torque with contact NORMALS only — the tangential friction reaction that holds the
robot on a slope, and its moment about the CoM, are absent. Two identities follow on any uniform
plane and hold to 1e-6:

```
sum_i N_i / (m g) = 1 / (cos pitch cos roll)
min_i N_i / (m g) = (|com_x| / rear_offset) / (cos pitch cos roll) = 0.2637 / (cp cr)
```

So the margin *rises* with tilt: 0.264 flat, 0.273 exactly where the `max_roll = 15°` gate fires,
0.410 at 50° of bank. It never crossed 0 across 12 shape worlds either. **Only the gate protects
against slope tip-over**, and a planner consuming this margin as a stability cost would be reading
a load-transfer diagnostic. Fixing it means adding the tangential moment to `normal_loads`, which
changes an existing output and was out of scope.

**Finding: the friction budget cannot come from `normal_loads` raw.** Same root cause — using
`Σ μᵢNᵢ` puts the crossing at `sin θ cos θ = μ`, which peaks at 0.5, so the certificate could never
fire on μ > 0.5 terrain. Only the load *ratios* are taken from the solve, where the bias cancels.

**Firing rates**, B=512 rollouts on synthetic terrain: `saturation` reaches 0.84 at 25° and μ=0.6,
and fires only on 15°+ slopes at μ=0.3. `stability` sat in [0.259, 0.291] everywhere. `stall` never
exceeded 0.49.

### 1.1 All three certificates were then removed

They were correct and they were inert, and the second fact turned out to cost more than the first
was worth. Measured at B=4096, T=25 on this machine:

| | ms | share |
|---|---|---|
| `main` | 1.23 | — |
| this branch, with certificates | 2.60 | |
| this branch, without them | **1.55** | **1.05 ms, 40%** |

IMPROVEMENTS.md §1 estimated "<1%". The gap is ~40x. It splits 0.41 ms for `contact_grip` and
0.65 ms for the saturation/stall arithmetic; the tip-over margin was genuinely free (one `min`
over registers) and was removed only because it does not measure what its name says.

**They could not have earned it at the current settings, and that is algebra rather than luck.**
Substituting `sin²p + cos²p sin²r = 1 - cos²p cos²r` into the demand, and `cos θ = cos p cos r`
for the total tilt, the whole certificate collapses to

    saturation  =  tan(θ) / mu_bar          (exactly, with the centripetal term at zero)

verified against the engine to 2.5e-6 at headings 0/45/90 deg on 10 and 20 deg ramps. Two
consequences. With a UNIFORM friction field `mu_bar` is a constant, so the per-step work computes
a fixed function of the pitch and roll the settle already has -- `contact_grip` samples the
terrain three times to recover a number that was known on the host. And the steepest total tilt
the existing gates admit is the box corner, `cos θ = cos 25 cos 15` → 28.9 deg, so saturation
caps at `tan(28.9°)/mu = 0.552/mu`: at the deployed `plan_friction = 0.8` it cannot exceed 0.69.
Not "rarely fires" — cannot fire.

Verified decision-neutral, not just argued: every surviving array in `golden_fixture.npz` stayed
bit-identical before the fixture was regenerated to drop the nine stale keys, and the whole
`scripts/model_ablation.py` sweep -- 280 arrays, per-candidate cost and endpoints over 5 model
levels x 28 scenarios -- is bit-identical, with the executed elite-mean command unchanged to 0.0.

**What to restore them for.** The certificate's only real content is making the tilt limit depend
on the LOCAL friction; the `max_roll` / `max_pitch_up` gates are mu-blind constants. Nothing in
the perception stack estimates mu per cell today (`elevation_node` calls `set_uniform_friction`),
so that content is unreachable. The day a per-cell mu exists, `contact_grip` is the piece that is
needed back, and this commit is the place to read it from. Until then, the same decision is
available for free on the host: set the tilt gate to `atan(mu)` at plan build, which also makes
the cost-to-go ROUTER friction-aware -- something the per-step certificate never did, since the
lattice solver runs no rollouts.

`scripts/saturation_from_bags.py` is unaffected: it computes the ratio from bag data in numpy and
imports no engine code, so it remains the way to decide whether the terrain ever warrants this.

---

## 2. Engine fidelity work

### Exact-arc pose integration — the one behaviour change

`integrate_pose` replaces forward Euler with the closed-form arc of a constant twist. Euler took
the chord and was first order in dt; on flat ground with constant wheel speeds, where the model's
own exact answer exists in closed form, it landed **10.9 / 18.7 / 17.7 cm** off over a 2.5 s horizon
at 2.1 m/s (gentle / hard / tight turn). Yaw was never wrong — only position lagged.

The engine now sits on the analytic arc to **0.000 cm at every dt**, and on bumpy terrain halving
dt moves the endpoint **0.7 mm** where Euler moved 93 mm. **That removes the case for dt = 0.05**:
dt = 0.1 is converged for trajectory accuracy even at 2.1 m/s.

The golden fixture was regenerated in that commit. Scale of the change: `controlled` 3.8 mm,
`derived` 0.3 mm, `loads` 0.02 N. The numpy reference in `reference/state.py` got the same update,
since it is the finite-difference oracle for this path.

### Actuator response — a lag, not a transport delay

`dynamics.MOTOR_TAU = 0.19 s` (first-order, via the engine's existing `tau_motor`) plus
`COMMAND_DELAY = 0.04 s` of residual dead time, which rounds to zero whole steps at dt = 0.1.
`SolverParams.command_delay` and `elevation_node`'s `command_history` implement the dead-time path
and stay in place for a shorter dt.

**This corrects an earlier reading in this same branch.** Fitting the delay first by
cross-correlation and the lag second reports ~170 ms of *pure* delay — because cross-correlation
returns the group delay of a slow rise. Fitting both jointly puts the drive wheels at tau
0.17–0.20 s with 0–50 ms of dead time, and fits better (RMSE 0.395 vs 0.412). The decisive
evidence is the step response: averaged over 34 setpoint steps the wheel is already moving 10 ms
in and passes 50% at ~140 ms. **There is no dead zone**, so there was nothing for a 170 ms
transport delay to describe.

What survives from that investigation: the lag is physical, not a timing artifact (every topic's
header sits within 2 ms of its log time; reported velocity tracks encoder position to 10–20 ms at
correlation 1.000; the IMU sees it too), and `/cmd_joints` reaches the LLC's echoed setpoint in
**10 ms**, so essentially all of it is downstream of ROS. Whether the remaining ~0.19 s is
controller tuning or the robot's own torque-limited acceleration is *not* settled: 105 Nm/wheel
gives a = 8.5 m/s², which reaches 1.4 m/s in 160 ms — the same order. A standing-start step in the
calibration drive would separate them.

### Traction: shear compliance, momentum, rolling resistance

All behind `SolverParams.shear_lk` (default 0 = legacy). The model solves the body twist from a
force balance where each contact follows the Janosi–Hanamoto shear curve, with
`λ = (|slip| / |Rω|) · (L/K)` — a slip *ratio*, not a velocity.

That distinction is the content. Under rigid Coulomb the front wheels supply any yaw moment at
essentially zero longitudinal slip, so **α is exactly 1.000**, and no rate-independent model can
depend on forward speed at all (adding a common speed leaves every slip velocity unchanged).
Finite compliance is what makes α exceed 1.

- **Momentum** is implicit, inside the same 3×3 Newton. Explicit would need dt ≈ 5 ms and a 20×
  horizon — the §9(a) wall. Stable and monotone at dt from 0.02 to **0.5 s**. `I_zz = 10.135 kg m²`
  is derived from the mass table, and a sweep over 0.1×–8× puts the optimum exactly there.
- **Rolling resistance** `μ_roll = 0.09` is measured, not fitted — the torque calibration's offset
  is 36–38 Nm ≈ 106 N ≈ 0.09 × weight. It is also what makes the robot coast to a stop.

**Scored on bags** — the engine driven at 100 Hz on *measured* wheel speeds, predicted yaw rate vs
gyro, 12,254 samples:

| model | RMS all | RMS quasi-static |
|---|---|---|
| legacy | 0.1512 | 0.0455 |
| shear | 0.1630 | 0.0384 |
| shear + momentum | **0.1368** | 0.0399 |

Better — but the criterion I set was whether all-sample RMS falls toward the 0.04 that quasi-static
samples reach, and it reached 0.137. **Inertia explains part of the transient residual, not most of
it.** The residual still correlates −0.136 with the gyro's own acceleration, i.e. the real robot
responds *faster* than the model.

**Open conflict.** One isotropic `(L/K, μ_roll)` does not fit both channels: α = 2.20 wants
L/K ≈ 8–12 at low resistance, while the forward gain of 0.915 wants μ_roll ≈ 0.15–0.25, which
drives α to 2.6–3.1. Either the isotropic `|slip|` treatment is wrong — terramechanics distinguishes
longitudinal from lateral shear moduli — or the forward target is soft, since it comes from
*commanded* wheels on Odin where no `/joint_states` exists to confirm 1:1 forward realisation.

---

## 3. Measurements from the existing bags

Reusable scripts, all in `scripts/`.

| quantity | value | how |
|---|---|---|
| wheel torque scale | `effort` = **0.1 Nm/unit** | two independent fits, corr 0.93 (`wheel_torque_from_bags.py`) |
| per-wheel torque | ≥ 105 Nm sustained, no plateau | a lower bound; nothing ever saturated |
| rolling resistance | 0.09 × weight | the same fit's offset |
| turn gain α | **2.20** measured wheels, 3.06 commanded | `fit_turn_gain.py`, two IMUs agree to 1% |
| drivetrain realisation | 0.60–0.70 of the commanded differential | the ratio of those two |
| actuator response | **τ = 0.17–0.20 s**, dead time 0–50 ms | `fit_actuator_lag.py`, joint fit |
| friction saturation, real terrain | ≤ 0.69 at μ=0.2, ≤ 0.23 at μ=0.6 | `saturation_from_bags.py` |
| L/K | 8–15 (pre-rolling-resistance) | `fit_traction.py`, quasi-static samples only |

**The friction budget never binds on terrain you have driven.** Tilt stays under 5.4° in those
bags; binding needs `tan θ = μ`, i.e. 31° at μ=0.6. That is a lower bound — these are trajectories
the robot drove and survived — and it says nothing about terrain you have not recorded.

**Torque never binds either.** 105 Nm/wheel is 900 N of traction, 0.86 × weight, so friction
saturates first for any μ < 0.86. IMPROVEMENTS.md §2's "you stall before you slip" does not hold
here; it reasoned from a placeholder an order of magnitude too small.

---

## 4. Claims retracted during the session

Recorded because the commit messages carry them but a reader of this report would not otherwise
see them.

- **"The drivetrain overshoots 45%."** It does not. That came from an instantaneous
  measured/commanded ratio on a continuously varying command, which cannot separate lag from
  overshoot. A second-order fit lands at ζ = 0.85–1.00 and does not beat first order.
- **"The force-balance model is refuted (α ≈ 1.02–1.08)."** That was a slip regulariser, not
  physics. The correct rigid-Coulomb answer is exactly 1.000; the model was missing shear
  compliance rather than being wrong.
- **"The command-to-response lag is ~170 ms of pure transport delay, and `tau_motor` is the wrong
  knob."** Both halves wrong. Sequential fitting (delay by cross-correlation, then lag) manufactures
  a dead time out of a slow rise. The step response has no dead zone at all, and the joint fit gives
  tau 0.19 s with under 50 ms of dead time — so `tau_motor`, which already existed, was the right
  knob throughout.
- **"Publishing `/cmd_joints` faster would cut the lag."** It would not — the LLC already holds the
  command. But asking exposed that both fitting scripts reconstructed commands by linear
  interpolation instead of a zero-order hold, inflating every fitted delay by ~50 ms.
- **"A calibration drive is needed to fit the traction model."** It was not: the fit uses measured
  wheel speeds against a gyro, with no command in the loop.
- **α(v) is NOT established.** The trend was fitted on transient-dominated data; the robot holds a
  turn command for a median of 3 ms, so the archive has almost no steady-state turning.
- **The left-wheel response asymmetry is not significant.** 8/11 segments, sign test p = 0.23.

---

## 5. What is safe to enable, and what is not

| change | default | confidence |
|---|---|---|
| the three certificates | **removed** | were analytic and correct; 40% of the rollout, and inert |
| exact-arc integration | **on** | high — exact against a closed form, no parameter |
| `wheel_width = 0.10` | off | high measurement, but changes planning |
| `tau_motor = 0.19` | **on** (planner) | jointly fitted; blend 0.53/step at dt=0.1 |
| `command_delay = 0.04` | on (planner) | residual dead time; 0 steps at dt=0.1 |
| `shear_lk`, `body_momentum` | off | mechanism sound, parameters unresolved |
| `k_turn` | unchanged at 1.0 | measured α says 2.20–2.87; deliberately not changed |

---

## 6. What to do next

1. **Drive the `calibrate` scenario** (added to `ros/record_odin.sh`, which now also records
   `/joint_states` and `/joint_setpoints`). Five minutes settles: whether Odin realises the forward
   channel 1:1 — the soft side of the traction conflict — plus the steady turn gain, whether α
   depends on speed, the delay's step count, and the standing-start prediction.
2. **Decide `k_turn`.** The planner currently expects ~60% more yaw than it gets. One line, but it
   interacts with the delay, so change them together and re-check together.
3. Only then reconsider the traction model. It is better and cheap to justify, but body inertia is
   the larger unmodelled effect and neither is yet the dominant residual.

---

## 7. Before merging

- **`studies/adjoint/harness.py` must be updated.** It hand-launches `step_kernel_bt`
  positionally, which now takes `twist_in` as an extra input and `twist` as an extra output (the
  three certificate outputs are gone again as of §1.1). It fails on arg count, not silently.
- **Merging with `study/adjoint-sensitivity` will conflict.** That branch's `clear_soft` work
  touches the same lines of `step_finalize`, the kernel signatures and `_alloc_rollout_buffers`.
  Mechanical, but certain. Regenerate the golden fixture afterwards — the merged engine is a
  different engine, and `golden.py` picks up `clear_soft` automatically if present.
- **`tests/engine/gradients.py` is broken on `main`** and still is (it launches 15 args at a kernel
  that wanted 17 even before this work). Pre-existing, already repaired on the study branch, left
  alone deliberately.
- The golden fixture is device- and Warp-version-specific: RTX A500, Warp 1.14.0.
