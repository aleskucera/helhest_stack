# Engine work — session report

Two stacked branches off `main` (`de96100`). Neither is merged.

```
main
 └── engine/tier1-certificates          IMPROVEMENTS.md Tier 1, complete and verified
      └── engine/exact-arc-integration  engine fidelity work + the measurement scripts
```

Everything is additive or opt-in, with one stated exception (§2).

---

## 1. Tier 1 — complete

The four commissioned items, in the order IMPROVEMENTS.md recommends. Each verified against a
closed-form answer rather than eyeballed. `tests/engine/golden.py` pins every simulator output for
a seeded batch and passed at every commit on this branch.

| item | state |
|---|---|
| §3 tip-over margin | exposed as `sim.stability` |
| §1 friction saturation | exposed as `sim.saturation`; crosses 1.0 at `tan θ = μ` to 1.2e-6 |
| §2 torque / stall | exposed as `sim.stall`; boundary independent of μ to 7e-7 |
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

### Command transport delay

`SolverParams.command_delay`, defaulting to the measured `dynamics.COMMAND_DELAY = 0.17 s`, with
`elevation_node` feeding `command_history` from the commands it actually issued.

Confirmed physical rather than a timing artifact, three ways: every topic's header stamp sits
within 2 ms of its bag log time; the reported wheel velocity tracks encoder position to 10–20 ms
(correlation 1.000, scale 0.996); and the IMU — a separate device and driver — sees the same lag on
yaw. Localised with `/joint_setpoints`: **10 ms** from `/cmd_joints` to LLC intake, **140–189 ms**
inside the velocity loop. So it is the motor controller, not comms, and may be tunable there.

Quantisation is a visible approximation: 0.17 s at dt = 0.1 rounds to 2 steps = 200 ms.

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
| command delay | 149–199 ms | `fit_actuator_lag.py` |
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
| the three certificates | on (inert) | high — analytic, nothing consumes them |
| exact-arc integration | **on** | high — exact against a closed form, no parameter |
| `wheel_width = 0.10` | off | high measurement, but changes planning |
| `command_delay = 0.17` | **on** (planner) | measured three ways; quantises to 200 ms |
| `motor_torque_limit = 105` | on | a lower bound; inert either way |
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

- **`studies/adjoint/harness.py` must be updated.** It hand-launches `step_kernel_bt` positionally,
  which now takes `twist_in` as an extra input and `stability`, `saturation`, `stall`, `twist` as
  extra outputs. It fails on arg count, not silently.
- **Merging with `study/adjoint-sensitivity` will conflict.** That branch's `clear_soft` work
  touches the same lines of `step_finalize`, the kernel signatures and `_alloc_rollout_buffers`.
  Mechanical, but certain. Regenerate the golden fixture afterwards — the merged engine is a
  different engine, and `golden.py` picks up `clear_soft` automatically if present.
- **`tests/engine/gradients.py` is broken on `main`** and still is (it launches 15 args at a kernel
  that wanted 17 even before this work). Pre-existing, already repaired on the study branch, left
  alone deliberately.
- The golden fixture is device- and Warp-version-specific: RTX A500, Warp 1.14.0.
