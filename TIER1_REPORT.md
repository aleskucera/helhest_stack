# Tier-1 engine certificates — session report

Branch `engine/tier1-certificates`, seven commits off `main` (`de96100`). Implements
IMPROVEMENTS.md §3, §1, §2, §4 in that order. Not merged; left for review.

Everything here is additive: new output arrays and two new opt-in `RobotParams` fields. With
default parameters every pre-existing simulator output is bit-identical to the pre-change
engine, proven at every commit by `tests/engine/golden.py`.

## Commits

| commit | item |
|---|---|
| `1256126` | golden bit-identity fixture (written FIRST, before any engine change) |
| `b195e61` | §3 tip-over margin |
| `a1ef147` | §1 friction saturation certificate |
| `8b73a5a` | §2 motor torque / stall certificate |
| `6070747` | §4 yaw-binned cylinder wheel envelope, behind `wheel_width` |
| `1297d27` | this report |
| `1b8b0b4` | `wheel_width` = 0.10 m from the Ostrich model's measured column; tests use it |

## File by file

**`src/helhest/engine/step.py`** — the physics.
- `stability_margin(robot, loads)` — `min_i(N_i) / (m g)`.
- `contact_grip(...)` — `Sum_i mu_i N_i` at a pose, mirroring the turning solve's sampling.
- `friction_saturation(...)` — friction-ellipse demand over the Coulomb budget.
- `torque_saturation(robot, pitch)` — required drive torque over `motor_torque_limit`.
- `body_twist(robot, om, alpha)` — extracted so the wheel-speed → `(v, psi_dot)` mapping has one
  home; `step_predict` and the certificates now share it. Bit-identical (golden).
- `yaw_bin(yaw, n_yaw)` — envelope-stack slice index; constant 0 for the spherical default.
- `step_predict` now returns `vec4 (x, y, yaw, alpha)`: `step_finalize` needs alpha to rebuild the
  twist, and re-reading the grad-tracked `turn_out` inside the kernel would have put a
  read-after-write on a taped buffer.
- `step_finalize` takes `fric_i`, `om`, `alpha`; writes the three certificates.
- `step_kernel`, `step_kernel_bt`, `rollout_kernel` each gained three output arrays, **appended at
  the end** so an out-of-tree hand-launch fails loudly on arg count rather than silently
  rebinding. `rollout_kernel`'s `envelope` is now `array3d` (the yaw stack).

**`src/helhest/engine/robot.py`** — two new `RobotParams` fields, both defaulting to the current
behaviour: `wheel_width: float | None = None` and `motor_torque_limit: float = inf`
(`Robot` carries the latter to the device). `motor_torque_limit` carries a `TODO(hardware)`;
`wheel_width` carries the measured 0.10 m and its source.

**`src/helhest/engine/envelope.py`** — `cylinder_offset_table(cell_size, wheel_radius, half_width,
yaw)` (rotated rectangle, cap from the along-travel offset only, own search radius so the corners
are not clipped) and `_contact_table_kernel` (the disk contact, for an arbitrary host-built
element). The spherical path still runs the original `_contact_kernel` untouched — the table
version computes its cap on the host in float64, which differs from the device's float32 by an
ULP, and that would have broken bit-identity.

**`src/helhest/engine/simulator.py`** — `stability`, `saturation`, `stall` buffers on both
simulators; `ForwardSimulator.envelope_stack` `[n_yaw, ny, nx]` with `envelope` kept as its 2D
slice-0 view (so existing readers, including `studies/adjoint/harness.py`, see the same object);
`set_terrain` dilates once per yaw bin when `wheel_width` is set; `YAW_BINS = 32`;
`DifferentiableSimulator` raises `NotImplementedError` for `wheel_width`.

**`tests/engine/golden.py` + `golden_fixture.npz`** — the non-interference proof. One
`ForwardSimulator` rollout on CPU and CUDA plus one `DifferentiableSimulator` forward+backward,
B=8, T=16, seeded terrain/friction/controls; 26 arrays compared exactly, except the two gradient
arrays which get a tolerance (the envelope adjoint scatters with atomics; measured run-to-run
spread 1.9e-9). `--write` regenerates.

**`tests/engine/certificates.py`** (new) — `selftest_ramp_margin`, `selftest_shape_margin`,
`selftest_friction_saturation`, `selftest_torque_stall`.

**`tests/engine/cylinder.py`** (new) — `selftest_transverse_ridge`, `selftest_lateral_ridge`,
`selftest_yaw_binning`.

**`tests/engine/step.py`** — updated for the new kernel signatures (buffers + the one-slice
envelope stack). `selftest_rollout_kernel` still reports fused == per-step at exactly 0.

## Verification, as measured

Analytic crossings (`python -m tests.engine.certificates`):

| check | result |
|---|---|
| `sum_i N_i / (m g) = 1 / (cos p cos r)` on tilted planes | worst dev 1.2e-6 |
| `saturation = tan(theta) / mu`, 10/20/30 deg, along and across slope | worst rel err 1.2e-6 |
| saturation at `mu = tan(theta)` | 1.0000 |
| centripetal `saturation = v psi_dot / (mu g)` on flat ground | worst rel err 2.9e-7 |
| stall crossing at 40 Nm | 19.21 deg, rel err 7e-7, identical at mu = 0.2 / 0.6 / 0.9 |
| stall at the default `motor_torque_limit = inf` | exactly 0 at every grade |

Cylinder (`python -m tests.engine.cylinder`):

| world | result |
|---|---|
| ridge across the path | envelopes agree to 3.0e-8 (one ULP), settled pose identical |
| ridge 0.30 m beside the left wheel | sphere rolls **7.57 deg**, cylinder **0.00e+00** |
| — sphere pose vs one-wheel-lifted hand geometry | roll/pitch/z match to <1% |
| — sphere lift vs continuous spherical cap | 0.0959 m vs 0.1215 m (21% low; the dilation can only reach the ridge at whole-cell offsets and dcap/dgap = -1.7 m/m here) |
| same ridge head-on (yaw 90 deg, bin 8/32) | sphere and cylinder identical, both -16.745 deg |

Cost, B=4096, T=25, 241x441 grid, RTX A500: rollout 1.144 → 1.164 ms (+1.7%) with the cylinder;
the per-frame dilation goes 0.228 → 2.250 ms for 32 slices. Memory 32 x the envelope grid.
Graph capture verified to record and replay with `wheel_width` both unset and set.

## Findings

**1. `min N_i` is not a tip-over test on this robot, and it never agrees with `max_roll`.**
This is the §3 cross-check, and it disagrees far more strongly than IMPROVEMENTS.md anticipated.
`normal_loads` balances vertical force and horizontal torque using the contact NORMALS only — the
tangential friction reaction that actually holds the robot on a slope, and the overturning moment
it exerts about the CoM at wheel-radius height, are absent from the equations. Two identities
follow on any uniform plane, and both hold to 1e-6:

```
sum_i N_i / (m g) = 1 / (cos pitch cos roll)
min_i N_i / (m g) = (|com_x| / rear_offset) / (cos pitch cos roll) = 0.2637 / (cp cr)
```

So the margin is the CoM's body-frame barycentric weight divided by a cosine: it **rises** with
tilt. It reads 0.264 flat, 0.273 exactly where the `max_roll = 15 deg` gate fires, 0.303 at the
29.5 deg front-axle tip angle and 0.410 at 50 deg of bank. Terrain shape does move it — that is
the only thing it responds to — but weakly: over 12 shape worlds (rocks and 0.6 m spikes under
each wheel, crests, valleys, saddles, roofs, tilts to 57 deg) it stayed inside [0.23, 0.33]. It
never crossed 0 in anything tried.

Consequence: **the margin cannot replace or validate the `max_roll` gate — only the gate protects
against slope tip-over**, and a planner consuming the margin as a stability cost would be reading
a load-transfer diagnostic, not a tip-over one. This also sharpens the study-branch note that
`min N_i` "never fell below 0.24 anywhere in the scene": that was not an accident of the scene, it
is structural. Fixing it means adding the tangential reaction moment to `normal_loads`, which
changes an existing output and so was out of scope here.

**2. The friction budget cannot be taken from `normal_loads` directly.** Same root cause. Using
`Sum_i mu_i N_i` as the budget puts the saturation crossing at `sin(theta) cos(theta) = mu`, which
peaks at 0.5 — the certificate would be structurally unable to fire on any terrain with mu > 0.5.
The implementation takes only the load RATIOS from the solve (where the bias cancels) and rebuilds
the budget from `m g cos(pitch) cos(roll)`. That is what makes the tan(theta) = mu crossing exact.

**3. The torque envelope is recoverable from the bags, and it says friction always binds first.**
`/joint_states.effort` is populated (three wheels, raw units). Newton along the body x axis pins
the scale: the accelerometer's specific force already contains gravity, so
`sum_i tau_i / R = m a_x` holds on grades as well as the flat, and a second estimator using
wheel-odometry acceleration touches no accelerometer at all. On the two post-fix bags
(`out_experiment_goal_unreachable0/1`) the two agree:

| bag | IMU fit | ODOM fit | corr |
|---|---|---|---|
| goal_unreachable0 | 9.73 raw/Nm | 10.60 raw/Nm | +0.93 / +0.90 |
| goal_unreachable1 | 9.67 raw/Nm | 10.38 raw/Nm | +0.93 / +0.89 |

So **effort is deci-newton-metres: 1 raw = 0.1 Nm at the wheel**, within ~10%. Two independent
consistency checks pass: the fit offset is 36-38 Nm of constant resistance, i.e. a rolling
coefficient of 0.09 on this robot's weight, and the implied accelerations match the odometry.

At that scale the front wheels hold **111-118 Nm for a full second** and peak at 130-136 Nm, with
no plateau (0.2-0.5% of samples within 5% of the peak), so this is a **lower bound**, not the
envelope. It is nevertheless enough to decide the question §2 was asked to settle:

| tau/wheel | traction | binds before friction for |
|---|---|---|
| 40 Nm (the old placeholder) | 343 N | mu < 0.33 |
| 105 Nm (measured bound) | 900 N | mu < 0.86 |

**Friction saturates before torque for any mu below 0.86** — so on this robot §1 does the work and
§2 is inert on realistic terrain. IMPROVEMENTS.md §2's "you stall before you slip on high-mu rough
terrain" is not true here; it was reasoning from a placeholder an order of magnitude too small.
Two approximations in `torque_saturation` are optimistic and were checked against this margin:
rolling resistance is excluded (~37 Nm total, measured) and the demand is split equally three ways
while the bags put ~2.5x more through each front wheel than the rear. Neither closes a 0.86-vs-0.6
gap.

Caveats worth keeping: bags before 2026-07-14 give a NEGATIVE correlation (the IMU was remounted)
and bags before 2026-07-27 have the `/cmd_joints` units bug, which corrupts the odometry
estimator specifically. The script prints both correlations so a bad era is obvious.

## Open hardware numbers

Checked against `~/projects/ostrich/examples/helhest_junior/robot_parameters.md`, whose provenance
table separates ruler-measured numbers from tuned ones. One of the three is answered there.

1. **Wheel width — ANSWERED: 0.10 m, ruler-measured** (§6 of that document; collision shape
   `cylinder r = 0.35, half-height 0.05`). So the half-width is **0.05 m**, half of what
   IMPROVEMENTS.md §4 assumed: the spherical envelope over-reaches sideways by **7x**, not 3.5x.
   `tests/engine/cylinder.py` now uses the measured value. `RobotParams.wheel_width` is still
   `None` by default — switching it changes planning behaviour, which is your call, not a
   side effect of this branch.
2. **Per-wheel torque — ANSWERED FROM THE BAGS as a lower bound: >= 105 Nm.** The Ostrich model
   has no torque limit (`TARGET_KE = 150 / TARGET_KD = 0` are fine-tuned servo gains, "not a
   datasheet motor constant"), but `/joint_states.effort` is populated on the real robot. See the
   finding below and `scripts/wheel_torque_from_bags.py`. `RobotParams.motor_torque_limit` now
   defaults to 105.0 instead of `inf`.
3. **`omega_max` — no hardware figure, but this repo's own bag calibration implies ~5.3 rad/s**
   (the drivetrain ceiling behind the turn-differential work; `plan_wmax` is set to 4 to stay
   under it). The Ostrich side only has a keyboard ramp (10 rad/s^2 accel toward ~5 rad/s), which
   is a UI limit, not hardware. Taking 5.3 rad/s: spin-in-place gives
   `psi_dot = R w / (half_track alpha) = 2.3 rad/s` at mu = 0.6, i.e. **13.2 deg of heading swept
   per 0.1 s step against 11.25 deg bins**. So §7's coupling is real at full spin — and note more
   bins do NOT fix it: the issue is that one step samples a single envelope snapshot while the
   robot sweeps through headings, which only a finer `dt` addresses. At the practical
   `plan_wmax = 4` it is 10 deg/step, just inside a bin.

## Things you should know before merging

- **`studies/adjoint/harness.py` will need three extra output arrays** when the research line
  reruns: it hand-launches `step_kernel_bt` with a positional `outputs=[...]` list, and the kernel
  now takes `stability`, `saturation`, `stall` appended at the end. It will fail loudly with an
  arg-count error, not silently. Nothing else in the repo launches these kernels directly.
- **Merging with `study/adjoint-sensitivity` will conflict.** That branch's `clear_soft` work
  touches the same lines of `step_finalize`, the three kernel signatures and
  `_alloc_rollout_buffers`. The conflicts are mechanical (both sides append), but they are certain.
  `tests/engine/golden.py` already picks up `clear_soft` automatically if the attribute exists —
  regenerate the fixture after the merge, since the merged engine is a different engine.
- **`tests/engine/gradients.py` is broken on `main`** and still is here: it launches `step_kernel`
  with 15 arguments (it needed 17 before this branch, 20 after). Pre-existing, and already
  repaired on the study branch; left alone deliberately.
- **Nothing is wired into the MPPI cost.** The certificates are computed and exposed only.
- **The cylinder envelope has a hard lateral edge.** Its underside is a straight line across the
  tread, so an obstacle inside the width lifts by its full height with no cap taper: the envelope
  steps 0 -> obstacle height across ONE cell at the tread edge, where the sphere's cap tapered
  smoothly over 0.35 m. Measured in `selftest_lateral_ridge` (a 0.30 m ridge under the tread reads
  0.175 m at the wheel centre, mid-ramp). Lateral behaviour is therefore cell-resolution-limited
  and non-smooth — a further reason not to hand this to the differentiable path unexamined.
- The golden fixture is device- and Warp-version-specific (float32 CUDA arithmetic is not portable
  across architectures). It was generated on the RTX A500 with Warp 1.14.0.
