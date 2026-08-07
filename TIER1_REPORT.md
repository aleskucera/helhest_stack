# Tier-1 engine certificates — session report

Branch `engine/tier1-certificates`, five commits off `main` (`de96100`). Implements
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
(`Robot` carries the latter to the device). Both carry a `TODO(hardware)`.

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
Graph capture verified to record and replay with both `wheel_width=None` and `0.2`.

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

**3. Which limit binds first is a real question and now measurable.** With mu = 0.6 the friction
certificate fires at 31.0 deg of grade; with a 40 Nm placeholder the stall certificate fires at
19.2 deg. On high-mu ground you stall before you slip, as IMPROVEMENTS.md §2 expected — but the
crossover sits exactly on the unmeasured torque number.

## Open hardware numbers, still needed from you

1. **`omega_max`** (IMPROVEMENTS.md open question 1) — decides whether §4 forces a finer `dt`. With
   32 bins and dt = 0.1 s the cylinder envelope is safe up to `psi_dot ~ 2 rad/s`, which
   spin-in-place reaches at roughly `omega_max = 4.5 rad/s`. Above that the rollout aliases across
   yaw bins and §4 and §7 have to land together. Documented on `yaw_bin`, not solved.
2. **Per-wheel motor torque envelope [Nm]** (open question 2) — `RobotParams.motor_torque_limit`,
   currently `inf`, which makes the stall certificate inert. Holding a 15 deg grade needs ~31.5 Nm
   per wheel here, so the answer is likely in the 20–60 Nm range where it changes planner
   behaviour.
3. **Real wheel width [m]** (open question 4) — `RobotParams.wheel_width`, currently `None`. The
   tests use 0.2 m (a 0.1 m half-width) as an assumption. This one has the largest behavioural
   effect of the three: at 0.2 m the robot stops being lifted by anything more than 0.1 m off its
   wheel track.

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
- The golden fixture is device- and Warp-version-specific (float32 CUDA arithmetic is not portable
  across architectures). It was generated on the RTX A500 with Warp 1.14.0.
