# Planner gating measurements

Step 2 of `PROBABILISTIC_PLANNING_PLAN.md` section 7: the two cheap measurements that the plan
flags as each able to invalidate a design choice in Part A. Run 2026-09-18, RTX A500.
Scripts: `studies/planner/{settle_sensitivity,lattice_timing}.py`; JSON in `studies/out/planner/`.

---

## 6.2 Does the settle's clamping mask the sensitivity? **No.**

The z-margin design (plan section 3.1) rests on the settle's analytic Jacobian carrying map
uncertainty into attitude uncertainty. The worry was that `settle()`'s damped Newton --
`max_step` per iteration, `tilt_clamp` every pass -- clips the response.

Finite difference against the closed form. With wheels at `(0, +b)`, `(0, -b)`, `(-l, 0)` and
small angles, `roll = (e1 - e2)/2b` and `pitch = (e3 - (e1+e2)/2)/l`, so
`d(roll)/de1 = 1/2b = 1.370` and `d(pitch)/de3 = 1/l = 1.333` rad/m.

| perturbation | measured `d(roll)/dh` | measured `d(pitch)/dh` | residual |
|---|---|---|---|
| front-L, 2 mm … 50 mm | **1.370 … 1.372** | −0.667 | ≤3e-7 |
| front-R, 2 mm … 50 mm | **−1.370 … −1.372** | −0.667 | ≤3e-7 |
| rear, 2 mm … 50 mm | 0.000 | **1.333 … 1.334** | ≤3e-7 |

Exact, over a 25× range of perturbation size. The cross term `d(pitch)/de1 = −1/2l = −0.667` is
also exact. Cross-slopes to 30° settle with residuals ≤2e-7, four orders under the `resid_tol`
of 1e-2, and `tilt_clamp` (60°) never comes near binding on a robot that tips at 15°.

**Section 3.1 stands: `J^-1` is the right carrier for sigma.**

## …but the path from a map cell to a support is not a plain max

Sweeping ONE raised cell along the front-left wheel's axis gives a response that is symmetric
about the wheel centre, decays with offset, and reaches zero at |dx| ≈ 0.20 m — not the 0.35 m
a flat max over the wheel footprint would give:

| offset from wheel centre | ±0.02 | ±0.06 | ±0.10 | ±0.14 | ±0.18 | ±0.22 m |
|---|---|---|---|---|---|---|
| support lift from a 5 cm cell | 0.028 | 0.023 | 0.023 | 0.006 | 0.006 | **0.000** m |

Two mechanisms, both real:

- **The envelope is a cylinder dilation, not a flat max.** A bump of height `h` touches a wheel
  of radius `R` only within `sqrt(R² − (R−h)²)` of the centre — 0.18 m for `h` = 5 cm, `R` =
  0.35 m, which is the measured cut-off. The dilation is offset-corrected.
- **The support is then bilinearly sampled** at the wheel centre across four envelope cells, so
  a single raised cell contributes only its bilinear weight. Raising a 5×5 patch instead
  recovers the full 1.370 sensitivity (table above).

**This corrects plan section 3.3.** The maximum the Clark fold addresses is over
*offset-corrected* heights `h_i − c(dx_i)`, not raw cell heights, and its result is then blended
across four envelope cells. A fold written against raw heights would be wrong in both the
support's mean and its sigma. The correction `c` is fixed robot geometry, so it costs nothing —
but it has to be there.

A second consequence worth noting: the cylinder envelope is `wheel_width` = 0.10 m wide
laterally against 0.70 m long. Contests are overwhelmingly LONGITUDINAL. A first attempt at the
contested-contact test placed the competing cells ±0.10 m apart in `y`, outside the 0.10 m
width, and measured exactly nothing.

---

## 6.7 Lattice solve time, and whether two solves fit. **7.9 ms; yes.**

Frame budget 69 ms at 14.5 Hz. Deployed settings: 16 m window, 0.24 m routing cell
(`plan_lat_coarsen` 3), `plan_n_theta` 24.

| config | poses | median | p90 | % frame |
|---|---|---|---|---|
| **deployed (coarsen 3)** | 107,736 | **7.86 ms** | 7.90 | **11.4%** |
| coarsen 4 | 60,000 | 3.53 | 3.86 | 5.1% |
| coarsen 2 | 240,000 | 22.42 | 22.82 | 32.5% |
| optimistic: 12 headings | 53,868 | 3.69 | 3.85 | 5.4% |
| optimistic: coarse + 12 headings | 30,000 | **2.00** | 2.02 | 2.9% |

(The params file records 10.1 ms for the deployed config on another machine; 7.9 here.)

**The two-solve gap of section 4.3 is cheap.** Pessimistic plus a coarse 12-heading optimistic
solve is 9.86 ms, 14.3% of a frame — 2 ms more than the single solve already costs. The
exploration trigger, the decision-relevant target selection, and the automatic diagnosis of an
ignorance-blocked goal all come for about 3% of a frame.

**And replan latency is not the carrot's risk.** At 7.9 ms and 0.5 m/s the robot commits 4 mm
blind. What bounds the carrot is the map update rate (69 ms) and its own tracking error, not the
planner. The concern raised when MPPI was dropped is resolved; plan section 6.6 (carrot and
pivot primitives do not compose) is the one that still matters.

---

## Net effect on the plan

- **6.2 closes.** No clamping, no masking; section 3.1 is sound as written.
- **6.7 closes.** Both the single solve and the two-solve scheme fit comfortably.
- **3.3 needs a correction** before implementation: fold over offset-corrected heights, and
  account for the bilinear blend across four envelope cells.
- Part A is otherwise unblocked.

---

# Part A built: the z-margin field (2026-09-18)

`CostToGo` now optionally measures feasibility in **sigmas** rather than raw thresholds
(`_margin_kernel`, `src/helhest/planning/costtogo.py`). `k_sigma = 0` is exactly the old
behaviour, so nothing changes until a caller opts in.

## What it computes

```
z_roll  = (max_roll - |roll|)        / sigma_roll
z_climb = (max_pitch_up + pitch)     / sigma_pitch      (climb = NEGATIVE pitch)
z_desc  = (max_pitch_down - pitch)   / sigma_pitch
z_clear = (clearance - clear_margin) / sigma_clear
z       = min over tests             -- the binding constraint, in sigmas

blocked |= z < k_sigma
tilt    += margin_weight * max(0, z_ref - z)
```

Dividing each test by its own sigma is what makes the `min` meaningful -- roll is in radians,
clearance in metres, and a raw `min` over those compares nothing.

## Sigma propagation, and an independent check on it

From the settle's closed-form rows: `roll = (e1 - e2)/2b`, `pitch = (e3 - (e1+e2)/2)/l`, both
verified exactly by finite difference in section 6.2 above. Both are **differences** of
supports, so the pose drift shared across cells cancels and what enters is the belief's
**measurement** sd, not its total. That is why `compute(sigma=...)` wants `meas_sd`.

| per-cell sigma | predicted sigma_roll | predicted sigma_pitch |
|---|---|---|
| 1.0 cm | 1.11 deg | 0.94 deg |
| **2.5 cm** | **2.77 deg** | **2.34 deg** |
| 5.0 cm | 5.55 deg | 4.68 deg |

`studies/calib/RESULTS.md` section 1 measured the attitude residual **end to end** -- settling
on a real accumulated map and comparing against SLAM -- at **2.84 deg roll, 2.11 deg pitch**. A
per-cell sigma of 2.5 cm reproduces 2.77 / 2.34 from geometry alone, within 3% on roll and 11%
on pitch. The two measurements share no machinery, so this is corroboration of the chain rather
than a restatement of it.

## Operating points

61x61 routing grid at 0.24 m, 12 headings, rolling terrain, uniform sigma:

| `k_sigma` | per-cell sigma | blocked | window reachable | z p50 |
|---|---|---|---|---|
| 0.0 | 2.5 cm | 0.0% | 96.7% | — |
| 2.0 | 0.5 cm | 9.8% | 84.7% | 4.0 |
| 2.0 | 1.0 cm | 9.8% | 84.7% | 4.0 |
| **2.0** | **2.5 cm** | **17.3%** | **76.0%** | **3.2** |
| 3.0 | 2.5 cm | 44.8% | 7.8% | 3.2 |
| 2.0 | 5.0 cm | 72.0% | 1.1% | 1.6 |

At the measured map quality, `k_sigma = 2` costs about 17% of poses and leaves three quarters
of the window reachable. The 0.5 and 1.0 cm rows are identical because `sigma_floor_m` (0.02 m)
dominates both -- the floor doing its job.

The last row is a warning worth stating plainly: **at a per-cell sigma of 5 cm, demanding two
sigmas leaves almost nothing reachable.** That is not the field misbehaving, it is the honest
consequence of a map that cannot resolve roll to better than 5.5 deg against a 15 deg envelope.
It is also exactly the situation the plan's optimistic/pessimistic gap (section 4.3) is meant to
detect and act on rather than sit in.

## Two approximations, both marked in the source

- **Sigma is sampled at each wheel centre**, not at the cell that won the envelope dilation.
  Elevation sigma varies smoothly with observation range (~3 cm/m measured), so over the
  <=0.35 m to the contact cell this is worth ~1 cm. The terrain max it stands in for is not
  smooth; sigma is.
- **The footprint maximum is not folded.** Reading sigma off one cell is the linearized, one-hot
  estimate, which overstates the sd at contested contacts. The Clark fold at the dilation stage
  is the fix, and per section 6.2 above it must fold **offset-corrected** heights, not raw ones.

`sigma_clear` also drops the (negative, helpful) cross term between the belly and its supports,
which overstates it -- the conservative direction.

Tests: `tests/planning/test_zmargin.py`, 7 cases.

---

# Part B built: the doubt field and the two-solve gap (2026-09-19)

Plan sections 4.2 and 4.3. `CostToGo` gains a `doubt` field, `solve_gap`, `gap_at` and
`doubt_targets`. Tests: `tests/planning/test_gap.py`, 9 cases.

## Doubt: ignorance is not the same as bad ground

Each pose is scored twice in one kernel pass — once against the believed map, once as if every
cell sat at `sigma_floor_m`:

| condition | meaning |
|---|---|
| `z_opt < k` | genuinely bad terrain; looking at it will not help |
| `z_opt >= k` but `z < k` | **blocked by ignorance** — worth going to look |

`doubt = z_opt − z` on the second, zero otherwise. One extra scalar per pose, no extra launch.

This is the distinction between purposeful exploration and wandering toward whatever is least
observed. On rolling terrain at a per-cell sigma of 5 cm, **99.5% of blocked poses turn out to
be blocked by ignorance rather than terrain** — the map, not the ground, is what stops the
robot.

## The gap: what ignorance costs, in metres

`solve_gap` solves twice and keeps both value functions; `gap_at` reads
`V_pessimistic − V_optimistic` at the robot's own pose. Uniform sigma, robot at one end of a
16 m window, goal at the other:

| per-cell sigma | V_pess | V_opt | gap | reachable (pess) | ignorance-blocked |
|---|---|---|---|---|---|
| 0.5 cm | 16.06 | 16.06 | **0.00** | yes | no |
| 2.0 cm | 16.06 | 16.06 | **0.00** | yes | no |
| 4.0 cm | capped | 16.06 | — | **no** | **yes** |
| 10 cm | capped | 16.06 | — | **no** | **yes** |

Below the floor the two solves coincide and the trigger stays silent — correct, since there is
nothing to gain by looking. Past 4 cm the goal is unreachable on the believed map and reachable
on a certain one, which is **the blind-cell failure diagnosing itself**: not bad terrain, not
knowing, with a defined response (go look, or relax `k_sigma`) instead of a planner that
reports no route.

Worth noting that `gap` and `doubt` are complementary, not redundant. At 0.5 cm the gap is zero
while 9.6% of poses still carry doubt — those are ignorance-blocked but not on the route, so
resolving them buys nothing. Only the gap knows that.

## Targets: where to look

`doubt_targets` follows the OPTIMISTIC policy greedily from the robot and collects the doubtful
poses along it — the cells whose resolution would unlock the better route. On a frontier-shaped
map (0.01 m behind the robot, 0.06 m ahead), every target returned lies in the uncertain half,
ranked by doubt. That is the cheap form of `SENSITIVITY_PLAN.md`'s C4: a policy rollout rather
than an adjoint.

## Timing, and a correction to section 6.7

Deployed shape (67×67 at 0.24 m, 24 headings), 69 ms frame:

| | time | % frame |
|---|---|---|
| single solve | 3.18 ms | 4.6% |
| **`solve_gap` (both)** | **11.58 ms** | **16.8%** |
| `doubt_targets` (host walk) | 1.06 ms | 1.5% |

**Section 6.7 estimated the two-solve scheme at about 2× the single solve. Measured, it is
3.6×.** The optimistic solve is the more expensive of the two: blocking fewer poses leaves a
larger reachable set, so the value iteration needs more sweeps to converge. The conclusion is
unchanged — 16.8% of a frame is affordable — but the estimate was low, and a coarser optimistic
solve is worth more than section 6.7 suggested, not less.

## What is left in the plan

- **4.1 frontier seeding** — zero `V` at frontier poses instead of at a goal. Not built.
- **3.3 the Clark fold** at the dilation stage, over offset-corrected heights.
- **6.6 carrot and pivot primitives do not compose** — untouched, and the only remaining item
  that is a design hole rather than an upgrade.
- Nothing has yet run on the robot.
