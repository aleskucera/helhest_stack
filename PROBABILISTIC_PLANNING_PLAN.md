# Probabilistic planning + exploration — plan

Written 2026-09-16, from a design discussion. Written to be picked up cold.

Target stack (post-Odin): Odin owns SLAM pose, a new CUDA mapping node owns the
probabilistic elevation map, `helhest.planning` owns routing, a simple carrot
controller replaces MPPI. This document covers the **planner** half: how a
probabilistic map enters the lattice, and how exploration falls out of the same
machinery.

Section 6 is the list of things to settle first, and it is the most important section.

**Status 2026-09-17.** Step 1 of section 7 (the sigma calibration harness) is BUILT and RUN --
`studies/calib/`, results in `studies/calib/RESULTS.md`, numbers in `studies/out/calib/*.json`.
It measured three things that revise this document: the map error is *not* drift-dominated
(section 2(b) below is wrong for this sensor), the deployed per-cell `max` reduction costs ~35%
of attitude accuracy, and self-occlusion is not a blocker (section 6.5 closes). Corrections are
inline below, each tagged MEASURED. Everything else is still unimplemented.

Scope of that run: it measures the **residual** of a mean-map plus deterministic-dilation
pipeline, which is ground truth for what the z-margin divides by. It does NOT yet check a
*predicted* sigma, because the Clark fold of section 3.3 is not built — the envelope is still a
morphological max of means, carrying no Jensen inflation and no `sigma_support`. Predicted
against measured is the calibration that section 6.1 ultimately asks for, and it lands with
step 5.

---

## 1. Background: how the lattice planner works today

Four layers, all in `src/helhest/planning/`.

**State.** A pose `(row, col, heading)` — a 3-D array `[ny, nx, n_theta]`. Heading
is in the state for two reasons: a min turn radius (0.5 m) means where you can go
depends on where you point, and the tilt envelope is asymmetric, so the same
hillside is a safe *pitch* head-on and a tipping *roll* sideways.

**Moves.** A fixed menu of motion primitives (`_build_primitives`,
`lattice_solver.py:128`): five equal-length arcs (curvature capped by the turn
radius) plus two point-turns for the skid-steer. Each stores its endpoint cell,
end heading, arc length, and **every cell it sweeps through** — so a thin wall
cannot be jumped.

**Value function.** `V[r,c,t]` = cost of the cheapest route from that pose to the
goal. Seeded `0` at the goal, `+inf` elsewhere, then relaxed:

    V_new[s] = min( V_old[s], min over primitives p of  cost(p) + V_old[next(s,p)] )

skipping primitives whose swept cells are not all free. A wavefront spreading
backwards from the goal. Every pose updates independently from the previous
sweep, so it is one GPU thread per pose, no priority queue, and the convergence
loop stays on-device (`capture_while`).

**Policy.** The robot never extracts a path. It looks up its own pose, tries each
primitive, and takes the one minimising `cost + V`. That is what makes a simple
carrot viable: the controller queries a field from wherever the robot actually
is, instead of re-attaching to a committed path.

**Terrain interface — exactly two fields per pose**, produced by `costtogo.py`:

| field | role |
|---|---|
| `blocked[r,c,t]` | hard veto |
| `tilt[r,c,t]`    | graded cost |

with `arc cost = arc_length * (1 + tilt_weight * mean tilt over swept cells)`.
`lattice_solver.py` is terrain-agnostic — swap those two tables and it plans on
something else.

**Where those come from — the settle.** Not a thresholded slope map: the robot is
placed on the terrain at every pose and the engine solves where it comes to rest
(`engine/step.py:226`, analytic 3x3 Newton). Out come body height, pitch, roll,
solver residual and belly clearance, for ~1.7 M poses in one kernel with no
readback. Then

    blocked = |roll| > max_roll OR pitch outside climb/descend limits
              OR residual > resid_tol OR clearance < clear_margin
    tilt    = w_roll*|roll| + w_pitch*|pitch|

Direction-awareness is free, because the same hillside yields pitch at one
heading and roll at another.

---

## 2. Two structural facts that shape everything below

**(a) The settle is translation-invariant in z.** Raise the whole map by a
constant and the body rises by the same constant: pitch, roll and clearance are
all unchanged. Consequence: **`h + k*sigma` is a no-op.** Inflating the map is
not a pessimism strategy — it only does anything where sigma varies spatially,
and then it is an arbitrary tilt toward the less-observed side, not a worst case.
Uncertainty must be *propagated through* the settle, never baked into its input.

**(b) Attitude and clearance are differences, so common-mode map error cancels.**
The pitch and roll rows of the settle's inverse Jacobian sum to zero; the
clearance rows sum to one against the belly cell. Either way a patch-wide offset
drops out. Pose drift — the dominant error source — is common-mode over a
footprint, so what actually threatens the robot is the *differential* error over
~0.5 m, not absolute map accuracy. That is a much easier property to ask of a
mapper, and it should be stated as a mapper requirement.

> **MEASURED 2026-09-17 — the premise is false for this sensor.** This section assumed the
> error is pose-drift dominated, hence common-mode, hence largely cancelled. The measured
> spatial correlation of the per-sweep residual is 0.30 at one 0.08 m cell and **0.04 beyond
> 0.5 m**: the variance is dominated by INDEPENDENT per-cell noise, not by a patch-wide
> offset. The cancellation does not happen. That is directly why the map cannot resolve roll
> at all today — predicted roll correlates with measured at +0.00, with 2.8 deg of spread
> against a 0.36 deg real signal (`studies/calib/RESULTS.md`, sections 1 and 4). The algebra
> above is still right; what is wrong is the claim that the error is the kind that cancels.


---

## 3. Part A — probabilistic feasibility: the z-margin field

Replace each pass/fail test with **how much room is left, in units of its own
uncertainty**:

    z_roll  = (max_roll - |roll|)          / sigma_roll
    z_climb = (max_pitch_up + pitch)       / sigma_pitch
    z_desc  = (max_pitch_down - pitch)     / sigma_pitch
    z_clear = (E[clearance] - clear_margin)/ sigma_clear

    z[r,c,t] = min over tests          # the binding constraint

then

    blocked = z < k
    tilt   += charge_per_sigma * max(0, z_charge - z)      # hinged graded penalty

`z` is a safety margin in sigmas: "this pose is 3.4 sigma from tipping."

Three properties worth naming:

- **One dial, not two.** The pessimism knob and the "penalty for being near a bad
  state" are the same quantity. `k = 2` means "I require two sigma of room on
  every test."
- **Units become comparable.** Roll is radians, clearance is metres; you cannot
  `min` over those. Dividing each by its own sigma makes the `min` meaningful.
- **Both ends degrade correctly.** Well-mapped ground: sigma small, `z` large,
  behaves as today. Frontier: sigma large, `z` small, the robot is cautious with
  no special-casing.

**Required detail: floor the denominator.** `sigma_eff = max(sigma, sigma_floor)`.
Without it, a perfectly-known map makes a 14.9-degree roll against a 15-degree
limit read as infinitely safe. `sigma_floor` is the irreducible error —
localisation, controller tracking, model error — which never goes to zero.

`residual` has no natural sigma (it is a solver diagnostic, not a belief), and
neither does the step-gate. Both stay as hard vetoes beside `z`.

### 3.1 Computing sigma_pitch / sigma_roll — cheaper than expected

`settle()` already forms the analytic Jacobian `J[i,:] = dc_i/d(z,pitch,roll)`
every Newton iteration, including the terrain-gradient terms `(gx, gy)`. Since
`c_i = wheel_center_z - height_i - r_wheel`, we have `dc_i/dheight_i = -1`, so at
the converged root

    d(z, pitch, roll) / d(support heights) = J^-1

which is exactly the affine map sigma has to travel through. **One extra
`solve3` per pose against the support-sigma vector.** No new math, no new
derivation, and it is the *true* local sensitivity rather than flat-ground
geometry, because `J` carries the terrain slope.

### 3.2 Computing sigma_clear

Clearance is `min over belly points i of (w_i.z - h(w_i.xy))`
(`chassis_clearance`, `engine/step.py:574`) — the smallest gap between the
chassis underside and the ground beneath it. Writing it against the three wheel
supports `e` and the belly cell `h_b`:

    c_i = u_i^T e - h_b            (u_i constant per belly point, sums to 1)

    Var[c_i] = u_i^T Sigma_e u_i + Var[h_b] - 2 u_i^T Cov(e, h_b)
               |__ body ______|   |_ ground _|  |___ cross term ___|

**The cross term is not optional.** Check the limits: perfectly correlated map
gives `sigma^2 + sigma^2 - 2 sigma^2 = 0` (correct — if everything is wrong by the
same amount, the gap is known exactly); independent cells give
`sigma^2 (sum u^2 + 1)` (correct — the errors add). Dropping the cross term always
returns the second answer, so with drift-dominated error it would hugely
overestimate `sigma_clear` and the robot would refuse ground it handles.

> **MEASURED 2026-09-17 — inverted.** The warning above assumed drift-dominated error would
> make the independent answer hugely pessimistic. At the ~0.5 m wheel-to-belly separation the
> measured correlation is **0.04**, so the independent answer is very nearly right and the
> cross term is a small correction. Keep it for correctness; it is not load-bearing on this
> record. Re-check outdoors and on the ICP pose path before assuming it stays small.


`Cov(e, h_b) = sigma_e sigma_b rho(d)` and the distances are fixed robot geometry,
so with **one spatial correlation length `L`** from the mapper every `rho` is a
compile-time constant.

The outer `min` over belly points is a minimum of correlated Gaussians, and
`min(A,B) = -max(-A,-B)` — so **the Clark fold applies directly, negated**, with
the matching consequence that `E[min] < min of means`: high-centring is more
likely than the mean map suggests, and the fold says by how much.

### 3.3 Where the Clark fold goes

The wheel supports read the **dilated envelope**, i.e. a max over the wheel
cylinder footprint, already computed as a map layer. So the maximum the fold
addresses happens at the *dilation* stage, not inside the settle (which only
bilinearly samples the result). The fold therefore belongs in the dilation
kernel, producing `envelope_mean` **and** `envelope_sigma` layers.

This is the fold's entire role in the runtime stack: a correct `sigma_support`,
where reading sigma off the argmax cell is ~29% wrong at contested contacts
(Clark paper, Sec. III-H). Per-pose, so the frozen-rollout assumption holds by
construction; no cross-arc covariance, and no path-level calibration law needed,
because no path-level standard deviation is being claimed.

Note the belly clearance samples **raw** elevation while the wheels sample the
dilated envelope — so two distinct sigma layers are needed, and the cross term in
3.2 spans both.

### 3.4 Unobserved cells

`sigma_unobserved = sigma_max`, a finite cap. The same `k*sigma` machinery then
makes unobserved terrain expensive-but-passable rather than a wall — RAMP's
"penetrable at a penalty" falling out of the existing mechanism instead of a
special case. This retires the constant blind-fill and its phantom plateau
(see `blind_fill_goal_unreachable`), and `measured` becomes redundant with
`sigma < sigma_max`.

### 3.5 What does NOT change

`lattice_solver.py`. Not one line. All of the above lands in the field-build
stage, and by the time the solver runs, uncertainty has been converted into the
same two plain tables it reads today.

**Rejected, with reasons** (do not relitigate without new evidence):

- *Mean-and-variance propagated through the Bellman update.* Coherent risk is
  time-inconsistent, so it is not a valid DP. The one DP-able form (entropic /
  exponential utility) separates exactly into `mu_i + (rho/2) sigma_i^2` per arc —
  i.e. it degenerates to a per-cell surcharge and cannot distinguish twenty
  mildly-risky arcs from one very risky arc, which is the distinction that
  matters for a single-event failure like tip-over.
- *Bottleneck (min-max) aggregation in the Bellman.* Valid DP and it does capture
  concentration, but the value function goes flat across regions sharing a worst
  cell, killing the gradient the carrot reads.
- *`h + k*sigma`.* See section 2(a). It is a no-op.

---

## 4. Part B — exploration

The sensor has a **limited FOV** (`/odin1/cloud_raw`, dTOF). That makes heading a
sensing decision, and the lattice state `(x, y, theta)` is therefore already a
viewpoint as well as a motion state. Exploration needs no new planner.

### 4.1 Same solver, different seed

`_init_lattice_kernel` zeroes `V` at the goal. **Zero it at every frontier pose
instead** and the identical solver gives frontier exploration — same primitives,
same settle feasibility, same direction-awareness, same policy readout. Value
iteration does multi-source wavefronts natively (A* would need a virtual node).

With limited FOV a frontier "pose" is a cell *plus a heading that looks at the
unknown*. Seeding only those headings makes the lattice route the robot to arrive
**already pointing the right way** — the approach-alignment behaviour that the
stress worlds already showed matters.

Frontier needs no new state with a probabilistic map: cells at
`sigma ~ sigma_max` adjacent to low-sigma cells.

### 4.2 The doubt field — danger vs ignorance

Compute `z` twice in the same kernel:

    z     = margin / sigma_eff      # what I know
    z_opt = margin / sigma_floor    # what I would know if the map were certain

| condition | meaning |
|---|---|
| `z_opt < k`                | genuinely bad terrain — looking will not help |
| `z_opt >> k` but `z < k`   | **blocked by ignorance** — worth resolving |

One extra scalar per pose. This is the difference between exploring everything
unknown and exploring what is actually in the way.

### 4.3 Two solves give the value of information, in metres

Solve the lattice twice — pessimistic (real sigma) and optimistic
(`sigma = sigma_floor`) — and compare at the robot's own pose:

    gap = V_pess[robot] - V_opt[robot]

`gap` is what ignorance costs, in the same units as the plan. It yields three
things:

- **A trigger.** Explore only when `gap` exceeds a threshold. Pessimism alone
  never explores (ignorance is expensive); optimism alone always explores
  (ignorance is free); the *gap* is the honest signal.
- **A target.** Roll out the *optimistic* policy from the robot's pose and
  collect the high-doubt poses along it. Those are the cells whose resolution
  would unlock the better route — decision-focused sensing by policy rollout,
  no adjoint required (the cheap version of C4 in `SENSITIVITY_PLAN.md`).
- **A safety net.** `V_pess[robot] = +inf` while `V_opt[robot] < inf` means the
  goal is unreachable *purely because of ignorance* — `blind_fill_goal_unreachable`
  diagnosed automatically, with a defined response (relax `k`, or go look) instead
  of declaring the goal unreachable and stopping.

The optimistic solve need not be full fidelity — it only answers "is there a much
better route if I were certain" — so coarser headings or grid is acceptable.

### 4.4 Cheap fallback

If only one thing gets built: **optimism with a pessimistic veto.** Route on the
optimistic map, but keep `z < k` as a hard gate on what is actually driven over.
Self-correcting (driving toward optimistically-good terrain is what reveals it),
principled (optimism in the face of uncertainty), paid for in occasional wasted
travel.

### 4.5 Not to be built

- **Entropy-maximising next-best-view.** Uncertainty is not relevance — already
  established in this repo's own prior-art work. Maximising information gain
  sends the robot to look at whatever is most uncertain, usually the far edge of
  the map.
- **A separate exploration planner.** Same solver, different seed. Anything else
  duplicates the feasibility model, and then the two disagree.
- **Adjoint map-cell sensitivity at runtime.** Keep it for the paper; the
  two-solve gap gets most of the benefit for a fraction of the complexity.

---

## 5. What the mapping node must publish

Derived from the above, not from mapper convenience:

1. **mean elevation** per cell — a mean, NOT the per-cell max deployed today. MEASURED: the
   `max` reduction costs 35% of attitude accuracy (3.23 -> 2.11 deg pitch, 4.48 -> 2.84 deg
   roll), and being an extremum it has no sigma to publish at all (RESULTS.md section 2).
   This replaces the max **within a cell** only. The max **across a wheel footprint** stays —
   it is the physics, and it is where the Clark fold lives (section 3.3).
2. **per-cell sigma**, keyed on OBSERVATION RANGE. MEASURED: 2.1 cm at 0.8 m, 4.7 at 1.2,
   7.3 at 1.8, 9.7 at 2.8, 11.2 at 3.2, 18.3 at 5.8 m (RESULTS.md section 3). A weighted
   linear fit gives `sigma(r) = 2.7 cm + 2.3 cm/m * r` but overestimates the near field by
   2x — use the table. Re-observing from close range is worth 5x, which is an exploration
   objective the section 4.3 gap can express directly.
3. **one scalar spatial correlation length `L`.** Not a covariance matrix; the
   robot geometry is fixed, so one length makes every needed `rho` a constant.
   MEASURED: `L ~ 0.1 m`, plus a long-range correlation floor of 0.04 that a single
   exponential does not capture — publish the floor too, or the model claims an independence
   it does not have at 1-2 m.
4. **sigma that shrinks on observation and grows with age.** Without the first,
   exploration never terminates; without the second, it never re-triggers and the
   dynamic-object work is invisible to the planner.
5. **sigma that depends on incidence angle and range**, not just observed/not.
   A grazing return gives poor elevation; with a low-mounted limited-FOV sensor
   most far-field returns are grazing.
6. **no constant fill of unobserved cells.** Unobserved is `sigma = sigma_max`.

The mapper is also where the *differential-over-0.5-m* accuracy requirement from
section 2(b) belongs: absolute bias mostly cancels, local inconsistency does not.

---

## 6. Open decisions — settle these before implementing

### 6.1 Prerequisite: sigma calibration harness -- DONE, see `studies/calib/RESULTS.md`

`z` divides by sigma, so **a 3x error in sigma is a 3x error in every margin.**
The Clark paper measured exactly this failure on BASEPROD: the +/-1 sigma band
covered surveyed supports at 20-21% of nodes against a nominal 68%, i.e. the
mapper's own belief was miscalibrated by roughly 3x, and it concluded that
"mapper miscalibration outweighs propagation error on this record."

Built as `studies/calib/` and run 2026-09-17 on `bags/ostrich*`. It compares ATTITUDE rather
than support height -- the engine's body origin and `odin1_base_link` differ by an unknown z,
which cancels in pitch/roll -- and measures sigma per cell from the spread across sweeps.

Delivered: `sigma_pitch ~ 2.1 deg`, `sigma_roll ~ 2.8 deg`, the `sigma(range)` table above, and
`L ~ 0.1 m`. At `k = 2` that spends 4-6 deg of a 15 deg roll envelope on map uncertainty.

STILL OPEN from this probe: the +1.3 to +2.3 deg roll bias is unexplained (a sensor mount roll
error is the natural candidate and is not ruled out; it needs the same flat patch driven at
several headings), and `sigma_clear` has no end-to-end check because only attitude was
compared.

### 6.2 Does the tilt clamp mask contested contacts?

The story in 3.3 is that a knife-edge contact produces a huge `sigma_roll` and so
a low `z`. But `settle()` runs a *damped* Newton with `solver.max_step` and clamps
pitch/roll to `solver.tilt_clamp` every iteration. Check whether a near-degenerate
settle actually produces the large sensitivity, or whether the clamp hides it. If
it is hidden, `J^-1` is not the right sigma carrier there and 3.1 needs revisiting.

### 6.3 The sigma-growth rate is load-bearing, and has no obvious default

With a limited FOV, everything outside the current view is ageing. Set growth too
fast and the world behind the robot becomes frontier, so it pivots constantly to
re-look; too slow and the map is stale. **This single parameter decides whether
the robot spins in place.** It needs a deliberate value and a test, not a default.

### 6.4 Pivot-to-look has no value in the cost model

`pivot_cost` is a fixed metre-equivalent penalty. A pivot that reveals the route
is worth paying for and one that does not is not, and nothing currently expresses
the difference. Options: leave it (the frontier seeding of 4.1 partly covers it,
since a look-at-frontier heading becomes a zero-cost goal), or add an explicit
information term to the pivot primitives. Prefer the former until measured.

### 6.5 Self-occlusion -- CLOSED, not a blocker

A low-mounted limited-FOV sensor shadows a lot of nearby ground — possibly
including where the wheels are about to go. Is the ground immediately under and
ahead of the robot ever observed at useful incidence? If not, the most
decision-relevant cells are structurally unobservable from the driving pose, and
the exploration story in Part B needs rethinking (or the sensor needs remounting).
Measure this from bags before building anything in Part B.

### 6.6 Carrot and pivot primitives do not compose

A pure-pursuit carrot always drives forward toward a lookahead point. If the
optimal policy says "pivot in place," the carrot cannot express it. Needs an
explicit mode switch and a rule for when to switch. This is a hole in the
"simple carrot" plan and it is independent of everything probabilistic.

Related and still open: which MPPI-era modules survive. `control/command.py`
(output conditioning, goal brake, turn boost), `control/terminal.py` (dock) and
`yaw_track.py` are optimizer-independent and should be kept; the robust-mu and
friction-certificate work lives only in rollouts and is lost with MPPI. Decide
explicitly whether MPPI is deleted or shelved, and if shelved, keep the lattice
output rich enough for it to consume later.

### 6.7 Lattice solve time is now a safety number

Unmeasured, and it was never critical while MPPI provided local reactivity. With
a carrot, the lattice is the sole safety authority, so replan latency bounds how
far the robot commits blind. It also decides whether the two-solve scheme of 4.3
is affordable. **Measure first.**

### 6.8 Arbitration: goal vs explore

`goal_source` currently takes `follow` / `click`. Exploration is a third mode.
Decide whether it is an always-on fallback (engage when `gap > threshold`) or an
explicit mission mode the operator selects. Affects the node's parameter surface
and the safety story.

### 6.9 One `k`, or one per mechanism?

Start with a single global `k`. If it proves too blunt in the field, the next step
is per-mechanism `k` (one for tilt, one for clearance) — **not** a return to
variance-in-the-Bellman.

### 6.10 What metric is allowed to decide any of this?

Median ranking regret was zero for every method in the Clark study, so it cannot
adjudicate. Candidates that can move: near-miss / intervention count across the
stress worlds, unreachable-goal rate, and frontier-progress rate for Part B.
Pick before running, not after.

---

## 7. Suggested ordering

The mapping node does not exist yet, but most of this does not wait on it.

1. ~~**sigma calibration harness (6.1)**~~ — DONE 2026-09-17, `studies/calib/RESULTS.md`.
2. **Measure (6.2) and (6.7)** — two cheap measurements that can each invalidate a design
   choice above. (6.5) is measured and closed.
3. **z-margin field (Part A)** against a *synthetic* sigma (e.g. derived from
   observation count) — exercises the whole machinery without waiting on the
   mapper.
4. **Doubt field + two-solve gap (4.2, 4.3)** — pure planner, testable in sim
   today.
5. **Clark fold at the dilation stage (3.3)** — swap in once a real sigma layer
   exists.
6. **Frontier seeding (4.1)** — after 6.3 and 6.5 have answers.

Carrot design (6.6) runs in parallel and is independent of all of it.
