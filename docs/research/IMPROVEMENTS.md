# helhest_stack — Motion Model Improvements

Candidate improvements to the planning simulator (`helhest.engine`), ordered by
value-per-cost. Each entry states **why** (what the model gets wrong today),
**how** (concretely, against the current code), **cost**, and **how to verify**.

Written 2026-08-05. Numbers marked *(measured)* were benchmarked on this
machine; numbers marked *(estimated)* are arithmetic from the model constants
and should be checked before anyone relies on them.

---

## 0. The measured baseline

Everything below is sized against this. `ForwardSimulator.rollout_launch`,
RTX A500, grid 241×441, T=25, forward only *(measured)*:

| B | ms/launch | rollout-steps/ms |
|---|---:|---:|
| 1024 | 0.947 | 27,026 |
| 4096 | 1.268 | 80,781 |
| 8192 | 2.206 | 92,837 |

**Two facts that shape every decision here.**

**(a) You are latency-bound below B≈4096.** 1024 → 4096 is 4× the work for
1.34× the wall time. Roughly 3× of extra per-step arithmetic is free at
planner scale — it hides in launch overhead.

**(b) You have ~2 orders of magnitude of headroom.** At dt=0.1 s the control
period is 100 ms; a full B=4096 rollout is 1.27 ms, i.e. **1.3% of the tick**.
Perception is 5.5 ms (`docs/performance.md`), so the whole loop is ~7%.

Consequence: **model fidelity is nearly free inside the current formulation.**
Every Tier 1 item below is ≤2×. What is *not* free is leaving the formulation —
see §9 (Non-goals) for the two walls, both of which are ~100×.

Robot constants used throughout (`model.py`, `robot.py`): m = 106.2 kg,
weight 1042 N, R = 0.35 m, half-track b = 0.365 m, rear offset 0.75 m,
CoM x ≈ −0.198 m, min turn radius 0.5 m, max_roll 15°.

---

## Tier 1 — free, and the model is currently wrong without them

### 1. Friction saturation certificate

**Why.** `step.py` already computes, every step, for every rollout:

```
grip_i = mu_i * N_i
total_grip = Σ grip_i
alpha  = 1 + k_turn * total_grip / (m*g)
x_icr  = grip_x / total_grip
```

**The friction budget is computed and then never spent.** It only bends the
turning geometry; it never limits achievable motion. Three consequences:

- **Inverted in the low-friction limit.** As μ→0, `total_grip`→0, so α→1 — the
  *ideal* differential-drive yaw rate — while forward speed stays exactly
  `r(ω_L+ω_R)/2`. The model says ice gives you perfect traction and your best
  turning. That is not "approximate", it is backwards, and it is optimistic on
  exactly the terrain you most want to avoid.
- **Gravity never enters the twist.** No downhill drift, no climb failure, no
  heading swing on a side slope.
- **No centripetal limit.** Turning at speed needs `m·v·ψ̇` laterally; nothing
  checks it.

Scale of the error *(estimated)*, at μ = 0.5 (budget 521 N):

| situation | demand | % of budget |
|---|---:|---:|
| hold station, 15° slope | 270 N | 52% |
| hold station, 30° slope | 521 N | **100% — cannot hold** |
| turn at 1 m/s, R=0.5 m | 212 N | 41% |
| 15° slope while turning | ~480 N | 92% |

So the model is fine on flat ground with grip, and **qualitatively wrong past
~15° of slope or on any low-μ patch** — always in the optimistic direction.

**How.** In the same kernel, after the settle and `normal_loads`:

```
demand_long = m*g*sin(pitch)          # + m*a  once momentum exists (§5)
demand_lat  = m*v*psi_dot + m*g*sin(roll)
demand      = sqrt(demand_long² + demand_lat²)     # friction ellipse
saturation  = demand / max(total_grip, floor)
```

Write `saturation` to a per-step output array alongside `loads`. **Guard the
denominator** — unguarded ratios blow up exactly where they mean least (a
near-unloaded contact reports enormous saturation while transmitting nothing).
This was learned the hard way in the Ostrich relaxation work.

Then choose what to do with it, in increasing order of commitment:

1. **Certificate only** — feed `max(saturation − 1, 0)` into the MPPI cost, the
   same way `resid_viol` is already handled in `mppi.py:310`. No dynamics
   change.
2. **Directional slip** — decompose demand, apply the ellipse, add a downhill
   drift velocity. ~50 flops.

**Do 1 first.** For a *ranking* problem, a penalty that stops you choosing the
icy traverse is worth more than an accurate simulation of sliding down it.
Skip the tempting middle option of isotropically scaling the twist — real grip
loss is directional (you keep going forward and lose the turn), so an isotropic
scale is the one behaviour that doesn't occur.

**Cost.** ~20 flops next to a 3×3 Newton with bilinear samples. **<1%.**

**Verify.** Build a constant-slope world at 10/20/30° and sweep μ. The
certificate must cross 1.0 at `tan(θ) = μ`, analytically. That is an exact
test, not a plausibility check.

---

### 2. Motor torque limit / stall

**Why.** Commanded wheel speed is *always* achieved — there is no torque model
anywhere, so the robot climbs any grade at full speed. On high-μ rough terrain,
which is most of the operating envelope, **you stall before you slip.**

Climbing 15° needs 270 N ÷ 3 wheels × 0.35 m = **31.5 Nm per wheel** at the
wheel, continuous *(estimated)*. Whether that is inside the real envelope is
the open question — see §10.

**How.** Same shape as §1 — required torque vs available, both already
derivable. Naturally fuses with §1 into a single "commanded twist is
unachievable" certificate with two reasons attached (`slip` / `stall`), which
matters because the *planner responses differ*: slip means pick another route,
stall means pick another speed.

**Cost.** Same as §1, shares the machinery. **<1%.**

**Verify.** Constant-slope sweep again; stall boundary must appear at the
grade where required torque crosses the limit, independent of μ. Crossing the
two sweeps (μ × grade) shows which limit binds first — that is a figure worth
having.

---

### 3. Tip-over margin from `min N_i`

**Why.** The settle forces all three wheels onto the envelope and solves 3
equations in `(z, pitch, roll)`. `normal_loads()` then solves a determinate 3×3
and **can return negative loads** — and negative `N_i` is exactly "CoM outside
the support triangle", i.e. the classical static tip-over test. It is already
computed every step and written to `loads_out[t,b]`. **Nothing consumes it.**

The subtler point: when a wheel *should* lift, the settle still forces it down,
so it converges to a pose satisfying three equations with no physical solution.
`z`, `pitch` and `roll` are **all** wrong from that step on, not just the
flagged wheel — the whole remaining trajectory is contaminated. So the
certificate matters even if the physics is never fixed: it marks where the
state stops being trustworthy.

**How.** `stability_margin = min_i(N_i) / (m*g)`. Feed the *continuous* margin
into the MPPI cost (not just a binary reject) so gradients/rankings degrade
smoothly as the robot approaches tipping. Reject below 0.

**Cost.** **Zero** — the quantity exists. This is wiring, not computation.

**Verify.** A ramp world with increasing bank angle: the margin must reach 0 at
the geometric tip angle from CoM height and half-track, computable by hand.
Cross-check against the existing `max_roll = 15°` gate — if they disagree, one
of them is mis-set, which is itself worth knowing.

---

### 4. Cylinder wheels instead of spheres

**Why.** `envelope.py` dilates the terrain by a **spherical cap**
`sqrt(R²−d²) − R` over a **disk** `|d| ≤ R`. The wheel is a sphere of radius
0.35 m — and wheel *width* is not a parameter anywhere in `model.py`, so the
assumption goes all the way down.

**The error is lateral.** A real wheel is a cylinder of half-width ~0.1 m, but
the sphere reaches **0.35 m sideways**. A rock 0.3 m to the side of the wheel
centre lifts and tilts the robot when in reality you straddle it. Systematically
**pessimistic in tight and rocky spaces**, and it corrupts pitch/roll, not just
z.

Along the direction of travel, sphere and cylinder are *identical* — the
cylinder's cross-section in the x–z plane is the same circle. **They differ
only laterally.** That is what makes the fix cheap and its effects predictable.

**How.** The obstacle is that the sphere's envelope is **yaw-invariant** —
which is exactly why one dilation can be shared across all B×T. A cylinder's is
not. Solution: **a yaw-binned envelope stack**, precomputed once per perception
frame, indexed by yaw at sample time. All three wheel axes are parallel on a
skid-steer, so one stack serves all three wheels.

- Replace the disk offset table in `wheel_offset_table()` with a **rectangle**
  2R × w, `cap = sqrt(R²−dx²) − R` depending only on the along-travel offset.
- Envelope becomes `[n_yaw, ny, nx]`; sample with the nearest bin.
- Bin count from `R·Δψ ≤ cell` → `Δψ ≤ 0.1/0.35 = 0.29 rad ≈ 16°` → **≥22
  bins, so use 32.**
- Add `wheel_width` to `RobotParams` / `model.py`. **Measure it on the real
  robot** — the 0.1 m half-width above is an assumption.

**Differentiability survives untouched.** The arg-max/gather split doesn't care
about the structuring element's shape; the IFT adjoint through the settle is
unchanged.

**Cost.**
- *Rollout:* unchanged — still one lookup.
- *Memory:* at the real 0.1 m cell and an 81×158 grid, 51 KB × 32 = **1.6 MB**.
- *Precompute:* the rectangle is ~14 cells vs the disk's ~37 at that
  resolution, so ~12× current dilation across 32 bins — still a small kernel,
  run once per frame, not per rollout.

**This is the best cost/benefit item in the document**: real geometric fidelity
for zero marginal rollout cost.

**⚠️ One coupled consequence — see §7.** The cylinder introduces a yaw
sampling constraint the sphere never had.

**Verify.** A world with a single rock offset laterally from the wheel track.
Sphere lifts the robot; cylinder should not. Compare predicted roll against the
geometry by hand.

---

## Tier 2 — cheap, conditional on where the robot is going

### 5. Body momentum

**Why — and why *not yet*.** There is no velocity state; `tau_motor` models
actuator lag but not body inertia. At current speeds this is genuinely
negligible *(estimated)*:

Froude number `v²/(gL)` with L = 0.75 m: at ω = 2 rad/s → v = 0.7 m/s →
**Fr = 0.067**. Quasi-static is not an approximation there, it is excellent.
You would need v ≈ 2.7 m/s to reach Fr ≈ 1.

Stopping distance `v²/(2μg)`:

| v | μ=0.5 | μ=0.2 |
|---|---:|---:|
| 0.7 m/s | 5 cm | 12 cm |
| 1.0 m/s | 10 cm | 25 cm |
| 2.0 m/s | 41 cm | 102 cm |

At 0.7 m/s the robot stops within *half a control tick*. **Adding momentum
would change nothing measurable.** It starts to matter around 2 m/s.

**How, when needed.** Body-frame `(v, ψ̇)` states with first-order lag, driven
by the commanded twist and capped by §1/§2. Stays inside the fused kernel; no
new solve.

**Cost.** ~1.2× *(estimated)*. Two extra states, no extra iteration.

**Trigger.** Do this when `v_max` goes above ~1.5 m/s, not before.

---

### 6. Two-contact re-solve on tipping

**Why.** §3 detects tipping but leaves the pose wrong. This fixes the pose
without leaving quasi-statics.

**Important structural note:** for a **rigid tripod**, a lifted wheel has *no
static equilibrium* — with two contacts the robot balances only if the CoM
projects exactly onto the line between them; generically it is *falling*. So a
2-contact state is inherently dynamic, and this improvement is a **better
approximation of a tipping pose**, not a correct simulation of tipping. Anything
more requires §5 plus small timesteps, which is the wall in §9.

**How.** When `min N_i < 0`, re-solve with that wheel free: rotate about the
line through the other two until it re-contacts or exceeds the tip limit. 2×2
plus a rotation, analytic, warm-started from the 3-contact solution.

**Cost.** ~2× the settle, **but only on flagged rollouts** *(estimated)*. If
tipping is rare the amortised cost is negligible; if it is common your planner
has bigger problems. Note the warp caveat in §9(c): one thread per rollout
means a divergent branch is paid by the whole warp. Consider a compaction pass
(§8) if the flagged fraction is large.

**Verify.** Tip margin from §3 must decrease monotonically through the
re-solve, and the resulting roll must match the geometric prediction on a
bank-angle ramp.

---

### 7. Decouple the control rate from the sensor rate

**Why.** Replanning is currently tied to point-cloud arrival. It needn't be —
map updates at sensor rate, EKF propagates state at control rate, MPPI replans
on the last map plus the propagated pose. The machinery all exists.

The gain is control-side, not fidelity-side: disturbance rejection, loop
latency, and dynamic obstacles. A pedestrian at 1.5 m/s moves 15 cm per 100 ms
tick versus 7.5 cm at 20 Hz.

**Note this does *not* require changing the rollout `dt`.** Keep T=25 at
dt=0.1 s and simply launch more often — 20 Hz then costs 2.5% duty cycle
*(measured basis)*.

**On changing `dt` itself — the honest criteria.** Two independent bounds:

| criterion | bound | at v=0.7 | at v=1.0 | at v=2.0 |
|---|---|---:|---:|---:|
| one grid cell per step | `dt ≤ cell/v`, cell = 0.1 m | 143 ms | 100 ms | 50 ms |
| envelope bandwidth (Nyquist at R/2) | `dt ≤ R/(2v)` | 250 ms | 175 ms | 88 ms |

**dt = 0.1 s is already adequate up to ~1 m/s.** (Note the real perception grid
is 0.1 m/cell per `docs/performance.md`; the 0.05 m in `heightmap.py` is a
synthetic-scene default and must not be used for this calculation.)

**The one criterion that does force a finer dt is §4's cylinder, via yaw
rate.** Once the envelope is yaw-dependent, `ψ̇·dt` must stay inside a yaw bin.
From the turning model, spin-in-place gives `ψ̇ ≈ 0.44·ω_max` (at μ=0.6,
α = 2.2) *(estimated)*:

| ω_max | ψ̇ | Δψ per 0.1 s | vs 32 bins (11.25°) |
|---|---:|---:|---|
| 2 rad/s | 0.87 rad/s | 5° | fine |
| 4 rad/s | 1.75 rad/s | 10° | marginal |
| 8 rad/s | 3.5 rad/s | **20°** | **aliases** |

So if `ω_max` is near 8 rad/s, **§4 and a ~25 Hz step are the same change** and
should land together: the cylinder is what makes finer stepping worth having,
and finer stepping is what makes the cylinder correct. **`ω_max` is the number
that decides this** and it is not recorded anywhere in the repo.

**Cost.** Architectural, not computational.

---

### 8. Per-rollout mode compaction

**Why.** If any improvement applies to only some rollouts (§6, or a promotion
scheme), a `if` inside the fused kernel is paid by the **whole warp** — one
thread per rollout means the slowest thread sets the price.

**How.** Compact rollout indices by mode, launch one kernel per mode over its
index list. No divergence, each mode stays graph-capturable. Per-rollout
granularity is natural since MPPI resamples every cycle; **per-step switching
would wreck the fused `rollout_kernel`** and should not be attempted.

**Cost.** One compaction pass. Only worth it if a mode's flagged fraction is
substantial.

---

## Tier 3 — research-grade, higher risk

### 9. Uncertainty bracketing instead of model promotion

**Why.** For a *ranking* problem the question is never "how accurate is this
model" but "does this uncertainty change which trajectory wins". Bracketing
answers that directly and needs no second model.

**How.** Run the existing model twice with optimistic and pessimistic μ (spread
sourced from the friction field's own uncertainty, or from
`filtering/variance_transformation.py`). If both agree on the verdict, no
further work. If they disagree, that candidate is decision-critical — penalise
or promote it.

Note `mppi.py` already has K slip samples for CVaR — this is the same
machinery pointed at model uncertainty rather than control noise, and may be
largely wiring.

**Cost.** 2× the cheapest thing you have, using the **existing kernel
unchanged**. Given §0(a) — latency-bound below B≈4096 — this may be closer to
1.3× in practice.

---

### 10. The Level-1 vs Level-2 ablation *(the paper-relevant one)*

**Why.** The multi-fidelity architecture in the prior-art documents assumes a
cascade from a quasi-static prefilter to an exact friction cone is worth
building. The cliff is real *(measured)*: **~93,000 rollout-steps/ms here
versus ~9 world-steps/ms for Ostrich forward-only** — about 10⁴.

But **Tiers 1–2 define an intermediate level that costs <2×**, and nobody has
measured whether it recovers most of the expensive level's *ranking*. If it
does, that is a significant finding about the method — far better discovered
here than by a reviewer. If it does not, it is the strongest possible
motivation for the cascade, measured rather than assumed.

**How.** Fix a scene set. For each candidate population compute the cost
ranking under (a) current model, (b) current + Tier 1, (c) Ostrich. Report
**top-K overlap and Kendall's τ**, not trajectory RMSE — only ordering affects
what MPPI does.

**Cost.** Offline study, not runtime.

**Caveat worth internalising.** In the Ostrich relaxation work, removing the
friction complementarity — the supposedly expensive structure — bought
*nothing*: μ=50 cost exactly what the full cone cost (5.22 vs 5.22 NR/step,
trajectories within 6.6e-4 m). Under saturation the binding cone turned out to
be a *stabiliser*, and every way of removing it was 2.4–2.8× worse. **Do not
assume which part of a model is expensive — measure it.** That lesson is the
main transferable result from that branch.

---

### 11. Downward modes — skip work on easy terrain

**Why.** The budget for expensive modes has to come from somewhere, and a large
fraction of rollouts cross locally-flat terrain where the settle converges in
~1 iteration and pitch/roll ≈ 0.

**How.** A flatness test on the heightmap patch (cheap: envelope gradient
magnitude over the wheel footprint) short-circuits the settle to
`z = envelope, pitch = roll = 0`.

**Cost.** Negative, on flat terrain. Combines with §8.

**Caveat.** Warp divergence again — worth it only with compaction, or if the
flat fraction is overwhelming.

---

## 12. Maintenance

`benchmarks/forward.py:21` and `benchmarks/differentiable.py:19` import
`helhest.control.reference._to_target_wheel_omega`, which no longer exists
anywhere in `src/`. Both benchmarks fail at import. Pre-existing; found while
benchmarking for §0.

---

## Non-goals — and precisely why

These are the two walls. Both are ~100×, and everything above is designed to
stay on the near side of them.

**(a) Reducing the timestep for stiff dynamics.** Nothing in the current model
is stiff — contact is *solved* quasi-statically at each pose, not *integrated*
— so there is no CFL-type bound on `dt` at all. That property is what buys
dt = 0.1 s and it should be defended deliberately. Contact dynamics needs
dt ~1–5 ms; at dt = 1 ms a 2.5 s horizon goes from T=25 to T=2500, which eats
the entire headroom by itself. **The expensive thing is never the model — it is
the timestep the model forces.**

**(b) Per-rollout terrain state.** Memory, not compute: 4096 rollouts ×
241×441 × 4 B = **1.7 GB** on a 4 GB card. This is already why
`DifferentiableSimulator` is batch-limited. Anything where terrain *changes per
rollout* — rutting, deformation, snow compaction, track laying — hits this wall
immediately. The shared-envelope design is load-bearing, not incidental.

**(c) Variable-iteration solves.** The settle is 3×3 with a fixed cap. One
thread per rollout means the warp pays the slowest thread, so a contact LCP
needing a data-dependent 10–50 iterations becomes worst-case-everywhere and
breaks clean graph capture.

**(d) Host sync per step.** Kills the fused kernel and graph capture. Costs far
more in lost fusion than in arithmetic.

**Deformable terrain / terramechanics** is excluded by (b). **Learned residual
correction** is not excluded on cost grounds but carries its own training risk
and is a different research direction.

---

## Recommended order

1. **§3** tip-over margin — zero cost, the quantity already exists
2. **§1** friction saturation — <1%, and the axis the model is blind *and*
   optimistic on
3. **§2** torque/stall — shares §1's machinery; determines which limit binds
4. **§4** cylinder envelope — free at rollout time, fixes a systematic
   pessimism (check §7's yaw constraint alongside)
5. **§10** the ablation — decides whether anything beyond this is needed
6. **§7** faster control loop — independent of the above, cheap, control-side win

§5, §6, §8, §9, §11 are conditional; their triggers are stated in place.

---

## Open questions that gate the above

1. **What is `ω_max`?** Decides whether §4 forces a finer `dt` (§7). Not
   recorded anywhere in the repo.
2. **What is the motor torque envelope?** Decides whether §1 or §2 binds first,
   and hence which certificate does the work.
3. **What is the target `v_max`?** Below ~1.5 m/s, §5 is unnecessary; above
   ~2 m/s it becomes required.
4. **What is the real wheel width?** §4 needs it; currently unparameterised.
5. **What does the robot actually get wrong in the field** — routing onto
   slopes it cannot hold (→ §1), or refusing gaps it could cross (→ §4)? Field
   logs beat all of the reasoning above.
