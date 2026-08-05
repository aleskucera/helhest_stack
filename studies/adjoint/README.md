# Study A — is the IFT settle adjoint correct?

Gate for the decision-focused-sensing project (`SENSITIVITY_PLAN.md` §5). Run:

```
.venv/bin/python -m studies.adjoint.study_a      # ~1 min on an RTX A500
```

Writes `studies/out/study_a.{png,npz,json}`. The console output carries the full stratified
table; the verdict block at the end summarises.

## What is actually being tested

`dJ/dh` reaches the elevation grid through **two** differentiable layers, and they fail for
different reasons at a curb. Conflating them would have made the study uninterpretable, so
each is isolated:

| layer | code | nature |
|---|---|---|
| `envelope → (z, pitch, roll)` | `step.py:settle_bt` / `adj_settle_bt` | hand-written IFT adjoint — **the thing under test** |
| `elevation → envelope` | `simulator.py:_contact` (off-tape) + `envelope.py:gather_bt` (on-tape) | frozen-arg-max Danskin subgradient |

Levels, each adding exactly one mechanism:

| | adds | leaf | dilation |
|---|---|---|---|
| **A0** | forward parity — replica == production, batched == fused, Warp == numpy | — | — |
| **A1** | one settle, no chaining | envelope | identity |
| **A2** | settle chain + `step_predict` + BPTT | envelope | identity |
| **A3** | the frozen-arg-max dilation | elevation | real |
| **A4** | the friction path, non-uniform μ | friction | real |

The oracle is finite differences of the **Warp forward itself**. An adjoint is only correct
*with respect to a forward*; differencing the numpy reference instead would conflate "the
adjoint is wrong" with "the two forwards have drifted". The numpy reference is used once, in
A0, as a model-parity check — it agrees to 5e-7 m.

## Design choices that carry weight

- **Non-uniform terrain, four labelled regions.** On flat ground `gx = gy = 0`, which
  collapses `J[i,1]` to `dp[2]` and zeroes the adjoint's entire `adj_pose[0]/[1]`
  accumulation — a flat-terrain check exercises none of those terms and cannot fail.
- **One-sided differences, not just central.** `D+` and `D-` separately give a per-cell kink
  ratio for the price of one extra forward per sweep. This is what separates *"the adjoint is
  wrong"* from *"the forward is not differentiable here"*. Without it, a central difference at
  a kink silently reports the mean of two one-sided slopes and reads as a halved gradient —
  which is exactly how the historic ~47% friction bug read, so the two are indistinguishable
  unless the sides are kept apart.
- **The zero set is probed.** Testing only cells where the adjoint is already nonzero cannot
  detect a *dropped* term — it reports zero, is never probed, and passes.
- **`atol = 0`, fixed Newton count.** Otherwise `h ± ε` can take different iteration counts
  and the forward map carries step discontinuities that FD reads as noise.
- **Linear functionals.** A quadratic tilt cost `pitch² + roll²` has an identically zero
  gradient wherever `pitch = roll = 0` — it would leave the flat lane with nothing to test.
- **B is the region sweep.** Terrain is per-rollout, so one perturbed forward yields B
  independent finite differences. The whole study is ~1 minute.

## Results

**1. THE GATE — PASS.** `adj_settle_bt` is correct on non-uniform terrain. Over A1 and A2,
across flat / slope / curb / rock: worst relative L2 error **9.5e-3**, worst regression slope
deviation **1.9e-3**, cosine 1.00000. Settle residual 6e-8 m, so the IFT premise (`c = 0` at
the root) holds. **Proceed to Study B.**

Note *where* this passes: the curb region is as clean as the flat one once the dilation is
taken out. The settle adjoint is not the fragile part.

**2. FINDING — the frozen-arg-max dilation has a validity radius, and it is quantization, not
physics.** The cap step between neighbouring offsets is `R − √(R² − cell²)` = **3.6 mm** at
0.05 m cells with R = 0.35 m. A height perturbation larger than that at an adjacent cell flips
the envelope arg-max. The measured kinked fraction goes 0% → 100% exactly as ε crosses it
(panel (c)); with the dilation off there is no such cliff. Even *below* the radius the
dilation costs 1–2 orders of accuracy on sharp terrain — same functional, dilation ON vs OFF:
flat 8.2e-4 / 4.2e-4, slope 8.7e-4 / 3.3e-4, curb **1.6e-2** / 2.2e-4, rock **2.7e-2** /
6.4e-4.

A follow-up numerical experiment established the *cause*: if the contact is allowed to slide
sub-cell instead of being restricted to cell centres, `∂env/∂δ` rises **smoothly** 0 → 1 over
0–7.5 mm rather than stepping 0 → 1 at 3.6 mm. So the cliff is an artifact of quantizing the
contact, and sub-cell refinement would remove most of it — while making the forward model
*more* accurate (today's dilation systematically under-estimates the wheel rest height). What
survives refinement is genuine ties: a wheel bridging two truly equal contacts, which is what
happens right at a curb edge.

> Note this is why the max must be **refined, not smoothed**. A softmax envelope biases the
> forward model (LSE ≥ max) and, to be smooth across a σ-sized perturbation, would have to
> blur the terrain at the σ scale — i.e. build a model that cannot see the curb. Sub-cell
> refinement has the opposite character: it improves accuracy and smooths as a side effect.

→ **For Study B:** map σ of a few cm is far above this radius, so first-order propagation
through the dilation is suspect for realistic uncertainty. B should compute FOSM **both ways**
— today's frozen-cell-centre gradient and a sub-cell-refined one — against the same
Monte-Carlo truth, so the size of the fix is measured rather than assumed.

**3. FINDING → FIXED, and the fix is measured.** `chassis_clearance` took a `min` over 18
belly points that all share one body-frame z, so on level ground they tied *exactly* (18 of 18
within 1 mm). `d(min)/dh` handed the whole gradient to one arbitrary point and reported a hard
zero at the rest. `step.chassis_clearance` now returns **both**: the `min` (unchanged, still
the feasibility gate) and `soft = Σᵢ max(clear_margin − cᵢ, 0)`, which has no tie.

Same functional, same terrain, regression slope against finite differences:

| region | `clear` (tied min) | `clear_soft` (hinge sum) |
|---|---|---|
| flat | 0.50 | **1.0000** |
| slope | 0.42 | **1.0000** |
| curb | 0.39 | **1.0000** |
| rock | 0.30 | **1.0001** |

Cosine 0.36–0.59 → **1.00000**; worst relative L2 ~1.0 → **2.6e-3**; kinked fraction 73% → 2%
on flat; false zeros gone. Panel (d): the tied min scatters into a cross, the hinge sum lies
on the identity.

Gate on `min`, differentiate `soft`. **`mppi.py`'s live penalty is deliberately unchanged** —
switching it changes the cost's scale and would need re-tuning against field behaviour.

**4. FINDING — the friction path is clean.** On a non-uniform μ field: worst relative L2
1.5e-2, worst slope deviation 7.9e-3. The historic `sample_field` position-gradient bug read
as ~47%, i.e. a slope of ~0.53. Nothing of that kind is present.

**5. FINDING — `min N_i` cannot serve as the linearisation-validity flag on this robot.** It
reaches zero only at static tip-over, which on a *tripod* is about a support-triangle edge
rather than about the track: front axle **29.5°**, left/right rear 34.6°. (The naive
`atan(half_track / CoM height)` = 46° is the wrong lever arm.) Planner limits are 15° roll and
15° pitch-down, so the switch is unreachable inside the feasible envelope — measured `min N_i
/(m·g)` never fell below 0.24 anywhere in the scene.

→ **Replaced.** `make_tiled_contact` now also emits `contact_margin` = winner − runner-up of
the dilation arg-max: exactly how far terrain must move for the wheel's contact to change,
i.e. the radius in which the frozen gradient is the true derivative. It costs one extra
register tile, never enters the envelope or any gradient, and it is discriminative
(panels (e), (f)):

| region | median margin | min |
|---|---|---|
| flat | 3.59 mm (= the geometric cap step) | 0.03 mm |
| slope | 1.35 mm | 0.13 mm |
| curb | 3.59 mm | **0.000 mm** |
| rock | 4.25 mm | **0.000 mm** |

Exact ties occur only at curb edges and on rocks — which is where the plan expected trouble,
just via a different mechanism than it named.

## Changes to production code made by this study

| file | change |
|---|---|
| `engine/step.py` | `chassis_clearance` returns `vec2(min, soft)`; `clear_soft` threaded through `step_finalize`, `step_kernel`, `step_kernel_bt`, `rollout_kernel` |
| `engine/simulator.py` | `clear_soft` buffer; `contact_margin` buffer |
| `engine/envelope.py` | tiled contact tracks the runner-up and emits `margin` |
| `tests/engine/gradients.py` | **was already broken** before this work (launched 15 args against a 17-arg kernel — same staleness class as `benchmarks/`); repaired, all five self-tests now pass |

`tests/engine/step.py` still reports `fused rollout_kernel == per-step  worst=0.00e+00`, so
the graph-captured hot path is unchanged bit-for-bit.

## Caveats

- Single robot geometry and one grid resolution (0.05 m). The 3.6 mm radius scales as
  `R − √(R² − cell²)`, so it is ~14.6 mm at the real 0.1 m perception grid — still well below
  a realistic σ. Re-run at 0.1 m before quoting a number in a paper.
- `STUDY_CLEAR_MARGIN = 0.30 m` (vs the production 0.05 m) is used so the belly hinge is
  active at all: the belly sits 0.25 m above the contact plane and the tallest scene obstacle
  is 0.20 m. It puts every belly point in violation, i.e. the maximal-tie case — the hardest
  test for the fix — and affects nothing else, since `clear_margin` never enters the settle or
  the twist.
- float32 throughout; the ε-sweep plateau guards against reading FD noise as error.
- The probe set is capped per region (`metrics.probe_cells`); counts are printed per row.
