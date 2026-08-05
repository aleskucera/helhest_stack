# Study A — is the IFT settle adjoint correct?

Gate for the decision-focused-sensing project (`SENSITIVITY_PLAN.md` §5). Run:

```
.venv/bin/python -m studies.adjoint.study_a      # ~1 min on an RTX A500
```

Writes `studies/out/study_a.{png,npz,json}`. The console output carries the full stratified
table; the verdict block at the end is the summary reproduced below.

## What is actually being tested

`dJ/dh` reaches the elevation grid through **two** differentiable layers, and they fail for
different reasons at a curb. Conflating them would have made the whole study
uninterpretable, so each is isolated:

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
  wrong"* from *"the forward is not differentiable here"*. Without it, a central difference
  at a kink silently reports the mean of two one-sided slopes and reads as a halved gradient.
- **The zero set is probed.** Testing only cells where the adjoint is already nonzero cannot
  detect a *dropped* term — it reports zero, is never probed, and passes. The historic
  `sample_field` position-gradient bug was exactly a dropped term.
- **`atol = 0`, fixed Newton count.** Otherwise `h ± ε` can take different iteration counts
  and the forward map carries step discontinuities that FD reads as noise.
- **Linear functionals.** A quadratic tilt cost `pitch² + roll²` has an identically zero
  gradient wherever `pitch = roll = 0` — it would leave the flat lane with nothing to test.
- **B is the region sweep.** Terrain is per-rollout, so one perturbed forward yields B
  independent finite differences. The whole study is ~1 minute.

## Results

**1. THE GATE — PASS.** `adj_settle_bt` is correct on non-uniform terrain. Over A1 and A2,
across flat / slope / curb / rock: worst relative L2 error **9.1e-3**, worst regression slope
deviation **1.2e-3**, cosine 1.00000. Settle residual 6e-8 m throughout, so the IFT premise
(`c = 0` at the root) holds. **Proceed to Study B.**

Note *where* this passes: the curb region is as clean as the flat one (`rel L2` 2.4e-4 vs
3.9e-4) once the dilation is taken out. The settle adjoint is not the fragile part.

**2. FINDING — the frozen-arg-max dilation has a hard validity radius of 3.6 mm.** The cap
step between neighbouring offsets is `R - sqrt(R² - cell²)` = 3.6 mm at 0.05 m cells with
R = 0.35 m. A height perturbation larger than that at an adjacent cell flips the envelope
arg-max, and the frozen index is then differentiating the wrong branch. The measured kinked
fraction goes 0% → 100% exactly as ε crosses it (panel (c)); with the dilation off there is
no such cliff. Even *below* the radius the dilation costs 1–2 orders of accuracy on sharp
terrain — same functional, dilation ON vs OFF:

| region | ON | OFF |
|---|---|---|
| flat | 7.7e-4 | 3.9e-4 |
| slope | 1.0e-3 | 3.3e-4 |
| curb | **1.6e-2** | 2.4e-4 |
| rock | **2.7e-2** | 6.3e-4 |

→ **For Study B:** map σ of a few cm is far *above* this radius, so first-order propagation
through the dilation is invalid for realistic uncertainty — independently of anything the
settle does. This is a stronger and more specific limitation than the plan anticipated, and
it is a property of the *envelope construction*, not of contact mechanics.

**3. FINDING — the belly-clearance gradient is structurally unreliable.**
`chassis_clearance` is a `min` over 18 belly points that all share one body-frame z
(`robot.py:_chassis_pts`), so on level ground they tie **exactly** — 18 of 18 within 1 mm at
the flat-lane pose. The adjoint hands the whole gradient to one tied point and reports a hard
zero at the other 17; a central difference at a tie returns the mean of the two one-sided
slopes. Measured regression slope 0.24–0.74 (0.5 being the straddled-kink signature, not a
halved gradient), with the zero-control catching false zeros at up to 0.8 of the group scale.
Visible as the cross in panel (d).

→ Exclude `clear` from any first-order attribution, or replace the `min` with a smooth
aggregation (softmin / sum of hinges). **Not changed here** — that is a production change and
belongs in discussion, not in a validation study.

**4. FINDING — the friction path is clean.** On a non-uniform μ field: worst relative L2
1.5e-2, worst slope deviation 7.9e-3. The historic `sample_field` position-gradient bug read
as ~47%, i.e. a slope of ~0.53. Nothing of that kind is present.

**5. FINDING — `min N_i / (m g)` never falls below 0.24 anywhere in this scene, and cannot
fall to 0 inside the feasible envelope.** `min N_i` reaches zero only at static tip-over,
which on a *tripod* is about a support-triangle edge rather than about the track: front axle
**29.5°**, left/right rear 34.6°. (The naive `atan(half_track / CoM height)` = 46° is the
wrong lever arm.) Planner limits are 15° roll and 15° pitch-down.

→ **For Study B:** the plan's §4 premise — that `min N_i → 0` is *the* linearisation-validity
flag — needs re-examining. On this robot the settle's contact-set switch is unreachable
within the planner's own tilt envelope, so that flag would essentially never fire. The switch
that actually breaks first order here is the **dilation arg-max** (finding 2). Study B should
stratify on both, and the free detector it wants is a near-tie margin in `_contact`, not
`min N_i`.

## Caveats

- Single robot geometry and one grid resolution (0.05 m). The 3.6 mm radius scales as
  `R - sqrt(R² - cell²)`, so it is ~14 mm at the real 0.1 m perception grid — still far below
  a realistic σ. Worth re-running at 0.1 m before quoting a number in a paper.
- float32 throughout; the ε-sweep plateau is the guard against reading FD noise as error.
- The probe set is capped per region (see `metrics.probe_cells`); counts are printed per row.
