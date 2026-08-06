# Adjoint map-cell sensitivity — consolidated findings

**What this is.** Every result from the study, stated once, in its final corrected form,
organised by *what we believe* rather than by the order it was discovered.
`RESULTS.md` remains the detailed record and carries the derivation history and the
correction banners; where the two differ, **this document is current**.
Every claim names the script that produces it.

**Scope of the evidence.** All synthetic, one robot geometry, σ is a placeholder (that is the
single biggest weakness — see *Not established*). Statistics are paired sign tests on per-seed
differences unless stated. Two results in this study reversed when n was increased, so nothing
below n ≈ 100 is reported as a finding.

---

## 1. The headline

The IFT adjoint through the quasi-static settle is **correct** — and at realistic map
uncertainty its **support** is useful while its **magnitude** is not.

That one distinction organises everything else. Choosing *where to look* needs only the support,
and works (decision-focused sensing beats information-theoretic sensing 200/200 seeds). Estimating
*how much risk* needs the magnitude, and fails (every analytic route loses to a free
per-timestep heuristic). Five separate attempts to fix the magnitude — sub-cell refinement,
analytic Hessian, cap-pooling, second-order FOSM, soft contact gradients — were each tested and
each failed, with a measured mechanism.

---

## 2. Established

### 2.1 The adjoint is correct  · `studies/adjoint/study_a.py`

Finite differences **of the Warp forward itself** (not of the numpy reference — an adjoint is
only correct with respect to a forward), on a labelled four-lane scene with flat / slope / curb /
rock strata. A flat-terrain check would have passed vacuously.

| | |
|---|---|
| worst relative L2 error, all strata | **9.5×10⁻³** |
| worst regression slope deviation | 1.9×10⁻³ |
| settle residual (the IFT premise) | 6×10⁻⁸ m |
| Warp float32 vs numpy float64 forward | 5×10⁻⁷ m |
| friction path, non-uniform μ | 1.46×10⁻² |

The curb is as clean as flat ground once the dilation is removed — the settle adjoint is not the
fragile part.

### 2.2 Two real defects found in the engine

- **`chassis_clearance` had a structurally broken gradient**: a `min` over 18 belly points that
  all share one body-frame z, so on level ground they tie *exactly* and the adjoint routed the
  whole gradient to one arbitrary winner. Fixed with a tie-free hinge sum alongside the min;
  regression slope **0.30–0.50 → 1.0000**.
- **`min N_i` cannot serve as the linearisation-validity flag** on this robot. It reaches zero
  only at static tip-over, which on a *tripod* is about a support-triangle edge, not the track:
  front axle 29.5°, rear edges 34.6°, against a 15° planner limit. It never fell below 0.24
  anywhere in the scene. Replaced by a contact-margin diagnostic emitted from the dilation.

Both were pre-existing and invisible to the test suite. `tests/engine/gradients.py` was itself
already broken (launching 15 args at a 17-arg kernel) and was repaired.

### 2.3 First order is inadequate at realistic σ — with a scale-free criterion  · `study_b.py`

Self-check first: at σ×0.03 the FOSM/Monte-Carlo ratio is 1.00 ± 0.02, validating both the noise
normalisation and the correlated FOSM formula. Every later deviation is physics.

Stratified by a **per-source validity radius** (`contact.source_slack`):

| σ / slack | median FOSM/MC ratio | worse than 2× | n |
|---|---|---|---|
| < 1 | **1.02** | **0.0%** | 265 |
| 1 – 10 | 1.71 | 41.2% | 690 |
| ≥ 10 | 2.61 | 63.7% | 314 |

**The catch:** at realistic σ almost nothing is inside — 3 of 846 probes — because the slack is
millimetres while σ is centimetres.

**The criterion transfers** (`generalise.py`): re-run at 0.10 m cells on fractal 1/f terrain with
a completely different spectrum, attribution stays accurate inside the radius (1% and 0% bad) and
degrades outside (14%, 15%).

**But the flat-ground formula does not.** `R − √(R²−cell²)` predicts 3.6 mm at 0.05 m; measured
**16.5 mm**. At 0.10 m it predicts 14.6 mm; measured **10.4 mm**. On broadband terrain the
contact is decided by the ground's own relief, not by the cap's step between offsets — so the
formula is a *flat-ground special case*, and at fine resolutions a pessimistic one.

**Two supporting results.** i.i.d. per-cell map noise is not a valid model here — the envelope is
a max over ~37 cells, so zero-mean noise biases it **3.4σ** against **0.4σ** correlated. And
per-cell and global adequacy come apart: the most globally accurate rollout has among the worst
per-cell ratios, so a `Var(J)` check alone would have been misleadingly reassuring.

### 2.4 Uncertainty ≠ relevance — the strongest positive result

**The decoy** (`study_b.py`): an unobserved patch holding **67.3% of the map's total σ²** attracts
**0.000%** of the FOSM variance, because no wheel can reach it.

**In plan ranking** (`ranking.py`, n = 200, real taped adjoint): 16 candidate plans over a
partially-observed fractal map, each policy reveals M cells, scored by Kendall τ of the re-ranked
plans against ground truth.

`disagreement − entropy` at 400 cells: **+0.633, 200/200 seeds, p = 1.2×10⁻⁶⁰**.

And **entropy delivers no measurable value over not sensing at all** — indistinguishable from
random at every budget (p ≥ 0.12). Its cells land ~2.0 m off the nearest plan, where the adjoint
is *exactly* zero, so revealing them cannot move any plan's cost. Task-aware policies sit at
~0.05 m off-plan.

**Robust to all three real error sources** (`noise.py`, `verify_noise.py`, n = 200/arm). Gated
before use: flat ground 100% visible, a wall hides 100% behind and 0% in front; sensor std 1.05×
configured, autocorrelation 0.89 at one cell, 2.67× larger error far than near; the localisation
field reproduced **exactly from three numbers** (1×10⁻⁸).

`disagreement − entropy` @400 across arms: clean +0.735, sensor +0.667, occlusion +0.520,
localisation +0.509, **all three +0.391** — every one p < 10⁻²⁹.

**And the leak it could have depended on does not matter.** σ over unobserved cells was derived
from the truth's roughness — a real leak, favouring the *baseline* (entropy is nothing but σ).
Controlled two ways: `random` **is** entropy-with-uniform-σ exactly, and a `--flat-sigma` arm
removes the leak for every policy at once. Nothing moves: `disagreement` 0.822 → 0.818 (clean),
0.538 → 0.537 (full noise).

### 2.5 When an adjoint is actually needed  · `compare_families.py`, n = 200/family

Three plan families varying **only** how much the candidates' coverage overlaps. Verified by
construction: `speed`'s plans finish **1 mm** apart with coverage spread 200× below `fan`'s.

| family | adjoint − geometry @400 |
|---|---|
| `fan` — 16 separate corridors | +0.092 |
| `hybrid` — 4 paths × 4 speed profiles | **+0.180** |
| `speed` — 1 shared path | **−0.069** |

**Non-monotonic, and the endpoint is the finding.** Confounding the plans *completely* makes the
problem **easier** for geometry: one shared path means one narrow corridor, 400 cells covers it,
and `swath` reaches **τ = 1.000**. *If your candidates all follow one path, do not compute an
adjoint — sense the corridor.* The adjoint pays in the middle, which is the MPPI elite-set regime.

**Sharpest number in the study:** within `hybrid`'s path-identical groups — where geometry is
blind by construction — `disagreement` +0.869 vs `swath` +0.672 at 400 cells: **+0.198,
161/185 seeds, p = 4×10⁻²⁶**, recovering 92% of the oracle.

### 2.6 What limits sensing: pose error, and it is irreducible  · `compare_noise.py`

Oracle τ@400 — the best *any* policy could do:

| clean | + sensor | + occlusion | + pose | all three |
|---|---|---|---|---|
| 0.911 | 0.802 | **0.940** *(rises)* | **0.601** | 0.565 |

Realistic occlusion *raises* the ceiling (the ray cast observes 19.5% vs the disc's 7.9%). **Pose
error nearly halves it**, and no cell-selection strategy recovers it: revealing more cells just
delivers more measurements carrying the same three wrong numbers.

Under full noise the adjoint's edge over a distance transform **vanishes**: +0.016, p = 0.10.
Geometry already captures 91% of the achievable gain against the adjoint's 95%.

**A modelling result found while building this:** the textbook first-order per-cell marginal of a
pose error, `|∇h|·d`, is **invalid at realistic pose error**. A 2° heading error over a 6 m lever
arm displaces the map ~2.2 cells while this terrain decorrelates in ~1 — a three-parameter
first-order fit explains only **R² = 0.23** of the field it generates.

### 2.7 The gradient's magnitude is unusable for risk  · `risk.py`, `softgrad.py`, `order2.py`

**Direct evidence.** FD against the adjoint at the highest-|g| cells on the belief: **2.3% error
at ε = 1 mm, 32% at ε = 1 cm** — and σ is 2–30 cm. The adjoint is exactly correct and exactly
useless at the scale the uncertainty lives at.

**Why, mechanically.** The gradient is evaluated on the belief, which is flat wherever
unobserved, so the contact arg-max there is near-degenerate. Measured: mean |g| is **0.0812 on
observed cells against 0.0164 on unobserved** — 5× larger where σ is *smallest*. So `g·σ`
concentrates its weight where the map is already known.

**Every analytic route lands in the same place** (τ against Monte-Carlo truth, n = 40 paired):

| | τ |
|---|---|
| `Σ_t σ_t` — per **timestep**, free | **+0.140** |
| σ summed over cells on the support | −0.129 |
| first-order FOSM | −0.130 |
| soft contact gradient (τ = σ) | −0.129 |
| second-order FOSM | −0.190 |

### 2.8 The one transferable engineering rule: **aggregate risk over time, not over cells**

`order2.py`, paired, n = 40:

- **domain:** `cell_sigma − time_sigma` = **−0.270, 3/39 seeds, p = 3.6×10⁻⁸**
- **gradient weighting at fixed domain:** −0.000, 20/40, **p = 1.00 — exactly neutral**

A per-cell sum grows with the *area* a plan sweeps, which is not what a plan's risk depends on.
A per-timestep sum has a fixed T terms whatever the plan. This is the same area-overcounting
pathology measured independently in §7c (a correction growing linearly, 21 → 303, with cells
summed).

**A related trap:** aggregating σ under the contact patch with a **max** — the physically
tempting choice, since the settle height *is* a max — **cannot rank plans at all**. Measured
spread of `Σ_t σ_t` across 16 plans: **0.008** with a max against 1.267 with an RMS, because a
saturating σ field pins the max to the cap under every footprint. Use RMS or mean.

### 2.9 Adding a risk term helps  · `risk.py`, n = 150

`step − none` = −0.872 regret, 36/47, **p = 3.5×10⁻⁴**. And the Jensen bias is first-order real:
**E[J] − J(belief) = +1.36**, so a mean-map planner is systematically optimistic before any risk
term is considered.

### 2.10 It fits in a control tick  · `local_contact_bench.py`

The curvature diagonal for 256 cells: 1180 ms serial → 39.5 ms batched → 7.4 ms with a windowed
arg-max refresh → **4.6 ms** with per-cell restore. Bit-identical throughout. (Affordability is
established; §3.4 says it is not worth spending.)

---

## 3. Refuted — with mechanisms

Each of these was a plausible fix. Each was tested and failed. The mechanism matters more than
the verdict, because it says what *not* to try next.

**3.1 Sub-cell contact refinement** (`subcell_eval.py`). Forward: a clear win — 44× more accurate
envelope, bias −0.172 → −0.004 mm. **Gradient: no.** Refinement shrinks each contact jump but
multiplies their frequency; the maximiser is still quantized, now to 1/8 cell. And even with a
*continuum* maximiser, `∂env/∂δ` sweeps its full 0 → 1 range over ~7.5 mm. A gradient that
traverses its entire range inside one σ is not a differentiability problem — the function is
smooth and strongly **curved**, and smoothing does not make a function linear.
*Cost of finding out: ~70 s, against the week it would have taken to build into the tiled kernel.*

**3.2 An analytic Hessian** (`hessian_split.py`). With the arg-max frozen the envelope is
*linear* in h, so a frozen-contact analytic Hessian contributes **exactly zero** from the
dilation. Against the true curvature it is 0.4% (flat), 1.2% (slope), 3.0% (curb), 35% (rock) —
two orders too small nearly everywhere. Rock is the cross-check: the one region with genuine
geometric terrain curvature.

**3.3 Cap-pooling the attribution score** — *partially* supported, after a correction. At the
correct radius, pooling to the contact support **helps at small budgets** (`magn_pooled −
magnitude` +0.057 @25, p = 0.014) and hurts at large ones (−0.061 @400). A crossover, not a
refutation. (First reported as refuted; that run pooled over 35 cells instead of 3.5 — see §4.1.)

**3.4 Second-order FOSM for decisions** (`second_order.py`, `order2.py`). §4 showed it cuts
badly-wrong variance estimates 34.1% → 3.2% with bias correlation r = 0.983. It does **not**
improve decisions, by any of three routes:
- *bias correction, no sensing*: τ +0.121 → +0.056, p = 0.57. The correction spreads the plans by
  **303** against a true cost spread of **6.5**, and its magnitude grows **linearly** in cells
  summed (21 → 34 → 77 → 226 → 303) — per-cell terms added as if independent through a max.
- *a truncation sweep found an interior optimum* (+0.087 → +0.161 at 100 cells) that **did not
  survive**: fixed in advance and run once on disjoint seeds, τ +0.129 → **+0.029** (p = 0.027
  *worse*).
- *as a risk estimator*: −0.054, p = 0.13, and calibration **degrades** 0.85 → 2.38.

So §4 and §5 stand as measured — the correction *is* more accurate and *does* fit in a tick — but
variance accuracy was the wrong objective if the decision is what matters.

**3.5 Soft contact gradients** — a real effect, too small to matter (`softgrad.py`). The correct
derivative under uncertainty is `∂E[max]/∂h = P(q is the arg-max)`, a soft distribution, so a
softmax with temperature **set by σ rather than tuned**. Only the envelope-mediated term may be
softened, and that is correctness not choice: a one-hot re-contraction reproduces the hard adjoint
**exactly** for `settle` (0.0000) and fails for `clear_soft` (1.87), because the belly samples the
raw heightmap bilinearly and has no max in its path. Result: **+0.006 τ, 43/66, p = 0.019** —
real, flat across 0.25σ–4σ, and nowhere near enough to undo the inversion.

**3.6 Single-plan attribution** — the form the closed-loop benchmark deployed. It loses to its own
geometric shadow (`swath_best`, the same corridor with no derivative) at 400 cells: **−0.187,
26/196, p = 4×10⁻²⁷**. Attribution must be computed over the *candidate set*, not the incumbent
plan. Relatedly, variance-across-plans adds nothing over raw summed sensitivity (p = 0.51) — the
elegant "only cells that can reorder plans matter" argument is not what does the work.

**3.7 The closed-loop benchmark as posed** (§6 of `RESULTS.md`). Underpowered (one claim
supported at p = 0.007; variant B's apparent result did not survive n = 12 → 32), and scoped to a
question the method cannot answer: per-cell Gaussian σ cannot represent *topology* — "is there a
way through" — only cost-shaping. The matched cost-decision variant is blocked by the planner
itself: with the full map, the **oracle chose the rough channel**, because cost-to-go max-pools
terrain at 0.24 m and the MPPI horizon is ~2.5 m, so route choice is not roughness-sensitive.

---

## 4. Corrections made during the study

Recorded because several are the same class of error, and because two headline numbers changed.

**4.1 `env_radius` units.** It is already in **cells** (`ceil(wheel_radius / cell_size)`), and
the pooling code divided it by the cell size again — pooling over 35 cells instead of 3.5. This
invalidated the claim that cap-pooling "made things worse" (§3.3). No other result used it.

**4.2 An oracle leak in the ranking experiment.** `Harness.adjoint` resets the terrain to the
*scene's* elevation; the scene had been built on ground truth, so every gradient policy was
differentiating at the answer. Worth about a third of the effect (`disagreement`@400 0.823 →
0.648). Fixed by building the scene on the belief.

**4.3 A confounded 2×2.** §7g attributed a risk-ordering inversion to gradient weighting at
p = 6×10⁻¹¹. Its "unweighted" rows summed over **timesteps** while its "weighted" rows summed
over **cells**, so the aggregation domain varied too. With the missing cell measured, weighting is
**neutral** (p = 1.00) and the domain is the whole effect (p = 3.6×10⁻⁸). This was the exact
confounding error that section was written to expose.

**4.4 An uninterpretable ratio.** §7f reported `est/true risk = 1.28` for the path-σ arm. `Σ_t σ_t`
has units of metres×timesteps, not cost, so its scale is accidental. Only FOSM's 2.04 is a genuine
cost-unit over-prediction.

**4.5 A silent no-op that nearly became a result.** The first `--flat-sigma` run returned numbers
*bit-identical* to the leaked run, including entropy — which under a flat σ must collapse onto
random. `black` had joined the `build_belief` call onto one line, so the edit threading the
argument through never matched. Caught only because identical-to-three-decimals across five
policies is impossible.

**4.6 A latent trap in the harness.** Only `_reset_terrain` loaded friction, so any caller driving
the terrain itself and calling `forward()` rolled out on μ = 0 and got a **NaN pose behind a
plausible-looking settle**. Friction is now loaded at construction.

**4.7 Two small-n reversals.** §6 variant B did not survive n = 12 → 32. §7f's FOSM advantage did
not survive n = 40 → 150 (3.741 < 4.315 became 4.002 < 4.363). Nothing below n ≈ 100 is reported.

---

## 5. Not established

- **No real data.** Every number is synthetic, one robot geometry, and σ is the placeholder the
  plan asked for. Absolute breakdown scales are therefore not meaningful — only curve shapes and
  contrasts. **This is the single biggest weakness.**
- **Ranking is not driving.** §7's results are open-loop over a fixed plan set: no commitment, no
  re-planning, no cost of looking. §6 is the standing evidence that this gap is real.
- **The plan sets are synthetic fans**, not MPPI elite sets, and the cost is `settle + clear_soft`
  rather than the live planner cost.
- **Reveals are modelled as perfect** in the clean arm. Under noise they deliver a measurement,
  which is right — but the magnitudes in the clean arm are optimistic.
- **`cvar` in §6 is not an independent baseline** — as implemented it is a sampling estimator of
  the same sensitivity attribution uses.
- **Study C (C8) not started**, and the ProTerrain / related-work pass that
  `SENSITIVITY_PLAN.md` §3 asks for has **not been done**.
- **The domain rule (§2.8) has been tested on one cost and one robot.** It is mechanistically
  clear and reproduced by three independent routes, but not shown to generalise.

---

## 6. Reproduction

```
.venv/bin/python -m studies.adjoint.study_a                 # ~1 min
.venv/bin/python -m studies.adjoint.study_b                 # ~3 min
.venv/bin/python -m studies.adjoint.subcell_eval            # ~70 s
.venv/bin/python -m studies.adjoint.curvature_eval          # ~2 min
.venv/bin/python -m studies.adjoint.hessian_split
.venv/bin/python -m studies.adjoint.generalise
.venv/bin/python -m studies.adjoint.local_contact_bench

.venv/bin/python -m studies.bench.verify_noise              # gates; must pass first
.venv/bin/python -m studies.bench.softgrad --gate           # gates; must pass first
for f in fan hybrid speed; do
    .venv/bin/python -m studies.bench.ranking --family $f --seeds 200; done
for n in clean sensor localisation occlusion all; do
    .venv/bin/python -m studies.bench.ranking --family hybrid --noise $n --seeds 200; done
.venv/bin/python -m studies.bench.ranking --family hybrid --noise all --flat-sigma --seeds 200
.venv/bin/python -m studies.bench.compare_families
.venv/bin/python -m studies.bench.compare_noise
.venv/bin/python -m studies.bench.illustrate --seed 7       # what each policy looks at
.venv/bin/python -m studies.bench.risk --seeds 150 --family hybrid --noise all      # ~45 min
.venv/bin/python -m studies.bench.softgrad --seeds 80 --family hybrid --noise all
.venv/bin/python -m studies.bench.order2 --seeds 40 --family hybrid --noise all     # ~25 min
```

**Figures:** `studies/out/bench/methods_seed7.png` (what each policy looks at — the clearest
single picture), `families.png` (when an adjoint is needed), `noise.png` (what survives realistic
error), `ranking_*.png`, `study_a.png`, `study_b.png`.

**Production changes** are confined to `engine/step.py`, `engine/simulator.py`,
`engine/envelope.py`, plus repairs to two already-broken test files. `rollout_kernel` remains
bit-identical to init+step. Nothing under `ros/` or `bags/` was touched.

---

## 7. What to do next

**The paper this is.** Not "a better planner component" — a **characterisation**: *when can you
propagate map uncertainty analytically through a contact-based traversability model, and when must
you sample?* Lead with the criterion (§2.3) and uncertainty ≠ relevance (§2.4); the refutations
(§3) are the evidence that the criterion is real rather than a modelling artifact.

**The highest-value remaining work is a real σ layer, validated on bags.** It is the one sentence
a reviewer will circle. Most ingredients exist: the heightmap builder already exports
max/mean/min/count per cell, the accumulator carries `_last_seen`, and `confidence/` has occlusion
and support masks. Missing: within-cell variance (add `sumsq` to the rasterize kernel), a
range/incidence term, staleness inflation, and a pose term (check whether the Odin publishes a
populated covariance). Then the validation that costs little and is worth a lot: **drive the same
ground twice and compare predicted σ against the observed disagreement between passes** — a
calibration curve on real data, which also independently tests §2.6.

**Do not build:** second-order attribution in the planner (§3.4), sub-cell refinement for
gradients (§3.1), or the §6 closed-loop benchmark as posed (§3.7).

**Before writing:** the related-work pass `SENSITIVITY_PLAN.md` §3 asks for, now more important
than when the plan was written because the study sits close to STEP (Fan et al., RSS 2021) and
the Cai / EVORA line.
