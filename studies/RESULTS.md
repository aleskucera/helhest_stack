# Adjoint map-cell sensitivity for decision-focused sensing — results

Everything here is reproducible from this repo. Each claim names the script that produces it.
Read `SENSITIVITY_PLAN.md` first for the framing; this document reports what was measured
against it, including the parts that came out against the plan.

**One-line summary.** The IFT adjoint through the quasi-static settle is correct. First-order
attribution is *not* adequate at realistic map uncertainty, and the reason is curvature rather
than non-differentiability — a second-order correction fixes 90% of it, at a cost that fits in
a control tick. In closed loop, decision-focused sensing beats information-theoretic sensing
when the decision-critical feature is opaque, and ties with it when that feature is an
aperture. Statistical power is the weakest part and is stated as such.

---

## 1. Study A — is the adjoint correct?  **PASS**

`studies/adjoint/study_a.py` · `studies/adjoint/README.md` · figure `studies/out/study_a.png`

Finite differences of the Warp forward itself (not of the numpy reference — an adjoint is only
correct *with respect to a forward*), on a four-lane labelled scene with flat / slope / curb /
rock strata. A flat-terrain check would pass vacuously: with `gx = gy = 0` most of the settle
adjoint's terms drop out of the arithmetic entirely.

| | result |
|---|---|
| worst relative L2 error, all strata | **9.5e-3** |
| worst regression slope deviation | **1.9e-3** |
| settle residual (the IFT premise) | 6e-8 m |
| Warp float32 vs numpy float64 forward | 5e-7 m |

The curb is as clean as flat ground once the dilation is taken out — the settle adjoint is not
the fragile part.

**Three findings beyond the gate.**

1. **The frozen-arg-max dilation has a validity radius** of `R − √(R²−cell²)` = 3.6 mm at
   0.05 m cells. The measured kinked fraction goes 0% → 100% exactly as ε crosses it.
2. **`chassis_clearance` had a structurally broken gradient** — a `min` over 18 belly points
   that all share one body-frame z, so on level ground they tie *exactly*. Fixed by returning a
   tie-free hinge sum alongside the min; regression slope 0.30–0.50 → **1.0000**.
3. **`min N_i` cannot serve as the linearisation-validity flag** on this robot. It reaches zero
   only at static tip-over, which on a *tripod* is about a support-triangle edge, not the
   track: nearest boundary 29.5° against a 15° planner limit. Replaced by a contact-margin
   diagnostic emitted from the dilation.

## 2. Study B — is first order adequate?  **NO, and we know exactly when**

`studies/adjoint/study_b.py` · figure `studies/out/study_b.png`

Self-check first: at σ×0.03 the FOSM/Monte-Carlo ratio is 1.00 ± 0.02, validating both the
noise normalisation and the correlated FOSM formula. Every later deviation is physics.

**A crisp, scale-free criterion.** Stratified by a per-source validity radius
(`contact.source_slack` — the margin must be per-*source*-cell; the engine's `contact_margin`
is indexed by output cell and answers a different question):

| σ / slack | median ratio | worse than 2× | n |
|---|---|---|---|
| < 1 | **1.02** | **0.0%** | 265 |
| 1 – 10 | 1.71 | 41.2% | 690 |
| ≥ 10 | 2.61 | 63.7% | 314 |

**The catch:** at realistic σ almost nothing is inside — 3 of 846 probes at σ×1 or above,
because slack is ~3.6 mm while σ is centimetres.

Two supporting results: **i.i.d. per-cell map noise is not a valid model here** (the envelope
is a max over ~37 cells, so zero-mean noise biases it 3.4σ against 0.4σ correlated), and
**per-cell and global adequacy come apart** — the most globally accurate rollout has among the
worst per-cell ratios, so a Var(J) check alone would have been misleadingly reassuring.

**The decoy result:** an unobserved patch holding **67.3% of the map's total σ²** attracts
**0.000%** of the FOSM variance. High uncertainty and high relevance are genuinely different.

## 3. The fix everyone pointed at — refuted

`studies/adjoint/subcell_eval.py`

Studies A and B both implicated contact quantization, so: let the contact slide sub-cell.

- **Forward: a clear win.** 44× more accurate envelope against a continuum reference, bias
  −0.172 → −0.004 mm. Worth having on its own merits.
- **Gradient: no.** Attribution does not improve, and the kinked-fraction cliff moves *in*.

**Why, which matters more than the answer.** Refinement shrinks each contact jump but
multiplies their frequency — the maximiser is still quantized, now to 1/8 cell. And even with a
*continuum* maximiser, `∂env/∂δ` sweeps its full 0 → 1 range over ~7.5 mm. A gradient that
traverses its entire range inside one σ is not a differentiability problem: the function is
smooth and strongly **curved**, and smoothing does not make a function linear.

Cost of finding out: ~70 s of compute, against the week it would have taken to build into the
graph-captured tiled kernel.

## 4. The fix that works — second-order FOSM

`studies/adjoint/curvature_eval.py`

`Var = g²σ² + ½c²σ⁴`, `E[J] − J₀ = ½cσ²`. Fraction of cell-σ pairs whose variance is wrong by
more than 2×:

| stratum | first order | second order |
|---|---|---|
| flat | 32% | **1%** |
| slope | 50% | **1%** |
| curb | 25% | 10% |
| rock | 21% | 18% |
| **overall** | **34.1%** | **3.2%** |

Bias correlation r = **0.983** — a quantity first order cannot express at all. It fails only
in genuinely *tied* cells, which Study B's σ/slack criterion already flags, so the two results
compose: **second-order FOSM everywhere, sampling fallback only where the radius says so.**

**Can the Hessian be analytic instead?** No — measured. `studies/adjoint/hessian_split.py`:
with the arg-max frozen the envelope is *linear* in h, so a frozen-contact analytic Hessian
contributes exactly zero from the dilation. Against the true curvature it is 0.4% (flat), 1.2%
(slope), 3.0% (curb), 35% (rock) — about two orders too small nearly everywhere. Rock is the
cross-check: it is the one region with genuine geometric terrain curvature.

## 5. Does it fit in a control tick?  **Yes**

`studies/adjoint/local_contact_bench.py`

The curvature diagonal needs `J(±δ)` per cell. Per-rollout terrain turns N cells into 2N
slices of one launch; two further optimisations remove the wasted work inside it.

| | 256 cells |
|---|---|
| serial (2 forwards per cell) | 1180 ms |
| batched | 39.5 ms |
| + windowed arg-max refresh | 7.4 ms |
| + per-cell restore | **4.6 ms** |

**8.6× over batched, 256× over serial, and bit-identical** — zero arg-max discrepancies
against a full recompute at every size. On a grid 2.3× *larger* than the real perception grid,
so under 5% of a 100 ms tick with margin.

Only ~2% of the map matters for a committed plan (~500 support cells of 29,161), and at a curb
just 65 cells carry 99% of the predicted variance — which is why probing a few hundred is
correct rather than a shortcut. Probing everything would cost 525 ms.

## 6. The §6 benchmark — decision-focused vs information-theoretic sensing

`studies/bench/` · figure `studies/out/bench/benchmark.png`

Closed loop: plan on an accumulated, occluded, optimistically-inpainted belief; drive on
ground truth. Once per 30 frames the robot may spend 8 frames aiming a long-range narrow-FOV
sensor at a chosen bearing — the budget is **capped at 4 and identical for every policy**, so
the overhead cancels and only the aiming decision is compared.

**Two variants, and the contrast between them is the result.**

| variant | mean time to goal (n=12) | | | |
|---|---|---|---|---|
| | none | entropy | **attribution** | oracle |
| **A** aperture — gap in a wall | 305 | 275 | **271** | 201 |
| **B** opaque — corridor floor | 231 | **255** | **216** | 171 |

In **A** the two tie. A gap is simultaneously the most decision-relevant *and* the most
information-rich thing to look at, because sight passes *through* an aperture — so "where can
I see most" and "where does my plan depend" coincide by construction.

In **B**, where the critical feature is opaque and small, they diverge: entropy-directed
sensing is **actively harmful** (+24 frames against never looking) while attribution-directed
sensing helps (−16). The behavioural mechanism is visible directly — entropy spends 1.5 of its
4 looks on the decoy and 0.4 on the critical cells; attribution spends 1.4 on the critical
cells and **0.0** on the decoy.

**Getting the baseline strong enough to be wrong took three corrections**, each of which would
otherwise have produced a straw man: an area-counting NBV is nearly direction-*indifferent*
(a fixed cone has the same area whichever way it points), so it was switched to σ-weighted
information gain; σ had to be *predicted from context* (roughness measured where observed,
carried into neighbouring unobserved blocks) rather than assumed uniform; and the decoy had to
sit near enough for its roughness to be observable at all.

## 7. What is **not** established

- **Statistical power is the weakest part.** At n = 12 no paired sign test reaches
  significance (p = 0.146–1.0) despite mean effects of 16–39 frames. The direction is
  consistent with the hypothesis; the benchmark as run does **not** establish it. A 32-seed
  run is in progress and this section will be updated with it.
- **No real data.** Every number is synthetic, on one robot geometry and one grid resolution.
  σ is the placeholder the plan asked for, so absolute breakdown scales are not meaningful —
  only curve shapes and contrasts.
- **`cvar` is not an independent baseline.** As implemented it is a sampling estimator of the
  same sensitivity attribution uses, and lands on identical bearings. It should be reimplemented
  as a genuine sample-the-map-and-replan baseline or dropped.
- **The sensitivity used in the loop is a geometric proxy**, not a taped adjoint per candidate
  bearing. It has the adjoint's defining property (support only where the committed plan can
  physically touch, zero on the decoy) but the loop does not exercise the engine's own gradient.
- **Study C (C8) not started**, and the ProTerrain methods comparison (§3) has not been done.
- The second-order curvature is finite-difference-derived at the exact σ — deliberately the
  *ceiling*. §5 shows it is affordable, but affordability is not the same as implemented.

## 8. Reproducing

```
.venv/bin/python -m studies.adjoint.study_a           # ~1 min
.venv/bin/python -m studies.adjoint.study_b           # ~3 min
.venv/bin/python -m studies.adjoint.subcell_eval      # ~70 s
.venv/bin/python -m studies.adjoint.curvature_eval    # ~2 min
.venv/bin/python -m studies.adjoint.hessian_split
.venv/bin/python -m studies.adjoint.local_contact_bench
.venv/bin/python -m studies.bench.verify_world        # gates, must pass before the rest
.venv/bin/python -m studies.bench.verify_policies
.venv/bin/python -m studies.bench.run_bench --seeds 32 --variant gap
.venv/bin/python -m studies.bench.run_bench --seeds 32 --variant corridor
.venv/bin/python -m studies.bench.analyse
```

Production changes made by this work are confined to `engine/step.py`,
`engine/simulator.py`, `engine/envelope.py` (plus repairs to two already-broken test files).
`rollout_kernel` remains bit-identical to init+step.
