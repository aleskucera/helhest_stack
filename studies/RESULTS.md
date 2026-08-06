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

## 4b. Does the criterion transfer?  **Yes — but the 3.6 mm figure does not**

`studies/adjoint/generalise.py`

Re-run at the real perception resolution (0.10 m) and on fractal 1/f terrain with ~12 cm RMS
relief, i.e. a completely different spectrum from the study scene's analytic primitives.

| cell | flat-ground prediction | measured slack | inside radius: bad | outside: bad |
|---|---|---|---|---|
| 0.05 m | 3.6 mm | **16.5 mm** | 1% | 14% |
| 0.10 m | 14.6 mm | **10.4 mm** | 0% | 15% |

**The criterion transfers**: inside the radius attribution stays accurate (median ratio 1.00
and 0.97), outside it degrades — on terrain the criterion was never tuned against.

**The radius does not follow the formula.** On broadband terrain the contact is decided by the
ground's *own* relief, not by the spherical cap's step between neighbouring offsets. So
`R − √(R²−cell²)` is a **flat-ground special case**, and at fine resolutions a pessimistic
one — rough ground determines its contact *more* decisively than flat ground, which is the
opposite of the intuition the 3.6 mm figure invites. This corrects the reach of §1's finding 1
without changing the practical conclusion: at 10–17 mm, centimetre-scale σ is still outside
the radius, so second order is still needed.

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

`studies/bench/` · figure `studies/out/bench/benchmark.png` · **n = 32 seeds per variant**

Closed loop: plan on an accumulated, occluded, optimistically-inpainted belief; drive on
ground truth. Once per 30 frames the robot may spend 8 frames aiming a long-range narrow-FOV
sensor at a chosen bearing. The budget is **capped at 4 and identical for every policy**, so
the overhead cancels and only the aiming decision is compared.

### The one claim that is statistically supported

**Attribution beats entropy** in variant A: 24/32 seeds, mean −22 frames, **sign test
p = 0.007**. The behavioural mechanism is directly visible — entropy spends 1.66 of its 4 looks
on the decoy and 0.47 on the critical cells; attribution spends 0.88 on the critical cells and
**0.00** on the decoy.

| variant A (aperture), n=32 | mean | hard | easy | vs none | headroom | reached |
|---|---|---|---|---|---|---|
| none | 315 | 424 | 220 | — | 0% | 32/32 |
| sigma | 394 | 530 | 273 | +78 | −68% | **26/32** |
| entropy | 292 | 330 | 258 | −23 | 20% | 32/32 |
| **attribution** | **270** | **258** | 281 | **−45** | **39%** | 32/32 |
| oracle | 200 | 201 | 199 | | | |

`sigma` — uncertainty-weighted *without* decision-weighting — is the worst arm and fails to
reach the goal on 6/32 seeds. Weighting by uncertainty alone is worse than not looking.

### What did **not** survive going from n=12 to n=32

At n = 12 variant B appeared to show the cleanest result: entropy actively harmful (+24 frames
against never looking) while attribution helped (−16). **That did not reproduce.** At n = 32:

| variant B (opaque), n=32 | mean | vs none | sign test vs none |
|---|---|---|---|
| none | 213 | — | — |
| entropy | 242 | +30 | — |
| **attribution** | **228** | **+16** | 5/32 wins, **p = 0.000 — significantly worse** |

Attribution vs entropy in B is 11/25 wins, p = 0.69 — directionally right, not significant.
The cause is visible in the regime split: only **5 of 32** seeds turned out hard (null baseline
> 1.5× oracle), because the corridor's detour is cheap to find once discovered. Variant B is a
weaker scenario than intended, and its n=12 result was small-sample noise. I am recording this
rather than quietly keeping the n=12 numbers.

### The honest summary

1. **Attribution > entropy** — supported in A (p = 0.007), directionally consistent but not
   significant in B.
2. **Neither beats never-looking on a per-seed basis.** In A attribution's *mean* is 45 frames
   better, but it wins on only 15/32 seeds: large wins where the default guesses wrong, small
   losses everywhere else. In B it is significantly worse.
3. The looks are spent **unconditionally**. I predicted that deciding *whether* to observe
   (C5) would keep the wins and drop the losses. **It did not — see §6c.**
4. Variant A's structural finding stands: where the critical feature is an **aperture**, sight
   passes through it, so "where can I see most" and "where does my plan depend" partly
   coincide — which is why entropy still recovers 20% of the headroom there.

### 6c. C5 — deciding whether to look. Implemented, measured, rejected

`studies/bench/compare_c5.py` · gated runs preserved as `results_*_c5.json`

Each policy skips its look when the best bearing would resolve less than 25% of its *own*
objective's total — one threshold, every arm, against its own total, chosen a priori and not
swept. Both variants re-run at n = 32.

| variant A | mean before | mean after | looks before | looks after |
|---|---|---|---|---|
| entropy | 292 | **326** (+34) | 4.00 | 1.84 |
| attribution | 270 | 273 (+3) | 4.00 | **4.00** |
| sigma | 394 | 303 (−91) | 4.00 | 4.00 |

**It does not work, and the reason is instructive.** The gate **never binds for attribution** —
its objective is concentrated exactly where a route-directed look resolves it, so the
resolvable fraction is always high. It binds far too aggressively for entropy, whose objective
is diffuse, cutting its looks by more than half and making it 34 frames *worse*. And it
weakened the one supported result: attribution vs entropy fell from 24/32 (p = 0.007) to
20/32 (p = 0.215).

**What this says about C5.** "What fraction of my objective could this resolve" is *not* value
of information. VoI asks whether the observation would **change the decision** — a different
and harder quantity than how much variance it removes. A cell can carry most of the plan's
predicted variance and still be worth nothing to observe if every outcome leads to the same
route. Implementing C5 properly means estimating the decision change, not the variance change.

The ungated results are therefore the primary ones; `LOOK_THRESHOLD` is set to 0 and the gated
run is kept for the record. **This is the third of my own proposed fixes that measurement
refuted** (sub-cell refinement, the analytic Hessian, and now this), which is the methodology
working rather than failing.

## 7. What is **not** established

- **Only one benchmark claim is statistically supported** (attribution > entropy in variant A,
  p = 0.007). Everything else is directional. Going from n = 12 to n = 32 overturned variant
  B's apparent result, which is a warning about how little 12 closed-loop seeds establish.
- **No policy beats never-looking per-seed**, because the look budget is spent
  unconditionally. C5 (deciding *whether* to observe) is unimplemented and is the obvious gap.
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
