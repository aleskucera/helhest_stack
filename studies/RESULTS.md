# Adjoint map-cell sensitivity for decision-focused sensing — results

Everything here is reproducible from this repo. Each claim names the script that produces it.
Read `SENSITIVITY_PLAN.md` first for the framing; this document reports what was measured
against it, including the parts that came out against the plan.

**One-line summary.** The IFT adjoint through the quasi-static settle is correct. First-order
attribution is *not* adequate at realistic map uncertainty, and the reason is curvature rather
than non-differentiability — a second-order correction fixes 90% of it, at a cost that fits in
a control tick. Decision-focused sensing beats information-theoretic sensing convincingly —
in open-loop plan ranking at n = 200, entropy is indistinguishable from random while the
adjoint recovers the true ranking (§7). But most of that margin is *task-awareness*, not the
derivative: "reveal what you are about to drive over" is as good as the adjoint at small
sensing budgets and is only overtaken at large ones. The closed-loop benchmark (§6) is weaker
and its statistical power is the weakest part of the whole study; it is stated as such.

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
   (C5) would keep the wins and drop the losses. **It did not — see §6c.** The actual cause is
   §6d: single-plan attribution is confirmatory, and no gating rule repairs that.
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

### 6d. Why no policy beats abstention — the confirmatory loop

`studies/bench/diagnose_confirmatory.py`

Attribution loses ~61 frames on easy seeds against a look budget of only 32, and the extra 30
come from **driving 2.5 m further**. Looking makes the route worse, not merely slower. On 9 of
17 easy seeds it finds the gap *later* than doing nothing (seed 27: frame 160 vs 75).

On seed 27 the gap is at bearing **−42°** from the start. Attribution's four looks go
**[−3°, +13°, +63°, −86°]** — the first three straight ahead or at the opposite side. The robot
drives to y = **+4.7 m**, the wrong way.

**The mechanism.** Attribution aims along the *current plan's* route, computed on a belief
where unknown = flat = passable. Look 1 reveals the wall on that route → the cost-to-go
reroutes toward the nearest apparently-free edge, which is merely the boundary of what has been
observed, an artifact of sensing rather than of the world → attribution aims along the *new*
route, confirming more wall on the side it just committed to → repeat. The policy chases its
own commitment. The null baseline, never looking, sweeps broadly with its 180° short-range
sensing and stumbles onto the gap.

> **Naive decision-focused sensing is confirmatory, not exploratory.** It looks where the plan
> already goes, which tends to *confirm* the plan's assumptions rather than test them. In a
> routing problem the informative look is at the alternative the plan **rejected** — precisely
> where sensitivity is low under the current plan. `∂J/∂h` for a single committed plan cannot
> see it, by construction.

**This is an implementation error, not a flaw in the idea.** `SENSITIVITY_PLAN.md` §1 specifies
attribution over *"the elite **set**'s cost variance"* — plural. Attributing over one committed
plan is what produces the loop. The fix follows directly: attribute over the elite set and look
where the plans **disagree**. A cell every candidate route agrees about is worthless to observe
however sensitive the chosen plan is to it; a cell that separates the top routes is worth
everything. **Untested** — it is the highest-value open experiment.

Consequences for reading §6: attribution > entropy (p = 0.007) stands and is if anything
*understated*, since it won while handicapped by this loop. "Beats entropy but not abstention"
should be read as a property of **single-plan** attribution specifically.

### 6e. Why the benchmark cannot be won as posed — the method's actual scope

Trying to build the experiment where attribution beats *both* entropy and abstention produced
the most useful result of the benchmark effort, and it is a scoping result.

**Per-cell Gaussian σ cannot represent a routing alternative.** Disagreement-based attribution
needs candidate plans that differ. Traced elite routes from one cost-to-go field do not differ
(0.54 m spread at any softmin temperature — a distance-like field has an essentially unique
geodesic). Sampling maps from the belief and re-planning on each gives the alternatives in
principle, but measured:

| σ over unobserved cells | routes traced | path spread |
|---|---|---|
| 0.12 m | 8 | 0.55 m — all straight |
| 0.30 m | **1** | — |
| 0.50 / 0.80 m | **none** | — |

At small σ no sampled map ever contains the 0.8 m barrier, so every sampled plan goes straight
and there is nothing to disagree about. At obstacle-scale σ the sampled maps are **rubble
everywhere** and no route exists at all. A wall is a coherent 10 m object; smoothed per-cell
noise is gravel. Optimistic inpainting compounds it: under "unknown = flat = passable" the
planner is not uncertain, it is confidently wrong.

> **Scope.** `∂J/∂h · σ` attribution is well-posed for **cost-shaping** uncertainty — ride
> roughness, tilt, clearance margin — where the route is fixed and the cost wobbles. It is
> **not** well-posed for **feasibility/topology** uncertainty — is there a way through — where
> the plan itself changes and a per-cell Gaussian cannot express the alternatives.

Variants A and B are both topology tasks. **The benchmark was built for a question the method
is structurally unsuited to**, which explains the confirmatory loop (§6d), entropy's strong
showing, and the failure to beat abstention.

### 6f. The matched benchmark, attempted and not delivered

`world.build_lanes` — two open channels, one rough beyond the point of commitment: a pure cost
decision, which is what the method is for. It does **not** yet work, and the blocker is not the
scenario:

**With the full map, the oracle chose the ROUGH channel** (372 frames) while the ignorant
baseline took the smooth one (165). Perfect information made it worse. The cost-to-go
max-pools terrain at 0.24 m and the MPPI horizon is ~2.5 m, so **neither selects routes on
terrain roughness** — "which of two open routes is cheaper" is not a decision this planner
makes sharply.

That is a property of the planner's cost, not of the scenario, and tuning the scenario around
it would manufacture a result rather than measure one. Making the matched benchmark work
requires making route selection genuinely roughness-sensitive first — a change to
`costtogo.py`'s traversability cost, deliberately not made here.

**What would be needed for the experiment to be winnable**, stated so the next attempt does
not rediscover it:
1. a planner whose route choice is sensitive to the terrain property being sensed;
2. a belief that can represent the alternatives — for topology that means object-level or
   occupancy uncertainty, not per-cell height σ;
3. headroom (oracle vs abstention) larger than the look budget — 20 frames against a 32-frame
   budget cannot pay, whatever the policy does.

## 7. The open-loop ranking experiment — the claim tested without a closed loop

`studies/bench/ranking.py`, n = 200 seeds, ~90 s. Figure: `studies/out/bench/ranking.png`.

§6 could not settle C4's claim because the closed loop entangled it with route selection, look
budgets, commitment dynamics, and the confirmatory bug of §6d. So the claim is posed directly,
as the thing a sampling planner actually does with a cost — **ranking**:

random fractal terrain (12 cm RMS, 0.1 m cells); the robot has observed only a 1.5 m disc and
inpaints the rest optimistically flat; 16 candidate plans fan out into the unknown. Each policy
reveals M cells; the plans are re-ranked on the updated belief and scored by **Kendall τ against
the ranking on ground truth**, plus top-1 accuracy and the true regret of the plan a planner
would pick. Gradients are the **real taped adjoint** from `DifferentiableSimulator` — the
engine's own derivative is on trial, not the geometric proxy §7's caveat list flags for §6.

Before sensing: τ = +0.035, top-1 correct 11 %. The belief's ranking is essentially uninformative.

| policy | what it knows | τ@100 | τ@400 | top-1@400 | regret@400 |
|---|---|---|---|---|---|
| random | nothing | +0.025 | +0.040 | 12 % | 2.53 |
| entropy | σ only | +0.017 | +0.015 | 12 % | 2.43 |
| swath_best | geometry of the incumbent plan | +0.093 | +0.330 | 36 % | 0.93 |
| attribution | ∂J/∂h of the incumbent plan | +0.095 | +0.142 | 16 % | 1.85 |
| swath | geometry of the whole plan set | **+0.197** | +0.556 | 45 % | 0.57 |
| disagreement | Var_k(∂J_k/∂h)·σ² over the set | +0.163 | **+0.648** | 56 % | 0.38 |
| *oracle* | *the actual belief error* | *+0.397* | *+0.847* | *74 %* | *0.13* |

### What this establishes

**1. Decision-focused beats information-theoretic, decisively.** `disagreement` − `entropy`:
+0.146 at 100 cells (150/199 seeds, p = 4×10⁻¹³) and +0.633 at 400 (**200/200 seeds**,
p = 1×10⁻⁶⁰). This is the §6 claim, now at n = 200 instead of n = 32 and with the real adjoint.

**2. Entropy is statistically indistinguishable from random** (p = 0.28 at 400 cells) and does
not improve with budget at all. σ here is a property of the ground (roughness-driven), so
entropy targets genuinely uncertain cells — they are simply not the cells that decide anything.
Panel (c) is the whole story: entropy and random reveal cells at 4.9 m mean range and 2.0 m off
the nearest plan; every task-aware policy sits at ~2.2 m range and ~0.05 m off-plan.

**3. But most of that win is *task-awareness*, not the derivative.** Purely geometric
"reveal what you are about to drive over" (`swath`) also crushes entropy (+0.180 at 100 cells,
p = 2×10⁻²⁰) — and at 100 cells it **beats** the adjoint (`disagreement` − `swath` = −0.034,
p = 0.027). The adjoint only overtakes geometry once the budget covers the corridor:
+0.092 at 400 cells (131/193, p = 8×10⁻⁷) against `swath`, +0.141 (p = 5×10⁻¹⁶) against
`swath_var`. **The honest claim is a crossover, not a dominance.**

**4. Single-plan attribution — the form §6 deployed — is the weak version, and the deficit is
not coverage.** Matched against its own geometric shadow (`swath_best`: the same corridor, no
derivative), it ties at 25 and 100 cells and then **loses badly** at 400: −0.187, 26/196 seeds,
p = 4×10⁻²⁷. It saturates at τ 0.142 while every plan-set score climbs past 0.6. This is §6d's
confirmatory loop showing up in open loop, and it is the concrete lesson: **attribution must be
computed over the candidate set, not the incumbent plan.**

**5. Discriminativeness adds nothing over raw sensitivity.** `disagreement` (variance across
plans) and `magnitude` (Σ|∂J_k/∂h|) are indistinguishable at every budget (p = 0.51 at 100 and
400). The elegant "only cells that can reorder the plans matter" argument is not what is doing
the work — having *any* per-plan gradient is.

### What was ruled out along the way

- **An oracle leak.** `Harness.adjoint` resets the terrain to the *scene's* elevation; building
  the scene on ground truth silently differentiated at the answer. Fixed by building the scene
  on the belief. It was worth roughly a third of the effect — `disagreement`@400 fell 0.823 → 0.648.
- **A rigged entropy baseline.** With a binary σ, entropy has no preference among unobserved
  cells and its "choice" is the argsort's tie order — it scored *below* random. Fixed with a
  spatially structured σ and a random tie-break applied to every policy.
- **The "inert reveals" hypothesis.** Since the envelope is a max over a spherical cap,
  revealing one gradient-hot cell among unrevealed neighbours might change nothing. Measured
  directly (`rch`: envelope cells moved per cell revealed) and **refuted** — `random` has the
  *highest* reach (7.8) and the *worst* τ. Cap-pooling the score, which the hypothesis implies,
  made things worse (0.163 → 0.104 at 100 cells).
- **`Harness.forward` on uninitialised friction.** Any caller that drove the terrain itself and
  called `forward` rolled out on μ = 0 and got a NaN pose with a plausible-looking settle. Only
  `_reset_terrain` loaded friction; it is now loaded at construction.

### What it does not establish

It is open-loop: no commitment, no re-planning, no cost of looking. A policy that ranks a fixed
plan set well need not drive better — §6 is the evidence that the gap is real. The plan set is a
fixed fan rather than an MPPI elite set, the cost is `settle + clear_soft` rather than the live
planner cost, and the terrain is still synthetic with a placeholder σ.

## 8. What is **not** established

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
  §7 closes this for the *open-loop* claim only — and finds that a geometric proxy is in fact
  competitive with the real adjoint below 400 revealed cells, so the §6 loop's use of a proxy
  is now known to be a smaller compromise than it looked, and its results correspondingly less
  attributable to the adjoint.
- **Ranking is not driving.** §7's win is over a fixed plan set with no commitment, no
  re-planning and no cost of looking. §6 is the standing evidence that this gap is real.
- **Study C (C8) not started**, and the ProTerrain methods comparison (§3) has not been done.
- The second-order curvature is finite-difference-derived at the exact σ — deliberately the
  *ceiling*. §5 shows it is affordable, but affordability is not the same as implemented.

## 9. Reproducing

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
.venv/bin/python -m studies.bench.ranking       # open-loop ranking, n=200, ~90 s
```

Production changes made by this work are confined to `engine/step.py`,
`engine/simulator.py`, `engine/envelope.py` (plus repairs to two already-broken test files).
`rollout_kernel` remains bit-identical to init+step.
