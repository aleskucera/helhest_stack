# Adjoint map-cell sensitivity for decision-focused sensing — results

> **Read `FINDINGS.md` first.** It states every result once, in its final corrected form,
> organised by what we believe rather than by the order it was found. This document is the
> detailed record: it keeps the derivation history and the in-place correction banners, so
> where the two differ, `FINDINGS.md` is current.

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
sensing budgets and is only overtaken at large ones. Varying how much the candidate plans
overlap locates where an adjoint is actually required (7b) -- and a matched test finds the
second-order correction of section 4 does NOT improve the decision, by either available
route (7c). Injecting all three real map-error sources -- sensor, occlusion and pose --
leaves the sigma-is-not-relevance claim intact but ERASES the adjoint's edge over a plain
distance transform, because pose error halves what any sensing could achieve and no
cell-selection strategy can recover it (7d). Tested on the OTHER use of the same machinery --
estimating a plan's risk for a CVaR cost, against Monte-Carlo truth and the STEP baseline --
the adjoint does not win there either (7f): adding a risk term clearly helps (p = 3e-04), but
first-order FOSM is twice as over-conservative as a per-timestep Gaussian heuristic and makes
no better decisions than it. Pulling that apart gives the study's unifying result (7g): the
adjoint's SUPPORT -- which cells a plan can respond to -- is correct and useful, but its
MAGNITUDE is not, at realistic sigma -- the adjoint FD-checks to 2.3% at 1 mm and 32% at 1 cm,
while sigma is 2-30 cm. Soft contact gradients (7h), the derivative of the EXPECTED envelope,
help measurably (p = 0.019) but only slightly. Testing them also corrected 7g's attribution:
what inverts a risk score is summing over CELLS rather than over TIME (a 0.229 swing), not
gradient weighting (neutral, p = 0.43) -- confirmed paired in 7i (domain -0.270, p = 4e-08;
weighting -0.000, p = 1.00). Second-order FOSM does NOT rescue the risk estimate either
(7i, p = 0.13, calibration degrading 0.85 -> 2.38). Every analytic route lands between
tau = -0.19 and -0.13 while a free per-timestep sigma sum lands at +0.14, so the transferable
result is an aggregation rule -- sum risk over TIME, not over CELLS. The closed-loop benchmark
(§6) is the weakest part of the study and is stated as such.

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

**2. Entropy delivers no measurable value over not sensing at all.** Paired against each seed's
own no-sensing τ, `random` is indistinguishable from doing nothing at every budget (p ≥ 0.12)
and `entropy` likewise (−0.020 at 400 cells, p = 0.015 — not a claim worth making across 12
comparisons), while `swath` gains +0.521 and `disagreement` +0.613 from the same budget. The
mechanism is simple rather than interesting: their cells land ~2.0 m off the nearest plan, where
the adjoint is *exactly* zero, so revealing them cannot move any plan's cost. The more
interesting explanation — that partial sensing actively hurts by breaking the common-mode
cancellation of the inpaint bias — was tested and **not supported** (correlation between how
close entropy's cells fell to a plan and its Δτ is −0.03).

σ here is a property of the ground (roughness-driven), so entropy targets genuinely uncertain
cells — they are simply not the cells that decide anything.
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
- **The "inert reveals" hypothesis** — *partly* supported, after a correction. Since the
  envelope is a max over a spherical cap, revealing one gradient-hot cell among unrevealed
  neighbours might change nothing. The `rch` diagnostic (envelope cells moved per cell revealed)
  does **not** support it: `random` has the *highest* reach (7.8) and the *worst* τ, so what
  matters is moving the *right* envelope cells, not more of them.

  **Correction.** This was first reported as fully refuted, on the grounds that cap-pooling the
  score made things worse. That run was invalid: `sim.env_radius` is already in **cells**
  (`ceil(wheel_radius / cell_size)`), and the code divided it by the cell size again, pooling
  over a 35-cell radius instead of 3.5. At the correct radius the conclusion **reverses at small
  budgets** — pooling helps exactly where the hypothesis says it should, when the budget is too
  small to cover a cell's neighbourhood:

  | | @25 | @100 | @400 |
  |---|---|---|---|
  | `magn_pooled − magnitude` | **+0.057** (p=0.014) | +0.002 (p=0.83) | −0.061 (p=2e-11) |
  | `disag_pooled − disagreement` | +0.029 (p=0.072) | −0.034 (p=0.14) | −0.057 (p=6e-11) |

  So the honest statement is a crossover, not a refutation: pooling to the contact support helps
  at small budgets and hurts at large ones, where the budget already covers the neighbourhood and
  pooling only blurs the score. Nothing else in §7 used `env_cells`, so no other number moves.
- **`Harness.forward` on uninitialised friction.** Any caller that drove the terrain itself and
  called `forward` rolled out on μ = 0 and got a NaN pose with a plausible-looking settle. Only
  `_reset_terrain` loaded friction; it is now loaded at construction.

### What it does not establish

It is open-loop: no commitment, no re-planning, no cost of looking. A policy that ranks a fixed
plan set well need not drive better — §6 is the evidence that the gap is real. The plan set is a
fixed fan rather than an MPPI elite set, the cost is `settle + clear_soft` rather than the live
planner cost, and the terrain is still synthetic with a placeholder σ.

### 7b. When is an adjoint actually needed?  **Only when plans overlap but do not coincide**

`studies/bench/ranking.py --family {fan,hybrid,speed}` · `compare_families.py` · n = 200 each.

§7's headline weakness was scenario design, not method: a fan of arcs makes *"which plan wins"*
and *"which cells does it cross"* nearly the same question, so a distance transform answers it
without a derivative. Three plan families now vary **only** how much the plans' coverage
overlaps, holding terrain, σ, budgets, policies and seeds fixed.

| family | plans | endpoints apart | coverage spread | τ@400 adjoint | τ@400 geometry | **adjoint − geometry** |
|---|---|---|---|---|---|---|
| `fan` | 16 separate corridors | 1.130 m | 0.0390 | 0.648 | 0.556 | **+0.092** |
| `hybrid` | 4 paths × 4 speed profiles | 1.069 m | 0.0346 | 0.819 | 0.639 | **+0.180** |
| `speed` | 1 shared path, 16 profiles | 0.001 m | 0.0002 | 0.931 | **1.000** | **−0.069** |

All three differences pass a sign test at p < 0.01. The construction is verified rather than
asserted: `speed`'s plans finish 1 mm apart and its coverage spread is 200× smaller than
`fan`'s, because scaling both wheels by a common factor retraces the identical path.

**The relationship is not monotonic, and the endpoint is the interesting part.** Confounding the
plans *completely* makes the problem **easier** for geometry, not harder: when all 16 candidates
share one path the decision-relevant terrain collapses to a single narrow corridor, 400 cells
covers all of it, and `swath` reaches **τ = 1.000** — perfect. This was the opposite of my
prediction and it is a clean scope statement: *if your candidates all follow one path, do not
compute an adjoint; sense the corridor.*

The adjoint pays in the middle case, where candidates span a broad area **and** overlap within
it — which is exactly the MPPI elite-set regime. `hybrid` doubles the advantage over `fan`.

**The sharpest result is the within-group ordering.** In `hybrid`, four plans share each path
exactly, so every geometric score makes identical reveals for all four and cannot rank them by
construction. On that component alone, at 400 cells:

| | τ within groups |
|---|---|
| no sensing | +0.133 |
| entropy | +0.185 |
| swath (geometry) | +0.672 |
| **disagreement (adjoint)** | **+0.869** |
| *oracle* | *+0.935* |

`disagreement − swath` = **+0.198, better on 161/185 seeds, p = 4×10⁻²⁶**, and the adjoint
recovers 92% of the oracle's within-group ordering.

**The budget crossover survives everywhere.** Geometry still wins at 25 and 100 cells in both
`fan` (−0.006, −0.034) and `hybrid` (−0.031, −0.072). Attribution's value is *efficiency inside
the relevant region*, and it only shows once the budget is large enough to have to prioritise
within that region rather than merely find it.

### 7c. Does the second-order correction improve the DECISION?  **No — by either route**

`studies/bench/second_order.py`, `hybrid` family. §4 showed second-order FOSM cuts badly-wrong
variance estimates 34.1% → 3.2%; §5 showed it fits in a control tick. Neither shows it makes a
better *decision*, which is the only thing a planner can use. Curvature can enter two ways, and
both were tested.

**Route 1 — bias correction, requiring no sensing at all.** `E[J_k] − J_k(belief) = ½ Σᵢ c_ki σᵢ²`
is *per-plan*, so unlike a common offset it moves the ranking. Measured (n = 50): τ +0.121 →
+0.056, p = 0.57. No effect — and one number gave the reason away: **the correction spreads the
plans by 303 against a true cost spread of 6.5.**

Chasing that spread reproduced a Study B result exactly. The correction's magnitude grows
**linearly** in the number of cells summed (21 → 34 → 77 → 226 → 303 for 10 → 600 cells), which
is the signature of adding per-cell effects as if independent — through a morphological *max*,
where they do not add. Study B's "i.i.d. per-cell map noise is not a valid model here" was about
the same structure.

A truncation sweep on seeds 0–19 showed an interior optimum (τ +0.087 → +0.161 at 100 cells).
**It did not survive.** Fixing that truncation in advance and evaluating once on disjoint seeds
100–199: τ +0.129 → **+0.029** (worse, 38/99, p = 0.027), τ within groups +0.169 → **−0.047**
(worse, 34/95, p = 0.007). The optimum was overfitting to the 20 seeds it was found on.

**Route 2 — cell selection.** Replacing `Var_k(g)σ²` with `Var_k(g)σ² + ½Var_k(c)σ⁴`, both arms
choosing from the same shortlist so the curvature term is the *only* difference: τ@100 0.250 →
0.178 (p = 0.021, worse), τ@400 0.793 → 0.730 (p = 0.079). It steers the budget toward
high-curvature cells, which are the tied and kinked ones Study B's σ/slack criterion already
flags as where the linearisation fails.

**The conclusion is a caveat on §4 and §5, and it is worth stating plainly:** being more
accurate about `Var(J)` did not translate into a better decision here, so the case for putting
second-order attribution into the planner is not made by these results. §4 and §5 stand as
measured — the correction *is* more accurate and *does* fit in a tick — but accuracy in the
variance was the wrong thing to optimise if the decision is what matters.

### 7d. Does any of this survive realistic map error?  **The entropy result yes, the geometry result no**

`studies/bench/noise.py` · `verify_noise.py` (gates) · `compare_noise.py` · `hybrid`, n = 200/arm.

Everything up to here ran on a belief corrupted by **one** source, modelled crudely: a disc of
"observed" inside which the map was exactly right, and reveals that handed over ground truth —
a perfect sensor with perfect localisation. All three real sources are now injected with the
structure they actually have, each gated before use:

- **occlusion** — 2.5-D ray-cast line of sight, so ridges cast *shadows* a disc cannot. Gated:
  flat ground 100% visible, a wall hides 100% behind it and 0% in front.
- **sensor** — correlated (0.15 m length) and range-growing. Gated: injected std matches
  configured to 1.05×, autocorrelation 0.89 at one cell, 2.67× larger error far than near.
- **localisation** — the whole patch written into the map at a wrong pose. Gated by *exact
  reconstruction*: the entire 8100-cell error field is reproduced from three numbers to 1e-8.

Reveals now hand over the **measurement**, not the truth — so sensing no longer converges on
perfect knowledge, which is the single biggest way the earlier setup flattered every policy.

**Building the model produced a result before any policy ran.** The textbook per-cell marginal
of a pose error is first order, `|∇h|·d`. It is *invalid at realistic pose error*: a 2° heading
error over a 6 m lever arm displaces the map by ~2.2 cells while this terrain decorrelates in
~1 (autocorrelation 0.63 at one cell), so the displaced map is nearly **independent** of the
truth rather than a perturbation of it — a three-parameter first-order fit explains only
**R² = 0.23** of the field it generates. That is Study B's lesson (first order runs out before
realistic magnitudes) reappearing for *pose* rather than map noise.

**The ceiling moves, and pose error is what moves it.** Oracle τ@400 — the best *any* policy
could do:

| arm | oracle @400 | what changed |
|---|---|---|
| clean | 0.911 | — |
| + sensor noise | 0.802 | reveals are now noisy |
| + ray-cast shadows | **0.940** | *rises*: the ray cast observes 19.5% vs the disc's 7.9% |
| + pose error | **0.601** | **halved** |
| all three | 0.565 | |

Pose error is the binding constraint, and it is **irreducible by sensing**: revealing more cells
just delivers more measurements carrying the same three wrong numbers. No cell-selection
strategy, however clever, recovers it.

**Result 1 — decision-focused beats information-theoretic: SURVIVES EVERYTHING.**
`disagreement − entropy` at 400 cells:

| clean | sensor | occlusion | localisation | all |
|---|---|---|---|---|
| +0.735 (200/200) | +0.667 (200/200) | +0.520 (200/200) | +0.509 (185/200) | **+0.391 (174/197)** |

Every one at p < 1e-29. This is the study's headline claim and it is robust to all three sources
together.

**Result 2 — the adjoint beating a distance transform: DOES NOT SURVIVE.**
`disagreement − swath` at 400 cells:

| clean | sensor | occlusion | localisation | all |
|---|---|---|---|---|
| +0.183 (p=9e-30) | +0.156 (p=2e-23) | +0.086 (p=1e-28) | +0.079 (p=2e-09) | **+0.016 (p=0.10)** |

Under all three together the edge is **gone**. The within-group component — where geometry is
blind by construction — tells the same story: +0.201 clean, +0.028 under all noise (p = 0.11).
Normalised against the moving ceiling, geometry captures 91% of the achievable gain under full
noise against the adjoint's 95%, so there is very little left to win.

**One source helps the adjoint rather than hurting it.** Realistic occlusion is the only arm
where the adjoint beats geometry at *every* budget, including 25 cells (+0.086, p = 6e-18) where
it loses in every other configuration. Shadows make the observed set geometrically complicated,
so "look along the path" stops being a good proxy for "look where it matters."

**What this changes.** The scientific claim (σ ≠ relevance) is now well supported under
realistic error. The engineering claim (compute the adjoint rather than a distance transform)
is not: at realistic pose error the two tie, and the dominant loss is an error neither can fix.
The highest-value next step is therefore **not** better cell selection — it is reducing or
modelling the pose error, e.g. propagating `∂J/∂h` through `∂h/∂pose` to get a 3-DoF
sensitivity, which is the same adjoint applied where the actual variance is.

### 7e. The σ leak, and whether anything depends on it

Over **unobserved** cells σ is derived from `_local_relief(truth)` — the roughness of ground the
robot has not seen. That is a genuine leak and it deserves to be stated plainly rather than in a
caveat list.

**Why it is there.** With a uniform σ, every unobserved cell scores identically under entropy,
so its "choice" degenerates to the tie-break order — a straw man, not a baseline. In the first
run that version of entropy scored *below* random. The roughness-driven σ gives entropy a real
and sensible target.

**Which way it cuts.** Towards the baseline. Entropy is *nothing but* σ, so the leak is the only
information it has; the adjoint score uses σ only as a weight on a gradient term that dominates
it. So the leak inflates the thing being beaten.

**It is already controlled for twice over.**

1. `random` **is** entropy-with-uniform-σ, exactly — equal scores, random tie-break, a uniform
   draw over unobserved cells. Both versions are therefore in every table already. Measured,
   entropy beats random by +0.004 to +0.032 τ, non-significant in 13 of 15 arm×budget tests.
2. A `--flat-sigma` arm removes the leak for **every** policy at once (n = 200, @400 cells):

| | entropy | random | swath | disagreement | oracle |
|---|---|---|---|---|---|
| clean, σ from truth | +0.088 | +0.056 | +0.639 | +0.822 | +0.911 |
| clean, **flat σ** | +0.072 | +0.056 | +0.639 | **+0.818** | +0.911 |
| all noise, σ from truth | +0.147 | +0.128 | +0.522 | +0.538 | +0.565 |
| all noise, **flat σ** | +0.141 | +0.128 | +0.522 | **+0.537** | +0.565 |

Nothing moves. With the leak removed the headline is unchanged — `disagreement − entropy` =
+0.746 clean (199/200, p = 3e-58) and +0.397 under full noise (174/199, p = 1e-28) — and so is
the negative result, `disagreement − swath` = +0.179 clean but +0.015 under full noise. Entropy
without the leak is statistically indistinguishable from random (p = 0.52 and 0.42).

**A process note, because it nearly went the other way.** The first `--flat-sigma` run returned
numbers *bit-identical* to the leaked run, including entropy — which under a flat σ must collapse
onto random. The flag was being silently dropped: `black` had reformatted the `build_belief`
call onto one line, so the edit threading the argument through never matched. Had the leak
genuinely made no difference, that no-op would have been indistinguishable from the real result
it was supposed to test.

### 7f. Risk estimation against the field's actual baseline — **the adjoint does not win here either**

`studies/bench/risk.py`, `hybrid` plans, full noise, **n = 150**, 256 terrain draws per seed.

§7's sensing framing left the adjoint tied with a distance transform, so this tests the other —
and more standard — use of the same machinery: **estimating a plan's risk** for a risk-aware
cost, which is what every uncertainty-aware off-road planner actually needs. The question's
virtue is that **Monte-Carlo is simultaneously the strongest baseline and the ground truth**:
draw terrain from the belief, roll every plan out on each draw, and the empirical CVaR *is* the
answer — no linearisation, no independence assumption, the envelope's max handled exactly.

Arms, all `J(belief) + κ·risk` with `κ = φ(Φ⁻¹(α))/(1−α) = 1.755` at α = 0.9, so only the risk
estimator differs. Common random numbers across plans, so plan differences are paired.

| estimator | regret | picked best | τ vs truth | est/true risk |
|---|---|---|---|---|
| `none` — mean map, no risk term | 4.874 | 4% | +0.041 | — |
| `sum_sigma` — σ over the swept area | 4.167 | 9% | +0.066 | — |
| **`step`** — Gaussian CVaR (Fan et al., RSS 2021) | **4.002** | 10% | +0.077 | **1.28** |
| **`fosm`** — ours, adjoint variance | 4.363 | 9% | +0.089 | **2.04** |
| *`mc`* — 4096 rollouts/seed | *0* | *100%* | *1.000* | *1.00* |

**What holds.** Adding a risk term helps: `step − none` = −0.872 regret, 36/47 non-tied seeds,
**p = 3.5×10⁻⁴**. And the Jensen bias is real and first-order — `E[J] − J(belief) = +1.36`, so a
mean-map planner is systematically optimistic before any risk term is considered.

**What does not.** **Our FOSM does not beat STEP.** It is slightly *worse* (+0.362 regret,
35/83, p = 0.19 — not significant in either direction), and its risk *values* are twice as
over-conservative as STEP's (ratio 2.04 vs 1.28). It does not significantly beat the no-risk
baseline either (p = 0.46). This is consistent with Study B rather than surprising: first-order
FOSM was measured there at ratio 1.7–2.6 for σ/slack ≫ 1, which is the regime realistic σ sits
in. The cost argument (one backward pass against 4096 rollouts) does not rescue it, because a
far cheaper per-timestep heuristic already does better.

**An n = 40 → n = 150 reversal, recorded because it nearly became the result.** At n = 40 the
ordering was `fosm` 3.741 < `step` 4.315 — ours winning. At n = 150 it is `step` 4.002 <
`fosm` 4.363. The n = 40 advantage was noise. This is the same failure that overturned variant
B in §6, and it is the second time in this study that a promising small-n result has not
survived; anything here below n ≈ 150 should be treated as unreported.

**One new positive finding, about the baselines rather than about us.** Aggregating σ under the
contact patch with a **max** — the physically tempting choice, since the settle height *is* a max
over the patch — **cannot rank the plans at all**: measured spread of `Σ_t σ_t` across the 16
plans is **0.008** with a max against 1.267 with an RMS. A saturating σ field pins the max to the
cap under every footprint of every plan. Anyone building a per-cell risk cost on a map with
capped/unknown-cell σ should aggregate with an RMS or a mean, not a max.

### 7g. **The gradient's support is informative; its magnitude is not.** The unifying result

`studies/bench/risk.py`, n = 100. This began as a check of the objection that FOSM *must* beat a
path-σ surrogate, "because it is the same thing multiplied by gradients, so we know where it
actually hurts." That objection confounds two changes at once — the **weighting** and the
**norm** — so both were varied independently, scored by the ordering each induces over plans
against the MC truth (units therefore cancel):

| surrogate | τ vs true risk |
|---|---|
| `unweighted_L1` = Σ_t σ_t | **+0.101** |
| `unweighted_L2` = √(Σ_t σ_t²) | +0.099 |
| `weighted_L1` = Σ_i \|g_i\|σ_i | **−0.179** |
| `weighted_L2` = √(Σ_i (g_iσ_i)²) | −0.158 |
| `fosm` (correlated) | −0.094 |

> ### ⚠ CORRECTION — this 2×2 was itself confounded, see §7h
>
> The "unweighted" rows sum over **timesteps** (`sig_t.sum(axis=0)`, T+1 terms) while the
> "weighted" rows sum over **cells** (`(g·σ).sum(axis=1)`, ~8100 terms). So varying "weighting"
> also changed the **aggregation domain**, and the p = 6×10⁻¹¹ below cannot be attributed to
> the gradient. §7h supplies the missing cell — unweighted *per-cell* — and finds the domain is
> the whole effect while gradient weighting is neutral (p = 0.43). This is the same confounding
> error this section was written to expose, made one section later. The measurements stand; the
> attribution below does not.

**Weighting by the gradient does not merely fail to help — it inverts the ordering.** At fixed
norm: L1 −0.280 (18/100, p = 6×10⁻¹¹), L2 −0.258 (19/98, p = 7×10⁻¹⁰). The norm is nearly
irrelevant by comparison (−0.001, p = 0.34 unweighted; +0.021, p = 0.010 weighted).

**It is not a bug.** Finite differences against the adjoint on this exact belief, at the
highest-\|g\| cells: worst relative error **2.3% at ε = 1 mm**. The gradient is right. At
**ε = 1 cm the same check is 32% off**, and σ here is 2–30 cm. So the adjoint is exactly correct
and exactly useless at the scale the uncertainty actually lives at — Study B's validity radius,
now demonstrated on the decision itself rather than on a variance ratio.

**Why it inverts, rather than merely degrading.** Two compounding reasons:

1. The gradient is evaluated on the **belief**, which is flat wherever unobserved. On a flat
   plateau the contact arg-max is near-degenerate, so the gradient there is small and largely
   arbitrary — it reports what *would* matter if the ground were flat, in exactly the region
   where we have no idea whether it is.
2. Measured consequence: mean \|g\| is **0.0812 on observed cells against 0.0164 on unobserved**
   — 5× larger where σ is smallest. The product `g·σ` therefore concentrates its weight where
   the map is already known and starves the region that carries the uncertainty.

**The unifying reading of the whole study.** The adjoint's *support* — which cells a plan's cost
can respond to at all — is correct and useful. Its *magnitude* — how much — is not, at realistic
σ. That single distinction explains results that otherwise look contradictory:

- **§7 sensing works** (`disagreement` beats entropy 200/200) because choosing *where to look*
  only needs the support: nonzero where a wheel can touch, zero elsewhere.
- **§7 sensing ties geometry** because a distance transform recovers that same support almost
  perfectly, for free, without any derivative.
- **§7f risk estimation fails** because a risk *magnitude* needs the gradient's value, and the
  value is wrong by 2× at realistic σ.

It also makes a falsifiable prediction for anything built on this next: **any use of the adjoint
that depends only on where it is nonzero will work, and will be matched by geometry; any use
that depends on how big it is will fail until the σ-scale problem is solved.** Second-order FOSM
(§4) is the one candidate fix that targets exactly this, and it is the one experiment left.

**A reporting correction to §7f.** The `est/true risk` column there listed 1.28 for the path-σ
arm. That number is **not interpretable**: `Σ_t σ_t` has units of metres×timesteps, not cost, so
its scale is accidental and only its ordering is meaningful. FOSM's 2.04 is a genuine cost-unit
over-prediction and stands. The two should not have appeared in one column. The decision-quality
comparison in §7f is also a statistical **tie** (p = 0.19), not a loss.

### 7h. Soft contact gradients — a real effect, and it corrects §7g's attribution

`studies/bench/softgrad.py`, n = 80. The proposal: the hard adjoint routes the whole gradient to
the one cell that *currently* wins the contact arg-max, but under uncertainty the winner is not
that cell — so spread the gradient over cells that are *almost* touching.

This is not a smoothing hack, it is the right derivative. What a risk estimate needs is
`∂E[max]/∂h`, and that derivative **is** `P(q is the arg-max)` — a soft distribution collapsing
to one-hot only as σ → 0. A softmax over the offset table with temperature τ is exactly that,
so **τ is set by σ rather than tuned**.

**Only the envelope-mediated term may be softened**, and that is a correctness requirement, not
a modelling choice: the wheels reach the map through `env = max_d(·)` and so have an arg-max;
the belly does not — `chassis_clearance` samples the raw heightmap by bilinear interpolation.
Measured: a one-hot re-contraction of `dJ/denv` reproduces the hard adjoint **exactly** for
`settle` (relative error 0.0000) and fails for `clear_soft` (1.87). Gate: at τ → 0 the soft
gradient collapses onto the engine's arg-max, and the per-plan risk score agrees to 2.4%.

| risk score | aggregation | τ vs true risk |
|---|---|---|
| `Σ_t σ_t` (from §7f) | per **timestep** | **+0.101** |
| σ summed over the gradient's support | per **cell** | −0.128 |
| hard adjoint, `√Σ(g σ)²` | per **cell** | −0.150 |
| soft, τ = 0.25σ … 4σ | per **cell** | −0.145 |

**Softening helps, and the effect is real but small**: +0.006 τ against the hard adjoint,
43/66 seeds, **p = 0.019** at τ = 0.25σ, and flat across the whole 0.25σ–4σ range. It does not
undo the inversion — every per-cell score stays at ≈ −0.145.

**And it exposes the real driver.** With the missing 2×2 cell finally measured — unweighted,
*per-cell* — the two effects separate:

- **Gradient weighting, at fixed per-cell domain:** −0.022, 44/80, **p = 0.43 — neutral.**
- **Aggregation domain:** per-timestep +0.101 against per-cell −0.128, a swing of **0.229**.

So §7g's headline was wrong: the inversion is **not** caused by gradient weighting. It is caused
by summing over **cells** rather than over **time**. A per-cell sum accumulates with the *area*
a plan sweeps, and swept area anti-correlates with true risk here; a per-timestep sum has a
fixed T terms whatever the plan. This is the same area-overcounting pathology measured in §7c
(the correction growing linearly, 21 → 303, with cells summed) arriving by a third route.

**Caveat on the evidence.** The weighting comparison is a clean paired within-run test. The
domain comparison is **across runs** (n = 100 vs n = 80), because no single run contains both
unweighted-per-timestep and unweighted-per-cell. A paired domain test is the obvious next run
and is not yet done — the 0.229 swing is large enough that it is very unlikely to be an artifact,
but it has not been measured to the standard the rest of this section holds.

**What survives of §7g.** The `support vs magnitude` reading still holds, but for a different
reason than stated: what makes the adjoint useful is *where* it is nonzero, and what ruins a
per-cell risk score is *summing over that support*, not weighting within it.

### 7i. Second-order FOSM as a risk estimator — **no**. And the domain test, now paired

`studies/bench/order2.py`, n = 40, both questions in one run so every comparison is
within-seed paired.

| score | aggregation | τ vs true risk |
|---|---|---|
| `time_sigma` — Σ over **timesteps** of footprint σ | time | **+0.140** |
| `cell_sigma` — Σ over **cells** of σ on the support | cell | −0.129 |
| `fosm1_full` — first-order FOSM | cell | −0.130 |
| `fosm1_short` — first order, shortlist only | cell | −0.135 |
| `fosm2_short` — **second order**, same shortlist | cell | −0.190 |
| `soft` — soft contact gradient at τ = σ | cell | −0.129 |

**Q2 — the aggregation domain is the driver. Confirmed, paired.** `cell_sigma − time_sigma` =
**−0.270, 3/39 seeds, p = 3.6×10⁻⁸**. And gradient weighting at a fixed per-cell domain is
**exactly neutral**: `fosm1_full − cell_sigma` = −0.000, 20/40, **p = 1.00**. §7h's correction
stands, now on the paired evidence it was missing. Summing a per-cell quantity over the swept
support is what inverts a risk score; the gradient has nothing to do with it.

**Q1 — second order does not rescue the risk estimate.** `fosm2_short − fosm1_short` = −0.054,
13/36, p = 0.13 — directionally *worse*, not significantly. And the calibration **degrades**:
est/true risk 0.85 first-order against **2.38** second-order. §4's variance improvement
(34.1% → 3.2% of badly-wrong per-cell estimates) does not carry through to the aggregate risk of
a plan.

**A tempting number that is not a result.** The arm including the second-order bias term,
`fosm2_bias`, scores **+0.239** and beats first order by +0.374 (31/40, **p = 6.8×10⁻⁴**) — the
best gradient-based score measured anywhere in this study. It is reported here only to be
discarded, because the quantity underneath it is demonstrably broken: the curvature-predicted
`E[J] − J(belief)` is **−130.9** against a measured **+2.158** — wrong sign, and sixty times too
large. It is the §7c pathology again (per-cell second-order terms summed as if independent
through a max), now severe enough to invert. A broken estimator that happens to rank well is an
accidental proxy, not a bias correction, and reporting the τ without the −130.9 would be
indefensible. A valid bias estimator would have to come first.

**Where this leaves analytic propagation.** Across §7f, §7g, §7h and §7i, every analytic
route — first-order FOSM, correlated FOSM, soft contact gradients, second-order FOSM — lands
between τ = −0.19 and −0.13 against the Monte-Carlo truth, while a per-timestep σ sum costing
nothing lands at +0.14. The adjoint's magnitude does not become usable for risk by softening it,
by adding curvature, or by changing the norm. **The one robust, transferable finding is the
aggregation rule: sum risk over time, not over cells** — a per-cell sum grows with the area a
plan sweeps, which is not what a plan's risk depends on.

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
.venv/bin/python -m studies.bench.ranking --family fan     --seeds 200   # ~90 s each
.venv/bin/python -m studies.bench.ranking --family hybrid  --seeds 200
.venv/bin/python -m studies.bench.ranking --family speed   --seeds 200
.venv/bin/python -m studies.bench.compare_families
.venv/bin/python -m studies.bench.second_order --seeds 50 --family hybrid   # ~7 min
.venv/bin/python -m studies.bench.second_order --validate-bias 100 --seeds 100 \
    --seed-offset 100 --family hybrid                                       # held-out
.venv/bin/python -m studies.bench.verify_noise      # gates; must pass before 7d is believed
for n in clean sensor localisation occlusion all; do \
    .venv/bin/python -m studies.bench.ranking --family hybrid --noise $n --seeds 200; done
.venv/bin/python -m studies.bench.compare_noise
.venv/bin/python -m studies.bench.ranking --family hybrid --noise all --flat-sigma --seeds 200
.venv/bin/python -m studies.bench.illustrate --seed 7    # what each policy looks at
.venv/bin/python -m studies.bench.risk --seeds 150 --family hybrid --noise all   # ~45 min
.venv/bin/python -m studies.bench.softgrad --gate     # must pass before 7h is believed
.venv/bin/python -m studies.bench.softgrad --seeds 80 --family hybrid --noise all
.venv/bin/python -m studies.bench.order2 --seeds 40 --family hybrid --noise all   # ~25 min
```

Production changes made by this work are confined to `engine/step.py`,
`engine/simulator.py`, `engine/envelope.py` (plus repairs to two already-broken test files).
`rollout_kernel` remains bit-identical to init+step.
