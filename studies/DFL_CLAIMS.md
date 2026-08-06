# DFL paper claims draft — for adversarial review

> **REVIEW BANNER (2026-08-07, methods red-team + live-elite check): the claims below are
> now known to overstate.** (1) A ONE-PARAMETER baseline — MSE weights + a tuned coefficient
> on the distance-to-observed feature ("fill lower the farther from data") — recovers 83%
> of the decision-training gap (regret 0.325 vs 0.269) and is statistically
> indistinguishable from it (p=0.18). The "shaped error / planning map" mechanism is largely
> a distance artifact (relevant cells sit 3.8x closer to observed ground, definitionally).
> (2) decision vs cost_space: p=0.099, n.s. (3) On real MPPI elites the decision-vs-MSE edge
> is n.s. (p=0.31-0.74; 56% ties); only decision-vs-zero-fill survives (p=4e-8). (4) The C2
> factorization claim has a live alternative explanation (sigma miscalibrated for the MSE
> model's actual residuals) pending a calibrated-sigma rerun. (5) "Virgin" seeds are virgin
> w.r.t. weights only; design choices were shaped on overlapping seed ranges. (6)
> "Pre-registered" = pilot-informed same-session; describe it as such. Rev-2 of this
> document will restructure around what survives: decision-training as a DISCOVERY procedure
> whose learned solution compresses to a shippable heuristic — the same
> discover-then-compress pattern as the sensing result (support -> corridor mask) and the
> risk result (CVaR -> bracket). Required experiments before any submission: distance-shift
> baseline in every table; distance-matched stratification; calibrated-sigma ablation rerun;
> clean seed block (5000+); route-level (lattice corridor) evaluation.

Written 2026-08-07 (evening of the first results). This is the contribution skeleton of the
intended SECOND paper (the learning paper; companion to the characterisation/workshop paper
in WORKSHOP_DRAFT.md + CLAIMS.md). Stated as it would be at submission, evidence status
marked. Reviewers: attack this document. Two experiments are IN FLIGHT tonight and marked
both now COMPLETE and folded in below (inert-fill: partial confirm; observability: ~75%
of decision-relevant mass unconstrained by driving).

## Working title

**Learning Planning Maps: Decision-Focused Training of Terrain Perception Through
Differentiable Contact**

## Abstract (draft)

Terrain perception for off-road robots is trained to reconstruct the world — minimize
height error, calibrate uncertainty — and its output is handed to a planner that must then
survive the reconstruction's mistakes. We show this hand-off discards the objective that
matters. Training a terrain inpainter by DECISION loss — the regret of the plan choice made
on its completed map, backpropagated through a differentiable quasi-static contact model —
halves plan-choice regret relative to an identically-structured MSE-trained model (0.27 vs
0.60, p=0.006, 80 pre-registered held-out seeds), while producing a 3x WORSE height map.
The model learns a planning map, not a mean map: its extra error concentrates ~5x in
decision-irrelevant cells, and its systematic low bias in unobserved regions halves the contact-support
mass resting on unknown cells (21.7% vs 44.1%; restoring the bias alone reverses it) —
ranking candidate plans on what is actually known. The standard alternative — a likelihood-trained map plus risk-aware
planning (per-step Gaussian CVaR, or CVaR over correlated map samples) — does not recover
the gap (p<=0.0012): the per-cell (mean, sigma) factorization provably loses the
correlation, direction, and relevance structure the decision depends on, and no planner-side
risk machinery reconstructs it. We ground the result in a certainty-equivalence argument
(for max-form contact costs the decision-optimal imputation differs from the conditional
mean), quantify which terrain directions trajectory-self-supervision can and cannot
constrain (median 75% of decision-relevant terrain mass — 95% among unobserved cells — is
unconstrained by the driven trajectory), and propose a falsifiable field-evaluation ladder — prospective
contact verification, payload-change falsification, and randomized paired trials — for a
learning paradigm whose usual evaluation cannot distinguish a terrain estimate from an
equivalence class of them.

## Contributions as claimed

- **C1 (headline): decision-focused training beats likelihood training for planning.**
  Regret 0.269 vs 0.598, p=0.006 paired sign test, 80 virgin seeds; pre-registered bar
  (p<0.05 AND >=10% of the zero-fill->MSE gap; achieved 33.8%) passed on first full-scale
  run. Gradient chain through the settle verified to 5e-13; training stable (2 clips/400).
  Status: DONE at linear-model scale, synthetic. The capacity ladder is the acknowledged
  decisive next test.
  LIVE-ELITE UPDATE (dfl_elites.json, n=80, both fixed-candidate and fill-in-the-loop
  levels): on the real MPPI's low-diversity elite sets the decision-vs-MSE edge shrinks to
  a non-significant trend (+14% of the gap, p=0.74 / p=0.31; 56% of seeds tie — both fills
  pick the same elite), while decision-vs-zero-fill stays decisive (p=4e-8). Same structural
  wall the sensing claim hit (CLAIMS.md C1 live-elite note): candidate-set diversity is the
  moderator of every edge-over-strong-baseline in this project. Scope accordingly: at the
  within-corridor elite level, any reasonable fill suffices; the untested level where fill
  quality should matter most on THIS stack is the ROUTE choice (lattice/cost-to-go corridor
  selection — where the real phantom-plateau incident lived). Route-level evaluation is now
  the priority experiment, ahead of the capacity ladder in importance for the field claim.
- **C2: the factorization result.** MSE map + STEP penalty (0.638) and MSE map + correlated
  CVaR@0.9, M=128 (1.131) both fail to recover decision-plain (0.269), p<=0.0012; CVaR
  hurts even its own base map (p=3e-4). Composition (decision map + CVaR) adds nothing
  (p=0.11). Declared caveat: regret is risk-NEUTRAL (argmin of true cost), which
  structurally disfavors risk-averse rules; a catastrophe-shaped-cost rematch is owed.
  Status: DONE (same scale/caveats as C1).
- **C3: mechanism.** (a) Shaped error budget: extra error ~5x concentrated in
  decision-irrelevant cells (two independent stratifications; corr(extra error, adjoint
  support) = -0.095, CI excluding 0). (b) Bias is DOWNWARD (-0.22 m unobserved), contra the
  pre-registered pessimism guess — recorded as a miss. (c) Inert-fill hypothesis (low fill
  pushes the envelope arg-max onto observed cells; "decide on what you know"): PARTIALLY
  CONFIRMED (dfl_inert.json). Unobserved cells carry 21.7% of adjoint-support mass under
  decision-fill vs 44.1% (mse) and 52.8% (zero-fill); shift control near-dispositive —
  undoing only the bias (+0.207 m) drives the share to 68.2%, above every baseline: the
  LOWNESS itself creates the inertness. Not confirmed: per-seed inertness-drop does not
  predict per-seed regret (r=-0.11, n.s.) — population-level mechanism, claim at that grain. (d) Cross-plan-family transfer holds marginally
  (hybrid-trained, fan-evaluated: p=0.047, 25/37 non-tied) — terrain-shaped, not
  plan-specific (the fill provably never sees plans; only the loss does), but weaker
  off-family.
- **C4: theory.** Certainty equivalence fails for max/threshold contact costs; the
  decision-optimal single map differs from the posterior mean (companion-paper measurement:
  E[J]-J(mean) ~ +1.4). To write: one toy proposition (regret-optimal imputation deviates
  from the conditional mean wherever the contact arg-max is uncertain) + softmin surrogate
  consistency. Plus the identifiability analysis: trajectory-observable subspace (dtau/dh
  Gramian) vs decision-relevant subspace (dJ/dh): DONE as a pilot (observability.json,
  n=40, sanity-gated — 99.2% of observability mass within wheel-reach of the track).
  Median 75% [p10 68, p90 88] of decision-relevant mass sits on cells the driven
  trajectory's outputs are structurally blind to (exact-zero gradient), rising to ~95%
  among unobserved cells — even though the relevant cells sit only ~0.5 m from the track:
  the contact arg-max reads a narrow band, and proximity is not observability. Consequence
  for the program: trajectory-fitting (MonoForce-style) CANNOT supply the labels planning
  needs; hindsight labels must come from later SENSING (the map), not from contact — this
  redirects weakness #7's label source and quantifies the user's ground-truth objection.
  Status: argument sketched, nothing proven yet.
- **C5: evaluation methodology for self-supervised terrain learning.** The falsifiable
  ladder: (i) observability/relevance overlap in sim; (ii) prospective contact verification
  (predict supporting height BEFORE contact, verify from wheel kinematics AT contact);
  (iii) mowed-plot gold set; (iv) payload-change cross-embodiment falsification; (v)
  randomized paired field trials for the decision claim. Status: DESIGNED, not executed.
  Claimed as a methodological contribution because trajectory-fitting alone validates an
  equivalence class, not a terrain — the known epistemic gap of this program's own lineage.

## Prior art and claimed deltas (to be fetch-verified by reviewers; flag anything wrong)

- SPO / decision-focused learning (Elmachtoub & Grigas; Mandi et al. survey): the loss
  family. Delta: first embodied instantiation through non-smooth contact physics for
  terrain perception; the RMSE-vs-regret dissociation demonstrated.
- Decision-aware model learning in RL (VAML, Farahmand; value equivalence, Grimm et al.;
  TaskMet): closest ML lineage — "fit the model where the decision cares." Delta: they
  learn DYNAMICS for control; we learn PERCEPTION/maps through fixed known physics.
- Differentiable planners in the training loop (VIN; Amos & Kolter differentiable MPC;
  Karkus et al.): machinery lineage. Delta: contact/terrain, not smooth or discrete
  dynamics; perception target, not policy.
- MonoForce / FusionForce: infrastructure and self-supervision lineage (loss in MOTION
  space — explain the driven trajectory). Delta: loss in DECISION space (pick the right
  plan); plus C5's answer to the shared ground-truth problem. Shared-group positioning
  sentence required.
- UNRealNet (+ neural-process elevation): the likelihood-trained inpainting baseline family.
- Triest et al. IRL costmaps: learns the COST through a planner; we learn the MAP under a
  fixed cost. Adjacent, must be distinguished.
- Known threat to check hard: end-to-end "planning-oriented" perception in autonomous
  driving (e.g., UniAD and successors) — perception trained with planning as the target, at
  scale. Delta must be argued, not assumed: modular map interface retained; physics-based
  differentiable contact rather than learned planner; off-road contact costs.

## Known weaknesses (pre-declared)

1. CAPACITY CONFOUND — the strongest objection: 6-feature linear models; MSE may close the
   gap with a real network. The capacity ladder is scheduled and its criterion will be
   pre-registered before running.
2. Synthetic everything: fractal terrain, placeholder-though-structured sigma, one robot,
   one cost function. No real data yet (hindsight self-supervision on bags is designed, not
   run).
3. Risk-neutral regret metric (see C2 caveat).
4. Decision vs cost-space conditions: decision wins the table (0.269 vs 0.427) but the
   PAIRED decision-vs-cost_space significance was not reported; the two share cosine-0.79
   weights and decision works partly THROUGH cost alignment. A reviewer may argue the
   simpler cost-space loss is the real story and softmin adds little. Must be tested.
5. Transfer holds only marginally off-family (p=0.047); cost-function transfer untested.
6. Test seeds 100-179 are virgin to the DFL experiments but the terrain GENERATOR and cost
   were used extensively across this project's other studies; a skeptic may ask whether
   design choices (features, sigma model) were indirectly tuned on this distribution.
7. Deployment story: retraining coupled to the planner's cost; hindsight labels assume the
   robot later observes what it inpainted (not always true — untraversed regions stay
   unlabeled, a selection bias the field version must handle).
