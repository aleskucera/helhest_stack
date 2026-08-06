# DFL paper claims draft — for adversarial review

Written 2026-08-07 (evening of the first results). This is the contribution skeleton of the
intended SECOND paper (the learning paper; companion to the characterisation/workshop paper
in WORKSHOP_DRAFT.md + CLAIMS.md). Stated as it would be at submission, evidence status
marked. Reviewers: attack this document. Two experiments are IN FLIGHT tonight and marked
[PENDING]: the inert-fill mechanism check and the observability/relevance overlap pilot.

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
decision-irrelevant cells, and and its systematic low bias in unobserved regions halves the contact-support
mass resting on unknown cells (21.7% vs 44.1%; restoring the bias alone reverses it) —
ranking candidate plans on what is actually known. The standard alternative — a likelihood-trained map plus risk-aware
planning (per-step Gaussian CVaR, or CVaR over correlated map samples) — does not recover
the gap (p<=0.0012): the per-cell (mean, sigma) factorization provably loses the
correlation, direction, and relevance structure the decision depends on, and no planner-side
risk machinery reconstructs it. We ground the result in a certainty-equivalence argument
(for max-form contact costs the decision-optimal imputation differs from the conditional
mean), quantify which terrain directions trajectory-self-supervision can and cannot
constrain [PENDING pilot], and propose a falsifiable field-evaluation ladder — prospective
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
  pushes the envelope arg-max onto observed cells; "decide on what you know"): [PENDING
  tonight, incl. a shift-control separating lowness from shape and a per-seed
  mechanism-to-regret correlation]. (d) Cross-plan-family transfer holds marginally
  (hybrid-trained, fan-evaluated: p=0.047, 25/37 non-tied) — terrain-shaped, not
  plan-specific (the fill provably never sees plans; only the loss does), but weaker
  off-family.
- **C4: theory.** Certainty equivalence fails for max/threshold contact costs; the
  decision-optimal single map differs from the posterior mean (companion-paper measurement:
  E[J]-J(mean) ~ +1.4). To write: one toy proposition (regret-optimal imputation deviates
  from the conditional mean wherever the contact arg-max is uncertain) + softmin surrogate
  consistency. Plus the identifiability analysis: trajectory-observable subspace (dtau/dh
  Gramian) vs decision-relevant subspace (dJ/dh) — claim only the overlap [PENDING pilot].
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
