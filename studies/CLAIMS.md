# Paper claims — revision 2, post red-team and post holdout

> **REVISION 3 BANNER (2026-08-11, correction cycle complete).** A code review found 13
> number-corrupting defects (commit 8e0aef5; ledger in `RERUNS.md`, narrative in
> `HANDOFF.md` §11); every affected artifact was rerun (commits 5f13dcb, 4c71c94). Outcome
> for this document, claim by claim:
> - **C1 clean-arm numbers stand UNCHANGED** — the hybrid/clean rerun is byte-identical to
>   the committed artifact (the clean arm never touches the fixed code paths, and tau-b
>   equals gamma absent exact ties), so +0.691/p=2.5e-60 and the clean-regime scope law
>   survive as written. All-noise/occlusion/localisation numbers are superseded by the
>   2026-08-10 jsons; the qualitative claims (entropy loses everywhere, corridor ties/beats
>   the adjoint under realistic noise, elites compress every edge, the motion-coupled
>   retraction) all reproduce on corrected masks.
> - **C2 stands**; the rare-event number tightens to FORM missing by 1.99 dex (stalled
>   attacks now censored rather than reported as finite Φ(−4)) vs subset simulation's 0.107.
> - **The "Follow-up direction validated: decision-focused perception training" section
>   below is RETRACTED (▼▼).** Under the corrected noise model the thread's one surviving
>   positive REVERSES: the decision-loss-trained inpainter loses to MSE at p=3.9e-10
>   (9/68 wins, dfl_full.json 2026-08-11); the cost-space arm is marginal (p=0.019, 15/19).
>   dfl_settle's rerun refused to complete because the retrained rung-2 model no longer
>   reproduces the recorded capacity result (0.157 vs 0.257) — the training landscape
>   itself moved with the noise fix. The p=0.006 headline, the ablation chain, and the
>   "symmetric law" sentence built on it describe the corrupted benchmark, not the
>   phenomenon. The section is kept below as the historical record; do not cite it.
>   The dual-consumer insight and the observability result (75% zero-gradient mass) were
>   derived from mechanisms, not that headline — they need re-verification before reuse,
>   not automatic retraction (observability.json was rerun 2026-08-10 and its qualitative
>   claim reproduces).

Written 2026-08-06 (rev 2, same day). Revision 1 was adversarially reviewed by three
independent referees (novelty / methodology / significance) and stress-tested by a
pre-registered sweep + holdout and a clean-worktree reproduction. This revision records
what survived. Changes from rev 1 are the point; they are marked ▲ (strengthened),
▼ (weakened/retracted), ● (reworded).

The fixed-sigma rerun (n=200 per arm, gate-verified non-saturating sigma: <1% of unobserved
cells at cap vs 63-65% before; corridor baselines added) is COMPLETE; its numbers are final
below. Results: studies/out/bench/ranking_hybrid{,_all}.json.

## Working title

● **When Does Sensing Need a Derivative? Decision-Focused Sensing and the Limits of
Uncertainty Propagation Through Contact**

(The rev-1 title claimed the method; the paper the evidence supports is the
characterisation, with the method as its constructive half.)

## Status of the contributions

- ▼ **C1 (constructive) — narrowed three times, now final.** Under the fair (non-saturating)
  sigma, the adjoint beats information-theoretic sensing DECISIVELY at every budget and
  noise level (vs entropy @400: clean +0.691, p=2.5e-60; all-noise +0.471, p=2.6e-35 — the
  fair sigma GREW this gap, because distance-dominant sigma pushes entropy's picks even
  further off-plan). Against the strongest fair baseline (corridor-masked uncertainty, no
  derivative) the scope law is a **headroom law**: the adjoint's edge tracks the
  gap between what corridor masking achieves and the oracle ceiling, capturing ~30-70% of
  that headroom wherever it exists. Both compressions kill it: FLOOR (budgets too small for
  any policy to matter — clean@25, oracle itself at 0.19, edge -0.03 n.s.) and CEILING
  (budgets or sensor cones large enough that masking saturates the achievable — full-noise
  @400, disagreement at 95% and corridor at 94% of oracle, edge +0.007 p=0.08; the
  wide-cone motion-coupled holdout is the same case at sensor scale). In between the edge
  is real and significant: clean @100/@400 +0.108/+0.210 (p<=5.4e-5), full noise @25/@100
  +0.041/+0.061 (p<=3.2e-3), within path-identical groups included. Central figure:
  studies/out/bench/scope_law.png.
  **Live-elite result (C5 first half, 2026-08-06, studies/bench/elites.py):** real MppiGpu
  elite sets on the belief map have low-but-nonzero diversity (path spread ~19% of hybrid's,
  100/100 seeds lower, never speed-degenerate) — a distinct regime no synthetic family
  covered. On them the adjoint STILL beats entropy decisively (all budgets, both noise arms,
  p<=1.3e-4) — uncertainty != relevance is fully robust. But its edge over corridor masking
  survives only under CLEAN noise (@100/@400: +0.176/+0.113, p<=3e-5) and is NOT significant
  under the primary full-noise arm at any budget (p>=0.097, win rates ~53-60%): low-diversity
  real candidates plus realistic noise together erase the derivative's measured advantage
  over masking. The honest constructive claim is therefore: sense where the candidates
  disagree (always beats entropy); compute the disagreement with a derivative only in
  low-noise, precise-look regimes — otherwise the corridor mask is all the decision-awareness
  the sensing can use.
  **The motion-coupled (wide-cone, body-fixed lidar) version of C1 is retracted**: on
  pre-registered holdout seeds the +0.045 edge over entropy evaporated (-0.005, p=0.83),
  and in the one regime where an edge over entropy replicates (narrow FOV, short range,
  +0.058, p<0.005), corridor-masked MI — no derivative — ties or significantly BEATS the
  adjoint (-0.035, p=0.033). Measured scope boundary, reported as a finding: at cone-scale
  sensing granularity, decision-AWARENESS is fully delivered by masking uncertainty to
  where the candidates drive; the derivative pays only when sensing granularity is finer
  than the candidate-separation scale.
- **C2 addendum (2026-08-06 evening, studies/bench/{certify,rare}.py):** the map-error
  robustness-certificate idea (minimal terrain perturbation flipping the plan choice) was
  tested and REFUTED in both its regimes: as a flip-probability predictor it never beat the
  cost gap or the 2-forward bracket (pre-registered n=100: tau 0.335 vs gap 0.360); as a
  rare-event estimator (FORM / attack-seeded IS vs subset simulation, 16-case pre-registered
  testbed) it failed by ~2 dex while derivative-free subset simulation matched truth to 0.1
  dex at less compute. The traced mechanism is the sharpest statement of C2 yet: contact
  decisions fail through NONLINEAR variance asymmetry (a rival's envelope-max cost has ~3x
  the variance and overtakes via ordinary excursions) — the true failure modes are not near
  the linearized boundary, so the gradient does not merely mis-measure the distance, it
  points at the wrong boundary. Eighth consecutive sampling-vs-gradient tie/loss.
- ▲ **C2 (characterisation) — now the paper's spine, per all three referees.** The validity
  radius (contact slack, mm, vs map sigma, cm); seven measured refutations of
  magnitude-based risk, incl. the bundled adjoint's heavy-tailed blow-up (single draws at
  5e10) connecting the Suh/Parmas finite-sample pathology to terrain. Reproduced exactly
  from committed code (12/12 studies, worktree, 2026-08-06). NEW obligation from review:
  engage the saltation-matrix / hybrid-systems-sensitivity literature (Hiskens & Pai 2000;
  arXiv:2306.06862; Salted KF arXiv:2007.12233) — our position: saltation corrects ODE-flow
  sensitivities across time-domain guard crossings; the quasi-static settle has no
  time-domain events — its discontinuity is a parametric active-set change in the h->pose
  map, for which the applicable frame is parametric-programming sensitivity (Fiacco), and
  the validity radius IS the measured distance-to-active-set-change. Write this argument
  explicitly; consider a small saltation-style experiment as the eighth refutation.
- ● **C3 (scope map) — reframed around the mechanism, not the metaphor.** Our own data:
  variance-across-plans adds nothing over summed |adjoint| (p=0.51), so "committee
  disagreement" is ancestry (cite QBC: Seung/Opper/Sompolinsky 1992, Freund et al. 1997 —
  the single-plan confirmatory failure is QBC's own >=2-members argument), not mechanism.
  The mechanism: coverage of the cells the candidate set's costs rest on, weighted by
  sigma. The family result (one corridor -> geometry suffices, tau=1.0; elite-set regime ->
  adjoint pays) stands and now includes the motion-coupled boundary above. The
  hierarchical-stack argument (router picks corridor, short-horizon MPC perturbs within it
  -> real elites ARE the elite-set regime) is supported by this stack's own architecture
  (verified by code trace) but must be measured on live elites (C5), incl. the NEW
  candidate-set-diversity check: real elites may be lower-diversity than synthetic fans,
  shrinking the signal (significance referee's compounding-risk attack — unmeasured).
- ● **C4 (risk findings) — language corrected, content intact.** (a) aggregate over time
  not cells (p=3.6e-8, reproduced exactly); RMS not max. (b) The Jensen gap: DROP "unstated
  in the literature" — mean-map optimism is the stated premise of STEP/EVORA/Cai; what is
  new is the quantification (E[J]-J(belief) = +1.4, first-order-real, uncorrected by second
  order) for max-based contact costs on geometry. (c) belief±sigma bracket beats STEP's
  per-step Gaussian CVaR at ranking (p=1.6e-3) using bounds elevation_mapping already
  publishes; nearest prior art Cai's CVaR-Dyn (traction-space, one-sided) — state the
  delta, expect the "geometric analogue" objection.
- **C5 (pending, unchanged in content, raised in priority):** real sigma from bags with
  two-pass calibration; live MPPI elite sets incl. diversity measurement; the corridor
  baselines carried into every future comparison. Still the gate to any main-track venue.

## Framing corrections (novelty referee; all writing-level, all mandatory)

- ▼ RETRACT "first embodied decision-focused sensing of any kind." Goal-oriented optimal
  experimental design (GOOED: Alexanderian, Petra & Ghattas 2018, arXiv:1802.06517; Attia
  et al. 2023; arXiv:2502.15062; arXiv:2507.02500) already places sensors by adjoint-weighted
  effect on a downstream quantity of interest; arXiv:2509.15961 already moves a sensor along
  an optimized path (entropy-flavoured). Replacement claim: "to our knowledge the first to
  combine goal-oriented (adjoint-weighted) sensing value with an embodied candidate-decision
  set through non-smooth contact physics" — and cite GOOED as the smooth-PDE antecedent.
- Cite expected-model-change active learning (Cai/Zhang/Zhou ICDM 2013) and the
  decision-focused active-learning area alongside QBC.
- Rewrite the POMDP-IR/belief-space delta: the right contrast is GOOED (cheap, localized,
  adjoint-based) vs us (non-smooth contact, embodied, candidate-set), not "VOI is expensive."
- ProTerrain/MonoForce/FusionForce: add an explicit shared-authorship positioning sentence;
  anchor the delta in C2 (the test-time use required the validity characterisation), not in
  "we run the gradient at a different pipeline stage."
- Terminology: lead with "decision-focused sensing"; VOI as ancestry (first-order surrogate
  for VOI over plan selection); DWR as an explicitly-flagged analogy; do not lead with
  "active perception."

## Reporting standards (methodology referee; adopted for every number in the paper)

- Primary baselines: corridor_sigma / corridor_mi (task-masked, no gradient) — not plain
  entropy. Entropy stays as the information-theoretic representative, with its sigma-field
  non-degeneracy demonstrated (tie-plateau check at every tested budget).
- Report median alongside mean for paired effects; report the oracle gap with equal
  prominence to the baseline gap.
- State the total hypothesis-test count; apply a family-wise correction to any p in the
  0.005-0.05 band before claiming it. (Casualties under Bonferroni, acknowledged: cap-pooling
  p=0.014, softgrad p=0.019 — and the retracted motion-coupled p=0.010.)
- The five noise arms are ONE set of 200 map/plan realisations under five corruption
  models — pseudo-replication, disclosed as such.
- Pre-registered holdout for every headline: design on one seed range, report from virgin
  seeds, no iteration after peeking. (This protocol is what caught the motion-coupled
  mirage; it is a selling point of the paper's methodology, not overhead.)

## Evidence provenance

- FINDINGS.md reproduced end-to-end from committed code in an isolated worktree
  (2026-08-06): 12/12 studies, headline numbers exact (study A/B, risk, order2) or within
  a few percent (rankings). The one flagged "discrepancy" was the checker comparing the
  clean-arm claim to the all-noise arm; both arms reproduce exactly.
- Sweep: 11 configs x 100 seeds (studies/out/sensing/sweep.json). Holdout: 2 frozen configs
  x 120 virgin seeds (holdout.json). Motion-coupled retraction is grounded in these.
- Fixed-sigma rerun: DONE 2026-08-06. New sigma: tanh-saturating frontier+distance+relief
  field (gate: 0.04-0.18% of unobserved cells within 1% of max, vs required <10%; ~390
  distinct values among entropy's top-400 picks). Corridor policies added to ranking.py.
  Caveat recorded: the old +0.633/+0.391 headline was the FAN family; the rerun is HYBRID,
  so the sigma fix is not isolated apples-to-apples — but the untouched-policy sanity check
  (disagreement-swath @400/all = +0.016) matches the prior pipeline exactly.

## Follow-up direction validated (2026-08-06 evening): decision-focused perception training
## ▼▼ RETRACTED 2026-08-11 — see the rev-3 banner; historical record only, do not cite

The one gradient use that PASSED its pre-registered bar (studies/bench/dfl.py,
dfl_full.json): train a terrain inpainter with a decision loss (softmin expected regret,
backpropagated through the settle adjoint) instead of MSE. Chained gradient verified to
5e-13 vs finite differences; training stable (2 clipped steps / 400). On 80 VIRGIN held-out
seeds: decision-trained regret 0.269 vs MSE-trained 0.598 (p=0.006 paired, +33.8% of the
zero-fill->MSE improvement; pre-registered bar was p<0.05 and >=10%). Signature mechanism:
the decision-trained model has ~3x WORSE height RMSE yet ~2.2x better decisions — it learns
a planning map, not a mean map (consistent with the measured Jensen optimism of mean maps).
Anomaly resolved: the softmin loss scalar sits on a temperature-entropy floor while argmin
quality improves beneath it; the learned weights are cosine-0.79 to cost-space-trained ones
but outperform them. Together with the eight test-time refutations this completes the
symmetric law: gradients through contact fail wherever trusted pointwise, and work where
consumed by an averaging optimizer. Ablation vs the standard architecture (dfl_ablation.json, user-requested): "MSE map +
risk-aware planning" does NOT recover the advantage — MSE+STEP-penalty 0.638 and
MSE+correlated-CVaR@0.9 (M=128) 1.131 vs decision-plain 0.269 (p=0.0012 and 1.8e-8);
CVaR even hurt its own base map (p=3e-4). The (mean, sigma) factorization loses
decision-relevant information (correlation/direction/relevance) that planner-side risk
handling cannot reconstruct. Composition (decision map + CVaR) added nothing (p=0.11).
Caveat: regret is measured against the risk-NEUTRAL truth (argmin of true cost), which
structurally disfavors risk-averse choice rules; with a catastrophe-shaped cost the CVaR
arms would deserve a rematch. THIS is the follow-up paper (CoRL-shaped, MonoForce
lineage — supervisor-aligned): capacity ladder, pessimism-mechanism analysis, sigma-head
variant, plan-family transfer, then hindsight self-supervision on real bags. Mechanism + transfer follow-ups (dfl_mechanism.json): the extra height error of the
decision-trained model is SHAPED, not noise — ~5x larger in decision-IRRELEVANT cells than
relevant ones (two independent stratifications agree; corr(extra error, adjoint support)
= -0.095, CI excludes 0): it economizes its error budget where no decision gradient reaches.
One pre-registration miss, recorded: the bias is DOWNWARD (fills low), not the predicted
pessimistic-upward. Leading revised hypothesis (UNVERIFIED): a low fill makes unknown cells
inert in the envelope max — the argmax lands on observed cells, so plans get ranked on what
is actually known ("decide on what you know" imputation, not pessimism). Testable: argmax
occupancy of unobserved cells under decision-fill vs MSE-fill — Phase B item. Plan-family
transfer HOLDS but marginally (hybrid-trained, fan-evaluated: p=0.047, wins 25/37 non-tied;
the fill provably never saw plans — only the loss did — so the caution is terrain-shaped,
though weaker off-family). Caveats:
linear 6-feature model (capacity confound unresolved), one cost, synthetic.

## Venue and sequencing (significance referee; endorsed)

1. Workshop paper now: the characterisation + the honestly-scoped sensing result +
   the negative motion-coupled finding ("when does sensing need a derivative?" is a good
   workshop conversation precisely because of the retraction).
2. Colleague conversation (Agishev/Zimmermann) with the C2 result as the gift.
3. C5 (real sigma, live elites, diversity check) — then T-RO for the full paper
   (characterisation + method + scope map; RA-L's page limit cannot carry it).

## Standing weaknesses (updated)

1. Everything synthetic; sigma placeholder (now non-saturating, gate-verified — still
   synthetic). C5 unchanged as the biggest gap.
2. Motion-coupled sensing: RETRACTED as a positive claim; now a measured negative scope
   boundary. The honest residual positive in that setting: task-masking (any form) beats
   pure entropy at narrow FOV (holdout-confirmed, p<0.005) — a result about masking, not
   about derivatives.
3. tau is a ranking proxy; closed-loop time-and-safety outcomes unmeasured.
4. RESOLVED (2026-08-06): live-elite diversity measured (studies/bench/elites.py, n=100+60).
   Real elites: ~1/5 of hybrid's spread, never degenerate; adjoint-vs-entropy survives on
   them; adjoint-vs-corridor survives only under clean noise. Remaining caveats: fixed goal,
   one replan cycle (not closed loop), elites keep MPPI's native rear=0 convention vs the
   synthetic families' rear=mean.
5. Audience-size ceiling: needs a differentiable contact model at deployment. Mitigation:
   the adjoint runs in a low-rate advisory side-channel (K taped rollouts per look
   decision); the control loop stays zeroth-order; MonoForce is an adoptable open model.
