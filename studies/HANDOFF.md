# HANDOFF — the complete experimental record (2026-08-06/07 research sessions)

Written for a cold start by a future session (any model). Read this first, then
`studies/CLAIMS.md` (sensing/characterisation paper) and `studies/DFL_CLAIMS.md` (the DFL
thread, incl. its review banners). Everything below is committed on branch
`study/adjoint-sensitivity`; every number has a JSON in `studies/out/bench/` and a script in
`studies/bench/` or `studies/adjoint/`; every experiment marked [PRE-REG] had its success
criteria written before running.

## 0. The one-paragraph story

The project asked: what should an off-road planner do with per-cell heightmap uncertainty,
and can a differentiable contact simulator's derivatives help? Answer, after ~30
experiments: derivatives through contact FAIL wherever trusted pointwise (validity radius
is mm, set by contact-switch slack; sigma is cm) — eight refuted uses — and WORK where
consumed by an averaging process. The two constructive survivors: (1) decision-aware
SENSING beats entropy everywhere but beats simple corridor-masking only when headroom
exists (scarce/precise looks, diverse candidates); (2) **Clark/SSTA analytic moment
propagation through the contact max** (the user's original "analytic per-cell uncertainty"
idea in its mathematically correct form) — first analytic risk estimator to pass every
pre-registered bar, apparently the first use of Clark 1961 in robotics (verified via full
citation-graph scan: 0 robotics papers among 728 citers).

## 1. Papers / documents state

- `studies/WORKSHOP_DRAFT.md` — COMPLETE workshop paper ("When Does Sensing Need a
  Derivative?"), all numbers final. Parked pending the Karel conversation. LaTeX port not
  started (IEEEtran + pdflatex available; figures in studies/out/... — note *.png is
  gitignored, regenerate via scripts).
- `studies/CLAIMS.md` — sensing/characterisation claims, rev 2, post 3-referee red-team +
  holdout. Trustworthy; FINDINGS.md beneath it was independently reproduced 12/12 from
  committed code in a worktree.
- `studies/DFL_CLAIMS.md` — the DFL thread with TWO review banners; read banners first,
  body second (body overstates, banners correct it).
- `studies/KAREL_BRIEF.md` — one-page supervisor brief (Karel Zimmermann is the user's
  supervisor; MonoForce/ProTerrain are his group's line). Slightly stale now: it predates
  the DFL settling verdict and Clark; update before the meeting.

## 2. SENSING thread (the workshop paper's positive half)

- Free-look ranking (fixed sigma, n=200/arm, `bench/ranking.py`): adjoint-support
  "disagreement" sensing beats entropy at every budget/noise (clean @400 +0.691 tau,
  p=2.5e-60; full noise +0.471, p=2.6e-35). Entropy ~= not sensing at all. [Fair-sigma
  version: the ORIGINAL entropy baseline was crippled — 63-65% of unobserved cells tied at
  the sigma cap; fixed to a tanh frontier+distance+relief field (<1% at cap) in
  `bench/noise.py`; the gap GREW under the fair sigma.]
- vs corridor-masked baselines (no derivative): **headroom law** — the adjoint's edge =
  30-70% of the gap between corridor-masking and the oracle; killed by floor (25-cell
  budgets, clean), ceiling (400-cell budgets under noise; adjoint at 95% of oracle), wide
  cones, and low-diversity candidates. Central figure: `bench/plot_scope.py` ->
  scope_law.png.
- Motion-coupled (body-fixed lidar, `sensing/pipeline.py`, sweep 11 configs + [PRE-REG]
  holdout seeds 200-319): design-set edge +0.045 EVAPORATED on holdout (-0.005, p=0.83);
  corridor_mi ties/beats adjoint everywhere. RETRACTED as a positive claim; kept as a
  measured scope boundary. Figures: `sensing/viz.py` -> pipeline_seed1.png, summary.png.
- Live MPPI elites (`bench/elites.py`, n=100): real elites have ~1/5 of synthetic hybrid
  diversity (never degenerate); adjoint-vs-entropy survives (p<=1.3e-4); adjoint-vs-
  corridor survives only clean noise. **Candidate-set diversity is the moderator of every
  edge-over-strong-baseline in this project.**

## 3. RISK thread (the characterisation + the Clark revival)

Refuted for risk MAGNITUDE (each with mechanism, all Taylor/pointwise-family):
1st-order FOSM; 2nd-order FOSM; soft contact gradients; sub-cell refinement; analytic
Hessian (piecewise-zero); cap-pooling; bundled adjoint (`bench/bundled.py` — fixes the
support catch-22, 0.60->0.83 mass on high-sigma cells, but single draws explode to 5e10 at
sampled contact switches, and never beats its own forwards used as plain MC at equal
draws); map-error certificates (`bench/certify.py`, [PRE-REG] n=100: never beats cost-gap
or bracket; `bench/rare.py` rare-event testbed: FORM/attack-IS miss by ~2 dex while
derivative-free subset simulation matches truth at 0.1 dex — mechanism: decisions fail
through NONLINEAR VARIANCE ASYMMETRY, the true failure modes are not near the linearized
boundary). What works cheaply: Sigma_t sigma_t (STEP-form); the belief±sigma BRACKET
(beats STEP at ranking p=1.6e-3; bounds already published by elevation_mapping); sampled-
map CVaR. Rules: aggregate risk over TIME not cells; RMS not max under footprint; Jensen
optimism of mean maps E[J]-J(belief) ~ +1.4 (quantified, uncorrected by 2nd order).

**CLARK (`bench/clark.py`, `bench/clark_full.py`) — the constructive analytic result.**
Moment propagation through the envelope max via Clark 1961 recursion (SSTA-style),
settle linearized to an exact 3x3 tripod map (gate-verified 3.4% median), correlated cell
covariances from the noise kernel (gate-verified 0.6%). [PRE-REG all pass, n=100
hybrid/all, settle-only cost]: sd-ratio vs MC median 0.927 IQR[0.89,0.97] (FOSM was
1.7-2.6); Jensen-bias capture corr 0.884; clark_cvar regret 0.042 vs step 0.181 (p=8e-5),
bracket 0.122 (p=1.3e-3), fosm 0.110 (p=0.07 n.s. on this reduced cost); picked-best 72%.
Budget curve: Clark ~= MC-with-128-256 draws. Wall honesty: Python Clark 8.5 ms/plan vs
GPU-batched 256 rollouts ~6 ms — needs a Warp kernel to win wall time (trivially parallel;
assessed feasible). CLEAR_SOFT HINGE EXTENSION FAILED its gate (~2.4x E-overshoot on the
belly-hinge term — likely missing correlation between the 18 belly points and the settle
pose; OPEN PROBLEM) -> full-cost run fell back to settle-only per pre-registered rule.
Literature: verified via OpenAlex full citation scan — Clark 1961 has ZERO robotics citers;
closest prior art Fankhauser RA-L'18 (supplies the Gaussian field, no max propagation) and
STEP (same risk quantity, by sampling). CAVEAT: Ono/JPL chance-constrained rover line was
NOT searchable that session (no IEEE/Scholar access) — must be checked before claiming.
FINAL VERDICTS (landed before handoff close):
ROBUSTNESS (clark_full.json): Clark is the only estimator never-bad across all three
regimes — hybrid/all 0.042 (best by far), fan/sensor 0.085 (near-best while FOSM COLLAPSES
to 0.583, clark beats it p=4e-11), hybrid/clean 0.129 (competitive; fosm best there, n.s.).
Beats the bracket significantly in ALL THREE regimes (p=0.0013/0.021/0.017) — the only
method to do so. vs STEP: decisive on hybrid/all only (n.s. elsewhere). The robustness
story is CONSISTENCY: FOSM swings best-to-catastrophic by regime; Clark stays calibrated.
GRADIENT VALIDITY (clark_grad.json, [PRE-REG]): FAILED for magnitudes — criteria (i) and
(ii) missed (median rel err 48% at 1cm, 75% at sigma; bar was 10%/25%). Even Clark's own
FUNCTION difference has ~56% error on single-cell cm perturbations: a one-cell change moves
E[J] by ~0.01% of J, below the recursion's approximation floor. So the per-cell MAGNITUDE
program is closed at ALL THREE levels tested (hard adjoint, bundled adjoint, analytically
smoothed adjoint) — Clark is a PLAN-level instrument (E, Var, ranking), not a cell-level
one. Partial positives, underpowered: attribution ranking tau 0.94 vs hard's 0.80 (n=1
seed — needs replication before any claim) and the catch-22 improves (38% of gradient mass
on unobserved cells vs hard's 15%).

## 4. DFL thread (decision-focused perception training) — method dead here, insights stand

Arc (all in `bench/dfl*.py`, jsons alongside): linear pilot p=0.006 -> methods red-team
reverse-engineered it (MSE + ONE tuned coefficient on distance-to-observed recovers 83% of
the gap, p=0.18 indistinguishable; "shaped error" mostly a distance artifact; decision-vs-
cost_space n.s. p=0.099; seeds virgin for weights only; "pre-registered" was
pilot-informed) -> capacity ladder (`dfl_capacity.py`): softmin decision loss DIVERGES at
capacity (7m-RMSE degenerate fills, worse than zero-fill); cost-space loss survives and
scales (rung-2 0.257 vs MSE 0.582, p=0.003); MSE is FLAT in capacity (conditional-mean
fills are the problem — irreducible-ambiguity theory confirmed) -> settling tests
(`dfl_settle.py`): cost-space beats the fair distance baseline at capacity (p=0.024, tweak
recovers 58%) BUT on real elites: n.s. at choice level (p=0.088) and **REVERSES at
generation level** (planner planning ON the cost-space fill: MSE better, p=0.0097).
**THE DUAL-CONSUMER INSIGHT (most publishable sentence of the thread): the planning map
has two consumers — candidate GENERATION wants geometric fidelity, candidate CHOICE wants
decision shaping — and optimizing one map for choice harms generation.**
Supporting findings that stand: inert-fill mechanism (partial: low fill halves unobserved
support mass 21.7% vs 44.1%, shift-control near-dispositive; no per-seed regret link);
observability pilot (`bench/observability.py`): **75% of decision-relevant terrain mass
(95% among unobserved cells) has exactly-zero gradient in the driven trajectory's outputs**
— trajectory self-supervision (MonoForce-style) cannot supply what planning needs;
hindsight labels must come from later SENSING, not contact. Evaluation ladder designed
(prospective contact verification, mowed-plot gold set, payload falsification, randomized
paired trials) — NOT executed. DFL novelty review never delivered its synthesis (its
sub-searches ran; report lost with the session if it never filed — reconstructible via the
threads named in DFL_CLAIMS prior-art section).

## 5. The meta-findings (candidate paper-thesis material)

1. **Pointwise-vs-averaged law**: gradients through contact fail when trusted at a point
   (8 refutations), work inside averaging consumers (Clark's E[J]; possibly its gradient —
   clark_grad pending).
2. **Headroom law** (sensing): edge over masking = fraction of oracle-minus-masking gap.
3. **Candidate-diversity moderator**: every edge-over-strong-baseline shrank on real
   low-diversity elites (sensing, DFL — twice).
4. **Discover-then-compress (x4)**: adjoint support -> corridor mask; CVaR -> bracket;
   decision-training -> distance discount; sensing oracle gap -> selection-not-budget.
   The differentiable machinery finds behaviors; cheap rules ship them.
5. **Dual-consumer** (DFL): generation vs choice need different maps.

## 6. Ship-ready practical items for the robot (independent of papers)

- Corridor-masked sensing score (uncertainty x plan-proximity) — best cheap look-picker.
- The belief±sigma bracket as an MPPI risk term (2 rollouts; beats STEP-form at ranking).
- Non-saturating sigma field (`bench/noise.py` model) — the old cap was degenerate.
- Ground-referenced/distance-aware blind fill (the phantom-plateau class of bugs; the
  measured caveat: aggressive low-fill corrupts candidate generation — dual-consumer).
- min N_i is NOT a linearisation flag (tripod geometry; use contact_margin instead).

## 7. Process rules (hard-won; keep them)

- Holdout-before-headline: design on one seed range, report virgin seeds, never iterate
  after peeking. Caught two mirages (+0.045 sensing; certificate optimizer overfit).
- Corridor/simple baselines PRIMARY; report medians + oracle gap + tie counts beside means.
- Family-wise correction on p in [0.005, 0.05]; noise arms are pseudo-replication.
- Verify every citation by fetching (the ADO hallucination; the 2604.01434 mis-cite trap:
  it is solver-compute allocation, NOT VOI sensing).
- Seed hygiene: "virgin" must cover design choices, not just weights (documented failure).
- Rel-error metrics near zero baselines lie (Clark Gate-2 artifact) — use absolute +
  correlation metrics too.
- Agent fleet economy: Sonnet for coding/review/search, Haiku for mechanical runs,
  Opus/top-tier only for hard reasoning. Red-team every claims doc before believing it.

## 8. Immediate next steps (priority order)

1. Read `studies/out/bench/clark_full.json` (robustness) and `clark_grad.json` (gradient
   validity) — both were in flight at handoff; fold verdicts into CLAIMS.md. If clark_grad
   passed, the attribution/sensing scores should be rebuilt on the Clark gradient and the
   key sensing comparisons rerun.
2. Clark open problems: the clear_soft hinge correlation structure (the 2.4x overshoot);
   the Warp kernelization for wall-time victory; the Ono/JPL prior-art check; then decide
   whether Clark is its own short paper or CLAIMS.md's constructive risk section.
3. Update KAREL_BRIEF.md (add Clark + DFL settling verdict), then the USER talks to Karel.
4. USER field afternoon: same occluded ground twice -> real sigma calibration + hindsight
   labels. Gates every main-track submission.
5. Workshop draft -> LaTeX (paper/workshop/figs/ already has copied figures) after Karel.
6. Owed-but-deferred: catastrophe-cost CVaR rematch; calibrated-sigma DFL ablation rerun;
   route-level (lattice corridor) fill evaluation; clean seed block (5000+) revalidation.

## 9. File map (all committed unless noted)

Docs: studies/{HANDOFF,CLAIMS,DFL_CLAIMS,WORKSHOP_DRAFT,KAREL_BRIEF,FINDINGS,RESULTS}.md
Sensing: studies/sensing/{pipeline,viz}.py; out/sensing/{results,sweep,holdout}.json
Studies: studies/bench/{ranking,noise,policies,risk,order2,softgrad,compare_*,illustrate,
  bundled,certify,rare,elites,observability,plot_scope,clark,clark_full,clark_grad,
  dfl,dfl_ablation,dfl_mechanism,dfl_inert,dfl_elites,dfl_capacity,dfl_settle}.py
Adjoint: studies/adjoint/{study_a,study_b,harness,scene,sigma,curb_direction,...}.py
Outputs: studies/out/bench/*.json (figures *.png are gitignored — regenerate via scripts)
Memory (cross-session): ~/.claude/.../memory/adjoint_sensing_paper_status.md + MEMORY.md
