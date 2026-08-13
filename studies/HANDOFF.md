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

## 10. Update — 2026-08-07 paper session (both gates closed, draft written)

**The paper now lives in its OWN repo: `~/projects/clark_paper`
(github.com/aleskucera/clark-paper, private), built from the axion-paper template
(IEEEtran, `compile.sh`, `check_pdf.sh`, anonymized for double-blind RA-L). This
repo keeps the evidence; that one keeps the prose.** Decisions taken by the user:
outline A (Clark-led), venue RA-L with no hard deadline, sensing thread reduced to
one discussion paragraph, simulation-study framing with a possible sigma upgrade
before submission.

**GATE 1a — Ono/JPL prior art: CLOSED, the novelty claim SURVIVES.** The
chance-constrained rover line and its neighbours all propagate worst-case interval
bounds through the contact kinematics (ACE, Otsu et al. JFR'20), fit an empirical
Gaussian to those bounds by offline MC plus linearization (p-ACE, Ghosh/Otsu/Ono
IROS'18 — closest by PROBLEM), or reach closed form only for a quadratic form and
fall back to a heuristic where a max appears (Tomita & Ho AAS 23-391 — closest by
MACHINERY). Nobody propagates moments through the max. Full record, including the
narrowed claim wording and what could not be reached (IEEE Xplore, two Ono papers
behind auth walls): `~/projects/clark_paper/PRIOR_ART.md`.

**GATE 1b — the clear_soft hinge: the model defect is FIXED; the decision benefit
is NOT significant.** (`bench/clark_hinge.py`, jsons `clark_hinge{,_design_all,
_virgin,_stage2}.json`.) Root cause of clark_full's 2.2x E-overshoot was
approximation (e) exactly as suspected, in two parts: the frozen pose drops the
envelope max's own Jensen uplift of the chassis, AND it drops the positive
correlation between the ground under the belly and the chassis riding up on it.
Fix: `w_z = a_i . e_t + const` with `a_i = M[z] - px_i M[pitch] + py_i M[roll]`
composed with settle_map's tripod map, so the hinge argument becomes
`const + g.U - a.N` and both terms fall out of covariances clark.py already
computes (no new fold, no new MC). A SECOND structural revision was required: the
same-timestep-only restriction on approximation (f) had been calibrated against
the buggy frozen model, so with (e) repaired it flipped from compensation to
deficit; reinstating all pairs as one global Phi-weighted quadratic form is both
correct and cheaper. Virgin cases (n=12): E-ratio 2.225 -> 0.775, |err|/mc_sd
5.91 -> 1.08, sd-ratio 0.825, corr 0.915. Gate H nonetheless FAILS its own
criterion (iv) (corr >= 0.93) — recorded as a defect in the BAR, not relaxed: the
frozen comparator scores 0.905 on the same virgin cases and n=12 Pearson has a 95%
CI of [0.72, 0.98]. Declared residual: the rollout TRAJECTORY is still frozen at
the belief path, which explains both the ~22% E undershoot and the 0.83 sd-ratio.
STAGE 2 (full cost, n=100, hybrid/all, run as EXPLORATORY after the user
authorized it): clark_cvar has the lowest mean regret 0.347 (bracket 0.465, fosm
0.467, step 0.501), the lowest median (0.000) and the best picked-best rate (54%),
but the paired sign tests do not separate (33/59 vs step p=0.43; 39/69 vs bracket
p=0.34). PRE-REGISTERED CRITERION FAILS -> the paper's headline stays on the
SETTLE cost and reports the extension as calibrated-but-not-decisive.

**Numbers that changed on recomputation from the JSONs** (the paper uses these;
older summary text in this file and in CLAIMS.md is superseded where they differ):
clark_grad's per-cell errors are 50% at 1 cm and 83% at sigma (not 48%/75%); its
attribution tau is 0.912 vs the hard adjoint's 0.894 over 10 seeds (not 0.94 vs
0.80 at n=1); its catch-22 gradient mass is 0.48 vs 0.30 (not 38% vs 15%); the
"bracket beats STEP (p=1.6e-3)" claim is specifically about Kendall TAU (+0.110,
76/117) and CVaR error (p=1.3e-3) — on REGRET the two are indistinguishable
(p=0.37), so the paper states the metric explicitly.

**Figures** regenerate from `bench/plot_paper.py` (Jensen explainer computed on
real footprints of the benchmark's own belief map; budget curve from
clark_full.json). The Jensen figure also exposes something a schematic would have
hidden and the paper now states: Clark over-predicts the PER-NODE Jensen gap by
~30%, against a first-order alternative that predicts no gap at all.

### 10.1 Corrections found while writing the paper (2026-08-07, later)

Three adversarial referee agents reviewed the draft (novelty / methodology / significance).
The methodology referee recomputed ~50 numeric claims from the JSONs and confirmed them
exactly; chasing its findings turned up two errors that were ours, not the paper's:

1. **`clark_full.bench_wall_costs` is broken and its 8.48 ms/plan is ~5x optimistic.** It
   reads `h.sim.controlled` BEFORE `h.forward()`, so it timed the Clark fold on the
   pre-rollout buffer: an all-zeros trajectory parked at the origin, where all 40 timesteps
   share ONE footprint (measured: 1 distinct xy vs 41, path length 0.0 m vs 4.7 m) and the
   candidate universe collapses. A warning comment is now in that function; the code and its
   json are left intact so the committed artifact stays reproducible. **Corrected numbers
   (`bench/clark_fast.py`, `out/bench/clark_fast.json`, idle machine, medians over 10 seeds x
   16 plans): clark.py 42.9 ms/plan, MC-256 6.07 ms/plan measured in the same run.** Any
   future wall-time claim must come from clark_fast, not from clark_full's `wall` field.
2. **The cap-pooling refutation was imported from the wrong study.** Its p=0.014 comes from
   `ranking.py::p_magnitude_pooled` -- a SENSING policy about where to point a look budget --
   not from any risk-magnitude experiment. It is out of the paper's ledger, which is now
   SEVEN routes, not eight. HANDOFF §3's list should be read the same way.
3. Smaller: `rare.json` says FORM misses by 2.97 decades (not ~2) and attack-seeded IS is
   censored on 16/16 cases (no finite estimate at all, not "~2 dex"); the "3.4 sigma vs 0.4
   sigma" iid-vs-correlated claim in FINDINGS §2.3 is not reproducible from study_b.json's b1
   under any single aggregation (it is scale-dependent: 0.34/0.03 at sigma_scale 0.03 rising
   to 13.2/1.19 at 3.0) -- the paper cites its own figure's measured 2.4x instead.

**NEW AND USEFUL: `bench/clark_fast.py` makes the estimator 3.7x faster with BIT-IDENTICAL
output** (max |dE| = max |dVar| = 0.000e+00 over 10 seeds x 16 plans). `clark_build` widens
its tracked covariance vector with the node's own K candidates so the recursion can read
Cov(running max, candidate i) -- but every candidate IS a universe cell, so that number is
already in the universe part of the vector. Dropping the redundant block means the
[3T x K x |U|] tensor (~70 MB/plan) never has to be built, sorted and concatenated. 42.9 ->
11.6 ms/plan. Still 1.9x slower than 256 batched GPU rollouts, so the Warp kernel remains the
open item for a wall-time claim -- but it is now a 2x gap, not a 7x one.

### 10.2 Optimization (2026-08-08): the estimator is now faster than sampling

`bench/clark_conv.py` (settle) and `bench/clark_hinge_fast.py` (full cost) rewrite the
estimator around two structural facts that the original implementation did not use:

1. **A moment-matched max node IS a fixed linear functional of its own candidates.** Clark's
   covariance recursion unrolls to `Cov(node,·) = Σ_i w_i Cov(X_i,·)` with
   `w_0 = Π Φ_i`, `w_i = (1−Φ_i) Π_{j>i} Φ_j`, `Σ w_i = 1` (verified against `clark_build`
   to 8e-16). The fold therefore needs only the node's own K×K covariance and can emit K
   weights; the [N, K, |U|] tensor is unnecessary.
2. **`rho_lookup` is separable with hard-zero support** (`ρ(dy,dx) = ρ1(dy)ρ1(dx)`, zero past
   11 cells). Any linear functional of cells has `Var = <G, (G*ρ1)*ρ1>` — scatter onto the
   corridor patch, two 11-tap convolutions. No covariance matrix is ever built.

O(3TK|U| + |U|²) → O(3TK² + P). Measured on an idle machine, medians over seeds × 16 plans:

| path | before | after | vs MC-256 (6.07 ms/plan) |
|---|---|---|---|
| settle, sphere | 42.9 ms | **1.6 ms** (batched over the candidate set) | 3.8× faster |
| settle, cylinder | — | 0.95 ms unbatched | 6.4× faster |
| full cost, sphere | 50.9 ms | **9.6 ms** | 1.6× slower |
| full cost, cylinder | 49.4 ms | **2.6 ms** | 2.3× faster |

Output is unchanged: E exact, Var to 5e-16 (settle) / 1.2e-15 (full cost). **The paper's
wall-time caveat is now a claim.** Three ingredients, in order of payoff: the weight/convolution
collapse (16×), the cylinder element (K 37→5-7, another 2.7×), batching the fold over the
candidate set (1.6×). Also `_rho` folds abs/clip/where into a zero-padded signed-lag table —
that alone was 77% of the first hinge version's runtime.

**Two process notes.** (a) `clark_hinge_fast` verifies against `clark_hinge` on every case
before it reports any timing; the cylinder's delta is PHYSICS and is emitted under a separate
json key so it can never be read as accuracy. (b) Clark was **91% of a stage-2 seed's cost**
(the 256 MC draws batch into 0.11 s/seed), so the full-cost head-to-head at n=500 costs ~2
minutes now and cost ~10 before — it was never compute-limited, which means the
non-significant p=0.43/0.34 result can be settled at higher n whenever wanted. IF THAT IS
RUN: the n=100 test already failed its pre-registration, so a larger-n rerun is a NEW test
that must be pre-registered separately and reported alongside the original failure, never in
place of it.

### 10.3 Where the belief comes from (2026-08-08/09) — the noise model stopped being invented

This is the largest change to the project's foundations since Clark itself, and none of it is in
the paper yet. New code: `sensing/lidar_belief.py`, `sensing/subcell_relief.py`,
`sensing/bag_belief.py`, `bench/clark_conv.py` (rank-M), `bench/realistic_sigma.py`,
`PREREG_realistic_sigma.md`. Outputs alongside in `out/bench/`.

**THE MOTIVATION: the study was circular.** Estimators were handed a belief, and the MC "truth"
they were graded against sampled from THAT SAME hand-built noise model. Every arm was being
scored on how well it predicted a distribution we wrote down.

**1. Simulate the sensing instead of assuming it** (`lidar_belief.py`). A Warp ray-march of an
Ouster-like 1024x64 pattern against a true height field; noise corrupts the RANGE, not the
height (so a given error lands in z at normal incidence and slides sideways at grazing, which
no additive per-cell noise can express); dropout grows at grazing incidence; a 6-DoF drifting
pose error; and the survivors rasterized by the robot's own `HeightMapBuilder`. Truth is the
terrain we generated, so nothing is assumed twice. Measured, 32 realizations, one scene:
sigma 3.1 cm median (the study assumes ~10), correlation TWO-SCALE and anisotropic with a floor
past 2.5 m (the study assumes one separable kernel, zero by 1 m), separability error 0.48,
excess kurtosis +9.0, 21% of cells never observed.

**2. The bias decomposition, and a claim of mine that was wrong.** I first reported "79% of the
error is systematic and no covariance model captures it". That was overstated and the
diagnostics caught it. FLAT+noiseless gives bias exactly 0.00 (ray-marcher acquitted); halving
the march step changes nothing; removing dropout does not shrink it. What remains is mostly
DEFINITIONAL: a rasterized cell holds the mean height of the points in it (an area average)
while the planner reads it as the height AT the cell centre, and on 10 cm cells over 12 cm RMS
relief those differ by ~2 cm. Genuine systematic sensing error is ~1 cm against a sigma of 3 cm.

**3. Discretization CAN be modelled, through the max** (`subcell_relief.py`). Max is
associative, so max over the footprint = max over cells of the max WITHIN each cell; the current
model silently substitutes the stored value for the inner max. Validated against the true
footprint max on a 5x finer surface: the mean layer is optimistic by **bias/tau = -2.39, -2.29,
-2.22** across a roughness sweep, i.e. a LAW (bias ~ -2.3 tau), not a property of one terrain.
Feeding the fold the within-cell max removes the bias at every roughness.
  - WITHDRAWN: "just use the mapper's max layer". On a noiseless surface it was the best arm;
    with real returns it is the WORST (+20 cm) because one bad return owns a cell. A p90 of the
    same returns recovers most of it.
  - tau from returns: bias removed, but corr(tau_hat, tau_true) ~0.03 per cell. Two hypotheses
    were wrong (drift contamination; range/hit-count -- corr is ~0 even at 127 hits/cell). The
    RIGHT answer: tau_true's own split-half reliability is 0.21-0.50 on homogeneous fractal
    terrain, so there is no per-cell roughness structure to find. On terrain that HAS roughness
    structure (patchy amplitude, as real ground does), corr rises to 0.37 per cell and
    **0.62 / 0.71 / 0.78 pooled over 3x3 / 5x5 / 9x9** against a reliability ceiling of 0.85.
    So sub-grid roughness is recoverable at PATCH scale (0.5-0.9 m), which is the scale it
    exists at. A within-cell max cannot be pooled; a variance can. That is the argument for
    modelling the discretization rather than measuring it.

**4. Real-bag robustness** (`bag_belief.py`). Reads mcap with `rosbags` -- NO ROS, no container --
recovers the Ouster mount by fitting a ground plane (this bag has no tf_static for it),
accumulates 40 scans on /odom_2d. 9x9 m window: 49% observed, 33 hits/cell, ground at -1.23 m.
  - BUG FOUND AND FIXED: the estimator accepted NaN cells and returned non-finite moments for
    all 16 plans without complaining. `clark_conv` now refuses non-finite input.
  - CONFIRMED: zero-filling unobserved cells more than DOUBLES the spread between candidates
    (45.9 vs 21.7) -- the phantom-plateau mechanism reaches the risk term as a RANKING change.
  - ON RECORD: 16 plans at ~3.15 ms is ~50 ms per replan on a real map (26 ms batched), against
    a plan/replan budget the profiling put at 4.7 ms. Real integration problem.

**5. rank-M separable kernels** (`clark_conv.separable_terms`). The convolution optimization
assumes rho(dy,dx) = rho1(dy) rho1(dx). Under the measured kernel that single term gets Var[J]
wrong by **60%**. Any kernel is a sum of separable products: PSD-projected rank 5 holds Var[J]
to 0.57%. Truncation alone breaks positive semi-definiteness (min spectrum -4e-4), so alternating
projections (truncate to rank M; clip the Fourier spectrum at zero, per Bochner) restore it at no
accuracy cost. NOTE rank 3 scored WORSE than rank 2 on dVar despite a smaller residual on rho --
choose M by measuring dVar, never by reading the spectrum.

**6. THE BELIEF MODEL THAT FITS: Sigma = rank-3 plane + compact stationary kernel.** The first
generator gate FAILED and was right to: the realized correlation missed by 0.06 and the miss did
not shrink with draws. The measured kernel never decays inside its window because a scan's pose
error shifts and TILTS every point together, and no stationary kernel can express that. Split it
off: **plane 30%, stationary 70%**, and the residual decorrelates within three cells
(rho +0.022 at 1 m, -0.004 at 2 m). Both halves are cheap AND EXACT for a linear functional of
cells -- the plane is a 3x3 form on (sum G, sum G x, sum G y), the residual uses the rank-M
convolution. Fitted parameters in `out/bench/belief_model.npz` (sigma_offset 1.4 cm, tilt sigma
~0.005 rad/m).

**7. THE RESULT** (`realistic_sigma.py`, criteria pre-registered and committed BEFORE the run).
Gate is end-to-end -- does the variance the ESTIMATOR predicts for a linear functional match the
variance the GENERATOR realizes? -- because the first attempt passed a kernel check and was still
wrong. Median error 2.2%. hybrid/all, n=100, mean regret:

    clark_cvar  0.0500   vs none    -0.2491  49/62  p=4.8e-06
    clark_mean  0.1053   vs step    -0.3110  58/67  p=6.8e-10
    bracket     0.1273   vs bracket -0.0773  28/40  p=1.7e-02
    fosm        0.2006   vs fosm    -0.1506  39/50  p=9.0e-05
    none        0.2991
    step        0.3610   pooled sd-ratio 0.911

**ALL FOUR PRE-REGISTERED CRITERIA PASS.** Our regret barely moves (0.042 -> 0.050) while every
arm that discards correlation degrades badly: STEP-form 0.181 -> 0.361, now WORSE than planning
on the mean map; FOSM 0.110 -> 0.201. That is the mechanism -- the advantage comes from consuming
a correlation structure the alternatives throw away, not from a noise model tuned to suit us.
Predictions recorded beforehand: two right (longer correlation helps us; the bracket becomes the
closest competitor), one WRONG (I predicted calibration would be the casualty; it passed most
comfortably at 0.911).

**CAVEATS.** The belief model is fitted to ONE terrain and ONE straight traverse, so the 30/70
split and the tilt magnitudes are that scenario's. The plane-within-footprint approximation
treats the plane's correlation as constant over 0.9 m (true to a fraction of a percent, stated
in the code). Vegetation, multi-echo, wet surfaces and dust are not modelled at all.

**WHAT THE PAPER DOES NOT YET CONTAIN — none of section 10.3.** In particular it still claims
the single-separable-kernel convolution (60% wrong under a measured belief; rank-M is the
repair), still reports the comparison under the invented sigma only, and says nothing about the
plane/stationary decomposition, the discretization law, or the real-bag robustness fix. Deciding
how much of this belongs in an 8-page RA-L versus a follow-up is the next call to make.

## 11. The 2026-08-10 defect session — 13 fixes, full rerun, and the verdicts that moved

An adversarially-verified code review of studies/bench + studies/sensing found 18 confirmed
defects; the 13 that corrupt numbers were fixed in commit 8e0aef5 (worst: occlusion rays
marched from the GRID CORNER instead of the sensor, so every occlusion/localisation/all
noise arm — including every clark*.json — was computed on wrong observation masks; also a
half-cell pose-resample shift, an in-sample budget curve, a frozen CVaR baseline rng, and
kendall_tau computing Goodman-Kruskal gamma instead of tau-b, inflating every tied tau).
`studies/RERUNS.md` is the ledger: fix -> stale artifacts -> rerun order. Reruns are
POST-HOC CORRECTIONS of pre-registered results: the corrected number is reported beside the
original everywhere, never in place of it. Numbers below supersede §3/§10 where they differ.

**Belief foundation (commit db26720) — stable, one qualitative correction.** Sigma median
3.11 cm, 80% observed, kurtosis +10, bias ~78% of MSE and still mostly definitional. The
36-scenario sweep keeps "the plane is sensing, the share is terrain", but the tilt
ANISOTROPY on curved paths flips orientation under the fixed body-frame drift: arc/turn
concentrate tilt into body-x (6.1 -> 8.0 mrad/m; body-y 7.9 -> 6.6). Straight rows are
bit-identical (yaw=0 no-op check). The belief model finally has a producer
(`sensing/fit_belief_model.py`, atomic --write, anti-circularity variance check); refit:
plane share 0.24 (was 0.30), sigma_offset 1.16 cm (was 1.40), tilts/residual rho in band.

**Clark chain (commit 0021b98).** Calibration UNTOUCHED: sd-ratio 0.927 (same to three
digits), Jensen corr 0.786 (was 0.884; bar 0.7). Robustness keeps its shape: FOSM still
collapses fan/sensor (clark beats it p=4e-11), bracket beaten in two regimes (p=0.021/
0.017). WEAKENED: settle-only decisive superiority was partly the bug's — clark_cvar vs
step p=8e-5 -> 0.079, vs bracket 0.0013 -> 0.24 (regret 0.077 vs 0.129 still first,
picked-best 69%). clark_grad still fails the magnitude bars (46%/73% vs 10%/25%) — Clark
stays a plan-level instrument; attribution tau 0.873 vs hard 0.637 (tau-b). FLIPPED: Gate H
passes all four criteria on the design set (corr 0.957 vs bar 0.93; sd-ratio 0.827 vs old
0.426) and stage 2 — same config as the twice-failed run — is decisive: clark_cvar 0.148 vs
step 0.419 (p=3.1e-6), bracket p=4.7e-4. CAVEATS on the flip: the committed pre-fix gate
json predates the config keys (its case set is ambiguous), the regenerated virgin-set json
came back bit-identical (under investigation), and a same-seed rerun after a code fix is
not a confirmation — hence:

**PREREG_stage2_virgin.md (commit 7705708), committed BEFORE its run:** stage 2 on virgin
seeds 5000-5099 (verified untouched repo-wide), criteria frozen (lowest mean regret; beats
step p<0.05; beats bracket p<0.05), predictions recorded, one shot, reported alongside both
prior results whatever it says.

**Measured belief (commit 957f40b) — the result that matters most survived its own
repair.** All three arms pass all four pre-registered criteria on the refit model:
hybrid/all clark_cvar 0.045 vs step 0.214 (p=7.3e-7), bracket p=0.011, fosm p=4.3e-4,
sd-ratio 0.886; hybrid/clean step p=1.6e-5; fan/sensor fosm p=6.8e-17. The held-out budget
curve (in-sample scoring was one of the 13 bugs) STRENGTHENS the claim: MC bottoms at
0.070 regret at 128 draws, above clark's 0.045 — n_star is None; the old N*=128 was the
artifact. Where the invented-noise settle benchmark lost step-significance, the measured
belief keeps clark decisive against every baseline: the paper's center of gravity moves
here.

**Sensing thread (commit 5f13dcb) — the retractions retract nothing new.** Under fixed
masks and tau-b: entropy still loses to every informed policy (to p=4.5e-20), corridor
still ties/beats the adjoint on realistic settings, real elites still compress every edge,
motion-coupled holdout still null (p=0.57-1.00). Occlusion/localisation arms have correct
masks for the first time. NOTE: clean/sensor ranking arms (C1's +0.691/+0.471 headline
class) still carry gamma taus — rerunning in stage 5; every tau shrinks where costs tie.

**The virgin verdicts (commit 730ef00).** Gate H on virgin cases PASSES all four criteria
under the fixed noise (corr 0.961 vs the 0.915 fail on record; the earlier bit-identical
virgin json was the stage-2 flow not recomputing it). The pre-registered virgin stage 2
(seeds 5000-5099, PREREG_stage2_virgin.md) FAILS its composite: lowest mean regret (0.196
vs step 0.393, bracket 0.335) and beats step p=0.044, but does not separate from the
bracket (p=0.31) — exactly the criterion the prereg predicted was at risk. Per the frozen
rule the full-cost claim is CALIBRATED BUT NOT INDEPENDENTLY CONFIRMED; the paper headline
stays on the settle/measured-belief results and reports the trajectory whole.

**Stage 5 (commit 4c71c94) — one reversal that outranks the bookkeeping.** The DFL
thread's single surviving positive (decision-loss training beats MSE, p=0.006) REVERSES
under the fixed noise: decision-trained loses to MSE at p=3.9e-10 (9/68 wins); cost-space
marginal (p=0.019). dfl_settle's rerun rc=1 by its own reproduction guard (retrained
rung-2 regret 0.157 vs recorded 0.257) — the training landscape moved with the fix.
CLAIMS.md rev 3 retracts the section. CONTROL: ranking hybrid/clean returned
BYTE-IDENTICAL to the committed artifact — the clean arm never touches the fixed paths and
tau-b equals gamma absent exact ties — so C1's clean-arm headline numbers stand unchanged.

**Refutation-ledger triage (2026-08-11, correcting the stage-5 commit message's "keep
their verdicts" line, which was wrong for one row):**
- **BUNDLED ADJOINT: REVERSED.** Same arms, same protocol, recomputed from both jsons:
  pre-fix bundled-vs-mc_small p=0.121 ("never beats its own forwards" — the refutation);
  post-fix mean −0.32, 35/44, p=1.1e-4 — it decisively BEATS them. The 0.61→0.83 catch-22
  and 5e10 blow-up figures have no counterpart in the current json (gradmass 0.052→0.062,
  max cvar_err 3.76). The paper's ledger now counts SIX refutations plus one reversal;
  a corrected-benchmark bundled-vs-Clark head-to-head has NOT been run (open item).
- Second-order: refutation STRONGER — fosm2−fosm1 paired Δτ −0.125, better on 7/40,
  p=3.7e-7 (was "does not improve, p=0.13"); ratio medians 0.78→1.54 (means outlier-blown
  by one near-singular contact — quote medians).
- Softened contact: direction holds (no temperature helps, all p≥0.27 vs hard) but the
  old "−0.13 / ~50× too small" figures have no field in the current json — superseded.
- rare/FORM: my stage-5 message said "1.99 dex with stalls censored" — imprecise. The
  overall median miss is 2.97 dex (UNCHANGED); 1.985 is the P<1e-2-regime median; the
  stall-censoring fix is real in code but dormant on this data (0/16 attacks stalled).
- bag_belief: zero-fill spread 45.9 vs 11.5 ground-referenced (phantom-plateau confirmed);
  NaN maps now refused.
- separable_report.json (NEW artifact + bench/separable_report.py): the paper's separable-
  kernel numbers finally persisted, measured on the refit kernel: worst-case separability
  0.14 (was 0.48), single-term Var[J] error 7.1% (was 60%), rank-5 1.4% (was 0.57%); the
  production assumed kernel (CORR_LEN=0.15) is 148% wrong against the measured stationary
  kernel — noted for any future default change. Paper edit list: scratchpad paper_number_inventory.md (~85 claims, 21/27
spot-checked numbers drifted; two framing flips: hinge verdict negative->positive pending
the virgin run, and "STEP worse than the mean map" no longer holds — STEP 0.214 vs none
0.313). Writing standard for the paper: clark_paper/CLAUDE.md (strict academic register,
every claim proven or cited, independent reviewer panel before finalization).

## 12. The de-sphering recompute (2026-08-13) — engine and element, separated

Campaign: merged engine (bbc98cf) at n=100, cylinder as candidate primary + sphere-on-
merged-engine baselines. All 15 runs rc=0; artifacts committed; logs log_desphere_*.

**Three-way, realistic_sigma hybrid/all (mean regret):**
old record: clark 0.045, step 0.214 (p=7.3e-7), bracket 0.098, PASSED.
merged-SPHERE: clark 0.073, step 0.256 (p=0.0145), PASSED — but bracket p=1.0, fosm
n.s.: the ENGINE change alone already costs most of the extra-baseline decisiveness.
merged-CYLINDER: clark 0.160, step 0.285 (p=0.081) — criterion iii FAILS, VERDICT
FAILED; bracket p=0.49, fosm p=0.34. All arms' regrets grow; separation collapses in
this one regime (clark still lowest by a wide mean margin).

**But cylinder is NOT weaker across the board:**
- hybrid/clean and fan/sensor: ALL FOUR criteria PASS decisively (step p<=0.0046,
  bracket <=0.0022, fosm ~0).
- Calibration IMPROVES under cylinder: pooled sd-ratio 0.955-0.971 vs sphere's 0.886.
- Full cost (hinge): Gate H passes on design AND virgin sets; stage 2 hugely decisive
  (vs step p=4.5e-7, vs bracket p=6.5e-7, mean diffs ~-1.1). Under the true geometry
  the FULL cost separates sharply while settle-only hybrid/all does not.
- clark settle benchmark: clark_cvar remains lowest-regret in all three worlds.
- conv timing: cylinder 1.13 ms/plan (40x) post-merge. (conv's rel-dE/dVar fields vs the
  sphere reference are the PHYSICS delta, not error — same caveat as ever.)

**Reading:** the story survives (ordering, calibration, 2/3 regimes, full cost) but the
flagship "all three regimes pass" sentence does NOT survive a straight cylinder re-base:
hybrid/all loses step-significance (0.081 at n=100), and the engine merge alone had
already thinned it (7.3e-7 -> 0.0145). Paper options now live with the author:
(a) cylinder re-base with a reframed headline (calibration 0.96 + 2/3 regimes + decisive
full cost — arguably a BETTER story: under the deployed geometry the deployed objective
separates); (b) paper stays on the frozen sphere/old-engine record (valid, reproducible,
in git), robot and records live on cylinder. Pre-reg bookkeeping either way: these are
configuration-change reruns, reported beside the originals, never replacing them.
