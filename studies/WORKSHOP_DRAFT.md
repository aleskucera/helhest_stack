# When Does Sensing Need a Derivative?
## Decision-focused sensing and the limits of uncertainty propagation through contact

*Workshop draft v1, 2026-08-06. Markdown master for a 4-page workshop submission; numbers
final per `studies/CLAIMS.md` rev 2 except [ELITES], pending the live-elite experiment.
Figures: `studies/out/bench/scope_law.png` (Fig. 1), `studies/out/sensing/pipeline_seed1.png`
(Fig. 2), the study-B stratification plot (Fig. 3), `studies/out/bench/methods_seed7.png`
(Fig. 4, optional).*

### Abstract

Off-road robots plan on elevation maps carrying per-cell uncertainty, and use it two ways:
penalise it (risk-aware planning) or reduce it where largest (information-theoretic active
perception). We characterise, on a differentiable quasi-static contact model, when each use
can exploit the map's *derivative* structure — and when it cannot. For **risk**, the answer
is never at realistic map noise: the plan-cost gradient's validity radius is set by
contact-switch slack (millimetres) while map sigma is centimetres; seven repair routes
(second-order, soft contact gradients, sub-cell refinement, analytic Hessian, cap-pooling,
and bundled randomized-smoothing adjoints — the last failing by heavy-tailed blow-up at
sampled contact switches) each fail with a measured mechanism, while a two-rollout bracket
on belief±sigma outperforms a per-step Gaussian CVaR at ranking. For **sensing**, the
gradient's *support* — which cells the choice among candidate plans rests on — is
informative where its magnitude is not: sensing directed by candidate-set cost-gradient
structure beats entropy-directed sensing at every budget and noise level (up to +0.69
Kendall tau, p=2.5e-60, n=200). Against the strongest gradient-free baseline, however —
uncertainty masked to the candidates' corridor — the derivative's edge follows a **headroom
law**: it captures 30–70% of the gap between corridor masking and an oracle, and vanishes
whenever that gap does — at budgets too small to matter, at budgets large enough that
masking saturates the achievable, for wide-FOV body-fixed looks (a claim we retracted
after our own pre-registered holdout), and on the live planner's own low-diversity elite
sets under realistic noise, where the residual edge over masking vanishes entirely. The
value of the derivative in sensing is thus real but conditional, and we give the condition.

### 1. Setting and contributions

Robot: skid-steer ground vehicle; 2.5D elevation belief h with per-cell sigma (occlusion,
sensor noise, pose error); a differentiable quasi-static settle maps (pose, h) to (z, pitch,
roll) through a wheel-envelope contact arg-max; plan cost J sums tilt/clearance terms over a
rollout. The adjoint dJ/dh attributes a plan's cost to individual map cells (validated
against finite differences on non-uniform terrain to ~1% worst-case).

Contributions: **(C2)** a validity-radius criterion for first-order uncertainty propagation
through contact, with seven measured refutations of gradient-magnitude risk estimation;
**(C1)** decision-focused sensing — candidate-set adjoint support × sigma as the sensing
score — with its scope mapped honestly against the strongest gradient-free baselines,
including a pre-registered holdout that retracted our own motion-coupled claim; **(C4)**
transferable risk-hygiene findings (aggregate over time not cells; RMS not max; the
quantified Jensen optimism of mean-map planning, E[J]−J(belief) ≈ +1.4; the belief±sigma
bracket). All results n≥100–200, paired sign tests, reproduced end-to-end from committed
code in a clean worktree.

### 2. Why the magnitude fails (risk)

The envelope contact is an arg-max: the linearisation is exact within the contact slack
(mm) and meaningless across sigma (cm) — measured 2.3% gradient error at 1 mm perturbation,
32% at 1 cm. A structural catch-22 compounds it: the belief is flat where unobserved, so
|dJ/dh| is ~5× smaller exactly where sigma is largest. Consequences, each measured: FOSM
mis-ranks plans against Monte-Carlo truth (worse than a free per-timestep sigma sum);
second-order fixes variance accuracy (34%→3% badly-wrong cells) but not decisions (its
correction grows linearly with cells summed); softmax contact gradients are real but ~50×
too small to matter; and the bundled adjoint — averaging gradients over sampled maps, the
smoothing literature's recommendation — repairs the catch-22 (support mass on high-sigma
cells 0.60→0.83) yet explodes (single draws at 5×10^10 when a sample lands on a
near-singular contact) and, at matched compute, never beats simply using its forward
rollouts as a small Monte-Carlo (p=0.12). We relate this to the contact-gradient pathology
literature (Suh et al.; Parmas; DiffMJX) — scoped there to optimization, measured here for
UQ — and to hybrid-systems sensitivity: saltation matrices correct flow sensitivities across
time-domain guard crossings, whereas the settle's discontinuity is a *parametric* active-set
change (Fiacco), for which our validity radius is precisely the measured
distance-to-active-set-change. What works instead: sampled-map CVaR (affordable at planner
scale), and a deterministic two-rollout bracket on belief±sigma — using bounds the standard
elevation mapper already publishes — which beats STEP's per-step Gaussian CVaR at ranking
(p=1.6e-3) and 16-draw empirical CVaR at an eighth of its cost.

### 3. Why the support works (sensing) — and exactly when

Setting: K=16 candidate plans on a partially observed map; a policy reveals M cells; plans
are re-costed by rollout on the updated belief; score = Kendall tau of the re-ranking
against truth (n=200/arm; sigma field gate-verified non-degenerate: <1% of unobserved cells
at its ceiling, ~390 distinct values among any policy's top-400 picks).

**Uncertainty is not relevance.** Entropy-directed sensing is flat in budget (tau ≈ 0.12
clean / 0.07 noisy at every M) — its picks land metres off any candidate, where dJ/dh ≡ 0;
an unobserved region holding 67% of total map sigma² receives 0.000% of decision-relevant
weight because no wheel can reach it. Sensing by candidate-set adjoint support beats it by
+0.47 to +0.69 tau at M=400 (p ≤ 2.6e-35). The single-plan variant fails (confirmatory
bias, p=4e-27) — the candidate set is load-bearing, which is query-by-committee's classical
argument, here with plans as the committee and a physical adjoint as the disagreement.

**The headroom law (Fig. 1).** The strongest fair baseline is not entropy but
corridor-masked uncertainty — sigma (or Gaussian MI) weighted by proximity to the candidate
paths, no derivative. Against it, the adjoint's edge equals a consistent 30–70% of the
headroom the mask leaves below an oracle, and disappears with it: at M=25 on clean maps
(floor: even the oracle only reaches 0.19), at M=400 under full noise (ceiling: the adjoint
sits at 95% and the mask at 94% of the oracle), and — at sensor scale — for wide-FOV
body-fixed looks, where one cone reveals ~10^3 cells: our pre-registered holdout on virgin
seeds erased the design-set edge over entropy (+0.045 → −0.005) and showed corridor masking
tying or beating the adjoint (p=0.033). In between, the edge is real: clean +0.108/+0.210
at M=100/400 (p ≤ 5.4e-5), full noise +0.041/+0.061 at M=25/100 (p ≤ 3.2e-3), including
within path-identical candidate groups where the mask is constant by construction.

**On the live planner's own elites** — extracted from the production MPPI/CEM solving on the
belief map (n=100 full-noise, n=60 clean) — the picture sharpens once more. Real elites
occupy a diversity regime no synthetic family covered (~1/5 of the hybrid family's path
spread, on 100/100 seeds, yet never degenerate). The adjoint still beats entropy decisively
on them at every budget in both noise arms (p ≤ 1.3e-4). Its residual edge over corridor
masking, however, survives only under clean sensing (+0.176/+0.113 at M=100/400,
p ≤ 3×10⁻⁵) and is not statistically distinguishable from zero under the full noise model
at any budget (p ≥ 0.097): low candidate diversity and realistic map error jointly consume
the headroom the derivative needs. We report this as the honest boundary of the method,
predicted in advance by an adversarial review of this work and confirmed by measurement.

**Practitioner summary.** Sense where your candidate plans disagree — but compute the
disagreement with a derivative only when your looks are precise and your budget scarce
relative to the candidates' corridor; otherwise mask your uncertainty map by the corridor
and spend the savings on rollouts.

### 4. Related work (compressed)

Forward propagation of terrain uncertainty through differentiable engines: ProTerrain
(pure forward Monte-Carlo; our C2 explains why sampling, not linearisation, is the sound
choice there); MonoForce/FusionForce use the terrain gradient as a training signal —
FusionForce names dCost/dTerrain as available and never deploys it; we turn it into a
test-time sensing signal. Risk-aware planning: STEP, EVORA, Cai et al. (whose CVaR-Dyn is
the one-sided traction-space relative of our bracket). Information-theoretic sensing: NBV,
IPP, Fisher Information Fields and MI-gradient mapping differentiate information w.r.t.
*pose*; SPOT and proximity-masked NBV inject task-awareness geometrically — our corridor
baselines instantiate exactly that idea, and our claim is measured *against* it, not around
it. Decision-theoretic ancestry: VOI (our score is a first-order surrogate for VOI over
plan selection), POMDP-IR, belief-space planning, CTP-with-remote-sensing; goal-oriented
OED (Alexanderian et al.) already places sensors by adjoint-weighted effect on a downstream
quantity in smooth linear inverse problems — we bring that principle to non-smooth contact
physics and an embodied candidate-decision set; DWR error estimation is the numerical
ancestry; query-by-committee the acquisition ancestry.

### 5. Limitations

All synthetic (one robot geometry; sigma structured but not calibrated — real-sigma
calibration by repeated traverses is the next step and the paper's largest gap); tau over a
fixed candidate set is a proxy for closed-loop outcome; five noise arms are one seed set
under five corruption models (pseudo-replication, disclosed); borderline p-values are
reported with the family-wise caveat (cap-pooling and soft-gradient effects would not
survive correction); the live-elite measurement used a fixed goal and one
replan cycle rather than a closed loop; and the method requires a differentiable contact
model at deployment — though
only in a low-rate advisory side-channel (K taped rollouts per look decision), with the
control loop unchanged.

### Reproducibility

Every number: `studies/` on branch `study/adjoint-sensitivity`; consolidated claims with
provenance in `studies/CLAIMS.md`; independent 12/12 reproduction from committed code
2026-08-06. Pre-registered holdouts: design seeds 0–119, reporting seeds 200–319, no
post-hoc iteration.
