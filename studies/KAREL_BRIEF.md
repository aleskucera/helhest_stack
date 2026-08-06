# Supervisor brief — differentiable terrain adjoints: what works, what doesn't, and the paper(s)

*Aleš, 2026-08-06. One page. Everything below is measured, pre-registered where it matters,
and reproduced from committed code (branch `study/adjoint-sensitivity`). Full provenance:
`studies/CLAIMS.md`; drafted workshop paper: `studies/WORKSHOP_DRAFT.md`; central figure:
`studies/out/bench/scope_law.png` (regenerate: `python -m studies.bench.plot_scope`).*

## The one-sentence result

**Gradients through the contact settle fail wherever they are trusted pointwise — eight
pre-registered refutations — and work where they are consumed by an averaging optimizer:
sensing support, and (new, tonight) decision-focused training of terrain perception.**

## The three findings

**1. Uncertainty propagation through contact cannot be linearized (the characterisation).**
The plan-cost adjoint dJ/dh is exact but valid over millimetres (contact-switch slack) while
map sigma is centimetres. Eight repair/usage attempts refuted with mechanisms: 1st/2nd-order
FOSM, soft gradients, sub-cell refinement, analytic Hessian (piecewise-zero), bundled
adjoints (heavy-tail blow-up at sampled switches, draws at 5e10), minimal-perturbation
certificates (as predictor and as rare-event FORM/IS — derivative-free subset simulation
wins by ~2 dex). Sharpest mechanism: decisions fail through *nonlinear variance asymmetry*
(a rival's envelope-max cost has ~3x the variance and overtakes via ordinary excursions) —
the gradient points at the wrong boundary. Relevant to ProTerrain: this *justifies* its
forward-sampling design choice quantitatively. What works instead: sampled-map CVaR, and a
two-rollout bracket on belief±sigma that beats STEP's per-step Gaussian CVaR at ranking
(p=1.6e-3) using bounds elevation_mapping already publishes.

**2. Decision-focused sensing works — within a measured boundary (the workshop paper).**
Sensing directed by candidate-set adjoint *support* beats entropy-directed sensing at every
budget and noise level (up to +0.69 tau, p=2.5e-60, n=200; entropy is indistinguishable from
not sensing). Against the strongest gradient-free baseline — uncertainty masked to the
candidates' corridor — the edge follows a *headroom law*: ~30-70% of the gap between
corridor-masking and an oracle, vanishing at tiny budgets (floor), huge budgets or wide-FOV
body-fixed looks (ceiling; our own pre-registered holdout retracted the wide-cone claim),
and on the live MPPI's low-diversity elites under full noise. Honest scope, honestly mapped.

**3. Decision-focused perception training (NEW — the direction I want to discuss).**
Train the map inpainter not by MSE against true heights but by *plan-choice regret*,
backpropagated through the settle adjoint (gradient chain verified to 5e-13; training
stable). On 80 virgin seeds: regret 0.27 vs 0.60 for MSE-trained (p=0.006) — with 3x WORSE
height RMSE. **It learns a planning map, not a mean map** (consistent with the measured
Jensen optimism of mean maps under max-based contact costs). The standard alternative —
MSE map + risk-aware planning (STEP penalty, or CVaR over 128 correlated sampled maps) —
does NOT recover the gap (p<=0.0012): the per-cell (mean, sigma) factorization provably
loses correlation/direction/relevance information. Relation to MonoForce: same
backprop-through-physics infrastructure and no-hand-labels philosophy; MonoForce's loss
lives in motion space (explain the driven trajectory), this one in decision space (pick the
right plan), with hindsight self-supervision (later observations label earlier inpaintings).
Caveats: linear 6-feature model on synthetic worlds — capacity ladder, sigma-head variant,
plan-family transfer, and real-bag hindsight training are the Phase-B list.

## The asks

1. **Workshop paper** (`WORKSHOP_DRAFT.md`, complete): co-authorship posture, venue pick
   (off-road / UQ workshop at ICRA or RSS), and your read on the framing before I port to
   LaTeX. Then T-RO for the full version after real-sigma validation.
2. **ProTerrain coordination**: we verified (full-methods read) no overlap — it is forward-MC
   only, no test-time terrain gradients, no sensing. Finding 1 strengthens its motivation;
   happy to cite/coordinate however the group prefers.
3. **The DFL direction (finding 3)**: is this a paper you want to supervise toward CoRL?
   It runs on the group's engine lineage and my Phase-B plan is ~a month of work.
4. One field afternoon with the robot (drive the same occluded ground twice) unlocks the
   real-sigma calibration for (2) and the hindsight labels for (3).
