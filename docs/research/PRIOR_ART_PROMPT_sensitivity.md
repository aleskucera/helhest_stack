# Deep-research prompt — decision-relevant map uncertainty via adjoint sensitivity

Copy everything below the line into a deep-research model.

---

# Prior-art and novelty search request

## What I need

A rigorous novelty assessment of a proposed robotics method. I need to know, per
claim, whether it is already published — and if it is close but not identical,
exactly how it differs and how I would have to differentiate my work.

**Critical instruction on what counts as novelty.** The underlying mathematics
(adjoint sensitivity analysis, first-order uncertainty propagation, dual-weighted
residual / goal-oriented error estimation) is **mature and standard** in
computational fluid dynamics, nuclear engineering, groundwater modelling and
finite-element analysis. I already know this. Do **not** report "this is standard
adjoint UQ" as if it settled the question. The question is whether this machinery
has been **applied in this specific robotics setting**. Please separate every
verdict into:

- **method novelty** — is the mathematical technique new? (I expect: no)
- **application novelty** — has it been used for this purpose, in this domain,
  with this architecture? (this is what I need to know)

## System context

A GPU navigation stack for a three-wheeled skid-steer ground robot on rough
outdoor terrain, comprising:

1. **Perception** — lidar point cloud → 2.5D elevation grid (heightmap) →
   traversability cost map. Cells with no returns (occlusion, range limits) are
   filled by multigrid diffusion inpainting.
2. **A differentiable, purely kinematic robot–terrain model.** Controlled DOF
   `(x, y, yaw)` from differential-drive wheel kinematics with friction-dependent
   turning parameters. Derived DOF `(z, pitch, roll)` from a **quasi-static
   settle**: a 3×3 Newton solve placing all three wheels on the terrain.
   The model is **differentiable with respect to the raw heightmap `h` and a
   per-cell friction field `μ`**, via a hand-written implicit-function-theorem
   adjoint through the settle. Note: it is differentiated w.r.t. the
   *environment*, not w.r.t. the controls.
3. **A sampling-based planner** — MPPI over ~4000–8000 rollouts, horizon ~25
   steps.

## The claims to assess

Give each a separate verdict.

**C1 — Adjoint sensitivity of a planning cost to individual map cells.**
Computing `∂J/∂h_i`, the derivative of a trajectory's planning cost with respect
to the elevation value of each individual map cell, through a differentiable
robot–terrain contact/support model.
*Please distinguish carefully from the far more common `∇h` — the **spatial**
gradient of the height field, i.e. terrain slope used as a cost feature. That is
not what I mean. I mean the adjoint sensitivity of the task cost to the map's
stored values.*

**C2 — First-order propagation of map uncertainty into decision-relevant cost
variance.** Combining `∂J/∂h_i` with a per-cell elevation uncertainty `σ_i` to
obtain `Var(J) ≈ Σ_i (∂J/∂h_i)² σ_i²`, i.e. how much a *specific plan's* cost is
uncertain because of *specific map cells*.

**C3 — Sensitivity-driven screening inside sampling-based MPC.** Using that
quantity to rank, screen, or promote candidate trajectories within an
MPPI/CEM/model-predictive-path-integral loop — deciding which candidates warrant
more computation or more information.

**C4 — Sensitivity-driven active perception.** Using `∂J/∂h_i` to identify
*which map cells the current decision depends on*, and directing sensing,
processing, or viewpoint selection at those cells.
*Contrast explicitly with information-theoretic active perception (entropy
reduction, mutual information, information gain), which is well established. My
criterion is task/decision sensitivity, not information content: a cell can be
maximally uncertain and completely irrelevant to the decision.*

**C5 — A defer-and-look controller behaviour.** Declining to commit to a plan
when the cost variance of the elite candidate set is dominated by unobserved or
inpainted cells — i.e. "I do not know enough to choose yet" — and re-observing
instead.

**C6 — Goal-oriented / dual-weighted-residual error estimation applied to robot
motion planning.** Using adjoint-weighted error estimation to decide model
fidelity, discretisation, or computational effort *in a robotics planning
context* (as opposed to mesh adaptivity in PDE solvers).

**C7 — Calibrated per-cell uncertainty for lidar elevation maps, including
inpainted cells.** A validated per-cell `σ` for a heightmap that covers
**both** observed cells (from within-cell return statistics) **and unobserved /
interpolated cells** (from a model such as distance-to-nearest-measurement,
inpainting residual, or local terrain gradient), verified against held-out
returns rather than asserted.
*Most elevation-mapping work I know of produces either a binary validity mask or
a variance only for observed cells. I want to know if anyone has produced and
**calibrated** uncertainty for the interpolated regions.*

**C8 — Differentiable contact/terrain simulators as diagnostics rather than
learners.** The claim that gradients through differentiable terrain-interaction
models are more reliable for *one-shot sensitivity attribution* than for
*gradient-based learning or trajectory optimisation*, because attribution
requires local correctness once whereas learning requires an unbiased descent
direction repeatedly. Has anyone made, tested, or refuted this argument?

## Prior art I have already found

Please **extend** this rather than rediscover it, and specifically look for
**follow-ups and citing works** for each.

*(Verified by me via search)*
- CoVO-MPC: Theoretical Analysis of Sampling-based MPC and Optimal Covariance
  Design — arXiv:2401.07369, L4DC 2024. Hessian → optimal MPPI sampling
  covariance.
- Do Differentiable Simulators Give Better Policy Gradients? — arXiv:2202.00817,
  ICML 2022.
- MonoForce: Self-supervised Learning of Physics-informed Model for Predicting
  Robot–terrain Interaction — arXiv:2309.09007, IROS 2024 (Agishev, Zimmermann,
  Kubelka, Pecka, Svoboda). Differentiable terrain physics; backpropagates
  trajectory error into perception **as a training loss**. **This is the closest
  work and the highest-priority thread — I need all follow-ups, 2024–2026.**
- Neural Elevation Models for Terrain Mapping and Path Planning —
  arXiv:2405.15227.
- CAHSOR: Competence-Aware High-Speed Off-Road Ground Navigation in SE(3) —
  arXiv:2402.07065. *Learned* competence awareness.
- Multigoal-oriented DWR error estimation with deep neural networks —
  arXiv:2112.11360; Neural-network-guided adjoint computations in DWR —
  arXiv:2102.12450.
- Discrete adjoint method for sensitivity and uncertainty analysis —
  arXiv:1805.01451.
- MetaTune: Adjoint-based Meta-tuning via Robotic Differentiable Dynamics —
  arXiv:2603.27313.

*(From an earlier prior-art report; identifiers unverified — please confirm they
exist and are correctly attributed)*
- Adaptive Dynamics Orchestration (ADO) — arXiv:2606.00085.
- Decremental Dynamics Planning / Adaptive Dynamics Planning —
  arXiv:2510.05330.
- Datamodels-based MPPI rollout pruning — arXiv:2512.00759.
- Surrogate-accelerated MPPI with Deep Koopman operator — arXiv:2603.05385.

## Fields and terminology to search under

The idea sits between literatures that use different vocabulary for the same
mathematics. Please search each explicitly:

- **Robotics/planning:** belief-space planning, POMDP motion planning, active
  perception, next-best-view, informative path planning, risk-aware MPC, CVaR
  planning, scenario MPC, traversability estimation under uncertainty,
  uncertainty-aware off-road navigation, perception-aware planning
- **Applied mathematics:** dual-weighted residual, goal-oriented error
  estimation, a posteriori error estimation, adjoint sensitivity, model
  adaptivity, certified model reduction
- **Uncertainty quantification:** first-order second-moment (FOSM), sensitivity
  analysis, Sobol indices, value of information, adjoint-based UQ
- **Machine learning:** decision-focused learning, predict-then-optimize,
  influence functions, data attribution, task-aware perception
- **Control:** parametric sensitivity in NMPC, real-time iteration scheme,
  sensitivity-based warm starting

## Groups and venues worth checking directly

- **CTU Prague, VRAS / Vision for Robotics and Autonomous Systems** (Zimmermann,
  Svoboda, Agishev, Pecka, Kubelka) — highest priority; I am in this group and
  need to know what neighbouring projects have published or preprinted.
- ETH Zurich RSL — elevation mapping and uncertainty (Fankhauser lineage)
- CMU Field Robotics; JPL (NeBula, traversability); MIT; Georgia Tech
  (Theodorou — MPPI origin); UT Austin (Xuesu Xiao — off-road navigation)
- Venues: ICRA, IROS, RSS, CoRL, RA-L, T-RO, L4DC, ICML, NeurIPS. Range
  2015–present, weighted heavily toward 2023–2026.

## Please also report

1. **Negative results and refutations.** Has anyone tried adjoint sensitivity
   through contact models for planning and reported that it *fails* — noisy
   gradients, non-smoothness at contact transitions, poor calibration? A
   published failure is as important to me as a published success.
2. **Anything contradicting the premise** that first-order sensitivity is
   adequate under realistic terrain-map uncertainty. My own concern is that the
   places with the largest `σ` (curb edges, occlusion boundaries) also have the
   largest curvature, so the linearisation may fail exactly where it matters.
3. **Terminology I am missing** — if this idea already has a name in some field,
   I need that name.

## Output format

For each claim C1–C8:

| Claim | Verdict (TAKEN / PARTIALLY TAKEN / OPEN) | Closest citation | Precise mechanism difference | How I would have to differentiate |

Then:

- **Top five threats, ranked** — the works that most endanger novelty, each with
  the specific structural difference stated precisely.
- **Method vs application verdict** — a clear statement of which parts are old
  mathematics and which combination, if any, is unpublished.
- **Reviewer simulation** — write the three most damaging objections a critical
  ICRA/RSS reviewer would raise, and whether each is answerable.
- **Recommended framing** — given everything found, the strongest honest
  positioning for this work, including the possibility that the answer is "this
  is not novel enough; here is the adjacent question that is."

Please be adversarial rather than encouraging. I would rather learn now that this
is taken than after six months of implementation.
