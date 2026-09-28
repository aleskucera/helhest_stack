# Decision-focused sensing via adjoint map-cell sensitivity — plan

Working plan, written 2026-08-05, revised against the novelty assessment in
`Adjoint Map-Cell Sensitivity for Off-Road MPPI Navigation_ Prior-Art and Novelty
Assessment.pdf`. Written to be picked up cold.

---

## 1. The idea, in one paragraph

The engine is differentiable w.r.t. the **terrain**, not the controls: an
implicit-function-theorem adjoint through the 3×3 quasi-static settle gives
`∂J/∂h_i` — how much a *committed plan's* cost depends on each individual
elevation cell. Combined with per-cell map uncertainty `σ_i`, that localises
*which map cells this decision actually rests on*. Used to (a) steer sensing at
decision-relevant cells rather than high-entropy ones, (b) abstain and re-observe
when the elite set's cost variance is dominated by unobserved cells, and (c)
allocate planner effort where model error is decision-relevant.

## 2. What the novelty assessment changed

**The framing is narrower than we first sketched. Adopt the narrow one.**

- **Drop the calibrated-σ paper.** C7 is nearly taken — UNRealNet
  (arXiv:2407.08720) and a Neural-Processes elevation model (arXiv:2508.03890)
  already produce *and held-out validate* uncertainty on inpainted/unobserved
  cells. **Consume their uncertainty; do not claim it.** (This reverses an
  earlier recommendation in this repo's discussion to split off a perception
  paper first — that paper is largely written already, by other people.)
- **C2 and C3 are components, not contributions.** First-order variance and
  in-MPPI screening are cited machinery (DM-MPPI arXiv:2512.00759, CoVO-MPC
  arXiv:2401.07369). Claiming them invites the "standard UQ in a new wrapper"
  rejection.
- **The contribution is C1 → C4 → C5 → C6 → C8**: the *backward,
  decision-focused use*. C4 (goal-oriented active perception) is the strongest
  claim; C8 (diagnostics-not-learners) is the conceptual spine and has no clean
  precedent.

**Adopt this terminology** — it already exists and signals the right lineage:
*decision-focused learning / predict-then-optimize / SPO+* (Elmachtoub–Grigas),
*goal-oriented (dual-weighted-residual) error estimation*, and for C5, *value of
information*. Name the method **goal-oriented active perception** or
**decision-focused sensing**.

## 3. ⚠️ ProTerrain — read before writing a line

**arXiv:2510.19364** — Raja, **Agishev**, Prágr, Pajarinen, **Zimmermann**,
Singh, Ghabcheloo. Oct 2025, submitted to ICRA 2026. Models spatially correlated
aleatoric terrain uncertainty and **propagates it through a differentiable
physics engine** for probabilistic trajectory forecasting.

That is our C1+C2 architecture **minus the adjoint**, from two of our own
authors. The structural difference is real but narrow:

| | ProTerrain | this work |
|---|---|---|
| direction | **forward** Monte-Carlo sampling of terrain uncertainty | **backward** adjoint attribution |
| output | a trajectory *distribution* | *per-cell* contribution to Var(J) |
| use | forecasting / traversability | sensing, abstention, effort allocation |

**Action before anything else: talk to Agishev and Zimmermann.** Confirm whether
ProTerrain computes any backward/adjoint quantity or attributes variance to
individual cells — the assessment only read its abstract/intro/related-work, not
its methods. If it does even partially, C1/C2 shrink further. This is also
simply the right thing to do with colleagues working the same seam.

## 4. The technical risk, and a mitigation we already own

**The linearisation fails exactly where it matters.** Largest `σ` (curbs,
occlusion boundaries) = largest curvature = **contact-set switches in the
settle** (which of the three wheels is in support changes as `h` varies). FOSM,
the one-shot adjoint, and the IFT-through-settle all assume a local smoothness
that fails there. The published contact-gradient literature predicts exactly this
(arXiv:2202.00817, arXiv:2506.14186, arXiv:2604.18161). A reviewer will find this
immediately; it is the objection most likely to sink the paper.

**We already compute the switch detector for free.** `normal_loads()` returns
per-wheel `N_i` every step, and `min N_i → 0` *is* the contact-set switch — it is
the same active-set boundary that dominated the Ostrich relaxation study (see
`ostrich/RELAXATION_PLAN.md`: every structural relaxation failed at, and had to
re-supply, the active-set criterion). So:

> **Use `min N_i / (m·g)` as a linearisation-validity flag.** Where the margin is
> small, the settle is near a switch, `Var(J)` is untrustworthy, and the planner
> falls back to local sampling in those cells only.

That turns the fatal objection into a stated, detected, and handled limitation —
with a detector that costs nothing because it is already in `loads_out[t, b]`.

Other mitigations to consider: curvature-corrected (second-order) FOSM;
randomized-smoothing "bundled" gradients (Suh–Pang–Tedrake, RA-L 2022).

## 5. Do these two studies before committing months

Both are offline, cheap, and each can kill or redirect the project.

### Study A — is the adjoint correct?
Finite-difference `∂J/∂h` against the numpy oracle **on non-uniform terrain**.
`ostrich/RELAXATION_BRANCH.md` records that a Warp `sample_field`
position-gradient bug once threw friction gradients off ~47% *and was invisible
under uniform fields*. Uniform terrain proves nothing.
→ **Gate:** if this fails and cannot be fixed, stop; it also invalidates the
existing calibration path.

### Study B — is first-order adequate? *(the decisive one)*
Compute `∂J/∂h_i` and FOSM `Var(J)`, then compare against brute-force
perturbation / Monte-Carlo ground truth, **stratified by cell type**: flat,
slope, curb edge, occlusion boundary — and cross-tabulated against the `min N_i`
margin.
→ **Gate:** if first-order stays within a small consistent factor at
high-σ/high-curvature cells, proceed. If it diverges there (the likely outcome),
either add the curvature correction or the sampling fallback — **and report it as
a finding, not a footnote.** The stratified plot is the paper's central figure
either way.

### Study C — test C8 directly (the most novel, least supported claim)
Show that one-shot local attribution still ranks the right cells in regimes where
BPTT-style repeated descent through the same settle diverges. If it holds, that
is a citable contribution against arXiv:2202.00817 / arXiv:2604.18161. If it
fails, C8 becomes a refutation — still publishable, still honest.

## 6. The benchmark that would confirm the framing

Not "we reduce cost by X%". The scenario the assessment names explicitly:

> **A case where the highest-entropy cell and the highest-`∂J/∂h` cell are
> different, and sensing the latter improves the task outcome while sensing the
> former does not.**

Build it in `worlds.py`: occluded far side of a crest, with a real step hidden in
the inpainted region, plus a decoy high-entropy region that is decision-
irrelevant (laterally offset, outside the wheel envelope). Randomise step height
and approach angle over seeds.

Baselines, in increasing strength: plain MPPI → σ-penalty → entropy/NBV-directed
sensing → **CVaR over sampled maps** (the real one; note we can *afford* it at
~10 ms, so never argue cost — argue attribution). Metrics: collision/tip-over
rate, stop-and-look rate, time-to-goal at equal safety.

## 7. Fallback if Study B kills the attribution path

Pivot to **C6**: a goal-oriented (adjoint-weighted) *error-vs-effort* criterion
for compute/fidelity allocation inside a sampling planner — "where is my
planner's model error decision-relevant?". No robotics precedent, and it does
**not** depend on fragile high-curvature attribution. This is the safe harbour;
keep it in view.

## 8. Positioning

- **Frame:** goal-oriented / decision-focused active perception for off-road
  navigation. Contrast explicitly with information-theoretic active perception
  (entropy/MI/NBV) and with forward risk-aware planning (ProTerrain, STEP).
- **Cite as borrowed machinery, not contribution:** DWR lineage
  (Becker–Rannacher), FOSM, adjoint UQ.
- **Realistic venue:** RA-L / ICRA / IROS base case. T-RO or JFR if the
  Study-B characterisation is rigorous and the field validation is broad. RSS
  only if Study B yields a crisp transferable criterion for when first-order
  attribution is valid.
- **C8 is politically loaded** — it implicitly critiques MonoForce's use of the
  same gradients. Get co-author buy-in before writing it, not after.

## 9. Housekeeping

- The earlier prior-art report cited **arXiv:2606.00085 "ADO"**, which **does not
  exist** — discard it and anything derived from it. Its arXiv:2510.05330 title
  was also wrong (the real paper is *Adaptive Dynamics Planning*).
- `benchmarks/forward.py:21` and `benchmarks/differentiable.py:19` import
  `helhest.control.reference._to_target_wheel_omega`, which no longer exists;
  both fail at import.
- Model-side improvements that are orthogonal to this and mostly free are in
  `docs/research/IMPROVEMENTS.md` — §3 (tip-over margin) is a prerequisite here anyway, since
  it is the same `min N_i` quantity §4 above depends on.
