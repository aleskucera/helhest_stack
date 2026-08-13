# Prior art — the routing follow-up ("tier 3": diverse routes + closed-form path CVaR)

Scanned 2026-08-12 (web, abstracts fetched not guessed). Candidate contribution: lattice
router generates diverse routes (corridor penalties); each complete route gets closed-form
E and Var of the contact-max cost through the measured correlated belief (kernel
convolutions, no sampling, no covariance matrix); CVaR selection strictly at path level
because path risk is not edge-additive. Companions: Jensen-corrected (Clark-E) edge
weights inside Dijkstra; chance-constrained per-pose rollover gates from closed-form
attitude moments.

## Verdict

Broad categories CROWDED: risk-aware off-road planning (STEP RSS'21/IJRR, EVORA T-RO'24,
RAMP, TRG-planner arXiv:2501.01806, URA*) and mean-variance shortest path (Nikolova ESA'06;
Nikolova & Stier-Moses OR 2014). The specific six-piece joint claim is THIN — nothing
found does more than 2-3 of: (i) global lattice router (ii) diverse full routes
(iii) because risk is non-edge-additive (iv) path-level CVaR (v) closed-form mean AND
variance of a contact-max cost (vi) under spatially correlated uncertainty incl. a
long-range pose-drift plane.

## The two fights that decide the paper

1. **STEP** (Fan et al., arXiv:2103.02828 + 2303.01614): global A* with per-cell
   closed-form Gaussian CVaR summed ADDITIVELY inside the search, cells independent, no
   path-level variance. Reviewer: "STEP with better math?" -> REQUIRED: a measured case
   where additive per-cell CVaR picks a materially riskier route than path-level CVaR.
   (Adjacent evidence already held: STEP-form degrades to worse-than-mean-map at PLAN
   ranking under the measured belief — realistic_sigma_hybrid_all.json. Route-level
   version of that demonstration is the paper's centerpiece experiment.)
2. **Nikolova** (ESA 2006): exact mean-variance shortest path via quasi-convex
   maximization — INDEPENDENT edges, and the paper itself flags correlation as open.
   REQUIRED: state explicitly why the exact machinery is inapplicable here (nonlinear
   contact-max cost + correlated field are exactly what it excludes); do not leave "why
   not Nikolova" to the reviewer.

## Nearest neighbors to cite and distinguish

- STEP / EVORA / RAMP / TRG-planner / URA* — global-or-local risk-aware terrain planning;
  all per-cell/per-step, additive, uncorrelated.
- Nikolova line — non-additivity is proven ground; independent-edge, exact-algorithm
  strategy, non-robotic.
- SCOS (arXiv:2509.19559, 2025) — spatially correlated obstacle fields, but Monte-Carlo/
  RL, expected cost only. Closest correlated-field routing thread.
- Neural-Process elevation mapping (arXiv:2508.03890, 2025) — correlated elevation
  uncertainty, perception-only, never routes. Cite as the mapping-side adjacent.
- p-ACE (Ghosh/Otsu/Ono) — per-pose probabilistic attitude/clearance bounds; the gates
  companion must be framed as "per-pose bound -> router-integrated chance gate" or it
  reads as p-ACE-for-wheeled-robots.
- DeHart et al. — chance-constrained rollover machinery (manipulator payload); same math,
  different input uncertainty.

## Surviving claim wording (scan's draft)

"To our knowledge, the first global route planner for off-road ground robots that selects
among diverse lattice-generated candidate routes using a closed-form path-level mean and
variance of a nonlinear (wheel-contact-max) terrain cost — computed via moment
propagation through the map's spatially correlated uncertainty (including a long-range
pose-drift component), without sampling or an explicit covariance matrix — with CVaR
selection applied strictly at the path level because path risk is not edge-additive,
distinct from prior work (STEP, TRG-planner) that sums closed-form per-cell risk
additively inside the graph search."

Unverified lead, chase if pursued: "Gaussian Traveler Problem" (GP-correlated edge costs,
possibly theoretical OR only — not fetched).
