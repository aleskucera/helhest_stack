# Session prompt: write THE paper (RA-L / ICRA)

Paste or reference this at the start of the paper-writing session(s). Decisions already
made with the supervisor (do not relitigate): **authors = Aleš Kučera and Karel Zimmermann**
(affiliation: CTU in Prague — confirm exact department string with the user before
camera-ready); **ONE merged paper** from this research line (not the earlier two-paper
plan); **target RA-L, with ICRA as the alternative** (both IEEEtran; RA-L = 6 pages + 2
paid; if ICRA's deadline (~mid-September) is the target, say so early because it compresses
everything).

## Read first, in this order
1. `studies/HANDOFF.md` — the complete experimental record. Non-negotiable context.
2. `studies/CLAIMS.md` — the claims with provenance (rev 2, post red-team).
3. `studies/WORKSHOP_DRAFT.md` — the existing complete draft (written pre-Clark; it is the
   TEXT BASE to cannibalize, not the paper).
4. `studies/DFL_CLAIMS.md` — READ THE BANNERS ONLY (the thread is closed as a method; only
   its standing insights may be cited in the paper's discussion).
5. Process rules: `studies/HANDOFF.md` §7. They bind every session. In particular:
   pre-register criteria before any new run; report from virgin seeds; VERIFY EVERY
   CITATION BY FETCHING ITS ABSTRACT (this project caught one fully hallucinated citation
   and one mis-cite trap — arXiv:2604.01434 is NOT a VOI-sensing paper); when unsure what a
   number means, recompute from the JSON in `studies/out/bench/`, not from summaries.

## The paper's shape (draft this FIRST, get user sign-off before writing prose)

Working thesis (the strongest sentence the record supports): **uncertainty in terrain maps
must be propagated through the contact max, not linearized at it** — with three supports:
(a) the characterisation: derivative/Taylor methods fail at map-noise scale, with the
validity-radius criterion and the refutation ledger (compressed to ~1 page + one table);
(b) the constructive result: Clark/SSTA moment propagation — calibrated (sd-ratio 0.927),
Jensen-aware (corr 0.884), beats STEP/bracket, only never-bad estimator across regimes,
~= 128-256 MC draws, deterministic; apparently first use of Clark 1961 in robotics;
(c) the practical ledger: bracket beats STEP; aggregate over time not cells; the Jensen
optimism (+1.4) quantified; sensing corollary (uncertainty != relevance + headroom law) as
a SHORT secondary section or discussion paragraph — at 6-8 pages it cannot carry equal
weight, and cutting it entirely is an acceptable proposal to put to the user.
Produce 2 candidate outlines (Clark-led vs characterisation-led) with page budgets and ask
the user to pick. Do not start prose before that decision.

## Work order

### 1. Close the two gates on Clark's claims (before the paper says them)
- **Ono/JPL prior-art check** (blocks "first in robotics"): search IEEE Xplore / Google
  Scholar for Masahiro Ono's chance-constrained rover planning line + planetary terrain
  risk assessment; verify none does closed-form moment propagation through a terrain
  max/min. Record verified citations either way. (Context: OpenAlex scan of all 728 Clark
  1961 citers found zero robotics; this is the one uncovered corner.)
- **The clear_soft hinge fix** (blocks "full cost"): studies/bench/clark_full.py's hinge
  extension overshot E by ~2.4x — suspected missing correlation between the ~18 belly
  points and the settle pose (clearance = body height MINUS terrain; the two are
  correlated through the shared cells; treating them independently inflates the hinge
  mean). Fix the joint model, gate it (absolute-error + correlation metrics vs 20k-draw
  MC — NOT raw rel-err, see HANDOFF §7 near-zero-baseline trap), then rerun the full-cost
  head-to-head [PRE-REG: clark_cvar beats step AND bracket at p<0.05 on the full cost].
  If the fix defeats you, the paper scopes Clark to the tilt cost honestly — acceptable.

### 2. The paper itself
- LaTeX in `paper/main/` (IEEEtran; pdflatex + latexmk installed; `paper/workshop/figs/`
  has seeded figures; all figures regenerate from scripts — *.png is gitignored).
- Figures: scope_law.png only if the sensing section survives the cut; the Clark section
  needs (i) a Jensen/staircase explainer panel (a teaching version exists conceptually —
  see the four-panel figure logic in the session record: max-inflation, correlation-tames-
  it, point-vs-neighborhood, covariance-decides-pose-error; rebuild ~2 panels of it as
  Fig. 1), (ii) the estimator comparison table, (iii) the budget curve (Clark vs
  MC-with-N-draws). Keep every number traceable to a JSON.
- Related work: anchor lists live in CLAIMS.md (verified) — STEP/EVORA/Cai (risk),
  Fankhauser (the Gaussian field input), Suh/Parmas/DiffMJX (contact-gradient pathology,
  scoped to optimization — our delta: measured for UQ), Lozano-Pérez C-space + morphology
  (the max IS standard dilation — the reviewer-proofing for "is max the right model"),
  SSTA literature (Clark's industrial lineage), GOOED (adjoint sensor placement in smooth
  PDEs) and QBC if the sensing section survives. Every ID re-verified by fetching at
  writing time.
- Honest-limitations section is mandatory and its content is pre-written in CLAIMS.md /
  HANDOFF.md: synthetic sigma (structured, non-saturating, but uncalibrated), one robot,
  settle-only vs full cost (per gate 1 outcome), Python wall-time vs GPU sampling (kernel
  future work unless done), rigid-terrain assumption, ranking-not-closed-loop.

### 3. Strengtheners, in priority order (do as time allows, none blocks a draft)
- Warp kernelization of the Clark fold (turns the wall-time caveat into a claim).
- The owed reruns from HANDOFF §8.6 that affect kept claims only (catastrophe-cost CVaR
  rematch matters IF the paper claims bracket>CVaR framing; clean seed block 5000+ for the
  headline comparisons).
- The engine Tier-1 branch (IMPROVEMENTS_PROMPT.md, separate session) + study-chain rerun
  = the "conclusions hold under an improved contact model" robustness sentence.
- Real-sigma field data (needs the user's field afternoon — coordinate with them; the
  paper is submittable without it for RA-L only if framed as simulation study; ASK the
  user how they and Karel want to play this).

## Rules of engagement
- The user + Karel decide: outline choice, what gets cut, venue/deadline, anything
  touching authorship or field-data framing. Sessions decide: everything mechanical.
- Branch: paper work on a branch off `study/adjoint-sensitivity` (it needs the study
  scripts); new experiments follow the same additive rules as always (new files under
  studies/bench/, nothing existing edited, outputs committed with the code).
- Commits in repo style, no Claude co-author line (user rule).
- Agent economy: Sonnet for coding/search/review, Haiku for mechanical runs; red-team the
  finished draft with 2-3 adversarial reviewer agents BEFORE showing it to Karel — that
  protocol caught every overclaim this project ever made.
