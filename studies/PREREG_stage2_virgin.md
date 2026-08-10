# PRE-REGISTRATION — stage-2 full cost on virgin seeds (committed BEFORE the run)

Date: 2026-08-10. Committed before the experiment executes; the run command, seed block,
and success criteria below are frozen by this commit.

## Why this experiment exists

The stage-2 full-cost head-to-head (clark_cvar vs step/bracket on settle + clear_soft,
hybrid/all) has a compromised history:

1. n=100, seeds 0–99, 2026-08-07: FAILED its pre-registration (p=0.43 vs step, p=0.34 vs
   bracket) — on record in `clark_hinge_stage2.json` (commit 92e1c9e).
2. n=100, seeds 0–99, 2026-08-10: PASSED decisively (p=3.1e-6 vs step, p=4.7e-4 vs
   bracket) after the 8e0aef5 noise-model fixes — but this is a POST-HOC correction of a
   failed test: same seeds, rerun after a code change, and reported as such (commit 0021b98).

A reviewer should not accept 2 as confirmation. This experiment is the independent test:
same protocol, virgin seed block never used by any design choice, criteria fixed in
advance. Whatever it says is reported alongside BOTH prior results, never in place of them.

## Frozen protocol

- Command: `.venv/bin/python -m studies.bench.clark_hinge --device cuda:0 --stage2
  --skip-gate --seeds 100 --seed-offset 5000 --tag _stage2_virgin`
- Code state: `clark_hinge.py` at the commit that introduces `--seed-offset` (this commit);
  estimator code otherwise as of 8e0aef5's fixes. `--skip-gate` because Gate H's verdict is
  a separate question (task: virgin-gate rerun) and stage 2's validity here is defined by
  the criteria below, not by re-running the gate on design cases.
- Seeds: 5000–5099. Virgin check: no script or committed artifact in this repo has used a
  seed ≥ 5000 for this benchmark family (HANDOFF §8.6 reserved the 5000+ block for clean
  revalidation; grep of `studies/` for seed offsets confirms nothing consumed it).
- Cost: full (settle + clear_soft, coupled moments). Family/noise: hybrid/all. Same paired
  sign test as the published comparison (`clark_full._sign_paired`), n=100.

## Success criteria (all fixed now, evaluated once)

The experiment PASSES iff, on the virgin block:

- (i) clark_cvar has the lowest mean regret among the estimator arms
  (clark_cvar, clark_mean, step, bracket, fosm, sum_sigma, none);
- (ii) clark_cvar beats step at p < 0.05 (paired sign test, negative mean diff);
- (iii) clark_cvar beats bracket at p < 0.05 (same test).

Predictions recorded now: (ii) passes with p between 1e-5 and 1e-2 (the seeds-0–99 effect
mean diff −0.27 is large; virgin sampling noise should not erase it, but p=3.1e-6 will not
replicate exactly); (iii) is the most at-risk criterion (mean diff −0.26 but bracket is the
strongest baseline in the measured-belief runs).

## Reporting rule

The paper reports the trajectory: failed (0–99, buggy noise) → passed (0–99, fixed noise,
post-hoc) → this result (virgin). If this FAILS any criterion, the full-cost claim reverts
to "calibrated but not independently confirmed" and the headline stays on the
settle/measured-belief results. No rerun of this block under any modification; a future
attempt needs a new pre-registration and a new virgin block.
