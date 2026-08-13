# PRE-REGISTRATION — full-cost decision benchmark, cylinder, three regimes (committed BEFORE the run)

Date: 2026-08-13. Frozen by this commit: command, seed block, criteria, reporting rule.

## Why

The de-sphering recompute (HANDOFF §12) showed that under the merged engine + cylinder
element the settle-only hybrid/all benchmark is a hard instance where NO estimator
separates (all paired tests n.s.), while the full cost (settle + clear_soft) separates
decisively on hybrid/all (p≈5e-7, seeds 0–99). If the paper re-bases on the cylinder,
the honest headline shifts to the full cost — but that claim currently rests on ONE
regime and on the same seeds 0–99 used throughout development. This experiment is the
pre-registered, virgin-seed, three-regime test of the reframed headline: "on the
deployed objective, under the deployed geometry, the analytic estimator is decisive."

## Frozen protocol

- Code state: study branch at the commit introducing this file (clark_hinge.py --family/
  --noise plumbing included; engine = merged bbc98cf; element machinery = stage A/B).
- Commands (one per regime), seeds 6000–6099 (virgin: no script or artifact in this repo
  has consumed a seed ≥ 6000; the 5000+ block was used only for 5000–5099):
  `... -m studies.bench.clark_hinge --device cuda:0 --stage2 --skip-gate --seeds 100
   --seed-offset 6000 --element cylinder --family {hybrid,hybrid,fan} --noise {all,clean,sensor}
   --tag _fc_{all,clean,sensor}`
- Cost: full (settle + clear_soft, coupled moments); trajectories and MC truth through
  the cylinder ForwardSimulator (matched element end to end); gradient arms sphere-locked
  as recorded in the json.

## Success criteria (frozen, evaluated once per regime)

Per regime, the benchmark PASSES iff:
- (i) clark_cvar has the lowest mean regret among the estimator arms;
- (ii) clark_cvar beats step at p < 0.05 (the existing paired sign test);
- (iii) clark_cvar beats bracket at p < 0.05.
The HEADLINE claim ("decisive on the deployed objective across regimes") requires all
three regimes to pass. Partial outcomes are reported as partial — no averaging across
regimes, no metric substitution, no rerun of this block.

## Predictions (recorded now)

hybrid/all passes (the seeds-0–99 margins were ~1.1, an order of magnitude above the
settle-only margins that noise drowned). hybrid/clean passes. fan/sensor is the risk:
the fan family's full-cost margins are unmeasured under cylinder; if any regime fails,
most likely this one, via criterion (iii) (bracket is the strongest baseline).

## Reporting rule

All three results are reported wherever any is, alongside the full stage-2 history
(sphere-era: failed 0–99 buggy noise → passed 0–99 fixed → virgin 5000–5099 mixed;
cylinder-era: passed 0–99) — the trajectory is the record. If the composite fails, the
paper's full-cost claim stays regime-qualified and the settle-only hard-instance framing
stands on its own.
