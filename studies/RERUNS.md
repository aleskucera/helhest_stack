# RERUNS — artifacts invalidated by the 2026-08-10 bug-fix session

A high-effort code review of `studies/bench` + `studies/sensing` found 18 confirmed defects;
the 13 worst were fixed in the working tree (NOT committed — review the diffs first). Every
fix was verified by its own targeted check; no full benchmark was rerun. This file logs which
`studies/out/` artifacts each fix invalidates and in what order to regenerate them.

**Pre-registration rule (applies to every rerun below):** a rerun of a pre-registered result
is a NEW test. Report it alongside the original number as a post-hoc correction — never in
place of it. Affected pre-registered results: the Clark gates (`clark.json`,
`clark_full.json`, `clark_grad.json`), the realistic-sigma criteria
(`realistic_sigma_hybrid_all.json`, PREREG_realistic_sigma.md), the certificate study
(`certify.json`), and the rare-event verdict (`rare.json`).

## The fixes (all uncommitted, working tree)

| # | File | Defect fixed |
|---|------|--------------|
| 1 | `bench/noise.py` | `visibility()` cast occlusion rays from grid corner (0,0) instead of the sensor's grid index; `apply_pose_error()` had a spurious −0.5 → half-cell (5 cm) diagonal shift at zero pose error |
| 2 | `bench/realistic_sigma.py` | matched-budget curve scored in-sample (pick and truth shared the same 256 draws → regret ≡ 0 at nd=256, n\* biased low); now 128-pick / 128-truth holdout, nd sweep tops out at 128 |
| 3 | `bench/plot_paper.py` | `fig_budget` read `clark_full.json` (old invented-noise kernel); now reads `realistic_sigma_hybrid_all.json` (measured belief) |
| 4 | `sensing/lidar_belief.py` | pose-drift pitch/roll applied about WORLD axes (yaw never entered); now body-frame via R(−yaw). `_ground` used round()/node convention vs the marcher's cell-center (~2.6 cm on slopes); now bilinear cell-center |
| 5 | `bench/policies.py` + `bench/loop.py` | cvar's 12 noise fields frozen at `default_rng(0)` forever; now per-episode rng `default_rng([seed, frame])` threaded through all policies. Zero-gain fan returned the maximally off-axis bearing; now points ahead. `_trace_route` corner→cell-center (~0.17 m diagonal) |
| 6 | `sensing/subcell_relief.py` | NaN holes → `nan_to_num` → fake cliffs zero-clamped tau_hat in a ring around every occlusion hole; gradient-stencil neighbours of count==0 cells now excluded from `usable` |
| 7 | `bench/rare.py` | stalled FORM attacks silently reported as finite p̂=Φ(−4); `stalled` flag now propagated end-to-end, `analyse()` censors them. `selected[:12]` rebind inside the loop was a no-op; now `del selected[12:]` |
| 8 | `bench/second_order.py` | stale 7-name unpack of build_case's 8-tuple (crashed on both entry paths); reveal handed truth instead of measured |
| 9 | `sensing/pipeline.py` | single mode parsed but never forwarded `--fov/--range/--looks` (silent default-config results); comment claiming 0.05 m "is" the sensor std corrected (actual 0.010–0.046 m, range-dependent) |
| 10 | `bench/ranking.py` | `kendall_tau` computed Goodman-Kruskal gamma, not tau-b (ties dropped from the denominator → inflated taus); now true tau-b, fuzz-verified vs scipy. Same 0.05 m comment corrected at `p_corridor_mi` |
| 11 | `sensing/bag_belief.py` | odom lookup took first-at-or-after, not nearest (forward-shifted poses inflate drift/sigma); one non-finite plan made E_spread infinite (now excluded but still tallied broken) |

## Stale artifacts, by cause (a file can be stale for several reasons)

### Fix 1 — noise.py (hits every `occlusion`/`localisation`/`all` noise arm)
The big one: everything computed with `noise="all"` (the default for most scripts).

- **Clark chain (paper headline numbers):** `clark.json`, `clark_full.json`, `clark_fast.json`,
  `clark_grad.json`, `clark_conv_sphere.json`, `clark_conv_cylinder.json`, `clark_hinge.json`,
  `clark_hinge_design_all.json`, `clark_hinge_virgin.json`, `clark_hinge_stage2.json`,
  `clark_hinge_fast_sphere.json`, `clark_hinge_fast_cylinder.json`
- **Ranking:** `ranking_hybrid_occlusion.json`, `ranking_hybrid_localisation.json`,
  `ranking_hybrid_all.json`, `ranking_hybrid_all_flatsigma.json`
- **DFL suite:** `dfl_stage1.json`, `dfl_full.json`, `dfl_ablation.json`, `dfl_capacity.json`,
  `dfl_elites.json`, `dfl_inert.json`, `dfl_mechanism.json`, `dfl_settle.json`
- **Risk/refutation studies:** `risk_hybrid_all.json`, `elites_all.json`, `certify.json`,
  `bundled_hybrid_all.json`, `order2_hybrid_all.json`, `softgrad_hybrid_all.json`,
  `observability.json`
- **Realistic sigma:** `realistic_sigma.json`, `realistic_sigma_hybrid_all.json`
- **Sensing pipeline:** `out/sensing/results.json`, `sweep.json`, `holdout.json`,
  `case_seed0.npz`, `case_seed1.npz` (+ derived pipeline/summary figures)

NOT affected by fix 1: `rare.json` (sensor arm), `elites_clean.json`, all clean/sensor ranking
arms (`ranking.json`, `ranking_fan.json`, `ranking_hybrid.json`, `ranking_speed.json`,
`ranking_hybrid_sensor.json`, `ranking_hybrid_flatsigma.json`),
`realistic_sigma_fan_sensor.json`, `realistic_sigma_hybrid_clean.json`.

### Fix 4 — lidar_belief.py (the measured-belief foundation)
- **`belief_sweep.json` — QUALITATIVE risk.** The 36-scenario trajectory-shape study
  (commit 81fe982) specifically compared straight/arc/turn to measure the body-fixed-tilt
  frame effect — which the bug made world-fixed, i.e. the study measured the broken variant.
  Its trajectory-shape conclusions must be re-derived, not just re-numbered.
- `lidar_belief.json`, `lidar_belief_diagnose.json` (+ png) — `_ground` fix only.
- **`belief_model.npz`** (and `rho_measured.npy` if regenerated) — the fitted plane+kernel
  belief model; refit needed (`_ground` fix only, effect ~cm-scale; likely directionally
  stable). Everything loading it inherits staleness: `realistic_sigma*.json`.
- `subcell_relief.json`, `subcell_relief_sweep.json`, `subcell_tau_hetero.json`,
  `subcell_tau_v2.json`, `out/subcell_eval.json` — yaw=0 trajectories, `_ground` fix only.
- NOT affected: `bag_belief.json` (real bags, never imports lidar_belief).

### Fix 10 — tau-b (every committed json with a `"tau"` field; values shrink where ties exist)
`bundled_hybrid_all.json`, `certify.json`, `dfl_ablation.json`, `dfl_capacity.json`,
`dfl_elites.json`, `dfl_full.json`, `dfl_pilot.json`, `dfl_mechanism.json`, `dfl_settle.json`,
`elites_all.json`, `elites_clean.json`, `ranking_hybrid.json`, `ranking_hybrid_all.json`,
`out/sensing/results.json`, `sweep.json`, `holdout.json` — plus tau numbers quoted in
`RESULTS.md`, `CLAIMS.md`, `WORKSHOP_DRAFT.md` and the paper draft.

### Fix 5 — policies/loop
`results_gap.json`, `results_gap_nogate.json`, `results_gap_c5.json`, `results_corridor.json`,
`results_corridor_nogate.json`, `results_corridor_c5.json` (produced by `run_bench.py`;
this also resolves their "orphaned producer" status) + the run/verify console logs.

### Fix 2 — budget curve
`realistic_sigma_hybrid_all.json` currently holds a 1-seed smoke result from the fix's
verification (the untracked buggy version was overwritten). Needs a full
`python -m studies.bench.realistic_sigma --seeds 100` BEFORE `fig_budget` is regenerated.
**The paper's N\* must be requoted** — the old n_star=128 was an artifact of the in-sample
scoring; semantics also changed (curve tops out at nd=128 now).

### Fix 7 — rare.py
`rare.json` — 13/16 FORM rows sit at the cap; "FORM misses by 2.97 decades" is provisional
until rerun. Cited only in HANDOFF.md (§3, §10.1), not in any paper draft.

### Fixes 6, 11 — already handled or cheap
`subcell_from_returns.json` was regenerated post-fix (modified in tree). `bag_belief.json` is
stale (odom shift + spread poisoning) — cheap to regenerate.

### No reruns needed
Fix 8 (`second_order.py` never ran successfully since the tuple grew — no committed
artifacts); fix 9 (sweep/holdout modes always forwarded configs correctly; only ad-hoc
single-mode runs with non-default flags were wrong, and none are tracked).

## Suggested rerun order (dependencies first)

1. `lidar_belief.py` run + `belief_sweep.py` (36 scenarios) → refit `belief_model.npz`
   (foundation for everything measured-belief).
2. `realistic_sigma.py --seeds 100` per family/noise (needs 1; also picks up fixes 1+2) →
   then `plot_paper.py` figures (budget curve, Jensen).
3. Clark chain under fixed noise: `clark.py`, `clark_full.py`, `clark_fast.py`,
   `clark_conv.py`, `clark_grad.py`, `clark_hinge.py`, `clark_hinge_fast.py`.
4. Ranking/sensing arms with occlusion/localisation/all + tau-b everywhere:
   `ranking.py`, `elites.py`, `sensing/pipeline.py` (sweep + holdout).
5. Lower priority (refuted-route bookkeeping, numbers only): DFL suite, `risk.py`,
   `certify.py`, `bundled.py`, `order2.py`, `softgrad.py`, `observability.py`, `rare.py`,
   `run_bench.py`, `bag_belief.py`.

## Known still-unfixed (flagged, deliberately left)

- `bench/clark_full.py::matched_budget_curve` has the SAME in-sample truth bug as fix 2
  (pick pool is a prefix of the truth pool). `fig_budget` no longer reads it, but any future
  use of `clark_full.json["budget"]` must fix it first.
- Review cleanup findings not applied (kept per surgical-changes rule): divergent
  `correlated_field` twins (`noise.py` vs `bundled.py`), triplicated cloud→grid binning with
  a live Bessel divergence, ~230 copy-pasted certify lines in `rare.py` (`*4` suffix), dead
  `world.py::build_lanes`, orphaned MC loop + dead import in the old `fig_jensen` path.
- `belief_model.npz` / `rho_measured.npy` still have no committed producer script —
  regenerate them from a script this time so the measured-belief chain is reproducible.
