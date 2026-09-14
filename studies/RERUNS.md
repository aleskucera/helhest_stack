
## 2026-08-31 v2 attitude-cost campaign (PREREG_attitude_cost.md @clark_paper 5f72226)
- studies/out/v2/: F1.json, gate.json (21/22 kept), heldout_score_{foresight,hindsight}.json,
  REPORT_v2.json, FINAL_REPORT_v2.md, dcheck_refit.json, runner.log.
- Runner: studies/baseprod/v2_runner.py @2ee6834+, executed on dasenka (tmux v2_run),
  single pass, all freeze conditions passed, verdicts V2-1/2/3 PASS both conditions.

## 2026-09-11 gated-fold alpha* sweep, DESIGN traverses only (post hoc, not pre-registered)
- studies/out/v2/gate_sweep/: design_sweep_summary.json, design_sweep_windows.json, run.log.
- Script: studies/baseprod/v2_gate_sweep.py (first run at this commit), executed on dasenka from
  /local/kuceral4/baseprod/studies_v2 (deployed copy identical to studies/ here), 24 workers,
  windows out/windows/{foresight,hindsight}_v2/{t1,t2} (the v2 design builds of 2026-08-31).
- Per pairwise fold: gate on alpha (exact, with covariance) or on the marginal-only lower bound
  (mu_lead - mu_j)/(sd_lead + sd_j); gated folds keep the running leader's moments (the FOSM
  step). alpha* = 0 reproduces the frozen fosm arm and alpha* = inf the frozen clark arm to 1e-9
  in every window (asserted). Referee: 20k draws from the window belief, seed crc32(name).
- Applies MIN_RETAINED = 0.50 (prereg floor); 94 windows, 1 flagged, 2 unscoreable, 46 + 46 scored.
- Headline (exact gate, medians F / H): P[fold alpha >= 2] = 0.684 / 0.730; at alpha* = 1.5 the
  gate skips 76% / 81% of folds, declines 3.6% / 4.7% of the Jensen lift, rel err E 1.5e-3 /
  8.3e-4 (clark 2.1e-3 / 7.5e-4); sd/MC 0.952 / 0.971 (clark 0.948 / 0.966); CVaR error minimum
  at alpha* = 0.75 (0.0046 / 0.0032, uncorrected) because the sd deficit accumulates in folds
  with alpha in ~0.5-2; below alpha* ~1 partial gating has a worst-window mean error of 15% (H).
- Not for the paper's held-out claims. If a gated arm is scored on held-out data, alpha* must be
  fixed from this table first and the result labelled post hoc (or a new pre-registration).

## 2026-09-11 post hoc readings of the frozen held-out record (review round; no new scoring)
- studies/out/v2/posthoc_review.json from studies/baseprod/v2_posthoc_review.py, a function of
  heldout_score_{foresight,hindsight}.json only: the reference CVaR's definition (empirical tail)
  and its gap to the Gaussian form on the referee's own moments; the referee's relative standard
  error (mean, sd); the hybrid arm (fold mean, linearization sd) CVaR error; Table II at two
  significant figures; exact McNemar on the fans' top-1 hits (clark-corr vs fosm / mc-32 / mean-map).

## 2026-09-11 maximum-level calibration of the wheel support against the DSM (post hoc)
- studies/out/v2/max_calibration/: max_calibration_summary.json, max_calibration_windows.json, run.log,
  from studies/baseprod/v2_max_calibration.py (first run at this commit), on dasenka, all v2 windows
  (design t1/t2 and held-out, labelled; the held-out truth was consumed by E2 pre-registered, this
  reading is post hoc). Per node: true support (truth_max + cap, max over the element), the fold's
  E and sd, the max of means, contest depth alpha. Per-traverse median of (truth - max of means)
  removed (the registration's constant term carries the wheel radius).
- Reading: the truth's lift over the max of means is within +-0.5 cm of zero in every alpha bin
  below 3 on design and held-out, against a predicted Jensen lift of +0.6 to +0.8 cm at alpha < 0.25;
  the bin means are negative (-1 to -6 cm at low alpha), the same regression-on-the-mean signature
  as the simulated clearance campaign (clark_paper sim_campaign/rubble/clearance/investigate/).
  Nodes at alpha > 3 (1-2 %) sit 0.15-0.9 m above the truth: phantom winners. Fold |z| <= 1
  coverage 0.20-0.26 (the belief's variance, as E2 measured).

## 2026-09-13 fold-order ablation, DESIGN traverses (post hoc)
- studies/out/v2/fold_order/: fold_order_summary.json, fold_order_windows.json, run.log, from
  studies/baseprod/v2_fold_order.py on dasenka (46 + 46 unflagged design windows, 20k-draw referee,
  seed crc32 of the window name). Orders: desc (the paper's), asc, var_desc, var_asc, random.
- Reading: median |dE/E| between an order and desc 1.3e-4 to 8.4e-4 (max 8.2e-3), median |dsd/sd|
  6.6e-4 to 3.2e-3 (max 2.7e-2); median rel err E against the referee 2.10-2.28e-3 (F) and
  7.5-9.5e-4 (H) for every order, sd/MC 0.946-0.949 / 0.965-0.966; desc is best or tied-best on E.

## 2026-09-13 tail levels and approximation-error decomposition, all v2 windows (post hoc)
- studies/out/v2/tail_decomp/: tail_decomp_summary.json, tail_decomp_windows.json, run.log, from
  studies/baseprod/v2_tail_decomp.py on dasenka (fast-BLAS venv /local/kuceral4/venv_fast, numpy 2.2.6);
  referee re-drawn, 20k draws, seed crc32 of the window name; design and held-out windows both
  included (363 foresight, 368 hindsight unflagged). Post hoc, not pre-registered.
- Reading: referee cost skew 0.28 / 0.21, excess kurtosis 0.17 / 0.09 (medians). The fold's sd error
  against the quadratic form on the referee's OWN support moments is 5.3 % / 3.4 % (the moment-matching
  deficit); the Gaussian quadratic-form assumption on true moments costs 0.7 % / 0.3 %. CVaR relative
  error medians at q = 0.90 / 0.95 / 0.99: clark-corr 1.1 / 1.5 / 2.6 % (F), 0.5 / 0.8 / 1.4 % (H);
  fosm 2.7 / 2.9 / 3.5 % and 1.3 / 1.5 / 1.8 %; the Gaussian CVaR on the referee's own moments (the
  floor of any Gaussian-tail estimator) 0.9 / 1.4 / 2.6 % and 0.6 / 0.9 / 1.7 %: at q = 0.99 the
  corrected fold is at the floor.

## 2026-09-13 cross-site check of the sd correction on Oxford Spires (post hoc)
- studies/out/v2/crosssite/: crosssite_summary_all.json (the reading; every Spires window carries
  BASEPROD's sigma flag, so crosssite_summary.json, unflagged only, is empty), crosssite_windows.json,
  run.log, from studies/baseprod/v2_crosssite.py on dasenka (fast-BLAS venv), 20k-draw referee, seed
  crc32 of the window name. Windows: the E1 campaign's (~/data/oxford_spires/out, copied to
  dasenka:/local/kuceral4/spires_windows), keble-college-02/03-default (design site) and the four
  virgin sequences (blenheim-palace-01/02, christ-church-02/03); -02-unclamped skipped.
- Reading (medians; keble 29, blenheim 38, christ-church 43 windows; alpha_w ~0.15 everywhere; belief
  sigma 0.29 / 0.35 / 0.84 m): rel err E clark 2.7 / 2.0 / 1.1 % vs fosm 8.2 / 7.4 / 6.2 %, clark below
  fosm in 28/29, 38/38, 42/43; raw fold sd ratio 0.90 / 0.91 / 0.96, with the FROZEN BASEPROD law
  (a = 0.110, b = 1.0209) 0.99 / 0.99 / 1.06; fosm sd ratio 0.97 / 1.01 / 1.02; corrected sd error
  below raw in 24/29, 31/38, 14/43 and below fosm in 15/29, 19/38, 14/43; corrected CVaR error below
  fosm in 7/29, 10/38, 15/43. Keble refit degenerate (b = 0.05, the grid floor): alpha_w does not
  vary enough on Spires to identify the law.

## 2026-09-13 unscented-transform baseline, all v2 windows (post hoc)
- studies/out/v2/unscented/ (standard UT, alpha = 1) and studies/out/v2/unscented_scaled/ (scaled UT,
  alpha in {1, 0.3, 0.1, 0.03, 0.01}, kappa = 0, beta = 2), each with unscented_summary.json,
  unscented_windows.json, run.log, from studies/baseprod/v2_unscented.py on dasenka (fast-BLAS venv);
  2N + 1 sigma points per window (median 5,041 foresight / 6,553 hindsight), referee re-drawn (20k,
  crc32 seeds), design and held-out windows both included (363 / 368 unflagged).
- Reading (medians): the standard UT is unusable at this dimension (rel err E 0.14 / 0.073, sd 17x /
  9.5x the referee's); the scaled UT is best at alpha = 0.03: rel err E 9.7e-3 / 4.3e-3 (fold 2.0e-3 /
  8.5e-4, fosm 2.4e-2 / 1.5e-2), sd/MC 1.42 / 1.26, CVaR abs err median 0.027 / 0.016 (p95 1.22 / 1.03)
  against clark-corr 0.0045 / 0.0019 and fosm 0.013 / 0.0075. clark-corr beats the standard UT in
  362 / 363 and 363 / 368 windows on E and in every window on sd and CVaR.

## 2026-09-13 lineage note: the GPU wall benchmark
- The 5.5 us/plan, 5.8x, 0.4x, ~5,000-draw and 290x figures of the paper's Section III-I come from
  helhest_stack-study studies/out/v2_att_wall_3090.json @ db0491f (studies/bench/v2_att_wall.py,
  RTX 3090, P = 256, T = 40, cell 0.10 m), a third repository; the 290x is the sampler's price at
  its 4,096-draw cap, and the equal-accuracy count 4,946 is the 1/sqrt(D) extrapolation.
- 2026-09-13: v2_posthoc_review.py binom_two_sided tolerance made relative (F-11 of the methodology
  review); the hindsight hybrid sign-test p changes from 2.4e-15 to 2.3e-18; no other number moves.

### 2026-09-14: registration-exclusion analysis extended (raw and fosm statistics)

`baseprod/posthoc_registration_exclusion.py` now also reports, over the same pooled
rows and the same frozen k, the raw (k = 1) sd ratios of clark and fosm and fosm at
the frozen k, as scored (asserted against the pooled tables in
`heldout_score_{foresight,hindsight}.json`) and excluding traverse
2023-07-20_19-12-27. No rebuild, no new draws. Excluding the traverse: raw clark
17.21 / 19.64 (fosm 15.86 / 17.80); frozen-k fosm coverage 0.7850 / 0.7245,
sd ratio 0.9429 / 1.0736. The existing fields are unchanged (clark 1.0231 / 1.1846,
coverage 0.7352 / 0.6904). Post hoc; E2-ii's pre-registered verdict stands. Cited by
the paper's Limitations paragraph.

## 2026-09-14 cross-site check rerun with the E1 overhang flag (post hoc)
- studies/out/v2/crosssite_overhang/: crosssite_rows.jsonl (per-window rows as they finished),
  crosssite_windows.json, crosssite_summary{,_all,_clean,_overhang}.json, run.log (+ two aborted
  attempts' logs), from studies/baseprod/v2_crosssite.py --overhang-json (the three E1 artifacts
  out/risk_calibration{,_heldout}/e{0,1}_*.json) on dasenka, fast-BLAS venv, 20k-draw referee,
  same seeds as the 2026-09-13 run: every arm and referee moment reproduces it to 9e-9 relative.
- The Spires walks pass under arches, cloisters and porches; both the belief and the TLS truth
  keep the max height per cell, so the roof is aliased onto the ground track (RISK_CAL_DATA_AUDIT
  R1). The E1 overhang flag (belief-to-ground gap > 1 m along the track, computed in E1 before any
  cross-site scoring) marks 57 of the 111 windows: keble 12/29, blenheim 10/38, christ-church
  34/43 (5 clean, 4 without an E1 record); 10 scored windows have no E1 record and are in
  neither subset.
- CLEAN windows (44; medians raw fold sd ratio / with the frozen BASEPROD law / fosm): keble 13
  windows 0.84 / 0.93 / 0.95; blenheim 26 windows 0.89 / 0.98 / 0.99; christ-church 5 windows
  0.97 / 1.07 / 1.00. OVERHANG windows (56): 0.95-0.96 raw, 1.05-1.06 corrected, 1.03-1.05 fosm
  on every site. The mean result is unchanged in both subsets (clark below fosm in 44/44 clean,
  54/56 overhang). Reading: the deficit on clean ground is at or below the law's floor and the
  correction lands at or under one; under overhangs the deficit is shallow (belief sigma 0.7-1.2 m
  there) and the law overshoots by 5 % at every site. "Over-corrects the third site" is an
  overhang effect; christ-church is the site made of overhang windows.
- Two aborted attempts (run.attempt{1,2}.log): with OMP/OPENBLAS threads = 2 per worker the pool
  deadlocked on the six largest christ-church-02 windows (workers asleep in the BLAS thread after
  fork); threads = 1 finished them. The script now persists rows per window and resumes.

## 2026-09-14 post hoc: single scalar vs the alpha_w law on the held-out record
- studies/out/v2/posthoc_scalar_law.json from studies/baseprod/v2_posthoc_scalar_law.py, reading
  the committed heldout_score_{foresight,hindsight}.json (no new draws, law frozen). Across the six
  alpha_w-quantile bins the raw fold sd ratio rises 0.027 / 0.028 while the law rises 0.043 /
  0.045; a single scalar (1.065 / 1.037) gives median |sd ratio - 1| 0.0301 / 0.0183 against the
  law's 0.0280 / 0.0180. Reading (the ICRA-style review's point): the alpha_w dependence of the
  correction is not established on the held-out record; a scalar does as well. Also records the
  reference CVaR_0.9 medians 0.500 / 0.630 (cost units) for the Table II caption.

## 2026-09-14 softmax (log-sum-exp) arms, all v2 windows (post hoc)
- studies/out/v2/softmax/: softmax_summary.json, softmax_windows.json, softmax_rows.jsonl, run.log,
  from studies/baseprod/v2_softmax.py on dasenka (fast-BLAS venv, 24 workers, single-threaded
  BLAS), referee re-drawn (20k, crc32 seeds), design and held-out windows both included (363 / 368
  unflagged). Arms: the softened contact at one global temperature (5 mm to 10 cm, first-order
  propagation through the softmax weights) and the belief-scaled softmax with the per-node
  temperature a / 1.702 of Section III-C (the ICRA-style review's construction).
- Reading (medians, foresight / hindsight): the global temperature is best at its smallest value
  (5 mm), where it is linearization (rel err E 2.4e-2 / 8.0e-3 against fosm 2.4e-2 / 1.5e-2), and
  degrades monotonically above it (10 cm: 0.13 / 0.12). The belief-scaled softmax recovers part of
  the mean (8.6e-3 / 5.3e-3, below fosm in 295 / 363 and 305 / 368 windows) but not the fold's
  (2.0e-3 / 8.5e-4; the fold below it in 335 / 363 and 349 / 368), and it understates the sd
  (sd/MC 0.896 / 0.935, worse than the uncorrected fold's 0.940 / 0.964), so its CVaR error
  (0.0148 / 0.0090) is above linearization's (0.0126 / 0.0075); clark-corr below it in 351 / 363 and
  350 / 368 windows, the hybrid in 328 / 363 and 318 / 368.

## 2026-09-14 path-selection fans rerun with the C-FOSM arm; C-FOSM threshold verdicts (post hoc)
- studies/out/v2/fan_cfosm/: fan_cfosm_summary.json, fan_cfosm_rows.jsonl, run.log, from
  studies/baseprod/v2_fan_cfosm.py on dasenka (fast-BLAS venv, 24 workers, single-threaded BLAS):
  the 284 / 289 scoreable fans of the frozen held-out pass, same candidate seed (17) and referee
  seed (18), 8,000 shared draws, with c-fosm (fold mean, fosm sd) added; every pre-registered
  arm's regret and top-1 reproduce the frozen artifact in all 573 fans (asserted per fan).
- Reading: c-fosm top-1 278 / 284 (97.9 %) and 278 / 289 (96.2 %) against clark-corr 279 / 283
  (98.2 / 97.9 %), fosm 272 / 270 (95.8 / 93.4 %), mc-32 255 / 256, mean map 255 / 255; median
  regret 0 for every arm. Exact McNemar on discordant hits: c-fosm vs fosm 7-1 (p = 0.070) and
  8-0 (p = 0.0078); vs clark-corr 2-3 (p = 1.0) and 0-5 (p = 0.0625); vs mc-32 24-1 and 26-4,
  vs mean map 25-2 and 26-3 (p < 1e-4 each). c-fosm ranks like the fold, is not separated from it,
  and beats linearization significantly only under hindsight.
- Threshold verdicts (v2_posthoc_review.py, 'hybrid' arm derived from the frozen clark E and fosm
  sd, no new draws; out/v2/posthoc_review.json): disagreement with the reference at the reference
  CVaR's q25/q50/q75/q90 thresholds 1.6 / 0.9 / 0.3 / 0.0 % (foresight) and 0.0 / 0.3 / 0.0 /
  0.0 % (hindsight), against clark-corr 1.6 / 0.6 / 0.6 / 0.0 and 0 / 0 / 0 / 0, fosm 1.3 / 1.6 /
  0.3 / 0.3 and 0.0 / 0.6 / 0.3 / 0.3. No separation from linearization on verdicts.

## 2026-09-14 C-FOSM traverse-level and window-level readings (post hoc, additive)
- studies/baseprod/v2_posthoc_review.py extended and rerun (out/v2/posthoc_review.json): the
  hybrid arm (the fold's E with linearization's sd through the Gaussian CVaR, no new draws) now
  also carries the per-traverse win fractions, the window counts against the baselines, and a
  cluster-bootstrap ratio, the readings the paper quotes now that the fitted-scale arm
  (clark-corr) is no longer reported in it. Purely additive: every pre-existing key in
  posthoc_review.json reproduces bit-for-bit, verified by a key-wise diff against the previous
  file. The script asserts V2-1_E_vs_fosm_hybrid == V2-1_E_vs_fosm (the hybrid's mean IS the
  fold's).
- Reading (foresight / hindsight): the hybrid's CVaR error is below linearization's in 225 / 317
  and 226 / 323 windows, a majority in 16 and 17 of the 18 traverses, cluster-bootstrap 95 %
  ratio of median errors 1.75-3.12 and 1.70-2.83. It beats mc-32 on BOTH moments in 216 / 317 and
  246 / 323 windows, a majority in 16 and 17 traverses; below mc-32 on CVaR in 194 / 317 and
  242 / 323. Threshold-verdict disagreement 1.6 / 0.9 / 0.3 / 0.0 % and 0.0 / 0.3 / 0.0 / 0.0 %,
  the same 0-1.6 % band as clark-corr and fosm.
- For comparison, the fitted-scale arm's own figures (still in the artifact, no longer in the
  paper): CVaR below fosm in a majority of 16 and 18 traverses, bootstrap 2.05-3.60 and
  2.44-4.16, both moments below mc-32 in 231 / 317 and 255 / 323.
