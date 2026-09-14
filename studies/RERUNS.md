
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
