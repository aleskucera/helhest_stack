
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
