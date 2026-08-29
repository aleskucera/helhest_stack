# E2 FINAL REPORT — BASEPROD risk calibration on contested terrain

Written 2026-08-28 23:09:47 +0200. Pipeline commit `a02dfd9bd5d63f64c8b0d6abf3f486695f0cc3e4`. Frozen at 2026-08-28 19:48:17 +0200. The held-out run happened once; there is no second draw.

The author ran this chain unattended, without prior review of the design tables (PREREG_baseprod.md, authorization update 2026-08-28 evening). The full design-phase tables are therefore reproduced in this report, after the held-out results, for post-hoc review.

Configuration: 4 m windows, 7 m look-back, arms ['clark', 'clark-diag', 'fosm', 'step-form', 'mc-2', 'mc-32'] + mean-map.

**Contents** — held-out results per condition, then the criteria, then the design-phase appendix.

## HINDSIGHT — contrast

357 window products, 334 retained, 334 unflagged.

Frozen recalibration: k = 275.0293778920172, tau = 0.0 m (fitted on design, not refitted here).

| arm | | mean NLL | \|z\|<=1 | \|z\|<=2 | sd-ratio | mean z | median sd |
|---|---|---|---|---|---|---|---|
| clark | raw | 1505.0359 | 0.0419 | 0.0778 | 53.8485 | -10.9167 | 0.358 |
| clark | recal | 8.2413 | 0.6707 | 0.9132 | 3.247 | -0.6583 | 5.9364 |
| clark-diag | raw | 1893.4442 | 0.0329 | 0.0749 | 59.7279 | -15.185 | 0.3144 |
| clark-diag | recal | 9.4761 | 0.5928 | 0.8683 | 3.6015 | -0.9156 | 5.2136 |
| fosm | raw | 1013.5967 | 0.0389 | 0.0988 | 44.2113 | -8.8516 | 0.3756 |
| fosm | recal | 6.5239 | 0.7036 | 0.9251 | 2.6659 | -0.5337 | 6.229 |
| step-form | raw | 14.2692 | 0.3323 | 0.5629 | 4.8114 | -1.3504 | 2.2063 |
| step-form | recal | 4.6709 | 0.991 | 0.997 | 0.2901 | -0.0814 | 36.589 |
| mc-2 | raw | 17009.1005 | 0.0419 | 0.0898 | 182.486 | -28.5992 | 0.2582 |
| mc-2 | recal | 64.0733 | 0.506 | 0.6976 | 11.0037 | -1.7245 | 4.282 |
| mc-32 | raw | 1230.2171 | 0.0419 | 0.0838 | 48.6178 | -10.1897 | 0.3707 |
| mc-32 | recal | 7.2765 | 0.6737 | 0.9132 | 2.9316 | -0.6144 | 6.1477 |

mean-map (point prediction, no variance): mean error -6.6785, median |error| 3.6418, median |error| per step 0.0958 m

### Criteria

| criterion | value | verdict |
|---|---|---|
| E2-i | 0.6707 | PASS |
| E2-ii | 3.247 | FAIL |
| E2-iii | {'a_median_rel_err_E': 3.2123584119804137e-07, 'a_median_le_5pct': True, 'a_n_clark_better_E': 334, 'a_n': 334, 'a_sign_test_p': 0.0, 'a_majority_and_significant': True, 'a_pass': True, 'b_sd_ratio_to_mc': 0.9564009949208846, 'b_band': [0.85, 1.05], 'b_pass': True} | PASS |
| E2-v | {'n_clark_better_both': 216, 'sign_test_p': 0.0} | PASS |
| E2-iv | 0.9117 | reported, no criterion |

### Belief referee — E2-iii / E2-v machinery, exercised in-sample

334 unflagged windows, 20000 draws each (seed 20260828; the mc-N arms are subsampled without replacement with seed 20260828). Raw Var, no truth, no recalibration.

| arm | median rel err E | median rel err sd | p95 E | p95 sd | sd-ratio to MC (median) |
|---|---|---|---|---|---|
| clark | 3.212e-07 | 4.360e-02 | 1.023e-06 | 1.261e-01 | 0.9564 |
| clark-diag | 1.027e-06 | 1.330e-01 | 7.858e-06 | 5.087e-01 | 0.8670 |
| fosm | 1.649e-05 | 1.934e-02 | 9.416e-05 | 8.495e-02 | 1.0134 |
| step-form | 1.649e-05 | 4.986e+00 | 9.416e-05 | 8.510e+00 | 5.9864 |
| mc-2 | 1.178e-05 | 5.047e-01 | 4.457e-05 | 9.966e-01 | 0.6850 |
| mc-32 | 2.782e-06 | 9.207e-02 | 1.219e-05 | 2.455e-01 | 0.9827 |
| mean-map | 1.649e-05 | — (no predicted sd) | — | — | — |

**E2-iii preview** (in-sample; the criterion is held-out only):

- (a) MEAN — clark's median relative error 3.212e-07 (<= 5%), below fosm's in 334 of 334 windows, two-sided sign test p = 0.0 (majority + significant)
- (b) SD — clark's sd-ratio to the belief MC, median **0.9564** against the band [0.85, 1.05] (inside); mean 0.9471, p5 0.8739, p95 0.9908. No sd criterion against fosm.

**E2-v preview** — clark below mc-32 on BOTH moments in 216 of 334 windows, exact two-sided sign test p = 0.0 (E alone 313, sd alone 231). Against mc-2: both moments 316 of 334, p = 0.0.

- clark vs step-form: both moments 334 of 334 (p = 0.0), E alone 334, sd alone 334
- clark vs clark-diag: both moments 256 of 334 (p = 0.0), E alone 297, sd alone 277

#### Reported characterization — the sd-deficit-vs-contest-depth curve

clark's belief-referee sd-ratio (sd_clark / sd_MC) binned by the window's cost-weighted median alpha. No criterion attaches. The historical record is 0.90-0.94; the derived worst case is ~0.89 at alpha ~ 0.

| alpha bin | n | alpha median | clark sd-ratio median | p5 | p95 |
|---|---|---|---|---|---|
| [0.0, 0.2) | 4 | 0.1377 | **0.9126** | 0.8755 | 0.9204 |
| [0.2, 0.4) | 57 | 0.3298 | **0.9398** | 0.8317 | 0.9804 |
| [0.4, 0.6) | 126 | 0.5267 | **0.9539** | 0.8772 | 0.9854 |
| [0.6, 0.8) | 80 | 0.6831 | **0.9594** | 0.9142 | 0.9912 |
| [0.8, 1.0) | 33 | 0.856 | **0.9685** | 0.8941 | 0.9953 |
| [1.0, 1.5) | 26 | 1.1194 | **0.9702** | 0.8971 | 0.9976 |
| [1.5, inf) | 8 | 1.7226 | **0.9656** | 0.9122 | 0.9964 |

corr(alpha, clark sd-ratio) = 0.2369

#### Reported characterization — belief-referee CVaR error (q = 0.9)

|CVaR_q(arm) − CVaR_q(MC)| per window: the single number a planner consumes, fusing both moments through the decision rule's own tail weight. The arm CVaR is the Gaussian form E + phi(z_q)/(1−q) · sd; the MC CVaR is the mean of the empirical upper tail. mean-map has no sd, so its CVaR is its point prediction — the floor a risk-blind planner sits at. No criterion attaches.

| arm | median | mean | p95 | max |
|---|---|---|---|---|
| clark | **0.0280** | 0.0393 | 0.1242 | 0.2498 |
| clark-diag | **0.0717** | 0.1617 | 0.6726 | 1.0112 |
| fosm | **0.2482** | 0.3994 | 1.4065 | 3.1101 |
| step-form | **2.9208** | 3.6139 | 7.6013 | 11.0314 |
| mc-2 | **0.3845** | 0.5156 | 1.5880 | 3.6148 |
| mc-32 | **0.0759** | 0.1034 | 0.3022 | 0.6145 |
| mean-map | **0.9098** | 1.2333 | 3.3072 | 5.0953 |

Lowest median absolute CVaR error: **clark**

Fixed before any result existed: the reality comparison between arms conflates mapper and estimator. If the belief is faithful, the fold's advantage is expected to appear against reality too; if the belief is biased, ANY arm -- including cruder ones -- can win, because an estimator faithful to a wrong belief amplifies the belief's error while a crude one ignores the term the error feeds. Such an outcome grades the mapper, not the estimator.

| statistic | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| sigma / relief | 0.0119 | 0.0521 | **0.0923** | 0.1458 | 0.9048 |
| alpha (cost-weighted median) | 0.1165 | 0.4561 | **0.5768** | 0.7362 | 2.1459 |
| fraction |alpha| < 1 | 0.108 | 0.6525 | **0.7547** | 0.8332 | 0.9946 |

## FORESIGHT — PRIMARY

337 window products, 332 retained, 332 unflagged.

Frozen recalibration: k = 282.99667205586184, tau = 0.0 m (fitted on design, not refitted here).

| arm | | mean NLL | \|z\|<=1 | \|z\|<=2 | sd-ratio | mean z | median sd |
|---|---|---|---|---|---|---|---|
| clark | raw | 905.3099 | 0.0331 | 0.1084 | 41.6958 | -8.8142 | 0.3139 |
| clark | recal | 5.8358 | 0.7139 | 0.9307 | 2.4786 | -0.524 | 5.2811 |
| clark-diag | raw | 1347.9559 | 0.0271 | 0.0964 | 50.3912 | -12.8449 | 0.2741 |
| clark-diag | recal | 7.2327 | 0.6446 | 0.8886 | 2.9955 | -0.7636 | 4.6118 |
| fosm | raw | 597.3982 | 0.0723 | 0.1295 | 33.916 | -6.9396 | 0.3425 |
| fosm | recal | 4.8375 | 0.762 | 0.9337 | 2.0161 | -0.4125 | 5.7618 |
| step-form | raw | 10.5448 | 0.4096 | 0.6596 | 4.006 | -1.2116 | 2.1993 |
| step-form | recal | 4.6645 | 0.991 | 0.997 | 0.2381 | -0.072 | 36.9975 |
| mc-2 | raw | 53956.6182 | 0.0422 | 0.0843 | 324.757 | -52.5864 | 0.213 |
| mc-2 | recal | 192.7303 | 0.5512 | 0.7199 | 19.3049 | -3.126 | 3.5838 |
| mc-32 | raw | 788.5834 | 0.0422 | 0.1235 | 38.9174 | -8.2101 | 0.3318 |
| mc-32 | recal | 5.4888 | 0.741 | 0.9187 | 2.3134 | -0.488 | 5.5813 |

mean-map (point prediction, no variance): mean error -5.7888, median |error| 2.9084, median |error| per step 0.07638 m

### Criteria

| criterion | value | verdict |
|---|---|---|
| E2-i | 0.7139 | PASS |
| E2-ii | 2.4786 | FAIL |
| E2-iii | {'a_median_rel_err_E': 3.148604710502136e-07, 'a_median_le_5pct': True, 'a_n_clark_better_E': 331, 'a_n': 332, 'a_sign_test_p': 0.0, 'a_majority_and_significant': True, 'a_pass': True, 'b_sd_ratio_to_mc': 0.9341397875929586, 'b_band': [0.85, 1.05], 'b_pass': True} | PASS |
| E2-v | {'n_clark_better_both': 193, 'sign_test_p': 0.00357} | PASS |
| E2-iv | 0.9284 | reported, no criterion |

### Belief referee — E2-iii / E2-v machinery, exercised in-sample

332 unflagged windows, 20000 draws each (seed 20260828; the mc-N arms are subsampled without replacement with seed 20260828). Raw Var, no truth, no recalibration.

| arm | median rel err E | median rel err sd | p95 E | p95 sd | sd-ratio to MC (median) |
|---|---|---|---|---|---|
| clark | 3.149e-07 | 6.586e-02 | 1.260e-06 | 1.800e-01 | 0.9341 |
| clark-diag | 8.990e-07 | 1.766e-01 | 7.282e-06 | 5.233e-01 | 0.8234 |
| fosm | 1.976e-05 | 2.528e-02 | 8.789e-05 | 9.541e-02 | 1.0093 |
| step-form | 1.976e-05 | 5.714e+00 | 8.789e-05 | 9.284e+00 | 6.7143 |
| mc-2 | 1.039e-05 | 5.307e-01 | 4.604e-05 | 9.793e-01 | 0.6345 |
| mc-32 | 2.833e-06 | 9.885e-02 | 1.351e-05 | 2.421e-01 | 0.9967 |
| mean-map | 1.976e-05 | — (no predicted sd) | — | — | — |

**E2-iii preview** (in-sample; the criterion is held-out only):

- (a) MEAN — clark's median relative error 3.149e-07 (<= 5%), below fosm's in 331 of 332 windows, two-sided sign test p = 0.0 (majority + significant)
- (b) SD — clark's sd-ratio to the belief MC, median **0.9341** against the band [0.85, 1.05] (inside); mean 0.9229, p5 0.8200, p95 0.9860. No sd criterion against fosm.

**E2-v preview** — clark below mc-32 on BOTH moments in 193 of 332 windows, exact two-sided sign test p = 0.00357 (E alone 306, sd alone 208). Against mc-2: both moments 297 of 332, p = 0.0.

- clark vs step-form: both moments 331 of 332 (p = 0.0), E alone 331, sd alone 332
- clark vs clark-diag: both moments 258 of 332 (p = 0.0), E alone 287, sd alone 276

#### Reported characterization — the sd-deficit-vs-contest-depth curve

clark's belief-referee sd-ratio (sd_clark / sd_MC) binned by the window's cost-weighted median alpha. No criterion attaches. The historical record is 0.90-0.94; the derived worst case is ~0.89 at alpha ~ 0.

| alpha bin | n | alpha median | clark sd-ratio median | p5 | p95 |
|---|---|---|---|---|---|
| [0.0, 0.2) | 8 | 0.1619 | **0.8929** | 0.8059 | 0.9248 |
| [0.2, 0.4) | 93 | 0.3266 | **0.9252** | 0.816 | 0.9724 |
| [0.4, 0.6) | 141 | 0.4889 | **0.9351** | 0.8304 | 0.9836 |
| [0.6, 0.8) | 52 | 0.6707 | **0.9518** | 0.8435 | 0.988 |
| [0.8, 1.0) | 22 | 0.8537 | **0.9511** | 0.8674 | 0.9845 |
| [1.0, 1.5) | 14 | 1.3202 | **0.9313** | 0.8691 | 0.9887 |
| [1.5, inf) | 2 | 1.8948 | **0.9662** | 0.9587 | 0.9736 |

corr(alpha, clark sd-ratio) = 0.1963

#### Reported characterization — belief-referee CVaR error (q = 0.9)

|CVaR_q(arm) − CVaR_q(MC)| per window: the single number a planner consumes, fusing both moments through the decision rule's own tail weight. The arm CVaR is the Gaussian form E + phi(z_q)/(1−q) · sd; the MC CVaR is the mean of the empirical upper tail. mean-map has no sd, so its CVaR is its point prediction — the floor a risk-blind planner sits at. No criterion attaches.

| arm | median | mean | p95 | max |
|---|---|---|---|---|
| clark | **0.0380** | 0.0497 | 0.1361 | 0.2305 |
| clark-diag | **0.0796** | 0.1542 | 0.5870 | 0.8195 |
| fosm | **0.2872** | 0.4214 | 1.3466 | 3.1368 |
| step-form | **3.0382** | 3.6726 | 7.9901 | 12.0674 |
| mc-2 | **0.3161** | 0.4666 | 1.6550 | 3.2121 |
| mc-32 | **0.0742** | 0.1089 | 0.3464 | 0.6667 |
| mean-map | **0.8877** | 1.1811 | 3.1746 | 5.1324 |

Lowest median absolute CVaR error: **clark**

Fixed before any result existed: the reality comparison between arms conflates mapper and estimator. If the belief is faithful, the fold's advantage is expected to appear against reality too; if the belief is biased, ANY arm -- including cruder ones -- can win, because an estimator faithful to a wrong belief amplifies the belief's error while a crude one ignores the term the error feeds. Such an outcome grades the mapper, not the estimator.

| statistic | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| sigma / relief | 0.0129 | 0.0617 | **0.1065** | 0.167 | 0.9623 |
| alpha (cost-weighted median) | 0.1274 | 0.3715 | **0.4831** | 0.6131 | 2.0846 |
| fraction |alpha| < 1 | 0.1837 | 0.7276 | **0.8374** | 0.9004 | 1.0 |

---

# Appendix — the design phase, in full

> **DESIGN-PHASE / IN-SAMPLE.** Two design traverses; per-condition (k, tau) was fitted on the same windows it is scored on. These numbers set the freeze record; they are not results. They are reproduced here because the chain ran without prior review of them.


Written 2026-08-28 19:48:17 +0200 by the detached E2 runner on `dasenka`, pipeline commit `a02dfd9bd5d63f64c8b0d6abf3f486695f0cc3e4`.

> **DESIGN-PHASE / IN-SAMPLE.** Two design traverses. Per-condition (k, tau) is fitted on the same windows it is scored on. No pre-registered criterion is evaluated here and none is implied; E2-i, E2-ii, E2-iii, E2-iv and E2-v are HELD-OUT criteria and the previews below only exercise their machinery. These numbers set the freeze record, they are not results.

Six author-approved prereg edits are in force (clark_paper `fab1a70`): windows 10 m -> 4 m (the depth range cap), step-form and mc-2/mc-32 join the arms, E2-iii splits by moment, E2-v is new, and two reported characterizations are added.

**The runner has stopped here by design.** The author narrowed the autonomous authorization on 2026-08-28: the held-out set is not touched until an explicit manual start (see *Starting the held-out run* below).

### Freeze conditions

| condition | verdict | statement |
|---|---|---|
| C1 | **PASS** | (a) the 10 m hindsight scoring path still reproduces rehearsal v3 within 1e-06 relative -- code parity preserved on the old configuration -- AND (b) the 4 m pipeline runs both conditions end to end with finite outputs |
| C2 | **PASS** | at least 60% of the 48 design windows (4 m) unflagged under foresight |
| C3 | **PASS** | all fits finite, the QC/registration machinery ran on both conditions without error, and NO NaN or degenerate value appears anywhere in the design tables (arm moments, pooled arm table, belief-referee table, regime statistics) |
| C4 | **PASS** | no held-out path accessed before the freeze record is written |

C1(a) code parity — the 10 m hindsight path was rebuilt and rescored by this same code (19 windows) and compared against rehearsal v3 at 1e-06 relative: **PASS**. That is what makes the drop to 4 m a configuration change and not a code change. C1(b) end-to-end at 4 m: **PASS**.

C2 detail — foresight produced 46 window products from the 48 design windows, 46 retained (>= 50% of steps scoreable), 46 unflagged (0.9583 of 48, threshold 60%). Excluded: 0. Flags raised: {}

Which QC rule binds under foresight (**diagnostic, not a criterion**): of 46 scoreable foresight windows, 0 fall below the 50% retention floor (QC flag 1) and 0 carry a coverage/sigma/void-fill flag (QC flags 2-3). A step is scoreable only when all twelve of its candidate cells carry a belief. At 10 m windows the retention floor was the binding rule and 11 of 17 windows failed it, because a 10 m window is never observed past the 4 m range cap from before its own start; matching the window length to the perception horizon is what this configuration change addresses.

| foresight window | retention | element-cell observed frac | median sigma (m) | flags |
|---|---|---|---|---|
| t1/window_001 | 1.0 | 1.0 | 0.01507 | - |
| t1/window_002 | 1.0 | 1.0 | 0.01314 | - |
| t1/window_003 | 1.0 | 1.0 | 0.01465 | - |
| t1/window_004 | 1.0 | 1.0 | 0.01499 | - |
| t1/window_005 | 1.0 | 1.0 | 0.01849 | - |
| t1/window_006 | 1.0 | 1.0 | 0.0218 | - |
| t1/window_007 | 1.0 | 1.0 | 0.01735 | - |
| t1/window_008 | 1.0 | 1.0 | 0.01486 | - |
| t1/window_009 | 1.0 | 1.0 | 0.01404 | - |
| t1/window_010 | 1.0 | 1.0 | 0.01842 | - |
| t1/window_011 | 1.0 | 1.0 | 0.02032 | - |
| t1/window_012 | 1.0 | 1.0 | 0.01269 | - |
| t2/window_001 | 1.0 | 1.0 | 0.02955 | - |
| t2/window_002 | 1.0 | 1.0 | 0.01456 | - |
| t2/window_003 | 0.8974 | 0.9977 | 0.02209 | - |
| t2/window_004 | 1.0 | 1.0 | 0.01473 | - |
| t2/window_005 | 1.0 | 1.0 | 0.01863 | - |
| t2/window_006 | 1.0 | 1.0 | 0.01977 | - |
| t2/window_007 | 1.0 | 1.0 | 0.0621 | - |
| t2/window_008 | 1.0 | 1.0 | 0.02509 | - |
| t2/window_009 | 1.0 | 1.0 | 0.02071 | - |
| t2/window_010 | 1.0 | 1.0 | 0.01208 | - |
| t2/window_011 | 1.0 | 1.0 | 0.01419 | - |
| t2/window_012 | 1.0 | 1.0 | 0.01725 | - |
| t2/window_013 | 1.0 | 1.0 | 0.01771 | - |
| t2/window_014 | 1.0 | 1.0 | 0.02067 | - |
| t2/window_015 | 1.0 | 1.0 | 0.03493 | - |
| t2/window_016 | 1.0 | 1.0 | 0.01544 | - |
| t2/window_017 | 1.0 | 1.0 | 0.01257 | - |
| t2/window_018 | 1.0 | 1.0 | 0.02124 | - |
| t2/window_019 | 1.0 | 1.0 | 0.014 | - |
| t2/window_020 | 1.0 | 1.0 | 0.05417 | - |
| t2/window_021 | 1.0 | 1.0 | 0.01334 | - |
| t2/window_022 | 1.0 | 1.0 | 0.01724 | - |
| t2/window_023 | 1.0 | 1.0 | 0.01972 | - |
| t2/window_024 | 1.0 | 1.0 | 0.02037 | - |
| t2/window_025 | 1.0 | 1.0 | 0.03886 | - |
| t2/window_026 | 1.0 | 1.0 | 0.01769 | - |
| t2/window_027 | 1.0 | 1.0 | 0.01798 | - |
| t2/window_028 | 1.0 | 1.0 | 0.02888 | - |
| t2/window_029 | 1.0 | 1.0 | 0.01729 | - |
| t2/window_030 | 1.0 | 1.0 | 0.02028 | - |
| t2/window_031 | 1.0 | 1.0 | 0.02356 | - |
| t2/window_032 | 1.0 | 1.0 | 0.01455 | - |
| t2/window_033 | 1.0 | 1.0 | 0.02471 | - |
| t2/window_034 | 1.0 | 1.0 | 0.01563 | - |

### Registration (truth side; no belief, no arm enters)

| traverse | selected model | residual sd | block-CV | plane-only CV | tilt (mm/m) | void-fill in corridor |
|---|---|---|---|---|---|---|
| t1 | plane | 0.0449 m | 0.0601 m | 0.0601 m | [38.18, 2.24] | 0.00078 |
| t2 | quadratic | 0.0322 m | 0.0654 m | 0.0704 m | [30.0, 5.02] | 0.00339 |

### The two pinned calibration constants, re-checked

- mount pitch offset: pinned **-1.4092 deg** (total 18.5908 deg); refit on this host **-1.4092 deg**, |diff| 1e-05 deg. The pinned value is what the build used; the refit is recorded only.
- stereo noise sigma_z(r) = a + b r^2: pinned **a = 0.05185, b = 0.00343**; refit **a = 0.05185, b = 0.00343**.
- intrinsics: fx = fy = 422.28125, cx = 425.4715881347656, cy = 236.69622802734375, zero distortion (CameraInfo; never refitted).

### Condition: HINDSIGHT (contrast — the belief at its lifetime best)

48 window products, 46 retained, 46 unflagged, 2 excluded. Flagged: `{}`. Excluded: `["t1/window_000", "t2/window_000"]`

Recalibration (ML on unflagged, arm `clark`, applied identically to every variance arm, E[cost] untouched): **k = 275.0293778920172**, **tau = 0.0 m**, sd multiplier 16.584009704893965, mean NLL at fit 3.4083805194530123. Pinned-tau (k = 1) alternative: tau = 0.34684467696608323 m, penalty 0.6008 nats.

#### Reality referee — arm table (pooled, unflagged)

| arm | | mean NLL | \|z\|<=1 | \|z\|<=2 | sd-ratio | mean z | median sd |
|---|---|---|---|---|---|---|---|
| clark | raw | 137.6146 | 0.0652 | 0.1087 | 15.6239 | -6.019 | 0.4195 |
| clark | recal | 3.4084 | 0.7826 | 0.9565 | 0.9421 | -0.3629 | 6.9572 |
| clark-diag | raw | 244.3254 | 0.0652 | 0.087 | 20.5979 | -8.5878 | 0.361 |
| clark-diag | recal | 3.6236 | 0.6739 | 0.8696 | 1.242 | -0.5178 | 5.9871 |
| fosm | raw | 112.8933 | 0.087 | 0.1087 | 14.3366 | -4.9361 | 0.4509 |
| fosm | recal | 3.394 | 0.8043 | 0.9565 | 0.8645 | -0.2976 | 7.4784 |
| step-form | raw | 6.3664 | 0.3043 | 0.5435 | 2.834 | -1.0444 | 2.3541 |
| step-form | recal | 4.7173 | 1.0 | 1.0 | 0.1709 | -0.063 | 39.0411 |
| mc-2 | raw | 7883.7218 | 0.087 | 0.087 | 126.2067 | -13.6788 | 0.2231 |
| mc-2 | recal | 30.7074 | 0.4565 | 0.6522 | 7.6101 | -0.8248 | 3.6991 |
| mc-32 | raw | 133.168 | 0.0652 | 0.1087 | 15.3699 | -5.9081 | 0.4552 |
| mc-32 | recal | 3.4582 | 0.7609 | 0.9348 | 0.9268 | -0.3563 | 7.5485 |

mean-map (point prediction, no variance): mean error -5.408, median |error| 3.9989, median |error| per step 0.10302 m

- clark minus fosm (recalibrated, negative = clark better): mean 0.0144, median -0.0171, clark better in 29 of 46, two-sided sign test p = 0.1038
- clark minus clark-diag (recalibrated, negative = clark better): mean -0.2152, median 0.0205, clark better in 17 of 46, two-sided sign test p = 0.1038
- clark minus step-form (recalibrated, negative = clark better): mean -1.3089, median -1.5006, clark better in 43 of 46, two-sided sign test p = 0.0
- clark minus mc-32 (recalibrated, negative = clark better): mean -0.0498, median -0.0411, clark better in 36 of 46, two-sided sign test p = 0.0002
- clark minus mc-2 (recalibrated, negative = clark better): mean -27.299, median -0.3038, clark better in 34 of 46, two-sided sign test p = 0.0016
- arm-NLL spread: 7877.3554 nats raw, 27.3134 nats recalibrated (288.4x compression)

_No pass/fail criterion attaches to any reality-side arm comparison (PREREG_baseprod.md)._

#### Belief referee — E2-iii / E2-v machinery, exercised in-sample

46 unflagged windows, 20000 draws each (seed 20260828; the mc-N arms are subsampled without replacement with seed 20260828). Raw Var, no truth, no recalibration.

| arm | median rel err E | median rel err sd | p95 E | p95 sd | sd-ratio to MC (median) |
|---|---|---|---|---|---|
| clark | 4.025e-07 | 3.885e-02 | 1.121e-06 | 1.419e-01 | 0.9612 |
| clark-diag | 1.351e-06 | 1.719e-01 | 7.488e-06 | 3.872e-01 | 0.8281 |
| fosm | 1.755e-05 | 1.986e-02 | 7.033e-05 | 7.685e-02 | 1.0185 |
| step-form | 1.755e-05 | 4.401e+00 | 7.033e-05 | 6.813e+00 | 5.4014 |
| mc-2 | 1.410e-05 | 6.654e-01 | 4.681e-05 | 9.722e-01 | 0.5373 |
| mc-32 | 4.346e-06 | 8.522e-02 | 1.121e-05 | 1.960e-01 | 1.0137 |
| mean-map | 1.755e-05 | — (no predicted sd) | — | — | — |

**E2-iii preview** (in-sample; the criterion is held-out only):

- (a) MEAN — clark's median relative error 4.025e-07 (<= 5%), below fosm's in 46 of 46 windows, two-sided sign test p = 0.0 (majority + significant)
- (b) SD — clark's sd-ratio to the belief MC, median **0.9612** against the band [0.85, 1.05] (inside); mean 0.9482, p5 0.8581, p95 0.9914. No sd criterion against fosm.

**E2-v preview** — clark below mc-32 on BOTH moments in 31 of 46 windows, exact two-sided sign test p = 0.0259 (E alone 44, sd alone 32). Against mc-2: both moments 43 of 46, p = 0.0.

- clark vs step-form: both moments 46 of 46 (p = 0.0), E alone 46, sd alone 46
- clark vs clark-diag: both moments 42 of 46 (p = 0.0), E alone 44, sd alone 42

##### Reported characterization — the sd-deficit-vs-contest-depth curve

clark's belief-referee sd-ratio (sd_clark / sd_MC) binned by the window's cost-weighted median alpha. No criterion attaches. The historical record is 0.90-0.94; the derived worst case is ~0.89 at alpha ~ 0.

| alpha bin | n | alpha median | clark sd-ratio median | p5 | p95 |
|---|---|---|---|---|---|
| [0.0, 0.2) | 0 | — | — | — | — |
| [0.2, 0.4) | 5 | 0.3628 | **0.9442** | 0.8758 | 0.9761 |
| [0.4, 0.6) | 16 | 0.5037 | **0.9378** | 0.8176 | 0.9736 |
| [0.6, 0.8) | 12 | 0.688 | **0.9673** | 0.9224 | 0.9902 |
| [0.8, 1.0) | 7 | 0.8482 | **0.9826** | 0.9611 | 0.9927 |
| [1.0, 1.5) | 5 | 1.1763 | **0.9838** | 0.9487 | 0.9915 |
| [1.5, inf) | 1 | 1.5111 | **0.9843** | 0.9843 | 0.9843 |

corr(alpha, clark sd-ratio) = 0.4455

##### Reported characterization — belief-referee CVaR error (q = 0.9)

|CVaR_q(arm) − CVaR_q(MC)| per window: the single number a planner consumes, fusing both moments through the decision rule's own tail weight. The arm CVaR is the Gaussian form E + phi(z_q)/(1−q) · sd; the MC CVaR is the mean of the empirical upper tail. mean-map has no sd, so its CVaR is its point prediction — the floor a risk-blind planner sits at. No criterion attaches.

| arm | median | mean | p95 | max |
|---|---|---|---|---|
| clark | **0.0287** | 0.0417 | 0.1116 | 0.2457 |
| clark-diag | **0.1011** | 0.1872 | 0.6069 | 0.8449 |
| fosm | **0.2556** | 0.3823 | 1.0412 | 1.5822 |
| step-form | **3.1033** | 3.8578 | 8.8094 | 9.1022 |
| mc-2 | **0.5240** | 0.6814 | 1.5550 | 2.1136 |
| mc-32 | **0.0660** | 0.1072 | 0.3973 | 0.4930 |
| mean-map | **0.9948** | 1.3175 | 2.9346 | 4.0032 |

Lowest median absolute CVaR error: **clark**

#### Regime

| statistic | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| belief sigma on element cells (m) | 0.0108 | 0.013 | **0.0146** | 0.0162 | 0.0537 |
| truth relief p95-p5 (m) | 0.0519 | 0.1183 | **0.1591** | 0.2614 | 0.604 |
| sigma / relief | 0.0277 | 0.0585 | **0.0903** | 0.1316 | 0.2704 |
| alpha (cost-weighted median) | 0.291 | 0.4997 | **0.6199** | 0.8233 | 1.5111 |
| fraction of cost-weighted nodes |alpha| < 1 | 0.3208 | 0.5869 | **0.6704** | 0.7977 | 0.9082 |
| QC element-cell observed fraction | 1.0 | 1.0 | **1.0** | 1.0 | 1.0 |
| QC median element-cell sigma (m) | 0.0108 | 0.013 | **0.0146** | 0.0162 | 0.0537 |
| retention | 1.0 | 1.0 | **1.0** | 1.0 | 1.0 |
| belief mu minus truth, per-window median (m) | -0.112 | -0.0124 | **0.0134** | 0.0331 | 0.7106 |

### Condition: FORESIGHT (PRIMARY — the planning-time belief)

46 window products, 46 retained, 46 unflagged, 0 excluded. Flagged: `{}`. Excluded: `[]`

Recalibration (ML on unflagged, arm `clark`, applied identically to every variance arm, E[cost] untouched): **k = 282.99667205586184**, **tau = 0.0 m**, sd multiplier 16.822504928097416, mean NLL at fit 3.2854412083375673. Pinned-tau (k = 1) alternative: tau = 0.34267602534331587 m, penalty 0.7078 nats.

#### Reality referee — arm table (pooled, unflagged)

| arm | | mean NLL | \|z\|<=1 | \|z\|<=2 | sd-ratio | mean z | median sd |
|---|---|---|---|---|---|---|---|
| clark | raw | 141.4611 | 0.087 | 0.1304 | 14.9245 | -8.0684 | 0.3698 |
| clark | recal | 3.2854 | 0.7609 | 0.913 | 0.8872 | -0.4796 | 6.2215 |
| clark-diag | raw | 237.0687 | 0.087 | 0.1304 | 19.3049 | -10.4884 | 0.3359 |
| clark-diag | recal | 3.4372 | 0.6739 | 0.8696 | 1.1476 | -0.6235 | 5.6501 |
| fosm | raw | 115.6715 | 0.087 | 0.1522 | 13.8166 | -6.6691 | 0.4342 |
| fosm | recal | 3.2893 | 0.8043 | 0.9565 | 0.8213 | -0.3964 | 7.3035 |
| step-form | raw | 5.7496 | 0.413 | 0.6957 | 2.5179 | -1.2388 | 2.4986 |
| step-form | recal | 4.7177 | 1.0 | 1.0 | 0.1497 | -0.0736 | 42.0331 |
| mc-2 | raw | 1185.6744 | 0.0652 | 0.1304 | 48.4453 | -8.7403 | 0.2666 |
| mc-2 | recal | 6.5289 | 0.6522 | 0.8043 | 2.8798 | -0.5196 | 4.4855 |
| mc-32 | raw | 128.6275 | 0.087 | 0.1304 | 14.2982 | -7.5642 | 0.4301 |
| mc-32 | recal | 3.2996 | 0.7391 | 0.913 | 0.8499 | -0.4496 | 7.2345 |

mean-map (point prediction, no variance): mean error -5.8118, median |error| 3.2855, median |error| per step 0.08424 m

- clark minus fosm (recalibrated, negative = clark better): mean -0.0039, median -0.0498, clark better in 32 of 46, two-sided sign test p = 0.0114
- clark minus clark-diag (recalibrated, negative = clark better): mean -0.1518, median 0.0115, clark better in 15 of 46, two-sided sign test p = 0.0259
- clark minus step-form (recalibrated, negative = clark better): mean -1.4322, median -1.6852, clark better in 42 of 46, two-sided sign test p = 0.0
- clark minus mc-32 (recalibrated, negative = clark better): mean -0.0142, median -0.0174, clark better in 31 of 46, two-sided sign test p = 0.0259
- clark minus mc-2 (recalibrated, negative = clark better): mean -3.2434, median -0.0096, clark better in 24 of 46, two-sided sign test p = 0.883
- arm-NLL spread: 1179.9248 nats raw, 3.2435 nats recalibrated (363.8x compression)

_No pass/fail criterion attaches to any reality-side arm comparison (PREREG_baseprod.md)._

#### Belief referee — E2-iii / E2-v machinery, exercised in-sample

46 unflagged windows, 20000 draws each (seed 20260828; the mc-N arms are subsampled without replacement with seed 20260828). Raw Var, no truth, no recalibration.

| arm | median rel err E | median rel err sd | p95 E | p95 sd | sd-ratio to MC (median) |
|---|---|---|---|---|---|
| clark | 3.420e-07 | 5.132e-02 | 1.398e-06 | 1.707e-01 | 0.9487 |
| clark-diag | 1.284e-06 | 1.926e-01 | 6.029e-06 | 4.542e-01 | 0.8074 |
| fosm | 1.979e-05 | 2.029e-02 | 5.752e-05 | 9.083e-02 | 1.0172 |
| step-form | 1.979e-05 | 5.297e+00 | 5.752e-05 | 8.426e+00 | 6.2968 |
| mc-2 | 8.076e-06 | 5.128e-01 | 3.718e-05 | 1.039e+00 | 0.6776 |
| mc-32 | 4.256e-06 | 1.042e-01 | 1.658e-05 | 2.496e-01 | 0.9798 |
| mean-map | 1.979e-05 | — (no predicted sd) | — | — | — |

**E2-iii preview** (in-sample; the criterion is held-out only):

- (a) MEAN — clark's median relative error 3.420e-07 (<= 5%), below fosm's in 46 of 46 windows, two-sided sign test p = 0.0 (majority + significant)
- (b) SD — clark's sd-ratio to the belief MC, median **0.9487** against the band [0.85, 1.05] (inside); mean 0.9319, p5 0.8293, p95 0.9863. No sd criterion against fosm.

**E2-v preview** — clark below mc-32 on BOTH moments in 32 of 46 windows, exact two-sided sign test p = 0.01135 (E alone 44, sd alone 33). Against mc-2: both moments 41 of 46, p = 0.0.

- clark vs step-form: both moments 46 of 46 (p = 0.0), E alone 46, sd alone 46
- clark vs clark-diag: both moments 40 of 46 (p = 0.0), E alone 41, sd alone 42

##### Reported characterization — the sd-deficit-vs-contest-depth curve

clark's belief-referee sd-ratio (sd_clark / sd_MC) binned by the window's cost-weighted median alpha. No criterion attaches. The historical record is 0.90-0.94; the derived worst case is ~0.89 at alpha ~ 0.

| alpha bin | n | alpha median | clark sd-ratio median | p5 | p95 |
|---|---|---|---|---|---|
| [0.0, 0.2) | 0 | — | — | — | — |
| [0.2, 0.4) | 12 | 0.318 | **0.9119** | 0.8034 | 0.9492 |
| [0.4, 0.6) | 18 | 0.517 | **0.9531** | 0.8602 | 0.9855 |
| [0.6, 0.8) | 13 | 0.6839 | **0.9658** | 0.8845 | 0.9921 |
| [0.8, 1.0) | 2 | 0.8438 | **0.9661** | 0.9559 | 0.9763 |
| [1.0, 1.5) | 1 | 1.1459 | **0.9672** | 0.9672 | 0.9672 |
| [1.5, inf) | 0 | — | — | — | — |

corr(alpha, clark sd-ratio) = 0.4549

##### Reported characterization — belief-referee CVaR error (q = 0.9)

|CVaR_q(arm) − CVaR_q(MC)| per window: the single number a planner consumes, fusing both moments through the decision rule's own tail weight. The arm CVaR is the Gaussian form E + phi(z_q)/(1−q) · sd; the MC CVaR is the mean of the empirical upper tail. mean-map has no sd, so its CVaR is its point prediction — the floor a risk-blind planner sits at. No criterion attaches.

| arm | median | mean | p95 | max |
|---|---|---|---|---|
| clark | **0.0352** | 0.0478 | 0.1241 | 0.2446 |
| clark-diag | **0.1333** | 0.1729 | 0.4705 | 0.6935 |
| fosm | **0.3019** | 0.3832 | 0.8650 | 1.6625 |
| step-form | **3.3103** | 3.9248 | 9.0471 | 10.5883 |
| mc-2 | **0.3240** | 0.4640 | 1.1855 | 2.1234 |
| mc-32 | **0.0820** | 0.1219 | 0.3669 | 0.6072 |
| mean-map | **1.0323** | 1.2650 | 2.5932 | 3.9563 |

Lowest median absolute CVaR error: **clark**

#### Regime

| statistic | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| belief sigma on element cells (m) | 0.0121 | 0.0148 | **0.0178** | 0.0211 | 0.0621 |
| truth relief p95-p5 (m) | 0.0517 | 0.1181 | **0.1589** | 0.2613 | 0.5998 |
| sigma / relief | 0.0339 | 0.0705 | **0.1134** | 0.179 | 0.3127 |
| alpha (cost-weighted median) | 0.2767 | 0.3999 | **0.5528** | 0.6572 | 1.1459 |
| fraction of cost-weighted nodes |alpha| < 1 | 0.4706 | 0.6637 | **0.7776** | 0.8812 | 0.9995 |
| QC element-cell observed fraction | 0.9977 | 1.0 | **1.0** | 1.0 | 1.0 |
| QC median element-cell sigma (m) | 0.0121 | 0.0148 | **0.0178** | 0.0211 | 0.0621 |
| retention | 0.8974 | 1.0 | **1.0** | 1.0 | 1.0 |
| belief mu minus truth, per-window median (m) | -0.1329 | -0.0211 | **0.0062** | 0.0312 | 0.7132 |

### Artifacts

- `/local/kuceral4/baseprod/out/RUN_LOG.md`
- `/local/kuceral4/baseprod/out/FREEZE.json`
- `/local/kuceral4/baseprod/out/freeze_conditions.json`
- `/local/kuceral4/baseprod/out/design_phase.json`
- `/local/kuceral4/baseprod/out/design_score_hindsight.json`
- `/local/kuceral4/baseprod/out/design_score_foresight.json`
- `/local/kuceral4/baseprod/out/design_build_hindsight.json`
- `/local/kuceral4/baseprod/out/design_build_foresight.json`
- `/local/kuceral4/baseprod/out/parity10m_score.json`
- `/local/kuceral4/baseprod/out/parity10m_build.json`
- `/local/kuceral4/baseprod/out/calibration_check.json`
- `/local/kuceral4/baseprod/out/registration/registration.json`
- `/local/kuceral4/baseprod/out/pipeline_commit.json`

### The held-out run

C1-C4 all hold, so this runner proceeded **directly** into the held-out run under the author's restored chained authorization — no GO file, no stop. Follow it in `RUN_LOG.md`; the result lands in `FINAL_REPORT.md`, which reproduces these design tables as an appendix.

The held-out stages refuse to run without `/local/kuceral4/baseprod/out/FREEZE.json`. It exists.

Expected cost once started: 223 GiB streamed across 22 traverses, one archive resident at a time, deleted after extraction; disk cap 100 GB.
