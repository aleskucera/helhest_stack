# Pre-registration: does the advantage survive a realistic correlation structure?

Written 2026-08-08, BEFORE any run. Criteria fixed here; results go in a separate document.

## Why

Every comparative number in the study is computed under a hand-built belief: sigma from a
frontier/distance/relief heuristic, correlation imposed as a single separable kernel with a
0.15 m length scale, and a Monte-Carlo "truth" that samples from that same model. The sensing
simulator (`studies/sensing/lidar_belief.py`) measured what the correlation actually looks like
when a lidar builds the map, and it disagrees in a way that matters: two-scale, anisotropic, and
non-separable, with a floor that persists past 2.5 m where the assumed kernel is zero by 1 m.

The published convolution optimization assumes exactly the separability that fails. Measured, a
single separable term gets Var[J] wrong by 60% under the real kernel; a PSD-projected rank-5
expansion holds it to 0.57% (`clark_conv.separable_terms`). So the fast path survives, but only
in the rank-M form -- and that has to be stated in the paper whatever this run shows.

## What changes, and what deliberately does not

CHANGED: the correlation kernel, in BOTH the Monte-Carlo truth generator and every estimator
that consumes correlation, replaced by the measured 2-D kernel.

UNCHANGED, so the comparison isolates one variable: the scenes, the plan families, the sigma
FIELD, the cost, the CVaR level, the seed ranges, and the arms.

## Criteria, fixed in advance

Primary regime hybrid/all, n = 100 seeds, same protocol as `clark.py`.

  (i)   clark_cvar keeps the lowest mean regret of all arms.
  (ii)  clark_cvar beats the belief-map-only arm at p < 0.05 (paired sign test).
  (iii) clark_cvar beats the STEP-form arm at p < 0.05.
  (iv)  calibration holds: pooled sd-ratio vs MC truth stays in [0.7, 1.4].

PASS = all four. Anything less is reported as-is; the criteria are not adjusted afterwards.

## What each outcome means, decided now rather than after

  ALL FOUR PASS      The paper's comparison stops resting on an invented noise model. The
                     limitation "synthetic sigma" narrows to "simulated rather than measured
                     sensing", which is a much smaller gap.
  (iv) FAILS ONLY    Clark's calibration is sensitive to the correlation structure it is given.
                     That is a real finding and it goes in the limitations, not the bin.
  (i)-(iii) FAIL     The advantage was an artifact of the assumed kernel. The method still
                     computes what it claims -- it takes any Sigma -- but the comparative claim
                     would have to be restated around the regimes where it does hold.

## Predictions, recorded so they can be wrong

Longer correlation should HELP the analytic estimator relative to STEP-form and the bracket,
because those two ignore correlation entirely while Clark consumes it exactly. FOSM should be
hurt least in the clean regime and most where the map is far outside the validity radius. I
expect (i)-(iii) to pass and (iv) to be the one at risk, because the measured kernel's long
floor raises plan-cost variance in a way the frozen-trajectory approximation does not track.
