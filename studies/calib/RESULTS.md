# Map-uncertainty calibration — results

Step 1 of `PROBABILISTIC_PLANNING_PLAN.md` section 7. Run 2026-09-17 on the newest bags
(`bags/ostrich0..13`, recorded 2026-08-17), against the **current** mapper path: Odin SLAM
pose (`/odin1/odometry`), raw dTOF cloud (`/odin1/cloud_raw`), no ICP.

Reproduce:

```
PYTHONPATH=studies:src .venv/bin/python studies/calib/probe_attitude.py $(cd bags && ls -d ostrich* | sort -V) --lead 1 2 4
PYTHONPATH=studies:src .venv/bin/python studies/calib/probe_sigma.py ostrich0 ostrich1 ostrich2 ostrich3 ostrich4 ostrich6 ostrich8 ostrich10 ostrich11 --stat mean
```

Outputs land in `studies/out/calib/*.json`; every number below is read from those.

---

## 1. Headline: the sigma the z-margin divides by

Probe A settles the robot on the belief built from sweeps up to `lead` seconds earlier and
compares against the SLAM attitude. Pooled over all 14 ostrich bags, `mount_dx = 0`:

| per-cell stat | lead | n | pitch bias | **pitch sd** | corr(pred,meas) | roll bias | **roll sd** |
|---|---|---|---|---|---|---|---|
| `max` (deployed) | 2 s | 70 | +0.67 deg | **3.23 deg** | +0.78 | +2.20 deg | **4.48 deg** |
| `max` | 4 s | 49 | +0.90 | 2.24 | +0.78 | +1.79 | 3.86 |
| `mean` | 2 s | 70 | +0.65 | **2.11** | **+0.90** | +1.53 | **2.84** |
| `mean` | 4 s | 49 | +0.75 | 1.81 | +0.83 | +1.32 | 2.67 |

So `sigma_pitch ~ 2.1 deg` and `sigma_roll ~ 2.8 deg` at best. Over the 0.75 m wheelbase and
0.73 m track that is 2.7 cm and 3.6 cm of differential height error respectively.

**Pitch is tracked, roll is not.** Predicted pitch correlates with measured at +0.78 to +0.90.
Predicted ROLL correlates at +0.00 to -0.19 while its own spread is 2.8-4.5 deg against a
measured roll spread of only 0.2-0.36 deg -- the floor these bags were driven on is level, so
essentially *all* of the predicted roll is map error. The map cannot currently resolve roll at
all on this terrain.

## 2. What the deployed per-cell reduction costs, against a mean-and-sigma belief

**This is a baseline measurement, not a choice between `max` and `mean`.** The plan is the
probabilistic map; `max` here is the CURRENT deployed pipeline and `mean` is the closest thing
this harness can build to what a probabilistic mapper publishes. The comparison says what the
deployed statistic costs, and nothing more.

Switching the per-cell per-sweep reduction from `max` (the deployed `primary: "max"`,
`_pipeline_common.py:187`) to `mean` cuts the pitch residual 3.23 -> 2.11 deg and the roll
residual 4.48 -> 2.84 deg, and raises the pitch correlation 0.78 -> 0.90.

### The max lives at two levels; only one of them goes away

Worth separating, because "stop using max" would be wrong:

1. **Within a cell** — reducing the points that fall in one cell to a height. Here `max` is a
   biased, high-variance estimator, and being an extremum it has no standard deviation to
   publish at all. **This is the one the probabilistic map replaces** with a mean and a sigma.
2. **Across the cells of a wheel footprint** — the envelope / dilation. **This one stays.** The
   physics is real: a wheel does rest on the highest point beneath it. It is exactly where the
   Clark fold belongs (plan section 3.3), and the fold returns `E[max]` WITH its Jensen
   inflation plus a correct `sigma_support` — which a morphological max cannot.

So the probabilistic map does not remove the max. It moves it from a per-cell point reduction,
where it is merely a bad estimator, to a footprint-level fold over Gaussians, where it is the
physics computed properly. The conservatism then comes from `k * sigma`, not from taking an
extremum and hoping.

### Neither arm is the probabilistic pipeline yet

Both arms feed `ForwardSimulator.set_terrain`, whose envelope is a deterministic morphological
max **of means**. That is not `E[max of Gaussians]`: it carries no Jensen inflation and yields
no `sigma_support`. What section 1 measures is therefore a *mean-map plus deterministic
dilation* pipeline, which is the right baseline but is not the target design.

Read the numbers accordingly:

- `sigma_pitch ~ 2.1 deg` / `sigma_roll ~ 2.8 deg` are the **measured residual** — how wrong
  the map's attitude prediction actually is. This is ground truth, and it is what the z-margin
  must be calibrated against.
- The **predicted** sigma — what the fold plus `J^-1` would claim — does not exist yet, because
  the fold is not built.

Calibration in the sense of plan section 6.1 (and of the Clark paper's 20-21%-against-68%
coverage finding) means *predicted against measured*. This run delivers the measured half only.
The harness is not finished until the fold exists and its claimed sigma can be checked against
these residuals. That check is the natural step 5 of the plan's ordering.

## 3. sigma(range) -- the model the mapping node has to implement

Probe B: each sweep is one independent estimate of a cell's surface, so the spread ACROSS
sweeps is the per-cell sigma. Pooled over 9 bags, `mean` stat, cells with >= 4 observations:

| range | measured sigma | cells |
|---|---|---|
| 0.8 m | 2.1 cm | 1465 |
| 1.2 m | 4.7 cm | 4631 |
| 1.8 m | 7.3 cm | 6741 |
| 2.2 m | 8.4 cm | 9848 |
| 2.8 m | 9.7 cm | 5627 |
| 3.2 m | 11.2 cm | 3041 |
| 3.8 m | 11.8 cm | 2412 |
| 4.8 m | 12.2 cm | 1649 |
| 5.8 m | 18.3 cm | 492 |

Weighted linear fit `sigma(r) = 2.7 cm + 2.3 cm/m * r`, R^2 = 0.685. The fit overestimates
below ~1.5 m (4.4 cm predicted against 2.1 cm measured), so **use the table, not the line**,
for the near field the robot actually drives on.

This is the single biggest lever available. The robot drove on cells whose mean observation
range was ~3 m, where sigma is ~10 cm; the same cells at 0.8 m carry 2 cm. Re-observing ground
from close range before committing to it is worth a factor of five.

## 4. The error is NOT drift-dominated -- this corrects the plan

Pooled spatial correlation of one sweep's residual against the consensus:

| lag | 0.08 m | 0.16 m | 0.24 m | 0.40 m | 0.56 m | >= 0.8 m |
|---|---|---|---|---|---|---|
| corr | +0.30 | +0.18 | +0.14 | +0.08 | +0.06 | **+0.04** |

The variance is dominated by **independent per-cell noise**, with only a ~4% spatially
correlated floor. Three consequences, all of which revise the plan:

- **Section 2(b) is wrong for this sensor.** It argued that pose drift is common-mode over a
  footprint, so absolute map error largely cancels in attitude and clearance. Measured, the
  common-mode component is ~4% of the variance. The cancellation does not happen -- which is
  exactly why the predicted roll in section 1 is pure noise rather than a tracked signal.
- **Section 3.2's warning about the clearance cross term is inverted.** It argued that dropping
  `Cov(e, h_b)` forces the independent-cells answer and would badly overestimate
  `sigma_clear`. At the ~0.5 m wheel-to-belly separation the measured correlation is 0.04, so
  the independent answer is very nearly right and the cross term is a small correction.
  Keep it for correctness, but it is not load-bearing here.
- **Section 5's "one scalar correlation length" still stands**, and its value is
  `L ~ 0.1 m` (correlation falls below 1/e within one 0.08 m cell), plus a long-range floor of
  0.04 that a single exponential does not capture. A mapper publishing one `L` should publish
  the floor too, or the model will claim independence it does not have at 1-2 m.

Caveat: these are indoor floors observed by a dTOF at 0.08 m cells. Outdoor vegetation, and
the longer ICP-based pose path (`icp_enable: true`), may well be more correlated. Re-run before
generalising.

## 5. Self-occlusion is not a blocker (plan section 6.5)

Measured on ostrich4, 30 sweeps, 1.43 M points, in `odin1_base_link`:

- horizontal FOV ~137 deg, vertical ~88 deg
- ground returns begin at **0.36 m** from the sensor; median ground return at 0.89 m
- straight ahead (+-20 deg azimuth) the nearest ground return is 0.36 m

So the ground immediately in front IS observed and the near blind zone is ~0.36 m. The concern
that the decision-relevant cells are structurally unobservable from the driving pose does not
hold for this sensor. **Section 6.5 closes.**

What does bite is coverage of the wheel contact disks, which is a different thing:

| lead | front-L >= 2 obs | front-R | rear |
|---|---|---|---|
| 0 s | 63.6% | 65.3% | **48.7%** |
| 2 s | 51.9% | 53.4% | 55.8% |
| 4 s | 50.8% | 53.1% | 60.9% |

The rear wheel is worst at short lead, as a forward-facing sensor implies. Requiring all three
disks observed rejects ~35% of frames. That is a real constraint on how far ahead the planner
can be trusted, and it argues for the plan's frontier-seeding (section 4.1) preferring
headings that fill in the ground the robot is about to put its wheels on.

## 6. What this does NOT establish

- **A mount-offset fit failed to identify anything.** Sweeping `mount_dx` from 0 to 1.2 m the
  residual rises monotonically (pitch sd 2.65 -> 7.5 deg on 5 bags), so `mount_dx = 0` is best
  and `odin1_base_link` is at or near the engine's body origin. That is a *bound*, not a
  measurement -- the sweep cannot distinguish a small offset from map noise.
- **The roll bias (+1.3 to +2.3 deg) is unexplained.** Regressing predicted roll on heading
  gives a body-fixed constant of -2.5 deg AND a world-frame tilt amplitude of 11 deg, with
  R^2 = 0.28 at n = 70. Both components are present and neither is identified. A sensor mount
  roll error is the natural candidate and it is NOT ruled out; it needs either more data or a
  deliberate calibration manoeuvre (drive the same flat patch at several headings).
- **Absolute height error.** Only attitude is compared, since the engine's body origin and
  `odin1_base_link` differ by an unknown z. `sigma_clear` therefore has no end-to-end check
  yet; section 3.2's decomposition is untested against a realised belly clearance.
- **Whether the tilt clamp masks contested contacts** (plan section 6.2). Not addressed.

## 7. What to do with this

1. The mapping node should publish `sigma` per the section 3 table, keyed on **observation
   range**, and carry `L ~ 0.1 m` plus a 0.04 long-range floor.
2. It should publish a **mean**, not a running max -- section 2 measures what the max costs.
3. `sigma_pitch ~ 2.1 deg` / `sigma_roll ~ 2.8 deg` are the denominators for the z-margin
   field. At `k = 2` that is a 4-6 deg attitude margin, against a `max_roll` of 15 deg: usable,
   but it means over a third of the envelope is spent on map uncertainty at current quality.
4. Re-observing ground from close range is worth 5x in sigma. That is an exploration objective
   the plan's section 4.3 gap can express directly.

---

# Carve gate re-fit (2026-09-18)

`elevation_belief`'s visibility-carve gates were ported from this project's spinning-LIDAR
tuning, where `max_range = 2.5 m` was fitted against 0.35 deg beams binned at 1.4 deg. The Odin
dToF is a 137 x 88 deg sensor returning ground from 0.36 m, so the gates were structurally right
and the numbers were not evidence. Re-fitted on `ostrich0/2/4/8/10` with
`studies/calib/fit_carve.py`; numbers in `studies/out/calib/fit_carve.json`.

## 0. A precondition the harness got wrong first

Fed raw points, the belief put the **ceiling** in the map: on ostrich4, 82% of cells sat 2 m or
more above the floor and the median cell was at **3.57 m**. Keep-the-highest fusion takes the
highest return in a column, and indoors the upward rays reach the roof. Every erosion figure
computed against that denominator is a statement about ceiling, not terrain.

The deployed pipeline crops height before gridding (`z_max`, `_pipeline_common.py:186`); the
harness did not. With a crop of +1.0 / -1.5 m about the sensor the baseline map drops from
~9700 to ~6200 cells and the fusion residual RMS from 0.139 to 0.065 m. **A height crop is a
precondition of this filter indoors, not an optimisation.**

## 1. Cost: what the carve removes

Cells a carved run loses relative to a no-carve run, pooled over four bags, binned by the range
the cell was observed at:

| `max_range` | lost overall | <2 m | 2-5 m | 5-8 m | >8 m | resid RMS | vs baseline |
|---|---|---|---|---|---|---|---|
| 1.0 m | **0.73%** | 3.30% | 0.22% | 0.00% | 0.00% | 0.0714 | +0.0010 |
| 2.5 m (ported) | 2.91% | 7.73% | 2.96% | 0.03% | 0.00% | 0.0844 | +0.0139 |
| 5.0 m | 4.54% | 7.73% | 4.28% | 6.15% | 0.23% | 0.0980 | +0.0275 |
| 10 m | 7.99% | 7.73% | 4.28% | 12.63% | 14.36% | 0.1058 | +0.0354 |
| 15 m | 11.82% | 7.73% | 4.28% | 12.63% | **36.59%** | 0.1049 | +0.0345 |

The far-field blow-up reproduces the original finding's shape on a completely different sensor:
past 8 m the carve removes a third of the map at `max_range = 15`. `margin` and `persist` both
trade monotonically -- larger is safer -- and neither buys anything back.

## 2. Benefit: not demonstrated on this record

The belief accumulates the fusion residual per cell, so "how far did the map stand from each
incoming measurement" is free. Carving stale geometry should shrink it. **It does not: the
residual is worse at every gate setting tested**, by +1.4% at the mildest and +23% at the ported
default.

The direct test is selectivity -- does the carve remove cells that were disagreeing with
measurements, or cells that were agreeing? On ostrich4 and ostrich8:

| | removed |
|---|---|
| worst-agreeing decile of cells | 2.2% |
| best-agreeing half of cells | 1.4% |

A ratio of **1.6x**, which is barely better than indiscriminate. A carve that is finding stale
geometry would show a large ratio. This one is removing a near-random ~3% of the map, and the
residual rises because carving a converged cell restarts it: the next measurement re-initialises
it and the ones after that are scored against a fresh, worse prior.

## 3. Recommended gates, and what they rest on

Across five bags:

| gates (`max_range`/`margin`/`persist`) | lost | vs baseline residual |
|---|---|---|
| 2.5 / 0.05 / 8 (ported) | 2.67% | +0.0160 |
| **1.0 / 0.10 / 8** | **0.42%** | **+0.0010** |
| 1.0 / 0.20 / 16 | 0.25% | +0.0007 |

**1.0 / 0.10 / 8** is the new default: 6x less erosion and 16x less residual penalty than the
ported values.

## 4. What this does NOT establish

**These bags cannot demonstrate the carve's benefit, only its cost.** They are 19 s of largely
static indoor scene; the tuning this replaces was done on bags containing a walking person,
which is the failure the carve exists for. The measured selectivity of 1.6x is evidence that
the carve is not finding stale geometry *here*, not evidence that it never does.

So the gates above are fitted to minimise damage on a record where there is nothing to gain.
Before loosening them, measure the benefit on a bag with real dynamics -- the selectivity ratio
in section 2 is the metric to use, and it should be large, not 1.6.

A second caveat: with `margin = 0.02` the erosion is highest and the residual worst, which says
the carve evidence at small margins is driven by measurement noise rather than moved geometry.
At a per-cell sigma of 2-18 cm (section 3 above), a margin below ~0.10 m cannot distinguish
"this surface moved" from "this surface reads high today".


---

## 4. Odin's own pose-drift rate `q_z`

`elevation_belief` grows every cell's height variance by `q_z * dt` between measurements, and
that growth is what makes a SEAM between old and new data expensive to plan over. The shipped
`DriftRates` are the Oxford Spires **dead-reckoning** calibration; Odin's pose is on-device SLAM.

Reproduce:

```
PYTHONPATH=studies:src .venv/bin/python studies/calib/fit_drift.py out_odin0
```

A cell measured at `t1` and re-measured at `t2` differs by measurement noise plus whatever drift
accrued in between, so `E[(h2-h1)^2] = 2*var_meas + q_z*dt`. Fitted over 1.41 M re-measurement
pairs within 6 m on `out_odin0` (141 m driven), restricted to cells with >= 4 returns and under
2 cm of within-cell roughness so a shifted sample pattern is not read as drift:

| | fitted | shipped default | ratio |
|---|---|---|---|
| `q_z` | **7.5e-05** m²/s | 7.43e-03 | **0.010x** |
| `var_meas` | 3.1e-04 m² (sd **1.75 cm**) | — | — |

Bins are weighted by how many pairs stand behind them; weighting them equally instead gives
`q_z = 4.9e-05`, so read the estimate as **5–8e-05**. The fitted `var_meas` is a free sanity
check and lands where section 1's sigma work put it.

**The point estimate is soft** — R² 0.31, the binned variances are not monotone in `dt`, and
cells re-measured after a long gap are seen from a different viewpoint, which inflates the long
bins. **The conclusion is not.** No observed bin comes within an order of magnitude of the
inherited rate:

| dt | observed sd | default predicts | ratio |
|---|---|---|---|
| 0.4 s | 0.018 m | 0.061 m | 3.3x |
| 7.9 s | 0.034 m | 0.243 m | 7.1x |
| 21 s | 0.065 m | 0.395 m | 6.1x |
| 96 s | 0.096 m | 0.846 m | 8.8x |
| 112 s | 0.062 m | 0.910 m | 14.7x |

### What it changes

The footprint age spread (1.11 s median within 5 m, p90 88 s at a revisit, 76 s in the coarse
layer) is a property of the **sensor's sparsity** and is unaffected. What the fitted rate changes
is what that spread COSTS:

| age spread | at 7.4e-03 | at 7.5e-05 |
|---|---|---|
| 1.11 s (median, fine window) | 0.095 m, 3.8x | 0.026 m, **1.07x** |
| 87.7 s (a revisit) | 0.808 m, 32.6x | 0.085 m, **3.43x** |
| 76.5 s (coarse ground) | 0.754 m, 30.4x | 0.080 m, **3.22x** |

So the drift-spread term is a **seam correction**, not a general one. In a continuously
re-measured fine window it moves the margin by a few per cent; where old data meets new it is
still worth a factor of three, in the optimistic direction.
