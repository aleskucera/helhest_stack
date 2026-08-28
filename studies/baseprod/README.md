# studies/baseprod — E2: risk calibration on contested terrain (BASEPROD)

The committed pipeline for `PREREG_baseprod.md`. It assembles what the design phase built as
throwaway exploration (`~/data/baseprod_audit/spike/`, documented in
`theory/BASEPROD_SPIKE.md`) into one study that runs unattended on a remote host, evaluates
the freeze conditions mechanically, and writes the F1 freeze record.

## What runs

`runner.py` is a resumable state machine. Stages, in the order the prereg's "Authorized
autonomous execution" section fixes:

| stage | what it does | artifact |
|---|---|---|
| 1 commit | record the pipeline commit hash F1 names | `out/pipeline_commit.json` |
| 2 design | register both design traverses; re-check the two pinned calibration constants; rebuild the 10 m hindsight path for the C1(a) parity check; build BOTH belief conditions at 4 m; fit (k, tau) per condition; draw the belief referee | `out/design_*.json`, `out/parity10m_*.json` |
| 3 freeze_eval | evaluate C1-C4 | `out/freeze_conditions.json` |
| 4 freeze | `FREEZE.json` if all hold (then straight on to stage 5), `STOPPED_AT.md` if any fails (then exit); either way `DESIGN_PHASE_DONE.md` | `out/FREEZE.json` \| `out/STOPPED_AT.md`, `out/DESIGN_PHASE_DONE.md` |
| 5 heldout | stream the 22 held-out traverses (download, extract, build both conditions, delete archive, next) — entered automatically when C1-C4 pass | `out/windows/<cond>/<traverse>/` |
| 6 score | score them once at the frozen (k, tau) | `out/heldout_score_<cond>.json` |
| 7 report | criteria, artifacts, and the full design tables as an appendix | `out/FINAL_REPORT.md` |

**The chain is unconditional on success.** If C1-C4 all hold, the runner writes FREEZE.json
and goes straight into the held-out run, unattended (the author restored the chained
authorization on the evening of 2026-08-28; `PREREG_baseprod.md`, clark_paper `0506da2`). If
any gate fails it writes STOPPED_AT.md and exits before unlock, and the held-out set stays
untouched. There is no second draw. `--design-only` stops after the freeze record.

Every stage appends to `out/RUN_LOG.md` and drops a done-marker in `out/state/`; rerunning the
same script resumes rather than repeats.

## Running it

```sh
export BASEPROD_ROOT=/local/kuceral4/baseprod
tmux new -d -s e2_run '/local/kuceral4/baseprod/studies/baseprod/runner.sh'
```

Data layout is in `paths.py`; every path derives from `BASEPROD_ROOT`. The design/held-out
split is a directory split (`data/design` vs `data/heldout`), and `runner.held_out_untouched`
is the C4 guard that refuses to proceed if held-out material exists before the freeze record.

## Where the numbers live

- `constants.py` — every pinned constant, with provenance. Nothing in the scored path may
  introduce a constant that is not here. It also carries the QC thresholds, the E2 criteria
  bands, and the banned-corrections list; the runner cannot alter any of them.
- `c1_reference_v3.json` — the rehearsal-v3 hindsight numbers C1(a) must reproduce to 1e-6
  relative, per window and pooled, on the LEGACY 10 m configuration.
- `traverse_manifest.json` — all 24 traverses with their roboshare tokens and listed sizes.
  The two design traverses are named in `paths.DESIGN_TRAVERSES`; the other 22 are held out.

## Module map

| file | item |
|---|---|
| `geometry.py` | rover geometry from the dataset's own TF tree (corrected rear bogie; IMU-to-body rotation) |
| `register.py` | per-traverse vertical registration with the frozen `SELECTION_RULE`, and the void-fill validity mask |
| `depth.py` | D435i deprojection at the frozen CameraInfo intrinsics, poses, the registered-truth sampler, `cam_transform` |
| `belief.py` | `StereoBelief` = spires `ElevationBelief` with exactly one method overridden (the stereo r^2 variance model) |
| `calibration.py` | the two design-phase fits (mount pitch, sigma model), re-run as checks on the pinned values |
| `build.py` | window products, both belief conditions, with the foresight cutoff |
| `score.py` | QC flags, the reality referee, the belief referee (E2-iii MC), the (k, tau) fit, the criteria |
| `runner.py` | the state machine |

The fold, the arms, the element, the settle and the cost are **imported** from
`studies/spires/` (`resample_track`, `build_nodes`, `cell_variance_split`, `cell_covariance`,
`arm_moments`, `mc_cost`, `recalibrate_sd`, `_fit_scalars`, `vehicle.py`), never copied, so
this study cannot drift from the code E1 was run with.

## Windows, arms, and criteria (prereg edits of 2026-08-28, clark_paper `fab1a70`)

**Windows are 4 m**, the depth camera's range cap and the horizon a planner predicts over.
The first design phase ran at 10 m and failed C2 structurally: a 10 m window is never observed
past ~4 m from before its own start, so foresight retention clustered at 0.40-0.55 and 11 of
17 windows fell below the retention floor. `LEGACY_WIN_LEN_M = 10.0` survives only for the
C1(a) code-parity check.

**Seven arms**, all on identical inputs, the recalibration applied identically to every
variance-predicting one:

| arm | what it is |
|---|---|
| `clark` | the recipe — the moment-matched fold with full covariance |
| `clark-diag` | ablation — the same fold with the off-diagonal removed |
| `fosm` | the canonical first-order route, constructed as the STRONGEST derivative baseline (the winner's one-hot weights through the same full covariance). No published system runs it; the paper labels it so. |
| `mean-map` | the point floor; no predicted variance |
| `step-form` | the DEPLOYED PRACTICE. Formula pinned in `constants.STEP_FORM_FORMULA`: `E = cost(mean map)`; `sd = sum_n |c_eff[n]| * sqrt(C[u(n), u(n)])` with `u(n)` the mean-map winner cell of node n — no max, no covariance. |
| `mc-2`, `mc-32` | the exact method at budget — sample statistics of 2 and 32 draws subsampled without replacement from the reference draws with `MC_SUBSAMPLE_SEED`. N = 2 is the equal-compute point (the estimator costs 1.33 rollouts); N = 32 is a generous real-time budget. |

**E2-iii splits by moment.** (a) MEAN: clark's median relative error to the belief MC <= 5%
AND below fosm's in a majority of windows, exact sign test p < 0.05. (b) SD: clark's sd-ratio
to the belief MC in `[0.85, 1.05]`, an absolute band that owns the declared fold deficit,
pooled by the median over unflagged windows (`E2_III_SD_POOL`, fixed in advance). There is no
sd criterion against fosm.

**E2-v** is new: clark's belief-referee error below `mc-32`'s on BOTH moments in a majority of
unflagged windows, exact sign test p < 0.05.

**Two reported characterizations, no criteria**: the sd-deficit-vs-contest-depth curve
(clark's belief-referee sd-ratio binned by cost-weighted alpha, fixed bin edges in
`ALPHA_BIN_EDGES`), and the belief-referee CVaR error `|CVaR_0.9(arm) - CVaR_0.9(MC)|` per
window for every arm.

**C1 changed meaning** when the window length dropped. It is now two parts: (a) the 10 m
hindsight scoring path still reproduces rehearsal v3 exactly — code parity preserved on the
old configuration, so the change is a configuration change and not a code change; and (b) the
4 m pipeline runs both conditions end to end with finite outputs. C2 is re-evaluated on the
4 m windows under the same >= 60% unflagged-under-foresight rule. C3 and C4 are unchanged.

## The two belief conditions

For a window spanning arc length `[a, b)` with `LOOKBACK_M` the shared look-back:

```
hindsight   a - LOOKBACK_M <= s(frame) <= b     full trajectory, the belief at its lifetime best
foresight   a - LOOKBACK_M <= s(frame) <  a     the planning-time belief
```

The look-back is a shared pipeline element, so the foresight frame set is the hindsight set
intersected with the cutoff — a strict subset, never a superset. Applying the cutoff without
the shared look-back would let late windows fuse the entire preceding traverse and make
"foresight" richer than "hindsight" for those windows, which is not the condition the prereg
defines.
