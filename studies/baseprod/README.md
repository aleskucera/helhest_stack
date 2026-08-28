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
| 2 design | register both design traverses; re-check the two pinned calibration constants; build BOTH belief conditions; fit (k, tau) per condition; draw the belief referee | `out/design_*.json` |
| 3 freeze_eval | evaluate C1-C4 | `out/freeze_conditions.json` |
| 4 freeze | `FREEZE.json` if all hold, `STOPPED_AT.md` if any fails; then `DESIGN_PHASE_DONE.md`; **exit** | `out/FREEZE.json` \| `out/STOPPED_AT.md`, `out/DESIGN_PHASE_DONE.md` |
| 5 heldout | stream the 22 held-out traverses (download, extract, build both conditions, delete archive, next) | `out/windows/<cond>/<traverse>/` |
| 6 score | score them once at the frozen (k, tau) | `out/heldout_score_<cond>.json` |
| 7 report | criteria and artifacts | `out/FINAL_REPORT.md` |

**Stages 5-7 are gated.** The author narrowed the autonomous authorization on 2026-08-28: the
runner freezes and stops so the design-phase results can be discussed before the held-out set
is spent. They run only when `--heldout` is passed AND `out/FREEZE.json` exists AND
`out/HELDOUT_GO` exists. There is no second draw.

Every stage appends to `out/RUN_LOG.md` and drops a done-marker in `out/state/`; rerunning the
same script resumes rather than repeats.

## Running it

```sh
export BASEPROD_ROOT=/local/kuceral4/baseprod
tmux new -d -s e2_run '/local/kuceral4/baseprod/studies/baseprod/runner.sh'   # design phase
touch $BASEPROD_ROOT/out/HELDOUT_GO                                           # after the go
tmux new -d -s e2_heldout '/local/kuceral4/baseprod/studies/baseprod/runner.sh --heldout'
```

Data layout is in `paths.py`; every path derives from `BASEPROD_ROOT`. The design/held-out
split is a directory split (`data/design` vs `data/heldout`), and `runner.held_out_untouched`
is the C4 guard that refuses to proceed if held-out material exists before the freeze record.

## Where the numbers live

- `constants.py` — every pinned constant, with provenance. Nothing in the scored path may
  introduce a constant that is not here. It also carries the QC thresholds, the E2 criteria
  bands, and the banned-corrections list; the runner cannot alter any of them.
- `c1_reference_v3.json` — the rehearsal-v3 hindsight numbers C1 must reproduce to 1e-6
  relative, per window and pooled.
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
