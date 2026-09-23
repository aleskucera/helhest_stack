# Driving on a belief

The first loop in which the probabilistic stack changes what the robot does.

Everything under `elevation_belief` and `terrain_value_field` has until now been computed, tested
and consumed by nothing. The thing that drives is handed a height array and a measured mask; the
per-cell measurement sd, the pose drift and the lattice's two readings of the map reach no
decision, and `z_veto` is never set. A quantity no decision depends on cannot be judged. So:

| | |
|---|---|
| reality | ostrich `odin_sim` -- the measured 256x192 dToF ray table, contact physics, wheel dynamics |
| perception | `elevation_belief`: per-cell Kalman fusion, the visibility carve, pose drift since each cell was last seen |
| routing | `terrain_value_field` via `CostToGo`, **given the sd and the drift**, so a pose is vetoed on its margin in SIGMAS |
| control | MPPI rollouts on a fine window cropped from the same belief |

Localization is ground truth from the simulator. That is deliberate: this tests the planner, not
ICP, and mixing the two would make a failure impossible to attribute.

## Running it

The simulator only exists on dasenka, and the planner only exists on `study/tvf-migration`, which
is not what dasenka's `helhest_stack` is checked out to. So it runs against a worktree, created
once:

```bash
# on dasenka, once
cd /local/kuceral4/projects/helhest_stack
git worktree add /local/kuceral4/projects/hs_tvf study/tvf-migration
cd /local/kuceral4/projects && git clone ssh://git@github.com/aleskucera/elevation_belief.git
git clone ssh://git@github.com/aleskucera/terrain_value_field.git
```

`env.sh` puts that worktree first on `PYTHONPATH`, so the main-branch checkout can never shadow
it. Then:

```bash
cd /local/kuceral4/projects/ostrich-odinsim
HELHEST_MOUNT=/local /local/kuceral4/projects/helhest-singularity/exec.sh bash -c \
  'source <this dir>/env.sh && python3 <this dir>/drive_sim.py'
```

### Every world at once

`sweep.sh` drives all six stress worlds, two at a time (one per GPU), and writes one npz each:

```bash
cd /local/kuceral4/projects/ostrich-odinsim
HELHEST_MOUNT=/local /local/kuceral4/projects/helhest-singularity/exec.sh bash -c \
  'source <this dir>/env.sh && bash <this dir>/sweep.sh'
```

`WORLDS=`, `FRAMES=`, `OUT=` and `EXTRA=` override the defaults; `EXTRA=--coarsen 0` runs the
same sweep with the coarse layer off, which is the A/B for it. Copy the npz back and draw them:

```bash
.venv/bin/python studies/closed_loop/sweep_figure.py --dir studies/closed_loop/out/sweep
```

Each world is drawn at its own extent. The belief is a rolling window and every run ends
somewhere different, so a panel is the map the robot had in front of it when it stopped -- not a
survey of the world. Ground it drove over earlier has scrolled out and is grey.

### The scrub page

`sweep_figure.py` draws the last frame. To scrub every frame of every layer instead, the run has
to have RECORDED them: `--history` defaults to 0 (off, because it reads back to the host) and
`sweep.sh` does not pass it, so a sweep run without it produces npz with no `hist_*` and
`build_scrub.py` refuses them. Ask for it explicitly:

```bash
EXTRA="--veto 1.0 --history 8" bash <this dir>/sweep.sh     # inside the container, as above
.venv/bin/python studies/closed_loop/build_scrub.py --dir <the npz dir> --out out/scrub
```

That writes one PNG per world per layer, a `manifest.json`, and `index.html` (a copy of
`scrub_page.html`, which is the source). Publish the whole `out/scrub` directory as one artifact:
the page fetches `manifest.json` relative to itself.

## The three windows

  belief + coarse   30 m, pooled to 1.0 m cells   -- which way round
  routing           10 m at 0.2 m, settle-based   -- how to get there
  MPPI               9 m                          -- where the rollouts live

Each is a centred crop of the one above, so every offset between them is a constant and the
captured replan graph stays valid. The routing window is small deliberately: a fine window larger
than the sensor's reliable coverage is not planning over terrain but over whatever filled the
unobserved cells, and a 14 m one was measured to produce no usable plan at all, its boundary ring
sitting outside the 6 m horizon. `--coarsen 0` turns the coarse layer off and returns the
single-layer behaviour.

## What the connected lattice was worth

`--veto 1.0`, 700-frame cap, same six worlds, before and after the arc step was made to reach
every heading instead of every second one (4dae1be; the diagnosis is in that commit and in
`tests/planning/test_heading_connectivity.py`):

| world | split ring | connected ring |
|---|---|---|
| gap | 187 | 170 |
| slalom | 376 | 322 |
| pillars | 246 | 224 |
| pocket | **did not reach**, 3.46 m short | **308** |
| ridge | 262 | 199 |
| bumpy | 313 | 257 |

Frames to the goal, so lower is better; 5/6 to 6/6. Every world improved, which is the part
worth keeping: the split ring was not a `pocket` problem that happened to show up there, it was
costing every run 9-24% and only `pocket` failed outright, because only `pocket` needed a
heading the orphaned half of the ring owned.

## The coarse layer, on trial

Once the heading ring was connected, the layer's old justification (fail -> reach on three
worlds) was gone, so it was re-measured against the case for deleting it.

| arm | total frames | verdict |
|---|---|---|
| with it, `--coarsen 5` | 1480 | baseline |
| **without it**, `--coarsen 0` | 1542 (+4.2%) | 6/6 anyway, but +15% `pillars`, +11% `ridge` |
| no pooling, `--coarsen 1` | 1472 (-0.5%) | identical, and +2.0 ms a frame |
| `--coarse-pass 0.1` | 1476 | within noise |
| `--coarse-pass 0.9` | 1473 | within noise |

**Keep the layer, keep the pooling, stop treating `min_pass_fraction` as a risk.** It earns ~4%
overall and 11-15% exactly where "which way round" binds -- `ridge` needs its notch found and
`pocket` is a C whose only opening is outside the routing window for most of the approach. The
pooling is free performance: factor 1 matches factor 5 and costs 2 ms.

And `min_pass_fraction`, carried since it was written as an untuned stand-in for per-edge
feasibility, turns out to move everything except the answer. Swept 9x it takes blocked coarse
cells from 1.5% to 13% and the coarse field by up to 48 m, and the loop moves under 2%,
non-monotonically. A coarse "which way out" survives a wholesale change of opinion about which
blocks are passable, so the stand-in does not need replacing -- which also means the walls
looking inconsistent frame to frame in the viewer is cosmetic, not a defect that reaches the
wheels. Caveat: measured on stress worlds, whose coverage is good.

## Judging the planner without a controller

A run's outcome mixes the planner and the controller, and every attribution made from
reach/no-reach in this study turned out to be confounded. `plan_quality.py` replays the recorded
maps, rebuilds the field, and walks the lattice's own policy from the pose the robot actually
held. No simulator, no controller: "was there a plan from here" is the planner's question alone.

```bash
.venv/bin/python studies/closed_loop/plan_quality.py --a out/ab_A --c out/ab_C
```

**These numbers were measured on the disconnected heading ring and are retained only as
history.** Half of every cell's headings held the "no route" cap, and this table read that as a
planner that could not find routes. See `incident_2026-09-22_lattice-heading-connectivity.md`.

| world | no routing layer | with it | what MPPI did | *re-measured, connected ring* |
|---|---|---|---|---|
| gap | 100% | 96% | reached both | 95% / 95% |
| slalom | 32% | 89% | fail → reach | 90% / 95% |
| pillars | 61% | 94% | fail → reach | 93% / 96% |
| pocket | 67% | 79% | fail → reach | 87% / 95% |
| ridge | 74% | 78% | reached both | 96% / 96% |
| bumpy | 35% | 39% | reached both | 100% / 100% |

The right-hand column is the same question asked of a connected lattice, and PAIRED -- one run's
frames judged under both settings, which removes the trajectory confound this script's own
docstring admits. Plan usability is 87-100% everywhere, and the routing layer's margin collapses
from +57 pp on slalom to +5. Most of what the original table measured was the split ring, not the
absence of a routing layer. The `bumpy` claim -- "no route 6 frames in 10, the controller carries
the run" -- is simply dead: it is 10 frames in 10.

The layer still earns its place, but for less. Driving all six worlds WITHOUT it (`--coarsen 0`)
reaches 6/6 -- so it is not fail->reach on three worlds any more -- at +4.2% frames overall, and
the cost is concentrated rather than spread: pillars 224 -> 258 (+15%) and ridge 199 -> 221
(+11%), the rest within noise. Note that is MORE than the paired column suggests, exactly as its
caveat warns: pairing judges a successful run's poses, so it cannot see that part of the layer's
value is keeping the robot on ground where plans exist. The two measurements answer different
questions; keep both.

## The carrot follower, and what it is not

`--controller carrot` follows the lattice's own policy by pure pursuit. It was built to isolate
the planner and it does not: a simpler controller sits in the same place in the causal chain and
brings its own failures. It measured its own bugs twice -- a turn rate coupled to a collapsing
forward speed, zero commands when the policy walk returned nothing, and then pure pursuit cutting
corners into a ridge. It works on `gap` and is kept for the one thing it alone can do: it
physically cannot enter a vetoed pose, so it can say whether a veto set is survivable. It is not
evidence about plan quality. `plan_quality.py` is.

## The self-filter, and why it is not optional

`odin_sim` casts every ray against the robot's own wheels and chassis, deliberately -- "exactly
as on the real sensor". 22% of returns land inside 0.5 m. Fed to the belief unfiltered they paint
the robot as a ~0.6 m obstacle that it then carries around with it: the 5x5 cells under a
stationary robot read +0.59, +0.64, +0.62 where the ground is at +0.005.

The planner is then walled in by its own body. Measured, with the filter off: 73% of seen cells
blocked at some heading, and the robot drives 3 m, circles, and stops 12 m short of the goal.

`--no-self-filter` reproduces that. The filter itself is `ScanPreprocessor`'s existing `self_box`
gate, run in the robot's base frame where the box is exact, with the survivors rotated to world
by `transform_points` -- the cloud is uploaded once and never comes back to the host.

## Unmeasured cells

Filled from the median measured height, never with 0.0. A zero fill reads as flat ground wherever
the robot has not looked, and where the ground itself sits below zero that is a phantom plateau
which walls the routing window off in a closed ring and makes the goal unreachable. That has been
diagnosed on real bags twice.
