# Driving on a belief

The first loop in which the probabilistic stack changes what the robot does.

Everything under `elevation_belief` and `terrain_value_field` has until now been computed, tested
and consumed by nothing. The thing that drives is handed a height array and a measured mask; the
per-cell measurement sd, the pose drift and the lattice's two readings of the map reach no
decision, and `k_sigma` is never set. A quantity no decision depends on cannot be judged. So:

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

## Judging the planner without a controller

A run's outcome mixes the planner and the controller, and every attribution made from
reach/no-reach in this study turned out to be confounded. `plan_quality.py` replays the recorded
maps, rebuilds the field, and walks the lattice's own policy from the pose the robot actually
held. No simulator, no controller: "was there a plan from here" is the planner's question alone.

```bash
.venv/bin/python studies/closed_loop/plan_quality.py --a out/ab_A --c out/ab_C
```

| world | no routing layer | with it | what MPPI did |
|---|---|---|---|
| gap | 100% | 96% | reached both |
| **slalom** | **32%** | **89%** | fail → reach |
| **pillars** | **61%** | **94%** | fail → reach |
| **pocket** | **67%** | **79%** | fail → reach |
| ridge | 74% | 78% | reached both |
| bumpy | 35% | 39% | reached both |

The three worlds the routing layer flipped for MPPI are the three where plan usability jumps; the
three MPPI reached either way are the three where it barely moves. The layer's benefit is a
**planner** effect. And `bumpy` is the reverse case: no route 6 frames in 10 either way, yet MPPI
reaches it comfortably -- there the controller carries the run, which is the same `bumpy` where
it drives through vetoed poses 20% of the time.

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
