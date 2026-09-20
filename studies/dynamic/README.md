# A walking person, with labels

The visibility carve has always been fitted on static traverses, which can show what it costs
but not what it is for. Its own docstring says so, and asks for a measurement on real dynamics.
This is that measurement.

A bag cannot provide it: you can see that a cell is tall and guess why. Here the mover's track is
an **input**, so every cell it ever occupied is labelled and so is every cell that is only ever
ground.

## Running it

The sim half needs `ostrich-odinsim` and the helhest Apptainer, which live on dasenka:

```bash
ssh dasenka                       # the config cd's you into /local/kuceral4
export HELHEST_MOUNT=/local       # exec.sh binds only $HOME by default
cd /local/kuceral4/projects/ostrich-odinsim
/local/kuceral4/projects/helhest-singularity/exec.sh bash -c \
    'source odin_env.sh && python3 <this dir>/person_sim.py'
```

It writes `person.npz`: clouds, robot poses, the person's position per frame. Copy that back and
analyse it anywhere:

```bash
.venv/bin/python studies/dynamic/person_carve.py
```

## The scenario

A 0.4 x 0.4 x 1.7 m box walks across the robot's view at x = +3 m, from y = -3.5 to +3.5 at
1 m/s, then stands still. The robot is stationary. 260 frames at 14.5 Hz -- the real sensor's
rate, with the real 256x192 ray table.

The person is an existing static shape relocated each frame. The ray-cast kernel recomputes
`X_wb * shape_transform[i]` per scan against every shape with no acceleration structure, so a
moved shape is visible to the sensor immediately and costs nothing to move. The shape chosen is
the one FURTHEST from the robot, so its absence from its original place cannot confound the near
field.

## What it found

The person is unmistakably in the map: ~830 returns land on it per frame, up to +1.36 m.

Score by HEIGHT, not by the `valid` flag. `carve` runs immediately before `measure_scan`, so a
retired cell is re-initialised from the ground behind it in the same frame and reads valid again
-- watching `valid` sees nothing at all, which cost an hour to notice.

| `max_range` | trail cells | still a ghost at the end |
|---|---|---|
| carve off | 140 | 76 (54%) |
| **1.0 m (shipped)** | 140 | **76 (54%)** |
| 3.0 m | 140 | 53 (38%) |
| 6.0 m | 140 | 28 (20%) |
| 10.0 m | 140 | 28 (20%) |

**The shipped default is indistinguishable from no carve at all.** It cannot reach a person at
3 m, so half the trail stands as a wall of ~0.9 m where somebody walked past.

Two things worth knowing beyond that. Cells the person never touched are not damaged -- they
read *better* with the carve on, 86 disturbed down to 53 -- so on this scene reach is not being
bought with erosion. And 20% of the trail survives even a 10 m carve, which is **not** occlusion:
only 3 of those 28 are within 1 m of where the person stopped, and the rest are spread along the
whole walk. That residual is unexplained and is the next thing to look at.
