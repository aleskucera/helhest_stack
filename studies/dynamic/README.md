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
cd /local/kuceral4
HELHEST_MOUNT=/local projects/helhest-singularity/exec.sh bash -c \
    'source projects/ostrich-odinsim/odin_env.sh && python3 <this dir>/person_sim.py'
```

`HELHEST_MOUNT=/local` is not optional -- `exec.sh` binds only `$HOME`, and the repos are under
`/local/kuceral4`. It writes `person_clear.npz`: clouds, robot poses, the person's position per
frame. Copy that to `studies/dynamic/out/` and analyse it anywhere:

```bash
.venv/bin/python studies/dynamic/person_carve.py     # the table below
.venv/bin/python studies/dynamic/person_figure.py    # the picture
```

## The scenario

A 0.9 x 0.9 x 1.0 m box, standing from 0.35 m to 1.35 m, walks across the robot's view at
x = +3 m, from y = -3.5 to +3.5 at 1 m/s, then stands still. The robot is stationary. 260 frames
at 14.5 Hz -- the real sensor's rate, with the real 256x192 ray table.

The person is an existing static shape relocated each frame. The ray-cast kernel recomputes
`X_wb * shape_transform[i]` per scan against every shape with no acceleration structure, so a
moved shape is visible to the sensor immediately and costs nothing to move. The world's other
boxes and its perimeter walls are **sunk 50 m** rather than switched off, so the person walks over
clear ground: a ghost next to a wall is not a ghost you can measure.

### Two ways of picking the mover that both give a scene worth nothing

Both of these produced tables that looked plausible and were meaningless. They are written down
because neither announced itself -- the script ran, the numbers came out, and only a sanity check
on the map heights caught them.

**Building with `solid_obstacles=False, bounding_walls=False` takes the ground away.** It looks
like the way to clear the scene. But with no obstacles and no walls the only static shape left is
the TERRAIN MESH, so "relocate the furthest static shape" walks the ground itself out from under
the robot. The robot falls: `robot.z = -344 m` by the end, the sensor sees nothing but its own
hull (6 k returns per frame instead of 25 k), and the map is a 536 m tall smear. Build the world
whole and sink what you do not want. `ground_only` stamps obstacle cells back to 0.0, so sinking
a box leaves flat ground and not a pit.

**"The static shape furthest from the robot" is a perimeter wall.** In `pillars` that is shape 21,
half-extent 0.10 x 5.24 m -- a 10.5 m wall walked across the view and scored as a person. The
selection now also demands a person-sized footprint (`--person-max-half`, default 0.6 m).

## What it found

The person is unmistakably in the map: it reads up to +1.35 m above the ground it stands on.

Score by HEIGHT, not by the `valid` flag. `carve` runs immediately before `measure_scan`, so a
retired cell is re-initialised from the ground behind it in the same frame and reads valid again
-- watching `valid` sees nothing at all, which cost an hour to notice.

| `max_range` | trail cells | still a ghost at the end | ground cells disturbed |
|---|---|---|---|
| carve off | 210 | 191 (91%) | 12 of 791 |
| **1.0 m (shipped)** | 210 | **191 (91%)** | 4 of 789 |
| 3.0 m | 210 | 131 (62%) | 4 of 789 |
| **6.0 m** | 210 | **7 (3%)** | 4 of 789 |
| 10.0 m | 210 | 7 (3%) | 4 of 789 |

**The shipped default is indistinguishable from no carve at all.** It cannot reach a person at
3 m, so 91% of the trail stands as a wall of about a metre where somebody walked past. At 6 m the
trail is gone, and a cell clears one frame (0.07 s) after the person steps off it.

**On this scene the reach is free.** 789 of the 791 ground cells survive a 10 m carve exactly as
they survive a 1 m one. That is worth stating carefully: it is flat ground, a stationary robot and
no pose error, which is the easiest case the carve will ever be given. The erosion this reach can
cause is real and was measured elsewhere -- `between_beam_gap_carve` on bags, and 0.53% of the map
on `out_odin0` -- but it is not a property of the reach alone.

**The 3% residual is occlusion, not a bug.** Of the 7 cells that survive a 10 m carve, 4 are
within 1 m of where the person came to rest: the person is standing in front of them, so no ray
reaches them. The cleared cells sit at a median 4.0 m from that spot, the survivors at 0.60 m.

## The two mechanisms that are shipped OFF

`carve(margin_sigmas=...)` scales the clearance by the cell's own height sd (ETH's
`visibilityCleanup` uses 3σ) and `carve(ref_range=...)` weights a ray's evidence by
`min(1, ref_range / ray_length)` (`elevation_mapping_cupy`'s `cleanup_step / (ray_length /
max_ray_length)`). Both exist to buy reach without eroding the far field. Measured here, at 10 m
reach:

| | ghost left | ground kept |
|---|---|---|
| fixed 0.10 m margin | 7 (3%) | 789 |
| `margin_sigmas=3`, `margin=0.05` | 7 (3%) | 789 |
| `ref_range=3` | 7 (3%) | 789 |

They change nothing, because on this scene there is nothing for them to fix. They stay off. This
is not evidence against them -- it is a scene with no far-field erosion to protect, so it cannot
be evidence either way. The measurement that would settle it is a moving person on **rolling**
terrain with a **driving** robot, which is the next scenario to build.
