# Step 0 -- baselines (2026-09-26, node at 429d427, laptop A500)

## Bag replays, current node (accumulator map), twice each

| bag | goals reached | blocks seen | sealed | bridged | path samples in sealed / bridged / unseen | frames no coarse route / no fine route at own heading |
|---|---|---|---|---|---|---|
| in_speed_odin0 run 1 | 0/1 | 276 | 138 | 12 | 0 / 0 / 19 of 316 | 0 / 2 of 79 |
| in_speed_odin0 run 2 | 0/1 | 276 | 138 | 12 | 0 / 0 / 18 of 315 | 0 / 1 of 79 |
| out_odin0 run 1 | 0/21 | 2382 | 319 | 4 | 33 / 0 / 0 of 2824 | 9 / 4 of 706 |
| out_odin0 run 2 | 0/21 | 2382 | 319 | 4 | 33 / 0 / 0 of 2769 | 9 / 4 of 693 |

The audit repeats. Steady-state stage times do not: outdoor `total` 94.7 vs 146.3 ms between two
runs of the same tree (every stage scaled together, i.e. the laptop, not the code). Step 4 must
interleave the two map sources in one session and compare within it.

Recordings: `out0/*.npz` (gitignored; not kept after 2026-09-28).

## A bag with moving objects: none found

`find_dynamic.py` counts 0.2 m columns that hold something tall in only some of the frames that
observe them. On every Odin bag the counts are dominated by the edges of static structure seen
from a changing viewpoint (maps in `out0/dynamic_*.png`), so the count alone ranks nothing.
`compact` looked like the exception -- the robot stands still -- but its tall-point count per frame
flips between ~70 and ~7300 from 62 to 91 s (the robot turning in place) and is constant to 1% from
91 s to the end: nothing moves in it.

So the carving decision (PLAN.md decision 2) cannot yet be checked on real data.
