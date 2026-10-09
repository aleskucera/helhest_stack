# Step 4 -- accumulator vs belief, simultaneous on dasenka (2026-09-26, 286bed9 + f105713)

Two rounds; in each, both map sources replay the same bag at the same time, one per RTX 3090,
GPUs swapped between rounds (`ab_dasenka.sh`, deleted after 5c7fabb: the node no longer has an
accumulator arm). Charge and z_veto 0 (step 3). The recordings (`out4/`) were not kept.

## Audit -- identical between rounds for both arms

| bag | map | sealed blocks | path samples in sealed | no coarse route at robot | no fine route at own heading |
|---|---|---|---|---|---|
| in_speed_odin0 | accumulator | 138 | 0 of 315 | 0 of 79 | 1 of 79 |
| in_speed_odin0 | belief | 132 | 0 of 315 | 0 of 79 | **0** of 79 |
| out_odin0 | accumulator | 319 | 33 of 2824 | 8-9 of 706 | 3-4 of 706 |
| out_odin0 | belief | **213** | **10** of 2824 | **3** of 706 | **0** of 706 |

The belief map is better on every count. (The single laptop replay before step 3 showed 9 no-route
frames outdoors; it ran with the old charge_per_sigma 0.5, which with the belief's REAL sigma
started charging the rough patch at 85 s. With the charge at 0 those stalls are gone -- a first
measurement of what the sigma path does once it is switched on.)

## Frame time, steady state [ms/frame]

| bag, round | accumulator total | belief total | build_maps acc -> belief | belief_update | plan:coarse acc -> belief |
|---|---|---|---|---|---|
| out r1 | 79.9 | 70.6 | 14.0 -> 5.3 | 1.7 | 4.9 -> 2.5 |
| out r2 | 82.6 | 73.5 | 15.3 -> 6.6 | 1.8 | 3.8 -> 2.4 |
| in_speed r1 | 42.1 | 46.8 | 7.9 -> 4.2 | 1.4 | 3.5 -> 1.9 |
| in_speed r2 | 87.4 | 77.3 | 15.3 -> 5.4 | 1.9 | 4.2 -> 2.9 |

Outdoors the belief path is ~9 ms/frame faster: the rasters' host round trips and the per-frame
HeightMapBuilder passes are gone (build_maps 14-15 -> 5-7 ms), the coarse raster comes free, and
the belief update costs 1.7 ms. in_speed is short and noisier (round 2 slowed both arms alike).

## Not yet checked

Moving objects: waiting for the recorded bag (STEP0.md).
