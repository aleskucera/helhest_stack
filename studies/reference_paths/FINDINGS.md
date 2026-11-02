# Reference paths: what goes wrong and what to change (study, 2026-10-30/31)

The reference stage (`helhest/control/reference_path.py`) builds every frame the curve MPPI tracks:
greedy descent of V's heading-minimum -> taut string (visibility graph within `v_tol` of V) ->
curvature profile optimised by Adam (100 steps warm, 400 cold) -> rejection test.

## Harness

| piece | what |
|---|---|
| `extract.py`, `score.py`, `events.py` | the curves the robot published (`/reference_path_marker`) out of bags, scored without a map |
| node `plan_ref_record`, `drive_sim --ref-record [--ref-record-full]` | the stage's inputs and outputs every frame |
| `replay_ref.py` | reruns the stage offline on a recording, bit-exact (0/127, 0/1500, 0/246 frames differ), with `--set field=value` variants; `--recorded` scores without rerunning |
| `viz.py` | contact sheets: V, forbidden cells, descent, taut string, curve, variant overlay |
| `sim_worlds.py` | 12 stress worlds added to drive_sim's |
| `campaign_table.py` | per world x configuration: reached, frames, wall touches, curve metrics |

## Problems found

**P1 -- the curve keeps a way round the taut string has left.** `_mode_kernel` tests whether the
taut string left last frame's curve only when the string's END moved more than `rebase_m`. With
the goal in the window the end never moves, so a curve warm-started on the other side of an
obstacle is polished there indefinitely: forest (curve loops back round a tree it already passed,
taut string straight on, ~4 s), facing_wall frame 100 (string north, curve south).

**P2 -- the way round flips while an obstacle is being discovered.** facing_wall: south, north,
south as more of the wall comes into view; the robot drove straight at it, ended against its face
and never left (baseline fails 2 of 3 runs). cornerL24: the first frames route out of the corridor
over the unseen end of its wall, the robot turns into the wall, then the corridor route takes
over and asks a 100 deg pivot the robot cannot make beside the wall (stuck 90 s).

**P2b -- the descent ignores the robot's heading.** cornerL24 (fails in every C-campaign run):
the robot starts facing east in a corridor whose north wall ends beside it. V's heading-minimum
descent ignores the turn-around and routes back west round the wall end, over unseen ground the
map later closes; the robot swings its nose into the wall end, its pose becomes vetoed (V capped
at its cell), the descent finds nothing (no curve, MPPI on V_escape) and it stays wedged 90 s. The
lattice-policy descent (`policy_descent`) prices the turn-around from the robot's own heading
bin (pivot_cost 0.3/bin -> ~3.6 m for a half turn), keeps going east, and reached in 250 frames.

**P3 -- clearance asked where the ends cannot have it.** Every knot is asked `offset` (clearance
+ half-width, ~1.2 m) of clearance. A goal 0.6 m from a wall cannot be reached at that, and the
optimiser bent the curve into a U-hook that arrives along the wall (goal_at_wall, 39% of frames
a loop, curve up to 3x the string's length). A robot already beside a wall had every curve
rejected (facing_wall: 30 s without a curve, MPPI on bare V).

**P4 -- a loop where a pivot belongs.** The start pivot is priced `w_pivot` = 2 per rad^2: a
half turn costs ~20 m-equivalent, a 1 m-radius loop ~3-6 m. So a robot facing away from its route
gets a loop, even beside a wall where a loop cannot fit and a skid-steer would spin.

**P5 -- sag after a turn.** uturn: after the start turn the curve stays 0.6-1 m off the straight
line to the goal for ~3 s of frames on open ground. `w_pivot` 0.5 removes most of it.

**P6 -- pigtails.** false_door: a full 360 deg circle beside the robot while the string runs
straight; a curvature profile cannot unwind a loop by small steps, and the warm start keeps it.

**P7 -- the curve leaves the window.** false_door: bulges ~1 m past the string beyond the routing
window, where the clearance map has no information.

**P8 -- facing away, the warm curve never re-pivots.** A cold start pivots outright when the way
on is > `pivot_init_rad` (60 deg) off, but warm frames never ask again: with the robot facing away
from the goal the curve swung metres out and back instead of turning (patek1 t 41-45 s, 163 s).

**Arrival frames are not failures.** The 3% of patek1 frames without a curve are all within 0.5 m
of a goal (the curve is < 3 points; MPPI brakes onto it). Metric `empty_far` excludes them.

**Sim vs robot (2).** drive_sim hands over to the dock controller at 3.0 m (its `--dock` default),
the robot at 1.5 m (`plan_dock_radius` node default): the robot builds references over the last
1.5-3 m, the sim never does -- which is exactly where patek1's end-of-run loop happened. drive_sim routes on 0.08 m cells (200x200 over 16 m); the robot routes on
0.24 m (`plan_lat_coarsen` 3, 66x66). The stage's descent, staircase and distance map are 3x
finer in the sim.

**Warm steps.** patek1 with the fixes, 3090: 100 steps 1.65 ms, 50 steps 1.11 ms with nearly the
same curves (wiggle frames 8.5 -> 11.3%, strays 2.5 -> 2.2%), 30 steps 0.85 ms and worse (strays
4%). 50 is available if the frame budget needs it.

**P5's cause: the length term has no shape gradient.** `w_len` prices the KNOT COUNT to the
arrival (an integer) times the spacing, so it moves only the spacing; once a curve reaches its
target only `w_k` / `w_dk` (0.02, 0.2) shape it and a gentle bulge is nearly free. gap (C2): the
curve bulged 1.3 m off a straight string, growing frame by frame while the robot crawled through
the gap at 4 cm/s, held there by the clearance governor. `w_taut` (a pull to the nearest point of
the taut string, up to the arrival) supplies the missing gradient: uturn mean deviation from the
string 0.15 -> 0.03 m at w_taut 3, detour p95 1.05 -> 1.01, +0.06 ms.

**Speed.** The stage is ~3.1 ms a frame on a 3090 (100 Adam steps); ~1.7 ms of that is the
single-thread descent / visibility / taut kernels, not the optimiser. Stopping Adam early
(`stop_tol`) saves ~1 ms but degrades the curve (single_on_line: detour 1.02 -> 1.09) -- rejected.

## On the robot's own data

Node replay of karlak_u_patek1 (robot config, the bag's own poses, 0.24 m routing) with
`plan_ref_record`; the offline rerun matches 2816 of 2898 frames. Curve metrics: 4.3% loops,
3% empty, 12.7% straying > 1 m from the taut string, detour p95 2.4x -- the same profile the robot
published (`score.py` on the bag: 3.5% loops, 3.6% empty).

The worst event is the end of the run (t 194-200 s, bag 171-200 s): 1-2.5 m from the goal, taut
string a short straight to it, the curve a 5 m loop round a cluster of trees and back -- the robot
FOLLOWED it (bag: up to 1.4 m off at 1.2 m/s, curve turning 200-320 deg). This is P1: the goal is
in the window, its end never moves, so the curve's way round is never revisited. At t 45.6 s the
curve runs ~5 m past the goal into unmapped ground and loops back (P6).

## Speed (bit-exact)

V's heading-minimum computed once a frame in parallel (the single-thread descent took 24 loads
per neighbour) and the clearance-room test run one thread per sample: 2.91 -> 2.05 ms a frame on
the A500, 0 of 127 replayed curves differ. The optimiser (1.87 ms, 100 Adam steps) is now nearly
all of it.

## Recommended set (opt-in, all defaults unchanged)

    -p plan_ref_cold_on_divergence:=1.0 -p plan_ref_clear_taper:=0.5 -p plan_ref_cold_on_loop:=1.0 \
    -p plan_ref_w_pivot:=0.5 -p plan_ref_w_taut:=3.0

Offline on the robot's own frames (node replay, stage rerun):

| metric | patek1 deployed | fixed | patek0 deployed | fixed |
|---|---|---|---|---|
| loop | 4.3% | 0.04% | 2.0% | 0.09% |
| > 1 m off the taut string | 12.7% | 0.14% | 14.8% | 0.43% |
| detour p95 | 2.36 | 1.06 | 2.15 | 1.06 |
| deviation p95 | 2.44 m | 0.11 m | 1.72 m | 0.13 m |
| no curve, goal > 1.5 m | 0.17% | 0.03% | 0.29% | 0.08% |
| clearance p5 | 0.47 m | 0.44 m | 0.53 m | 0.37 m |

patek0's lower clearance p5 is the taper letting the curve approach goals beside obstacles
(the tightest frames are 1.6-2.4 m from such a goal, where the deployed stage had no curve or
a hook).

`w_taut` pulls only where the point already has its clearance (the taut string hugs what it
bends round; an unrestricted pull cost clearance near trunks).

Rejected: `sticky_tol_m` (more loops on real data), `cold_on_detour` 1.3 (repeated colds in
dense trees: sim forest stray 0.6 -> 8.9%; at 2.0 a small real gain, kept optional),
always-cold, `stop_tol`, `cold_on_repivot` (negligible), the taper's 0.1 m margin (facing_wall's
rejections were a pivot's tail swing from a pose already against the wall).

## Live in the node (patek1 replayed with the parameters set)

| | deployed | fix set | fix set + policy descent |
|---|---|---|---|
| loops | 4.34% | 0.04% | 0.07% |
| > 1 m off string | 12.74% | 0.14% | 0.21% |
| no curve, goal > 1.5 m | 0.17% | 0.03% | 0.83% |
| clearance p5 | 0.47 m | 0.44 m | 0.49 m |

The live fix-set run matches the offline rerun exactly. The policy descent's empty frames are
one 2 s stretch (t 182.8-184.8 s) where its route flipped north/south of a tree cluster and the
southern curve's start pivot swung the tail into it: it needs route hysteresis first.

Two more real runs, offline (deployed -> fix set): delay_neural0 stray 11.5 -> 0.06%, detour
p95 1.68 -> 1.03, clearance p5 0.68 -> 0.82 m; new3 stray 10.5 -> 2.8%, deviation p95 0.84 ->
0.09 m, frame jumps 2.8 -> 4.1% (the one metric that got worse).

## Closed loop, robot-faithful (`--route-coarsen 3 --fine 12 --dock 1.5`), 2 runs each

| | R0 baseline | R6 fixes | R7 fixes + policy descent |
|---|---|---|---|
| reached | 43/46 | 41/46 | 43/46 |
| total frames | 19165 | 18380 | 16818 |

Clear wins for the fixes: facing_wall 672 -> 293 frames, goal_at_wall 808 -> 124, gap 613 ->
162. The policy descent fixes cornerL24 (2/2 at 293 frames vs 1/2) and false_door (2/2 at 754
vs 1/2 at 1278). corridor24/26 (2.4/2.6 m) fail in most runs of every configuration: at 0.24 m
routing the start pose 0.5 m off the centre line is already vetoed, no route exists on the first
frame, the robot drifts into the wall -- a router/resolution limit, not the reference. Final
set with 4 runs (R8) and 4 baseline runs: running.

## Changes under test (all opt-in, default off)

| param | fixes |
|---|---|
| `cold_on_divergence` | P1: test the divergence every frame |
| `clear_taper` | P3: clearance asked <= each end's own + taper x distance; rejection eased the same way |
| `w_pivot` 0.5 | P4, P5 |
| `sticky_tol_m` | P2: descent prefers last frame's side among near-equal ways |
| `cold_on_loop` | P6: a curve turning 270 deg more than the string is a cold start |
| `policy_descent` | the route from the robot's own heading: the lattice's policy rollout instead of the heading-minimum walk |

## 2026-11-01 night: the band -- the curve as points on the taut string (`band`)

The curvature profile is the root of P1, P4, P6 and P8: the curve is integrated from the robot,
so a small change near the start swings the whole tail, a loop cannot be unwound by small steps,
and every fix so far (cold-start triggers, taper, w_taut) patches a symptom of that. `band`
replaces the optimiser: every frame the taut string is resampled from the robot to the target
(<= 64 legs) and relaxed, one lane per point, by Jacobi steps on

    w_smooth sum |p_j-1 - 2 p_j + p_j+1|^2 + w_tension sum |p_j+1 - p_j|^2
    + w_clear (clearance deficit of the front disc, and of the rear disc past the first 0.75 m)^2
    + w_veto (the settle's veto at the point's own heading)

both ends held. The start heading is free (a robot facing away gets a pivot from MPPI's heading
term, not a loop). Defaults 16 / 1 / 150 steps; 0.7-0.9 ms a frame (3090) against 1.8 ms.

**Bugs on the way.** The rear axle disc 0.75 m behind each point along the curve's tangent is
right for a profile whose first tangent is the robot's heading; a band starts along the route, so
near the start the disc lay behind the robot in the kerb it had backed to: start hooks and 11 %
of the Zviretice parking-lot frames rejected. Points nearer the start than `rear` carry no rear
disc now (in the band's clearance term and its rejection). The scoring had the same blind spot
for every variant with a start pivot: `wall_run_m` / `run_p5` / `run_neg` drop that disc.

**Offline, the 7 real runs (robot's own frames), frame-weighted:**

| | loop | > 1 m off string | detour p95 | clearance p5 | touches | no curve, goal > 1.5 m | jump > 0.3 m | flick p95 | s-bends |
|---|---|---|---|---|---|---|---|---|---|
| robot's stage | 3.40 % | 14.07 % | 1.87 | 0.65 m | 0 | 0.12 % | 1.54 % | 0.12 m | 12.1 % |
| fix set | 0.35 | 0.92 | 1.06 | 0.59 | 0 | 0.21 | 1.83 | 0.13 | 8.8 |
| band | 0.23 | 0 | 1.00 | 0.65 | 0 | 2.45 | 1.86 | 0.14 | 4.0 |
| band + keep 0.75 | 0.22 | 0.01 | 1.03 | 0.63 | 0 | 2.60 | 1.86 | 0.06 | 3.5 |

(The string-relative columns favour the band by construction; clearance, loops, empties, jumps
and bends do not.) The band's empties are the parking lot (11 %): a goal ON a car or a kerb,
the band held to it, its last points on forbidden ground -- see `band_end_trunc_m` and the goal
region below.

**Jumps are the string's, not the optimiser's.** new3: every one of the band's 46 jumps is a
frame where the taut string's first 4 m moved > 0.3 m (34 of them 0.3-0.6 m: the 0.24 m
descent staircase reshuffling). 600 steps instead of 150 or 4x the tension do not remove them
(4.1 -> 4.0 %), so they are not under-convergence. `band_keep_m` (start each point from the
nearest point of last frame's band when the new string lies within keep_m of it everywhere)
halves the typical motion (flick p95 0.24 -> 0.10 m on new3) but not the count of big jumps.
`v_tol_per_m` (a per-metre allowance in the visibility test, for the open-ground 1.5 m bulge
flipping in and out at new3 t 49-57 s) did nothing (jump 3.9-4.7 %): rejected.

**Closed loop, robot-faithful, 23 worlds x 4 runs (band before the rear-disc fix):**

| | reached | frames |
|---|---|---|
| R0 robot's stage | 86/92 | 37761 |
| R9 fix set | 81/92 | 38513 |
| R10 band | 86/92 | 32559 (-14 %) |

Band wins: goal_at_wall 720 -> 118 frames, facing_wall 3/4 -> 4/4 and 2.2x faster, gap 385 ->
130, pocket 562 -> 331, ridge 349 -> 247, corridor26 3/4 -> 4/4; false_door 2/4 as the base
(the fix set's 0/4 is gone). Losses: cornerL24 2/4 -> 0/4 (the case policy_descent fixed),
slalom 332 -> 478 frames, trunks 502 -> 571, chicane 474 -> 514.

**Goal region (`plan_goal_region_m`, CostToGo).** The parking-lot figure-eights are the ROUTE:
a goal on the edge of a car is reachable as one cell only at the headings whose footprint fits,
and V routed round to the far side to arrive at one (the taut string itself wraps the car).
Seeding every cell within r of the goal at its straight distance lets the route stop at the
nearest standable point on the near side (priced seeds: a blocked one stays blocked).

**Closed loop, 8 runs, the 11 worlds that separate the variants** (band + taper 0.5 + end
truncation 1.5 m; T0 = rear disc everywhere, T75 = none within 0.75 m of the start, KT0 = T0 +
keep 0.75):

| world | R0 robot's stage | T0 | KT0 | T75 |
|---|---|---|---|---|
| chicane | 7/8 509 | 8/8 500 | 8/8 438 | 7/8 535 |
| cornerL24 | 2/8 | 2/8 | 1/8 | 1/8 |
| corridor24 | 6/8 198 | 7/8 178 | 7/8 205 | 6/8 242 |
| corridor26 | 7/8 251 | 8/8 163 | 8/8 200 | 7/8 178 |
| facing_wall | 7/8 341 | 8/8 243 | 8/8 242 | 8/8 258 |
| false_door | 5/8 1141 | 3/8 1334 | 2/8 1104 | 2/8 1272 |
| gap | 8/8 275 | 8/8 159 | 8/8 151 | 8/8 243 |
| pocket | 7/8 464 | 8/8 292 | 8/8 299 | 8/8 310 |
| slalom | 8/8 364 | 8/8 411 | 8/8 376 | 8/8 451 |
| trunks | 8/8 664 | 8/8 538 | 8/8 489 | 8/8 512 |
| wall_on_line | 8/8 204 | 8/8 202 | 8/8 208 | 8/8 212 |
| reached | 75/88 | 78/88 | 77/88 | 73/88 |

The rear disc near the start stays (T0): without it the band's first points hug the wall the
robot starts beside. Offline the skip made fewer bends (s-bend 4.5 vs 5.7 %), closed loop it
lost. false_door is a time-limit race in every configuration (reached runs take 950-1460 of the
1500 frames); the band loses it in the northern passage (north -> x 18: 548 frames against 326),
at 0.35-0.56 m mean clearance against 0.8-2.3 m, where the clearance governor slows it.

**The stage's clearance map misses most obstacles.** In that passage the field the stage keeps
clear of (`forbidden_distance`) reads its 1.68 m cap across the whole width: the walls are not
in it. It is the distance to cells the settle forbids at EVERY heading, and a wall or a car is
straddled at some headings. Zviretice parking lot: 5-47 cells a frame forbidden at every heading,
172-289 at half of them or more -- as many as V caps (173-289). The clearance term, in the
deployed stage and the band alike, sees a few percent of the obstacles; the curves stay clear
mostly because the taut string follows V. `forbid_frac` (plan_ref_forbid_frac) takes cells
forbidden at a fraction of headings.

Scored against the half-heading map (replay_ref.SCORE_FRAC 0.5, all variants), 7 real runs:

| | crosses (run_neg) | loop | stray | s-bend | no curve, goal > 1.5 m |
|---|---|---|---|---|---|
| robot's stage | 3.87 % | 3.40 % | 14.1 % | 12.1 % | 0.12 % |
| robot's stage, frac 0.5 | 0.04 | 5.83 | 18.9 | 17.2 | 1.44 |
| fix set | 4.81 | 0.35 | 0.92 | 8.8 | 0.21 |
| fix set, frac 0.5 | 2.99 | 0.32 | 0.57 | 13.7 | 3.97 |
| band T0 | 3.31 | 0.30 | 0 | 5.7 | 0 |
| band T0, frac 0.5 | 0.45 | 1.28 | 0.03 | 15.8 | 0.09 |

On the parking lot 13.5 % of the robot's own curves cross the half-heading map. The denser map
makes every stage safer and wigglier; whether that pays is a closed-loop question (S_bandT050,
S_base50, S_bandT075 campaigns).

**What the denser map needs: a smaller clearance ask.** The asked clearance (0.8 m + half-width)
was calibrated against the every-heading map; the half-heading map already reaches past the
faces, so with it every gap read too narrow and the band centred itself between blobs (s-bends
5.7 -> 15.8 %). Asking 0.2 m (`plan_ref_clearance_m`) instead, 7 real runs, half-heading scoring:

| | loop | crosses | clearance p5 | s-bend | no curve, goal > 1.5 m |
|---|---|---|---|---|---|
| robot's stage | 3.40 % | 3.87 % | 0.18 m | 12.1 % | 0.12 % |
| band T0 | 0.30 | 3.31 | 0.24 | 5.7 | 0 |
| band, frac 0.5, ask 0.8 | 1.28 | 0.45 | 0.35 | 15.8 | 0.09 |
| band, frac 0.5, ask 0.4 | 0.26 | 0.39 | 0.31 | 6.3 | 0.14 |
| **band, frac 0.5, ask 0.2 (b50c2)** | **0.09** | **0.25** | **0.30** | **4.7** | 0.29 |

frac 0.75 is no middle ground (crosses 2.3 %, s-bend 14.3 %). The live node with b50c2 on the
parking-lot bag matches the offline rerun (loop 0.25 %, crosses 1.07 %, empties 1.24 % live vs
0.25 / 1.04 / 1.24 offline).

**Closed loop, 23 worlds x 8 runs:**

| | reached | frames |
|---|---|---|
| robot's stage | 167/184 | 80283 |
| band T0 | 172/184 | 64543 |
| **b50c2** | **174/184** | **63606** |
| b50c2 + policy_descent | 174/184 | 66324 |
| robot's stage, frac 0.5 (11 worlds) | 75/88 vs 75/88 | |

b50c2 fixes the band's slalom slowdown (343 frames, base 364) and lifts cornerL24 to 4/8 and
false_door to 4/8. Policy descent on top: cornerL24 7/8 and false_door 8/8 (933 frames, faster
than base), but corridor26 4/8, facing_wall 6/8, forest and pillars 7/8, narrow_s +50 %; and on
the parking lot its same-goal frame jumps are 5.1 % against 3.2 % (robot 2.0 %; patek1 0.63 vs
0.82). `policy_sticky_m` (hysteresis among near-tie moves) changes nothing (5.0 %): the flips are
not ties, the walk starts from a different heading bin as the robot manoeuvres. Not recommended
until the route is consistent.

Jump metric note: `jump` counts goal switches and newly seen obstacles too; `jump2.py` recounts
excluding frames whose goal changed.

**Recommended (opt-in), b50c2:**

    -p plan_ref_band:=1.0 -p plan_ref_clear_taper:=0.5 -p plan_ref_band_end_trunc_m:=1.5 \
    -p plan_ref_forbid_frac:=0.5 -p plan_ref_clearance_m:=0.2

**End truncation, second pass.** Near a goal on structure the band can dip onto forbidden ground
0.2-1 m before the goal and come out again at the goal itself; truncation walked back only over
a failing TAIL, so those frames were still rejected (park0: 21 of 35 empties a front-disc
failure in the last metre). It now cuts before the first failing point anywhere in the last
1.5 m: frames without a curve, goal > 1.5 m, 0.29 -> 0.10 % over the 7 runs (park0 1.24 ->
0.32 %), robot's stage 0.12 %.

**16 runs a world on the 11 worlds that separate the variants** (b50c2 vs the robot's stage):

| world | robot's stage | b50c2 |
|---|---|---|
| chicane | 14/16 | 16/16 |
| cornerL24 | 2/16 | 7/16 |
| corridor24 | 11/16 | 11/16 |
| corridor26 | 13/16 | 16/16 |
| facing_wall | 14/16 | 16/16 |
| false_door | 8/16 | 8/16 |
| gap | 16/16 (239 f) | 16/16 (142 f) |
| pocket | 15/16 | 16/16 |
| slalom | 16/16 (419 f) | 15/16 (340 f) |
| trunks | 16/16 (592 f) | 16/16 (528 f) |
| wall_on_line | 16/16 | 16/16 |

All runs together (these at 16, the other 12 worlds at 8): 249/272 against 235/272, 18 % fewer
frames. The one slalom failure is a wedge: nose against the second gate's slab end, facing away
from the route; the string and the band go back round the slab end first, which a forward-only
robot pressed to it cannot start (1300 frames there).

**Truncation, third pass: only at the goal.** The second pass (cut before the first failing
point in the last 1.5 m) also cut curves that end at the window's edge, short of a crowded edge:
closed loop corridor26 6/8 and trunks 6/8 (165/184 over 23 worlds x 8). Gated on the goal being
in the window (87e18e2) both are 8/8 again.

**Final validation of the recommended code (87e18e2)**, 16 runs on the 11 separating worlds, 8
on the other 12: **246/272 reached against 235/272 for the robot's stage, 16 % fewer frames**
(114011 vs 135647); b50c2 with the first-pass truncation 249/272 (within noise). Offline, 7
real runs: loop 0.09 % (robot 3.40), crosses 0.31 % (3.87), s-bends 4.7 % (12.1), stray 0
(14.1), no curve with the goal > 1.5 m 0.15 % (0.12), same-goal frame jumps 1.89 % (1.54).
Live node: patek1 loop 0.21 % (4.34), crosses 0.17 % (1.71), no empties (0.17), same-goal jumps
0.31 % (0.82); castle run (hrad0) no loops, no crossings, 0.42 % empties (0.07), s-bends
11.9 % (0.5: a 0.2-0.3 m swing off an obstacle beside the robot as it leaves).

**`band_hold` (route hysteresis) -- calmer on real data, rejected in closed loop.** Keeping last
frame's band (from the robot on, extended to the new target) while it clears every forbidden
cell and is <= hold x the new string + 0.5 m: 7 real runs, same-goal jumps 1.89 -> 1.44 % at
1.05 and 1.35 % at 1.2 (robot's stage 1.54), parking lot 3.42 -> 2.49 %, flick p95 0.14 ->
0.06 m, loops and crossings unchanged. But closed loop, 8 runs on the 11 separating worlds,
hold 1.05 reaches 70/88 against ~75 for the recommended set: cornerL24 0/8, corridor24 5/8,
corridor26 6/8, false_door 3/8 -- the worlds where the first way round must be given up as the
map fills in. A way that still clears and is nearly as short in the window can still be the
wrong one. Opt-in, off; the remaining jumps are the price of following V.

**Band smoothness sweep** (8 runs, 11 separating worlds, recommended set): band_w_smooth 8 ->
77/88, 16 (default, from its 16-run rate) ~75/88, 32 -> 75/88. Within noise; default kept.

**Speed on the laptop's A500** (new3, 1111 frames): the deployed stage 1.80 ms median, 2.42 p95,
7.59 max (its 400-step cold starts); the recommended band 0.97 / 1.14 / 1.39 ms -- it has no
cold starts, so the worst frame is 5x better. (17 of 1111 deployed-stage frames differ from the
dasenka recording: cross-GPU FMA fusion, see warp_codegen_order_nondeterminism.)
