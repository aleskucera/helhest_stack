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

**Sim vs robot.** drive_sim routes on 0.08 m cells (200x200 over 16 m); the robot routes on
0.24 m (`plan_lat_coarsen` 3, 66x66). The stage's descent, staircase and distance map are 3x
finer in the sim.

**Speed.** The stage is ~3.1 ms a frame on a 3090 (100 Adam steps); ~1.7 ms of that is the
single-thread descent / visibility / taut kernels, not the optimiser. Stopping Adam early
(`stop_tol`) saves ~1 ms but degrades the curve (single_on_line: detour 1.02 -> 1.09) -- rejected.

## Changes under test (all opt-in, default off)

| param | fixes |
|---|---|
| `cold_on_divergence` | P1: test the divergence every frame |
| `clear_taper` | P3: clearance asked <= each end's own + taper x distance; rejection eased the same way |
| `w_pivot` 0.5 | P4, P5 |
| `sticky_tol_m` | P2: descent prefers last frame's side among near-equal ways |
| `cold_on_loop` | P6: a curve turning 270 deg more than the string is a cold start |
| `policy_descent` | the route from the robot's own heading: the lattice's policy rollout instead of the heading-minimum walk |
