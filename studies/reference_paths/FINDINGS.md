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

## Changes under test (all opt-in, default off)

| param | fixes |
|---|---|
| `cold_on_divergence` | P1: test the divergence every frame |
| `clear_taper` | P3: clearance asked <= each end's own + taper x distance; rejection eased the same way |
| `w_pivot` 0.5 | P4, P5 |
| `sticky_tol_m` | P2: descent prefers last frame's side among near-equal ways |
| `cold_on_loop` | P6: a curve turning 270 deg more than the string is a cold start |
| `policy_descent` | the route from the robot's own heading: the lattice's policy rollout instead of the heading-minimum walk |
