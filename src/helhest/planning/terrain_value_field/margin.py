"""Feasibility measured in standard deviations of room, from any set of constraints.

The one idea this library is built around. Each constraint on a state is asked how much room is
left in units of ITS OWN uncertainty:

    z_i = margin_i / max(sigma_i, floor_i)
    z   = min over i                          -- the binding constraint

Dividing by each constraint's own sigma is what makes the `min` meaningful. A tilt margin is in
radians and a clearance margin in metres; a raw `min` over those compares nothing. In sigmas
they are the same quantity, and the smallest one is genuinely the one about to be violated.

One knob follows: `z_veto`, how many standard deviations of room the robot insists on. The
graded penalty comes off the same number, so "how pessimistic am I" and "how close is this to
bad" are not two separately-tuned things that fight.

Nothing here knows what a constraint means. A geometric producer supplies slope and step
margins, a physics producer supplies tilt and belly clearance, a learned one supplies whatever
it scores -- the arithmetic is identical, and that is the whole reason this is a library rather
than a part of one robot's planner.

`floor_i` is not optional. Without it a perfectly known map makes a state at 14.9 degrees of
roll against a 15 degree limit read as infinitely safe. The floor is the irreducible error --
localisation, controller tracking, model mismatch -- that no map improvement removes.

IGNORED IS NOT "UNMEASURED", AND CONFUSING THE TWO IS SILENT. A constraint set to `IGNORED`
declines to speak about a state, and `min` then skips it. If EVERY constraint declines, `z` stays
at IGNORED -- enormously above any `z_veto`, and above `z_charge` too -- so the state is not vetoed,
carries no penalty, and produces NO DOUBT. An unmeasured cell becomes indistinguishable from
perfect flat ground, and nothing anywhere says so. Measured on a goal placed past the horizon,
with the 25 columns between the robot and the goal never observed:

    unknown encoded as  V at robot  V optimistic   doubt out there
   observed everywhere       17.50         17.50              0.00
              declines       17.50         17.50              0.00   <-- silently optimistic
             uncertain UNREACHABLE         17.50             38.86
                filled UNREACHABLE   UNREACHABLE              0.00   <-- silently pessimistic

`declines` drives straight at a goal through terrain nobody has seen, at exactly the cost it
would charge for measured ground. `filled` -- an invented height reported with a confident sigma
-- is the opposite failure: the planner believes a wall is there and the doubt field says there
is nothing to learn, which is indistinguishable from a genuinely walled-off map.

Unmeasured belongs in the SIGMA. Report the best guess with an honest sigma and all three
readings do their job: the pessimistic solve refuses to drive blind, the optimistic one shows a
route exists if the ground is as good as it looks, and `doubt` says which cells are responsible.
`IGNORED` is for a constraint that does not APPLY to a state -- a slope test where no plane was
fitted -- not for one whose input was never measured.

HARD CONSTRAINTS. `floor_i = 0` marks a test with no uncertainty to divide by -- a solver that
failed to resolve, a step taller than a gate. It is a pure sign test: `margin < 0` vetoes the state
and sets `hard`, and the constraint takes no part in `z`, the graded charge or `doubt`. Dividing by
a made-up tiny sigma instead would drag `z` to +-1e9 and swamp every real margin in the `min`.

POSE COST. The veto and the graded cost travel as ONE field, with the veto in the sign:

    pose_cost = penalty          a state the robot may occupy, penalty >= 0
              = -1 - penalty     a vetoed state

The graded penalty is never negative, so the sign bit is free, and -1 - (-1 - p) = p recovers
it -- unlike a sentinel. Not to the bit, though: in float32, -1 - p drops every part of p below
half an ulp of 1 (6e-8), so a vetoed state's penalty comes back approximately. A robot that adds
its own costs to vetoed states and re-packs them reads `penalty`, written alongside, instead.

This is not tidiness. The relax kernel reads
this field about 35 times per thread per sweep, once for every swept cell of every primitive,
and it is bandwidth-bound; as two arrays that was two loads from two cache lines for one
decision. `blocked = pose_cost < 0` wherever you want to look at it separately.
"""

from __future__ import annotations

import warp as wp

# A margin whose sigma is unknown is not evidence of safety. Constraints may opt out by writing
# this, and the reduction ignores them rather than treating them as infinitely comfortable.
IGNORED = wp.constant(1.0e30)


@wp.kernel
def margin_to_fields_kernel(
    margin: wp.array4d(dtype=wp.float32),  # [constraint, row, col, heading]
    sigma: wp.array4d(dtype=wp.float32),  # [constraint, row, col, heading]
    floor: wp.array(dtype=wp.float32),  # [constraint]
    z_veto: wp.array(dtype=wp.float32),  # [1]
    z_charge: wp.float32,
    charge_per_sigma: wp.float32,
    z: wp.array3d(dtype=wp.float32),  # [row, col, heading]
    z_certain: wp.array3d(dtype=wp.float32),  # [row, col, heading]
    hard: wp.array3d(dtype=wp.float32),  # [row, col, heading] 1 = a hard constraint fails
    pose_cost: wp.array3d(dtype=wp.float32),  # [row, col, heading]
    penalty: wp.array3d(dtype=wp.float32),  # [row, col, heading] the graded cost, >= 0
    doubt: wp.array3d(dtype=wp.float32),  # [row, col, heading]
):
    """Reduce the constraints and classify the result, in one pass.

    `margin_to_z_kernel` and `classify_kernel` below do the same work split in two, and remain
    the reference this is tested against. They are kept because they are easier to read, not
    because either is a fallback.

    Fusing them is worth doing for one reason, and it is measurable: `classify` reads exactly
    what the reduction just wrote, so splitting the two sends `z` and `z_certain` out to DRAM
    and straight back in for nothing. On 2.56 M states with two constraints that is 1.31 ms
    against 1.10 ms, a 1.20x saving (`dev/bench_margin.py`).

    `z_veto` is an array rather than a float so a captured CUDA graph can be retuned
    without re-recording it.

    There is nothing else to win here. The reduction runs at 82.6 GB/s against a measured peak
    of 80 GB/s on this device -- it is already at the memory wall, with no compute to hide and
    no data reused between threads, which is also why tiling it would cost rather than help.
    """
    r, c, t = wp.tid()
    best = float(IGNORED)
    best_certain = float(IGNORED)
    failed = float(0.0)
    for i in range(margin.shape[0]):
        m = margin[i, r, c, t]
        if m >= IGNORED:
            continue  # this constraint declines to speak about this state
        f = floor[i]
        if f <= 0.0:  # hard: a sign test, see HARD CONSTRAINTS above
            if m < 0.0:
                failed = 1.0
            continue
        s = wp.max(sigma[i, r, c, t], f)
        best = wp.min(best, m / s)
        best_certain = wp.min(best_certain, m / f)
    z[r, c, t] = best
    z_certain[r, c, t] = best_certain
    hard[r, c, t] = failed
    k = z_veto[0]
    pen = wp.where(best < z_charge, charge_per_sigma * (z_charge - best), 0.0)
    # veto in the sign; see POSE COST above
    pose_cost[r, c, t] = wp.where(best < k or failed > 0.5, -1.0 - pen, pen)
    penalty[r, c, t] = pen
    doubt[r, c, t] = wp.where(best < k and best_certain >= k, best_certain - best, 0.0)


@wp.kernel
def margin_to_z_kernel(
    margin: wp.array4d(dtype=wp.float32),  # [constraint, row, col, heading]
    sigma: wp.array4d(dtype=wp.float32),  # [constraint, row, col, heading]
    floor: wp.array(dtype=wp.float32),  # [constraint]
    z: wp.array3d(dtype=wp.float32),  # [row, col, heading]
    z_certain: wp.array3d(dtype=wp.float32),  # [row, col, heading]
    hard: wp.array3d(dtype=wp.float32),  # [row, col, heading] 1 = a hard constraint fails
):
    """Reduce per-constraint (margin, sigma) to the binding margin, twice.

    `z` uses the supplied sigma; `z_certain` uses only the floor -- the same state scored as if
    the map carried no uncertainty beyond the irreducible. The pair is what separates a state
    that is bad ground from one that is merely unknown, and neither reading alone can.

    The library runs `margin_to_fields_kernel`, which is this plus `classify_kernel` in one
    pass. This split pair is the readable statement of what that computes, and the oracle it is
    held to in `test_fusion_matches_the_split_pair`.
    """
    r, c, t = wp.tid()
    best = float(IGNORED)
    best_certain = float(IGNORED)
    failed = float(0.0)
    for i in range(margin.shape[0]):
        m = margin[i, r, c, t]
        if m >= IGNORED:
            continue  # this constraint declines to speak about this state
        f = floor[i]
        if f <= 0.0:  # hard: a sign test, see HARD CONSTRAINTS above
            if m < 0.0:
                failed = 1.0
            continue
        s = wp.max(sigma[i, r, c, t], f)
        best = wp.min(best, m / s)
        best_certain = wp.min(best_certain, m / f)
    z[r, c, t] = best
    z_certain[r, c, t] = best_certain
    hard[r, c, t] = failed


@wp.kernel
def classify_kernel(
    z: wp.array3d(dtype=wp.float32),  # [row, col, heading]
    z_certain: wp.array3d(dtype=wp.float32),  # [row, col, heading]
    hard: wp.array3d(dtype=wp.float32),  # [row, col, heading] 1 = a hard constraint fails
    z_veto: wp.array(dtype=wp.float32),  # [1]
    z_charge: wp.float32,
    charge_per_sigma: wp.float32,
    pose_cost: wp.array3d(dtype=wp.float32),  # [row, col, heading]
    penalty: wp.array3d(dtype=wp.float32),  # [row, col, heading] the graded cost, >= 0
    doubt: wp.array3d(dtype=wp.float32),  # [row, col, heading]
):
    """Turn the two margins into a veto, a graded cost, and a doubt field.

    See `margin_to_fields_kernel`: the library fuses this with the reduction. Kept as the
    readable reference.

    `doubt` is the distinction that makes an uncertainty-aware value function worth more than a
    conservative one: a state that fails on ANY map is bad ground, and no amount of looking at
    it will help; a state that passes on a certain map and fails on this one is blocked by
    IGNORANCE, and is worth resolving. Reported per state; what a robot does about it is not
    this library's business.
    """
    r, c, t = wp.tid()
    k = z_veto[0]
    zz = z[r, c, t]
    zc = z_certain[r, c, t]
    pen = wp.where(zz < z_charge, charge_per_sigma * (z_charge - zz), 0.0)
    # veto in the sign
    pose_cost[r, c, t] = wp.where(zz < k or hard[r, c, t] > 0.5, -1.0 - pen, pen)
    penalty[r, c, t] = pen
    doubt[r, c, t] = wp.where(zz < k and zc >= k, zc - zz, 0.0)


@wp.kernel
def pack_pose_cost_kernel(
    blocked: wp.array3d(dtype=wp.float32),  # [row, col, heading] > 0.5 = vetoed
    penalty: wp.array3d(dtype=wp.float32),  # [row, col, heading]
    pose_cost: wp.array3d(dtype=wp.float32),  # [row, col, heading]
):
    """Pack a veto and a graded cost into the one signed field the solver reads (POSE COST).

    For a robot that edits the classified fields -- erodes the veto, adds costs of its own -- and
    hands the result back. The clamp at zero is not defensive noise: a negative penalty would read
    as a veto and quietly make a passable state impassable, so the encoding's one precondition is
    enforced where it is produced.
    """
    r, c, t = wp.tid()
    pen = wp.max(penalty[r, c, t], 0.0)
    pose_cost[r, c, t] = wp.where(blocked[r, c, t] > 0.5, -1.0 - pen, pen)
