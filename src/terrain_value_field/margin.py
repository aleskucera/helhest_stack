"""Feasibility measured in standard deviations of room, from any set of constraints.

The one idea this library is built around. Each constraint on a state is asked how much room is
left in units of ITS OWN uncertainty:

    z_i = margin_i / max(sigma_i, floor_i)
    z   = min over i                          -- the binding constraint

Dividing by each constraint's own sigma is what makes the `min` meaningful. A tilt margin is in
radians and a clearance margin in metres; a raw `min` over those compares nothing. In sigmas
they are the same quantity, and the smallest one is genuinely the one about to be violated.

One knob follows: `k_sigma`, how many standard deviations of room the robot insists on. The
graded penalty comes off the same number, so "how pessimistic am I" and "how close is this to
bad" are not two separately-tuned things that fight.

Nothing here knows what a constraint means. A geometric producer supplies slope and step
margins, a physics producer supplies tilt and belly clearance, a learned one supplies whatever
it scores -- the arithmetic is identical, and that is the whole reason this is a library rather
than a part of one robot's planner.

`floor_i` is not optional. Without it a perfectly known map makes a state at 14.9 degrees of
roll against a 15 degree limit read as infinitely safe. The floor is the irreducible error --
localisation, controller tracking, model mismatch -- that no map improvement removes.
"""

from __future__ import annotations

import warp as wp

# A margin whose sigma is unknown is not evidence of safety. Constraints may opt out by writing
# this, and the reduction ignores them rather than treating them as infinitely comfortable.
IGNORED = wp.constant(1.0e30)


@wp.kernel
def margin_to_z_kernel(
    margin: wp.array4d(dtype=wp.float32),  # [constraint, row, col, heading]
    sigma: wp.array4d(dtype=wp.float32),  # same shape; per-constraint standard deviation
    floor: wp.array(dtype=wp.float32),  # [constraint] irreducible sd, in that constraint's units
    z: wp.array3d(dtype=wp.float32),
    z_certain: wp.array3d(dtype=wp.float32),
):
    """Reduce per-constraint (margin, sigma) to the binding margin, twice.

    `z` uses the supplied sigma; `z_certain` uses only the floor -- the same state scored as if
    the map carried no uncertainty beyond the irreducible. The pair is what separates a state
    that is bad ground from one that is merely unknown, and neither reading alone can.
    """
    r, c, t = wp.tid()
    best = float(IGNORED)
    best_certain = float(IGNORED)
    for i in range(margin.shape[0]):
        m = margin[i, r, c, t]
        if m >= IGNORED:
            continue  # this constraint declines to speak about this state
        f = floor[i]
        s = wp.max(sigma[i, r, c, t], f)
        best = wp.min(best, m / s)
        best_certain = wp.min(best_certain, m / f)
    z[r, c, t] = best
    z_certain[r, c, t] = best_certain


@wp.kernel
def classify_kernel(
    z: wp.array3d(dtype=wp.float32),
    z_certain: wp.array3d(dtype=wp.float32),
    k_sigma: wp.array(dtype=wp.float32),  # device scalar, so a captured graph can be retuned
    z_ref: wp.float32,
    penalty_weight: wp.float32,
    blocked: wp.array3d(dtype=wp.float32),
    penalty: wp.array3d(dtype=wp.float32),
    doubt: wp.array3d(dtype=wp.float32),
):
    """Turn the two margins into a veto, a graded cost, and a doubt field.

    `doubt` is the distinction that makes an uncertainty-aware value function worth more than a
    conservative one: a state that fails on ANY map is bad ground, and no amount of looking at
    it will help; a state that passes on a certain map and fails on this one is blocked by
    IGNORANCE, and is worth resolving. Reported per state; what a robot does about it is not
    this library's business.
    """
    r, c, t = wp.tid()
    k = k_sigma[0]
    zz = z[r, c, t]
    zc = z_certain[r, c, t]
    blocked[r, c, t] = wp.where(zz < k, 1.0, 0.0)
    penalty[r, c, t] = wp.where(zz < z_ref, penalty_weight * (z_ref - zz), 0.0)
    doubt[r, c, t] = wp.where(zz < k and zc >= k, zc - zz, 0.0)
