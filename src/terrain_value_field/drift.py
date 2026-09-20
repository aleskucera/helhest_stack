"""How much the pose drift VARIES across a footprint, which is the part of it that matters.

A belief map built from a moving robot carries two kinds of height uncertainty. The measurement
part is independent per cell. The drift part is one shared random walk on the robot's pose, so
two cells share whatever drift accrued over their common interval and differ only by the drift
accrued since the older of them was last seen:

    Var(h_A - h_B) = var_meas_A + var_meas_B + |drift_A - drift_B|

A planner's margins are differences -- tilt across a footprint, clearance against the ground the
robot settles onto -- so this, and not the absolute height variance, is what it should divide by.
Feeding it `var_h` would veto a perfectly measured patch purely because a minute passed; feeding
it `var_meas` alone assumes every cell under the robot was measured at the same instant.

That assumption is not true, and not because of revisits. On a 14.5 Hz sensor whose returns fill
only about a third of 0.2 m cells per frame, neighbouring cells inside one footprint are painted
several sweeps apart. Measured on a real run, within 5 m of the robot: the age spread across a
footprint is 1.11 s at the median -- the same early in the run and late, so it is the sensor's
sparsity rather than the driving -- with a p90 reaching 88 s wherever the robot crosses its own
earlier track. Out at 5-15 m the median reaches 76 s.

What that costs depends on the PLATFORM's drift rate, and by a lot. Those same age spreads, on a
map whose measurement sd is 1.75 cm:

    age spread          q_z = 7.4e-03 (dead reckoning)   q_z = 7.5e-05 (on-device SLAM)
    1.11 s  (median)              0.095 m   3.8x                 0.026 m   1.07x
    87.7 s  (a revisit)           0.808 m  32.6x                 0.085 m   3.43x
    76.5 s  (coarse ground)       0.754 m  30.4x                 0.080 m   3.22x

So on a well-localised robot this is a SEAM correction and not a general one: where the ages
under a footprint match, which is the ordinary case in a window that is continuously re-measured,
it moves the answer by a few per cent. Where old data meets new -- a revisit, a window edge, the
coarse layer's aged ground -- it is worth a factor of three, in the optimistic direction, which
is the direction that drives a robot into things.

Fit your own rate. The two columns above differ by 100x and the conclusions differ with them.

`max - min` over a footprint bounds `|drift_A - drift_B|` for every pair inside it, so a producer
that inflates with it is conservative and never optimistic. Which is the right way to be wrong.
"""

from __future__ import annotations

import warp as wp


@wp.kernel
def drift_spread_kernel(
    drift: wp.array2d(dtype=wp.float32),  # [row, col]
    radius: wp.int32,
    spread: wp.array2d(dtype=wp.float32),  # [row, col]
):
    """`max(drift) - min(drift)` over the square of `radius` cells about each cell.

    `drift` is the belief's `var_h - var_meas` [m^2]: non-negative wherever a height exists, and
    NEGATIVE where none does. Unmeasured cells are skipped rather than counted as zero drift,
    which would read as a fresh measurement and shrink the spread exactly where the map is worst.
    A cell with no measured neighbour at all gets 0, since there is no pair to be uncertain about.
    """
    r, c = wp.tid()
    rows = drift.shape[0]
    cols = drift.shape[1]
    lo = float(1.0e30)
    hi = float(-1.0e30)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            rr = r + dr
            cc = c + dc
            if rr >= 0 and rr < rows and cc >= 0 and cc < cols:
                d = drift[rr, cc]
                if d >= 0.0:
                    lo = wp.min(lo, d)
                    hi = wp.max(hi, d)
    spread[r, c] = wp.where(hi >= lo, hi - lo, 0.0)


def footprint_drift_spread(
    drift: wp.array,
    radius: int,
    out: wp.array | None = None,
) -> wp.array:
    """Host-side wrapper. `radius` is in CELLS and is the producer's business -- only it knows how
    big the robot is, and the spread is footprint-shaped."""
    if radius < 0:
        raise ValueError(f"radius must be >= 0 cells, got {radius}")
    if out is None:
        out = wp.zeros_like(drift)
    wp.launch(
        drift_spread_kernel,
        dim=drift.shape,
        inputs=[drift, int(radius)],
        outputs=[out],
        device=drift.device,
    )
    return out
