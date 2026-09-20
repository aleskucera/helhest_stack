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
sparsity rather than the driving -- which is 0.095 m of sd on a height difference against the
0.028 m that `var_meas` alone would claim. A factor of 3.4, in the optimistic direction, in the
freshest part of the map. Out at 5-15 m the median spread reaches 76 s and 0.75 m.

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
