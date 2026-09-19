"""Control sets: the moves a robot may make from a state, and what each one sweeps.

A control set is the edge structure of the state space -- Pivtoraiko and Kelly's term for the
canonical set of repeating moves that a state lattice is built from. Two are provided, and they
produce the same tables, so the solver never learns which it was given:

  `arc_control_set`   forward arcs capped by a minimum turn radius, optionally with point
                      turns. Orientation matters, so the state is (x, y, heading).
  `omni_control_set`  eight neighbours at a single heading bin. A holonomic robot has no
                      orientation-dependent constraints, and with `n_theta = 1` the solver
                      degenerates to grid value iteration -- the same kernel, no special case.

Each set gives, per (heading bin, move): the endpoint cell offset, the heading it ends at, its
cost, and every cell it crosses. The swept cells are the point: checking only the endpoint lets
a robot step over a wall thinner than one move.

LATTICE CLOSURE. An arc is integrated in continuous space and then snapped to the lattice, so the
heading the table records is the heading the robot actually reaches ONLY if the turn lands on a
bin boundary. When it does not, the recorded end heading is wrong by up to half a bin on every
move -- and since the margin field is indexed by heading, feasibility gets evaluated at a pose
the robot will not occupy. Distinct turn rates also collapse onto one lattice transition, so the
solver relaxes duplicates. At n_theta=8, step=0.3, turn_radius=0.5 that is 17.2 deg of heading
error per move and 20% of the primitives redundant. `closing_step` gives a step that closes, and
`arc_control_set` warns when handed one that does not.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

# The five turn rates, as fractions of the sharpest arc the turn radius allows.
_TURN_FRACTIONS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def closing_step(n_theta: int, turn_radius: float, bins: int = 2) -> float:
    """Arc length whose sharpest turn spans exactly `bins` heading bins.

    `bins` must be EVEN: the half-rate arcs turn by `bins / 2`, and those have to land on a bin
    boundary too. An odd `bins` leaves them on a half-bin, which is the worst case for closure.
    """
    if bins <= 0 or bins % 2 != 0:
        raise ValueError(f"bins must be a positive even number (half-rate arcs), got {bins}")
    return float(turn_radius) * (2.0 * math.pi / int(n_theta)) * int(bins)


def _arc_length(chord: float, turned: float) -> float:
    """Length of the circular arc with this chord and this central angle.

    chord = 2 r sin(turned/2) and length = r * turned, so length = chord * turned /
    (2 sin(turned/2)), which tends to the chord as `turned` tends to zero.
    """
    if abs(turned) < 1.0e-9:
        return chord
    return chord * turned / (2.0 * math.sin(0.5 * turned))


def arc_control_set(
    n_theta: int,
    resolution: float,
    step: float,
    turn_radius: float,
    max_sweep: int,
    nseg: int,
    turn_weight: float = 0.0,
    pivot_cost: float | None = None,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Host-side forward-arc motion primitives. For each heading bin and each turn rate, integrate
    the arc of length `step`, and record: endpoint cell offset (dr, dc), resulting heading bin, arc
    cost, and the swept cells (offsets) the arc passes through (for collision). Min turn radius
    `turn_radius` caps the turn rate, so turning costs space -- the whole point.

    Cost is the REALIZED arc length: recomputed from the snapped endpoint and the snapped heading
    change, not the nominal `step`. Snapping moves the endpoint by up to a cell, which spreads the
    ground a move actually covers over ~26% at n_theta=8, step=0.3; charging the nominal step for
    all of them hands the planner a discount on whichever arc happens to round outwards, and it
    will take it. `turn_weight` [m per rad of heading change] is charged on top of that length, so
    a straight is cheaper than an arc covering the same distance. 0 = distance only.

    `pivot_cost` [m per heading bin] appends two POINT-TURN primitives (heading +-1 bin in place)
    -- the skid-steer can rotate on the spot, so `goal behind` routes as pivot-then-drive instead
    of a wide loop (or +inf). A half turn is n_theta/2 of them, so the price already scales with
    angle; 3-5x `step` keeps pivots to the dead ends that need them. Endpoint-heading feasibility
    is enforced for free: a blocked pose holds V = +inf, so a pivot into it never helps. None =
    forward arcs only. Zero is rejected -- a free move with no displacement is a zero-cost cycle.
    """
    if pivot_cost is not None and pivot_cost <= 0.0:
        raise ValueError(f"pivot_cost must be > 0 or None, got {pivot_cost}")
    dth = 2.0 * math.pi / n_theta
    sharpest = step / turn_radius  # dtheta over the step, at the min turn radius
    turns = [f * sharpest for f in _TURN_FRACTIONS]  # dtheta over the step
    if any(abs(t / dth - round(t / dth)) > 1.0e-6 for t in turns):
        bins = max(2, 2 * int(round(0.5 * sharpest / dth)))
        warnings.warn(
            f"arc_control_set: step={step:.4f} does not close on the lattice for "
            f"turn_radius={turn_radius}, n_theta={n_theta} -- recorded end headings will be off "
            f"by up to {0.5 * math.degrees(dth):.1f} deg per move and some primitives will be "
            f"duplicates. Nearest closing step is "
            f"{closing_step(n_theta, turn_radius, bins):.4f} = "
            f"closing_step({n_theta}, {turn_radius}, bins={bins}).",
            stacklevel=2,
        )
    n_arc = len(turns)
    n_prim = n_arc + (2 if pivot_cost is not None else 0)
    prim_dr = np.zeros((n_theta, n_prim), np.int32)
    prim_dc = np.zeros((n_theta, n_prim), np.int32)
    prim_heading = np.zeros((n_theta, n_prim), np.int32)
    prim_cost = np.zeros((n_theta, n_prim), np.float32)
    sweep_dr = np.zeros((n_theta, n_prim, max_sweep), np.int32)
    sweep_dc = np.zeros((n_theta, n_prim, max_sweep), np.int32)
    sweep_n = np.zeros((n_theta, n_prim), np.int32)
    for it in range(n_theta):
        th = (it + 0.5) * dth
        for p, dth_p in enumerate(turns):
            x, y = 0.0, 0.0
            cells = []
            for s in range(1, nseg):
                cth = th + dth_p * (float(s) - 0.5) / float(nseg - 1)
                x += (step / float(nseg - 1)) * math.cos(cth)
                y += (step / float(nseg - 1)) * math.sin(cth)
                cells.append((int(round(y / resolution)), int(round(x / resolution))))
            dc_p = int(round(x / resolution))
            dr_p = int(round(y / resolution))
            end_bin = int(math.floor(((th + dth_p) % (2.0 * math.pi)) / dth)) % n_theta
            prim_dc[it, p] = dc_p
            prim_dr[it, p] = dr_p
            prim_heading[it, p] = end_bin
            # Charge the move the LATTICE makes, not the one that was integrated: the arc through
            # the snapped endpoint, turning by the snapped heading change. On a closing set the two
            # agree to the endpoint rounding; on a non-closing one they do not, hence the warning.
            turned = float((end_bin - it + n_theta // 2) % n_theta - n_theta // 2) * dth
            chord = math.hypot(dc_p * resolution, dr_p * resolution)
            prim_cost[it, p] = _arc_length(chord, turned) + turn_weight * abs(turned)
            uniq = sorted(set(cells))[:max_sweep]
            for s, (cr, cc) in enumerate(uniq):
                sweep_dr[it, p, s] = cr
                sweep_dc[it, p, s] = cc
            sweep_n[it, p] = len(uniq)
        if pivot_cost is not None:
            for p, dbin in ((n_arc, -1), (n_arc + 1, +1)):
                # in place: endpoint = same cell, heading one bin over; sweep = the cell itself so
                # the pivot picks up the pose's graded tilt like any arc
                prim_heading[it, p] = (it + dbin) % n_theta
                prim_cost[it, p] = pivot_cost
                sweep_n[it, p] = 1
    return n_prim, prim_dr, prim_dc, prim_heading, prim_cost, sweep_dr, sweep_dc, sweep_n


def omni_control_set(
    resolution: float,
    diagonal_cost_scale: float = math.sqrt(2.0),
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Eight neighbours at a single heading bin: grid value iteration, same kernel.

    For a robot whose feasibility does not depend on which way it faces. Diagonals cost
    sqrt(2) so the field approximates Euclidean distance rather than Chebyshev.
    """
    moves = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    n_prim = len(moves)
    prim_dr = np.zeros((1, n_prim), np.int32)
    prim_dc = np.zeros((1, n_prim), np.int32)
    prim_heading = np.zeros((1, n_prim), np.int32)
    prim_cost = np.zeros((1, n_prim), np.float32)
    sweep_dr = np.zeros((1, n_prim, 1), np.int32)
    sweep_dc = np.zeros((1, n_prim, 1), np.int32)
    sweep_n = np.ones((1, n_prim), np.int32)
    for p, (dr, dc) in enumerate(moves):
        prim_dr[0, p], prim_dc[0, p] = dr, dc
        diagonal = dr != 0 and dc != 0
        prim_cost[0, p] = resolution * (diagonal_cost_scale if diagonal else 1.0)
        # One swept cell -- the destination. A single-cell step cannot straddle anything.
        sweep_dr[0, p, 0], sweep_dc[0, p, 0] = dr, dc
    return n_prim, prim_dr, prim_dc, prim_heading, prim_cost, sweep_dr, sweep_dc, sweep_n
