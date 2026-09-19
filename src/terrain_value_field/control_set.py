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
"""

from __future__ import annotations

import math

import numpy as np


def arc_control_set(
    n_theta: int,
    resolution: float,
    step: float,
    turn_radius: float,
    max_sweep: int,
    nseg: int,
    pivot_cost: float = 0.0,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Host-side forward-arc motion primitives. For each heading bin and each turn rate, integrate
    the arc of length `step`, and record: endpoint cell offset (dr, dc), resulting heading bin, arc
    cost, and the swept cells (offsets) the arc passes through (for collision). Min turn radius
    `turn_radius` caps the turn rate, so turning costs space -- the whole point.

    pivot_cost > 0 appends two POINT-TURN primitives (heading +-1 bin in place, cost = pivot_cost
    [m-equivalent] per bin) -- the skid-steer can rotate on the spot, so `goal behind` routes as
    pivot-then-drive instead of a wide loop (or +inf). Endpoint-heading feasibility is enforced for
    free: a blocked pose holds V = +inf, so a pivot into it never helps. 0 = forward-arcs only."""
    dth = 2.0 * math.pi / n_theta
    turns = [
        -step / turn_radius,
        -step / turn_radius / 2.0,
        0.0,
        step / turn_radius / 2.0,
        step / turn_radius,
    ]  # dtheta over the step
    n_arc = len(turns)
    n_prim = n_arc + (2 if pivot_cost > 0.0 else 0)
    prim_dr = np.zeros((n_theta, n_prim), np.int32)
    prim_dc = np.zeros((n_theta, n_prim), np.int32)
    prim_heading = np.zeros((n_theta, n_prim), np.int32)
    prim_cost = np.full((n_theta, n_prim), step, np.float32)
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
            prim_dc[it, p] = int(round(x / resolution))
            prim_dr[it, p] = int(round(y / resolution))
            prim_heading[it, p] = int(math.floor(((th + dth_p) % (2.0 * math.pi)) / dth)) % n_theta
            uniq = sorted(set(cells))[:max_sweep]
            for s, (cr, cc) in enumerate(uniq):
                sweep_dr[it, p, s] = cr
                sweep_dc[it, p, s] = cc
            sweep_n[it, p] = len(uniq)
        if pivot_cost > 0.0:
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
