"""Cost-to-go V(row, col, heading) over a discretised state space, value-iterated on the GPU.

Takes only the per-state `blocked` and graded-`penalty` fields that a feasibility producer makes,
a control set, and a set of zero-cost seed states -- it never learns what a constraint meant or
what produced it. Value iteration runs to a fixed point over EVERY state rather than searching
for one path, so the output is a field you can query from wherever the robot actually is, not a
trajectory it has to re-attach to.

That is the difference from a lattice planner in the usual sense. Those search the lattice with
A* and use a precomputed cost-to-go as the heuristic; this computes that cost-to-go and stops.
Every state updates from the previous sweep's values with no priority queue and no ordering, so
it is one GPU thread per state, and the convergence loop runs ON DEVICE (`capture_while`) --
the whole solve is CUDA-graph-capturable.

A state from which the seeds are unreachable under the control set keeps cost +inf. For a
forward-only robot that is not a failure but information: it is exactly the misaligned approach
a 2-D geodesic cannot express.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from .control_set import arc_control_set


@wp.kernel
def _free_seeds_kernel(
    seeds: wp.array(dtype=wp.float32, ndim=3),
    blocked: wp.array(dtype=wp.float32, ndim=3),
):
    """A seeded state is a source, so it may not also be vetoed -- the iteration needs somewhere
    to start. Nothing else is cleared."""
    r, c, t = wp.tid()
    if seeds[r, c, t] > 0.5:
        blocked[r, c, t] = 0.0


@wp.kernel
def _init_kernel(
    seeds: wp.array(dtype=wp.float32, ndim=3),
    inf: wp.float32,
    dist: wp.array(dtype=wp.float32, ndim=3),
):
    """Seed: 0 wherever the mask is set, +inf elsewhere.

    A MASK rather than a single goal cell, because value iteration takes multiple sources for
    free where a graph search would need a virtual node. One seeded state is goal-seeking; a
    seeded frontier is exploration; a seeded set of docks is "reach any of these". Which of
    those a robot wants is not this library's decision -- it only has to be expressible.
    """
    r, c, t = wp.tid()
    dist[r, c, t] = wp.where(seeds[r, c, t] > 0.5, 0.0, inf)


@wp.kernel
def _keep_going_kernel(
    changed: wp.array(dtype=wp.int32),
    iter_count: wp.array(dtype=wp.int32),
    cap: wp.int32,
    keep_running: wp.array(dtype=wp.int32),
):
    """Device-side loop condition (so capture_while keeps the value iteration on the GPU, no host
    sync): keep going while the last body improved SOME cell and we're under the cap. dim=1."""
    it = iter_count[0] + 1
    iter_count[0] = it
    if changed[0] != 0 and it < cap:
        keep_running[0] = 1
    else:
        keep_running[0] = 0


@wp.kernel
def _relax_kernel(
    dist_in: wp.array(dtype=wp.float32, ndim=3),
    blocked: wp.array(
        dtype=wp.float32, ndim=3
    ),  # [h, w, n_theta] PER-POSE feasibility (1 = blocked)
    penalty: wp.array(
        dtype=wp.float32, ndim=3
    ),  # [rows, cols, headings] graded cost from the margin
    prim_dr: wp.array(dtype=wp.int32, ndim=2),  # [n_theta, n_prim] endpoint row offset
    prim_dc: wp.array(dtype=wp.int32, ndim=2),  # endpoint col offset
    prim_heading: wp.array(dtype=wp.int32, ndim=2),  # heading bin the arc ends at
    prim_cost: wp.array(dtype=wp.float32, ndim=2),  # arc length
    sweep_dr: wp.array(dtype=wp.int32, ndim=3),  # [n_theta, n_prim, max_sweep] swept-cell offsets
    sweep_dc: wp.array(dtype=wp.int32, ndim=3),
    sweep_n: wp.array(dtype=wp.int32, ndim=2),  # [n_theta, n_prim] swept-cell count
    n_prim: wp.int32,
    penalty_weight: wp.float32,  # 0 -> pure distance; >0 -> prefer states with more margin
    inf: wp.float32,
    dist_out: wp.array(dtype=wp.float32, ndim=3),
    changed: wp.array(dtype=wp.int32),
):
    """One min-relaxation sweep of the (row, col, heading) value function:

        V_out[s] = min( V_in[s], min over forward-arc primitives p
                         of  cost(p) + V_in[ next(s, p) ]   if p's swept cells are all free )

    cost(p) = arc_length * (1 + penalty_weight * mean penalty over the swept cells), so with
    penalty_weight > 0 the geodesic PREFERS flatter poses (not just avoids blocked ones). Feasibility and
    graded cost come from the PER-POSE field (the robot's settle), sampled at the swept cells using the
    state's own heading t (the arc rotates little over one step), so a wall face -- where a body tilts or
    high-centers -- blocks the crossing arc while flat ground stays cheap. The whole swept arc must be
    clear (not just the endpoint), so the robot can't jump a thin wall. Iterating to a fixed point gives
    the forward-only cost-to-go; a misaligned pose from which the goal is unreachable stays +inf."""
    r, c, t = wp.tid()
    h = dist_in.shape[0]
    w = dist_in.shape[1]
    if blocked[r, c, t] > 0.5:
        dist_out[r, c, t] = inf
        return
    best = dist_in[r, c, t]
    for p in range(n_prim):
        ok = int(1)
        ns = sweep_n[t, p]
        tsum = float(0.0)
        for s in range(ns):
            sr = r + sweep_dr[t, p, s]
            sc = c + sweep_dc[t, p, s]
            inb = int(0)
            if sr >= 0 and sr < h and sc >= 0 and sc < w:
                inb = 1
            scr = wp.clamp(sr, 0, h - 1)
            scc = wp.clamp(sc, 0, w - 1)
            if inb == 0 or blocked[scr, scc, t] > 0.5:
                ok = 0
            tsum += penalty[scr, scc, t]
        if ok == 1:
            nr = r + prim_dr[t, p]
            nc = c + prim_dc[t, p]
            if nr >= 0 and nr < h and nc >= 0 and nc < w:
                arc = prim_cost[t, p]
                if ns > 0:
                    arc = arc * (1.0 + penalty_weight * tsum / float(ns))
                best = wp.min(best, arc + dist_in[nr, nc, prim_heading[t, p]])
    dist_out[r, c, t] = best
    if best < dist_in[r, c, t]:
        changed[0] = 1


class ValueSolver:
    def __init__(
        self,
        resolution: float,
        height: int,
        width: int,
        n_theta: int = 16,
        turn_radius: float = 0.6,
        step: float | None = None,
        pivot_cost: float = 0.0,  # [m-equiv] per heading bin; > 0 adds point-turn primitives
        control_set: tuple | None = None,  # from control_set.py; None builds forward arcs
        device: wp.Device | None = None,
    ):
        self.resolution = resolution
        self.height = height
        self.width = width
        self.n_theta = n_theta
        self.device = wp.get_device(device)
        self._inf = 1.0e30
        self._step = float(step) if step is not None else 2.0 * self.resolution

        # a single arc can sweep ~step/resolution cells; size the swept-cell buffer + arc sampling
        # to that ratio so fine grids don't truncate the collision check and jump thin walls.
        if control_set is None:
            step_cells = self._step / self.resolution
            max_sweep = max(6, int(math.ceil(step_cells)) * 2 + 3)
            nseg = max(8, int(step_cells * 4))
            control_set = arc_control_set(
                self.n_theta,
                self.resolution,
                self._step,
                float(turn_radius),
                max_sweep,
                nseg,
                pivot_cost=float(pivot_cost),
            )
        n_prim, prim_dr, prim_dc, prim_heading, prim_cost, sweep_dr, sweep_dc, sweep_n = control_set
        self.n_prim = n_prim
        # motion-primitive table on device, indexed [heading_bin, primitive]: where each forward arc
        # lands + what it crosses (see _build_primitives). The relax kernel reads these every sweep.
        with wp.ScopedDevice(self.device):
            self._prim_dr = wp.array(prim_dr, dtype=wp.int32)  # endpoint row offset of the arc
            self._prim_dc = wp.array(prim_dc, dtype=wp.int32)  # endpoint col offset
            # heading bin the arc ends at
            self._prim_heading = wp.array(prim_heading, dtype=wp.int32)
            # arc length (the move's base cost)
            self._prim_cost = wp.array(prim_cost, dtype=wp.float32)
            # row offsets of the cells the arc crosses
            self._sweep_dr = wp.array(sweep_dr, dtype=wp.int32)
            self._sweep_dc = wp.array(sweep_dc, dtype=wp.int32)  # col offsets of those swept cells
            self._sweep_n = wp.array(sweep_n, dtype=wp.int32)  # how many swept cells each arc has
            # two value buffers, ping-ponged each sweep (read one, write the other, swap); +changed flag
            self._dist_a = wp.zeros((self.height, self.width, self.n_theta), dtype=wp.float32)
            self._dist_b = wp.zeros((self.height, self.width, self.n_theta), dtype=wp.float32)
            # >0 if any cell improved this sweep (convergence)
            self._changed = wp.zeros(1, dtype=wp.int32)
            # device while-condition for capture_while
            self._keep_running = wp.zeros(1, dtype=wp.int32)
            self._iter = wp.zeros(1, dtype=wp.int32)
        self._cap = self.height + self.width  # max bodies (each = 2 sweeps -> 2*(h+w) sweeps total)

    def _relax(
        self,
        dist_in: wp.array,
        dist_out: wp.array,
        blocked: wp.array,
        penalty: wp.array,
        penalty_weight: float,
    ) -> None:
        """One min-relaxation sweep dist_in -> dist_out (race-free pull); raises self._changed if any
        cell improved."""
        wp.launch(
            _relax_kernel,
            dim=(self.height, self.width, self.n_theta),
            inputs=[
                dist_in,
                blocked,
                penalty,
                self._prim_dr,
                self._prim_dc,
                self._prim_heading,
                self._prim_cost,
                self._sweep_dr,
                self._sweep_dc,
                self._sweep_n,
                self.n_prim,
                float(penalty_weight),
                self._inf,
            ],
            outputs=[dist_out, self._changed],
            device=self.device,
        )

    def _record_solve(
        self,
        blocked: wp.array,
        penalty: wp.array,
        seeds: wp.array,
        penalty_weight: float,
        capture: bool,
    ) -> wp.array:
        """Record the device-side value iteration: free the seeds, initialise, then min-relax to a fixed
        point. The convergence loop runs ON DEVICE via capture_while (graph-safe) -- a 2-SWEEP body
        (a->b->a) keeps the ping-pong buffers fixed inside the captured graph, so the result always
        lands in self._dist_a. capture=False uses an eager host while-loop (CPU / no-graph fallback).
        """
        grid_dim = (self.height, self.width, self.n_theta)

        wp.launch(
            _free_seeds_kernel,
            dim=grid_dim,
            inputs=[seeds],
            outputs=[blocked],
            device=self.device,
        )
        wp.launch(
            _init_kernel,
            dim=grid_dim,
            inputs=[seeds, self._inf],
            outputs=[self._dist_a],
            device=self.device,
        )
        self._keep_running.fill_(1)
        self._iter.zero_()

        def body() -> None:
            self._changed.zero_()
            self._relax(self._dist_a, self._dist_b, blocked, penalty, penalty_weight)
            self._relax(self._dist_b, self._dist_a, blocked, penalty, penalty_weight)
            wp.launch(
                _keep_going_kernel,
                dim=1,
                inputs=[self._changed, self._iter, self._cap, self._keep_running],
                device=self.device,
            )

        if capture:
            wp.capture_while(self._keep_running, body)
        else:
            while True:
                body()
                if int(self._keep_running.numpy()[0]) == 0:
                    break
        return self._dist_a
