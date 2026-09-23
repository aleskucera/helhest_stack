"""Which way round, decided where that is the only question worth asking.

The fine router answers "can the robot stand here", pose by pose, from the settle. That answer is
only meaningful on ground the robot has actually measured, and Odin's dToF measures 4 m ahead
completely, 74-85% at 6 m, 41-49% at 8 m and 12-14% at 12 m (`out_odin0`, 141 m driven). Past
that a finer window is not planning over terrain, it is planning over whatever filled the
unobserved cells.

Coarsening changes that, because coverage is a question of whether ANY return landed in a cell.
Pooled to 1.0 m the same map is 85% known at 10 m where the 0.2 m grid is 26%. So the coarse
layer can see the shape of the world at a range where the fine layer cannot see anything -- which
is the range at which "go left or go right" is decided.

This layer must be STRICTLY more permissive than the fine one. That is the property that stops
the two disagreeing in a loop: a route the coarse layer promises and the fine layer then refuses
makes the robot oscillate between them. Everything below follows from it, and the first version
of this file got each point backwards, so each carries the measurement that corrected it.

  It pools PASSABILITY, not height, and by FRACTION. The question here is "is there room to
  cross this block", which neither extreme answers. Pooling height with MAX answers the fine
  layer's question -- "how bad is the worst thing in this block" -- and seals every passage
  narrower than a cell or two: measured on the `gap` world, whose wall has a 1.8 m opening, MAX
  read that opening as sealed in every frame from the sixteenth on, priced the far side at 3.4 m
  and the robot's own side at 18.5, and left the two unconnected. Pooling with ANY overshoots the
  other way: a 0.4 m wall inside a 1.0 m block leaves climbable ground beside it, so the block
  reads open and the layer routes straight through a wall spanning the world.

  So a block is passable when at least `min_pass_fraction` of its measured cells are climbable.
  That is a proxy for width, and it is the honest one available: a block is crossable when most
  of it is drivable. On the measured cases it separates cleanly -- a 1.8 m doorway pools near
  1.0, a block straddling a 0.4 m wall pools near 0.4. It is a heuristic, and the reason a
  heuristic is needed is that a per-CELL cost cannot express "you may stand in this block but not
  cross it"; saying that properly needs per-EDGE feasibility, which the solver does not take.

  Passability is a property of a cell and its own neighbours, never of the block next door. The
  version that vetoed a block when a NEIGHBOURING block differed in height by more than a step
  blocked everything within a metre of an obstacle, so a corridor needed about 3 m of clearance
  before its centre survived. No 1.8 m gap passes that at any grid alignment.

  No settle. A 1.0 m cell is smaller than the robot, so placing a 1.5 m chassis on one and
  solving its contacts says nothing the pooling has not already said. A fine cell is climbable
  when the step to its immediate neighbours is one a wheel could drive up. That is a smaller
  effective footprint than the fine layer's, deliberately: this layer is allowed to promise a gap
  the robot then turns out not to fit through, and not allowed to hide one it would have fitted.

  Unmeasured is not blocked, but past a frontier it is not free either. Ignorance about ground
  the robot has not reached is no reason to route around it -- that is the fine layer's job once
  it arrives. Taken without limit, though, it lets this layer route through terrain that does not
  exist: on `gap`, 748 of 818 routable cells had never been measured and 483 of those lay outside
  the world's own 16 x 10 m extent, so it costed an 18 m detour around the OUTSIDE of the world
  and called it a route. Cells within `frontier_m` of measured ground stay free; past that each
  costs `void_penalty`. Still reachable -- a goal in terrain nobody has seen must stay reachable
  or the robot will never go and look at it -- but no longer cheaper than the real way round.
"""

from __future__ import annotations

import numpy as np
import warp as wp
from terrain_value_field import omni_control_set
from terrain_value_field.solver import ValueSolver

from ..engine import GridParams


@wp.kernel
def _climb_kernel(
    elevation: wp.array2d(dtype=wp.float32),  # fine [ny, nx]
    measured: wp.array2d(dtype=wp.float32),  # fine [ny, nx], 1 = observed
    max_step: wp.float32,  # [m] the largest rise a wheel can drive up
    climbable: wp.array2d(dtype=wp.float32),  # fine [ny, nx], 1 = a wheel could cross it
):
    """A fine cell is climbable when the step to its immediate neighbours is one the robot can
    drive up.

    Unmeasured cells, and unmeasured neighbours, are skipped rather than counted: the height
    there is a fill, and treating a fill as ground manufactures a cliff at the edge of the map.
    """
    r, c = wp.tid()
    if measured[r, c] < 0.5:
        climbable[r, c] = 0.0
        return
    rows = elevation.shape[0]
    cols = elevation.shape[1]
    h = elevation[r, c]
    ok = float(1.0)
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            rr = r + dr
            cc = c + dc
            if rr >= 0 and rr < rows and cc >= 0 and cc < cols:
                if measured[rr, cc] > 0.5:
                    if wp.abs(elevation[rr, cc] - h) > max_step:
                        ok = 0.0
    climbable[r, c] = ok


@wp.kernel
def _pool_kernel(
    elevation: wp.array2d(dtype=wp.float32),  # fine [ny, nx]
    measured: wp.array2d(dtype=wp.float32),  # fine [ny, nx]
    climbable: wp.array2d(dtype=wp.float32),  # fine [ny, nx]
    factor: wp.int32,
    passable: wp.array2d(dtype=wp.float32),  # coarse [cy, cx], climbable FRACTION in 0..1
    seen: wp.array2d(dtype=wp.float32),  # coarse [cy, cx], 1 = some fine cell observed
    floor: wp.array2d(dtype=wp.float32),  # coarse [cy, cx], lowest measured ground in the block
):
    """What fraction of the block's MEASURED cells a wheel could drive over.

    Over the measured ones, not all of them, so a half-observed block is not marked impassable
    for the half nobody has looked at -- that half's price is the frontier's business.

    `floor` is the MINIMUM measured height, and is for looking at rather than for deciding --
    the decision is `passable`. Minimum because the question a floor answers is "what is the
    ground under this block", and a block holding an obstacle still has ground beside it.
    """
    r, c = wp.tid()
    ny = elevation.shape[0]
    nx = elevation.shape[1]
    lo = float(1.0e30)
    n_seen = float(0.0)
    n_climb = float(0.0)
    for dr in range(factor):
        for dc in range(factor):
            fr = r * factor + dr
            fc = c * factor + dc
            if fr < ny and fc < nx:
                if measured[fr, fc] > 0.5:
                    lo = wp.min(lo, elevation[fr, fc])
                    n_seen += 1.0
                    if climbable[fr, fc] > 0.5:
                        n_climb += 1.0
    passable[r, c] = wp.where(n_seen > 0.0, n_climb / n_seen, 0.0)
    seen[r, c] = wp.where(n_seen > 0.0, 1.0, 0.0)
    floor[r, c] = wp.where(n_seen > 0.0, lo, 0.0)


@wp.kernel
def _cost_kernel(
    passable: wp.array2d(dtype=wp.float32),  # coarse [cy, cx], climbable fraction
    seen: wp.array2d(dtype=wp.float32),  # coarse [cy, cx]
    min_pass: wp.float32,  # climbable fraction a block needs to count as crossable
    frontier: wp.int32,  # [cells] how far past measured ground stays free
    void_penalty: wp.float32,  # [m] charged per cell beyond that
    pose_cost: wp.array3d(dtype=wp.float32),  # coarse [cy, cx, 1]
):
    """Pack passability into the one signed field the solver reads.

    Sign convention is the solver's, shared with `costtogo._pose_cost_kernel`: the veto rides in
    the sign, so a free cell is `+penalty` and a vetoed one `-1 - penalty`. Measured ground is
    free or vetoed outright. Unmeasured ground is free near the frontier and priced beyond it,
    never vetoed -- a goal in terrain nobody has seen has to stay reachable, or the robot will
    not go and look at it.
    """
    r, c, _t = wp.tid()
    if seen[r, c] > 0.5:
        pose_cost[r, c, 0] = wp.where(passable[r, c] >= min_pass, 0.0, -1.0)
        return
    rows = seen.shape[0]
    cols = seen.shape[1]
    near = float(0.0)
    for dr in range(-frontier, frontier + 1):
        for dc in range(-frontier, frontier + 1):
            rr = r + dr
            cc = c + dc
            if rr >= 0 and rr < rows and cc >= 0 and cc < cols:
                if seen[rr, cc] > 0.5:
                    near = 1.0
    pose_cost[r, c, 0] = wp.where(near > 0.5, 0.0, void_penalty)


@wp.kernel
def _seed_goal_kernel(
    goal_rc: wp.array(dtype=wp.int32),  # [2]
    inf: wp.float32,
    seeds: wp.array3d(dtype=wp.float32),  # coarse [cy, cx, 1]
):
    r, c, t = wp.tid()
    seeds[r, c, t] = wp.where(r == goal_rc[0] and c == goal_rc[1], 0.0, inf)


@wp.kernel
def _goal_cell_kernel(
    goal_xy: wp.array(dtype=wp.float32),  # [2], in the coarse grid's own frame
    origin_x: wp.float32,
    origin_y: wp.float32,
    cell_size: wp.float32,
    rows: wp.int32,
    cols: wp.int32,
    goal_rc: wp.array(dtype=wp.int32),  # [2]
):
    """Resolve and CLAMP the goal into the window, so a goal beyond it becomes a carrot at the
    edge rather than no goal at all."""
    c = int((goal_xy[0] - origin_x) / cell_size)
    r = int((goal_xy[1] - origin_y) / cell_size)
    goal_rc[0] = wp.clamp(r, 0, rows - 1)
    goal_rc[1] = wp.clamp(c, 0, cols - 1)


class CoarseRouter:
    """A heading-free cost-to-go over the whole window, at a cell size where coverage is good.

    `factor` is how many fine cells go into one coarse cell and `max_step_m` the rise a wheel can
    drive up. `min_pass_fraction` is how much of a block must be climbable before it counts as
    crossable. `frontier_m` is how far past measured ground stays free, and `void_penalty` what
    each cell costs beyond it -- in metres, so 1.0 doubles the price of crossing a metre of
    terrain nobody has looked at.

    DO NOT TUNE `min_pass_fraction`. It was carried as a heuristic standing in for per-edge
    feasibility, i.e. as an untuned risk. Measured across the six stress worlds it is a knob that
    moves everything except the answer: swept 0.1 -> 0.9 it takes blocked coarse cells from ~1.5%
    to ~13% and shifts this field by up to 48 m on `pocket`, and the closed loop changes by under
    2%, non-monotonically -- 1476 / 1504 / 1473 total frames at 0.1 / 0.5 / 0.9, all 6/6. What
    this layer contributes is a coarse "which way out of the routing window", and that survives
    a wholesale change of opinion about which blocks are passable. So the stand-in does not need
    replacing with per-edge feasibility; it needs leaving alone.

    The insensitivity is measured on stress worlds, whose coverage is good. On real maps -- far
    patchier, much more of the window unmeasured -- it may bind, and `frontier_m`/`void_penalty`
    (which decide what unseen ground costs) are the more likely levers there anyway.

    The layer as a whole DOES earn its place: driving all six worlds with it off (`--coarsen 0`)
    still reaches 6/6, but costs +4.2% frames overall and +15% on `pillars`, +11% on `ridge` --
    the two worlds where "which way round" actually binds. Pooling at `factor` 1 rather than 5
    buys nothing and costs 2.0 ms a frame (0.31 -> 2.32 ms), so the pooling stays too.
    """

    def __init__(
        self,
        fine_grid: GridParams,
        factor: int = 5,
        max_step_m: float = 0.25,
        min_pass_fraction: float = 0.5,
        frontier_m: float = 3.0,
        void_penalty: float = 1.0,
        device: wp.Device | str | None = None,
    ) -> None:
        if factor < 1:
            raise ValueError(f"factor must be >= 1 fine cells per coarse cell, got {factor}")
        if max_step_m <= 0.0:
            raise ValueError(f"max_step_m must be > 0, got {max_step_m}")
        if frontier_m < 0.0 or void_penalty < 0.0:
            raise ValueError(
                f"frontier_m and void_penalty must be >= 0, got {frontier_m}, {void_penalty}"
            )
        self.device = wp.get_device(device)
        self.factor = int(factor)
        self.max_step_m = float(max_step_m)
        self.min_pass_fraction = float(min_pass_fraction)
        self.void_penalty = float(void_penalty)
        # ceil, so the coarse grid covers the fine one even when it does not divide evenly
        cy = (fine_grid.cells_y + self.factor - 1) // self.factor
        cx = (fine_grid.cells_x + self.factor - 1) // self.factor
        self.grid = GridParams(
            cells_x=cx,
            cells_y=cy,
            cell_size=fine_grid.cell_size * self.factor,
            origin_x=fine_grid.origin_x,
            origin_y=fine_grid.origin_y,
        )
        self.frontier = int(round(float(frontier_m) / self.grid.cell_size))
        with wp.ScopedDevice(self.device):
            self._climb = wp.zeros((fine_grid.cells_y, fine_grid.cells_x), dtype=wp.float32)
            self.passable = wp.zeros((cy, cx), dtype=wp.float32)
            self.seen = wp.zeros((cy, cx), dtype=wp.float32)
            self.floor = wp.zeros((cy, cx), dtype=wp.float32)
            self._pose_cost = wp.zeros((cy, cx, 1), dtype=wp.float32)
            self._seeds = wp.zeros((cy, cx, 1), dtype=wp.float32)
            self._goal_rc = wp.zeros(2, dtype=wp.int32)
            self._goal_xy = wp.zeros(2, dtype=wp.float32)
        self.solver = ValueSolver(
            self.grid.cell_size,
            cy,
            cx,
            n_theta=1,
            control_set=omni_control_set(self.grid.cell_size),
            device=self.device,
        )
        self.V = wp.zeros((cy, cx, 1), dtype=wp.float32, device=self.device)

    def solve(
        self,
        elevation: wp.array,
        measured: wp.array,
        goal_xy: tuple[float, float],
    ) -> wp.array:
        """Fine `elevation` and `measured` [ny, nx] -> coarse V [cy, cx, 1], device-resident.

        `goal_xy` is in the fine grid's frame, which this layer shares by construction.
        """
        self._goal_xy.assign(np.asarray(goal_xy[:2], np.float32))
        wp.launch(
            _climb_kernel,
            dim=self._climb.shape,
            inputs=[elevation, measured, self.max_step_m],
            outputs=[self._climb],
            device=self.device,
        )
        wp.launch(
            _pool_kernel,
            dim=self.passable.shape,
            inputs=[elevation, measured, self._climb, self.factor],
            outputs=[self.passable, self.seen, self.floor],
            device=self.device,
        )
        wp.launch(
            _cost_kernel,
            dim=self._pose_cost.shape,
            inputs=[
                self.passable,
                self.seen,
                self.min_pass_fraction,
                self.frontier,
                self.void_penalty,
            ],
            outputs=[self._pose_cost],
            device=self.device,
        )
        wp.launch(
            _goal_cell_kernel,
            dim=1,
            inputs=[
                self._goal_xy,
                self.grid.origin_x,
                self.grid.origin_y,
                self.grid.cell_size,
                self.grid.cells_y,
                self.grid.cells_x,
            ],
            outputs=[self._goal_rc],
            device=self.device,
        )
        wp.launch(
            _seed_goal_kernel,
            dim=self._seeds.shape,
            inputs=[self._goal_rc, self.solver._inf],
            outputs=[self._seeds],
            device=self.device,
        )
        # penalty_scale 1.0: the void penalty is already in metres, so it adds to the move's own
        # length as itself rather than being weighted a second time
        self.V = self.solver.value_iterate(self._pose_cost, self._seeds, 1.0)
        return self.V
