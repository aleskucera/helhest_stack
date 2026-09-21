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

Two deliberate differences from the fine layer, both of which make this layer STRICTLY more
permissive. That is the property that stops the two from disagreeing in a loop: a route the
coarse layer promises and the fine layer then refuses makes the robot oscillate between them.

  No settle. A 1.0 m cell is smaller than the robot, so placing a 1.5 m chassis on one and
  solving its contacts says nothing the pooling has not already said. Feasibility here is a step
  test: a cell is blocked when the height difference to a neighbour exceeds what the robot can
  climb. It answers "is there a way through", not "can it stand there".

  Unmeasured is not blocked. Ignorance about ground the robot has not reached is not a reason to
  route around it -- deciding whether that ignorance is a problem is the fine layer's job, once
  the robot is close enough to have measured it. This is the `certain=True` reading in
  `terrain_value_field.hierarchical`, in the one place it belongs.

The pooling is a MAX over each block, so an obstacle is never averaged away by the ground around
it. Combined with the step test that makes the layer pessimistic about obstacles and optimistic
about ignorance, which is the right way round for both.
"""

from __future__ import annotations

import numpy as np
import warp as wp
from terrain_value_field import omni_control_set
from terrain_value_field.solver import ValueSolver

from ..engine import GridParams


@wp.kernel
def _pool_kernel(
    elevation: wp.array2d(dtype=wp.float32),  # fine [ny, nx]
    measured: wp.array2d(dtype=wp.float32),  # fine [ny, nx], 1 = observed
    factor: wp.int32,
    pooled: wp.array2d(dtype=wp.float32),  # coarse [cy, cx]
    seen: wp.array2d(dtype=wp.float32),  # coarse [cy, cx], 1 = any fine cell observed
):
    """MAX-pool `factor` x `factor` blocks. Max, not mean: a mean lets a block of ground average
    an obstacle away, and this layer is the one that decides which side of it to pass."""
    r, c = wp.tid()
    ny = elevation.shape[0]
    nx = elevation.shape[1]
    hi = float(-1.0e30)
    any_seen = float(0.0)
    for dr in range(factor):
        for dc in range(factor):
            fr = r * factor + dr
            fc = c * factor + dc
            if fr < ny and fc < nx:
                if measured[fr, fc] > 0.5:
                    hi = wp.max(hi, elevation[fr, fc])
                    any_seen = 1.0
    pooled[r, c] = wp.where(any_seen > 0.5, hi, 0.0)
    seen[r, c] = any_seen


@wp.kernel
def _step_cost_kernel(
    pooled: wp.array2d(dtype=wp.float32),  # coarse [cy, cx]
    seen: wp.array2d(dtype=wp.float32),  # coarse [cy, cx]
    max_step: wp.float32,  # [m] the largest rise the robot can climb
    pose_cost: wp.array3d(dtype=wp.float32),  # coarse [cy, cx, 1]
):
    """Block a cell when the step to any 8-neighbour exceeds what the robot can climb.

    Sign convention is the solver's and is shared with `costtogo._pose_cost_kernel`: the veto
    rides in the sign, so a free pose is `+penalty` and a vetoed one `-1 - penalty`. This layer
    grades nothing, so the penalty is zero and the values are exactly 0.0 and -1.0.

    A step is only counted between two MEASURED cells. Against an unmeasured neighbour there is
    no step to measure -- the pooled height there is a fill, and treating a fill as ground would
    manufacture a cliff at the edge of the map and wall the window off in a ring.
    """
    r, c, _t = wp.tid()
    rows = pooled.shape[0]
    cols = pooled.shape[1]
    blocked = float(0.0)
    if seen[r, c] > 0.5:
        h = pooled[r, c]
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr = r + dr
                cc = c + dc
                if rr >= 0 and rr < rows and cc >= 0 and cc < cols:
                    if seen[rr, cc] > 0.5:
                        if wp.abs(pooled[rr, cc] - h) > max_step:
                            blocked = 1.0
    pose_cost[r, c, 0] = wp.where(blocked > 0.5, -1.0, 0.0)


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

    `factor` is how many fine cells go into one coarse cell. `max_step_m` is the rise the robot
    can climb; the tip-over envelope is what makes a slope impassable in the fine layer, and this
    is its blunt equivalent -- deliberately blunt, because this layer must never be the stricter
    of the two.
    """

    def __init__(
        self,
        fine_grid: GridParams,
        factor: int = 5,
        max_step_m: float = 0.25,
        device: wp.Device | str | None = None,
    ) -> None:
        if factor < 1:
            raise ValueError(f"factor must be >= 1 fine cells per coarse cell, got {factor}")
        if max_step_m <= 0.0:
            raise ValueError(f"max_step_m must be > 0, got {max_step_m}")
        self.device = wp.get_device(device)
        self.factor = int(factor)
        self.max_step_m = float(max_step_m)
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
        with wp.ScopedDevice(self.device):
            self.pooled = wp.zeros((cy, cx), dtype=wp.float32)
            self.seen = wp.zeros((cy, cx), dtype=wp.float32)
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
            _pool_kernel,
            dim=self.pooled.shape,
            inputs=[elevation, measured, self.factor],
            outputs=[self.pooled, self.seen],
            device=self.device,
        )
        wp.launch(
            _step_cost_kernel,
            dim=self._pose_cost.shape,
            inputs=[self.pooled, self.seen, self.max_step_m],
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
        self.V = self.solver.value_iterate(self._pose_cost, self._seeds, 0.0)
        return self.V
