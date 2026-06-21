"""Settle-based orientation-aware cost-to-go V(x, y, theta).

Feasibility comes from the robot's OWN settle, not a thresholded traversability map: for every pose
(x, y, theta) the robot is placed on the terrain and the engine's residual / clearance / tilt are
read. A pose is blocked iff residual > resid_tol OR clearance < clear_margin OR tilt > tilt_max --
the SAME validity the MPPI rollouts use, so the cost-to-go and the rollouts agree by construction
(no arbitrary obstacle threshold). tilt is the graded arc cost (prefer flat). So walls block because
the robot can't sit on their face, and rough terrain is costly-but-passable rather than a false
obstacle. The static (zero-control) settle is friction-independent, so compute() needs only the
elevation and the goal.

Self-contained on purpose: this is the cost-to-go we keep; the traversability/2D variants in
costtogo.py are on their way out.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

if TYPE_CHECKING:
    import warp.context

    from ..engine import GridParams
    from ..engine import RobotParams
    from ..engine import SolverParams


@wp.kernel
def _clamp3d_kernel(
    v_in: wp.array3d(dtype=wp.float32), vcap: wp.float32, v_out: wp.array3d(dtype=wp.float32)
):
    """Copy V, replacing the solver's +inf (unreachable) with a large finite cap so the cost
    kernel's trilinear sampling never blends to inf."""
    r, c, t = wp.tid()
    val = v_in[r, c, t]
    if val > vcap:
        v_out[r, c, t] = vcap
    else:
        v_out[r, c, t] = val


class CostToGoLatticeSettle:
    """Orientation-aware cost-to-go with feasibility from the robot's settle (see module docstring).
    compute(elevation, goal) -> clamped V[ny, nx, n_theta]."""

    def __init__(
        self,
        grid: GridParams,
        robot_params: RobotParams,
        solver_params: SolverParams,
        device: wp.context.Device | str | None = None,
        n_theta: int = 24,
        turn_radius: float = 0.5,
        robot_radius: float = 0.3,
        step: float = 0.3,
        resid_tol: float = 1e-2,
        clear_margin: float = 0.05,
        tilt_max_deg: float = 40.0,
        tilt_weight: float = 2.0,
    ) -> None:
        # robot_params / solver_params are MANDATORY (like grid): the caller passes the same vehicle
        # it gave the planner, so the cost-to-go settles exactly the robot the rollouts drive -- no
        # silent fallback that could quietly disagree with the planner.
        try:
            from terrain_toolkit import LatticeValueSolver
        except ImportError as e:
            raise ImportError(
                "orientation-aware cost-to-go needs terrain_toolkit; install it, e.g. "
                "`uv pip install -e ../terrain_toolkit --no-deps`"
            ) from e
        from ..engine import Simulator
        from ..heightmap import Heightmap

        self.nx, self.ny, self.cell = int(grid.cells_x), int(grid.cells_y), float(grid.cell_size)
        self.x0, self.y0 = float(grid.origin_x), float(grid.origin_y)
        self.bounds = (self.x0, self.x0 + self.nx * self.cell, self.y0, self.y0 + self.ny * self.cell)
        self.n_theta = int(n_theta)
        self.device = wp.get_device(device)  # resolve None -> default once, reuse everywhere
        self.resid_tol, self.clear_margin = float(resid_tol), float(clear_margin)
        self.tilt_max = float(np.radians(tilt_max_deg))
        self.tilt_weight = float(tilt_weight)
        self._vcap = float(1.5 * (self.nx + self.ny) * self.cell * (1.0 + tilt_weight))
        # world coords of every cell center -> the pose grid we settle (heading added per bin)
        cols, rows = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        self._X = (self.x0 + cols * self.cell).ravel().astype(np.float32)
        self._Y = (self.y0 + rows * self.cell).ravel().astype(np.float32)
        self.settle_sim = Simulator(robot_params, solver_params, grid, self.nx * self.ny, 1, self.device)
        # the zero-control settle is a static resting-pose solve -> friction-independent (verified
        # bit-identical across mu). The sim just needs SOME friction array set; a dummy uniform does.
        self._mu = Heightmap(np.full((self.ny, self.nx), 0.8, np.float32), (self.x0, self.y0), self.cell)
        self.solver = LatticeValueSolver(
            self.cell, self.ny, self.nx, n_theta=self.n_theta,
            turn_radius=turn_radius, robot_radius=robot_radius, step=step, device=self.device,
        )
        self.V = wp.zeros((self.ny, self.nx, self.n_theta), dtype=wp.float32, device=self.device)

    def _settle_fields(self, elevation: np.ndarray) -> tuple[wp.array, wp.array]:
        """Settle every pose; return blocked[ny,nx,n_theta], tilt[ny,nx,n_theta] (rad) as wp.arrays."""
        sim, B = self.settle_sim, self.nx * self.ny
        sim.set_terrain(wp.array(np.ascontiguousarray(elevation, np.float32), dtype=wp.float32, device=self.device))
        sim.set_friction(self._mu)
        blocked = np.zeros((self.ny, self.nx, self.n_theta), np.float32)
        tilt = np.zeros((self.ny, self.nx, self.n_theta), np.float32)
        two_pi = 2.0 * np.pi
        for t in range(self.n_theta):
            th = (float(t) + 0.5) * two_pi / float(self.n_theta)  # bin-center heading
            sim.start_pose.assign(np.stack([self._X, self._Y, np.full(B, th, np.float32)], 1))
            sim.omega.zero_()
            sim.rollout_launch()
            der = sim.derived.numpy()[0]  # (z, pitch, roll) settled at each pose
            res = sim.residual.numpy()[0]
            clr = sim.clearance.numpy()[0]
            ti = np.arccos(np.clip(np.cos(der[:, 1]) * np.cos(der[:, 2]), -1.0, 1.0))
            blk = (res > self.resid_tol) | (clr < self.clear_margin) | (ti > self.tilt_max)
            blocked[:, :, t] = blk.reshape(self.ny, self.nx)
            tilt[:, :, t] = ti.reshape(self.ny, self.nx)
        return (wp.array(blocked, dtype=wp.float32, device=self.device),
                wp.array(tilt, dtype=wp.float32, device=self.device))

    def compute(self, elevation: np.ndarray, goal_xy: np.ndarray) -> wp.array:
        """elevation [ny, nx] + world goal -> clamped V[ny, nx, n_theta] (no friction needed)."""
        blocked, tilt = self._settle_fields(elevation)
        v = self.solver.compute_from_fields(blocked, tilt, goal_xy, self.bounds, tilt_weight=self.tilt_weight)
        wp.launch(_clamp3d_kernel, dim=(self.ny, self.nx, self.n_theta),
                  inputs=[v, self._vcap], outputs=[self.V], device=self.device)
        return self.V
