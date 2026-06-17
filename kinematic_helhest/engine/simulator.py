"""Preallocated forward-simulation context.

Build once from (robot params, solver params, grid spec, B, T); then `set_terrain`,
`set_uniform_friction`, and `rollout` reuse every device buffer — no allocation in the loop.
Replaces the old planning-side `BatchRollout`: the engine now owns the simulation
state (the rollout buffers + the terrain/envelope/friction grids), and the planner
just feeds controls in.

The grid (nx, ny, cell, origin) is FIXED at construction. In the robot-centered
rolling map the window dimensions never change — only the terrain *values* do — so
each cycle overwrites the owned buffers in place.

Forward-only for now (no requires_grad / tape); the differentiable calibration path
keeps its own buffers in tests/engine/gradients.py.
"""
import numpy as np
import warp as wp
from warp import Device

from .envelope import _contact_kernel
from .envelope import _gather_kernel
from .robot import RobotParams
from .step import init_state
from .step import SolverParams
from .step import step as wstep
from .terrain import GridParams


class Simulator:
    def __init__(
        self,
        robot_params: RobotParams,
        solver_params: SolverParams,
        grid_params: GridParams,
        B: int,  # Batch size
        T: int,  # Number of timesteps
        device: Device | str | None = None,
    ):

        self.device = wp.get_device(device)

        self.B = B
        self.T = T

        self.robot = robot_params.build(device)  # device Robot struct
        self.solver = solver_params.build()  # device Solver struct
        self.grid = grid_params.build()  # Grid (fixed)
        self.wheel_radius = grid_params.R
        self.env_radius = int(np.ceil(grid_params.R / grid_params.cell))
        ny, nx = grid_params.ny, grid_params.nx

        with wp.ScopedDevice(self.device):
            # terrain: owned envelope + friction, plus the arg-max scratch; raw is borrowed.
            self.elevation = wp.zeros((ny, nx), dtype=wp.float32)
            self.envelope = wp.zeros((ny, nx), dtype=wp.float32)
            self.friction = wp.zeros((ny, nx), dtype=wp.float32)
            self._contact_iy = wp.zeros((ny, nx), dtype=wp.int32)
            self._contact_ix = wp.zeros((ny, nx), dtype=wp.int32)
            self._cap = wp.zeros((ny, nx), dtype=wp.float32)

            # rollout buffers + control inputs, allocated ONCE.
            self.planar = wp.zeros((T + 1, B), dtype=wp.vec3f)
            self.tilt = wp.zeros((T + 1, B), dtype=wp.vec3f)
            self.loads = wp.zeros((T, B), dtype=wp.vec3f)
            self.turning = wp.zeros((T, B), dtype=wp.vec2f)
            self.clearance = wp.zeros((T, B), dtype=wp.float32)
            self.residual = wp.zeros((T, B), dtype=wp.float32)
            self.omega = wp.zeros((T, B), dtype=wp.vec3f)
            self.start_pose = wp.zeros(B, dtype=wp.vec3f)

    def set_terrain(self, elevation: wp.array):
        wp.copy(self.elevation, elevation)

        wp.launch(
            kernel=_contact_kernel,
            dim=self.elevation.shape,
            inputs=[
                self.elevation,
                self.grid.cell,
                self.wheel_radius,
                self.env_radius,
            ],
            outputs=[
                self._contact_iy,
                self._contact_ix,
                self._cap,
            ],
            device=self.device,
        )

        wp.launch(
            kernel=_gather_kernel,
            dim=self.elevation.shape,
            inputs=[
                self.elevation,
                self._contact_iy,
                self._contact_ix,
                self._cap,
            ],
            outputs=[self.envelope],
            device=self.device,
        )

    def set_uniform_friction(self, value):
        """Uniform friction: overwrite the owned friction grid in place."""
        self.friction.fill_(float(value))

    def set_friction(self, friction_hm):
        """Per-cell friction from a numpy Heightmap matching the grid (copied in place)."""
        self.friction.assign(np.ascontiguousarray(friction_hm.H, np.float32))

    @classmethod
    def for_scene(cls, robot_params, solver_params, scene, mu, B, T, device="cuda"):
        """Convenience for a static numpy scene: size the grid from `scene`, upload it,
        dilate, set friction. (The non-preallocated, build-once use.)"""
        sim = cls(
            robot_params,
            solver_params,
            GridParams.from_heightmap(scene, R=robot_params.wheel_radius),
            B,
            T,
            device,
        )
        sim.set_terrain(
            wp.array(np.ascontiguousarray(scene.H, np.float32), dtype=wp.float32, device=device)
        )
        sim.set_friction(mu)
        return sim

    def rollout(self, omega_np, init_pose):
        """omega_np [T, B, 3], init_pose (x,y,yaw) shared by all rollouts. Returns
        planar [T+1,B,3] (x,y,yaw), tilt [T+1,B,3] (z,pitch,roll), clear/resid [T,B]."""
        self.omega.assign(np.ascontiguousarray(omega_np, np.float32))
        self.start_pose.assign(
            np.ascontiguousarray(
                np.tile(np.asarray(init_pose, np.float32), (self.B, 1)), np.float32
            )
        )
        wp.launch(
            init_state,
            self.B,
            inputs=[self.envelope, self.grid, self.robot, self.solver, self.start_pose],
            outputs=[self.planar, self.tilt],
            device=self.device,
        )
        for t in range(self.T):
            wp.launch(
                kernel=wstep,
                dim=self.B,
                inputs=[
                    t,
                    self.envelope,
                    self.elevation,
                    self.grid,
                    self.friction,
                    self.grid,
                    self.robot,
                    self.solver,
                    self.omega,
                ],
                outputs=[
                    self.planar,
                    self.tilt,
                    self.loads,
                    self.turning,
                    self.clearance,
                    self.residual,
                ],
                device=self.device,
            )
        return self.planar.numpy(), self.tilt.numpy(), self.clearance.numpy(), self.residual.numpy()
