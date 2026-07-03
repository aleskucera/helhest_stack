"""Preallocated simulation contexts: a forward-only planner sim and a differentiable calibration sim.

Build once from (robot params, solver params, grid spec, batch_size, n_steps); then `set_terrain`,
`set_friction`, and the rollout reuse every device buffer -- no allocation in the loop. The grid
(cells_x, cells_y, cell_size, origin) is FIXED at construction; in the robot-centered rolling map the
window dimensions never change -- only the terrain *values* do -- so each cycle overwrites in place.

Three classes:
  - `BaseSimulator`          -- shared device structs (robot/solver/grid), grid scalars, the
                                wheel-envelope dilation scratch + machinery, and buffer allocation.
  - `ForwardSimulator`       -- forward-only batched rollouts for MPPI (one fused `rollout_kernel`,
                                graph-capturable, no gradients).
  - `DifferentiableSimulator`-- taped per-step rollout for gradient-based calibration: terrain and
                                state buffers carry `.grad`; controls/start pose do NOT (we don't
                                differentiate w.r.t. control). Terrain is ALWAYS per-rollout
                                `[B, ny, nx]` -- each rollout calibrates its own grid.
The standalone FD oracle in tests/engine/gradients.py still keeps its own buffers.
"""

from collections.abc import Callable

import numpy as np
import warp as wp
from warp import Device

from .envelope import _contact_kernel
from .envelope import _gather_kernel
from .envelope import gather_bt
from .envelope import make_tiled_contact
from .envelope import pad_edge
from .envelope import wheel_offset_table
from .robot import RobotParams
from .step import init_state_kernel_bt
from .step import rollout_kernel
from .step import SolverParams
from .step import step_kernel_bt
from .terrain import GridParams

DILATE_TILE = 16  # output tile size for the batched tiled dilation (DifferentiableSimulator)


@wp.kernel
def _final_x_loss(controlled: wp.array2d(dtype=wp.vec3), step: int, loss: wp.array(dtype=float)):
    b = wp.tid()
    wp.atomic_add(loss, 0, controlled[step, b][0])


def demo_loss(sim: "DifferentiableSimulator") -> wp.array:
    """Default `loss_fn` for `rollout_taped`: sum of final-row x over the batch -- a stand-in for
    benchmarks. A `loss_fn` launches its scalar-loss kernel(s) over `sim`'s rollout buffers (it
    runs inside the tape) and returns the [1] loss array; swap it for a real objective (e.g.
    trajectory matching against logged data)."""
    loss = wp.zeros(1, dtype=float, device=sim.device, requires_grad=True)
    wp.launch(
        _final_x_loss,
        sim.batch_size,
        inputs=[sim.controlled, sim.n_steps],
        outputs=[loss],
        device=sim.device,
    )
    return loss


class BaseSimulator:
    """Shared state for the forward/differentiable simulators: the built device structs, the
    grid-derived scalars, and the wheel-envelope dilation scratch + machinery. Subclasses own the
    terrain/rollout buffers (their grad flags and shapes differ). Not meant to be instantiated."""

    def __init__(
        self,
        robot_params: RobotParams,
        solver_params: SolverParams,
        grid_params: GridParams,
        batch_size: int,
        n_steps: int,
        device: Device | str | None = None,
    ):
        self.device = wp.get_device(device)
        self.batch_size = batch_size
        self.n_steps = n_steps

        self.robot = robot_params.build(device)  # device Robot struct
        self.solver = solver_params.build()  # device Solver struct
        self.grid = grid_params.build()  # Grid (fixed)
        self.wheel_radius = robot_params.wheel_radius
        self.env_radius = int(np.ceil(robot_params.wheel_radius / grid_params.cell_size))
        self.cells_y = grid_params.cells_y
        self.cells_x = grid_params.cells_x
        self.cell_size = grid_params.cell_size
        # Terrain + dilation buffers are shape-specific (2D forward, [B, ny, nx] differentiable),
        # so each subclass allocates its own.

    def _alloc_rollout_buffers(self, requires_grad: bool, control_grad: bool) -> None:
        """Allocate the per-rollout state/output/input buffers (shapes shared by both subclasses).
        `requires_grad` covers the state-carrying + diagnostic buffers; controls/start pose follow
        `control_grad` (the differentiable path skips control gradients).
        `current_wheel_omega` and `init_current_wheel_omega` are never differentiated (tau_motor is a non-diff
        struct scalar; promote to wp.array if d(loss)/d(tau) is ever needed)."""
        B, T = self.batch_size, self.n_steps
        rg = requires_grad
        with wp.ScopedDevice(self.device):
            self.controlled = wp.zeros((T + 1, B), dtype=wp.vec3f, requires_grad=rg)
            self.derived = wp.zeros((T + 1, B), dtype=wp.vec3f, requires_grad=rg)
            self.loads = wp.zeros((T, B), dtype=wp.vec3f, requires_grad=rg)
            self.turning = wp.zeros((T, B), dtype=wp.vec2f, requires_grad=rg)
            self.clearance = wp.zeros((T, B), dtype=wp.float32, requires_grad=rg)
            self.residual = wp.zeros((T, B), dtype=wp.float32, requires_grad=rg)
            self.current_wheel_omega = wp.zeros((T + 1, B), dtype=wp.vec3f)
            self.target_wheel_omega = wp.zeros((T, B), dtype=wp.vec3f, requires_grad=control_grad)
            self.start_pose = wp.zeros(B, dtype=wp.vec3f, requires_grad=control_grad)
            self.init_current_wheel_omega = wp.zeros(B, dtype=wp.vec3f)  # like start_pose

    def _dilate(
        self,
        elevation: wp.array,
        contact_iy: wp.array,
        contact_ix: wp.array,
        cap: wp.array,
        envelope: wp.array,
    ) -> None:
        """Wheel-envelope dilation of a 2D `elevation` field (or [b] slice view) into `envelope`,
        via the arg-max contact + gather kernels on the 2D scratch. The shared dilation machinery.
        """
        wp.launch(
            _contact_kernel,
            dim=elevation.shape,
            inputs=[elevation, self.grid.cell_size, self.wheel_radius, self.env_radius],
            outputs=[contact_iy, contact_ix, cap],
            device=self.device,
        )
        wp.launch(
            _gather_kernel,
            dim=elevation.shape,
            inputs=[elevation, contact_iy, contact_ix, cap],
            outputs=[envelope],
            device=self.device,
        )

    def set_uniform_friction(self, value: float) -> None:
        """Uniform friction: overwrite the owned friction grid in place (works for 2D or 3D)."""
        self.friction.fill_(float(value))

    def set_friction(self, friction_hm: np.ndarray) -> None:
        """Per-cell friction from a numpy Heightmap matching the grid (copied in place). 2D only --
        `DifferentiableSimulator` overrides this to take a [B, ny, nx] device `wp.array`."""
        self.friction.assign(np.ascontiguousarray(friction_hm.H, np.float32))


class ForwardSimulator(BaseSimulator):
    """Forward-only batched rollouts for MPPI planning: the whole rollout (init + T steps) runs in
    ONE fused `rollout_kernel` (graph-capturable, no host I/O, register state carry). No gradients --
    terrain and buffers are plain arrays. The differentiable path lives in DifferentiableSimulator
    (the fused kernel's register carry isn't autodiffable)."""

    def __init__(
        self,
        robot_params: RobotParams,
        solver_params: SolverParams,
        grid_params: GridParams,
        batch_size: int,
        n_steps: int,
        device: Device | str | None = None,
    ):
        super().__init__(robot_params, solver_params, grid_params, batch_size, n_steps, device)
        with wp.ScopedDevice(self.device):
            self.elevation = wp.zeros((self.cells_y, self.cells_x), dtype=wp.float32)
            self.envelope = wp.zeros((self.cells_y, self.cells_x), dtype=wp.float32)
            self.friction = wp.zeros((self.cells_y, self.cells_x), dtype=wp.float32)
            self._contact_iy = wp.zeros((self.cells_y, self.cells_x), dtype=wp.int32)
            self._contact_ix = wp.zeros((self.cells_y, self.cells_x), dtype=wp.int32)
            self._cap = wp.zeros((self.cells_y, self.cells_x), dtype=wp.float32)
        self._alloc_rollout_buffers(requires_grad=False, control_grad=False)

    def set_terrain(self, elevation: wp.array) -> None:
        wp.copy(self.elevation, elevation)
        self._dilate(self.elevation, self._contact_iy, self._contact_ix, self._cap, self.envelope)

    def rollout_launch(self) -> None:
        """Launch the whole rollout (init + T steps) in ONE fused kernel; NO host I/O.

        `self.target_wheel_omega` must already hold the controls, `self.start_pose` the init pose,
        and `self.init_current_wheel_omega` the initial lagged wheel speed (e.g. encoder reading;
        zeros if starting from rest). Results stay on device -- the graph-capturable core."""
        wp.launch(
            rollout_kernel,
            self.batch_size,
            inputs=[
                self.n_steps,
                self.envelope,
                self.elevation,
                self.friction,
                self.grid,
                self.robot,
                self.solver,
                self.start_pose,
                self.init_current_wheel_omega,
                self.target_wheel_omega,
            ],
            outputs=[
                self.controlled,
                self.derived,
                self.current_wheel_omega,
                self.loads,
                self.turning,
                self.clearance,
                self.residual,
            ],
            device=self.device,
        )

    def rollout(
        self,
        target_wheel_omega: np.ndarray,
        init_pose: tuple[float, float, float],
        init_wheel_omega: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """target_wheel_omega [T, B, 3], init_pose (x,y,yaw) shared by all rollouts.
        `init_wheel_omega` is the initial actual wheel speed (shape [3] or [B, 3]); defaults to
        zeros (wheels at rest). Returns controlled [T+1,B,3] (x,y,yaw),
        derived [T+1,B,3] (z,pitch,roll), clear/resid [T,B]."""
        # Start pose:
        self.start_pose.assign(
            np.ascontiguousarray(
                np.tile(np.asarray(init_pose, np.float32), (self.batch_size, 1)), np.float32
            )
        )
        # Wheel omega:
        self.target_wheel_omega.assign(np.ascontiguousarray(target_wheel_omega, np.float32))
        if init_wheel_omega is None:
            self.init_current_wheel_omega.zero_()
        else:
            init_wheel_omega_np = np.asarray(init_wheel_omega, np.float32)
            if init_wheel_omega_np.ndim == 1:
                init_wheel_omega_np = np.tile(init_wheel_omega_np, (self.batch_size, 1))
            self.init_current_wheel_omega.assign(np.ascontiguousarray(init_wheel_omega_np))

        # Launch rollout:
        self.rollout_launch()
        return (
            self.controlled.numpy(),
            self.derived.numpy(),
            self.clearance.numpy(),
            self.residual.numpy(),
        )


class DifferentiableSimulator(BaseSimulator):
    """Forward-and-backward simulator for gradient-based calibration. ALWAYS per-rollout terrain.

    Records init + T per-step launches (`step_kernel_bt`) on a `wp.Tape` so `backward` can backprop
    to the terrain. The fused `rollout_kernel` isn't autodiffable (register carry), so this pays one
    launch per step. Usage:

        sim = DifferentiableSimulator(robot, solver, grid, B, T, device)
        sim.set_terrain(elev_stack)              # elev_stack is [B, ny, nx]
        sim.set_friction(mu_stack)               # mu_stack is a [B, ny, nx] wp.array; or set_uniform_friction(value)
        sim.target_wheel_omega.assign(...); sim.start_pose.assign(...)
        sim.rollout_taped(my_loss_fn); sim.backward()   # loss_fn defaults to demo_loss
        g = sim.friction.grad.numpy()            # d(loss)/d(friction), shape [B, ny, nx]

    Terrain is `[B, ny, nx]`: rollout b runs on its own elevation/envelope/friction and gets its own
    grad slice. Control and start pose are already per-rollout, so this covers both "B terrains,
    fixed control" (fill the controls identically) and "B terrains, B controls" (a calibration
    dataset of independent episodes); for one shared terrain, broadcast it across the B slices.

    Gradients flow to the raw terrain (`elevation`/`friction`) and the state buffers
    (`controlled`/`derived`); controls (`target_wheel_omega`) and `start_pose` are NOT grad-tracked -- we
    don't differentiate w.r.t. control, and a non-grad leaf simply skips its gradient without
    breaking the terrain chain. The dilation is split: a shared-memory tiled arg-max CONTACT runs
    off-tape (fast, non-diff) and the GATHER runs on-tape (envelope = elevation[contact] + cap), so
    `d(loss)/d(raw elevation)` flows through the cheap scatter adjoint -- the analytical gradient,
    not autodiff of the convolution.

    CUDA-only (the tiled contact needs GPU shared memory). NOTE: per-rollout terrain costs B x the
    grid memory (x2 for grads), so B is the number of calibration episodes (10s-100s), not the
    thousands used for planning.
    """

    def __init__(
        self,
        robot_params: RobotParams,
        solver_params: SolverParams,
        grid_params: GridParams,
        batch_size: int,
        n_steps: int,
        device: Device | str | None = None,
    ):
        super().__init__(robot_params, solver_params, grid_params, batch_size, n_steps, device)

        if not self.device.is_cuda:
            raise RuntimeError(
                "DifferentiableSimulator is CUDA-only: the tiled arg-max contact needs GPU shared "
                "memory. Build it with device='cuda'."
            )

        self.tape: wp.Tape | None = None
        self._loss: wp.array | None = None

        ny, nx = self.cells_y, self.cells_x
        R, T = self.env_radius, DILATE_TILE
        dy, dx, cap = wheel_offset_table(R, self.cell_size, self.wheel_radius)

        # `_best_k` is the off-tape contact (arg-max offset per cell); the offset table feeds both the
        # tiled contact and the gather. Edge-padded, tile-aligned halo buffer for the tiled contact.
        self._tiled_contact = make_tiled_contact(R, T)
        self._n_tiles = ((ny + T - 1) // T, (nx + T - 1) // T)
        pny, pnx = self._n_tiles[0] * T + 2 * R, self._n_tiles[1] * T + 2 * R
        with wp.ScopedDevice(self.device):  # per-rollout terrain [B, ny, nx]
            self.elevation = wp.zeros((batch_size, ny, nx), dtype=wp.float32, requires_grad=True)
            self.envelope = wp.zeros((batch_size, ny, nx), dtype=wp.float32, requires_grad=True)
            self.friction = wp.zeros((batch_size, ny, nx), dtype=wp.float32, requires_grad=True)
            self._best_k = wp.zeros((batch_size, ny, nx), dtype=wp.float32)  # contact offset
            self._elev_pad = wp.zeros((batch_size, pny, pnx), dtype=wp.float32)
            self._off_dy = wp.array(dy, dtype=wp.int32)
            self._off_dx = wp.array(dx, dtype=wp.int32)
            self._off_cap = wp.array(cap, dtype=wp.float32)

        # State/diagnostic buffers carry grad; controls + start pose do NOT (no control gradients).
        self._alloc_rollout_buffers(requires_grad=True, control_grad=False)

    def _contact(self) -> None:
        """Off-tape shared-memory tiled arg-max -> self._best_k (the contact offset per cell).
        Non-differentiable (the gather supplies the gradient), so the fast tiled path costs no
        backward. Edge-pads first so the halo loads never go out of bounds."""
        wp.launch(
            pad_edge,
            dim=self._elev_pad.shape,
            inputs=[self.elevation, self.env_radius, self._elev_pad],
            device=self.device,
        )
        wp.launch_tiled(
            self._tiled_contact,
            dim=(self.batch_size, self._n_tiles[0], self._n_tiles[1]),
            inputs=[self._elev_pad, self._off_dy, self._off_dx, self._off_cap, self._best_k],
            block_dim=128,
            device=self.device,
        )

    def _gather(self) -> None:
        """envelope = elevation[contact] + cap using the fixed `self._best_k`. Recorded ON the tape
        in `rollout_taped` -- its scatter adjoint IS d(envelope)/d(elevation), the analytical
        gradient. This is the ONLY place envelope is built, so it isn't valid until `rollout_taped`.
        """
        wp.launch(
            gather_bt,
            dim=self.elevation.shape,
            inputs=[self.elevation, self._best_k, self._off_dy, self._off_dx, self._off_cap],
            outputs=[self.envelope],
            device=self.device,
        )

    def set_terrain(self, elevation: wp.array) -> None:
        """Load a [B, ny, nx] device stack into the owned `elevation` buffer (copy + shape check).
        A convenience init helper -- `elevation` is a public calibration parameter you may also
        modify in place; `rollout_taped` re-derives the contact from it each call either way."""
        assert (
            elevation.shape == self.elevation.shape
        ), f"elevation {elevation.shape} must match the sim's [B, ny, nx] {self.elevation.shape}"
        wp.copy(self.elevation, elevation)

    def set_friction(self, friction: wp.array) -> None:
        """Load a [B, ny, nx] device stack into the owned `friction` buffer (copy + shape check).
        Like `set_terrain`, a convenience init helper for a public calibration parameter you may
        also modify in place. Overrides the base Heightmap setter (friction here is a wp.array)."""
        assert (
            friction.shape == self.friction.shape
        ), f"friction {friction.shape} must match the sim's [B, ny, nx] {self.friction.shape}"
        wp.copy(self.friction, friction)

    def rollout_taped(self, loss_fn: Callable | None = demo_loss) -> wp.array | None:
        """Record init + T per-step launches (+ optional `loss_fn(self)`) on a fresh tape; return the
        loss array (or None). `self.target_wheel_omega`/`self.start_pose` must already hold controls/pose;
        set `self.init_current_wheel_omega` before calling if the robot is already moving (default: zeros).

        Two backward paths follow:
          - scalar loss: pass a `loss_fn` (default `demo_loss`); it runs INSIDE the tape so its
            launches are recorded, then call `backward()`. Injected (not fixed) because the
            objective is domain-specific (calibration vs benchmark).
          - VJP / framework bridge (torch, jax, ...): pass `loss_fn=None` to just record the
            rollout, then seed output cotangents via `backward_from_cotangents(...)`.

        The arg-max CONTACT is recomputed off-tape here (fresh for the current `elevation`, so
        in-place parameter updates need no `set_terrain`), then the cheap GATHER runs ON the tape so
        `d(loss)/d(raw elevation)` connects through envelope = elevation[contact] -- its scatter
        adjoint is the analytical gradient. `elevation`/`friction` must already be set."""
        self._contact()  # off-tape arg-max -> best_k, fresh for the current elevation
        wp.copy(
            self.current_wheel_omega[0], self.init_current_wheel_omega
        )  # seed current_wheel_omega[0] off-tape: boundary condition, not differentiated.
        self.tape = wp.Tape()
        with self.tape:
            self._gather()  # on-tape: envelope = elev[contact] + cap; scatter adjoint -> d/d elevation
            wp.launch(
                init_state_kernel_bt,
                self.batch_size,
                inputs=[self.envelope, self.grid, self.robot, self.solver, self.start_pose],
                outputs=[self.controlled, self.derived],
                device=self.device,
            )
            for t in range(self.n_steps):
                wp.launch(
                    step_kernel_bt,
                    self.batch_size,
                    inputs=[
                        self.envelope,
                        self.elevation,
                        self.friction,
                        self.grid,
                        self.robot,
                        self.solver,
                        self.target_wheel_omega[t],
                        self.current_wheel_omega[t],
                        self.controlled[t],
                        self.derived[t],
                    ],
                    outputs=[
                        self.current_wheel_omega[t + 1],
                        self.controlled[t + 1],
                        self.derived[t + 1],
                        self.loads[t],
                        self.turning[t],
                        self.clearance[t],
                        self.residual[t],
                    ],
                    device=self.device,
                )
            self._loss = loss_fn(self) if loss_fn is not None else None
        return self._loss

    def backward(self) -> None:
        """Backprop the recorded scalar loss. Gradients land in the `.grad` of the terrain
        (elevation/friction) and the state buffers. Use `backward_from_cotangents` instead if you
        recorded with `loss_fn=None`."""
        if self.tape is None:
            raise RuntimeError("call rollout_taped() before backward()")
        if self._loss is None:
            raise RuntimeError(
                "no scalar loss recorded (rollout_taped(loss_fn=None)); use backward_from_cotangents"
            )
        self.tape.backward(loss=self._loss)

    def backward_from_cotangents(self, adj_controlled: wp.array, adj_derived: wp.array) -> None:
        """VJP boundary for a framework bridge (torch/jax autograd): seed output cotangents on the
        state buffers and backprop. `adj_controlled`/`adj_derived` are [T+1, B] vec3 = dL/d(controlled)
        and dL/d(derived) from downstream; gradients land in `elevation.grad`/`friction.grad`. Record
        with `rollout_taped(loss_fn=None)` first. (Seeds via Warp's `grads=` dict -- no scalar loss.)
        """
        if self.tape is None:
            raise RuntimeError("call rollout_taped(loss_fn=None) before backward_from_cotangents()")
        assert (
            adj_controlled.shape == self.controlled.shape
        ), f"adj_controlled {adj_controlled.shape} must match controlled {self.controlled.shape}"
        assert (
            adj_derived.shape == self.derived.shape
        ), f"adj_derived {adj_derived.shape} must match derived {self.derived.shape}"
        self.tape.backward(grads={self.controlled: adj_controlled, self.derived: adj_derived})

    def zero_grad(self) -> None:
        """Zero the grads recorded on the tape (call between optimizer iterations)."""
        if self.tape is not None:
            self.tape.zero()
