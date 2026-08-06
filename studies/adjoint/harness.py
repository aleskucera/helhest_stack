"""Adjoint-vs-finite-difference harness for Study A.

The oracle is finite differences of the WARP forward itself, not of the numpy reference.
An adjoint is only correct *with respect to a forward*; differencing a different forward
conflates "the adjoint is wrong" with "the two forwards have drifted apart". The numpy
reference is used once, separately, as a model-parity check (level A0).

Two differentiability layers sit on the h -> J path and they fail for different reasons at
a curb, so the harness can isolate them:

  envelope -> (z, pitch, roll)   `settle_bt` / `adj_settle_bt`, the IFT adjoint
  elevation -> envelope          the dilation: arg-max computed OFF-tape (`_contact`) and
                                 then FROZEN while `gather_bt` runs on-tape. That is a
                                 Danskin subgradient, exact only where the arg-max is
                                 unique and not near-tied -- which a curb edge violates.

`dilate=False` replaces the dilation with the identity (the offset table's (0, 0) entry has
cap 0, so `gather_bt` becomes `envelope = elevation`) and the `elevation` buffer is loaded
with envelope VALUES. `elevation.grad` is then d/d(envelope) and only the settle adjoint is
under test. `dilate=True` is the production path.

`_rollout` mirrors `DifferentiableSimulator.rollout_taped` so the dilation can be swapped;
level A0 asserts the two agree bit-for-bit, so the replica is not a second implementation
being validated in place of the real one.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import warp as wp

from helhest.engine import DifferentiableSimulator
from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams
from helhest.engine.envelope import wheel_offset_table
from helhest.engine.step import init_state_kernel_bt
from helhest.engine.step import step_kernel_bt

from . import subcell

# Five scalar functionals of one rollout, kept SEPARATE (not summed) so a failure is
# attributable to a path rather than to "the loss". They share a single forward pass; only
# the backward is repeated, once per term, with a one-hot cotangent.
TERM_NAMES = ("settle0", "settle", "pose", "clear", "clear_soft")
N_TERMS = len(TERM_NAMES)
TERM_DOC = {
    "settle0": "w . (z, pitch, roll) of the INIT settle only -- ONE settle, no chaining",
    "settle": "sum over t>=1 of w . (z, pitch, roll) -- the settle chain through the rollout",
    "pose": "w . final (x, y, yaw) -- the step_predict / BPTT path",
    "clear": "sum of MIN belly clearance -- the tied min, kept to show the defect",
    "clear_soft": "sum of the tie-free belly-margin violation -- the fix, measured against it",
}
# `clear` and `clear_soft` are the same physical quantity aggregated two ways, reported side by
# side so the fix for Study A finding 3 is a measurement rather than an assertion.
#
# The hinge in `clear_soft` is only active when the belly is within clear_margin of the ground,
# and the belly sits 0.25 m above the contact plane while the tallest scene obstacle is 0.20 m
# -- so at the production margin of 0.05 m the term would be identically zero and untestable.
# Raising it to 0.30 m puts EVERY bottom-face point in violation by a similar amount, which is
# the maximal-tie case: precisely the configuration that breaks the min. It affects nothing
# else -- clear_margin never enters the settle, the twist, or any other output.
STUDY_CLEAR_MARGIN = 0.30
# The first three are LINEAR functionals. A quadratic tilt cost (pitch^2 + roll^2) is more
# plan-like but its gradient vanishes identically wherever pitch = roll = 0, i.e. over the
# whole flat lane -- it would leave the flat region with nothing to test. The plan-realistic
# cost belongs in Study B; Study A wants functionals that never degenerate.
# Literals, not tuples: a Warp kernel cannot close over a host tuple.
DERIV_WZ = 1.0
DERIV_WPITCH = 0.7
DERIV_WROLL = 0.5
POSE_WX = 1.0
POSE_WY = 0.5
POSE_WYAW = 0.3


@wp.kernel
def _terms_kernel(
    controlled: wp.array2d(dtype=wp.vec3),  # [T+1, B] (x, y, yaw)
    derived: wp.array2d(dtype=wp.vec3),  # [T+1, B] (z, pitch, roll)
    clearance: wp.array2d(dtype=wp.float32),  # [T, B] min belly clearance
    clear_soft: wp.array2d(dtype=wp.float32),  # [T, B] tie-free margin violation
    n_steps: int,
    terms: wp.array2d(dtype=wp.float32),  # [N_TERMS, B] -- zeroed before launch
):
    """Per-rollout scalar functionals. Accumulated with atomic_add into a pre-zeroed array
    (a plain indexed assignment has a fussier adjoint; the atomic's is unambiguous).

    The `float(0.0)` wrappers are required by Warp to declare a mutable local inside a
    dynamic loop -- a bare `0.0` is a compile-time constant and fails to build. ruff's
    UP018 will strip them if allowed to; keep the noqa.
    """
    b = wp.tid()
    d0 = derived[0, b]
    wp.atomic_add(terms, 0, b, DERIV_WZ * d0[0] + DERIV_WPITCH * d0[1] + DERIV_WROLL * d0[2])

    settle = float(0.0)  # noqa: UP018
    for t in range(1, n_steps + 1):
        d = derived[t, b]
        settle += DERIV_WZ * d[0] + DERIV_WPITCH * d[1] + DERIV_WROLL * d[2]
    wp.atomic_add(terms, 1, b, settle)

    p = controlled[n_steps, b]
    wp.atomic_add(terms, 2, b, POSE_WX * p[0] + POSE_WY * p[1] + POSE_WYAW * p[2])

    clear = float(0.0)  # noqa: UP018
    soft = float(0.0)  # noqa: UP018
    for t in range(n_steps):
        clear += clearance[t, b]
        soft += clear_soft[t, b]
    wp.atomic_add(terms, 3, b, clear)
    wp.atomic_add(terms, 4, b, soft)


@wp.kernel
def _perturb_cell(arr: wp.array3d(dtype=wp.float32), iy: int, ix: int, delta: float):
    """Add `delta` to cell (iy, ix) of EVERY rollout slice, on device.

    One launch perturbs all B slices, and because the slices are independent this yields B
    finite differences from a single forward pass -- the reason the whole sweep is cheap.
    """
    b = wp.tid()
    arr[b, iy, ix] = arr[b, iy, ix] + delta


class Harness:
    """Owns the simulator, the pristine terrain copies, and the adjoint/FD drivers."""

    def __init__(
        self,
        scene,
        poses: np.ndarray,
        omega: np.ndarray,
        dt: float = 0.1,
        newton_iters: int = 20,
        device: str = "cuda",
    ):
        ny, nx = scene.shape
        self.batch_size = poses.shape[0]
        self.n_steps = omega.shape[0]
        self.scene = scene
        self.newton_iters = newton_iters
        # atol = 0 disables the settle's early exit, so h+eps and h-eps always take the SAME
        # number of Newton iterations. Otherwise the iteration count itself is a function of
        # h and the forward map carries small step discontinuities that FD reads as noise.
        solver = SolverParams(dt=dt, newton_iters=newton_iters, atol=0.0)
        grid = GridParams(nx, ny, scene.cell, scene.origin_x, scene.origin_y)
        self.robot_params = RobotParams(clear_margin=STUDY_CLEAR_MARGIN)
        self.sim = DifferentiableSimulator(
            self.robot_params, solver, grid, self.batch_size, self.n_steps, device
        )
        self.device = self.sim.device

        self.sim.target_wheel_omega.assign(np.ascontiguousarray(omega, np.float32))
        self.sim.start_pose.assign(np.ascontiguousarray(poses, np.float32))
        self.sim.init_current_wheel_omega.zero_()

        stack = lambda a: np.ascontiguousarray(np.tile(a, (self.batch_size, 1, 1)), np.float32)
        self._raw_np = stack(scene.elevation)
        self._fric_np = stack(scene.friction)
        with wp.ScopedDevice(self.device):
            self.terms = wp.zeros((N_TERMS, self.batch_size), dtype=wp.float32, requires_grad=True)
            self._raw0 = wp.array(self._raw_np, dtype=wp.float32)
            self._fric0 = wp.array(self._fric_np, dtype=wp.float32)

        # The (0, 0) offset (cap 0) turns `gather_bt` into the identity -- how `dilate=False`
        # bypasses the dilation without touching the production kernels.
        dy, dx, _ = wheel_offset_table(
            self.sim.env_radius, scene.cell, self.robot_params.wheel_radius
        )
        self._k_identity = int(np.flatnonzero((dy == 0) & (dx == 0))[0])

        # The envelope-leaf levels run on the elevation buffer loaded with the EXACT envelope
        # the production dilation produces, so they test the same terrain the real path sees.
        # Optional sub-cell contact refinement (studies/adjoint/subcell.py). Off by default so
        # every earlier result is unaffected; `use_subcell = True` swaps BOTH the forward and
        # the backward, which is required for a fair comparison against Monte-Carlo truth.
        self.use_subcell = False
        self._subcell = subcell.SubcellDilation(
            self.sim, scene.cell, self.robot_params.wheel_radius
        )

        # Friction must be loaded here, not only in `_reset_terrain`: a caller that drives the
        # terrain itself and then calls `forward` would otherwise roll out on mu = 0 and get a
        # NaN pose out of the traction solve, with the settle still looking plausible.
        self.sim.set_terrain(self._raw0)
        wp.copy(self.sim.friction, self._fric0)
        self.sim._contact()
        self.sim._gather()
        with wp.ScopedDevice(self.device):
            self._env0 = wp.array(self.sim.envelope.numpy(), dtype=wp.float32)
        self._env_np = self._env0.numpy()

    # --- terrain source for a level ------------------------------------------------------
    def base_terrain(self, dilate: bool) -> np.ndarray:
        """Host copy of the elevation buffer's pristine contents for this level."""
        return self._raw_np if dilate else self._env_np

    def _reset_terrain(self, dilate: bool) -> None:
        wp.copy(self.sim.elevation, self._raw0 if dilate else self._env0)
        wp.copy(self.sim.friction, self._fric0)

    # --- forward / tape ------------------------------------------------------------------
    def _launches(self) -> None:
        sim = self.sim
        self._subcell.gather() if self.use_subcell else sim._gather()
        wp.launch(
            init_state_kernel_bt,
            self.batch_size,
            inputs=[sim.envelope, sim.grid, sim.robot, sim.solver, sim.start_pose],
            outputs=[sim.controlled, sim.derived],
            device=self.device,
        )
        for t in range(self.n_steps):
            wp.launch(
                step_kernel_bt,
                self.batch_size,
                inputs=[
                    sim.envelope,
                    sim.elevation,
                    sim.friction,
                    sim.grid,
                    sim.robot,
                    sim.solver,
                    sim.target_wheel_omega[t],
                    sim.current_wheel_omega[t],
                    sim.controlled[t],
                    sim.derived[t],
                ],
                outputs=[
                    sim.current_wheel_omega[t + 1],
                    sim.controlled[t + 1],
                    sim.derived[t + 1],
                    sim.loads[t],
                    sim.turning[t],
                    sim.clearance[t],
                    sim.clear_soft[t],
                    sim.residual[t],
                ],
                device=self.device,
            )
        wp.launch(
            _terms_kernel,
            self.batch_size,
            inputs=[sim.controlled, sim.derived, sim.clearance, sim.clear_soft, self.n_steps],
            outputs=[self.terms],
            device=self.device,
        )

    def _rollout(self, dilate: bool, tape: wp.Tape | None = None) -> None:
        """One forward rollout; on `tape` if given. Mirrors `rollout_taped` except that the
        arg-max contact can be replaced by the identity."""
        sim = self.sim
        if dilate and self.use_subcell:
            self._subcell.contact()  # off-tape: discrete arg-max, then parabolic refinement
        elif dilate:
            sim._contact()  # off-tape arg-max, recomputed for the CURRENT elevation
        else:
            sim._best_k.fill_(float(self._k_identity))
        wp.copy(sim.current_wheel_omega[0], sim.init_current_wheel_omega)
        self.terms.zero_()
        if tape is None:
            self._launches()
        else:
            with tape:
                self._launches()

    def forward(self, dilate: bool) -> np.ndarray:
        """Forward only. Returns terms [N_TERMS, B]."""
        self._rollout(dilate)
        return self.terms.numpy()

    # --- gradients -----------------------------------------------------------------------
    def adjoint(self, dilate: bool, leaf: str) -> tuple[np.ndarray, np.ndarray]:
        """Record once, back-propagate each term with a one-hot cotangent on `terms`.

        Returns (grads [N_TERMS, B, ny, nx] w.r.t. `leaf`, terms [N_TERMS, B]).
        """
        self._reset_terrain(dilate)
        tape = wp.Tape()
        self._rollout(dilate, tape)
        terms = self.terms.numpy()
        target = self.sim.elevation if leaf == "elevation" else self.sim.friction
        seed = wp.zeros_like(self.terms)
        grads = np.zeros((N_TERMS, self.batch_size, *self.scene.shape), np.float32)
        for k in range(N_TERMS):
            tape.zero()
            onehot = np.zeros((N_TERMS, self.batch_size), np.float32)
            onehot[k] = 1.0
            seed.assign(onehot)
            tape.backward(grads={self.terms: seed})
            grads[k] = target.grad.numpy()
        tape.zero()
        return grads, terms

    def finite_differences(
        self, dilate: bool, leaf: str, cells: Iterable[tuple[int, int]], eps: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """One-sided differences of the same forward, each [n_cells, N_TERMS, B].

        Returns (D_plus, D_minus). The central difference is their mean; keeping them apart
        costs one extra forward for the whole sweep (the unperturbed evaluation is shared)
        and buys a per-cell KINK DETECTOR: at a smooth point D+ == D-, at a kink they differ
        and the central difference silently reports their average. That distinction is what
        separates "the adjoint is wrong" from "the forward is not differentiable here", and
        it is the quantity Study B stratifies on.

        With `dilate=True` the arg-max is recomputed inside each perturbed forward, so this
        differences the TRUE forward including contact switching -- which is what lets it
        expose the frozen-arg-max subgradient at a curb edge.
        """
        target = self.sim.elevation if leaf == "elevation" else self.sim.friction
        cells = list(cells)
        self._reset_terrain(dilate)
        base = self.forward(dilate).copy()
        d_plus = np.zeros((len(cells), N_TERMS, self.batch_size), np.float32)
        d_minus = np.zeros_like(d_plus)
        for n, (iy, ix) in enumerate(cells):
            for sign, out in ((1.0, d_plus), (-1.0, d_minus)):
                self._reset_terrain(dilate)  # restore from a pristine copy, never by -eps
                wp.launch(
                    _perturb_cell,
                    self.batch_size,
                    inputs=[target, iy, ix, sign * eps],
                    device=self.device,
                )
                out[n] = sign * (self.forward(dilate) - base) / eps
        self._reset_terrain(dilate)
        return d_plus, d_minus

    # --- diagnostics carried alongside (Study B's stratification axes) --------------------
    def diagnostics(self, dilate: bool) -> dict[str, np.ndarray]:
        """Run one clean forward and return the per-step quantities Study B stratifies on:
        the tip-over margin min N_i / (m g) and the settle residual (the IFT premise)."""
        self._reset_terrain(dilate)
        self._rollout(dilate)
        loads = self.sim.loads.numpy()  # [T, B, 3]
        weight = self.robot_params.mass * self.robot_params.gravity
        return {
            "chassis_tie": self._chassis_tie(),
            "n_chassis_pts": int(self.robot_params._chassis_pts().shape[0]),
            "loads": loads,
            "stability_margin": loads.min(axis=2) / weight,  # [T, B]
            "residual": self.sim.residual.numpy(),  # [T, B]
            # winner - runner-up of the dilation arg-max: Study B's linearisation-validity flag
            "contact_margin": self.sim.contact_margin.numpy()[0],  # [ny, nx], shared terrain
            "derived": self.sim.derived.numpy(),  # [T+1, B, 3]
            "controlled": self.sim.controlled.numpy(),
        }

    def tip_over_angles(self) -> dict[str, float]:
        """Static tip-over angle about each edge of the wheel support TRIANGLE.

        Not atan(half_track / CoM height): on a tripod the robot tips about the line joining
        two contacts, and the front-axle edge -- not the track -- is the nearest boundary
        because the CoM sits behind it. min N_i reaches 0 exactly at these angles, so they
        bound where the settle's contact set can switch at all.
        """
        rp = self.robot_params
        wheels = np.array(
            [[0.0, rp.half_track], [0.0, -rp.half_track], [-rp.rear_offset, 0.0]], float
        )
        com = np.array(rp.com[:2], float)
        com_height = rp.wheel_radius  # body origin sits one wheel radius above the contacts
        out = {}
        for i, j, name in ((0, 1, "front_axle"), (0, 2, "left_rear"), (1, 2, "right_rear")):
            edge = wheels[j] - wheels[i]
            lever = abs(float(np.cross(edge, com - wheels[i]))) / float(np.linalg.norm(edge))
            out[name] = float(np.degrees(np.arctan2(lever, com_height)))
        out["min"] = min(v for k, v in out.items() if k != "min")
        return out

    def _chassis_tie(self) -> np.ndarray:
        """How many of the belly sample points sit within 1 mm of the minimum clearance, per
        rollout, at the LAST step.

        `chassis_clearance` is a `min` over these points, so a tie is a kink: the adjoint
        routes the whole gradient to one tied point and reports a hard zero at the others,
        while a finite difference at any of them moves the min. All the belly points share
        one body-frame z, so on level ground the tie is total.
        """
        from helhest.reference import placement

        pts = self.robot_params._chassis_pts().astype(np.float64)
        controlled = self.sim.controlled.numpy()[-1]  # [B, 3]
        derived = self.sim.derived.numpy()[-1]
        counts = np.zeros(self.batch_size, np.int32)
        for b in range(self.batch_size):
            x, y, yaw = controlled[b]
            z, pitch, roll = derived[b]
            hm = _numpy_heightmap(self.scene)
            R = placement.euler_zyx(yaw, pitch, roll)
            world = np.array([x, y, z])[None, :] + pts @ R.T
            clear = world[:, 2] - hm.sample(world[:, 0], world[:, 1])
            counts[b] = int((clear - clear.min() < 1e-3).sum())
        return counts


def _numpy_heightmap(scene):
    from helhest.heightmap import Heightmap

    return Heightmap(scene.elevation, (scene.origin_x, scene.origin_y), scene.cell)
