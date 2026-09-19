"""The value field: constraints in, cost-to-go out.

`TerrainValueField` is the whole library's front door. It owns the buffers, reduces a
producer's per-constraint margins to a safety margin in sigmas, turns that into a veto and a
graded cost, and value-iterates to a fixed point.

It estimates and it stops. There is no controller here, no frontier policy, no goal-versus-
explore arbitration: those are decisions about what a particular robot should do, and a robot
that disagrees with ours should not have to fork a planner to say so. What comes out is a field
-- `V`, and the `z`, `pose_cost` and `doubt` it was built from -- and the robot decides the rest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from . import margin as _margin
from .solver import ValueSolver


@dataclass
class Constraints:
    """A producer's output: per-state margins, their uncertainties, and their floors.

    `margin[i, r, c, t]` is how much room constraint `i` has left at that state, in whatever
    units that constraint uses; `sigma` is its standard deviation in the same units;
    `floor[i]` is the irreducible part that no better map removes. A constraint may write
    `margin >= margin.IGNORED` to decline to speak about a state.
    """

    margin: wp.array  # [n_constraints, rows, cols, headings]
    sigma: wp.array  # [n_constraints, rows, cols, headings]
    floor: wp.array  # [n_constraints]


class TerrainValueField:
    def __init__(
        self,
        rows: int,
        cols: int,
        resolution: float,
        n_theta: int = 16,
        *,
        k_sigma: float = 2.0,
        z_ref: float = 4.0,
        penalty_weight: float = 0.0,
        penalty_scale: float = 1.0,
        control_set: tuple | None = None,
        turn_radius: float = 0.6,
        step: float | None = None,
        turn_weight: float = 0.0,
        pivot_cost: float | None = None,  # None = 8x the equal-turn arc; inf = none
        device: wp.Device | str | None = None,
    ) -> None:
        """`k_sigma` is the one knob: how many standard deviations of room a state must hold.

        Two penalty knobs, and they act in different places -- read them together once:

        `penalty_weight` is charged in the MARGIN kernel. It turns "how many sigmas of room is
        left below `z_ref`" into a per-state cost, so it sets the units. `penalty_weight = 0`
        gives a pure veto and no gradient at all.

        `penalty_scale` is applied in the SOLVER. It multiplies the mean per-state cost along a
        move's swept cells into that move's price. It trades route length against room: 0 makes
        the field a pure shortest path over the unvetoed states.

        Both must be >= 0. The solver's own parameter is also called `penalty_scale`; nothing
        downstream of here is called `penalty_weight`.
        """
        # Both knobs must be non-negative, and not as a matter of taste. `penalty_weight` < 0
        # makes the graded penalty negative, and `pose_cost` carries the veto in its sign -- so
        # every free state would read as vetoed and the whole map would go unreachable, silently.
        # `penalty_scale` < 0 can drive a move's cost below zero, which breaks min-plus outright.
        if penalty_weight < 0.0:
            raise ValueError(
                f"penalty_weight must be >= 0 (the veto rides in the sign of the "
                f"graded cost; see margin.POSE COST), got {penalty_weight}"
            )
        if penalty_scale < 0.0:
            raise ValueError(
                f"penalty_scale must be >= 0 or a move can cost less than "
                f"nothing, got {penalty_scale}"
            )
        self.device = wp.get_device(device)
        self.rows, self.cols, self.n_theta = int(rows), int(cols), int(n_theta)
        self.resolution = float(resolution)
        self.k_sigma = float(k_sigma)
        self.z_ref = float(z_ref)
        self.penalty_weight = float(penalty_weight)
        self.penalty_scale = float(penalty_scale)

        shape = (self.rows, self.cols, self.n_theta)
        with wp.ScopedDevice(self.device):
            self.z = wp.zeros(shape, dtype=wp.float32)
            self.z_certain = wp.zeros(shape, dtype=wp.float32)
            # graded cost with the veto in the sign; see margin.POSE COST
            self.pose_cost = wp.zeros(shape, dtype=wp.float32)
            self.doubt = wp.zeros(shape, dtype=wp.float32)
            self.V = wp.zeros(shape, dtype=wp.float32)
            self.V_certain = wp.zeros(shape, dtype=wp.float32)
            self._seeds = wp.zeros(shape, dtype=wp.float32)
            self._k = wp.array([self.k_sigma], dtype=wp.float32)

        self.solver = ValueSolver(
            self.resolution,
            self.rows,
            self.cols,
            n_theta=self.n_theta,
            turn_radius=turn_radius,
            step=step,
            turn_weight=turn_weight,
            pivot_cost=pivot_cost,
            control_set=control_set,
            device=self.device,
        )

    # -- seeds ----------------------------------------------------------------------------
    def seed_states(self, mask: np.ndarray | wp.array) -> None:
        """Set the zero-cost states directly. `mask` is [rows, cols, headings], non-zero = seed."""
        if isinstance(mask, wp.array):
            wp.copy(self._seeds, mask)
        else:
            self._seeds.assign(np.ascontiguousarray(mask, dtype=np.float32))

    def seed_cell(self, row: int, col: int) -> None:
        """Seed one cell at every heading -- the ordinary "drive to here" case."""
        m = np.zeros((self.rows, self.cols, self.n_theta), np.float32)
        m[int(row), int(col), :] = 1.0
        self._seeds.assign(m)

    # -- solve ----------------------------------------------------------------------------
    def solve(self, constraints: Constraints, certain: bool = False) -> wp.array:
        """Reduce the constraints, classify, and value-iterate. Returns `V` (device-resident).

        `certain=True` scores every state as if the map carried no uncertainty beyond the
        floors. That is not a cheaper approximation -- it is the second half of the pair that
        makes `doubt` meaningful, and on its own it is the optimistic reading.
        """
        self._k.assign(np.array([self.k_sigma], np.float32))
        wp.launch(
            _margin.margin_to_fields_kernel,
            dim=(self.rows, self.cols, self.n_theta),
            inputs=[
                constraints.margin,
                constraints.sigma,
                constraints.floor,
                self._k,
                self.z_ref,
                self.penalty_weight,
            ],
            outputs=[self.z, self.z_certain, self.pose_cost, self.doubt],
            device=self.device,
        )
        if certain:
            # The optimistic reading classifies `z_certain` against itself. Re-running the
            # classify half alone is the cheap way to say that without a second fused kernel
            # whose only difference is which of two registers it compares.
            wp.launch(
                _margin.classify_kernel,
                dim=(self.rows, self.cols, self.n_theta),
                inputs=[
                    self.z_certain,
                    self.z_certain,
                    self._k,
                    self.z_ref,
                    self.penalty_weight,
                ],
                outputs=[self.pose_cost, self.doubt],
                device=self.device,
            )
        result = self.solver.value_iterate(self.pose_cost, self._seeds, self.penalty_scale)
        wp.copy(self.V, result)
        return self.V

    def solve_pair(self, constraints: Constraints) -> tuple[wp.array, wp.array]:
        """Solve believing the map, then as if it were certain. Returns (V, V_certain).

        Their difference at a state is what IGNORANCE costs there, in the same units as the
        plan cost. Neither solve gives that alone: scored pessimistically, not knowing is
        expensive everywhere; scored optimistically it is free everywhere. Only the pair
        separates "this route is blocked by ground" from "this route is blocked by the map".

        The certain solve is usually the DEARER of the two -- it vetoes fewer states, so more
        of the space is reachable and the iteration needs more sweeps. Budget for that rather
        than assuming twice the cost.
        """
        self.solve(constraints, certain=False)
        # Park every field the believed-map solve produced. `solve` reuses these buffers, so
        # without this the object would end up describing the certain solve while claiming to
        # hold the believed one -- a mismatch that reads as a library bug from the outside.
        parked = {n: wp.clone(getattr(self, n)) for n in ("V", "z", "pose_cost", "doubt")}
        self.solve(constraints, certain=True)
        wp.copy(self.V_certain, self.V)
        for n, buf in parked.items():
            wp.copy(getattr(self, n), buf)
        return self.V, self.V_certain

    # -- read -----------------------------------------------------------------------------
    def converged(self) -> bool:
        """Did the last solve reach a fixed point? See `ValueSolver.converged`. Syncs.

        Worth asking before trusting an `unreachable` reading: a capped solve reports the same
        thing a genuinely walled-off map does.
        """
        return self.solver.converged()

    def unreachable_value(self) -> float:
        """Values at or above this mean "no route under the control set"."""
        return float(self.solver._inf) * 0.99

    def at(self, row: int, col: int, heading_bin: int = 0) -> dict:
        """Read the fields at one state. Scalars, so the host is the right place for them."""
        r, c, t = int(row), int(col), int(heading_bin)
        cap = self.unreachable_value()
        v = float(self.V.numpy()[r, c, t])
        vc = float(self.V_certain.numpy()[r, c, t])
        return {
            "v": v,
            "v_certain": vc,
            "gap": v - vc,
            "z": float(self.z.numpy()[r, c, t]),
            "doubt": float(self.doubt.numpy()[r, c, t]),
            "reachable": v < cap,
            "reachable_if_certain": vc < cap,
            "unreachable_by_ignorance": v >= cap and vc < cap,
        }
