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

from . import hierarchical as _hier
from . import margin as _margin
from ...grid import Grid
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


@wp.kernel
def _seed_doubt_kernel(
    doubt: wp.array3d(dtype=wp.float32),  # [row, col, heading]
    min_doubt: wp.float32,
    inf: wp.float32,
    seeds: wp.array3d(dtype=wp.float32),  # [row, col, heading]
):
    """Make every sufficiently doubted state a zero-cost source.

    The value field that comes out is then "how far to the nearest thing worth looking at",
    which is the other half of the explore-or-push trade: the gap says what ignorance costs,
    this says what it costs to go and resolve some of it.
    """
    r, c, t = wp.tid()
    seeds[r, c, t] = wp.where(doubt[r, c, t] > min_doubt, 0.0, inf)


class TerrainValueField:
    def __init__(
        self,
        rows: int,
        cols: int,
        resolution: float,
        n_theta: int = 16,
        *,
        z_veto: float = 2.0,
        z_charge: float = 4.0,
        charge_per_sigma: float = 0.5,
        penalty_scale: float = 1.0,
        control_set: tuple | None = None,
        turn_radius: float = 0.6,
        step: float | None = None,
        turn_weight: float = 0.0,
        pivot_cost: float | None = None,  # None = 8x the equal-turn arc; inf = none
        sweep_spacing: float | None = None,  # [m] between checked poses; see arc_control_set
        free_blocked_seeds: bool = True,  # False when seeds come from a coarser layer
        device: wp.Device | str | None = None,
    ) -> None:
        """`z_veto` is the one knob: how many standard deviations of room a state must hold.

        Two penalty knobs, and they act in different places -- read them together once:

        `z_veto` and `z_charge` are both denominated in SIGMAS, which means neither survives a
        change of `floor` unless it is rescaled with it. Halving a floor halves every sigma and
        doubles every z, so the same two numbers then describe a different band entirely -- a
        caller who dropped a floor from 2 cm to 0.5 cm found the penalty band go from 8.8 degrees
        of roll wide to 1.1, i.e. a ramp too narrow to steer by, with no error anywhere. If you
        move a floor, move these with it.

        `charge_per_sigma` defaulted to 0 -- veto only -- which makes feasibility a CLIFF: a state
        at 2.01 sigmas is free and one at 1.99 is impossible, with no gradient in between, so
        nothing prefers five sigmas of room to two. Measured on a robot driving rough ground, the
        planner parked it AT the edge and stalled, because sitting there cost nothing. A penalty
        cannot make a state unreachable, only make roomy ground cheaper than marginal ground, and
        it is what gives `z_charge` anything to do at all.

        `charge_per_sigma` is charged in the MARGIN kernel. It turns "how many sigmas of room is
        left below `z_charge`" into a per-state cost, so it sets the units. `charge_per_sigma = 0`
        gives a pure veto and no gradient at all.

        `penalty_scale` is applied in the SOLVER. It multiplies the mean per-state cost along a
        move's swept cells into that move's price. It trades route length against room: 0 makes
        the field a pure shortest path over the unvetoed states.

        Both must be >= 0. The solver's own parameter is also called `penalty_scale`; nothing
        downstream of here is called `charge_per_sigma`.
        """
        # Both knobs must be non-negative, and not as a matter of taste. `charge_per_sigma` < 0
        # makes the graded penalty negative, and `pose_cost` carries the veto in its sign -- so
        # every free state would read as vetoed and the whole map would go unreachable, silently.
        # `penalty_scale` < 0 can drive a move's cost below zero, which breaks min-plus outright.
        if charge_per_sigma < 0.0:
            raise ValueError(
                f"charge_per_sigma must be >= 0 (the veto rides in the sign of the "
                f"graded cost; see margin.POSE COST), got {charge_per_sigma}"
            )
        if penalty_scale < 0.0:
            raise ValueError(
                f"penalty_scale must be >= 0 or a move can cost less than "
                f"nothing, got {penalty_scale}"
            )
        self.device = wp.get_device(device)
        self.rows, self.cols, self.n_theta = int(rows), int(cols), int(n_theta)
        self.resolution = float(resolution)
        self.z_veto = float(z_veto)
        self.z_charge = float(z_charge)
        self.charge_per_sigma = float(charge_per_sigma)
        self.penalty_scale = float(penalty_scale)
        self.solver_inf = 1.0e30  # the "unreachable"/"not a seed" sentinel

        shape = (self.rows, self.cols, self.n_theta)
        with wp.ScopedDevice(self.device):
            self.z = wp.zeros(shape, dtype=wp.float32)
            self.z_certain = wp.zeros(shape, dtype=wp.float32)
            self.hard = wp.zeros(shape, dtype=wp.float32)  # 1 = a hard constraint fails
            # graded cost with the veto in the sign; see margin.POSE COST
            self.pose_cost = wp.zeros(shape, dtype=wp.float32)
            self.penalty = wp.zeros(shape, dtype=wp.float32)  # the graded cost alone, exact
            self.doubt = wp.zeros(shape, dtype=wp.float32)
            self.V = wp.zeros(shape, dtype=wp.float32)
            self.V_certain = wp.zeros(shape, dtype=wp.float32)
            # +inf = not a seed. zeros would mean EVERY state is a free goal. Public so device-side
            # seeders (`hierarchical`) can write it inside a captured graph.
            self.seeds = wp.full(shape, float(self.solver_inf), dtype=wp.float32)
            # a device scalar, so a captured graph can be retuned without re-recording it
            self._k = wp.array([self.z_veto], dtype=wp.float32)

        self.solver = ValueSolver(
            self.resolution,
            self.rows,
            self.cols,
            n_theta=self.n_theta,
            turn_radius=turn_radius,
            step=step,
            turn_weight=turn_weight,
            pivot_cost=pivot_cost,
            sweep_spacing=sweep_spacing,
            control_set=control_set,
            free_blocked_seeds=free_blocked_seeds,
            device=self.device,
        )

    def set_z_veto(self, z_veto: float) -> None:
        """Retune the veto. A host-to-device copy, so call it outside any graph capture; a graph
        recorded around `classify` reads the new value on its next replay."""
        self.z_veto = float(z_veto)
        self._k.assign(np.array([self.z_veto], np.float32))

    # -- seeds ----------------------------------------------------------------------------
    def seed_states(self, mask: np.ndarray | wp.array) -> None:
        """Zero-cost sources from a 0/1 mask, `[rows, cols, headings]`, non-zero = seed.

        A MASK rather than a single goal cell, because value iteration takes multiple sources for
        free where a graph search would need a virtual node. One seeded state is goal-seeking; a
        seeded frontier is exploration; a seeded set of docks is "reach any of these". Which of
        those a robot wants is not this library's decision -- it only has to be expressible.
        """
        m = mask.numpy() if isinstance(mask, wp.array) else np.asarray(mask)
        self.seed_values(np.where(m > 0.5, 0.0, self.solver_inf).astype(np.float32))

    def seed_cell(self, row: int, col: int) -> None:
        """Seed one cell at every heading -- the ordinary "drive to here" case."""
        v = np.full((self.rows, self.cols, self.n_theta), self.solver_inf, np.float32)
        v[int(row), int(col), :] = 0.0
        self.seed_values(v)

    def seed_values(self, values: np.ndarray | wp.array) -> None:
        """Seed with COSTS rather than a mask: `[rows, cols, headings]`, +inf = not a seed.

        The seed field is the initial value function. A goal mask is the special case where every
        source costs 0; what this adds is sources that already carry a price. That is what a
        coarser layer hands down -- "leaving the window here still costs you this much" -- and it
        is what makes the fine solve prefer the right exit instead of treating every boundary cell
        as equally good. See `seed_from_coarse`.
        """
        if isinstance(values, wp.array):
            wp.copy(self.seeds, values)
        else:
            self.seeds.assign(np.ascontiguousarray(values, dtype=np.float32))

    def seed_from_coarse(
        self,
        coarse_value: wp.array,
        coarse_grid: Grid,
        fine_grid: Grid,
        goal_xy: tuple[float, float],
        band: int | None = None,
    ) -> None:
        """Seed this window's border from a coarser layer's cost-to-go. See `hierarchical`.

        `fine_grid` says where the window sits in the world THIS frame -- it moves with the robot,
        so it is an argument rather than state. `goal_xy` is in the same world frame: a goal inside
        the window is seeded alone and the ring is not, since the heading-free coarse layer
        understates distance in this layer's metric. `band` is the ring thickness in cells and
        defaults to the furthest a single move reaches, because a thinner ring can be jumped
        clean over.
        """
        if coarse_value.shape[2] != 1:
            raise ValueError(
                f"the coarse layer is expected to be heading-free (n_theta=1, omni_control_set); "
                f"got {coarse_value.shape[2]} headings"
            )
        # the kernel works from min corners; the grids carry cell centres
        fine_x0 = float(fine_grid.origin_x) - 0.5 * float(fine_grid.cell_size)
        fine_y0 = float(fine_grid.origin_y) - 0.5 * float(fine_grid.cell_size)
        coarse_x0 = float(coarse_grid.origin_x) - 0.5 * float(coarse_grid.cell_size)
        coarse_y0 = float(coarse_grid.origin_y) - 0.5 * float(coarse_grid.cell_size)
        wp.launch(
            _hier.seed_goal_and_ring_kernel,
            dim=(self.rows, self.cols, self.n_theta),
            inputs=[
                wp.array(np.asarray(goal_xy[:2], np.float32), device=self.device),
                coarse_value,
                wp.array(np.array([coarse_x0, coarse_y0], np.float32), device=self.device),
                float(coarse_grid.cell_size),
                fine_x0,
                fine_y0,
                float(fine_grid.cell_size),
                int(self.solver.reach_cells if band is None else band),
                float(self.solver_inf),
            ],
            outputs=[self.seeds],
            device=self.device,
        )

    def seed_doubt(self, min_doubt: float = 0.0) -> None:
        """Seed every state whose `doubt` exceeds `min_doubt`. Solve after this and the field is
        the cost of reaching the nearest state that is blocked by ignorance rather than by ground.

        `doubt` is whatever the last solve left behind, so this follows a solve rather than
        replacing one.
        """
        wp.launch(
            _seed_doubt_kernel,
            dim=(self.rows, self.cols, self.n_theta),
            inputs=[self.doubt, float(min_doubt), float(self.solver_inf)],
            outputs=[self.seeds],
            device=self.device,
        )

    # -- solve ----------------------------------------------------------------------------
    def classify(self, constraints: Constraints, certain: bool = False) -> wp.array:
        """Reduce the constraints and classify: fills `z`, `z_certain`, `hard`, `pose_cost`,
        `penalty` and `doubt`, and returns `pose_cost`. No host sync and no host-to-device copy, so it can be
        recorded into a larger graph; a robot that edits `pose_cost` before `iterate` (erosion,
        costs of its own) does it here, between the two.

        `certain=True` scores every state as if the map carried no uncertainty beyond the
        floors. That is not a cheaper approximation -- it is the second half of the pair that
        makes `doubt` meaningful, and on its own it is the optimistic reading.
        """
        wp.launch(
            _margin.margin_to_fields_kernel,
            dim=(self.rows, self.cols, self.n_theta),
            inputs=[
                constraints.margin,
                constraints.sigma,
                constraints.floor,
                self._k,
                self.z_charge,
                self.charge_per_sigma,
            ],
            outputs=[
                self.z,
                self.z_certain,
                self.hard,
                self.pose_cost,
                self.penalty,
                self.doubt,
            ],
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
                    self.hard,
                    self._k,
                    self.z_charge,
                    self.charge_per_sigma,
                ],
                outputs=[self.pose_cost, self.penalty, self.doubt],
                device=self.device,
            )
        return self.pose_cost

    def iterate(self, capture: bool = True) -> wp.array:
        """Value-iterate `pose_cost` from `seeds` to a fixed point. Returns the solver's buffer,
        device-resident and overwritten by the next solve; `solve` copies it into `V`.

        `capture=False` inside a caller's own capture: `ValueSolver.value_iterate` would
        otherwise try to open a second one. Its loop still becomes a device-side conditional
        node of the outer graph.
        """
        return self.solver.value_iterate(
            self.pose_cost, self.seeds, self.penalty_scale, capture=capture
        )

    def solve(self, constraints: Constraints, certain: bool = False) -> wp.array:
        """`classify` then `iterate`. Returns `V` (device-resident)."""
        self.classify(constraints, certain=certain)
        wp.copy(self.V, self.iterate())
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

    def value_of_looking(
        self,
        constraints: Constraints,
        row: int,
        col: int,
        heading_bin: int = 0,
        min_doubt: float = 0.0,
    ) -> dict:
        """Should the robot go and look, or push on? Three solves and the trade between them.

        `gap` is what ignorance costs at this state, in the plan's own units: solve believing the
        map, solve as if it were certain, subtract. It is an UPPER BOUND on what any amount of
        looking could save, because the certain solve is what the robot would do if every doubt
        were resolved in its favour.

        `cost_to_look` is what it costs to get somewhere that would resolve some of it -- the same
        field seeded on the doubted states instead of the goal, solved optimistically because the
        route to a place you have not seen runs through places you have not seen.

        Looking is worth it when the gap exceeds the detour. Both sides are metres, so the
        comparison needs no tuned threshold -- but `worth_looking` is a convenience and the
        decision belongs to the robot: a detour costs time and risk this does not model.

        Three cases have no number. `blocked_by_ignorance` is the strongest signal there is:
        believing the map the goal is unreachable, and it would be reachable if the doubts went
        the robot's way -- not knowing is costing the entire route, so `gap` is infinite. If
        neither solve reaches the goal the map is genuinely walled off and there is nothing to
        learn. And if nothing is doubted there is nowhere to go and `cost_to_look` is infinite.
        """
        self.solve_pair(constraints)
        here = self.at(row, col, heading_bin)
        n_doubted = int((self.doubt.numpy() > min_doubt).sum())

        parked = {n: wp.clone(getattr(self, n)) for n in ("V", "z", "pose_cost", "doubt")}
        parked_seeds = wp.clone(self.seeds)
        self.seed_doubt(min_doubt)
        # optimistic: the way to somewhere unseen runs through the unseen, so the believed
        # reading would refuse to plan the very trip this is costing
        look = float(self.solve(constraints, certain=True).numpy()[row, col, heading_bin])
        for n, buf in parked.items():
            wp.copy(getattr(self, n), buf)
        wp.copy(self.seeds, parked_seeds)

        cap = self.unreachable_value()
        blocked_by_ignorance = here["unreachable_by_ignorance"]
        gap = float("inf") if blocked_by_ignorance else here["gap"]
        cost_to_look = look if look < cap else float("inf")
        return {
            "gap": gap,
            "cost_to_look": cost_to_look,
            "worth_looking": gap > cost_to_look,
            "blocked_by_ignorance": blocked_by_ignorance,
            "walled_off": not here["reachable"] and not here["reachable_if_certain"],
            "doubted_states": n_doubted,
            "v": here["v"],
            "v_certain": here["v_certain"],
        }

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
