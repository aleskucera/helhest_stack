# terrain_value_field

**Cost-to-go over a terrain belief, where feasibility is measured in standard deviations of margin.**

Give it per-state constraints with their uncertainties and it returns a value function over every state — plus the safety margin and doubt fields it was built from. It estimates and it stops: no controller, no exploration policy, no goal-versus-explore arbitration. Those are decisions about what a particular robot should do, and a robot that disagrees should not have to fork a planner to say so.

Built on [NVIDIA Warp](https://github.com/NVIDIA/warp). Everything stays on the GPU.

## The idea

Every constraint on a state is asked how much room is left **in units of its own uncertainty**:

```
z_i = margin_i / max(σ_i, floor_i)
z   = min over i                     ← the binding constraint
```

Dividing by each constraint's own σ is what makes the `min` meaningful. A tilt margin is in radians and a clearance margin in metres; a raw `min` over those compares nothing. In sigmas they are the same quantity, and the smallest genuinely is the one about to be violated.

One knob follows — `k_sigma`, how many standard deviations of room the robot insists on — and the graded penalty comes off the same number, so *how pessimistic am I* and *how close is this to bad* are not two separately-tuned things that fight.

`floor_i` is not optional. Without it a perfectly known map makes a state at 14.9° of roll against a 15° limit read as infinitely safe. The floor is the irreducible error — localisation, controller tracking, model mismatch — that no map improvement removes.

## Doubt: ignorance is not bad ground

Every state is scored twice — once against the believed map, once as if σ were at the floor:

| | meaning |
|---|---|
| fails both | **bad ground.** Looking at it cannot help. |
| passes when certain, fails when believed | **blocked by ignorance.** Worth resolving. |

`solve_pair` returns both value functions. Their difference at a state is what ignorance costs *there*, in the same units as the plan cost — the signal neither solve gives alone, since pessimistically not knowing is expensive everywhere and optimistically it is free everywhere.

What a robot *does* about that is not this library's business. It publishes the field.

## The one seam

Nothing here knows what a constraint means. A producer supplies `(margin, σ, floor)` per state and the arithmetic is identical:

| producer | margins |
|---|---|
| `GeometricProducer` (shipped) | `max_slope − slope`, `max_step − step` |
| physics (a settle, a contact solve) | tilt limits, belly clearance |
| learned | whatever it scores, with its own σ |

With σ = 0 everywhere it degenerates to ordinary deterministic value iteration — a useful baseline to have for free.

## Not a lattice planner

The state space *is* a [state lattice](https://www.ri.cmu.edu/pub_files/pub4/pivtoraiko_mihail_2007_1/pivtoraiko_mihail_2007_1.pdf) — nodes plus a control set of feasible local moves. But lattice planners search it with A\* and use a precomputed cost-to-go as the *heuristic*; this computes that cost-to-go and stops. Every state updates from the previous sweep with no priority queue and no ordering, so it is one GPU thread per state.

The output is a field you query from wherever the robot actually is, not a trajectory it must re-attach to.

Two control sets ship, and the solver never learns which it was given:

- `arc_control_set` — forward arcs capped by a minimum turn radius, optional point turns. Orientation matters, so states are (x, y, heading).
- `omni_control_set` — eight neighbours at one heading bin. With `n_theta = 1` the same kernel is grid value iteration for a holonomic robot. No special case.

An arc is integrated in continuous space and then snapped to the lattice, so it records the heading the robot actually reaches only if its turn lands on a bin boundary. When it does not, every move is off by up to half a bin — and since the margin field is indexed by heading, feasibility gets checked at a pose the robot will not occupy. `closing_step(n_theta, turn_radius, bins)` gives a step that closes (`bins` even, because the half-rate arcs have to close too), and `arc_control_set` warns when handed one that does not. A move costs the **realized** arc length through its snapped endpoint rather than the nominal step, so no curvature gets a rounding discount on ground covered; `turn_weight` [m per rad] is charged on top of that, and `pivot_cost` [m per heading bin] buys point turns.

Seeds are a **mask**, not a goal cell: value iteration takes many sources for free where a graph search needs a virtual node. One seed is goal-seeking, a seeded frontier is exploration, a seeded set of docks is "reach any of these".

## Usage

```python
import numpy as np, warp as wp
from terrain_value_field import TerrainValueField, build_grid, omni_control_set
from terrain_value_field.producers import GeometricProducer

grid  = build_grid(rows, cols, 0.1, origin_x, origin_y)
field = TerrainValueField(rows, cols, 0.1, n_theta=1,
                          k_sigma=2.0, control_set=omni_control_set(0.1))
produce = GeometricProducer(rows, cols, 1, max_slope_rad=0.45, max_step_m=0.15)

constraints = produce(height, height_sd, grid)    # both wp.array [rows, cols]
field.seed_cell(goal_row, goal_col)
V, V_certain = field.solve_pair(constraints)      # device-resident

state = field.at(robot_row, robot_col)
# {'v', 'v_certain', 'gap', 'z', 'doubt', 'reachable',
#  'reachable_if_certain', 'unreachable_by_ignorance'}
```

## The shipped producer is not a toy

A great many robots decide traversability exactly this way — too steep, or too tall a step — and for them this is the whole feasibility model with the uncertainty handled properly.

One detail worth knowing, because the obvious implementation is wrong: **step is the largest departure from the local plane, not the footprint's peak-to-trough.** On a smooth 20° incline an 0.8 m footprint spans 0.29 m top to bottom with no step present, so a peak-to-trough measure re-reports the slope and the two constraints stop being independent. Removing the plane the slope term already fitted leaves roughness, which is what a step limit is about. `test_a_plane_has_no_step_however_steep` pins it.

## Tests

```sh
PYTHONPATH=src python -m pytest tests -q
```

56 tests. Lattice closure and the cost model are pinned by their own file — the arc cost is checked against a numerically integrated circle rather than the closed form it uses. The margin reduction is checked against the algebra rather than another implementation; the producer against planes and steps of known geometry; and there is a guard, with tests, for a map whose shape disagrees with its grid — that reads out of bounds and returns *plausible nonsense* rather than failing, which cost real debugging time.
