"""Two layers: a cheap global field that says which way, a fine local one that says how.

Cost goes as roughly the cube of the span, because the state count is quadratic in it and the
number of sweeps is linear -- information crosses one primitive per sweep. So halving the window
is worth about 8x, and that is enough to buy heading resolution rather than merely pocket it.
Measured on an A500 at 0.2 m cells, corner seed: a 40 m window at 16 headings takes 121 ms, a
20 m window at 32 headings takes 40 ms, a 14 m window at 32 headings takes 14 ms.

The window should be about as large as the map is INFORMATIVE, not as large as the map. On Odin's
dTOF, the accumulated map ahead of the robot into new terrain is complete to 4 m, 74-85% at 6 m,
41-49% at 8 m and 12-14% at 12 m (`out_odin0`, 141 m driven). Past that a bigger fine window is
not planning over terrain, it is planning over whatever filled the unobserved cells.

The composition:

    coarse.solve(coarse_constraints, certain=True)   # whole map, omni_control_set, n_theta=1
    fine.seed_from_coarse(coarse.V, coarse_grid, fine_grid, goal_xy)
    fine.solve(fine_constraints)

`certain=True` on the coarse layer is not a shortcut, it is the point: the optimistic reading
scores states as if the map carried no uncertainty beyond the floors, so global routing is not
blocked by ignorance about terrain the robot has not reached yet. Deciding whether that ignorance
is actually a problem is the fine layer's job, once the robot is close enough to have measured it.

Two failure modes worth knowing before relying on this. The coarse layer can promise a route the
fine layer then refuses, and the robot oscillates between them; keeping the coarse layer strictly
more permissive than the fine one is what prevents that, which is the other reason for
`certain=True`. And the seed ring has to be thicker than one cell or an arc can step straight over
it -- see `ValueSolver.reach_cells`, which is what the default band comes from.
"""

from __future__ import annotations

import warp as wp


@wp.kernel
def goal_cell_kernel(
    goal_xy: wp.array(dtype=wp.float32),  # [2], in the same frame as the origin
    origin_x: wp.float32,
    origin_y: wp.float32,
    cell_size: wp.float32,
    rows: wp.int32,
    cols: wp.int32,
    goal_rc: wp.array(dtype=wp.int32),  # [2]
):
    """Resolve and CLAMP the goal into the window, so a goal beyond it becomes a carrot at the
    edge rather than no goal at all. `origin` is the window's min corner, so `int` floors to the
    cell that contains the goal. On device, so a captured graph can take a new goal per replay."""
    c = int((goal_xy[0] - origin_x) / cell_size)
    r = int((goal_xy[1] - origin_y) / cell_size)
    goal_rc[0] = wp.clamp(r, 0, rows - 1)
    goal_rc[1] = wp.clamp(c, 0, cols - 1)


@wp.kernel
def seed_goal_kernel(
    goal_rc: wp.array(dtype=wp.int32),  # [2]
    inf: wp.float32,
    seeds: wp.array3d(dtype=wp.float32),  # [row, col, heading]
):
    """Seed the goal cell at every heading, on DEVICE.

    The goal cell is resolved inside the captured graph (`goal_cell_kernel`), so the seeding has
    to be too -- reading it back to call a host-side seeder would put a sync in the middle of the
    graph and defeat the point of capturing it.
    """
    r, c, t = wp.tid()
    seeds[r, c, t] = wp.where(r == goal_rc[0] and c == goal_rc[1], 0.0, inf)


@wp.kernel
def seed_goal_and_ring_kernel(
    goal_xy: wp.array(dtype=wp.float32),  # [2], this window's frame -- UNCLAMPED on purpose
    coarse_value: wp.array3d(dtype=wp.float32),  # [cy, cx, 1], the coarse layer's cost-to-go
    coarse_origin: wp.array(dtype=wp.float32),  # [2], the coarse grid in THIS window's frame
    coarse_cell: wp.float32,
    origin_x: wp.float32,  # this window's own origin, same frame
    origin_y: wp.float32,
    cell_size: wp.float32,
    band: wp.int32,  # ring thickness, in fine cells
    inf: wp.float32,
    seeds: wp.array3d(dtype=wp.float32),  # [row, col, heading]
):
    """Seed the goal AND the window's border, in one pass because each writes every cell.

    A border cell is seeded at what the coarse layer says it costs to carry on from there, so the
    fine solve pays the real price of each exit and prefers the right one instead of treating
    every way out of the window as equally good. Fused with the goal seeding because each writes
    every cell, and on device because the goal is resolved inside a captured graph.

    The goal is taken UNCLAMPED here, unlike the single-layer path. Clamping an out-of-window goal
    onto the border is what makes one layer work at all -- it becomes a carrot the window drags
    along -- but with a coarse layer it is actively wrong: a zero-cost seed on the border beats
    every finite coarse value, so the ring is overridden and the robot chases the exit nearest the
    goal even when the coarse layer knows that exit is a dead end. When the goal is outside, the
    ring IS the goal information and nothing else should be seeded.

    The coarse value is read from the NEAREST coarse cell, never interpolated: unreachable cells
    hold +inf, and blending that with a finite neighbour yields a large finite number -- a cell
    that reads as reachable at an invented price, which is worse than either truth.

    """
    r, c, t = wp.tid()
    rows = seeds.shape[0]
    cols = seeds.shape[1]
    gc = int((goal_xy[0] - origin_x) / cell_size)
    gr = int((goal_xy[1] - origin_y) / cell_size)
    if gr >= 0 and gr < rows and gc >= 0 and gc < cols:
        # The goal is in the window, so the window is not missing anything and the ring is not
        # seeded at all. It would not merely be redundant: the coarse layer is omnidirectional and
        # pays no turn cost, so it UNDERSTATES distance in the fine layer's own metric, and a ring
        # priced that way reads as a shortcut. The fine solve would route the robot out of the
        # window and back to reach a goal sitting a few metres in front of it. Seeding the ring
        # at a 5 m margin instead was tried: no better on false_door (2/3 either way), and it
        # took pocket from 3/3 to 2/3 with a run three times slower.
        seeds[r, c, t] = wp.where(r == gr and c == gc, 0.0, inf)
        return
    v = inf
    if r < band or r >= rows - band or c < band or c >= cols - band:
        # origin + c*cell, NOT + (c + 0.5)*cell: the cost-to-go places a pose at
        # `origin + c * cell` and resolves the goal the same way. It feeds a NEAREST-cell read of
        # the coarse field rather than a smooth interpolation, so at --coarsen 1 a 0.1 m offset
        # flips the rounding for about half the ring and reads a neighbour's value.
        x = origin_x + float(c) * cell_size
        y = origin_y + float(r) * cell_size
        cc = int(wp.round((x - coarse_origin[0]) / coarse_cell))
        cr = int(wp.round((y - coarse_origin[1]) / coarse_cell))
        if cr >= 0 and cr < coarse_value.shape[0] and cc >= 0 and cc < coarse_value.shape[1]:
            v = coarse_value[cr, cc, 0]
    seeds[r, c, t] = v
