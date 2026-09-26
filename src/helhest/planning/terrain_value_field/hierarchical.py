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
    fine.seed_from_coarse(coarse.V, coarse_grid, fine_grid)
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

from ...grid import Grid
from ...grid import world_of


@wp.kernel
def boundary_seeds_kernel(
    coarse_value: wp.array3d(dtype=wp.float32),  # [rows, cols, 1]
    coarse: Grid,
    fine: Grid,
    band: wp.int32,
    inf: wp.float32,
    seeds: wp.array3d(dtype=wp.float32),  # [rows, cols, heading]
):
    """Seed the window's border with what the coarse layer says it costs to leave there.

    Interior cells are not seeds. A border cell is seeded at the coarse cost-to-go read at its
    world position, so the fine solve pays the real price of each exit and prefers the right one
    instead of treating every way out as equally good.

    The coarse value is read from the NEAREST cell rather than interpolated. Unreachable cells
    hold +inf, and blending that with a finite neighbour would produce a large finite number --
    a cell that reads as reachable at an invented price, which is worse than either truth.
    """
    r, c, t = wp.tid()
    rows = seeds.shape[0]
    cols = seeds.shape[1]
    if r >= band and r < rows - band and c >= band and c < cols - band:
        seeds[r, c, t] = inf
        return
    p = world_of(fine, r, c)
    cc = int(wp.round((p[0] - coarse.origin_x) / coarse.cell_size))
    cr = int(wp.round((p[1] - coarse.origin_y) / coarse.cell_size))
    if cr < 0 or cr >= coarse.cells_y or cc < 0 or cc >= coarse.cells_x:
        seeds[r, c, t] = inf  # the window overhangs the coarse map: no exit known here
    else:
        seeds[r, c, t] = coarse_value[cr, cc, 0]
