"""The two `Grid` structs carry the same five numbers and mean different things by `origin`.

`helhest.engine.terrain.Grid` and `terrain_value_field.grid.Grid` both hold cells_x, cells_y,
cell_size, origin_x, origin_y. helhest's origin is the MIN CORNER of the map; tvf's is the CENTRE
of cell (0, 0). They are half a cell apart.

They are distinct warp types, so passing one where the other is expected is a type error and the
easy mistake is caught. The dangerous one is not: passing loose floats -- an origin read off a
`GridParams` and handed to a kernel that does tvf's arithmetic -- shifts everything by half a
cell, silently, with no shape or type to disagree about. `CostToGo.set_coarse` already hand-rolls
its cell-centre arithmetic for exactly this reason rather than pass a struct across the boundary.

This file pins both conventions so neither can drift, and states the offset between them so the
conversion is written down somewhere executable. The real fix is one struct and one convention;
until that lands, this is what stops the two quietly diverging further.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import helhest.engine.terrain as ht
import terrain_value_field.grid as tg
from helhest.engine.terrain import GridParams
from terrain_value_field.grid import build_grid

CELLS, CELL = 10, 1.0
PROBE = 5.0  # mid-grid, away from the border clamp that hides the difference at (0, 0)


@wp.kernel
def _probe_helhest(g: ht.Grid, x: wp.float32, out: wp.array(dtype=wp.float32)):
    c = ht._locate(g, x, x)
    out[0] = c[0]  # cell index
    out[1] = c[2]  # fraction into it


@wp.kernel
def _probe_tvf(g: tg.Grid, x: wp.float32, out: wp.array(dtype=wp.float32)):
    c = tg.locate(g, x, x)
    out[0] = c[0]
    out[1] = c[2]


@wp.kernel
def _probe_world_of(g: tg.Grid, row: wp.int32, col: wp.int32, out: wp.array(dtype=wp.float32)):
    p = tg.world_of(g, row, col)
    out[0] = p[0]
    out[1] = p[1]


def _run(kernel, *inputs) -> np.ndarray:
    out = wp.zeros(2, dtype=wp.float32)
    wp.launch(kernel, dim=1, inputs=list(inputs), outputs=[out])
    return out.numpy().copy()


def test_helhest_origin_is_the_min_corner():
    """`(x - origin)/cell - 0.5`: the centre of cell i sits at origin + (i + 0.5) * cell."""
    g = GridParams(CELLS, CELLS, CELL, 0.0, 0.0).build()
    cell, frac = _run(_probe_helhest, g, PROBE)
    assert (cell, frac) == (4.0, 0.5), "world 5.0 should be halfway between cells 4 and 5"


def test_tvf_origin_is_the_centre_of_cell_zero():
    """`(x - origin)/cell`: the centre of cell i sits at origin + i * cell."""
    g = build_grid(CELLS, CELLS, CELL, 0.0, 0.0)
    cell, frac = _run(_probe_tvf, g, PROBE)
    assert (cell, frac) == (5.0, 0.0), "world 5.0 should land exactly on cell 5"
    wx, wy = _run(_probe_world_of, g, 0, 0)
    assert (wx, wy) == (0.0, 0.0), "and cell (0,0)'s centre is the origin itself"


def test_the_two_are_exactly_half_a_cell_apart():
    """The number a conversion has to carry. Stated here so it is executable rather than folklore.

    Same five values into both: the index they report differs by 0.5 cells throughout, which is
    what makes a loose-float hand-off between the two silently wrong rather than loudly wrong.
    """
    gp = GridParams(CELLS, CELLS, CELL, 0.0, 0.0)
    for probe in (2.5, 5.0, 7.25):
        h = _run(_probe_helhest, gp.build(), probe)
        t = _run(_probe_tvf, build_grid(CELLS, CELLS, CELL, 0.0, 0.0), probe)
        assert (h[0] + h[1]) + 0.5 == (t[0] + t[1]), f"at {probe}: {h} vs {t}"


def test_the_conversion_that_makes_them_agree():
    """Shifting the origin by half a cell aligns them, which is what a `from_corner` would do."""
    gp = GridParams(CELLS, CELLS, CELL, 0.0, 0.0)
    shifted = build_grid(CELLS, CELLS, CELL, gp.origin_x + CELL / 2, gp.origin_y + CELL / 2)
    for probe in (2.5, 5.0, 7.25):
        h = _run(_probe_helhest, gp.build(), probe)
        t = _run(_probe_tvf, shifted, probe)
        np.testing.assert_allclose(h, t, atol=1e-6, err_msg=f"at {probe}")
