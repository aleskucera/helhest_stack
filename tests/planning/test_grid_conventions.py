"""One grid convention, and the conversion that keeps the host-facing API where it was.

There used to be two. `helhest.engine.terrain.Grid` took `origin` to be the map's MIN CORNER and
`terrain_value_field.grid.Grid` took it to be the CENTRE of cell (0, 0) -- the same five fields,
half a cell apart, as distinct warp types. The type difference caught the easy mistake and hid
the dangerous one: a kernel taking loose FLOATS shifts a whole map by half a cell with no shape
or type to disagree about. Both defects that produced were real. `_margin_kernel` placed a pose
at `origin + c*cell` and then sampled sigma through the min-corner `_locate`; and the cost-to-go's
boundary ring read the coarse field half a cell off, which mattered because it feeds a
NEAREST-cell lookup rather than an interpolation (518876a).

So the kernels now share tvf's convention, and `GridParams.build()` adds the half cell once. That
keeps `GridParams.origin_x` meaning what a caller measuring a window computes -- the min corner --
while everything on the device sees cell centres.

What these tests pin is that the conversion is exact. `(x - (o + c/2))/c` is algebraically the
`(x - o)/c - 0.5` it replaced, so sampling through `GridParams` must be unchanged to the bit;
anything else means a caller was reading the struct's origin directly and assuming the old
meaning.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import helhest.engine.terrain as ht
import terrain_value_field.grid as tg
from helhest.engine.terrain import GridParams
from terrain_value_field.grid import build_grid

CELLS, CELL = 10, 1.0
# mid-grid: at world (0, 0) the border clamp pulls both to cell 0, which hides any disagreement
PROBES = (2.5, 5.0, 7.25)


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


def _run(kernel, *inputs) -> np.ndarray:
    out = wp.zeros(2, dtype=wp.float32)
    wp.launch(kernel, dim=1, inputs=list(inputs), outputs=[out])
    return out.numpy().copy()


def _params() -> GridParams:
    return GridParams(CELLS, CELLS, CELL, 0.0, 0.0)


def test_build_moves_the_origin_to_the_centre_of_cell_zero():
    """The half cell is added once, here, and nowhere else."""
    p = _params()
    g = p.build()
    assert g.origin_x == p.origin_x + CELL / 2
    assert g.origin_y == p.origin_y + CELL / 2


def test_sampling_through_grid_params_is_what_it_always_was():
    """The regression guard for the whole change: a min-corner origin plus the old `-0.5` and a
    centre origin plus no offset are the same arithmetic, so these are the values the old code
    produced. If one moves, a caller read the struct's origin and assumed the old meaning."""
    g = _params().build()
    for probe, want_cell, want_frac in ((2.5, 2.0, 0.0), (5.0, 4.0, 0.5), (7.25, 6.0, 0.75)):
        cell, frac = _run(_probe_helhest, g, probe)
        assert (cell, frac) == (want_cell, want_frac), f"at {probe}"


def test_the_two_locates_now_agree_given_the_same_struct():
    """The point of the merge. Hand both the identical grid and they place a point identically --
    which is what makes it safe for one struct to serve both, and what was not true before."""
    g_h = _params().build()
    g_t = build_grid(CELLS, CELLS, CELL, g_h.origin_x, g_h.origin_y)
    for probe in PROBES:
        np.testing.assert_allclose(
            _run(_probe_helhest, g_h, probe), _run(_probe_tvf, g_t, probe), atol=1e-6
        )


def test_a_raw_min_corner_origin_still_reads_half_a_cell_out():
    """Why the conversion has to live in `build()` and not in the caller's head. Passing a
    min-corner origin straight to the device -- the loose-float hand-off that caused both real
    defects -- still lands half a cell from where `build()` would put it."""
    p = _params()
    raw = build_grid(CELLS, CELLS, CELL, p.origin_x, p.origin_y)  # forgot the half cell
    for probe in PROBES:
        good = _run(_probe_helhest, p.build(), probe)
        bad = _run(_probe_tvf, raw, probe)
        assert (good[0] + good[1]) + 0.5 == (bad[0] + bad[1]), f"at {probe}: {good} vs {bad}"
