"""The shape guard: a map that disagrees with the grid must fail loudly.

`locate` clamps to the GRID's extent, not the array's, so a mismatched map reads out of bounds
and returns plausible nonsense -- on a (1, N) row broadcast as an (N, N) map it reported a
0.31 m step where the true plane residual was 0.03 m, and every downstream field followed it
without complaint. Silent and wrong is the worst failure mode a library can have.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from terrain_value_field import build_grid
from terrain_value_field.producers import GeometricProducer

N, CELL = 21, 0.1


def _wp(a):
    return wp.array(np.ascontiguousarray(a, np.float32), dtype=wp.float32)


def test_a_broadcast_row_is_rejected():
    g = build_grid(N, N, CELL, 0.0, 0.0)
    p = GeometricProducer(N, N, 1)
    with pytest.raises(ValueError, match="height is"):
        p(_wp(np.zeros((1, N))), _wp(np.zeros((N, N))), g)


def test_a_mismatched_sigma_map_is_rejected():
    g = build_grid(N, N, CELL, 0.0, 0.0)
    p = GeometricProducer(N, N, 1)
    with pytest.raises(ValueError, match="height_sd is"):
        p(_wp(np.zeros((N, N))), _wp(np.zeros((N, N + 2))), g)


def test_a_producer_built_for_another_grid_is_rejected():
    g = build_grid(N, N, CELL, 0.0, 0.0)
    p = GeometricProducer(N + 4, N + 4, 1)
    with pytest.raises(ValueError, match="producer built for"):
        p(_wp(np.zeros((N, N))), _wp(np.zeros((N, N))), g)


def test_matching_shapes_pass():
    g = build_grid(N, N, CELL, 0.0, 0.0)
    p = GeometricProducer(N, N, 1)
    cons = p(_wp(np.zeros((N, N))), _wp(np.zeros((N, N))), g)
    assert tuple(cons.margin.shape) == (2, N, N, 1)
