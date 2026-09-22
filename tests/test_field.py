"""The solver, the geometric producer, and the two together.

The producer tests matter more than they look: `step` was originally peak-to-trough over the
footprint, which re-reported slope as roughness and made the two constraints redundant. The
plane tests below are what pin the fix.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from terrain_value_field import build_grid
from terrain_value_field import omni_control_set
from terrain_value_field import TerrainValueField
from terrain_value_field.producers import GeometricProducer

N = 41
CELL = 0.1


def _grid():
    return build_grid(N, N, CELL, -N * CELL / 2, -N * CELL / 2)


def _field(**kw):
    kw.setdefault("z_veto", 2.0)
    return TerrainValueField(N, N, CELL, n_theta=1, control_set=omni_control_set(CELL), **kw)


def _produce(height, sd, **kw):
    p = GeometricProducer(N, N, 1, **kw)
    return p(
        wp.array(np.ascontiguousarray(height, np.float32), dtype=wp.float32),
        wp.array(np.ascontiguousarray(sd, np.float32), dtype=wp.float32),
        _grid(),
    )


# Cell centres exactly as the producer computes them, so terrain built here lands on the grid.
XS = -N * CELL / 2 + np.arange(N) * CELL


def _plane(slope_rad: float) -> np.ndarray:
    return np.tile(XS * np.tan(slope_rad), (N, 1))


def _ridge(amp: float = 0.35, k: float = 1.1) -> np.ndarray:
    """A full (N, N) map. Broadcasting a single row instead is a shape bug the producer now
    rejects -- it used to read out of bounds and return plausible nonsense."""
    return np.tile(amp * np.sin(XS * k), (N, 1))


# --- the producer --------------------------------------------------------------------------


@pytest.mark.parametrize("slope_deg", [0.0, 10.0, 20.0])
def test_slope_margin_is_correct_on_a_known_plane(slope_deg):
    cons = _produce(_plane(np.radians(slope_deg)), np.zeros((N, N)), max_slope_rad=np.radians(30.0))
    inner = cons.margin.numpy()[0, 8:-8, 8:-8, 0]
    expected = np.radians(30.0 - slope_deg)
    assert inner.mean() == pytest.approx(expected, abs=np.radians(0.5))


@pytest.mark.parametrize("slope_deg", [0.0, 10.0, 20.0])
def test_a_plane_has_no_step_however_steep(slope_deg):
    """The fix. Peak-to-trough over a 0.8 m footprint on a 20 degree incline reads 0.29 m of
    'step' with no step present; a plane residual reads zero, as it must."""
    cons = _produce(_plane(np.radians(slope_deg)), np.zeros((N, N)), max_step_m=0.15)
    step_margin = cons.margin.numpy()[1, 8:-8, 8:-8, 0]
    assert step_margin.mean() == pytest.approx(
        0.15, abs=0.01
    ), "a plane must leave the step budget intact"


def test_a_real_step_is_caught():
    h = np.zeros((N, N), np.float32)
    h[:, N // 2 :] = 0.25  # a 25 cm wall, well over the 15 cm limit
    cons = _produce(h, np.zeros((N, N)), max_step_m=0.15)
    step_margin = cons.margin.numpy()[1, :, :, 0]
    assert step_margin.min() < 0.0, "the step must break its budget somewhere"
    near = step_margin[:, N // 2 - 3 : N // 2 + 3]
    assert near.min() < 0.0 and near.min() == pytest.approx(step_margin.min(), abs=1e-3)


def test_map_uncertainty_propagates_into_the_constraint_sigmas():
    flat = np.zeros((N, N))
    a = _produce(flat, np.full((N, N), 0.01)).sigma.numpy()[:, 8:-8, 8:-8, 0]
    b = _produce(flat, np.full((N, N), 0.02)).sigma.numpy()[:, 8:-8, 8:-8, 0]
    ratio = b.mean(axis=(1, 2)) / a.mean(axis=(1, 2))
    assert ratio.tolist() == pytest.approx([2.0] * len(ratio), rel=1e-3)


def test_a_wider_footprint_lowers_the_slope_uncertainty():
    """The baseline divides the differenced sd, so a wide footprint is less uncertain, not only
    more conservative."""
    flat, sd = np.zeros((N, N)), np.full((N, N), 0.02)
    narrow = _produce(flat, sd, footprint_m=0.2).sigma.numpy()[0, 8:-8, 8:-8, 0].mean()
    wide = _produce(flat, sd, footprint_m=0.6).sigma.numpy()[0, 8:-8, 8:-8, 0].mean()
    assert wide < narrow / 2.0


# --- the solver ----------------------------------------------------------------------------


def test_free_ground_gives_a_distance_field():
    f = _field()
    cons = _produce(np.zeros((N, N)), np.zeros((N, N)))
    f.seed_cell(N // 2, N // 2)
    v = f.solve(cons).numpy()[:, :, 0]
    assert v[N // 2, N // 2] == pytest.approx(0.0, abs=1e-6)
    # Eight-connected with sqrt(2) diagonals approximates Euclidean to a few percent.
    for dc in (5, 10, 15):
        assert v[N // 2, N // 2 + dc] == pytest.approx(dc * CELL, rel=0.05)


def test_a_wall_makes_the_far_side_unreachable():
    h = np.zeros((N, N), np.float32)
    h[:, N // 2] = 1.0  # a full-height barrier across the map
    f = _field()
    f.seed_cell(N // 2, 2)
    v = f.solve(_produce(h, np.zeros((N, N)))).numpy()[:, :, 0]
    cap = f.unreachable_value()
    assert v[N // 2, 5] < cap, "the near side must stay reachable"
    assert v[N // 2, N - 3] >= cap, "the far side must not"


def test_multiple_seeds_are_free():
    """Value iteration takes many sources where a graph search would need a virtual node."""
    f = _field()
    mask = np.zeros((N, N, 1), np.float32)
    mask[N // 2, 2] = 1.0
    mask[N // 2, N - 3] = 1.0
    f.seed_states(mask)
    v = f.solve(_produce(np.zeros((N, N)), np.zeros((N, N)))).numpy()[:, :, 0]
    assert v[N // 2, 2] == pytest.approx(0.0, abs=1e-6)
    assert v[N // 2, N - 3] == pytest.approx(0.0, abs=1e-6)
    mid = v[N // 2, N // 2]
    assert mid == pytest.approx((N // 2 - 2) * CELL, rel=0.1), "the midpoint is equidistant"


# --- the pair ------------------------------------------------------------------------------


def test_zero_uncertainty_degenerates_to_deterministic_planning():
    f = _field()
    cons = _produce(np.zeros((N, N)), np.zeros((N, N)))
    f.seed_cell(N // 2, N // 2)
    f.solve_pair(cons)
    assert (f.pose_cost.numpy() < 0).mean() == 0.0
    assert (f.doubt.numpy() == 0.0).all()
    np.testing.assert_allclose(f.V.numpy(), f.V_certain.numpy(), rtol=1e-6)


def test_uncertainty_blocks_by_ignorance_and_the_certain_solve_says_so():
    h = _ridge() + np.tile(0.25 * np.cos(XS * 0.9), (N, 1)).T
    f = _field()
    f.seed_cell(N // 2, N - 3)
    f.solve_pair(_produce(h, np.full((N, N), 0.05)))
    st = f.at(N // 2, 2)
    assert not st["reachable"]
    assert st["reachable_if_certain"]
    assert st["unreachable_by_ignorance"]
    assert (f.doubt.numpy() > 0).mean() > 0.5


def test_solve_pair_leaves_the_believed_reading_in_place():
    """Every field but V_certain must describe the believed map, or the object lies about itself."""
    cons_args = (_ridge(), np.full((N, N), 0.04))
    f = _field()
    f.seed_cell(N // 2, N - 3)
    f.solve(_produce(*cons_args))
    believed = {n: getattr(f, n).numpy().copy() for n in ("V", "z", "pose_cost", "doubt")}
    f.solve_pair(_produce(*cons_args))
    for n, was in believed.items():
        np.testing.assert_allclose(getattr(f, n).numpy(), was, rtol=1e-5, err_msg=n)
    assert not np.allclose(f.V_certain.numpy(), f.V.numpy())


def test_a_negative_charge_per_sigma_is_rejected():
    """It would invert the sign encoding: every FREE state would read as vetoed and the whole
    map would go unreachable, with nothing to show for it. See margin.POSE COST."""
    with pytest.raises(ValueError, match="charge_per_sigma"):
        _field(charge_per_sigma=-1.0)


def test_a_negative_penalty_scale_is_rejected():
    """A move that costs less than nothing breaks min-plus outright."""
    with pytest.raises(ValueError, match="penalty_scale"):
        _field(penalty_scale=-1.0)


def test_the_solver_rejects_a_negative_scale_at_its_own_boundary():
    from terrain_value_field.solver import ValueSolver

    s = ValueSolver(CELL, 8, 8, n_theta=1, control_set=omni_control_set(CELL))
    buf = wp.zeros((8, 8, 1), dtype=wp.float32)
    with pytest.raises(ValueError, match="penalty_scale"):
        s.value_iterate(buf, buf, -0.5)
