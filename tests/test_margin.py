"""The margin reduction: constraints in whatever units, out in sigmas of room.

This is the library's one idea, so it is tested against the arithmetic rather than against
another implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from terrain_value_field import margin as M


def _reduce(margins, sigmas, floors, shape=(1, 1, 1)):
    n = len(margins)
    mar = np.zeros((n, *shape), np.float32)
    sig = np.zeros((n, *shape), np.float32)
    for i, (m, s) in enumerate(zip(margins, sigmas)):
        mar[i], sig[i] = m, s
    z = wp.zeros(shape, dtype=wp.float32)
    zc = wp.zeros(shape, dtype=wp.float32)
    wp.launch(
        M.margin_to_z_kernel,
        dim=shape,
        inputs=[
            wp.array(mar, dtype=wp.float32),
            wp.array(sig, dtype=wp.float32),
            wp.array(np.array(floors, np.float32), dtype=wp.float32),
        ],
        outputs=[z, zc],
    )
    return float(z.numpy()[0, 0, 0]), float(zc.numpy()[0, 0, 0])


def test_single_constraint_is_margin_over_sigma():
    z, _ = _reduce([0.30], [0.10], [0.01])
    assert z == pytest.approx(3.0, rel=1e-5)


def test_the_binding_constraint_wins_regardless_of_units():
    """A tilt margin in radians and a clearance margin in metres are comparable only in sigmas.

    Here the clearance margin is numerically the LARGER number but the tighter constraint, and
    a reduction that compared raw margins would pick the wrong one.
    """
    z, _ = _reduce([0.05, 0.30], [0.01, 0.20], [0.001, 0.001])
    assert z == pytest.approx(1.5, rel=1e-5), "0.30/0.20 must bind over 0.05/0.01"


def test_the_floor_bounds_confidence_on_a_perfect_map():
    """Without it, a state a hair inside its limit reads as infinitely safe."""
    z_floored, _ = _reduce([0.05], [0.0], [0.02])
    assert z_floored == pytest.approx(2.5, rel=1e-5)


def test_floors_are_per_constraint():
    """Each constraint's floor is in its own units; one global floor would be meaningless."""
    z, _ = _reduce([0.20, 0.20], [0.0, 0.0], [0.02, 0.10])
    assert z == pytest.approx(2.0, rel=1e-5), "the constraint with the larger floor must bind"


def test_a_constraint_may_decline_to_speak():
    """Silence is not evidence of safety, but it must not veto either."""
    z, _ = _reduce([float(M.IGNORED), 0.30], [0.0, 0.10], [0.01, 0.01])
    assert z == pytest.approx(3.0, rel=1e-5)


def test_certain_reading_uses_only_the_floor():
    z, zc = _reduce([0.20], [0.10], [0.02])
    assert z == pytest.approx(2.0, rel=1e-5)
    assert zc == pytest.approx(10.0, rel=1e-5)


def test_a_map_at_the_floor_has_nothing_left_to_learn():
    z, zc = _reduce([0.20], [0.005], [0.02])
    assert z == pytest.approx(zc, rel=1e-6)


def _classify(z, zc, k=2.0, z_ref=4.0, w=1.0):
    shape = (1, 1, 1)
    az = wp.array(np.full(shape, z, np.float32), dtype=wp.float32)
    azc = wp.array(np.full(shape, zc, np.float32), dtype=wp.float32)
    out = [wp.zeros(shape, dtype=wp.float32) for _ in range(3)]
    wp.launch(
        M.classify_kernel,
        dim=shape,
        inputs=[az, azc, wp.array(np.array([k], np.float32), dtype=wp.float32), z_ref, w],
        outputs=out,
    )
    return [float(o.numpy()[0, 0, 0]) for o in out]


@pytest.mark.parametrize("z,expect", [(1.99, 1.0), (2.01, 0.0)])
def test_the_veto_switches_at_k_sigma(z, expect):
    blocked, _, _ = _classify(z, 99.0)
    assert blocked == expect


def test_the_penalty_is_hinged_at_z_ref():
    assert _classify(5.0, 99.0)[1] == pytest.approx(0.0)
    assert _classify(3.0, 99.0)[1] == pytest.approx(1.0, rel=1e-5)


def test_doubt_is_ignorance_and_only_ignorance():
    """Bad ground fails on any map, so looking at it cannot help and it carries no doubt."""
    assert _classify(1.0, 0.5)[2] == pytest.approx(0.0), "fails even when certain: bad ground"
    assert _classify(1.0, 9.0)[2] == pytest.approx(8.0, rel=1e-5), "passes when certain: ignorance"
    assert _classify(9.0, 9.0)[2] == pytest.approx(0.0), "not blocked at all"
