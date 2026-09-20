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
    out = [wp.zeros(shape, dtype=wp.float32) for _ in range(2)]
    wp.launch(
        M.classify_kernel,
        dim=shape,
        inputs=[az, azc, wp.array(np.array([k], np.float32), dtype=wp.float32), z_ref, w],
        outputs=out,
    )
    pose_cost, doubt = (float(o.numpy()[0, 0, 0]) for o in out)
    return [*_decode(pose_cost), doubt]


def _decode(pose_cost):
    """(blocked, penalty) from the sign-encoded field -- see margin.POSE COST."""
    pc = np.asarray(pose_cost)
    return np.where(pc < 0.0, 1.0, 0.0), np.where(pc < 0.0, -1.0 - pc, pc)


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


def test_fusion_matches_the_split_pair():
    """The fused kernel must reproduce the readable pair exactly, not approximately.

    Fusing is a performance change and nothing else, so anything but bit-identical output means
    the two have drifted apart and one of them is now wrong.
    """
    rng = np.random.default_rng(7)
    shape = (5, 9, 11, 3)  # constraints, rows, cols, headings
    out3 = shape[1:]
    margin = rng.uniform(-0.3, 0.6, shape).astype(np.float32)
    margin[0, ::7] = float(M.IGNORED)  # exercise the opt-out branch too
    mar = wp.array(margin, dtype=wp.float32)
    sig = wp.array(rng.uniform(0.0, 0.2, shape).astype(np.float32), dtype=wp.float32)
    flo = wp.array(rng.uniform(0.01, 0.05, shape[0]).astype(np.float32), dtype=wp.float32)
    k = wp.array(np.array([2.0], np.float32), dtype=wp.float32)
    z_ref, w = 4.0, 1.5

    split = [wp.zeros(out3, dtype=wp.float32) for _ in range(4)]
    wp.launch(M.margin_to_z_kernel, dim=out3, inputs=[mar, sig, flo], outputs=split[:2])
    wp.launch(
        M.classify_kernel,
        dim=out3,
        inputs=[split[0], split[1], k, z_ref, w],
        outputs=split[2:],
    )

    fused = [wp.zeros(out3, dtype=wp.float32) for _ in range(4)]
    wp.launch(
        M.margin_to_fields_kernel,
        dim=out3,
        inputs=[mar, sig, flo, k, z_ref, w],
        outputs=fused,
    )

    names = ("z", "z_certain", "pose_cost", "doubt")
    for name, a, b in zip(names, split, fused):
        np.testing.assert_array_equal(a.numpy(), b.numpy(), err_msg=f"{name} drifted")
    assert (fused[2].numpy() < 0).any(), "the scene must actually block something"
    assert (fused[3].numpy() > 0).any(), "and produce some doubt, or this proves little"


def test_the_pose_cost_encoding_loses_nothing_at_a_vetoed_state():
    """Why the veto rides in the SIGN rather than replacing the value with a sentinel.

    A blocked state still has a graded penalty, and it is still exactly recoverable. That is
    what lets `_free_seeds_kernel` un-block a seeded state by flipping the sign back instead of
    inventing a number for it.
    """
    blocked, penalty, _ = _classify(1.0, 99.0)  # z=1 < k=2, so vetoed, and z < z_ref so graded
    assert blocked == 1.0
    assert penalty == pytest.approx(3.0, rel=1e-5)  # w * (z_ref - z) = 1.0 * (4 - 1)
    # and the round trip is exact, not approximate
    encoded = -1.0 - penalty
    assert float(_decode(encoded)[1]) == pytest.approx(penalty, rel=0, abs=0)


def test_a_state_every_constraint_declines_to_judge_reads_as_perfect_ground():
    """The trap the module docstring warns about, pinned so the warning cannot go stale.

    `IGNORED` means "this constraint does not apply here", and a state where every constraint
    declines comes out unvetoed, unpenalised and undoubted -- identical to flat, certain, ideal
    terrain. A producer that spells "unmeasured" this way gets silent optimism. Unmeasured
    belongs in the sigma, where it produces a veto AND a doubt signal.
    """
    shape = (2, 1, 1, 1)  # two constraints, one state
    mar = wp.array(np.full(shape, float(M.IGNORED), np.float32), dtype=wp.float32)
    sig = wp.array(np.full(shape, 0.5, np.float32), dtype=wp.float32)
    flo = wp.array(np.array([0.01, 0.01], np.float32), dtype=wp.float32)
    out = [wp.zeros(shape[1:], dtype=wp.float32) for _ in range(4)]
    wp.launch(
        M.margin_to_fields_kernel,
        dim=shape[1:],
        inputs=[mar, sig, flo, wp.array(np.array([2.0], np.float32), dtype=wp.float32), 4.0, 1.0],
        outputs=out,
    )
    z, _, pose_cost, doubt = (float(o.numpy()[0, 0, 0]) for o in out)
    assert z >= float(M.IGNORED), "nothing spoke, so nothing bounds the margin"
    assert pose_cost >= 0.0, "NOT vetoed"
    assert pose_cost == pytest.approx(0.0), "and not even penalised"
    assert doubt == pytest.approx(0.0), "and no doubt raised -- this is the silent part"
