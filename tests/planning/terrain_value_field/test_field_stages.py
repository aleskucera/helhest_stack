"""The field in stages: classify, (the robot's edits), iterate -- and all of it in one graph.

A robot that keeps its whole frame on the device records seeding, classification and the value
iteration into ONE captured graph and replays it with new inputs. That only works if nothing in
those stages touches the host, and if every input that changes per frame -- the goal, the veto,
where a world-anchored coarse layer sits -- is read from device memory at replay time.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from helhest.planning.terrain_value_field import TerrainValueField
from helhest.planning.terrain_value_field.field import Constraints
from helhest.planning.terrain_value_field.hierarchical import goal_cell_kernel
from helhest.planning.terrain_value_field.hierarchical import seed_goal_and_ring_kernel
from helhest.planning.terrain_value_field.hierarchical import seed_goal_kernel

N, NT, CELL = 24, 8, 0.2
COARSE_N, COARSE_CELL = 8, 0.8


def _constraints() -> Constraints:
    """A soft margin that narrows toward a ridge, and a hard constraint failing in a block."""
    rng = np.random.default_rng(3)
    soft = rng.uniform(0.01, 0.3, (N, N, NT)).astype(np.float32)
    soft[:, N // 2, :] = -0.05  # a soft wall down the middle ...
    soft[N // 3 : N // 3 + 3, N // 2, :] = 0.2  # ... with a doorway in it
    hard = np.ones((N, N, NT), np.float32)
    hard[2:5, 2:5, :] = -1.0
    margin = np.stack([soft, hard])
    sigma = np.stack([np.full((N, N, NT), 0.05, np.float32), np.zeros((N, N, NT), np.float32)])
    return Constraints(
        margin=wp.array(margin, dtype=wp.float32),
        sigma=wp.array(sigma, dtype=wp.float32),
        floor=wp.array([0.01, 0.0], dtype=wp.float32),
    )


def _field() -> TerrainValueField:
    return TerrainValueField(N, N, CELL, n_theta=NT, z_veto=2.0, turn_radius=0.4)


def test_the_stages_are_solve():
    c = _constraints()
    a, b = _field(), _field()
    a.seed_cell(N - 2, N - 2)
    b.seed_cell(N - 2, N - 2)
    whole = a.solve(c).numpy()
    b.classify(c)
    staged = b.iterate().numpy()
    np.testing.assert_array_equal(whole, staged)
    assert (whole < 1e29).any() and (whole >= 1e29).any(), "the scene must block something"
    assert (b.hard.numpy() > 0).sum() == 9 * NT


class _Frame:
    """One device-resident frame: seed on device, classify, iterate. Recordable."""

    def __init__(self, ring: bool) -> None:
        self.f = _field()
        self.c = _constraints()
        self.ring = ring
        self.goal_xy = wp.zeros(2, dtype=wp.float32)
        self.goal_rc = wp.zeros(2, dtype=wp.int32)
        self.coarse_origin = wp.zeros(2, dtype=wp.float32)
        yy, xx = np.mgrid[0:COARSE_N, 0:COARSE_N]
        cv = (np.hypot(xx - 7, yy - 3) * COARSE_CELL).astype(np.float32)[:, :, None]
        self.coarse_value = wp.array(cv, dtype=wp.float32)
        self.V = wp.zeros((N, N, NT), dtype=wp.float32)

    def run(self) -> None:
        f = self.f
        if self.ring:
            wp.launch(
                seed_goal_and_ring_kernel,
                dim=(N, N, NT),
                inputs=[
                    self.goal_xy,
                    self.coarse_value,
                    self.coarse_origin,
                    COARSE_CELL,
                    0.0,
                    0.0,
                    CELL,
                    2,
                    float(f.solver_inf),
                ],
                outputs=[f.seeds],
            )
        else:
            wp.launch(
                goal_cell_kernel,
                dim=1,
                inputs=[self.goal_xy, 0.0, 0.0, CELL, N, N],
                outputs=[self.goal_rc],
            )
            wp.launch(
                seed_goal_kernel,
                dim=(N, N, NT),
                inputs=[self.goal_rc, f.solver_inf],
                outputs=[f.seeds],
            )
        f.classify(self.c)
        wp.copy(self.V, f.iterate(capture=False))

    def set(self, goal: tuple[float, float], z_veto: float, origin=(0.0, 0.0)) -> None:
        self.goal_xy.assign(np.array(goal, np.float32))
        self.coarse_origin.assign(np.array(origin, np.float32))
        self.f.set_z_veto(z_veto)


@pytest.mark.parametrize("ring", [False, True])
def test_one_graph_follows_the_goal_the_veto_and_the_coarse_origin(ring):
    if not wp.get_device().is_cuda:
        pytest.skip("graph capture needs CUDA")
    frame = _Frame(ring)
    frame.set((1.0, 1.0), 2.0)
    with wp.ScopedCapture() as cap:
        frame.run()

    # ring=True with the goal outside the window: the ring alone carries the goal
    changes = [((1.0, 1.0), 2.0, (0.0, 0.0)), ((3.9, 0.5), 0.5, (-1.6, -0.8))]
    if ring:
        changes = [((9.0, 2.0), 2.0, (0.0, 0.0)), ((9.0, 2.0), 0.5, (-1.6, -0.8))]
    results = []
    for goal, k, origin in changes:
        frame.set(goal, k, origin)
        wp.capture_launch(cap.graph)
        replayed = frame.V.numpy().copy()
        frame.run()  # the same frame, eagerly
        np.testing.assert_array_equal(replayed, frame.V.numpy(), err_msg=f"{goal} {k} {origin}")
        results.append(replayed)
    assert not np.array_equal(results[0], results[1]), "the change must matter, or this is empty"


@pytest.mark.parametrize("factor", [1, 3, 4])
def test_the_ring_reads_the_coarse_cell_under_each_fine_cell(factor):
    """A ring cell is seeded with the coarse value of the block its CENTRE lies in. Rounding the
    min corner against the coarse min corner instead is right only when the cells match: at a
    factor k it reads 0.5 * (1 - 1/k) coarse cells toward +x/+y."""
    n, fine = 12, 0.24
    coarse = fine * factor
    cn = n // factor + 4
    c_origin = -2.0 * coarse  # the coarse grid starts two blocks before the window, in its frame
    cols = np.broadcast_to(np.arange(cn, dtype=np.float32), (cn, cn))
    value = wp.array(np.ascontiguousarray(cols[:, :, None]), dtype=wp.float32)
    seeds = wp.zeros((n, n, 1), dtype=wp.float32)
    wp.launch(
        seed_goal_and_ring_kernel,
        dim=(n, n, 1),
        inputs=[
            wp.array([1.0e6, 1.0e6], dtype=wp.float32),  # the goal is outside: the ring alone
            value,
            wp.array([c_origin, c_origin], dtype=wp.float32),
            coarse,
            0.0,
            0.0,
            fine,
            n,  # the whole window is ring
            1.0e30,
        ],
        outputs=[seeds],
    )
    got = seeds.numpy()[0, :, 0]
    centres = (np.arange(n) + 0.5) * fine
    want = np.floor((centres - c_origin) / coarse).astype(np.float32)
    np.testing.assert_array_equal(got, want)
