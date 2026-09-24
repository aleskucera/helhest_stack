"""Reverse is allowed only over ground the robot has measured behind it."""

from __future__ import annotations

import numpy as np
import warp as wp

from helhest.control.reverse_gate import ReverseGate
from helhest.control.reverse_gate import strip_measured_fraction


def _mask(n=60, measured_x_min=None):
    m = np.ones((n, n), np.float32)
    if measured_x_min is not None:  # nothing measured west of this column (in metres at 0.2)
        m[:, : int(measured_x_min / 0.2)] = 0.0
    return m


def test_the_kernel_matches_the_host_reference():
    g = ReverseGate(device="cuda")
    for yaw in (0.0, 0.7, 2.5, -1.9):
        m = np.random.default_rng(1).random((60, 60)).astype(np.float32) > 0.4
        m = m.astype(np.float32)
        pose = (6.03, 6.07, yaw)  # off the cell lattice: no sample sits on a grid line
        g.clear(wp.array(m, device="cuda"), (0.0, 0.0), 0.2, pose)
        assert abs(g.fraction - strip_measured_fraction(m, (0.0, 0.0), 0.2, pose)) < 0.02


def test_open_only_when_the_strip_behind_is_measured():
    g = ReverseGate(device="cuda")
    # facing east at x = 6: the strip lies at x 4.5..5.7. Measured from x = 3 on: open.
    assert g.clear(
        wp.array(_mask(measured_x_min=3.0), device="cuda"), (0.0, 0.0), 0.2, (6.0, 6.0, 0.0)
    )
    # measured from x = 5 on: half the strip is blind -> locked
    assert not g.clear(
        wp.array(_mask(measured_x_min=5.0), device="cuda"), (0.0, 0.0), 0.2, (6.0, 6.0, 0.0)
    )
    # the start of a run: nothing behind is measured
    assert not g.clear(
        wp.array(np.zeros((60, 60), np.float32), device="cuda"), (0.0, 0.0), 0.2, (6.0, 6.0, 0.0)
    )


def test_off_the_window_counts_as_blind():
    g = ReverseGate(device="cuda")
    # facing west near the east edge: the strip behind runs off the window
    assert not g.clear(wp.array(_mask(), device="cuda"), (0.0, 0.0), 0.2, (11.5, 6.0, np.pi))
