"""The settle producer reads sigma under the wheels of the pose it settled, at that pose's heading.

It used to sample at the bin MIDPOINT, (t + 0.5) * dth, while the settle placed the robot at
t * dth: a margin in sigmas assembled from one pose's attitude and another pose's map uncertainty.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from helhest.engine import GridParams
from helhest.engine import RobotParams
from helhest.engine import SolverParams
from helhest.planning.settle_producer import CLIMB
from helhest.planning.settle_producer import ROLL
from helhest.planning.settle_producer import SettleProducer

N, CELL, NT = 21, 0.24, 4  # four headings: half a bin off is 45 deg, far from any wheel
FLOOR = 0.005


def _sigma_at_rear_wheel_of_heading_zero() -> tuple[SettleProducer, int]:
    robot = RobotParams()
    grid = GridParams(N, N, CELL, -N * CELL / 2, -N * CELL / 2)
    p = SettleProducer(grid, robot, SolverParams(), NT, FLOOR)
    centre = N // 2  # pose (centre, centre) sits at world (0, 0)
    xs = -N * CELL / 2 + (np.arange(N) + 0.5) * CELL
    xx, yy = np.meshgrid(xs, xs)
    sd = np.where(np.hypot(xx + robot.rear_offset, yy) < 0.2, 0.5, 0.0).astype(np.float32)
    zeros = wp.zeros((N, N), dtype=wp.float32)
    ones = wp.full((N, N), 1.0, dtype=wp.float32)
    p.settle(zeros)
    p.run(zeros, ones, wp.array(sd, dtype=wp.float32), zeros, wp.array([1.0], dtype=wp.float32))
    return p, centre


def test_sigma_is_read_under_the_rear_wheel_of_the_heading_that_was_settled():
    p, c = _sigma_at_rear_wheel_of_heading_zero()
    sigma = p.sigma.numpy()
    floor_pitch = float(p.floor.numpy()[CLIMB])
    # heading 0: the rear wheel sits on the uncertain spot, so pitch is uncertain
    assert sigma[CLIMB, c, c, 0] > 10.0 * floor_pitch
    # heading 180 deg: the rear wheel is 1.5 m away, on ground at the floor
    assert sigma[CLIMB, c, c, 2] == floor_pitch
    # roll differences the two front wheels only, and neither is on the spot
    assert sigma[ROLL, c, c, 0] == float(p.floor.numpy()[ROLL])
