"""The three-wheel settle, linearised: d(z, pitch, roll) / d(wheel contact heights).

Moved here from `studies/bench/clark.py` when the contact estimator entered `src/`. It is pure
robot geometry read off `RobotParams` -- nothing study-specific -- and both the risk estimator
and the study benchmarks now read this one definition.
"""

from __future__ import annotations

import numpy as np


def settle_map(rp) -> np.ndarray:
    """d(z, pitch, roll) / d(env_L, env_R, env_rear), rows (z, pitch, roll), cols (L, R, rear).

    Small-angle IFT of the 3-wheel settle: with wheel body positions L=(0,+b), R=(0,-b),
    rear=(-l,0) and R = Rz(yaw) Ry(pitch) Rx(roll), the z-component of R @ wheel_i is, to first
    order in (pitch, roll) and for ANY yaw (yaw only rotates x,y, never z):
        z_final_i ~= -pitch * p_ix + roll * p_iy
    so contact_z_i = z + z_final_i - wheel_radius = env_i gives 3 linear equations in
    (z, pitch, roll); solved once, symbolically, below. Time- and yaw-INDEPENDENT.
    """
    b, l = rp.half_track, rp.rear_offset
    return np.array(
        [
            [0.5, 0.5, 0.0],
            [-0.5 / l, -0.5 / l, 1.0 / l],
            [1.0 / (2.0 * b), -1.0 / (2.0 * b), 0.0],
        ]
    )
