"""Pose algebra for the node.

Pure numpy, **no rclpy** — so it is unit-testable without a ROS install. Poses are 4x4
homogeneous SE(3) matrices `T` such that a point in the source frame maps to the target frame as
`T @ [x, y, z, 1]`.
"""

from __future__ import annotations

import numpy as np


def invert_pose(T: np.ndarray) -> np.ndarray:
    """Inverse of an SE(3) pose, exploiting `R^-1 = R^T` (no general solve)."""
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=T.dtype)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def matrix_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix (top-left 3x3 of a pose) → quaternion `(x, y, z, w)`.

    Shepperd's method: pick the largest diagonal branch for numerical stability.
    """
    m = R[:3, :3]
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)
