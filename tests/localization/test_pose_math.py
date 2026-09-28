"""Unit tests for the pose algebra (`localization.pose_math`).

Pure numpy — no rclpy — so this runs without a ROS install.

Run: python tests/localization/test_pose_math.py
"""

from __future__ import annotations

import numpy as np
from helhest.localization.pose_math import invert_pose
from helhest.localization.pose_math import matrix_to_quaternion


def _rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _pose(roll: float, pitch: float, yaw: float, t: tuple[float, float, float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy_to_R(roll, pitch, yaw)
    T[:3, 3] = t
    return T


def _random_poses(n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    out = []
    for _ in range(n):
        rpy = rng.uniform(-np.pi, np.pi, 3)
        t = rng.uniform(-10.0, 10.0, 3)
        out.append(_pose(rpy[0], rpy[1], rpy[2], tuple(t)))
    return out


def test_invert_pose() -> None:
    for T in _random_poses(20):
        assert np.allclose(T @ invert_pose(T), np.eye(4), atol=1e-9)


def test_matrix_to_quaternion() -> None:
    # Round-trip each branch of Shepperd's method against a known good path:
    # rebuild the rotation from the quaternion and compare.
    for T in _random_poses(20):
        x, y, z, w = matrix_to_quaternion(T)
        # quaternion (x,y,z,w) -> rotation matrix
        n = x * x + y * y + z * z + w * w
        s = 2.0 / n
        R = np.array(
            [
                [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
            ]
        )
        assert np.allclose(R, T[:3, :3], atol=1e-9), T[:3, :3]


def main() -> None:
    test_invert_pose()
    test_matrix_to_quaternion()
    print("PASS: pose_math (2 tests)")


if __name__ == "__main__":
    main()
