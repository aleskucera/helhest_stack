"""Pose algebra for the node and the scan deskew.

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


def _so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → axis-angle vector `omega` (‖omega‖ = angle).

    First-order fallback near zero; the theta≈π branch is skipped because the
    per-sweep rotations this is used for are always small.
    """
    cos_theta = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1.0e-9:
        return 0.5 * axis  # sin(theta) ≈ theta → omega ≈ ½·vee(R − Rᵀ)
    return (theta / (2.0 * np.sin(theta))) * axis


def deskew_scan(points: np.ndarray, alphas: np.ndarray, sweep_delta: np.ndarray) -> np.ndarray:
    """Motion-compensate a swept cloud to the sweep-end pose (constant velocity).

    A spinning LiDAR measures each point at a slightly different instant, so a
    moving robot smears the sweep. `points` (N, 3, base frame) were measured at
    sweep fractions `alphas` (N,) in [0, 1] — 0 = sweep start, 1 = sweep end
    (the reference). `sweep_delta` is `base_start_T_base_end`, the base motion
    over the sweep (i.e. the odom frame-to-frame delta). Returns every point
    re-expressed in the sweep-end base frame.

    Constant-velocity model: interpolate rotation on the screw axis and shift
    translation linearly. Derivation — with `d = sweep_delta = [R_d | t_d]` and
    `R(α) = exp(α·log R_d)`, the point measured at α maps to the end frame by
    `p' = R_dᵀ · (R(α)·p + (α−1)·t_d)`. (α=1 → identity; α=0 → inv(d).)
    """
    if points.shape[0] == 0:
        return points.reshape(0, 3)
    R_delta = sweep_delta[:3, :3]
    t_delta = sweep_delta[:3, 3]
    omega = _so3_log(R_delta)
    theta = float(np.linalg.norm(omega))
    if theta < 1.0e-9:
        rotated = points  # R(α) ≈ I for every point
    else:
        axis = omega / theta
        angle = (alphas * theta)[:, None]
        # Rodrigues applied per point (shared axis, per-point angle α·θ).
        cross1 = np.cross(axis, points)
        cross2 = np.cross(axis, cross1)
        rotated = points + np.sin(angle) * cross1 + (1.0 - np.cos(angle)) * cross2
    shifted = rotated + (alphas[:, None] - 1.0) * t_delta
    return shifted @ R_delta  # right-multiply by R_delta ≡ apply R_deltaᵀ per row
