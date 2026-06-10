"""Differential-drive twist + SE(2) integration (numpy reference, batched).

The no-slip skid-steer kinematics with friction-dependent turning lumped into
two ICR parameters:

    vx  = R (wL + wR) / 2
    wz  = R (wR - wL) / (2 b alpha)        # alpha >= 1 widens the effective track
    vy  = -x_ICR * wz                      # lateral drift from a longitudinal ICR offset

In Phase 1 (alpha, x_ICR) are scalar constants; in Phase 4 they come from the
friction field via the moment-centroid map. The rear wheel is kinematically
redundant here (consistent when driven at the L/R average) and does not enter
the twist.

Conventions: body X fwd, Y left, yaw CCW+. Arrays carry a leading batch dim B.
"""
import numpy as np

from .model import HALF_TRACK
from .model import WHEEL_RADIUS


def wheel_twist(omega, alpha=1.0, x_icr=0.0, R=WHEEL_RADIUS, b=HALF_TRACK):
    """Body twist (vx, vy, wz) from wheel speeds.

    omega : [..., 3] angular velocities [left, right, rear] (rad/s).
    alpha, x_icr : scalars or [...] broadcastable arrays.
    Returns vx, vy, wz each shaped like omega[..., 0].
    """
    omega = np.asarray(omega, dtype=np.float64)
    wL, wR = omega[..., 0], omega[..., 1]
    vx = R * (wL + wR) / 2.0
    wz = R * (wR - wL) / (2.0 * b * np.asarray(alpha, dtype=np.float64))
    vy = -np.asarray(x_icr, dtype=np.float64) * wz
    return vx, vy, wz


def se2_integrate(pose, vx, vy, wz, dt):
    """Exact SE(2) exponential integration of a constant body twist over dt.

    pose : [..., 3] = (x, y, yaw). vx, vy, wz : [...]. Returns new pose [..., 3].
    """
    x, y, psi = pose[..., 0], pose[..., 1], pose[..., 2]
    theta = wz * dt

    small = np.abs(wz) < 1e-9
    wz_safe = np.where(small, 1.0, wz)
    s, c = np.sin(theta), np.cos(theta)
    # Body-frame displacement (left Jacobian of SO(2) applied to the linear vel).
    dxb = np.where(small, vx * dt, (vx * s + vy * (c - 1.0)) / wz_safe)
    dyb = np.where(small, vy * dt, (vx * (1.0 - c) + vy * s) / wz_safe)

    cp, sp = np.cos(psi), np.sin(psi)
    x_new = x + cp * dxb - sp * dyb
    y_new = y + sp * dxb + cp * dyb
    psi_new = psi + theta
    return np.stack([x_new, y_new, psi_new], axis=-1)


def pose2_to_pose7(pose2, z=WHEEL_RADIUS):
    """Planar pose (x, y, yaw) [..., 3] -> SE(3) pose7 (px,py,pz, qx,qy,qz,qw).

    Flat-ground placement: chassis origin at z = wheel radius, level (Phase 1).
    """
    x, y, psi = pose2[..., 0], pose2[..., 1], pose2[..., 2]
    zz = np.full_like(x, z)
    qx = np.zeros_like(x)
    qy = np.zeros_like(x)
    qz = np.sin(psi / 2.0)
    qw = np.cos(psi / 2.0)
    return np.stack([x, y, zz, qx, qy, qz, qw], axis=-1).astype(np.float32)
