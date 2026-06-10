"""Flat-ground forward rollout (Phase 1), batched over B parameter sets.

Drives the kinematic twin with a recorded wheel-speed sequence and integrates the
planar pose. No terrain solve yet: z = wheel radius, level. Phases 2+ replace the
flat placement with the heightmap settling solve.
"""
import numpy as np

from . import placement
from . import twist
from .model import HALF_TRACK
from .model import WHEEL_RADIUS


def rollout_flat(setpoints, dt, alpha=1.0, x_icr=0.0, init_pose=None,
                 R=WHEEL_RADIUS, b=HALF_TRACK):
    """Roll out a wheel-speed sequence on flat ground.

    setpoints : [T, 3] wheel speeds [left, right, rear] (rad/s).
    alpha, x_icr : scalar, or [B] arrays to roll out B parameter sets at once.
    init_pose : (x, y, yaw) or [B, 3]; defaults to origin.

    Returns pose2 [B, T, 3] (x, y, yaw); B is squeezed out if params were scalar.
    """
    setpoints = np.asarray(setpoints, dtype=np.float64)
    T = setpoints.shape[0]

    alpha = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
    x_icr = np.atleast_1d(np.asarray(x_icr, dtype=np.float64))
    B = max(alpha.shape[0], x_icr.shape[0])
    alpha = np.broadcast_to(alpha, (B,))
    x_icr = np.broadcast_to(x_icr, (B,))
    scalar = (np.ndim(np.squeeze(alpha)) == 0) and B == 1

    if init_pose is None:
        pose = np.zeros((B, 3), dtype=np.float64)
    else:
        pose = np.broadcast_to(np.asarray(init_pose, dtype=np.float64), (B, 3)).copy()

    poses = np.empty((B, T, 3), dtype=np.float64)
    for t in range(T):
        omega = np.broadcast_to(setpoints[t], (B, 3))
        vx, vy, wz = twist.wheel_twist(omega, alpha, x_icr, R, b)
        pose = twist.se2_integrate(pose, vx, vy, wz, dt)
        poses[:, t] = pose

    return poses[0] if scalar else poses


def rollout_terrain(setpoints, dt, hm, alpha=1.0, x_icr=0.0, init_pose=(0.0, 0.0, 0.0),
                    mu_field=None, k=2.0, R=WHEEL_RADIUS, b=HALF_TRACK):
    """Roll out on a heightmap: settle every step, advance through the tilt.

    The body-frame forward/lateral velocity is rotated into the world by the
    settled orientation, so climbing pitches the velocity up and horizontal
    progress slows by ~cos(pitch). z, roll, pitch are taken from the settle.

    If `mu_field` is given, the turning params (alpha, x_ICR) are computed each
    step from the friction sampled at the 3 contacts + the normal loads (Phase 4
    moment-centroid map, coefficient `k`); otherwise the scalar args are used.

    Returns dict of arrays over T: pose7 [T,7], pose2 [T,3] (x,y,yaw),
    loads [T,3] (N_i), alpha [T], x_icr [T], pitch/roll/residual [T],
    chassis_clear [T], high_center [T]. Single rollout (no batch).
    """
    from . import heightmap as _hm
    from . import turning
    setpoints = np.asarray(setpoints, dtype=np.float64)
    T = setpoints.shape[0]
    x, y, yaw = (float(v) for v in init_pose)

    # Placement runs against the wheel-envelope (sphere-wheel) surface.
    surf = _hm.wheel_envelope(hm, R)

    pose7 = np.empty((T, 7), dtype=np.float32)
    pose2 = np.empty((T, 3), dtype=np.float64)
    loads = np.empty((T, 3), dtype=np.float64)
    fz = np.empty(T)
    chassis_clear = np.empty(T)  # min chassis-point clearance (raw terrain)
    alpha_log = np.empty(T); xicr_log = np.empty(T)
    pitch = np.empty(T); roll = np.empty(T); resid = np.empty(T)

    place = None
    for t in range(T):
        init = None if place is None else (place["z"], place["pitch"], place["roll"])
        place = placement.settle(x, y, yaw, surf, init=init)
        N = placement.normal_loads(place, x, y)

        if mu_field is not None:
            c = place["contacts"]
            mu_i = mu_field.sample(c[:, 0], c[:, 1])
            alpha_t, xicr_t = turning.turning_params(mu_i, N, k)
        else:
            alpha_t, xicr_t = alpha, x_icr

        vx, vy, wz = twist.wheel_twist(setpoints[t], alpha_t, xicr_t, R, b)
        # Body velocity rotated to world; horizontal part advances (x, y).
        v_world = place["R"] @ np.array([vx, vy, 0.0])
        x += v_world[0] * dt
        y += v_world[1] * dt
        yaw += wz * dt

        cc, _ = placement.chassis_clearance(place["R"], x, y, place["z"], hm)

        pose7[t] = placement.place_pose7(place, x, y)
        pose2[t] = (x, y, yaw)
        loads[t] = N
        fz[t] = float(N @ place["normals"][:, 2])  # vertical force balance (== mg)
        chassis_clear[t] = float(cc.min())
        alpha_log[t] = alpha_t; xicr_log[t] = xicr_t
        pitch[t] = place["pitch"]; roll[t] = place["roll"]; resid[t] = place["residual"]

    return {"pose7": pose7, "pose2": pose2, "loads": loads, "fz": fz,
            "chassis_clear": chassis_clear, "high_center": chassis_clear < 0.0,
            "alpha": alpha_log, "x_icr": xicr_log,
            "pitch": pitch, "roll": roll, "residual": resid}


def cruise_decomposition(pose2, setpoints, dt, x_max=0.9, t_min=0.3, R=WHEEL_RADIUS):
    """Flat-ground cruise check (pre-box window). Mirrors replay_real's metric.

    Returns dict with commanded wheel speed, no-slip ground speed (= w*R), and
    the realized ground speed from the integrated path (== w*R by construction,
    since the kinematic model has zero slip)."""
    x = pose2[:, 0]
    t = np.arange(len(x)) * dt
    win = (t > t_min) & (x < x_max)
    if win.sum() < 5:
        return None
    cmd = np.abs(setpoints[win]).mean()
    speed = np.linalg.norm(np.diff(pose2[:, :2], axis=0), axis=1)[win[:-1]] / dt
    return {
        "commanded_wheel_speed": float(cmd),
        "noslip_ground_speed": float(cmd * R),
        "ground_speed": float(np.median(speed)),
        "window_steps": int(win.sum()),
    }
