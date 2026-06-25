"""Numpy reference for the MPPI cost + control packing (the GPU planner lives in mppi).

`_cost` is the readable, independent ORACLE that the GPU `_cost` kernel is differential-tested
against (tests/control/test_mppi.py): it defines, in plain numpy, what the per-rollout cost
MEANS, so a kernel that diverges from intent is caught even when the robot still roughly reaches
the goal. `_to_omega` packs the [B, T, 2] wheel-speed controls into the engine's [T, B, 3] layout.
"""

import numpy as np


def _to_omega(Ub):
    """Controls [B, T, 2] (wL, wR) -> omega [T, B, 3] (rear = mean)."""
    bt = np.transpose(Ub, (1, 0, 2))  # [T, B, 2]
    rear = bt.mean(2, keepdims=True)
    return np.concatenate([bt, rear], axis=2).astype(np.float32)


def _cost(controlled, derived, clear, resid, Ub, goal, clear_margin, resid_tol, w):
    """Per-rollout cost [B]. goal [2]. `derived` is [T+1, B, 3] = (z, pitch, roll).

    `w["tilt"]` (optional) penalizes body tilt past `tilt_free` along the rollout, split
    PER AXIS with roll weighted > pitch (`roll_cost_weight : pitch_cost_weight`, the robot's
    shape) — roll is the dangerous axis. The settle gives the true body pitch/roll for each
    candidate trajectory, so this is trajectory-aware (a diagonal slope crossing rolls
    differently than a straight climb), not a static per-cell terrain slope.
    """
    xy = controlled[:, :, :2]  # [T+1, B, 2]
    d = np.linalg.norm(xy - goal[None, None, :], axis=2)  # [T+1, B]
    # graded validity (option C): how far past margin/tol, weighted by how early (T,B -> B)
    T = clear.shape[0]
    early = ((T - np.arange(T)) / T)[:, None]  # [T, 1]
    clear_viol = np.maximum(clear_margin - clear, 0.0)  # [T, B]
    resid_viol = np.maximum(resid - resid_tol, 0.0)  # [T, B]
    # robot stability envelope (same limits as the cost-to-go): tipping is invalid. climbing is
    # nose-up = NEGATIVE pitch, so the climb limit is on -pitch. Large default = inactive.
    pitch, roll = derived[:T, :, 1], derived[:T, :, 2]  # [T, B]
    roll_viol = np.maximum(np.abs(roll) - w.get("max_roll", 1e3), 0.0)
    climb_viol = np.maximum(-pitch - w.get("max_pitch_up", 1e3), 0.0)
    descend_viol = np.maximum(pitch - w.get("max_pitch_down", 1e3), 0.0)
    inv = (early * (clear_viol + resid_viol + roll_viol + climb_viol + descend_viol)).sum(0)  # [B]
    eff = (Ub**2).sum((1, 2))
    smooth = (np.diff(Ub, axis=1) ** 2).sum((1, 2))
    J = w["term"] * d[-1] ** 2 + w["run"] * (d**2).mean(0) + w["eff"] * eff + w["smooth"] * smooth
    if w.get("tilt", 0.0) > 0.0:
        # split per axis, roll weighted > pitch (the robot's roll-vs-pitch shape): roll is the
        # dangerous axis. deadzone: tilt below `tilt_free` PER AXIS is free (drivable ramps), so the
        # robot still climbs gentle slopes to reach a goal; only steep tilt is penalized.
        roll_over = np.maximum(np.abs(derived[:, :, 2]) - w.get("tilt_free", 0.0), 0.0)  # [T+1, B]
        pitch_over = np.maximum(np.abs(derived[:, :, 1]) - w.get("tilt_free", 0.0), 0.0)
        rw, pw = w.get("roll_cost_weight", 1.0), w.get("pitch_cost_weight", 0.5)
        J = J + w["tilt"] * (rw * roll_over**2 + pw * pitch_over**2).mean(0)
    if w.get("head", 0.0) > 0.0:
        # heading: penalize facing away from the goal (1 - cos angle); drives the U-turn
        dx = controlled[:, :, 0] - goal[0]
        dy = controlled[:, :, 1] - goal[1]  # [T+1, B]
        dist = np.hypot(dx, dy)
        cos_align = -(
            np.cos(controlled[:, :, 2]) * dx + np.sin(controlled[:, :, 2]) * dy
        ) / np.maximum(dist, 1e-3)
        head = np.where(dist > 1e-3, 1.0 - cos_align, 0.0).mean(0)  # [B]
        J = J + w["head"] * head
    return J + inv * w["invalid"], inv > 0
