"""Model-based state filter utilities."""

from __future__ import annotations

import numpy as np

from ..dynamics import DT
from ..reference.state import State
from ..reference.state import step


def model_innovation(
    current_state: State,
    omega: np.ndarray,
    estimate: np.ndarray,
    surf,
    hm,
    dt: float = DT,
) -> float:
    """L2 distance between the model-predicted planar pose and `estimate`.

    Steps `current_state` forward one timestep with `omega`, then returns
    ||(x_pred - x_hat, y_pred - y_hat, wrap(psi_pred - psi_hat))||_2.

    omega    : [3] wheel angular velocities [L, R, rear] (rad/s)
    estimate : [3] external pose estimate (x_hat, y_hat, psi_hat)
    surf     : wheel-envelope heightmap (heightmap.wheel_envelope)
    hm       : raw heightmap (for chassis clearance in settle)
    """
    predicted = step(current_state, omega, surf, hm, dt)
    dx = predicted.x - estimate[0]
    dy = predicted.y - estimate[1]
    dpsi = (predicted.yaw - estimate[2] + np.pi) % (2 * np.pi) - np.pi
    return float(np.sqrt(dx**2 + dy**2 + dpsi**2))
