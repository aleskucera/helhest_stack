"""The kinematic State and the single physics step.

A `State` is a *valid* configuration: a settled, non-penetrative pose. It bundles
the planar pose (x, y, yaw) with its terrain placement (z, roll, pitch,
orientation R, contacts, normal loads) and a validity flag.

`make_state` settles an initial pose into a valid State. `step` advances one
timestep and returns the next valid State — the core invariant is

    step :  valid State  +  wheel speeds  ->  valid State

It is predict->project:
  predict : use the CURRENT state's orientation + loads to build the twist and
            advance the planar pose (climbing slows horizontal progress).
  project : settle the new pose onto the terrain -> the next valid State.
"""

from dataclasses import dataclass

import numpy as np

from . import placement
from . import turning
from . import twist
from ..model import HALF_TRACK
from ..model import WHEEL_RADIUS


@dataclass
class State:
    x: float
    y: float
    yaw: float
    place: dict  # settle result: z, pitch, roll, R, contacts, normals, residual
    loads: np.ndarray  # contact normal loads N_i [3] (Newtons)
    chassis_clear: float  # min chassis-point clearance vs raw terrain [m]
    valid: bool  # not high-centered (chassis clears)
    alpha: float = float("nan")  # turning params USED to reach this state
    x_icr: float = float("nan")  # (nan for an initial settled-only state)
    omega_actual: np.ndarray | None = None  # lagged wheel speed that was used for this step

    @property
    def pose7(self):
        """SE(3) pose (px,py,pz, qx,qy,qz,qw)."""
        return placement.place_pose7(self.place, self.x, self.y)

    @property
    def pose2(self):
        return np.array([self.x, self.y, self.yaw], dtype=np.float64)

    @property
    def fz(self):
        """Vertical force balance Sum N_i n_z (== m g at a valid settle)."""
        return float(self.loads @ self.place["normals"][:, 2])


def _settle(x, y, yaw, surf, hm, init=None, alpha=float("nan"), x_icr=float("nan"), omega_actual=None):
    """Project a planar pose onto the terrain -> valid State."""
    place = placement.settle(x, y, yaw, surf, init=init)
    N = placement.normal_loads(place, x, y)
    cc, _ = placement.chassis_clearance(place["R"], x, y, place["z"], hm)
    cmin = float(cc.min())
    return State(x, y, yaw, place, N, cmin, cmin >= 0.0, alpha, x_icr, omega_actual)


def make_state(x, y, yaw, surf, hm):
    """Settle an initial pose into a valid State.

    `surf` is the wheel-envelope placement surface (heightmap.wheel_envelope);
    `hm` is the raw heightmap (for chassis non-penetration).
    """
    return _settle(x, y, yaw, surf, hm)


def turning_of(state, mu_field, k, alpha=1.0, x_icr=0.0):
    """Turning params (alpha, x_ICR) for a step taken FROM `state`."""
    if mu_field is None:
        return alpha, x_icr
    c = state.place["contacts"]
    mu_i = mu_field.sample(c[:, 0], c[:, 1])
    return turning.turning_params(mu_i, state.loads, k)


def _motor_lag_step(omega_actual: np.ndarray, omega_cmd: np.ndarray, dt: float, tau: float) -> np.ndarray:
    """First-order actuator lag: mirrors the device motor_lag_step @wp.func exactly."""
    alpha = min(dt / max(tau, 1e-6), 1.0)
    return omega_actual + alpha * (omega_cmd - omega_actual)


def step(
    state,
    omega,
    surf,
    hm,
    dt,
    mu_field=None,
    k=2.0,
    alpha=1.0,
    x_icr=0.0,
    R=WHEEL_RADIUS,
    b=HALF_TRACK,
    tau_motor: float = 0.0,
    omega_actual: np.ndarray | None = None,
):
    """Advance one timestep: valid State + wheel speeds [L,R,rear] -> valid State."""
    omega = np.asarray(omega, dtype=np.float64)
    if omega_actual is None:
        omega_actual = np.zeros(3, dtype=np.float64)

    alpha_t, xicr_t = turning_of(state, mu_field, k, alpha, x_icr)

    # Apply lag first (update-then-use): tau_motor=0 gives omega_eff = omega_cmd exactly,
    # preserving backward compatibility with the original instantaneous-tracking model.
    omega_eff = _motor_lag_step(omega_actual, omega, dt, tau_motor)

    # Predict: project the body velocity through the CURRENT orientation, step.
    vx, vy, wz = twist.wheel_twist(omega_eff, alpha_t, xicr_t, R, b)
    v_world = state.place["R"] @ np.array([vx, vy, 0.0])
    x = state.x + v_world[0] * dt
    y = state.y + v_world[1] * dt
    yaw = state.yaw + wz * dt

    # Project: settle the new pose -> next valid State (warm-started).
    # omega_eff is the filter state for the next step (carry-through for the lag chain).
    init = (state.place["z"], state.place["pitch"], state.place["roll"])
    return _settle(x, y, yaw, surf, hm, init=init, alpha=alpha_t, x_icr=xicr_t,
                   omega_actual=omega_eff)
