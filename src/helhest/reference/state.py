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
    current_wheel_omega: np.ndarray | None = None  # lagged wheel speed that was used for this step

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


def _settle(
    x, y, yaw, surf, hm, init=None, alpha=float("nan"), x_icr=float("nan"), current_wheel_omega=None
):
    """Project a planar pose onto the terrain -> valid State."""
    place = placement.settle(x, y, yaw, surf, init=init)
    N = placement.normal_loads(place, x, y)
    cc, _ = placement.chassis_clearance(place["R"], x, y, place["z"], hm)
    cmin = float(cc.min())
    return State(x, y, yaw, place, N, cmin, cmin >= 0.0, alpha, x_icr, current_wheel_omega)


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


def _integrate_pose(x, y, yaw, v_world, wz, dt):
    """Exact constant-twist (arc) pose update -- the oracle twin of engine.step.integrate_pose.

    Forward Euler takes the chord instead of the arc and is only first order in dt; the twist is
    constant within a step, so the arc has a closed form that is exact at any dt. Kept in lockstep
    with the device version, since this module is the finite-difference oracle for it.
    """
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    u_x = cos_yaw * v_world[0] + sin_yaw * v_world[1]
    u_y = -sin_yaw * v_world[0] + cos_yaw * v_world[1]
    theta = wz * dt
    if abs(theta) > 1.0e-6:
        integral_cos = np.sin(theta) / wz
        integral_sin = (1.0 - np.cos(theta)) / wz
    else:
        integral_cos, integral_sin = dt, 0.5 * theta * dt
    local_x = integral_cos * u_x - integral_sin * u_y
    local_y = integral_sin * u_x + integral_cos * u_y
    return (
        x + cos_yaw * local_x - sin_yaw * local_y,
        y + sin_yaw * local_x + cos_yaw * local_y,
        yaw + theta,
    )


def _motor_lag_step(
    current_wheel_omega: np.ndarray, target_wheel_omega: np.ndarray, dt: float, tau: float
) -> np.ndarray:
    """First-order actuator lag: mirrors the device motor_lag_step @wp.func exactly."""
    alpha = 1.0 - np.exp(-dt / max(tau, 1e-6))  # exact first-order step; see engine.motor_lag_step
    return current_wheel_omega + alpha * (target_wheel_omega - current_wheel_omega)


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
    current_wheel_omega: np.ndarray | None = None,
):
    """Advance one timestep: valid State + wheel speeds [L,R,rear] -> valid State."""
    omega = np.asarray(omega, dtype=np.float64)
    if current_wheel_omega is None:
        current_wheel_omega = np.zeros(3, dtype=np.float64)

    alpha_t, xicr_t = turning_of(state, mu_field, k, alpha, x_icr)

    # Apply lag first (update-then-use): tau_motor=0 gives current_wheel_omega_eff = target_wheel_omega exactly,
    # preserving backward compatibility with the original instantaneous-tracking model.
    current_wheel_omega_eff = _motor_lag_step(current_wheel_omega, omega, dt, tau_motor)

    # Predict: project the body velocity through the CURRENT orientation, step along the arc.
    vx, vy, wz = twist.wheel_twist(current_wheel_omega_eff, alpha_t, xicr_t, R, b)
    v_world = state.place["R"] @ np.array([vx, vy, 0.0])
    x, y, yaw = _integrate_pose(state.x, state.y, state.yaw, v_world, wz, dt)

    # Project: settle the new pose -> next valid State (warm-started).
    # current_wheel_omega_eff is the filter state for the next step (carry-through for the lag chain).
    init = (state.place["z"], state.place["pitch"], state.place["roll"])
    return _settle(
        x,
        y,
        yaw,
        surf,
        hm,
        init=init,
        alpha=alpha_t,
        x_icr=xicr_t,
        current_wheel_omega=current_wheel_omega_eff,
    )
