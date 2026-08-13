"""Canonical robot + solver dynamics: the SINGLE source of truth for the vehicle model.

The planner, the interactive driver, and the settle-feasibility sampler must all simulate the
*same* vehicle -- otherwise the plan describes a different robot than the one being driven (the
plan->real lag/clip bug came from exactly this: plan dt=0.1 vs drive dt=0.05). Everything that
builds a ForwardSimulator pulls its RobotParams / SolverParams / timestep from here instead of writing
its own, so the configs can't drift.

Two solver fidelities share the same dt and turn gain:
  planning_solver   -- the B-batch MPPI rollouts: fewer Newton iters (speed across thousands)
  execution_solver  -- the single driven / settled robot: more Newton iters (accuracy)
"""

from .engine import RobotParams
from .engine import SolverParams

DT = 0.1  # control timestep -- the plan horizon step AND the driver frame step (must match)
# skid-steer turn gain: alpha = 1 + K_TURN*mu sets the turn resistance (yaw rate = ideal / alpha).
# The gain is TERRAIN-dependent -- outdoor grass/dirt grips harder, so the skid-steer resists turning
# more (understeers). Both values are ICP-truth calibrated against real manual-drive bags:
#   indoor  alpha ~= 1.48  (experiment0 gyro fit, corr 0.95)        -> K_TURN 0.6
#           (was 0.4 from rotate_in_place0 + arc_diff0 alpha~1.33; bumped after the experiment0 fit)
#   outdoor alpha ~= 1.82  (manual_drive_outdoor0, turns ~0.72x)    -> K_TURN 1.0
# A THIRD surface, 2026-08-10: flat PAVED outdoor (tilt under 5.3 deg) measures alpha ~= 1.50, so
# k = 0.62 -- i.e. it behaves like the INDOOR preset, not the outdoor one. Three independent
# estimates agree (steady arcs 1.52, spins 1.50-1.70, lag fit 1.55) over mean speeds 0.37-3.73
# rad/s and both directions, and converged Project Chrono independently predicts ~1.6. See
# CALIBRATION_RESULTS.md. Neither preset is changed by this: 0.6 gives 1.48 and is CONFIRMED, and
# the outdoor 1.0 was calibrated on grass/dirt, which this is not. What it does mean is that
# "outdoor" is about the SURFACE, not about being out of doors -- ros/odin/odin_elevation.params
# .yaml pins k_turn 1.0, which over-predicts turn resistance by 20% on tarmac, so the robot yaws
# more than the planner expects and overshoots turns.
# Pick per environment via k_turn_for(); a single constant can't be right for both. (Forward gain
# measured ~0.95-0.97 both -> wheel_radius unchanged; /cmd_joints is all-positive-forward, no flip.
# The 2026-08-10 bags put forward gain at 0.932, consistent with the 0.906-0.925 on record.)
# TODO: online K_TURN/friction estimation would remove this manual switch. See
# wheel_sign_convention_calibration memory.
K_TURN_INDOOR = 0.6
K_TURN_OUTDOOR = 1.0
# Wheel actuator response, MEASURED on out_experiment_goal_unreachable0/1 by fitting dead time and
# first-order lag JOINTLY (scripts/fit_actuator_lag.py). The drivetrain is a LAG, not a transport
# delay: averaged over 34 setpoint steps the wheel is already moving 10 ms in and reaches 50% at
# ~140 ms, with no dead zone at all. Fitting the delay first by cross-correlation and the lag
# second -- which is what an earlier revision of this file did -- reports ~170 ms of pure delay,
# because cross-correlation returns the GROUP DELAY of a slow rise. The joint fit puts the drive
# wheels at tau 0.17-0.20 s with 0-50 ms of dead time, and fits better (RMSE 0.395 vs 0.412).
MOTOR_TAU = 0.19  # [s] first-order actuator lag; blend = dt/tau = 0.53 at DT = 0.1
# What genuinely is transport: /cmd_joints to the LLC's echoed setpoint measures 10 ms, and the
# joint fit leaves 0-50 ms of dead time on top. Quantised to whole rollout steps it rounds to 0 at
# DT = 0.1, so it only bites if dt is shortened.
COMMAND_DELAY = 0.04
K_TURN = K_TURN_INDOOR  # module default (used by WarpDriver / demos when not overridden)


def k_turn_for(terrain: str) -> float:
    """Calibrated turn gain for 'indoor' / 'outdoor' (falls back to the module default)."""
    return {"indoor": K_TURN_INDOOR, "outdoor": K_TURN_OUTDOOR}.get(terrain, K_TURN)


def robot_params(wheel_width=None):
    """The canonical robot geometry/mass model.

    `wheel_width` None keeps the SPHERE wheel envelope, which reaches the full 0.35 m radius
    SIDEWAYS and is therefore systematically pessimistic in tight and rocky places -- a rock 0.3 m
    beside the wheel lifts and tilts the robot when in reality it is straddled. 0.10 (the
    ruler-measured tread) switches to the yaw-binned CYLINDER envelope, which is honest laterally.
    That trades away a conservative margin, so clearances tuned around the sphere's pessimism are
    worth re-checking.
    """
    return RobotParams(wheel_width=wheel_width)


def planning_solver(dt=DT, k_turn=K_TURN, command_delay=COMMAND_DELAY, tau_motor=MOTOR_TAU, momentum=True):
    """Solver for the MPPI rollouts (B in the thousands): shallow + loose settle, for speed.

    `tau_motor` defaults to the measured MOTOR_TAU: the wheels take ~0.19 s to reach a commanded
    speed, which the planner previously ignored entirely. `command_delay` is the small remaining
    dead time; a caller that does NOT feed `sim.command_history` should pass 0.0, since it would
    otherwise roll out against an all-zero history.

    momentum=True (default): body speed is grip-limited (see SolverParams.momentum) so the plan's
    braking/launch distances are mu-dependent -- required for the 1.5-2.5 m/s regime. Both solvers
    share the flag so the plan and the driven robot stay the same vehicle."""
    return SolverParams(
        dt=dt,
        k_turn=k_turn,
        newton_iters=6,
        atol=1e-4,
        command_delay=command_delay,
        tau_motor=tau_motor,
        momentum=momentum,
    )


def execution_solver(dt=DT, k_turn=K_TURN, command_delay=0.0, momentum=True):
    """Solver for the single driven / settled robot: deeper settle for fidelity.

    `command_delay` defaults to 0 here, unlike `planning_solver`: this path drives or settles a
    single robot from commands the caller already has in hand, so there is nothing in flight.
    """
    return SolverParams(
        dt=dt,
        k_turn=k_turn,
        newton_iters=12,
        tilt_clamp=1.2,
        command_delay=command_delay,
        momentum=momentum,
    )
