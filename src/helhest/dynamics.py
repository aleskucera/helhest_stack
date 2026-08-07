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
# Pick per environment via k_turn_for(); a single constant can't be right for both. (Forward gain
# measured ~0.95-0.97 both -> wheel_radius unchanged; /cmd_joints is all-positive-forward, no flip.)
# TODO: online K_TURN/friction estimation would remove this manual switch. See
# wheel_sign_convention_calibration memory.
K_TURN_INDOOR = 0.6
K_TURN_OUTDOOR = 1.0
# Transport delay [s] between publishing a wheel command and the wheels acting on it. MEASURED at
# 149-199 ms per wheel on out_experiment_goal_unreachable0/1 (scripts/fit_actuator_lag.py), with
# the command reconstructed as a zero-order hold -- interpolating it as a ramp inflates this by
# half a publish interval. Confirmed physical rather than a timing artifact: every topic's header
# stamp sits within 2 ms of its bag log time, the reported wheel velocity tracks the encoder
# position to 10-20 ms, and the IMU (separate device and driver) sees the same lag on yaw.
#
# QUANTISATION: the rollout can only delay by whole steps, so at DT = 0.1 this rounds to 2 steps
# = 200 ms, about 30 ms more than measured. Getting closer needs a finer dt, not a finer constant.
COMMAND_DELAY = 0.17
K_TURN = K_TURN_INDOOR  # module default (used by WarpDriver / demos when not overridden)


def k_turn_for(terrain: str) -> float:
    """Calibrated turn gain for 'indoor' / 'outdoor' (falls back to the module default)."""
    return {"indoor": K_TURN_INDOOR, "outdoor": K_TURN_OUTDOOR}.get(terrain, K_TURN)


def robot_params():
    """The canonical robot geometry/mass model."""
    return RobotParams()


def planning_solver(dt=DT, k_turn=K_TURN, command_delay=COMMAND_DELAY):
    """Solver for the MPPI rollouts (B in the thousands): shallow + loose settle, for speed.

    `command_delay` [s] defaults to the measured COMMAND_DELAY. A caller that does NOT feed
    `sim.command_history` should pass 0.0: it would otherwise roll out the first two steps against
    an all-zero history, i.e. predict the robot standing still for 200 ms.
    """
    return SolverParams(
        dt=dt, k_turn=k_turn, newton_iters=6, atol=1e-4, command_delay=command_delay
    )


def execution_solver(dt=DT, k_turn=K_TURN, command_delay=0.0):
    """Solver for the single driven / settled robot: deeper settle for fidelity.

    `command_delay` defaults to 0 here, unlike `planning_solver`: this path drives or settles a
    single robot from commands the caller already has in hand, so there is nothing in flight.
    """
    return SolverParams(
        dt=dt, k_turn=k_turn, newton_iters=12, tilt_clamp=1.2, command_delay=command_delay
    )
