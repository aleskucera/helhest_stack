"""Fast inner yaw-rate loop: correct the commanded differential so the REALIZED yaw matches the
command that was actually sent.

Everything downstream of MPPI is feedforward. The planner maps a differential to a yaw rate through
wz = R*(wR - wL) / (2b*alpha) with alpha = 1 + k_turn*mu, and then simply publishes it -- if the
real ground has a different alpha the robot turns by a different amount and nobody notices until
the next replan sees the pose error. Because alpha DIVIDES the yaw rate, ground softer than the
planner's assumed mu gives a smaller alpha and therefore MORE yaw than asked for: measured in the
engine at k_turn 0.6 against a planner believing mu 0.8, real mu 0.4 over-turns by 19% and mu 0.2
by 32%. On the robot the gap is at least as large -- the `compact` bag reads alpha 1.68-2.01 where
the planner assumes 1.48.

This closes that gap directly instead of estimating it. `AdaptiveTurnBoost` in turn_adapt.py is the
other half of the same problem and deliberately slow (an EMA over seconds, so it cannot fight the
replanning loop); it identifies the terrain GAIN. This is a regulator: it rejects the error now,
and it does not care why the error is there.

WHAT IT TRACKS MATTERS, in two ways the obvious implementation gets wrong.

The reference is the UNCORRECTED plan put through the output conditioner -- after the turn brake,
the slew limiter and the omega clamp, but without this loop's own correction. Tracking the raw
plan instead would make the loop push the differential back up to defeat the turn brake, undoing
the one knob measured to soften a corner without costing clearance. Tracking the FINAL published
command is worse still: the correction is inside it, so reference and measurement both scale with
the correction, the error has no fixed point, and the loop inflates the turn even at zero model
error (measured: peak yaw 0.517 -> 0.575 rad/s on ground the planner had exactly right).

The reference is then passed through the actuator's own first-order lag before comparison, because
the wheels legitimately trail a ramping command by tau_motor. Comparing against the instantaneous
reference makes the loop integrate that known lag as model error and overshoot -- it RAISED peak
yaw where it should have lowered it. With the lag applied, on flat ground at a commanded
differential of 2.0 with the planner believing mu 0.8:

    real mu      0.80    0.40    0.20
    open loop    0.0%   19.4%   32.1%    steady-state yaw error
    closed       0.4%    1.0%    1.1%    peak yaw also drops, 0.617 -> 0.597 at mu 0.4

WHAT SETS THE USEFUL BANDWIDTH is the drivetrain, not the loop rate: tau_motor is 0.19 s (measured
over 34 step responses), about 0.8 Hz, so no gain makes the wheels arrive sooner than that. Running
at 400 Hz buys nothing over ~30-50 Hz; the value here is closing the loop at all.

Guardrails, each for a failure this project has already hit:
  * anti-windup -- the integrator freezes at wheel saturation. This drivetrain does saturate, and
    reading saturation as a fixed gain loss is what produced the retracted "49% turn-differential
    defect".
  * deadband -- on a straight the reference is ~0 and the error is gyro noise plus the rest bias
    (~0.002 rad/s on /odin1/imu), so the loop bleeds its state instead of integrating drift.
  * a hard clamp on the correction, so no amount of bad gyro can command a spin.
  * stale measurements decay the correction rather than hold it.
"""

from __future__ import annotations


class YawRateTracker:
    """PI on yaw-rate error, returning an additive wheel-speed differential correction [rad/s]."""

    def __init__(
        self,
        *,
        yaw_per_diff: float,  # wz per unit differential: R / (2b*alpha) [rad/s per rad/s]
        kp: float = 0.4,  # dimensionless: fraction of the yaw error corrected immediately
        ki: float = 1.0,  # [1/s] integral gain, in yaw-rate units
        deadband: float = 0.05,  # [rad/s] below this the loop idles -- see the module docstring
        max_correction: float = 1.5,  # [rad/s] hard clamp on the differential correction
        stale_s: float = 0.15,  # [s] measurements older than this are not trusted
        tau_ref: float = 0.19,  # [s] actuator lag the reference is passed through (MOTOR_TAU)
    ) -> None:
        # A near-zero yaw_per_diff would turn the unit conversion into a divide-by-zero; it is a
        # pure geometry constant (~0.32 for this robot) so the floor is a guard, not a regime.
        self._yaw_per_diff = max(float(yaw_per_diff), 1e-3)
        self._kp = float(kp)
        self._ki = float(ki)
        self._deadband = float(deadband)
        self._max = float(max_correction)
        self._stale_s = float(stale_s)
        self._tau_ref = max(float(tau_ref), 0.0)
        self._integral = 0.0  # [rad/s] of yaw, converted to a differential only on output
        self._correction = 0.0
        self._ref_lagged = 0.0  # the reference seen through the actuator's own first-order lag

    @property
    def correction(self) -> float:
        """The differential correction [rad/s] to add to the next commanded (wR - wL)."""
        return self._correction

    @property
    def integral(self) -> float:
        """Accumulated yaw-rate bias [rad/s]; exposed for logging, not for control."""
        return self._integral

    def set_gains(self, kp: float, ki: float, deadband: float, max_correction: float) -> None:
        """Retune live. Kept off the constructor path so `ros2 param set` can adjust gains during
        a drive without rebuilding the planner -- a rebuild stalls the plan long enough to trip
        the command timer's staleness stop, i.e. tuning a gain would brake the robot."""
        self._kp = float(kp)
        self._ki = float(ki)
        self._deadband = float(deadband)
        self._max = float(max_correction)

    def reset(self) -> None:
        """Drop all state. Call on e-stop, on losing actuation, and when a new plan is committed
        after a gap -- a stale integrator applied to a fresh manoeuvre is a lurch."""
        self._integral = 0.0
        self._correction = 0.0
        self._ref_lagged = 0.0

    def update(
        self,
        yaw_ref: float,  # [rad/s] yaw implied by the command actually published
        yaw_meas: float | None,  # [rad/s] measured yaw rate, None if unavailable
        dt: float,  # [s] time since the previous update
        *,
        age: float = 0.0,  # [s] how old the measurement is
        saturated: bool = False,  # any wheel at its omega clamp -> the loop has no authority
    ) -> float:
        if dt <= 0.0:
            return self._correction
        if yaw_meas is None or age > self._stale_s:
            # No trustworthy measurement: bleed toward feedforward rather than hold a correction
            # that was computed for a manoeuvre which may already be over.
            self._integral *= 0.5 ** (dt / 0.25)
            self._correction *= 0.5 ** (dt / 0.25)
            return self._correction

        # The wheels cannot be at the reference yet: tau_motor is 0.19 s, so during a ramp the
        # robot legitimately trails the command. Comparing against the RAW reference makes the loop
        # integrate that known lag as if it were model error, which winds up and then overshoots --
        # measured as peak yaw rising 0.617 -> 0.654 rad/s where the loop should have LOWERED it.
        # Passing the reference through the same first-order lag leaves only genuine model error.
        if self._tau_ref > 0.0:
            blend = dt / (dt + self._tau_ref)
            self._ref_lagged += blend * (float(yaw_ref) - self._ref_lagged)
        else:
            self._ref_lagged = float(yaw_ref)
        error = self._ref_lagged - float(yaw_meas)
        if abs(self._ref_lagged) < self._deadband and abs(error) < self._deadband:
            # Straight running: the gain is unobservable here and the error is noise plus the
            # sensor's rest bias, so integrating it would walk the robot off a straight line.
            self._integral *= 0.5 ** (dt / 0.5)
            self._correction = self._integral / self._yaw_per_diff
            return self._correction

        if not saturated:  # anti-windup: no authority means no learning
            self._integral += self._ki * error * dt
            # Bound the integral in yaw units so the clamp means the same thing at every gain.
            limit = self._max * self._yaw_per_diff
            self._integral = max(-limit, min(limit, self._integral))

        corr = (self._kp * error + self._integral) / self._yaw_per_diff
        self._correction = max(-self._max, min(self._max, corr))
        return self._correction
