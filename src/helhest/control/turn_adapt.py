"""Optional online turn-boost from yaw feedback: self-tune the commanded turn differential so the
REALIZED yaw matches what MPPI planned -- across terrains and the drivetrain differential defect.

The vehicle turn model is  wz = R*(wR - wL) / (2b*alpha),  alpha = 1 + k_turn*mu  (engine/step.py).
MPPI plans assuming that model; the real robot may realize a different yaw for the same command
(grippier terrain -> understeer -> higher alpha; the two drive motors equalize a commanded
differential -> under-turn). Rather than re-identify the model online, close a slow loop: from the
ACTUALLY-COMMANDED differential and the MEASURED yaw rate, estimate the boost that would make the
realized yaw equal the model's prediction, and low-pass it:

    tb_target = model_yaw(diff_cmd) / yaw_meas = [R*diff_cmd / (2b*alpha_model)] / yaw_meas

Note diff_cmd cancels in the ideal (both numerator and yaw_meas scale with it), so this is a direct
estimate of model_gain / real_gain -- exactly the multiplier condition_command needs. It is a smarter,
self-tuning generalization of the fixed `plan_turn_boost` (docs/turn_differential_hotfix.md): it
adapts to whatever terrain the robot is on instead of a hand-picked indoor/outdoor constant.

Guardrails: only updates while genuinely turning (the gain is unobservable on straights, and a small
yaw_meas would blow up the ratio); ignores steps where command and measurement disagree in sign
(noise / transients); clamps to a safe band; EMA time constant of seconds so it cannot fight the MPPI
replanning loop.
"""

from __future__ import annotations

import math


class TurnGainEstimator:
    """Online estimate of the EFFECTIVE friction the turn model should use, from commanded
    differential vs measured yaw rate -- the model-side counterpart of AdaptiveTurnBoost (which
    compensates at the command). Each valid sample inverts the turn model for the effective mu:

        yaw = R*diff / (2b * alpha_real)  ->  alpha_real = R*diff / (2b*yaw)
        alpha = 1 + k_turn*mu_eff (flat ground, full load)  ->  mu_eff = (alpha_real - 1) / k_turn

    and EMA-tracks mu_eff's mean and spread. The planner consumes `band()` as a friction-scale
    band (center, span) relative to its nominal mu: MppiGpu.set_mu_band(center, span) recenters
    every rollout's dynamics on the REALIZED turn gain (fixing both understeer and oversteer --
    unlike the boost, this moves in both directions) and sizes the robust-mu replicas by how noisy
    the estimate is. Same guardrails as AdaptiveTurnBoost: only updates while genuinely turning,
    skips sign disagreements, slow EMA, hard clamps."""

    def __init__(
        self,
        *,
        k_turn: float,  # the planner solver's friction->alpha gain (must match the sim)
        mu_nominal: float,  # the planner's uniform friction value (plan_friction)
        wheel_radius: float,
        half_track: float,
        dt: float,
        tau_s: float = 5.0,  # EMA time constant [s]; slow, so it can't fight the replanning loop
        clamp: tuple[float, float] = (0.2, 3.0),  # bounds on the mu-scale CENTER
        span_floor: float = 0.15,  # never report less uncertainty than this (model error exists)
        span_ceil: float = 0.8,
        min_diff: float = 1.0,  # only adapt when |wR - wL| exceeds this [rad/s]
        min_yaw: float = 0.05,  # ...and |yaw_meas| exceeds this [rad/s]
    ):
        self._k_turn = float(k_turn)
        self._mu_nom = float(mu_nominal)
        self._c = wheel_radius / (2.0 * half_track)  # yaw per unit differential at alpha = 1
        self._beta = min(1.0, dt / max(tau_s, 1e-6))
        self._lo, self._hi = clamp
        self._span_floor, self._span_ceil = span_floor, span_ceil
        self._min_diff = min_diff
        self._min_yaw = min_yaw
        self.center = 1.0  # mu-scale center (1 = trust the nominal mu)
        self._var = 0.0  # EMA variance of the per-sample center estimate

    def band(self) -> tuple[float, float]:
        """(center, span) for MppiGpu.set_mu_band: mu-scale band the robust replicas should cover."""
        span = min(max(2.0 * float(self._var**0.5), self._span_floor), self._span_ceil)
        return self.center, span

    def update(self, diff_cmd: float, yaw_meas: float) -> tuple[float, float]:
        """diff_cmd: the differential actually commanded (wR - wL) [rad/s]; yaw_meas: measured yaw
        rate [rad/s]. Returns band(). Holds the last estimate on straights (gain unobservable)."""
        if abs(diff_cmd) < self._min_diff or abs(yaw_meas) < self._min_yaw:
            return self.band()
        if self._c * diff_cmd * yaw_meas <= 0.0:  # sign disagreement -> transient/noise, skip
            return self.band()
        alpha_real = self._c * diff_cmd / yaw_meas
        mu_eff = max(alpha_real - 1.0, 0.0) / self._k_turn
        sample = min(max(mu_eff / self._mu_nom, self._lo), self._hi)
        self.center += self._beta * (sample - self.center)
        self._var += self._beta * ((sample - self.center) ** 2 - self._var)
        return self.band()


class AdaptiveTurnBoost:
    """Slow yaw-feedback estimate of the turn_boost that makes realized yaw match the plan."""

    def __init__(
        self,
        *,
        alpha_model: float,  # planner's turn resistance 1 + k_turn*mu (the model the plan assumes)
        wheel_radius: float,
        half_track: float,
        dt: float,
        tau_s: float = 3.0,  # EMA time constant [s] -- slow, so it can't fight the replanning loop
        init: float = 1.0,  # starting boost (1.0 = no compensation)
        clamp: tuple[float, float] = (1.0, 3.0),  # safe band; a bad gyro moment can't run it away
        min_diff: float = 1.0,  # only adapt when |wR - wL| exceeds this [rad/s] (observable turning)
        min_yaw: float = 0.05,  # ...and |yaw_meas| exceeds this [rad/s] (avoid divide-by-noise)
    ):
        self._c = wheel_radius / (2.0 * half_track * alpha_model)  # model yaw per unit differential
        self._tau_s = max(tau_s, 1e-6)
        self._beta = self._blend(dt)  # fallback blend, for callers that don't time their updates
        self._lo, self._hi = clamp
        self._min_diff = min_diff
        self._min_yaw = min_yaw
        self.turn_boost = float(min(max(init, self._lo), self._hi))

    def _blend(self, dt: float) -> float:
        """EMA weight for a step of `dt` at time constant tau_s.

        Exact rather than the linear dt/tau: linear is only accurate for dt << tau (it needs a
        min(1.0, .) clamp to stay stable at all), and the whole point of this class is a tau
        measured in SECONDS, which only holds if the discretisation does.
        """
        return 1.0 - math.exp(-dt / self._tau_s)

    def update(self, diff_cmd: float, yaw_meas: float, dt: float | None = None) -> float:
        """diff_cmd: the differential actually commanded (wR - wL) [rad/s]; yaw_meas: measured yaw
        rate (gyro) [rad/s]. Returns the current turn_boost; only moves it when there is turning
        signal, so on straights it holds the last learned value.

        `dt` is the time since the previous update. Pass the MEASURED interval: tau_s is a wall-clock
        time constant, so a nominal period that doesn't match the real update rate scales it by the
        ratio (the ROS node updates at the plan rate, ~69 ms on Odin, not the nominal 0.1 s).
        """
        model_yaw = self._c * diff_cmd
        if abs(diff_cmd) < self._min_diff or abs(yaw_meas) < self._min_yaw:
            return self.turn_boost
        if model_yaw * yaw_meas <= 0.0:  # command vs measurement disagree in sign -> skip (noise)
            return self.turn_boost
        tb_target = model_yaw / yaw_meas  # >1 when under-realizing, <1 when over-realizing
        tb_target = min(max(tb_target, self._lo), self._hi)
        beta = self._beta if dt is None else self._blend(dt)
        self.turn_boost += beta * (tb_target - self.turn_boost)
        return self.turn_boost
