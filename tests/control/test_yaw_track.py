"""Contract tests for the inner yaw-rate loop (guardrails, sign, and the fixed point)."""

from helhest.control.yaw_track import YawRateTracker

YPD = 0.324  # rad/s of yaw per rad/s of differential, this robot at the planner's alpha
DT = 0.02  # 50 Hz, the intended command-timer rate


def _settle(trk: YawRateTracker, ref: float, gain: float, steps: int = 400) -> float:
    """Run the loop against a plant that realises `gain` x the modelled yaw. Returns the yaw."""
    yaw = 0.0
    for _ in range(steps):
        yaw = (ref / YPD + trk.correction) * YPD * gain
        trk.update(ref, yaw, DT)
    return yaw


def test_no_correction_when_the_model_is_right():
    # The loop must be inert on ground the planner already models correctly, or it would bias
    # every turn away from what MPPI planned.
    trk = YawRateTracker(yaw_per_diff=YPD, tau_ref=0.0)
    yaw = _settle(trk, 0.65, 1.0)
    assert abs(yaw - 0.65) < 1e-3
    assert abs(trk.correction) < 1e-3


def test_over_turn_is_corrected_downward():
    # Soft ground: alpha is smaller than modelled, so the robot turns MORE than asked. The loop
    # must REDUCE the differential -- the sign that matters, and the field symptom.
    trk = YawRateTracker(yaw_per_diff=YPD, tau_ref=0.0)
    yaw = _settle(trk, 0.65, 1.20)
    assert trk.correction < 0.0
    assert abs(yaw - 0.65) < 0.02


def test_under_turn_is_corrected_upward():
    trk = YawRateTracker(yaw_per_diff=YPD, tau_ref=0.0)
    yaw = _settle(trk, 0.65, 0.80)
    assert trk.correction > 0.0
    assert abs(yaw - 0.65) < 0.02


def test_deadband_ignores_noise_on_a_straight():
    # A straight line must not be regulated: the gain is unobservable and the error is gyro noise
    # plus the sensor's rest bias, which would otherwise integrate the robot off its line.
    trk = YawRateTracker(yaw_per_diff=YPD, deadband=0.05, tau_ref=0.0)
    for i in range(500):
        trk.update(0.0, 0.002 * (1 if i % 2 else -1), DT)
    assert abs(trk.correction) < 1e-6


def test_integrator_freezes_at_saturation():
    # No authority means no learning: this drivetrain saturates, and integrating through it is
    # what makes a loop wind up and then lurch when it comes back into range.
    trk = YawRateTracker(yaw_per_diff=YPD, tau_ref=0.0)
    for _ in range(200):
        trk.update(0.65, 0.0, DT, saturated=True)
    frozen = trk.integral
    assert abs(frozen) < 1e-9
    for _ in range(10):
        trk.update(0.65, 0.0, DT, saturated=False)
    assert trk.integral > 0.0  # ... and resumes once there is authority again


def test_correction_is_clamped():
    trk = YawRateTracker(yaw_per_diff=YPD, max_correction=1.5, tau_ref=0.0)
    for _ in range(2000):
        trk.update(3.0, 0.0, DT)  # a yaw error no differential could ever close
    assert abs(trk.correction) <= 1.5 + 1e-6


def test_stale_measurement_bleeds_the_correction():
    # A correction computed for a manoeuvre that may already be over must not be held.
    trk = YawRateTracker(yaw_per_diff=YPD, stale_s=0.15, tau_ref=0.0)
    for _ in range(100):
        trk.update(0.65, 0.4, DT)
    assert abs(trk.correction) > 0.1
    for _ in range(200):
        trk.update(0.65, 0.4, DT, age=1.0)
    assert abs(trk.correction) < 0.01


def test_reset_clears_state():
    trk = YawRateTracker(yaw_per_diff=YPD, tau_ref=0.0)
    for _ in range(100):
        trk.update(0.65, 0.4, DT)
    trk.reset()
    assert trk.correction == 0.0
    assert trk.integral == 0.0


def test_reference_lag_suppresses_the_startup_transient():
    # The wheels trail a ramping command by tau_motor. Without the lag the loop reads that as
    # model error and winds up; with it, a perfectly-modelled plant provokes almost no correction.
    lagged = YawRateTracker(yaw_per_diff=YPD, tau_ref=0.19)
    raw = YawRateTracker(yaw_per_diff=YPD, tau_ref=0.0)
    yaw_l = yaw_r = 0.0
    peak_l = peak_r = 0.0
    for _ in range(60):  # 1.2 s of a step reference against a first-order plant, gain exactly 1
        for trk, yaw, name in ((lagged, yaw_l, "l"), (raw, yaw_r, "r")):
            target = (0.65 / YPD + trk.correction) * YPD
            yaw = yaw + (DT / (DT + 0.19)) * (target - yaw)
            trk.update(0.65, yaw, DT)
            if name == "l":
                yaw_l, peak_l = yaw, max(peak_l, yaw)
            else:
                yaw_r, peak_r = yaw, max(peak_r, yaw)
    assert peak_l < peak_r  # the lagged reference overshoots less on the same plant
    assert peak_l < 0.65 * 1.05
