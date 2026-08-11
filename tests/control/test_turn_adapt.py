"""Contract tests for the adaptive turn-boost yaw-feedback estimator."""
from helhest.control.turn_adapt import AdaptiveTurnBoost
from helhest.control.turn_adapt import TurnGainEstimator

# indoor model: alpha = 1 + 0.6*0.8 = 1.48; R=0.35, b=0.365 (dynamics/robot defaults)
KW = dict(alpha_model=1.48, wheel_radius=0.35, half_track=0.365, dt=0.1)


def _model_yaw(diff, alpha=1.48, R=0.35, b=0.365):
    return R * diff / (2.0 * b * alpha)


# --- TurnGainEstimator: model-side mu adaptation (both directions) ---
GE_KW = dict(k_turn=0.6, mu_nominal=0.8, wheel_radius=0.35, half_track=0.365, dt=0.1)


def _real_yaw(diff, mu_real, k_turn=0.6, R=0.35, b=0.365):
    return R * diff / (2.0 * b * (1.0 + k_turn * mu_real))


def test_gain_estimator_understeer():
    # grippier ground than the model assumes (mu_real 1.2 vs nominal 0.8 -> the robot under-turns):
    # the center must converge to mu_real/mu_nominal = 1.5.
    est = TurnGainEstimator(tau_s=1.0, **GE_KW)
    for _ in range(400):
        est.update(4.0, _real_yaw(4.0, mu_real=1.2))
    center, span = est.band()
    assert abs(center - 1.5) < 0.05
    assert span <= 0.2  # noiseless samples -> span collapses to the floor


def test_gain_estimator_oversteer():
    # slick ground (mu_real 0.4 -> over-turns): center must drop BELOW 1 -- the case the
    # turn-boost hotfix (clamp floor 1.0) can never correct.
    est = TurnGainEstimator(tau_s=1.0, **GE_KW)
    for _ in range(400):
        est.update(4.0, _real_yaw(4.0, mu_real=0.4))
    center, _ = est.band()
    assert abs(center - 0.5) < 0.05


def test_gain_estimator_holds_on_straights():
    est = TurnGainEstimator(tau_s=1.0, **GE_KW)
    for _ in range(50):
        est.update(0.0, 0.0)
    center, _ = est.band()
    assert center == 1.0


def test_gain_estimator_noise_widens_band():
    # noisy yaw measurements must widen the reported span (the robust replicas' job grows)
    import numpy as np

    rng = np.random.default_rng(0)
    est = TurnGainEstimator(tau_s=1.0, **GE_KW)
    for _ in range(400):
        est.update(4.0, _real_yaw(4.0, mu_real=0.8) * float(rng.uniform(0.6, 1.6)))
    _, span = est.band()
    assert span > 0.15  # above the floor


def test_no_signal_holds():
    est = AdaptiveTurnBoost(init=1.7, **KW)
    # straight (tiny differential) and tiny yaw -> no adaptation, holds the value
    for _ in range(50):
        est.update(0.0, 0.0)
    assert est.turn_boost == 1.7


def test_converges_to_inverse_realization():
    # the robot realizes only 50% of the commanded differential -> yaw is half the model prediction,
    # so the boost must converge toward 1/0.5 = 2.0.
    est = AdaptiveTurnBoost(init=1.0, tau_s=1.0, **KW)
    diff = 4.0
    for _ in range(400):
        yaw_meas = _model_yaw(diff) * 0.5  # under-realized
        est.update(diff, yaw_meas)
    assert abs(est.turn_boost - 2.0) < 0.05


def test_over_realization_backs_off():
    # realizes 130% of commanded -> should drop below 1... but clamp floor is 1.0, so it lands at 1.0.
    est = AdaptiveTurnBoost(init=2.0, tau_s=1.0, clamp=(1.0, 3.0), **KW)
    diff = 4.0
    for _ in range(400):
        est.update(diff, _model_yaw(diff) * 1.3)
    assert abs(est.turn_boost - 1.0) < 1e-3


def test_clamps_high():
    # severe under-realization (10%) would want boost 10, but the band caps it at 3.0.
    est = AdaptiveTurnBoost(init=1.0, tau_s=1.0, clamp=(1.0, 3.0), **KW)
    diff = 4.0
    for _ in range(400):
        est.update(diff, _model_yaw(diff) * 0.1)
    assert est.turn_boost <= 3.0 + 1e-6 and est.turn_boost > 2.9


def test_sign_disagreement_skipped():
    # measured yaw opposite the command (a glitch) must not push the boost around
    est = AdaptiveTurnBoost(init=1.5, tau_s=1.0, **KW)
    for _ in range(50):
        est.update(4.0, -_model_yaw(4.0))  # wrong sign
    assert est.turn_boost == 1.5


def test_negative_differential_symmetric():
    # turning the other way (negative differential) tracks the same realization factor
    est = AdaptiveTurnBoost(init=1.0, tau_s=1.0, **KW)
    diff = -4.0
    for _ in range(400):
        est.update(diff, _model_yaw(diff) * 0.5)
    assert abs(est.turn_boost - 2.0) < 0.05
