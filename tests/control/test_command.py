"""Contract tests for the /cmd_joints command conditioning (pass-through, rear, slew, clamp)."""

import numpy as np

from helhest.control.command import condition_command
from helhest.control.command import in_flight_history
from helhest.control.command import plan_control_at
from helhest.control.command import to_engine_order
from helhest.control.command import JOINT_NAMES

Z = np.zeros(3, np.float32)


def test_joint_order():
    assert JOINT_NAMES == ("left_wheel_j", "rear_wheel_j", "right_wheel_j")


def test_forward_all_positive():
    # /cmd_joints input convention: forward = all wheels positive (no sign flip).
    cmd = condition_command(2.0, 2.0, Z, max_omega=4.0, max_slew=1e6, dt=0.1)
    assert cmd[0] > 0 and cmd[2] > 0
    np.testing.assert_allclose(cmd, [2.0, 2.0, 2.0], atol=1e-5)


def test_rear_is_mean():
    cmd = condition_command(1.0, 3.0, Z, max_omega=10.0, max_slew=1e6, dt=0.1)
    np.testing.assert_allclose(cmd, [1.0, 2.0, 3.0], atol=1e-5)  # rear = mean(1,3) = 2


def test_slew_limit():
    # target [4,4,4] from rest, but max_slew*dt = 10*0.1 = 1 -> clipped to +1.
    cmd = condition_command(4.0, 4.0, Z, max_omega=4.0, max_slew=10.0, dt=0.1)
    np.testing.assert_allclose(cmd, [1.0, 1.0, 1.0], atol=1e-5)


def test_magnitude_clamp():
    # huge command, prev already near target so slew doesn't bind -> magnitude clamp to max_omega.
    prev = np.array([9.0, 9.0, 9.0], np.float32)
    cmd = condition_command(10.0, 10.0, prev, max_omega=4.0, max_slew=1e6, dt=0.1)
    np.testing.assert_allclose(cmd, [4.0, 4.0, 4.0], atol=1e-5)


def test_stop_ramps_down():
    prev = np.array([3.0, 3.0, 3.0], np.float32)
    cmd = condition_command(0.0, 0.0, prev, max_omega=4.0, max_slew=10.0, dt=0.1)
    # toward zero, but only by max_slew*dt = 1 per joint
    np.testing.assert_allclose(cmd, [2.0, 2.0, 2.0], atol=1e-5)


def test_turn_boost_amplifies_diff_keeps_mean():
    # boost=2 doubles the (wr-wl) differential but leaves the forward mean (rear) unchanged.
    base = condition_command(1.0, 3.0, Z, max_omega=10.0, max_slew=1e6, dt=0.1)  # [1, 2, 3]
    boosted = condition_command(1.0, 3.0, Z, max_omega=10.0, max_slew=1e6, dt=0.1, turn_boost=2.0)
    np.testing.assert_allclose(boosted, [0.0, 2.0, 4.0], atol=1e-5)  # mean 2 kept, diff 2 -> 4
    assert boosted[1] == base[1]  # rear (forward) unchanged


def test_turn_boost_default_is_noop():
    a = condition_command(1.0, 3.0, Z, max_omega=10.0, max_slew=1e6, dt=0.1)
    b = condition_command(1.0, 3.0, Z, max_omega=10.0, max_slew=1e6, dt=0.1, turn_boost=1.0)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_turn_direction():
    # planner turn: wr > wl -> intended +yaw (left/CCW). Right wheel commanded faster than left.
    cmd = condition_command(1.0, 2.0, Z, max_omega=4.0, max_slew=1e6, dt=0.1)
    assert cmd[2] > cmd[0]  # right faster than left -> left/CCW turn


def test_goal_brake_scales_forward_not_turn():
    # brake_dist=4, goal_dist=2 -> forward scaled by 2/4 = 0.5; the turn differential is untouched.
    cmd = condition_command(
        1.0, 3.0, Z, max_omega=10.0, max_slew=1e6, dt=0.1, goal_dist=2.0, brake_dist=4.0
    )
    # unbraked would be [1, 2, 3] (mean 2, diff 2). mean -> 1, diff stays 2 -> [0, 1, 2].
    np.testing.assert_allclose(cmd, [0.0, 1.0, 2.0], atol=1e-5)


def test_goal_brake_noop_far_and_off():
    # beyond brake_dist the brake is a no-op; brake_dist=0 (default) disables it entirely.
    far = condition_command(
        2.0, 2.0, Z, max_omega=10.0, max_slew=1e6, dt=0.1, goal_dist=9.0, brake_dist=3.0
    )
    off = condition_command(2.0, 2.0, Z, max_omega=10.0, max_slew=1e6, dt=0.1)
    np.testing.assert_allclose(far, [2.0, 2.0, 2.0], atol=1e-5)
    np.testing.assert_allclose(off, [2.0, 2.0, 2.0], atol=1e-5)


def test_to_engine_order_moves_the_rear_wheel_last():
    # /cmd_joints is (left, rear, right); the engine wants (wL, wR, w_rear)
    np.testing.assert_allclose(to_engine_order([1.0, 2.0, 3.0]), [1.0, 3.0, 2.0])


def test_in_flight_history_is_oldest_first_and_broadcast():
    rows = [np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0]), np.array([3.0, 3.0, 3.0])]
    h = in_flight_history(rows, 2, 4)
    assert h.shape == (2, 4, 3)
    # only the last `steps` commands are still in flight, oldest of those first
    np.testing.assert_allclose(h[0, :, 0], 2.0)
    np.testing.assert_allclose(h[1, :, 0], 3.0)


def test_in_flight_history_pads_short_and_empty_histories():
    short = in_flight_history([np.array([5.0, 5.0, 5.0])], 3, 2)
    np.testing.assert_allclose(short[:, :, 0], 5.0)  # repeat the oldest: it was being held
    empty = in_flight_history([], 2, 2)
    np.testing.assert_allclose(empty, 0.0)  # nothing published yet -> standing still


def test_plan_control_walks_the_plan():
    U = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], np.float32)
    np.testing.assert_allclose(plan_control_at(U, 0.0, 0.1), [1.0, 2.0])
    np.testing.assert_allclose(plan_control_at(U, 0.1, 0.1), [3.0, 4.0])
    np.testing.assert_allclose(plan_control_at(U, 0.05, 0.1), [2.0, 3.0])  # interpolated, not held
    np.testing.assert_allclose(plan_control_at(U, 0.15, 0.1), [4.0, 5.0])


def test_plan_control_clamps_at_both_ends():
    U = np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)
    np.testing.assert_allclose(plan_control_at(U, -1.0, 0.1), [1.0, 2.0])
    np.testing.assert_allclose(plan_control_at(U, 99.0, 0.1), [3.0, 4.0])  # stale plan -> hold last
    np.testing.assert_allclose(plan_control_at(U[:1], 5.0, 0.1), [1.0, 2.0])


def test_joint_states_identity_mapping():
    # current LLC: /joint_states is all-positive-forward wheel rad/s (verified on
    # bags/motors0 + steps_air) -> identity mapping, reordered to engine (wL, wR, rear),
    # robust to message ordering.
    from helhest.control.command import joint_states_to_model

    om = joint_states_to_model(["rear_wheel_j", "left_wheel_j", "right_wheel_j"], [3.0, 1.0, 2.0])
    assert om is not None and np.allclose(om, [1.0, 2.0, 3.0])


def test_joint_states_missing_joint_is_none():
    from helhest.control.command import joint_states_to_model

    assert joint_states_to_model(["left_wheel_j"], [1.0]) is None


def test_spin_side_holds_through_a_spin_and_releases_on_driving():
    from helhest.control.command import spin_side

    left_spin = np.array([-2.0, 0.0, 2.0], np.float32)
    assert spin_side(0.0, left_spin) == 1.0
    assert spin_side(1.0, np.array([2.0, 0.0, -2.0], np.float32)) == -1.0  # a new spin decides
    assert spin_side(1.0, np.zeros(3, np.float32)) == 1.0  # stopped between frames: still left
    assert spin_side(1.0, np.array([1.5, 1.5, 1.8], np.float32)) == 0.0  # drove off: released
    # the planner's choice locks the side before the jerk-limited command has ramped up
    still = np.array([0.1, 0.0, -0.1], np.float32)
    assert spin_side(0.0, still, planned=np.array([2.0, -2.0], np.float32)) == -1.0
    assert spin_side(1.0, still, planned=np.array([2.0, -2.0], np.float32)) == 1.0  # held


def test_llc_not_driving_needs_both_a_moving_command_and_idle_setpoints():
    from helhest.control.command import llc_not_driving

    moving = np.array([1.5, 1.5, 1.5], np.float32)
    assert llc_not_driving(moving, idle_for_s=0.8)  # e-stop held
    assert not llc_not_driving(moving, idle_for_s=0.1)  # the LLC has not picked the command up yet
    at_rest = np.array([0.2, 0.2, 0.2], np.float32)
    assert not llc_not_driving(at_rest, idle_for_s=5.0)


def test_traction_scale_fades_out_as_the_route_falls_behind():
    import math

    from helhest.control.command import traction_scale

    assert traction_scale(math.radians(30), 60, 120) == 1.0
    assert abs(traction_scale(math.radians(90), 60, 120) - 0.5) < 1e-9
    assert traction_scale(math.radians(-170), 60, 120) == 0.0
    assert traction_scale(math.radians(170), 60, 0.0) == 1.0  # off_deg 0: always fully


def test_anchor_pulls_a_command_the_wheels_are_not_following():
    from helhest.control.command import anchor_to_wheels

    held = np.zeros(3, np.float32)  # brake trigger: the wheels stand still
    prev = np.array([4.0, 4.0, 4.0], np.float32)
    accel = np.array([3.0, 3.0, 3.0], np.float32)
    cmd, acc = anchor_to_wheels(prev, accel, held, max_lead=1.5)
    np.testing.assert_allclose(cmd, [1.5, 1.5, 1.5])
    np.testing.assert_allclose(acc, 0.0)  # the ramp restarts rather than carrying on at full slope
    # backward too
    cmd, _ = anchor_to_wheels(-prev, -accel, held, max_lead=1.5)
    np.testing.assert_allclose(cmd, [-1.5, -1.5, -1.5])


def test_anchor_leaves_a_following_robot_and_never_raises_the_command():
    from helhest.control.command import anchor_to_wheels

    prev = np.array([3.0, 3.0, 3.0], np.float32)
    accel = np.array([2.0, 2.0, 2.0], np.float32)
    following = np.array([2.6, 2.7, 2.8], np.float32)  # the motor lag, well inside the lead
    cmd, acc = anchor_to_wheels(prev, accel, following, max_lead=1.5)
    np.testing.assert_array_equal(cmd, prev)
    np.testing.assert_array_equal(acc, accel)
    # braking: the wheels lag ABOVE a command that is ramping down -- not dragged back up
    cmd, _ = anchor_to_wheels(np.full(3, 1.0, np.float32), accel, np.full(3, 4.0, np.float32), 1.5)
    np.testing.assert_allclose(cmd, 1.0)


def test_anchor_turns_a_held_ramp_into_a_smooth_release():
    """Stromovka 13_43_24 in miniature: the wheels are held at 0 for 10 s under a 4 rad/s target.
    Unanchored, the command is already at the target when they are let go; anchored, it is within
    the lead and ramps from there."""
    from helhest.control.command import anchor_to_wheels

    dt, target = 0.1, 4.0

    def release_step(max_lead: float) -> float:
        prev, accel, wheels = Z.copy(), Z.copy(), Z.copy()
        for _ in range(100):  # held
            if max_lead > 0.0:
                prev, accel = anchor_to_wheels(prev, accel, wheels, max_lead)
            cmd = condition_command(
                target,
                target,
                prev,
                max_omega=5.0,
                max_slew=3.0,
                dt=dt,
                prev_accel=accel,
                max_jerk=5.0,
            )
            accel, prev = (cmd - prev) / dt, cmd
        return float(prev[1] - wheels[1])  # what the LLC is asked for the instant it is let go

    assert release_step(0.0) > 3.9
    assert release_step(1.5) <= 1.5 + 5.0 * dt * dt + 1e-6  # the lead plus one jerk-limited tick
