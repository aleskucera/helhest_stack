"""Map a planner wheel command to the real robot's /cmd_joints, with motor-safety conditioning.

The planner works in a self-consistent "both wheels >= 0 = forward" convention (the MPPI box has
wmin=0). The robot's /cmd_joints INPUT convention is the same: forward = all wheels POSITIVE
([+wl, +rear, +wr]), so the command passes straight through -- no sign flip. Verified live on the
robot 2026-07-10 (an all-positive /cmd_joints drove straight forward).

  NOTE the LLC's /joint_setpoints OUTPUT echoes forward as [-, -, +] (it negates left/rear
  internally). That output convention is what the manual-drive bags recorded, but it does NOT
  apply to commands -- the LLC does its own internal sign mapping on the /cmd_joints input.

This is the single place all actuator-safety logic lives, so it is auditable and unit-tested:
  1. rear-as-follower                               (rear = mean(L, R); left/right pass through)
  2. asymmetric accel/decel rate limit, optionally jerk-limited  (a jumpy MPPI step can't shock
     the drivetrain; the planner's rollouts run the same `jerk_limited_step`)
  3. a hard per-joint magnitude clamp               (final backstop below the motor's safe max)
"""

from __future__ import annotations

import math

import numpy as np

# /cmd_joints JointState.name order -- matches /joint_setpoints on the robot. The [left, rear, right]
# arrays returned by condition_command are in THIS order.
JOINT_NAMES = ("left_wheel_j", "rear_wheel_j", "right_wheel_j")


def joint_states_to_model(names: list[str], velocities: list[float]) -> np.ndarray | None:
    """LLC /joint_states -> model-convention wheel speeds [wL, wR, w_rear], matching joints by
    NAME so message ordering doesn't matter. Returns None when any drive joint is missing
    (partial message -> caller falls back to the last command).

    On the CURRENT LLC the measured stream is already all-positive-forward in wheel rad/s --
    the same convention and units as the /cmd_joints input, so the mapping is identity.
    Verified on bags/motors0 + bags/steps_air (2026-08-11/12, post the 2026-07-27 unit fix):
    all three joints read POSITIVE under a pure forward command, measured/commanded median
    0.976-0.998. The old motor-side [-left, -rear, +right] convention survives only in the
    /joint_setpoints echo (see the module header) -- do NOT apply it here."""
    vel_by_name = dict(zip(names, velocities))
    try:
        return np.array(
            [float(vel_by_name[j]) for j in ("left_wheel_j", "right_wheel_j", "rear_wheel_j")],
            np.float32,
        )
    except KeyError:
        return None


def turn_first(
    wl: float,
    wr: float,
    heading_error: float,
    *,
    start_deg: float = 45.0,
    full_deg: float = 110.0,
    min_scale: float = 0.1,
    prev_diff: float | None = None,
    commit_deg: float = 90.0,
    speed: float | None = None,
    stop_below: float = 0.3,
) -> tuple[float, float]:
    """Slow the FORWARD component when the way on is well off the robot's heading, so a large
    turn is made in place before driving rather than as an arc that also advances.

    `heading_error` [rad] is the angle from the heading to where the route falls away. Below
    `start_deg` nothing changes; from there the mean speed scales linearly down to `min_scale`
    at `full_deg` and beyond. The differential is kept, so at a full brake the command is the
    spin the planner asked for without the advance.

    Why: forward-only, a minimum-radius arc that turns 52 deg advances 1.4 m (false_door, the
    anchored-map runs). Begun 1.9 m from a wall it ends 0.55 m from it, inside the robot's own
    turning clearance, where every forward and spinning rollout is vetoed and MPPI freezes.

    `prev_diff` is last frame's published differential (wr - wl). While the error is past
    `commit_deg` a spin already under way keeps its direction: with the way on straight behind,
    left and right cost the same, and the planner's elite picked a different one every frame
    (corridor: four reversals in 60 frames). A spin here is not in place -- 0.16 m sideways per
    95 deg -- so each reversal walked the robot toward a wall until the last 50 deg would have
    swept a corner into it and everything was vetoed. One direction, decided once.

    `speed` [m/s] is the robot's measured ground speed. At a full brake with the robot still
    moving faster than `stop_below`, the command is a plain stop and the spin waits: a spin
    begun at cruise carries the cruise momentum through the skid -- 0.9 m sideways per 120 deg
    in the corridor -- where a spin from rest travels 0.2 m (measured in the same physics at
    1.5-6 rad/s). Stop, then turn.
    """
    e = abs(math.degrees(heading_error))
    if e <= start_deg or full_deg <= start_deg:
        return wl, wr
    scale = max(min_scale, 1.0 - (e - start_deg) / (full_deg - start_deg) * (1.0 - min_scale))
    if e >= full_deg and speed is not None and speed > stop_below:
        return 0.0, 0.0
    mean = 0.5 * (wl + wr)
    half_diff = 0.5 * (wr - wl)
    if (
        prev_diff is not None
        and e > commit_deg
        and abs(prev_diff) > 1.0
        and half_diff * prev_diff < 0.0
    ):
        half_diff = -half_diff
    return mean * scale - half_diff, mean * scale + half_diff


# [s] the LLC's own wheel setpoints held at zero this long under a moving command = not driving
IDLE_FOR_S = 0.3
# [rad/s] a front-wheel command above this counts as asking for motion
MOVING_CMD = 0.5


def llc_not_driving(cmd: np.ndarray, idle_for_s: float) -> bool:
    """True when the published command ([L, rear, R]) asks for motion but the LLC has held its
    own front-wheel setpoints (/joint_setpoints) at zero for IDLE_FOR_S: an e-stop or a driver
    cut-out. /estop_active does not show the button (false throughout the e-stop below).

    The command tracker continues from its own last output, so without this it keeps ramping
    while nothing moves and the LLC jumps to the stale command on release: `turns_twist`
    (2026-09-29), an e-stop held 9 s while the forward command ramped to 1.85 m/s, then the
    setpoints went 0 -> 6 rad/s in 0.4 s. Keyed on the SETPOINTS, not the wheels: in `drive` the
    wheels stood still for 4-7 s under setpoints of 2-4 rad/s -- stalls, where holding the
    command at rest would keep the wheels from ever breaking loose.
    """
    return idle_for_s > IDLE_FOR_S and max(abs(float(cmd[0])), abs(float(cmd[2]))) > MOVING_CMD


def traction_scale(heading_error: float, on_deg: float, off_deg: float) -> float:
    """How much of the traction cost applies this frame: 1 while the route lies within `on_deg`
    of the heading, 0 beyond `off_deg`, linear between; `off_deg` <= 0 = always 1.

    Traction (CostParams.traction) makes a rolling turn cheaper than a spin, which is right while
    the way on is roughly ahead and wrong when it lies behind: under the corner limit a rolling
    U-turn needs a radius of ~1.7 m at 1 m/s, and in `tree2` (2026-09-30) goals 148-172 deg behind
    took 25-29 m of driving for 10-15 m of straight line. Turned off there, MPPI plans the spin.
    """
    if off_deg <= 0.0:
        return 1.0
    e = abs(math.degrees(heading_error))
    if e <= on_deg:
        return 1.0
    if e >= off_deg:
        return 0.0
    return (off_deg - e) / (off_deg - on_deg)


def spin_side(prev_side: float, cmd: np.ndarray, planned: np.ndarray | None = None) -> float:
    """Which way MPPI may spin next frame: +1 left, -1 right, 0 either.

    `cmd` is the command just published ([L, rear, R]); `planned` the planner's first step
    (wL, wR), when there is a fresh one. A spin -- barely advancing, clearly turning -- fixes the
    side as soon as the PLANNER picks it; the published command would be too late, because under
    the jerk limit a planner that flips side each frame never lets it ramp past the threshold
    (drive_sim, goal straight behind: +-5 deg of dithering for 20 s, wheels at +-0.3 rad/s). The
    side holds while the robot stands or keeps spinning and is released only once it drives off
    (or the caller resets it for a new goal), so a spin finishes the way it started. With the
    goal behind, left and right cost the same and the planner alternated every ~1.6 s on the
    robot (`turns` bag, 2026-09-29).
    """
    mean = 0.5 * (float(cmd[0]) + float(cmd[2]))
    diff = float(cmd[2]) - float(cmd[0])
    if abs(mean) >= 0.5:  # [rad/s] driving: the turn is over
        return 0.0
    if abs(diff) > 1.0:  # [rad/s] a spin under way
        return math.copysign(1.0, diff)
    if prev_side == 0.0 and planned is not None:
        p_mean = 0.5 * (float(planned[0]) + float(planned[1]))
        p_diff = float(planned[1]) - float(planned[0])
        if abs(p_mean) < 0.5 and abs(p_diff) > 1.0:  # the planner chose a spin
            return math.copysign(1.0, p_diff)
    return prev_side


def condition_command(
    wl: float,
    wr: float,
    prev: np.ndarray,
    *,
    max_omega: float,
    max_slew: float,
    dt: float,
    max_decel: float | None = None,
    turn_boost: float = 1.0,
    goal_dist: float | None = None,
    brake_dist: float = 0.0,
    turn_brake_a_max: float = 0.0,
    lat_gain: float = 0.0,
    turn_brake_scale: float = 1.0,
    prev_accel: np.ndarray | None = None,
    max_jerk: float = 0.0,
) -> np.ndarray:
    """Planner (wl, wr) -> conditioned [left, rear, right] wheel-velocity command for /cmd_joints.

    wl, wr: planner wheel speeds (model convention, >= 0; both positive = forward).
    prev: the previously PUBLISHED [left, rear, right] command. Pass zeros on the first call /
        after an e-stop so the slew limiter ramps up from rest.
    max_omega: hard cap on |wheel velocity| [rad/s] -- set to the motor's safe max.
    max_slew: cap on |d(command)/dt| for a joint SPEEDING UP [rad/s^2] (acceleration).
    max_decel: cap on |d(command)/dt| for a joint SLOWING toward rest [rad/s^2] (deceleration,
        incl. the stop ramp). None = use max_slew (symmetric limit, the old behaviour).
    goal_dist: current robot->goal distance [m]. With brake_dist > 0, scales the FORWARD speed down
        on the final approach (the goal brake below). None/0 disables the brake.
    brake_dist: [m] start braking within this range of the goal. 0 = no brake.
    turn_brake_a_max: [m/s^2] lateral-acceleration ceiling for the TURN BRAKE below. 0 = off.
    lat_gain: [m/s^2 per (rad/s)^2] converts mean*diff into lateral acceleration for this robot,
        = wheel_radius^2 / (2*half_track*alpha). Only read when turn_brake_a_max > 0.
    turn_brake_scale: extra speed scale in [0, 1] from the caller's LOOKAHEAD along the committed
        plan (1.0 = no anticipation). Applied the same way as the reactive brake.
    prev_accel: [left, rear, right] acceleration of the previous command [rad/s^2], i.e.
        (prev - the one before) / its dt. Only read when max_jerk > 0; None = at rest.
    max_jerk: [rad/s^3] cap on how fast each joint's acceleration may change. 0 = off (plain
        rate limit). With it on, max_slew and max_decel are the acceleration bounds.
    Returns [left, rear, right] velocities to publish. To STOP, call with wl = wr = 0 -- the slew
    limiter ramps the command down to rest.
    """
    # Split into forward (mean) + turn (differential). The drivetrain realizes the forward command
    # 1:1 but only ~half the TURN differential (the two motors equalize under load; measured over
    # outdoor bags). turn_boost amplifies the commanded differential to compensate so the wheels
    # actually deliver the yaw MPPI intended -- forward speed (mean) is untouched. 1.0 = no boost;
    # ~2.0 recovers the measured ~0.5 realization. Tune in the field.
    # *** HOTFIX / stopgap for a drivetrain defect -- NOT a real fix. Read
    #     docs/field/turn_differential_hotfix.md before changing/removing this: what it papers
    #     over, and what to actually fix. ***
    mean = 0.5 * (wl + wr)  # forward speed; also the rear follower target (rear = mean of L/R)
    # GOAL BRAKE: the robot is forward-only (wmin=0) -- it cannot pivot in place to re-aim, so if it
    # arrives fast and slightly off it flies PAST the goal and orbits (a hard stop-radius misses an
    # offset flyby entirely). Scaling forward speed linearly to 0 over the last brake_dist metres
    # makes it nose in slow -> settles AT the goal. The turn differential is NOT scaled (tighter arc
    # at low speed helps the final aim) and the far-field cruise is untouched (no slow-down until
    # inside brake_dist). Verified in sim vs a term_v MPPI cost + a sqrt profile: this linear output
    # brake settled cleanest (~0.2 m, zero overshoot) across straight/offset/sharp goals.
    if brake_dist > 0.0 and goal_dist is not None:
        mean *= min(1.0, float(goal_dist) / float(brake_dist))
    diff = (wr - wl) * float(turn_boost)  # turn differential, amplified
    # TURN BRAKE: slow INTO a corner the way a driver does, instead of carrying cruise speed
    # through it. v = R*mean and wz = R*diff/(2*half_track*alpha), so lateral acceleration is
    # a_lat = v*wz = lat_gain * mean * diff. Scaling `mean` ALONE (what the goal brake does) would
    # leave diff untouched and TIGHTEN the arc -- lifting off mid-corner while holding the same
    # steering lock. Scaling mean and diff by the SAME s keeps v/wz, hence the radius, and drops
    # a_lat by s^2, so the robot tracks the planned path at a lower speed. The rate limiter below
    # then gives the gentle accelerate-out for free (max_slew < max_decel).
    scale = float(np.clip(turn_brake_scale, 0.0, 1.0))  # caller's lookahead cap
    if turn_brake_a_max > 0.0 and lat_gain > 0.0:
        a_lat = abs(lat_gain * mean * diff)  # uses the BOOSTED diff -- what the robot will do
        if a_lat > turn_brake_a_max:
            scale = min(scale, math.sqrt(float(turn_brake_a_max) / a_lat))
    mean *= scale
    diff *= scale
    # /cmd_joints input convention: forward = all positive, no sign flip -- the LLC applies its own
    # internal signs. Verified: forward (wl=wr=v) -> [+v, +v, +v] drove the robot straight forward.
    target = np.array([mean - 0.5 * diff, mean, mean + 0.5 * diff], dtype=np.float32)
    prev = np.asarray(prev, dtype=np.float32)
    # Asymmetric rate limit: a joint speeding UP (|cmd| growing) is capped by max_slew (accel); a
    # joint slowing DOWN toward rest (|cmd| shrinking, incl. the stop ramp) by max_decel. Per-joint
    # because in a turn one wheel accelerates while the other decelerates. None = symmetric.
    decel = float(max_slew if max_decel is None else max_decel)
    if max_jerk > 0.0:
        accel = np.zeros(3, np.float32) if prev_accel is None else prev_accel
        # Clamp the TARGET, so the tracker ramps into the limit; the backstop below cutting a
        # ramp off at max_omega would be a jerk spike of its own.
        target = np.clip(target, -float(max_omega), float(max_omega))
        cmd = jerk_limited_step(target, prev, accel, float(max_slew), decel, max_jerk, dt)
    else:
        d_acc = float(max_slew) * float(dt)
        d_dec = decel * float(dt)
        lim = np.where(np.abs(target) >= np.abs(prev), d_acc, d_dec)  # per joint: accel vs decel
        cmd = prev + np.clip(target - prev, -lim, lim)  # rate limit
    cmd = np.clip(cmd, -float(max_omega), float(max_omega))  # hard magnitude backstop
    return cmd.astype(np.float32)


def jerk_limited_step(
    target: np.ndarray,
    prev: np.ndarray,
    prev_accel: np.ndarray,
    max_accel: float,
    max_decel: float,
    max_jerk: float,
    dt: float,
) -> np.ndarray:
    """One tick of a per-joint tracker whose acceleration is bounded and changes at most max_jerk.

    A plain rate limit bounds the acceleration but lets it flip from +max to -max in one tick,
    and MPPI replanning every frame asks for exactly that: the `turns` bag reversed each front
    wheel's acceleration 2.7 times a second, half the time at the cap. Here the acceleration
    itself ramps. Mirrored on device by mppi._jerk_limited_step, so the rollouts plan with the
    command the wheels will get; keep the two identical.

    Returns the new command. Its acceleration is (cmd - prev) / dt, the caller's next prev_accel.
    """
    target = np.asarray(target, np.float32)
    prev = np.asarray(prev, np.float32)
    accel = np.asarray(prev_accel, np.float32)
    error = target - prev
    lim = np.where(np.abs(target) >= np.abs(prev), max_accel, max_decel)
    # The largest acceleration from which ramping down to zero at max_jerk still ends ON the
    # target: the ramp a, a - j*dt, ... moves the command by a^2/(2j) + a*dt/2.
    reach = max_jerk * (np.sqrt(0.25 * dt * dt + 2.0 * np.abs(error) / max_jerk) - 0.5 * dt)
    wanted = np.sign(error) * np.minimum(lim, reach)
    new_accel = np.clip(wanted, accel - max_jerk * dt, accel + max_jerk * dt)
    # Land exactly when both the landing tick and the stop after it are within the jerk limit;
    # snapping otherwise would be the unbounded jerk this exists to prevent, so it overshoots a
    # little and comes back instead.
    land = error / dt
    can_land = (np.abs(land - accel) <= max_jerk * dt) & (np.abs(land) <= max_jerk * dt)
    new_accel = np.where(can_land, land, new_accel)
    return (prev + new_accel * dt).astype(np.float32)


def to_engine_order(cmd: np.ndarray) -> np.ndarray:
    """/cmd_joints order (left, rear, right) -> the engine's wheel vec3 (wL, wR, w_rear).

    JOINT_NAMES puts the REAR wheel in the middle; every kernel expects it last. One place for the
    swap so a caller can't get it silently backwards.
    """
    cmd = np.asarray(cmd, dtype=np.float32)
    return np.array([cmd[0], cmd[2], cmd[1]], dtype=np.float32)


def in_flight_history(commands, steps: int, batch: int) -> np.ndarray:
    """[steps, batch, 3] buffer for `ForwardSimulator.command_history`, oldest first.

    `commands` are the wheel commands already published but not yet acted on, in ENGINE order and
    oldest first; row k acts on rollout step k. Fewer than `steps` of them (at startup, or after a
    delay change) repeats the oldest, i.e. the robot is assumed to have been holding it; none at
    all gives zeros, i.e. standing still.
    """
    rows = [np.asarray(c, dtype=np.float32) for c in commands]
    if not rows:
        rows = [np.zeros(3, dtype=np.float32)]
    while len(rows) < steps:
        rows.insert(0, rows[0])
    stacked = np.asarray(rows[-steps:], dtype=np.float32)[:, None, :]
    return np.ascontiguousarray(np.repeat(stacked, batch, axis=1), dtype=np.float32)


def plan_control_at(nominal: np.ndarray, elapsed: float, plan_dt: float) -> np.ndarray:
    """The committed (wL, wR) at `elapsed` seconds into a plan sampled every `plan_dt`.

    A plan is a trajectory, not a single command, so a controller ticking faster than the planner
    should WALK it rather than hold its first step. Linear interpolation between plan steps: at
    dt = 0.1 s and a 50 ms tick that is the difference between two distinct commands per plan and
    the same one twice.

    Clamped at both ends -- before the start it gives the first step, past the horizon the last, so
    a stale plan degrades to holding its final command rather than indexing off the end.
    """
    nominal = np.asarray(nominal, dtype=np.float32)
    if nominal.ndim != 2 or nominal.shape[1] < 2:
        raise ValueError(f"nominal must be [T, >=2], got {nominal.shape}")
    if len(nominal) == 1:
        return nominal[0, :2].astype(np.float32)
    position = float(np.clip(elapsed / plan_dt, 0.0, len(nominal) - 1))
    step = int(np.floor(position))
    if step >= len(nominal) - 1:
        return nominal[-1, :2].astype(np.float32)
    frac = position - step
    return ((1.0 - frac) * nominal[step, :2] + frac * nominal[step + 1, :2]).astype(np.float32)
