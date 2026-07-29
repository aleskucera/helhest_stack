"""Odin1-driven elevation_node: map / traversability / MPPI, no ICP, no EKF.

Consumer side of the Odin pipeline (start the sensor with odin_driver.launch.py first).
Restarting this launch retunes params / resets the accumulated map without touching the
driver -- /odin1/cloud_raw + /odin1/odometry keep flowing.

elevation_node consumes the Odin raw cloud + SLAM pose with icp_enable=False, so the pose
is trusted directly (pure dead-reckoning, no scan-to-map correction) and the cloud only
feeds the map / traversability / planning. The plain node has no EKF. It owns the whole
map -> odom_odin -> odin1_base_link TF chain (publish_map_tf + publish_odom_tf).

TUNING: the speed / turn knobs are exposed as launch arguments, so e.g.
    ros2 launch <repo>/ros/odin/odin_elevation.launch.py plan_wmax:=5.0 k_turn:=0.6
actually applies them (defaults match elevation_node). See TUNABLE below.

SAFETY: plan_actuate is hardcoded False here -- this launch ALWAYS comes up viz-only. To
drive, set it live yourself: `ros2 param set /elevation plan_actuate true` (motor ceiling
~5.3 rad/s; plan is device-centered until a base_link->odin1_base_link mount TF exists).

Run (source ~/.rosrc + the helhest install; zenoh router up; driver launch running):

    ros2 launch <repo>/ros/odin/odin_elevation.launch.py [arg:=value ...]
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# name -> (default, one-line help). Defaults match elevation_node's own defaults, so a
# no-arg launch behaves exactly as before. All are floats.
TUNABLE = {
    "plan_wmax": ("4.0", "planner top per-wheel speed [rad/s] -- the real speed cap (<= plan_max_omega, motor ~5.3)"),
    "plan_max_omega": ("5.0", "hard output clamp on |wheel vel| [rad/s] -- motor safe max (<= ~5.3)"),
    "k_turn": ("-1.0", "turn gain override; <0 = use terrain preset"),
    "plan_turn": ("0.03", "wheel-differential penalty -> higher = gentler/straighter (<= ~0.1)"),
    "plan_max_slew": ("6.0", "accel cap on d(cmd)/dt [rad/s^2] -> lower = smoother turn ease-in"),
    "plan_goal_running": ("0.3", "progress pull -> higher = faster toward goal"),
    "plan_effort": ("0.001", "wheel-speed^2 penalty -> lower = faster"),
}


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument(name, default_value=default, description=help_)
        for name, (default, help_) in TUNABLE.items()
    ]
    tuned = [
        {name: ParameterValue(LaunchConfiguration(name), value_type=float)}
        for name in TUNABLE
    ]

    elevation = Node(
        package="helhest_stack_ros",
        executable="elevation_node",
        name="elevation",
        output="screen",
        parameters=[
            {"lidar_topic": "/odin1/cloud_raw"},  # raw dTOF cloud replaces Ouster
            {"odom_topic": "/odin1/odometry"},  # Odin SLAM pose replaces /odom_2d + ICP/EKF
            {"imu_topic": "/odin1/imu"},
            {"icp_enable": False},  # trust the Odin pose directly -- no scan-to-map correction
            {"deskew_enable": False},  # Odin per-point time is seconds; the deskew assumes ns
            # The gyro rotation prior exists to patch skid-degraded wheel-odom yaw. The Odin
            # odometry is a full 6-DOF SLAM pose (orientation trustworthy) AND the Odin IMU is
            # ~90deg-rotated from base with no correct extrinsic here -> the prior rolls the map
            # pose ~40deg on flat ground. Off: predict() uses the (level) Odin odom rotation.
            {"imu_rotation_prior": False},
            {"base_frame": "odin1_base_link"},  # cloud is already in this frame -> sensor TF = identity
            # Robot self-filter box in odin1_base_link (x fwd, y left). Measured from cloud_raw
            # with nothing but the robot within 1 m: the robot's above-floor returns are a tight
            # body (99% within x<=0.22, |y|<=0.48) plus a front-wheel rim to x~0.55, |y|~0.75.
            # This box catches ~99.95%. The default (x[0.10,0.60] y[+-0.55]) was Ouster-on-base_link.
            {"self_filter_enable": True},
            {"self_x_min": -0.05},
            {"self_x_max": 0.55},
            {"self_y_min": -0.75},
            {"self_y_max": 0.75},
            # odin1_base_link sits ~0.49 m above the floor (measured from cloud_raw); the
            # default 0.4 assumed base_link. Wrong height stamps a raised square under the
            # robot in overwrite mode -> bad elevation_global.
            {"footprint_robot_height": 0.49},
            {"map_frame": "map"},
            {"publish_map_tf": True},  # elevation owns map -> odom_odin ...
            {"publish_odom_tf": True},  # ... and odom_odin -> odin1_base_link
            # VIZ ONLY. Hardcoded False (not a launch arg) so this launch NEVER comes up armed.
            # Drive only by a deliberate live `ros2 param set /elevation plan_actuate true`.
            {"plan_actuate": False},
            *tuned,  # the launch-argument-driven speed/turn knobs
        ],
    )

    return LaunchDescription([*args, elevation])
