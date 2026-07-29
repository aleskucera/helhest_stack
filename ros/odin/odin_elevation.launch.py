"""Odin1 odometry + raw cloud -> elevation_node, no ICP, no EKF.

Brings up the whole perception+planning pipeline driven entirely by the Odin1
solid-state LiDAR instead of the Ouster + wheel-odom + ICP/EKF stack:

  * host_sdk_sample (odin_ros_driver) publishes the RAW dTOF cloud
    (/odin1/cloud_raw) and the on-device SLAM pose (/odin1/odometry_bad_twist).
  * odom_twist_to_child_frame (nav_utils) fixes the odom twist frame and
    republishes it as /odin1/odometry.
  * elevation_node consumes both with icp_enable=False -> the Odin pose is
    trusted directly (pure dead-reckoning, no scan-to-map correction) and the
    raw cloud only feeds the map / traversability / MPPI planning.

TF ownership: elevation_node broadcasts the whole map -> odom_odin ->
odin1_base_link chain (publish_map_tf + publish_odom_tf). We therefore do NOT
run the driver's odom_to_tf node (cras_odin_driver's launch does) -- it would
publish odin1_base_link -> odom_odin, the inverse edge, and split the tree.

Run (source ~/.rosrc, the helhest install, and the odin_ws install first;
a zenoh router must be up):

    ros2 launch <repo>/ros/odin/odin_elevation.launch.py
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch_ros.actions import Node

_HERE = os.path.dirname(os.path.abspath(__file__))
_ODIN_CONFIG = os.path.join(_HERE, "control_command_raw.yaml")
# The cras yaml-patch lets host_sdk_sample load a custom config via env (its own
# `config_file` param is ignored upstream). Same mechanism as cras_odin_driver.
_YAML_PATCH = os.path.join(
    get_package_prefix("cras_odin_driver"), "lib", "libodin_yaml_parser_patched.so"
)


def generate_launch_description() -> LaunchDescription:
    odin_driver = Node(
        package="odin_ros_driver",
        executable="host_sdk_sample",
        name="odin1",
        output="screen",
        # Scope the config env + yaml-patch preload to THIS process only. A global
        # LD_PRELOAD would inject the C++ yaml-cpp override into elevation_node's
        # rclpy/Warp process too and segfault it.
        additional_env={
            "ODIN_ROS_CONFIG_FILE": _ODIN_CONFIG,
            "LD_PRELOAD": _YAML_PATCH,
        },
        # raw odom carries a bad twist frame; nav_utils fixes it into /odin1/odometry
        remappings=[("odin1/odometry", "odin1/odometry_bad_twist")],
    )

    fix_odom_twist = Node(
        package="nav_utils",
        executable="odom_twist_to_child_frame",
        name="odin1_fix_odom",
        remappings=[
            ("odom", "odin1/odometry_bad_twist"),
            ("odom_out", "odin1/odometry"),
        ],
        parameters=[
            {"transform_linear": True},
            {"transform_angular": False},
            {"parent_frame": "odom_odin"},
        ],
    )

    # The Odin publishes /odin1/imu in frame imu_link but no TF connecting it to
    # the tree, so elevation_node's 400 Hz IMU->base lookup floods warnings. The
    # IMU is unused here (icp + deskew off), so identity just completes the tree
    # and silences it. Replace with the measured Odin IMU mount if deskew/gravity
    # are ever turned back on.
    imu_static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="odin1_imu_tf",
        arguments=[
            "--frame-id", "odin1_base_link",
            "--child-frame-id", "imu_link",
        ],
    )

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
            # VIZ ONLY. plan_actuate DEFAULTS TRUE -> a /goal_pose would publish /cmd_joints and
            # DRIVE the robot. Here the plan is relative to odin1_base_link (the device, offset
            # from the real base_link), so any command would be geometrically wrong. Keep off
            # until the base_link->odin1_base_link mount TF exists and driving is intended.
            {"plan_actuate": False},
        ],
    )

    return LaunchDescription(
        [
            odin_driver,
            fix_odom_twist,
            imu_static_tf,
            elevation,
        ]
    )
