"""Odin1 sensor bring-up: raw dTOF cloud + fused SLAM odometry (+ IMU frame TF).

Sensor side of the Odin pipeline, split from the consumer so elevation_node can be
restarted (param tuning / map reset) without the ~7 s USB + SLAM re-init here:

  * host_sdk_sample (odin_ros_driver) publishes the RAW dTOF cloud (/odin1/cloud_raw)
    and the on-device SLAM pose (/odin1/odometry_bad_twist).
  * odom_twist_to_child_frame (nav_utils) fixes the odom twist frame and republishes
    it as /odin1/odometry.
  * a static odin1_base_link->imu_link TF (identity) connects the IMU frame so a
    consumer's IMU->base lookup does not flood warnings. It lives here (sensor mount)
    so it persists across elevation restarts.

We deliberately do NOT run the cras odom_to_tf node -- elevation_node owns the
map -> odom_odin -> odin1_base_link chain; odom_to_tf would publish the inverse edge
and split the tree.

Run (source ~/.rosrc, the helhest install, and the odin_ws install; zenoh router up):

    ros2 launch <repo>/ros/odin/odin_driver.launch.py
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
        # LD_PRELOAD would inject the C++ yaml-cpp override into a Python/Warp consumer
        # process too and segfault it.
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

    # Identity odin1_base_link->imu_link so a consumer's 400 Hz IMU->base lookup resolves.
    # Replace with the measured Odin IMU mount if deskew/gravity are ever turned back on.
    imu_static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="odin1_imu_tf",
        arguments=[
            "--frame-id", "odin1_base_link",
            "--child-frame-id", "imu_link",
        ],
    )

    return LaunchDescription([odin_driver, fix_odom_twist, imu_static_tf])
