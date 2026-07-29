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
            {"base_frame": "odin1_base_link"},  # cloud is already in this frame -> sensor TF = identity
            {"map_frame": "map"},
            {"publish_map_tf": True},  # elevation owns map -> odom_odin ...
            {"publish_odom_tf": True},  # ... and odom_odin -> odin1_base_link
        ],
    )

    return LaunchDescription(
        [
            odin_driver,
            fix_odom_twist,
            elevation,
        ]
    )
