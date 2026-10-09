#!/bin/bash
# Record a GPX / QR mission: navigation_node's inputs and outputs (as record_nav.sh) PLUS the mission
# side -- the route the follower sends, the commander bridge's state, the follower's state and
# events, and the Fixposition pose and fusion status. Everything is on one ROS graph (Zenoh), so
# this one recorder on the jetson captures both machines.
#
# Usage:  ./record_mission.sh <name>     (start before the mission, Ctrl-C to stop)
#
# Bags land in ~/bags/<name> as MCAP with zstd chunk compression.
set -e

TOPICS=(
  # --- navigation_node: sensors, frames, planning, the drive chain (as record_nav.sh) ---
  /odin1/cloud_filtered /odin1/odometry /odin1/imu /odom_2d
  /tf /tf_static
  /goal_pose                # the goals navigation_node was given (RViz or the bridge)
  /planned_path             # what MPPI committed to, each frame
  /cmd_vel /cmd_joints      # the drive output (cmd_output twist -> /cmd_vel)
  /joint_setpoints /joint_states /estop_active
  /debug/angle /debug/setpoint /debug/error
  /odin1/image/compressed
  # --- the mission: route, bridge, follower ---
  /goal_sequence            # the whole route the follower sent (PoseArray, FP_ECEF)
  /goal_waypoint            # single goals (goto)
  /crl_commander/state      # the bridge's mode: STOP / GOTO / SEQUENCE
  /road_follower/state /road_follower/event /road_follower/route_path /road_follower/route_source
  /road_follower/home
  /gps_waypoints_markers    # the route as RViz markers
  # --- the Fixposition: where the robot is in the world, and whether its fusion is healthy ---
  /fixposition/odometry_llh /fixposition/odometry_enu /fixposition/ypr /fixposition/fpa/odomstatus
)

if [[ -z "${1:-}" || "$1" == "-h" || "$1" == "--help" ]]; then
  sed -n '2,/^set -e/p' "$0" | grep '^#' | sed 's/^# \?//'
  exit 0
fi

NAME="$1"
DEST="$HOME/bags/$NAME"
if [[ -e "$DEST" ]]; then
  read -r -p "~/bags/$NAME already exists -- overwrite? [y/N] " ans || ans=""
  [[ "$ans" == [yY]* ]] || { echo "aborted (delete it or record under a different name)."; exit 1; }
  rm -rf "$DEST"
fi

# ROS setup scripts read unset variables; keep set -u out of their way
source ~/.rosrc >/dev/null 2>&1
source ~/workspaces/helhest_ws/install/setup.bash >/dev/null 2>&1
mkdir -p ~/bags

# QoS override so the 400 Hz /odin1/imu is not silently dropped on the recorder side.
QOS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../config" && pwd)/rosbag2_qos.yaml"

echo "recording mission -> ~/bags/$NAME   (Ctrl-C to stop)"
exec ros2 bag record -o "$DEST" -s mcap --storage-preset-profile zstd_fast \
  --qos-profile-overrides-path "$QOS" --topics "${TOPICS[@]}"
