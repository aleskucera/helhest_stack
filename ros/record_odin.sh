#!/bin/bash
# Record the Odin1 elevation pipeline's INPUT topics for offline replay / param tuning.
# Mirrors record_bag.sh but for the Odin stack (odin_driver + odin_elevation launches):
# the Odin SLAM pose + raw dTOF cloud replace /odom_2d + /ouster/points.
#
# Usage:  ./record_odin.sh <scenario>     (do the maneuver, then Ctrl-C to stop)
#         ./record_odin.sh                (list the standard scenarios)
#
# Bags land in ~/bags/<scenario>. cloud_raw is ~13 MB/s -- add
#   --compression-mode file --compression-format zstd
# to the ros2 command below for smaller (slower) bags on long runs.
set -e

# Topics: the inputs elevation_node consumes, plus frames and the planning I/O. Recording
# INPUTS (not /elevation_* outputs) lets a live node regenerate the map/plan on replay.
TOPICS=(
  # --- sensor inputs (what elevation consumes) ---
  /odin1/cloud_raw          # raw dTOF cloud (192x256, ~14.5 Hz) -- the lidar input
  /odin1/odometry           # Odin SLAM pose, twist-fixed (~15 Hz) -- replaces /odom_2d + ICP/EKF
  /odin1/imu                # 400 Hz IMU -- unused now (icp+deskew off) but capture for future
  # --- frames ---
  /tf                       # dynamic map->odom_odin->odin1_base_link (from elevation, if running)
  /tf_static                # odin1_base_link->imu_link + any static mounts
  # --- planning I/O (plan_actuate on: a goal drives) ---
  /goal_pose                # planning goal (the input)
  /cmd_joints               # wheel command elevation sends to the LLC (the drive output)
)

# Standard scenarios: name -> maneuver to perform while recording.
declare -A SCENARIOS=(
  [static]="hold still -- baseline: floor plane, self-filter, specular-reflection check"
  [spin]="in-place spin (~1 rev) -- rotation stress: pose/map consistency, self-filter rings"
  [translate]="slow straight drive (~10 m) forward/back -- accumulator + odom drift"
  [drive_goal]="set a /goal_pose and let it drive to it -- planning + actuation capture (clear space!)"
  [dynamic]="people/objects moving through a static scene -- dynamic visibility-carve tuning"
)
ORDER=(static spin translate drive_goal dynamic)

list_scenarios() {
  echo "scenarios:"
  for k in "${ORDER[@]}"; do printf "  %-12s %s\n" "$k" "${SCENARIOS[$k]}"; done
}

if [[ -z "$1" || "$1" == "-h" || "$1" == "--help" || "$1" == "list" ]]; then
  echo "usage: record_odin.sh <scenario>   (do the maneuver, then Ctrl-C to stop)"
  list_scenarios
  exit 0
fi

NAME="$1"
if [[ -n "${SCENARIOS[$NAME]:-}" ]]; then
  echo "scenario '$NAME': ${SCENARIOS[$NAME]}"
else
  echo "note: '$NAME' is not a standard scenario (recording anyway)." >&2
  list_scenarios >&2
fi

# ros2 bag record refuses to write into an existing dir; overwrite on confirm so a re-run
# reuses the canonical ~/bags/<name> (replay tooling finds it by name).
DEST="$HOME/bags/$NAME"
if [[ -e "$DEST" ]]; then
  read -r -p "~/bags/$NAME already exists -- overwrite? [y/N] " ans || ans=""
  [[ "$ans" == [yY]* ]] || { echo "aborted (delete it or record under a different name)."; exit 1; }
  rm -rf "$DEST"
fi

source ~/.rosrc >/dev/null 2>&1
source ~/workspaces/helhest_ws/install/setup.bash >/dev/null 2>&1
mkdir -p ~/bags

# QoS override so the 400 Hz /odin1/imu is not silently dropped on the recorder side.
QOS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/odin/rosbag2_qos.yaml"

echo "recording -> ~/bags/$NAME   (Ctrl-C to stop)"
exec ros2 bag record -o "$DEST" --qos-profile-overrides-path "$QOS" "${TOPICS[@]}"
