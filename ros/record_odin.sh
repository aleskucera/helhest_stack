#!/bin/bash
# Record the Odin1 elevation pipeline's INPUT topics for offline replay / param tuning.
# Mirrors record_bag.sh but for the Odin stack (odin_driver.launch.py + elevation_node):
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
  /yaw_track                # inner yaw loop (x=reference, y=gyro, z=correction) -- the
                            # correction is NOT recoverable from /cmd_joints, which
                            # already contains it
  # --- drivetrain response (100 Hz, from the LLC) ---
  # Without these a bag can show WHAT was commanded but not what the wheels did, which blocks
  # every drivetrain question: the turn gain (alpha from measured rather than commanded wheels),
  # the command->response delay, the torque scale from `effort`, and the differential the LLC
  # actually realizes. The Odin bags so far have none of it. Cheap: ~100 Hz of 3 floats.
  /joint_setpoints          # per-wheel target the LLC is acting on -- splits transport from loop
  /joint_states             # measured wheel position/velocity/effort -- the actual response
  # --- camera (compressed only; raw / undistorted / intensity_gray are heavy and unused here) ---
  /odin1/image/compressed   # JPEG camera stream -- light, handy for reviewing a follow-me run
)

# Tracking namespaces recorded via --regex (below), so EVERY radio/uwb/bluetooth topic -- and any
# new anchor that appears -- is captured without listing each. Covers the follow-me target
# (/radio/estimate_pose, in the 'locator' frame; with /tf it reconstructs the chased point in map),
# the UWB two-way-ranging estimates + per-anchor distances, and the bluetooth AoA stack.
TOPIC_REGEX="^/(radio|uwb|bluetooth)/"

# Motor identification needs NONE of the above except the drivetrain. cloud_raw alone is ~13 MB/s
# (fast_experiment0 came to 8.5 GB), and the fit reads only the command, the wheel response and
# the gyro -- so the steps_* scenarios record just those three and land in the tens of MB.
MOTOR_TOPICS=(/cmd_joints /joint_setpoints /joint_states /odin1/imu)

# Standard scenarios: name -> maneuver to perform while recording.
declare -A SCENARIOS=(
  [static]="hold still -- baseline: floor plane, self-filter, specular-reflection check"
  [spin]="in-place spin (~1 rev) -- rotation stress: pose/map consistency, self-filter rings"
  [translate]="slow straight drive (~10 m) forward/back -- accumulator + odom drift"
  [drive_goal]="set a /goal_pose and let it drive to it -- planning + actuation capture (clear space!)"
  [dynamic]="people/objects moving through a static scene -- dynamic visibility-carve tuning"
  [calibrate]="HOLD each command 3-5 s: straight at ~2/4/6 rad/s, then turns (differential ~1/2/4) at each speed, then a few sharp starts from rest -- the only maneuver that gives STEADY-STATE turning (the planner never holds a command longer than ~3 ms) plus clean step responses"
  [relax]="the SAME differential step (e.g. +1.5 rad/s) applied from three different forward
speeds -- ~0.5, ~1.0, ~2.0 rad/s mean -- held 4 s each, 5 repeats per speed, both directions.
Separates a yaw lag keyed to TIME from one keyed to DISTANCE: only the distance form has a
response time that scales as 1/v. Chrono says the distance form (sigma ~0.15 m) fits better; this
is the measurement that confirms or kills it on the real robot."
  [compact]="the relaxation sweep for a SMALL site: spins in place at four wheel speeds instead
of driving arcs. Same measurement, 1.9 x 1.8 m instead of 41 x 29 m, because what sets a tyre's
relaxation is the speed the CONTACT travels over the ground and that is nonzero in a spin. Drive
it with ros/calibrate_drive.py compact --go."
  [steps_air]="WHEELS OFF THE GROUND, robot chocked or strapped down. Drive it with
ros/calibrate_drive.py steps --go (98 s). No breakaway, no load, no slip, so the response IS the
motor plus wheel inertia -- the only clean look at the actuator this robot can give. Records the
drivetrain topics only, no lidar."
  [steps_ground]="THE SAME program on the ground, needs ~4.2 m of run-out (steps alternate
forward/reverse, so it nets to zero displacement). The pair air+ground is the measurement: the
difference between them is load, breakaway and slip, which is what separates the motor model from
the traction model. Fit both with scripts/fit_motor_steps.py."
  [slope]="drive a slope of 10 deg or more: straight up, straight down, and ACROSS it in both
directions, 4-5 s each, plus a turn while on the cross-slope. Nothing in the archive exceeds 5.4
deg of tilt, so the load-transfer fix (normal_loads, validated only against Chrono) has never been
seen on real data. The across-slope runs are the ones that matter -- that is where the old model
predicted zero lateral transfer and Chrono predicts 0.4 m g."
)
ORDER=(static spin translate drive_goal dynamic calibrate relax compact steps_air steps_ground slope)

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

# steps_* are actuator identification: drivetrain topics only, and no --regex (the tracking
# namespaces are irrelevant here and only add size).
if [[ "$NAME" == steps_* ]]; then
  echo "recording -> ~/bags/$NAME   drivetrain only, no lidar   (Ctrl-C to stop)"
  exec ros2 bag record -o "$DEST" --qos-profile-overrides-path "$QOS" "${MOTOR_TOPICS[@]}"
fi

echo "recording -> ~/bags/$NAME   (Ctrl-C to stop)"
exec ros2 bag record -o "$DEST" --qos-profile-overrides-path "$QOS" "${TOPICS[@]}" --regex "$TOPIC_REGEX"
