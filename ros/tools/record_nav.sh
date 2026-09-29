#!/bin/bash
# Record navigation_node's INPUT topics for offline replay / param tuning, on the helhest-nav
# stack (cras_odin_driver + navigation_node, ros/sessions/helhest-nav.tmuxinator.yml).
#
# Usage:  ./record_nav.sh <scenario>     (do the maneuver, then Ctrl-C to stop)
#         ./record_nav.sh                (list the standard scenarios)
#
# Bags land in ~/bags/<scenario> as MCAP with zstd chunk compression. The cloud is ~13 MB/s before
# compression and everything else together is well under 1 MB/s.
set -e

# Topics: the inputs navigation_node consumes, plus frames and the planning I/O. Recording
# INPUTS (not /elevation_* outputs) lets a live node regenerate the map/plan on replay -- with
# today's code, not the code that ran.
TOPICS=(
  # --- sensor inputs (what the node consumes) ---
  /odin1/cloud_filtered     # dTOF cloud (192x256, ~14.5 Hz) with the robot's body masked by
                            # cras_odin_driver -- the node's lidar input. /odin1/cloud_raw is the
                            # same cloud unmasked; record it only to look at the mask itself.
  /odin1/odometry           # Odin SLAM pose, twist in the body frame (~14.5 Hz)
  /odin1/imu                # 400 Hz gyro -- the yaw-rate loop and the friction estimate read it
  /odom_2d                  # wheel odometry: a second pose to hold the Odin's against (the cras
                            # watchdog resets the Odin when the two disagree)
  # --- frames ---
  /tf                       # odin1_base_link->odom_odin (driver) + the wheel joints
  /tf_static                # base_link->odin1_base_link, and the Odin calibration down to odin1_lidar
  # --- planning I/O (plan_actuate on: a goal drives) ---
  /goal_pose                # planning goal (the input)
  /planned_path             # what MPPI committed to, each frame
  /cmd_joints               # wheel command the node sends to the LLC (the drive output)
  /cmd_vel                  # the other LLC input: (v, yaw rate), used when cmd_output:=twist
  # --- the LLC's own yaw loop on /cmd_vel (its internals, as it publishes them) ---
  /debug/angle
  /debug/setpoint
  /debug/error
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
  /estop_active             # when a human stopped the robot -- so a stop is not read as the planner's
  # --- camera (compressed only; raw / undistorted / intensity_gray are heavy and unused here) ---
  /odin1/image/compressed   # JPEG camera stream -- light, handy for reviewing a follow-me run
)

# Tracking namespaces recorded via --regex (below), so EVERY radio/uwb/bluetooth topic -- and any
# new anchor that appears -- is captured without listing each. Covers the follow-me target
# (/radio/estimate_pose, in the 'locator' frame; with /tf it reconstructs the chased point in map),
# the UWB two-way-ranging estimates + per-anchor distances, and the bluetooth AoA stack.
TOPIC_REGEX="^/(radio|uwb|bluetooth)/"

# Standard scenarios: name -> maneuver to perform while recording.
declare -A SCENARIOS=(
  [static]="hold still -- baseline: floor plane, body mask, specular-reflection check"
  [spin]="in-place spin (~1 rev) -- rotation stress: pose/map consistency, the spin speed cap"
  [drive_goal]="set a /goal_pose and let it drive to it -- planning + actuation capture (clear space!)"
  [turns]="on ONE surface, goals that need turning: 90 deg left and right, a goal straight behind
(a spin), and a slalom through 3-4 goals, at the default speed and again with plan_wmax lowered.
The controlled measurement of how much of each commanded turn the robot realises, with and
without the yaw-rate loop (ros2 param set /navigation plan_yaw_track false for a second run)."
  [tilt_cal]="drive over the SAME flat patch (a car park, not grass) four times, heading N, E, S and
W, ~10 m each, slowly. A mount pitch/roll error makes the patch read higher from close than from
far, differently per heading; four headings over one patch is what separates the mount's roll and
pitch from the SLAM's attitude drift."
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
it with ros/tools/calibrate_drive.py compact --go."
  [slope]="drive a slope of 10 deg or more: straight up, straight down, and ACROSS it in both
directions, 4-5 s each, plus a turn while on the cross-slope. Nothing in the archive exceeds 5.4
deg of tilt, so the load-transfer fix (normal_loads, validated only against Chrono) has never been
seen on real data. The across-slope runs are the ones that matter -- that is where the old model
predicted zero lateral transfer and Chrono predicts 0.4 m g."
)
ORDER=(static spin drive_goal turns tilt_cal dynamic calibrate relax compact slope)

list_scenarios() {
  echo "scenarios:"
  for k in "${ORDER[@]}"; do printf "  %-12s %s\n" "$k" "${SCENARIOS[$k]}"; done
}

if [[ -z "$1" || "$1" == "-h" || "$1" == "--help" || "$1" == "list" ]]; then
  echo "usage: record_nav.sh <scenario>   (do the maneuver, then Ctrl-C to stop)"
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
QOS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../config" && pwd)/rosbag2_qos.yaml"

echo "recording -> ~/bags/$NAME   (Ctrl-C to stop)"
# MCAP with zstd chunk compression: ~2-3x smaller, and read directly by ros2 bag play and by our
# Python tools (Cloudini would compress the cloud more, but every analysis would need a decode step).
exec ros2 bag record -o "$DEST" -s mcap --storage-preset-profile zstd_fast \
  --qos-profile-overrides-path "$QOS" --topics "${TOPICS[@]}" --regex "$TOPIC_REGEX"
