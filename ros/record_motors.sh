#!/bin/bash
# Record a MOTOR IDENTIFICATION bag: the drivetrain only, no perception.
#
# Usage:  ./record_motors.sh steps_air      (wheels off the ground)
#         ./record_motors.sh steps_ground   (same program, on the ground)
#         ./record_motors.sh <any-name>     (anything else you want to fit)
#
# Separate from record_odin.sh on purpose. That script records the elevation pipeline's INPUTS,
# which means /odin1/cloud_raw at ~13 MB/s -- fast_experiment0 came to 8.5 GB. None of it is read
# by the motor fit, which wants the command, what the wheels did, and the gyro. This records those
# and nothing else, so a 98 s run is tens of MB.
#
# Drive the manoeuvre with:  python3 ros/calibrate_drive.py steps --go
# Fit it with:               python scripts/fit_motor_steps.py ~/bags/steps_air ~/bags/steps_ground
set -e

TOPICS=(
  /cmd_joints        # what elevation / calibrate_drive asked the wheels to do
  /joint_setpoints   # what the LLC is acting on -- splits transport from the control loop
  /joint_states      # measured position, velocity AND effort (the effort channel is the point:
                     # air-vs-ground at matched speed prices the terrain load in motor units)
  /odin1/imu         # 400 Hz gyro -- yaw on +z, so a turn bag is fittable from the same file
)

usage() {
  cat <<'EOF'
usage: record_motors.sh <name>     (start this, run the manoeuvre, then Ctrl-C)

  steps_air     WHEELS OFF THE GROUND, robot chocked or strapped down -- the rim reaches
                1.4 m/s at 4 rad/s. No breakaway, no load, no slip, so the response IS the
                motor plus wheel inertia. This is the only clean look at the actuator.
  steps_ground  The SAME program on the ground. Needs ~4.2 m of run-out; the steps alternate
                forward/reverse so it nets to zero displacement.

The PAIR is the measurement -- the difference between them is load, breakaway and slip, which
is what separates the motor model from the traction model. Neither run alone settles anything.

BEFORE EITHER: plan_actuate must be OFF, or the planner and calibrate_drive both publish on
/cmd_joints and the manoeuvre is not what you drove:
    ros2 param set /elevation plan_actuate false
    ros2 topic hz /joint_states      # must be live, or nothing is fittable
EOF
}

if [[ -z "$1" || "$1" == "-h" || "$1" == "--help" || "$1" == "list" ]]; then
  usage
  exit 0
fi

NAME="$1"
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

case "$NAME" in
  steps_air)    echo "AIR run -- is the robot chocked or strapped down?" ;;
  steps_ground) echo "GROUND run -- ~4.2 m of run-out needed" ;;
esac
echo "recording -> ~/bags/$NAME   drivetrain only, no lidar   (Ctrl-C to stop)"
exec ros2 bag record -o "$DEST" --qos-profile-overrides-path "$QOS" "${TOPICS[@]}"
