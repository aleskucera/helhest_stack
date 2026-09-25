#!/usr/bin/env bash
# Replay one bag through elevation_node headlessly and keep the planner's frame history.
#
#   studies/bag_replay/run_bag.sh <bag dir> <out.npz> [rate] [extra -p args...]
#
# Runs inside the helhest Apptainer (ros/README.md): dev-shell env, the large-SHM Fast DDS
# profile on both participants, an isolated ROS_DOMAIN_ID. The node is the DEPLOYED Odin
# configuration (ros/odin/odin_elevation.params.yaml), actuation INCLUDED: the command chain
# (turn-first brake, conditioner) only runs when the node actuates, and /cmd_joints on an
# isolated domain drives nothing. The bag's recorded /goal_pose messages set the goals. On the bag's end the node gets SIGINT, which is when the
# recording (plan_debug_record) is written.
set -u  # the inner shell cannot: ROS setup.bash reads unset variables
BAG=$(realpath "$1"); OUT=$(realpath -m "$2"); RATE=${3:-0.5}; shift 3 2>/dev/null || shift $#
REPO=$(cd "$(dirname "$0")/../.." && pwd)
EXEC_SH=${EXEC_SH:-$HOME/projects/helhest-singularity/exec.sh}
LOG="${OUT%.npz}.log"
"$EXEC_SH" bash -c '
REPO="$1"; BAG="$2"; OUT="$3"; RATE="$4"; LOG="$5"; shift 5
source "$REPO/ros/dev-shell.sh"
export PYTHONPATH="$REPO/ros/helhest_stack_ros:$PYTHONPATH"
# editable installs (elevation_belief, terrain_value_field) live in .pth files, which python
# ignores on PYTHONPATH entries -- add their source dirs by hand
for f in "$REPO"/.venv/lib/python*/site-packages/_editable_impl_*.pth; do
  export PYTHONPATH="$(cat "$f"):$PYTHONPATH"
done
export FASTRTPS_DEFAULT_PROFILES_FILE="$REPO/ros/fastdds_shm.xml"
export FASTDDS_DEFAULT_PROFILES_FILE="$REPO/ros/fastdds_shm.xml"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-71}
python3 -m helhest_stack_ros.elevation_node --ros-args \
  --params-file "$REPO/ros/odin/odin_elevation.params.yaml" \
  -p plan_debug_record:="$OUT" "$@" > "$LOG" 2>&1 &
NODE=$!
sleep 30  # Warp JIT + graph capture before the first cloud arrives
ros2 bag play "$BAG" --rate "$RATE" \
  --topics /odin1/cloud_raw /odin1/odometry /odin1/imu /tf_static /goal_pose > /dev/null 2>&1
sleep 5
kill -INT "$NODE"; wait "$NODE"
' _ "$REPO" "$BAG" "$OUT" "$RATE" "$LOG" "$@"
ls -la "$OUT" 2>/dev/null || { echo "NO RECORDING -- tail of $LOG:"; tail -20 "$LOG"; }
