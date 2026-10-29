#!/usr/bin/env bash
# Replay one bag through navigation_node headlessly and keep the planner's frame history.
#
#   studies/bag_replay/run_bag.sh <bag dir> <out.npz> [rate] [extra -p args...]
#
# Runs inside the helhest Apptainer (ros/README.md): dev-shell env, the large-SHM Fast DDS
# profile on both participants, an isolated ROS_DOMAIN_ID. The node is the DEPLOYED Odin
# configuration, actuation INCLUDED: the command chain (turn-first brake, conditioner) only runs
# when the node actuates, and its output on an isolated domain drives nothing. The bag's recorded
# /goal_pose messages set the goals. On the bag's end the node gets SIGINT, which is when the
# recording (plan_debug_record) is written.
#
# Two kinds of bag, told apart by their topics:
#   cras_odin_driver (the robot since 2026-10, /odin1/cloud_filtered): odin.params.yaml +
#     odin_cras.params.yaml, exactly as the helhest-nav session starts the node
#   the vendor driver (older bags, /odin1/cloud_raw): odin.params.yaml alone
#
# The node's inputs are cut into <bag>_lite once and reused: the full bag also carries the camera
# and plays several times too slowly. The replay overrides the recorded QoS (play_qos.yaml next
# to this script): bags recorded over Zenoh mark /odin1/odometry, /joint_states and /tf_static
# avoid_ros_namespace_conventions, which hides them from every subscriber on replay -- the node
# then never pairs a cloud with an odometry message and plans nothing.
#
# Other machines: ENV_SH=<script> sources that environment instead of this checkout's venv, and
# EXEC_SH=<exec.sh> picks the container. The container sees only $HOME and the working directory,
# so a checkout or bag elsewhere needs HELHEST_MOUNT=<dir above both>. dasenka:
#   HELHEST_MOUNT=/local/kuceral4 ENV_SH=<env script> \
#     EXEC_SH=/local/kuceral4/projects/helhest-singularity/exec.sh run_bag.sh ...
#
# The first plan on a cold Warp kernel cache stalls the node ~60 s and frames are dropped: after
# a code change, run the replay twice and keep the second recording.
set -u  # the inner shell cannot: ROS setup.bash reads unset variables
BAG=$(realpath "$1"); OUT=$(realpath -m "$2"); RATE=${3:-0.5}; shift 3 2>/dev/null || shift $#
REPO=$(cd "$(dirname "$0")/../.." && pwd)
EXEC_SH=${EXEC_SH:-$HOME/projects/helhest-singularity/exec.sh}
LOG="${OUT%.npz}.log"
LITE="${BAG%/}_lite"

for p in "$REPO" "$BAG" "$(dirname "$OUT")"; do
  case "$p" in
    "$HOME"/*) ;;
    *) if [ -z "${HELHEST_MOUNT:-}" ]; then
         echo "$p is outside \$HOME: the container cannot see it. Set HELHEST_MOUNT to a directory above it." >&2
         exit 1
       fi ;;
  esac
done

if grep -q "name: /odin1/cloud_filtered" "$BAG/metadata.yaml"; then
  CLOUD=/odin1/cloud_filtered
  PARAMS="$REPO/ros/config/odin.params.yaml $REPO/ros/config/odin_cras.params.yaml"
else
  CLOUD=/odin1/cloud_raw
  PARAMS="$REPO/ros/config/odin.params.yaml"
fi

HELHEST_MOUNT=${HELHEST_MOUNT:-} "$EXEC_SH" bash -c '
REPO="$1"; BAG="$2"; OUT="$3"; RATE="$4"; LOG="$5"; ENV_SH="$6"; LITE="$7"; CLOUD="$8"; PARAMS="$9"
shift 9
if [ -n "$ENV_SH" ]; then
  # a machine without this checkout s venv (dasenka): its own environment script instead
  source "$ENV_SH"
else
  source "$REPO/ros/tools/dev-shell.sh"
  # editable installs (elevation_belief) live in .pth files, which python
  # ignores on PYTHONPATH entries -- add their source dirs by hand
  for f in "$REPO"/.venv/lib/python*/site-packages/_editable_impl_*.pth; do
    export PYTHONPATH="$(cat "$f"):$PYTHONPATH"
  done
fi
export PYTHONPATH="$REPO/ros/helhest_stack_ros:$PYTHONPATH"
# ...and the own src of this checkout ahead of them: the helhest editable install points at whichever
# checkout the venv was made in, so a worktree would otherwise replay the code of the main tree
export PYTHONPATH="$REPO/src:$PYTHONPATH"
export FASTRTPS_DEFAULT_PROFILES_FILE="$REPO/ros/config/fastdds_shm.xml"
export FASTDDS_DEFAULT_PROFILES_FILE="$REPO/ros/config/fastdds_shm.xml"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-71}
if [ ! -d "$LITE" ]; then
  CUT=$(mktemp --suffix=.yaml)
  printf "output_bags:\n- uri: %s\n  storage_id: mcap\n  topics: [%s, /odin1/odometry, /odin1/imu, /tf_static, /goal_pose, /joint_states]\n" \
    "$LITE" "$CLOUD" > "$CUT"
  ros2 bag convert -i "$BAG" -o "$CUT" > /dev/null 2>&1 || { echo "cutting $BAG failed" >&2; exit 1; }
  rm -f "$CUT"
fi
P_ARGS=""
for p in $PARAMS; do P_ARGS="$P_ARGS --params-file $p"; done
python3 -m helhest_stack_ros.navigation_node --ros-args $P_ARGS \
  -p plan_debug_record:="$OUT" "$@" > "$LOG" 2>&1 &
NODE=$!
sleep 30  # Warp JIT + graph capture before the first cloud arrives
ros2 bag play "$LITE" --rate "$RATE" \
  --qos-profile-overrides-path "$REPO/studies/bag_replay/play_qos.yaml" > /dev/null 2>&1
sleep 5
kill -INT "$NODE"; wait "$NODE"
' _ "$REPO" "$BAG" "$OUT" "$RATE" "$LOG" "${ENV_SH:-}" "$LITE" "$CLOUD" "$PARAMS" "$@"
ls -la "$OUT" 2>/dev/null || { echo "NO RECORDING -- tail of $LOG:"; tail -20 "$LOG"; }
