#!/usr/bin/env bash
# Headless A/B replay of an Odin bag through elevation_node, for parameter experiments.
#
# Runs the REAL node (not a reimplementation) with `debug_frames` on, so each frame logs
# `carved=<n>/<map>` -- the map-erosion trace. One run per config; logs land in OUT_DIR
# and ros/odin/replay_report.py turns them into a comparison table.
#
# Usage:
#   ./replay_ab.sh <bag_dir> <label> [-p param:=value ...]
#
#   ./replay_ab.sh ../../bags/out_odin0 default
#   ./replay_ab.sh ../../bags/out_odin0 matched -p dynamic_az_bins:=692 -p dynamic_el_bins:=390
#
# Env: RATE (bag play rate, default 1.0), OUT_DIR (default /tmp/odin_replay),
#      EXEC_SH (apptainer wrapper), DOMAIN (ROS_DOMAIN_ID, default 91 -- isolate from any
#      live stack, see ros/README.md).
set -euo pipefail

if [[ $# -lt 2 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "usage: replay_ab.sh <bag_dir> <label> [-p param:=value ...]" >&2
    exit 2
fi

BAG="$(cd "$1" && pwd)"
LABEL="$2"
shift 2
OVERRIDES=("$@")

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PARAMS="$REPO/ros/odin/odin_elevation.params.yaml"
RATE="${RATE:-1.0}"
OUT_DIR="${OUT_DIR:-/tmp/odin_replay}"
DOMAIN="${DOMAIN:-91}"
EXEC_SH="${EXEC_SH:-$HOME/projects/helhest-singularity/exec.sh}"
LOG="$OUT_DIR/$LABEL.log"
mkdir -p "$OUT_DIR"

# plan_actuate off: a replay must never publish drive commands. debug_frames on: the
# per-frame carve counters are the measurement.
FIXED=(-p plan_actuate:=false -p debug_frames:=true)

echo "replay: bag=$(basename "$BAG") label=$LABEL rate=$RATE overrides=${OVERRIDES[*]:-none}"

"$EXEC_SH" bash -c "
set -e
source '$REPO/ros/dev-shell.sh' >/dev/null 2>&1
# Double quotes: single quotes would keep \$PYTHONPATH literal and wipe ROS's own path.
export PYTHONPATH=\"$REPO/ros/helhest_stack_ros:\$PYTHONPATH\"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID=$DOMAIN
# Large-SHM profile: without it the dTOF clouds drop on UDP (see ros/README.md).
export FASTRTPS_DEFAULT_PROFILES_FILE=\"$REPO/ros/fastdds_shm.xml\"
export FASTDDS_DEFAULT_PROFILES_FILE=\"\$FASTRTPS_DEFAULT_PROFILES_FILE\"

python3 -m helhest_stack_ros.elevation_node --ros-args \
    --params-file '$PARAMS' ${FIXED[*]} ${OVERRIDES[*]:-} > '$LOG' 2>&1 &
NODE=\$!
# The planner builds CUDA graphs on startup; feeding the bag before that is up loses frames.
until grep -q 'ElevationNode:' '$LOG' 2>/dev/null; do
    sleep 2
    kill -0 \$NODE 2>/dev/null || { echo 'node died on startup:' >&2; tail -20 '$LOG' >&2; exit 1; }
done
sleep 5

ros2 bag play '$BAG' --rate $RATE \
    --topics /odin1/cloud_raw /odin1/odometry /odin1/imu /tf_static /goal_pose >/dev/null 2>&1
sleep 3
kill \$NODE 2>/dev/null || true
wait \$NODE 2>/dev/null || true
" 2>&1 | grep -v -e gocryptfs -e 'Terminating squashfuse' -e 'Timeouts can be caused'

echo "  -> $LOG  ($(grep -c 'carved=' "$LOG" || true) carve frames logged)"
