#!/usr/bin/env bash
# Launch rviz2 with the navigation config, no matter the current directory.
# Usage:  ros/tools/rviz.sh                 # opens the repo's nav.rviz (fixed frame odom_odin)
#         ros/tools/rviz.sh --extra args    # any extra args pass straight through to rviz2
set -eo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$REPO/ros/helhest_stack_ros/rviz/nav.rviz"

exec rviz2 -d "$CONFIG" "$@"
