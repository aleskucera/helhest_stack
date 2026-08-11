#!/usr/bin/env bash
# Fetch a rosbag from the robot into this directory (or a given destination).
#
# Usage:  ./fetch_bag.sh <bag_name> [destination_dir]
#         ./fetch_bag.sh -l                 (list the bags on the robot)
#
# Host/path are env-overridable, e.g. REMOTE_HOST=robot@10.0.0.7 ./fetch_bag.sh spin
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-robot@192.168.18.5}"
REMOTE_BAGS_DIR="${REMOTE_BAGS_DIR:-/home/robot/bags}"
LOCAL_BAGS_DIR="$(cd "$(dirname "$0")" && pwd)"
# Fail fast instead of hanging on the TCP connect when the robot is off/on another network.
SSH_OPTS=(-o ConnectTimeout=5)

usage() {
    cat <<EOF
usage: fetch_bag.sh <bag_name> [destination_dir]   (default destination: $LOCAL_BAGS_DIR)
       fetch_bag.sh -l                             (list the bags on the robot)

env: REMOTE_HOST (default $REMOTE_HOST), REMOTE_BAGS_DIR (default $REMOTE_BAGS_DIR)
EOF
}

case "${1:-}" in
    -h | --help)
        usage
        exit 0
        ;;
    -l | --list)
        ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "ls -1 '$REMOTE_BAGS_DIR'" || {
            echo "error: cannot reach $REMOTE_HOST" >&2
            exit 1
        }
        exit 0
        ;;
    "")
        usage >&2
        exit 2
        ;;
esac

BAG_NAME="$1"
DEST_DIR="${2:-$LOCAL_BAGS_DIR}"
TARGET="$DEST_DIR/$BAG_NAME"

# One round trip answers both questions: does the bag exist, and can we resume with rsync.
# `exit 0` so a missing rsync doesn't read as an unreachable host.
probe=$(
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" \
        "[ -d '$REMOTE_BAGS_DIR/$BAG_NAME' ] && echo BAG_OK
         command -v rsync >/dev/null && echo RSYNC_OK
         exit 0"
) || {
    echo "error: cannot reach $REMOTE_HOST" >&2
    exit 1
}

if [[ "$probe" != *BAG_OK* ]]; then
    echo "error: no bag '$BAG_NAME' in $REMOTE_HOST:$REMOTE_BAGS_DIR (try: $0 -l)" >&2
    exit 1
fi

mkdir -p "$DEST_DIR"
if [[ "$probe" == *RSYNC_OK* ]]; then
    # Bags are multi-GB over the robot's wifi: --partial resumes a dropped transfer instead
    # of restarting from zero, and an interrupted re-fetch skips what already landed.
    rsync -a --partial --info=progress2 -e "ssh ${SSH_OPTS[*]}" \
        "$REMOTE_HOST:$REMOTE_BAGS_DIR/$BAG_NAME/" "$TARGET/"
else
    # scp can't resume, and copying onto an existing dir nests it as <target>/<bag_name>
    # (a silently broken bag), so refuse rather than produce one.
    if [[ -e "$TARGET" ]]; then
        echo "error: $TARGET exists and the robot has no rsync, so a re-fetch would nest it." >&2
        echo "       Remove it first, or install rsync on the robot to resume instead." >&2
        exit 1
    fi
    scp -r "${SSH_OPTS[@]}" "$REMOTE_HOST:$REMOTE_BAGS_DIR/$BAG_NAME" "$DEST_DIR/"
fi

# A bag without metadata.yaml won't replay -- catch a truncated transfer here, not in tmuxinator.
if [[ ! -f "$TARGET/metadata.yaml" ]]; then
    echo "error: $TARGET/metadata.yaml missing -- transfer looks incomplete" >&2
    exit 1
fi
echo "ok: $TARGET"
