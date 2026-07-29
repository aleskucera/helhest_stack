#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="robot@192.168.18.5"
REMOTE_BAGS_DIR="/home/robot/bags"
LOCAL_BAGS_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <bag_name>"
    exit 1
fi

BAG_NAME="$1"
scp -r "${REMOTE_HOST}:${REMOTE_BAGS_DIR}/${BAG_NAME}" "${LOCAL_BAGS_DIR}/"
