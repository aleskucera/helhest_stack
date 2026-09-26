#!/usr/bin/env bash
# Step 4's A/B on dasenka: the accumulator map and the belief map replayed SIMULTANEOUSLY, one per
# GPU, each on its own ROS domain -- so both see the same machine load -- and the GPUs swapped
# between rounds so neither arm owns the faster card.
#
#   studies/belief_mapping/ab_dasenka.sh <out dir> [rounds] [bag ...]
#
# Runs on the dasenka host (not in the container: run_bag.sh enters it). Afterwards, locally:
#   python studies/bag_replay/audit.py <out dir>/*.npz
set -u
OUT=$(realpath -m "$1"); ROUNDS=${2:-2}; shift 2 2>/dev/null || shift $#
BAGS=${*:-in_speed_odin0 out_odin0}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
export ENV_SH=${ENV_SH:-/local/kuceral4/tmp/env.sh}
export EXEC_SH=${EXEC_SH:-/local/kuceral4/projects/helhest-singularity/exec.sh}
export HELHEST_MOUNT=${HELHEST_MOUNT:-/local/kuceral4}
BAGDIR=${BAGDIR:-/local/kuceral4/projects/hs_bags}
mkdir -p "$OUT"
for r in $(seq 1 "$ROUNDS"); do
  for b in $BAGS; do
    ga=$(( r % 2 )); gb=$(( (r + 1) % 2 ))
    ( CUDA_VISIBLE_DEVICES=$ga ROS_DOMAIN_ID=71 "$REPO/studies/bag_replay/run_bag.sh" \
        "$BAGDIR/$b" "$OUT/${b}_acc_r$r.npz" 0.5 -p profile_stages:=true \
        -p map_source:=accumulator ) &
    ( CUDA_VISIBLE_DEVICES=$gb ROS_DOMAIN_ID=72 "$REPO/studies/bag_replay/run_bag.sh" \
        "$BAGDIR/$b" "$OUT/${b}_belief_r$r.npz" 0.5 -p profile_stages:=true \
        -p map_source:=belief ) &
    wait
    echo "round $r $b done"
  done
done
