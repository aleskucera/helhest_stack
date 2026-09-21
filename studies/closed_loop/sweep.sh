#!/usr/bin/env bash
# Drive every stress world closed-loop and dump one npz each, for `sweep_figure.py`.
#
# Run it INSIDE the container, after sourcing env.sh -- see this directory's README:
#
#   cd /local/kuceral4/projects/ostrich-odinsim
#   HELHEST_MOUNT=/local /local/kuceral4/projects/helhest-singularity/exec.sh bash -c \
#     'source <this dir>/env.sh && bash <this dir>/sweep.sh'
#
# Two runs at a time, one per GPU. The worlds are independent, so the only thing shared is the
# box; dasenka has two 3090s and a single run does not fill one.
#
#   WORLDS=  which worlds            OUT=      where the npz go
#   FRAMES=  cap per run             EXTRA=    passed through to drive_sim (e.g. --coarsen 0)
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
WORLDS=${WORLDS:-"gap slalom pillars pocket ridge bumpy"}
OUT=${OUT:-/local/kuceral4/tmp/sweep}
FRAMES=${FRAMES:-600}
EXTRA=${EXTRA:-}
GPUS=${GPUS:-2}

mkdir -p "$OUT"
i=0
for w in $WORLDS; do
    gpu=$((i % GPUS))
    i=$((i + 1))
    (
        CUDA_VISIBLE_DEVICES=$gpu python3 "$HERE/drive_sim.py" \
            --world "$w" --frames "$FRAMES" --out "$OUT/$w.npz" --report 100 $EXTRA \
            >"$OUT/$w.log" 2>&1
        printf '%-8s exit %s  %s\n' "$w" "$?" "$(grep -E 'REACHED|did not reach' "$OUT/$w.log" | tail -1)"
    ) &
    while [ "$(jobs -r | wc -l)" -ge "$GPUS" ]; do wait -n; done
done
wait
echo "--- sweep done, npz in $OUT ---"
