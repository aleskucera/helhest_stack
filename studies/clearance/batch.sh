#!/usr/bin/env bash
# Run a job list through run.py, 8 at a time, alternating GPUs.
#
#   studies/clearance/batch.sh jobs.txt out_dir
#
# jobs.txt: one run per line, "<tag> <world> [--set KEY=VALUE ...]". Each run writes
# out_dir/<tag>.npz and .log (history every 4th frame). Set PY to the interpreter and, on a
# machine that needs it (dasenka: inside the container), source the environment first.
# Afterwards: python studies/clearance/analyze.py out_dir <arm> [<arm> ...]
set -u
JOBS=$(realpath "$1"); OUT=$(realpath -m "$2"); mkdir -p "$OUT"
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python3}
NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}; [ "$NGPU" -ge 1 ] || NGPU=1
k=0
while read -r tag world args; do
  [ -z "$tag" ] && continue
  ( CUDA_VISIBLE_DEVICES=$((k % NGPU)) $PY "$HERE/run.py" --world "$world" --frames 1200 \
      --report 50 --history 4 $args --out "$OUT/$tag.npz" > "$OUT/$tag.log" 2>&1 ) &
  k=$((k + 1)); while [ "$(jobs -rp | wc -l)" -ge 8 ]; do sleep 2; done
done < "$JOBS"
wait
echo "done: $k runs in $OUT"
