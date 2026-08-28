#!/usr/bin/env bash
# Detached entry point for the E2 runner. Everything heavy runs nice -n 15 ionice -c3.
#
#   ./runner.sh                 the whole chain: design phase -> C1-C4 -> FREEZE.json ->
#                               held-out streaming run -> scoring -> FINAL_REPORT.md.
#                               If any gate fails it writes STOPPED_AT.md and exits before
#                               unlock, leaving the held-out set untouched.
#   ./runner.sh --design-only   stop after the freeze record (dry run).
#
# Re-running is safe and resumes: every stage has a done-marker under out/state/.
set -uo pipefail

export BASEPROD_ROOT="${BASEPROD_ROOT:-/local/kuceral4/baseprod}"
STUDIES="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$STUDIES${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# keep BLAS from stampeding a shared box; the work is embarrassingly serial anyway
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

PY="${BASEPROD_PY:-$BASEPROD_ROOT/venv/bin/python}"
mkdir -p "$BASEPROD_ROOT/out"
LOG="$BASEPROD_ROOT/out/runner.stdout.log"

{
  echo "=== runner.sh start $(date -Is) host=$(hostname) args=$* ==="
  echo "    BASEPROD_ROOT=$BASEPROD_ROOT PYTHONPATH=$PYTHONPATH PY=$PY"
} >> "$LOG"

nice -n 15 ionice -c3 "$PY" -m baseprod.runner "$@" >> "$LOG" 2>&1
rc=$?
echo "=== runner.sh exit $rc $(date -Is) ===" >> "$LOG"
exit $rc
