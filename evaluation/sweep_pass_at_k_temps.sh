#!/bin/bash
# Temperature sweep for Pass@k exploration curves.
# Calls an existing single-job driver once per temperature.
# At T=0.0 uses TURNS=1 (greedy); otherwise uses full TURNS (default 1..8).
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/run_7B_math_4qa_hf_single_job.sh}"
TEMPERATURES="${TEMPERATURES:-0.0 0.2 0.4 0.6 0.8 1.0}"
TURNS="${TURNS:-1 2 3 4 5 6 7 8}"

if [[ ! -f "$JOB_SCRIPT" ]]; then
  echo "JOB_SCRIPT not found: $JOB_SCRIPT"
  exit 1
fi

echo "Sweep job: $JOB_SCRIPT"
echo "Temperatures: $TEMPERATURES"
echo "TURNS (T>0): $TURNS"

for T in $TEMPERATURES; do
  if [[ "$T" == "0" || "$T" == "0.0" ]]; then
    TURNS_THIS="1"
  else
    TURNS_THIS="$TURNS"
  fi
  echo "========================================"
  echo "Temperature=$T  TURNS=$TURNS_THIS"
  echo "========================================"
  TEMPERATURE="$T" TURNS="$TURNS_THIS" bash "$JOB_SCRIPT"
done

echo "Sweep complete."
