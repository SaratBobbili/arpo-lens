#!/bin/bash
# ARPO system-prompt sweep: calls run_7B_math_4qa_hf_single_job.sh per prompt.
# Does not modify the job script on disk. Builds a one-shot temp copy where
# model / prompt / tool budgets read from the environment (job hardcodes would
# otherwise clobber env overrides).
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/run_7B_math_4qa_hf_single_job.sh}"
PROMPT_TYPES="${PROMPT_TYPES:-base math search code_search gemini react}"

# Hub ARPO defaults (overridable by caller env).
ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-dongguanting/Qwen2.5-7B-ARPO}"
REASON_MODEL_PATH="${REASON_MODEL_PATH:-${ACTOR_MODEL_PATH}}"
REASON_MODEL_NAME="${REASON_MODEL_NAME:-Qwen2.5-7B-ARPO}"
RAW_ACTOR_CHECKPOINT_PATH="${RAW_ACTOR_CHECKPOINT_PATH:-}"
TRAINING_RECIPE_PATH="${TRAINING_RECIPE_PATH:-}"

if [[ ! -f "$JOB_SCRIPT" ]]; then
  echo "JOB_SCRIPT not found: $JOB_SCRIPT" >&2
  exit 1
fi

budget_for_prompt() {
  case "$1" in
    base)        echo "0 0" ;;
    math)        echo "3 0" ;;
    search)      echo "0 3" ;;
    code_search) echo "3 3" ;;
    gemini)      echo "3 3" ;;
    claude)      echo "3 3" ;;
    react)       echo "0 10" ;;
    *)           echo "3 3" ;;
  esac
}

# Line-safe rewrite of knobs only (never writes back to JOB_SCRIPT).
materialize_job() {
  local out="$1"
  python3 - "$JOB_SCRIPT" "$out" <<'PY'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()
# Config-block single-line assignments only (start of line).
repl = {
    r'^USE_HF_HUB_MODEL=.*$': 'USE_HF_HUB_MODEL="${USE_HF_HUB_MODEL:-false}"',
    r'^ACTOR_MODEL_PATH=.*$': 'ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH}"',
    r'^REASON_MODEL_PATH=.*$': 'REASON_MODEL_PATH="${REASON_MODEL_PATH}"',
    r'^RAW_ACTOR_CHECKPOINT_PATH=.*$': 'RAW_ACTOR_CHECKPOINT_PATH="${RAW_ACTOR_CHECKPOINT_PATH}"',
    r'^REASON_MODEL_NAME=.*$': 'REASON_MODEL_NAME="${REASON_MODEL_NAME}"',
    r'^TRAINING_RECIPE_PATH=.*$': 'TRAINING_RECIPE_PATH="${TRAINING_RECIPE_PATH}"',
    r'^PROMPT_TYPE=.*$': 'PROMPT_TYPE="${PROMPT_TYPE}"',
    r'^MAX_PYTHON_TIMES=.*$': 'MAX_PYTHON_TIMES="${MAX_PYTHON_TIMES}"',
    r'^MAX_SEARCH_TIMES=.*$': 'MAX_SEARCH_TIMES="${MAX_SEARCH_TIMES}"',
}
for pat, rep in repl.items():
    text, n = re.subn(pat, rep, text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit(f"expected 1 match for {pat!r}, got {n}")
# Disable hard coded Hub-Instruct rewrite block.
text, n = re.subn(
    r'if \[\[ "\$USE_HF_HUB_MODEL" == "true" \]\]; then',
    'if false; then',
    text,
    count=1,
)
if n != 1:
    raise SystemExit(f"expected USE_HF_HUB_MODEL if-block, got {n}")
# Convert VERL actor only for local dirs missing config.json (not HF repo ids).
text, n = re.subn(
    r'if \[\[ "\$USE_HF_HUB_MODEL" != "true" && ! -f "\$\{ACTOR_MODEL_PATH\}/config\.json" \]\]; then',
    'if [[ -d "$ACTOR_MODEL_PATH" && ! -f "${ACTOR_MODEL_PATH}/config.json" ]]; then',
    text,
    count=1,
)
if n != 1:
    raise SystemExit(f"expected convert if-block, got {n}")
open(dst, "w", encoding="utf-8").write(text)
PY
  chmod +x "$out"
}

TMP_JOB="$(mktemp "${TMPDIR:-/tmp}/arpo_prompt_job.XXXXXX.sh")"
cleanup() { rm -f "$TMP_JOB"; }
trap cleanup EXIT

materialize_job "$TMP_JOB"

echo "Base job (unmodified): $JOB_SCRIPT"
echo "Materialized job:      $TMP_JOB"
echo "Model: ${ACTOR_MODEL_PATH}  name=${REASON_MODEL_NAME}"
echo "Prompt types: $PROMPT_TYPES"

for P in $PROMPT_TYPES; do
  read -r MAX_PYTHON_TIMES MAX_SEARCH_TIMES < <(budget_for_prompt "$P")
  echo "========================================"
  echo "PROMPT_TYPE=$P  MAX_PYTHON_TIMES=$MAX_PYTHON_TIMES  MAX_SEARCH_TIMES=$MAX_SEARCH_TIMES"
  echo "========================================"
  USE_HF_HUB_MODEL="${USE_HF_HUB_MODEL:-false}" \
  ACTOR_MODEL_PATH="$ACTOR_MODEL_PATH" \
  REASON_MODEL_PATH="$REASON_MODEL_PATH" \
  RAW_ACTOR_CHECKPOINT_PATH="$RAW_ACTOR_CHECKPOINT_PATH" \
  REASON_MODEL_NAME="$REASON_MODEL_NAME" \
  TRAINING_RECIPE_PATH="$TRAINING_RECIPE_PATH" \
  PROMPT_TYPE="$P" \
  MAX_PYTHON_TIMES="$MAX_PYTHON_TIMES" \
  MAX_SEARCH_TIMES="$MAX_SEARCH_TIMES" \
  bash "$TMP_JOB"
done

echo "Sweep complete."
