#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
export PYTHONPATH="$(pwd):$PYTHONPATH"
mkdir -p logs

# Root output directory created by inference script.
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hf_math_4qa}"

# Set true to add LLM-as-judge scoring; keep false for pure math/F1 scoring.
USE_LLM="${USE_LLM:-false}"

# Judge endpoint/model only used when USE_LLM=true.
API_BASE_URL="${API_BASE_URL:-http://localhost:8001/v1}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-72B-Instruct}"

# Async evaluator limits.
CONCURRENT_LIMIT="${CONCURRENT_LIMIT:-50}"
TIMEOUT="${TIMEOUT:-3600}"

declare -A TASK_MAP=(
  ["aime24"]="math"
  ["aime25"]="math"
  ["math500"]="math"
  ["gsm8k"]="math"
  ["math"]="math"
  ["hotpotqa"]="qa"
  ["2wiki"]="qa"
  ["musique"]="qa"
  ["bamboogle"]="qa"
)

shopt -s nullglob
FILES=( "$OUTPUT_DIR"/*/*_output_*.json )
shopt -u nullglob

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No output files found under $OUTPUT_DIR"
  exit 1
fi

for file_path in "${FILES[@]}"; do
  filename="$(basename "$file_path")"
  dataset_name="${filename%%_output_*}"
  task="${TASK_MAP[$dataset_name]:-}"

  if [[ -z "$task" ]]; then
    echo "Skipping unsupported dataset output: $dataset_name"
    continue
  fi

  CMD=(python evaluate.py)
  CMD+=(--output_path "$file_path")
  CMD+=(--task "$task")
  CMD+=(--concurrent_limit "$CONCURRENT_LIMIT")
  CMD+=(--timeout "$TIMEOUT")

  if [[ "$USE_LLM" == "true" ]]; then
    CMD+=(--use_llm)
    CMD+=(--api_base_url "$API_BASE_URL")
    CMD+=(--model_name "$MODEL_NAME")
  fi

  echo "Evaluating $dataset_name with task=$task"
  "${CMD[@]}" | tee -a logs/evaluate_math_4qa_hf.log
  echo "Finished $dataset_name"
  echo "-----------------------------------"
done
