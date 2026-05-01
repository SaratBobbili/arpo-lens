#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
mkdir -p logs

# Root output directory created by inference script.
OUTPUT_DIR="${OUTPUT_DIR:-outputs/hf_math_4qa}"

# LLM-as-judge is the default author-style metric path; set false only for parser-only debugging.
USE_LLM="${USE_LLM:-true}"

# Judge endpoint/model only used when USE_LLM=true.
API_BASE_URL="${API_BASE_URL:-http://localhost:8001/v1}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-72B-Instruct}"

# Async evaluator limits.
CONCURRENT_LIMIT="${CONCURRENT_LIMIT:-50}"
TIMEOUT="${TIMEOUT:-3600}"

# Inference-time prompt schema; forwarded so evaluate.py can run the trainer's
# format validator on ECHO outputs. Default empty => no format-pass-rate metric.
PROMPT_TYPE="${PROMPT_TYPE:-}"
# ECHO validator profile (c1..c5); only consulted when PROMPT_TYPE=echo.
VALIDATOR_PROFILE="${VALIDATOR_PROFILE:-c1}"

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
  # Mixed math+qa val set (100+80). Independent task so it can be tested in
  # isolation; uses evaluate_grpo_mix_prediction whose normalize_answer mirrors
  # the training scorer (deep_research_echo.compute_score / get_f1_score) by
  # stripping articles + punctuation. Fallback accuracy is token F1; headline
  # metric is still llm_equal when USE_LLM=true.
  ["grpo_mix"]="grpo_mix"
)

shopt -s nullglob
FILES=( "$OUTPUT_DIR"/*/*_output_*.json )
shopt -u nullglob

RAW_FILES=()
for file_path in "${FILES[@]}"; do
  filename="$(basename "$file_path")"
  if [[ "$filename" =~ _output_[0-9]+\.json$ ]]; then
    RAW_FILES+=( "$file_path" )
  fi
done

if [[ ${#RAW_FILES[@]} -eq 0 ]]; then
  echo "No output files found under $OUTPUT_DIR"
  exit 1
fi

for file_path in "${RAW_FILES[@]}"; do
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

  # Pass prompt_type only when explicitly set; evaluate.py treats absence as
  # "non-ECHO eval" and skips the format validator.
  if [[ -n "$PROMPT_TYPE" ]]; then
    CMD+=(--prompt_type "$PROMPT_TYPE")
    CMD+=(--validator_profile "$VALIDATOR_PROFILE")
  fi

  echo "Evaluating $dataset_name with task=$task"
  "${CMD[@]}" | tee -a logs/evaluate_math_4qa_hf.log
  echo "Finished $dataset_name"
  echo "-----------------------------------"
done
