#!/bin/bash
set -euo pipefail

# Run root so relative paths in called scripts remain stable.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Cache/log folder for this orchestrator.
mkdir -p logs

# Required for tool-enabled inference; export this before running.
: "${BING_API_KEY:?BING_API_KEY must be set}"

# Main model checkpoint or HuggingFace model id used for reasoning generation.
REASON_MODEL_PATH="${REASON_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"

# API alias served by reasoning endpoints; must match infer DEFAULT_MODEL.
REASON_MODEL_NAME="${REASON_MODEL_NAME:-Qwen2.5-7B-Instruct}"

# Summarization helper checkpoint/model id used by SDS tool flow.
SUMM_MODEL_PATH="${SUMM_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"

# API alias served by summarization endpoints; must match infer SUMM_MODEL_NAME.
SUMM_MODEL_NAME="${SUMM_MODEL_NAME:-Qwen2.5-7B-Instruct}"

# Inference mode: completion_sds uses summarization model; completion/default does not.
INFER_MODE="${INFER_MODE:-completion_sds}"

# Number of examples per dataset for this run.
COUNTS="${COUNTS:-50}"

# Output directory root for inference artifacts and metrics.
OUTPUT_PATH="${OUTPUT_PATH:-outputs/hf_math_4qa}"

# Toggle LLM-as-judge at evaluation time.
USE_LLM="${USE_LLM:-false}"

# Optional judge endpoint and model; only used when USE_LLM=true.
API_BASE_URL="${API_BASE_URL:-http://localhost:8001/v1}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-Qwen2.5-72B-Instruct}"

# Delay to let vLLM endpoints come up before inference starts.
SERVER_BOOT_WAIT_SECONDS="${SERVER_BOOT_WAIT_SECONDS:-60}"

REASON_PID=""
SUMM_PID=""

cleanup() {
  if [[ -n "$REASON_PID" ]]; then
    kill "$REASON_PID" 2>/dev/null || true
  fi
  if [[ -n "$SUMM_PID" ]]; then
    kill "$SUMM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT SIGINT SIGTERM

echo "[1/4] Starting reasoning servers..."
MODEL_PATH="$REASON_MODEL_PATH" MODEL_NAME="$REASON_MODEL_NAME" \
  bash vllm_scripts/echo_vllm_launch_reasoning_model_hf_cuda4-7.sh \
  > logs/run_reasoning_wrapper.log 2>&1 &
REASON_PID=$!

if [[ "$INFER_MODE" == "completion_sds" ]]; then
  echo "[2/4] Starting summarization servers (SDS mode)..."
  MODEL_PATH="$SUMM_MODEL_PATH" MODEL_NAME="$SUMM_MODEL_NAME" \
    bash vllm_scripts/echo_vllm_launch_summarize_model_hf_cuda0-3.sh \
    > logs/run_summarization_wrapper.log 2>&1 &
  SUMM_PID=$!
else
  echo "[2/4] Skipping summarization servers because INFER_MODE=$INFER_MODE"
fi

echo "Waiting $SERVER_BOOT_WAIT_SECONDS seconds for server warmup..."
sleep "$SERVER_BOOT_WAIT_SECONDS"

echo "[3/4] Running inference on math + 4QA..."
MODEL_PATH="$REASON_MODEL_PATH" \
DEFAULT_MODEL="$REASON_MODEL_NAME" \
SUMM_MODEL_PATH="$SUMM_MODEL_PATH" \
SUMM_MODEL_NAME="$SUMM_MODEL_NAME" \
INFER_MODE="$INFER_MODE" \
COUNTS="$COUNTS" \
OUTPUT_PATH="$OUTPUT_PATH" \
BING_API_KEY="$BING_API_KEY" \
bash echo_infer_math_4qa_hf.sh | tee logs/run_infer_math_4qa_hf.log

echo "[4/4] Evaluating outputs..."
OUTPUT_DIR="$OUTPUT_PATH" \
USE_LLM="$USE_LLM" \
API_BASE_URL="$API_BASE_URL" \
MODEL_NAME="$JUDGE_MODEL_NAME" \
bash echo_evaluate_passk_math_4qa.sh | tee logs/run_eval_math_4qa_hf.log

echo "Run completed successfully. Outputs: $OUTPUT_PATH"
