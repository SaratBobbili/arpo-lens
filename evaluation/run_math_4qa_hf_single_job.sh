#!/bin/bash
set -euo pipefail

# Run root so relative paths in called scripts remain stable.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Cache/log folder for this orchestrator.
mkdir -p logs

# -------------------- Editable Run Config --------------------
# Bing search key used by tool-enabled inference.
BING_API_KEY="e39479c5-a6f0-4043-899d-d19fd26de1c4"
# Bright Data proxy zone for Bing search requests.
BING_ZONE="${BING_ZONE:-serp_api1}"
# Bright Data proxy country code for Bing search (cc URL parameter).
BING_LOCATION="${BING_LOCATION:-us}"

# Main reasoning model checkpoint/HF id served on ports 8002/8003.
REASON_MODEL_PATH="${REASON_MODEL_PATH:-dongguanting/Qwen2.5-3B-ARPO}"

# Served model alias for reasoning endpoints; must match infer DEFAULT_MODEL.
REASON_MODEL_NAME="${REASON_MODEL_NAME:-dongguanting/Qwen2.5-3B-ARPO}"

# Summarization helper checkpoint/HF id served on ports 8004/8005.
SUMM_MODEL_PATH="${SUMM_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"

# Served model alias for summarization endpoints; must match infer SUMM_MODEL_NAME.
SUMM_MODEL_NAME="${SUMM_MODEL_NAME:-Qwen2.5-7B-Instruct}"

# completion_sds enables SDS with summarization; completion/default skips summarization.
INFER_MODE="${INFER_MODE:-completion_sds}"

# Conda root and env used by the Python tool executor.
CONDA_PATH="${CONDA_PATH:-/scratch/user/saratb_tamu.edu/miniconda3}"
CONDA_ENV="${CONDA_ENV:-evaluation}"
# Directory for NLTK tokenizer data inside the selected conda env.
NLTK_DATA_DIR="${NLTK_DATA_DIR:-$CONDA_PATH/envs/$CONDA_ENV/nltk_data}"

# Number of samples per dataset.
# Use a very large default so infer.py processes the full dataset via min(dataset_size, COUNTS).
COUNTS="${COUNTS:-1000000}"

# Restrict this launcher to math benchmarks only (aime24/aime25/math500/gsm8k/math).
DATASET_GROUP="${DATASET_GROUP:-math}"

# Short tag appended to OUTPUT_PATH and log filenames; lets variant scripts
# (e.g. longctx / greedy) write to separate folders without clobbering the baseline.
RUN_TAG="${RUN_TAG:-}"

# Model-tagged output directory so predictions and metrics are grouped by evaluated model and dataset group.
# MODEL_OUTPUT_TAG replaces "/" to keep the model name in a single path segment.
MODEL_OUTPUT_TAG="${MODEL_OUTPUT_TAG:-${REASON_MODEL_NAME//\//__}}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/hf_math_4qa/${MODEL_OUTPUT_TAG}/${DATASET_GROUP}${RUN_TAG:+_$RUN_TAG}}"

# Enable LLM-as-judge at evaluation time.
USE_LLM="${USE_LLM:-false}"

# Judge endpoint/model used only when USE_LLM=true.
API_BASE_URL="${API_BASE_URL:-http://localhost:8001/v1}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-Qwen2.5-72B-Instruct}"

# Pass@k turns (one output file per turn); comma or space separated list.
TURNS="${TURNS:-1 2 3}"

# Sampling temperature (0.0 => greedy decoding).
TEMPERATURE="${TEMPERATURE:-0.6}"

# Max new tokens per model call; raise for long reasoning traces.
MAX_TOKENS="${MAX_TOKENS:-4096}"

# End-to-end timeout for a single sample, in seconds.
SAMPLE_TIMEOUT="${SAMPLE_TIMEOUT:-900}"

# Warmup wait time before starting inference.
SERVER_BOOT_WAIT_SECONDS="${SERVER_BOOT_WAIT_SECONDS:-60}"

# Max seconds to wait for each endpoint health check.
ENDPOINT_READY_TIMEOUT_SECONDS="${ENDPOINT_READY_TIMEOUT_SECONDS:-300}"
# -------------------------------------------------------------

if [[ -z "$BING_API_KEY" ]]; then
  echo "BING_API_KEY is empty. Set it in this script or via environment."
  exit 1
fi

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

wait_for_endpoint() {
  local endpoint="$1"
  local timeout="$2"
  local waited=0
  echo "Waiting for endpoint: $endpoint"
  while (( waited < timeout )); do
    if curl -sS --max-time 3 "${endpoint}/models" >/dev/null 2>&1; then
      echo "Endpoint ready: $endpoint"
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  echo "Endpoint did not become ready within ${timeout}s: $endpoint"
  return 1
}

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

wait_for_endpoint "http://localhost:8002/v1" "$ENDPOINT_READY_TIMEOUT_SECONDS"
wait_for_endpoint "http://localhost:8003/v1" "$ENDPOINT_READY_TIMEOUT_SECONDS"
if [[ "$INFER_MODE" == "completion_sds" ]]; then
  wait_for_endpoint "http://localhost:8004/v1" "$ENDPOINT_READY_TIMEOUT_SECONDS"
  wait_for_endpoint "http://localhost:8005/v1" "$ENDPOINT_READY_TIMEOUT_SECONDS"
fi

echo "[3/4] Running inference on math benchmarks..."
mkdir -p "$NLTK_DATA_DIR"
export NLTK_DATA="$NLTK_DATA_DIR"
MODEL_PATH="$REASON_MODEL_PATH" \
DEFAULT_MODEL="$REASON_MODEL_NAME" \
SUMM_MODEL_PATH="$SUMM_MODEL_PATH" \
SUMM_MODEL_NAME="$SUMM_MODEL_NAME" \
INFER_MODE="$INFER_MODE" \
CONDA_PATH="$CONDA_PATH" \
CONDA_ENV="$CONDA_ENV" \
COUNTS="$COUNTS" \
OUTPUT_PATH="$OUTPUT_PATH" \
BING_API_KEY="$BING_API_KEY" \
BING_ZONE="$BING_ZONE" \
BING_LOCATION="$BING_LOCATION" \
DATASET_GROUP="$DATASET_GROUP" \
TURNS="$TURNS" \
TEMPERATURE="$TEMPERATURE" \
MAX_TOKENS="$MAX_TOKENS" \
SAMPLE_TIMEOUT="$SAMPLE_TIMEOUT" \
bash echo_infer_math_4qa_hf.sh | tee "logs/run_infer_math_4qa_hf${RUN_TAG:+_$RUN_TAG}.log"

echo "[4/4] Evaluating outputs..."
OUTPUT_DIR="$OUTPUT_PATH" \
USE_LLM="$USE_LLM" \
API_BASE_URL="$API_BASE_URL" \
MODEL_NAME="$JUDGE_MODEL_NAME" \
bash echo_evaluate_passk_math_4qa.sh | tee "logs/run_eval_math_4qa_hf${RUN_TAG:+_$RUN_TAG}.log"

echo "Run completed successfully. Outputs: $OUTPUT_PATH"
