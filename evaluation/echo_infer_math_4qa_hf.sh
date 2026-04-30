#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
mkdir -p logs

# Dataset group selector:
# - math      : run only math
# - aime      : run aime24 + aime25
# - gsm8k     : run only gsm8k
# - math500   : run only math500
# - math_all  : run all math datasets
DATASET_GROUP="${DATASET_GROUP:-math}"
# Optional explicit dataset list override (space-separated).
# Example: DATASET_NAMES="math500 gsm8k"
# When set, this takes priority over DATASET_GROUP.
DATASET_NAMES="${DATASET_NAMES:-}"

MATH_DATASETS=(
  "aime24"
  "aime25"
  "math500"
  "gsm8k"
  "math"
)

QA_DATASETS=(
  "hotpotqa"
  "2wiki"
  "musique"
  "bamboogle"
)

if [[ -n "$DATASET_NAMES" ]]; then
  # Split on spaces into a bash array.
  read -r -a DATASETS <<< "$DATASET_NAMES"
elif [[ "$DATASET_GROUP" == "math" ]]; then
  DATASETS=("math")
elif [[ "$DATASET_GROUP" == "aime" ]]; then
  DATASETS=("aime24" "aime25")
elif [[ "$DATASET_GROUP" == "gsm8k" ]]; then
  DATASETS=("gsm8k")
elif [[ "$DATASET_GROUP" == "math500" ]]; then
  DATASETS=("math500")
elif [[ "$DATASET_GROUP" == "math_all" ]]; then
  DATASETS=("${MATH_DATASETS[@]}")
else
  echo "Unsupported DATASET_GROUP=$DATASET_GROUP (expected: math, aime, gsm8k, math500, or math_all)"
  exit 1
fi

# Main reasoning model endpoints (must be served before running this script).
# Space-separated list, for example: "http://localhost:8002/v1 http://localhost:8003/v1".
ENDPOINTS_STR="${ENDPOINTS:-http://localhost:8002/v1 http://localhost:8003/v1}"
read -r -a ENDPOINTS <<< "$ENDPOINTS_STR"

# Set completion_sds to enable full tool + SDS path; set completion to disable SDS summarization.
INFER_MODE="${INFER_MODE:-completion_sds}"

# Model path used to load tokenizer for the reasoning model.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"

# Served model alias for reasoning endpoints.
DEFAULT_MODEL="${DEFAULT_MODEL:-Qwen2.5-7B-Instruct}"

# Output root where each dataset gets its own subfolder.
OUTPUT_PATH="${OUTPUT_PATH:-outputs/hf_math_4qa}"

# Dataset directory root.
DATA_PATH="${DATA_PATH:-data}"

# Pass@k turns (creates one output file per turn).
TURNS="${TURNS:-1 2 3}"

# Prompt style given to the reasoning model.
PROMPT_TYPE="${PROMPT_TYPE:-code_search}"

# Max tool-call budget per sample.
MAX_PYTHON_TIMES="${MAX_PYTHON_TIMES:-5}"
MAX_SEARCH_TIMES="${MAX_SEARCH_TIMES:-8}"

# Max number of examples loaded per dataset for quick smoke testing.
COUNTS="${COUNTS:-50}"

# End-to-end timeout for a single sample, in seconds.
SAMPLE_TIMEOUT="${SAMPLE_TIMEOUT:-900}"

# Sampling temperature passed to vLLM (0.0 => greedy decoding).
TEMPERATURE="${TEMPERATURE:-0.6}"

# Max new tokens per model call; raise for long reasoning traces (AIME, MATH).
MAX_TOKENS="${MAX_TOKENS:-4096}"

# Async throughput knobs.
MAX_CONCURRENT="${MAX_CONCURRENT:-32}"
PYTHON_MAX_CONCURRENT="${PYTHON_MAX_CONCURRENT:-16}"

# Python tool execution environment.
CONDA_PATH="${CONDA_PATH:-/scratch/user/saratb_tamu.edu/miniconda3}"
CONDA_ENV="${CONDA_ENV:-evaluation}"

# Bing credentials for search tool.
BING_API_KEY="${BING_API_KEY:-}"
BING_ZONE="${BING_ZONE:-serp_api1}"
# Bing proxy country code used in Bright Data target URL (maps to cc=... in Bing URL).
BING_LOCATION="${BING_LOCATION:-us}"

# Search retrieval controls.
SEARCH_MAX_RESULTS="${SEARCH_MAX_RESULTS:-10}"
SEARCH_RESULT_LENGTH="${SEARCH_RESULT_LENGTH:-1000}"
BING_REQUESTS_PER_SECOND="${BING_REQUESTS_PER_SECOND:-8.0}"
BING_MAX_RETRIES="${BING_MAX_RETRIES:-3}"
BING_RETRY_DELAY="${BING_RETRY_DELAY:-1.0}"

# Summarization helper endpoints and tokenizer path (used by completion_sds).
SUMM_MODEL_URLS="${SUMM_MODEL_URLS:-http://localhost:8004/v1 http://localhost:8005/v1}"
SUMM_MODEL_NAME="${SUMM_MODEL_NAME:-Qwen2.5-7B-Instruct}"
SUMM_MODEL_PATH="${SUMM_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"

# Persistent SQLite caches for search calls and fetched URLs.
SEARCH_CACHE_FILE="${SEARCH_CACHE_FILE:-search_cache_math_4qa.db}"
URL_CACHE_FILE="${URL_CACHE_FILE:-search_url_cache_math_4qa.db}"

CMD=(python -u infer.py)
CMD+=(--infer_mode "$INFER_MODE")
CMD+=(--endpoints "${ENDPOINTS[@]}")
CMD+=(--model_path "$MODEL_PATH")
CMD+=(--default_model "$DEFAULT_MODEL")
CMD+=(--dataset_name "${DATASETS[@]}")
CMD+=(--output_path "$OUTPUT_PATH")
CMD+=(--data_path "$DATA_PATH")
CMD+=(--turns $TURNS)
CMD+=(--prompt_type "$PROMPT_TYPE")
CMD+=(--counts "$COUNTS")
CMD+=(--max_concurrent_requests "$MAX_CONCURRENT")
CMD+=(--max_python_times "$MAX_PYTHON_TIMES")
CMD+=(--max_search_times "$MAX_SEARCH_TIMES")
CMD+=(--sample_timeout "$SAMPLE_TIMEOUT")
CMD+=(--temperature "$TEMPERATURE")
CMD+=(--max_tokens "$MAX_TOKENS")
CMD+=(--top_p 0.95)
CMD+=(--top_k 20)
CMD+=(--min_p 0.0)
CMD+=(--repetition_penalty 1.1)
CMD+=(--include_stop_str_in_output true)
CMD+=(--python_max_concurrent "$PYTHON_MAX_CONCURRENT")
CMD+=(--conda_path "$CONDA_PATH")
CMD+=(--conda_env "$CONDA_ENV")
CMD+=(--bing_api_key "$BING_API_KEY")
CMD+=(--bing_zone "$BING_ZONE")
CMD+=(--bing_location "$BING_LOCATION")
CMD+=(--search_max_results "$SEARCH_MAX_RESULTS")
CMD+=(--search_result_length "$SEARCH_RESULT_LENGTH")
CMD+=(--bing_requests_per_second "$BING_REQUESTS_PER_SECOND")
CMD+=(--bing_max_retries "$BING_MAX_RETRIES")
CMD+=(--bing_retry_delay "$BING_RETRY_DELAY")
CMD+=(--summ_model_urls $SUMM_MODEL_URLS)
CMD+=(--summ_model_name "$SUMM_MODEL_NAME")
CMD+=(--summ_model_path "$SUMM_MODEL_PATH")
CMD+=(--search_cache_file "$SEARCH_CACHE_FILE")
CMD+=(--url_cache_file "$URL_CACHE_FILE")

echo "Running: ${CMD[*]}"
"${CMD[@]}" | tee logs/infer_math_4qa_hf.log
