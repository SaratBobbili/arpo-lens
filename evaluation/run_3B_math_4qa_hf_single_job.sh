#!/bin/bash
set -euo pipefail

# Run root so relative paths in called scripts remain stable.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Cache/log folder for this orchestrator.
mkdir -p logs

# -------------------- Editable Run Config --------------------
# Bing search key used by tool-enabled inference.
BING_API_KEY="9c221824-9a57-4261-b1b7-979959492235"
# Bright Data proxy zone for Bing search requests.
BING_ZONE="serp_api1"
# Bright Data proxy country code for Bing search (cc URL parameter).
BING_LOCATION="us"

# Main reasoning model checkpoint/HF id served on ports 8002/8003.
CHECKPOINT_DIR="/scratch/project/prj-02-llm-reasoning-shakkottai/saratb/ECHO"
# Raw VERL actor checkpoint directory to convert before serving.
RAW_ACTOR_CHECKPOINT_PATH="${CHECKPOINT_DIR}/checkpoint_snapshots/echo3BInstruct/global_step_40/actor"
# Base HF model used as the config/template during VERL->HF merge.
REASON_BASE_MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
# Converted HF model directory served by vLLM.
ACTOR_MODEL_PATH="${CHECKPOINT_DIR}/checkpoint_snapshots/echo3BInstruct/global_step_40/hf"
REASON_MODEL_PATH="${ACTOR_MODEL_PATH}"
# Served model alias for reasoning endpoints; must match infer DEFAULT_MODEL.
REASON_MODEL_NAME="Qwen2.5-3B-Instruct"

# Summarization helper checkpoint/HF id served on ports 8004/8005.
SUMM_MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
# Served model alias for summarization endpoints; must match infer SUMM_MODEL_NAME.
SUMM_MODEL_NAME="Qwen2.5-7B-Instruct"

# completion_sds enables SDS with summarization; completion/default skips summarization.
INFER_MODE="completion"

# System prompt schema:
#   base        -> no tools (pure CoT, table row "Qwen2.5-3B-Instruct")
#   math        -> python only (table row "+ TIR Prompting")
#   search      -> search only
#   code_search -> python + search (default for ARPO/AEPO trained checkpoints)
#   echo        -> ECHO <select>/<tool> schema; loads system prompt from
#                  ECHO_SYSTEM_PROMPT_YAML below instead of a hardcoded literal.
PROMPT_TYPE="echo"

# Per-sample tool-call budgets enforced by the SampleProcessor; set to 0 to disable a tool entirely.
# When PROMPT_TYPE=echo these are overridden below to the combined ECHO budget so
# the combined-budget gate in SampleProcessorCompletion fires before the per-tool
# gate (which would inject an OOD "limit exceeded" feedback message ECHO never saw).
MAX_PYTHON_TIMES="5"
MAX_SEARCH_TIMES="5"

# ---- ECHO-only config (consumed only when PROMPT_TYPE=echo) ----
# Single source of truth for the ECHO system prompt: shared with the trainer at
# ARPO/verl_arpo_entropy/recipe/echo/config/echo_system_prompts.yaml.
ECHO_SYSTEM_PROMPT_YAML="${SCRIPT_DIR}/../ARPO/verl_arpo_entropy/recipe/echo/config/echo_system_prompts.yaml"
# Selects system_prompt_N inside the YAML; must equal data.active_system_prompt
# used during ECHO training (echo_trainer.yaml).
ECHO_ACTIVE_SYSTEM_PROMPT="1"
# Combined per-sample tool budget (matches vLLMRolloutECHO.tool_call_limit, default 5).
ECHO_TOOL_CALL_LIMIT="5"
# Validator profile id (c1..c5) matching the trainer's mask_categories signature;
# routes which format checks gate HL vs LL inside deep_research_echo.compute_score.
# c1 = plan/reason/answer HL; tool choice + payload LL (the v1_ll_hl recipes).
ECHO_VALIDATOR_PROFILE="c1"

# Conda root and env used by the Python tool executor.
CONDA_PATH="/scratch/user/saratb_tamu.edu/miniconda3"
CONDA_ENV="evaluation"
# Directory for NLTK tokenizer data inside the selected conda env.
NLTK_DATA_DIR="$CONDA_PATH/envs/$CONDA_ENV/nltk_data"

# Samples per dataset; large value => full dataset via min(dataset_size, COUNTS) in infer.py.
COUNTS="1000000"

# Restrict this launcher to math benchmarks only (aime24/aime25/math500/gsm8k/math).
DATASET_GROUP="math"

# Pass@k turns (one output file per turn); space separated list.
TURNS="1 2 3"

# Sampling temperature (0.0 => greedy decoding).
TEMPERATURE="0.6"

# Max new tokens per model call; raise for long reasoning traces.
MAX_TOKENS="4096"

# End-to-end timeout for a single sample, in seconds.
SAMPLE_TIMEOUT="900"

# Short tag appended to OUTPUT_PATH and log filenames so different configs land in
# different folders. Auto-composed from the decoding/runtime knobs above; set to ""
# to reuse a plain baseline folder.
RUN_TAG="T${TEMPERATURE}_K${TURNS// /-}_mt${MAX_TOKENS}_to${SAMPLE_TIMEOUT}"
# Checkpoint folder (e.g., global_step_40) inferred from ACTOR_MODEL_PATH.
CHECKPOINT_TAG="$(basename "$(dirname "$ACTOR_MODEL_PATH")")"
CUSTOM_RUN_TAG="LLM_as_judge/${REASON_MODEL_NAME}/${CHECKPOINT_TAG}"

# Model-tagged output directory; "/" -> "__" keeps the model name in one path segment.
MODEL_OUTPUT_TAG="${REASON_MODEL_NAME//\//__}"
OUTPUT_PATH="outputs/hf_math_4qa/${CUSTOM_RUN_TAG}/${DATASET_GROUP}${RUN_TAG:+_$RUN_TAG}"

# Enable LLM-as-judge at evaluation time (true => --use_llm passed to evaluate.py).
USE_LLM="true"
# HF id / local checkpoint of the LLM judge launched by this orchestrator on port 8001.
JUDGE_MODEL_PATH="Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"
# Served alias for the judge endpoint; must match --model_name passed to evaluate.py.
JUDGE_MODEL_NAME="Qwen2.5-72B-Instruct"
# Endpoint URL the evaluator queries; matches the judge launcher PORT.
API_BASE_URL="http://localhost:8001/v1"

# Warmup wait time before starting inference.
SERVER_BOOT_WAIT_SECONDS="60"
# Max seconds to wait for each endpoint health check.
ENDPOINT_READY_TIMEOUT_SECONDS="300"
# Separate (longer) health-check budget for the 72B-GPTQ judge: torch.compile + KV cache
# init alone takes ~5min on first launch, so the smaller reasoning timeout is too tight.
JUDGE_ENDPOINT_READY_TIMEOUT_SECONDS="900"
# Grace period after stopping a server group so VRAM is released before the next launch.
SERVER_TEARDOWN_WAIT_SECONDS="20"

# Set to "true" to skip [1/5]-[3/5] (server bring-up + inference) and jump straight to
# [4/5]-[5/5] (judge launch + evaluation). Use this when inference outputs already exist
# under OUTPUT_PATH and only the judge/eval stage needs to be re-run.
RESUME_FROM_EVAL="false"
# -------------------------------------------------------------

# When PROMPT_TYPE=echo: export ECHO env vars for prompt_manager.PromptManager
# to read, and pin both per-tool budgets to the combined ECHO budget so the
# combined gate in SampleProcessorCompletion is the only one that ever fires.
if [[ "$PROMPT_TYPE" == "echo" ]]; then
  export ECHO_SYSTEM_PROMPT_YAML ECHO_ACTIVE_SYSTEM_PROMPT ECHO_TOOL_CALL_LIMIT
  MAX_PYTHON_TIMES="$ECHO_TOOL_CALL_LIMIT"
  MAX_SEARCH_TIMES="$ECHO_TOOL_CALL_LIMIT"
fi

# Summarization servers are only required when the prompt advertises search AND the
# inference engine is the SDS variant; base/math prompts never emit <search>, so the
# summ pool would just waste GPUs 0-3.
NEEDS_SUMM="false"
if [[ "$INFER_MODE" == "completion_sds" && "$PROMPT_TYPE" != "base" && "$PROMPT_TYPE" != "math" ]]; then
  NEEDS_SUMM="true"
fi

# Convert VERL/FSDP actor shards into a vLLM-loadable HF directory once.
if [[ -d "$RAW_ACTOR_CHECKPOINT_PATH" ]]; then
  if [[ ! -f "${ACTOR_MODEL_PATH}/config.json" ]]; then
    echo "[0/5] Converting VERL actor checkpoint to HF format..."
    python ../ARPO/merge_ckpt/convert_checkpoint_from_verl_to_hf.py merge \
      --backend fsdp \
      --hf_model_path "$REASON_BASE_MODEL_PATH" \
      --local_dir "$RAW_ACTOR_CHECKPOINT_PATH" \
      --target_dir "$ACTOR_MODEL_PATH"
  else
    echo "[0/5] Found converted HF checkpoint, skipping merge: $ACTOR_MODEL_PATH"
  fi
fi

# Hard fail early when the reasoning model is not loadable by vLLM.
# Valid inputs are either:
#   1) local HF directory with config.json, or
#   2) Hugging Face repo id in the form "namespace/model".
if [[ -d "$ACTOR_MODEL_PATH" ]]; then
  if [[ ! -f "${ACTOR_MODEL_PATH}/config.json" ]]; then
    echo "ERROR: ACTOR_MODEL_PATH points to a local directory without config.json: $ACTOR_MODEL_PATH" >&2
    echo "       Provide a converted HF directory (run VERL->HF merge) or set a valid HF repo id." >&2
    exit 1
  fi
elif [[ "$ACTOR_MODEL_PATH" == /* || "$ACTOR_MODEL_PATH" == ./* || "$ACTOR_MODEL_PATH" == ../* ]]; then
  echo "ERROR: ACTOR_MODEL_PATH looks like a local path but does not exist: $ACTOR_MODEL_PATH" >&2
  echo "       Check the path or run checkpoint conversion before launch." >&2
  exit 1
elif [[ "$ACTOR_MODEL_PATH" != */* ]]; then
  echo "ERROR: ACTOR_MODEL_PATH is neither a local HF directory nor a valid HF repo id: $ACTOR_MODEL_PATH" >&2
  echo "       Expected HF repo id format: namespace/model" >&2
  exit 1
fi

REASON_PID=""
SUMM_PID=""
JUDGE_PID=""

# Send SIGTERM to the whole process group of a wrapper launched via `setsid`,
# so the underlying `vllm serve` workers also exit (kill -- -$PID).
stop_server() {
  local pid_var="$1"
  local pid="${!pid_var}"
  [[ -z "$pid" ]] && return 0
  kill -TERM -- "-$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  printf -v "$pid_var" '%s' ""
}

cleanup() {
  stop_server REASON_PID
  stop_server SUMM_PID
  stop_server JUDGE_PID
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

if [[ "$RESUME_FROM_EVAL" != "true" ]]; then
  echo "[1/5] Starting reasoning servers..."
  setsid env MODEL_PATH="$REASON_MODEL_PATH" MODEL_NAME="$REASON_MODEL_NAME" \
    bash vllm_scripts/echo_vllm_launch_reasoning_model_hf_cuda4-7.sh \
    > logs/run_reasoning_wrapper.log 2>&1 < /dev/null &
  REASON_PID=$!

  if [[ "$NEEDS_SUMM" == "true" ]]; then
    echo "[2/5] Starting summarization servers (SDS mode)..."
    setsid env MODEL_PATH="$SUMM_MODEL_PATH" MODEL_NAME="$SUMM_MODEL_NAME" \
      bash vllm_scripts/echo_vllm_launch_summarize_model_hf_cuda0-3.sh \
      > logs/run_summarization_wrapper.log 2>&1 < /dev/null &
    SUMM_PID=$!
  else
    echo "[2/5] Skipping summarization servers (INFER_MODE=$INFER_MODE, PROMPT_TYPE=$PROMPT_TYPE)"
  fi

  echo "Waiting $SERVER_BOOT_WAIT_SECONDS seconds for server warmup..."
  sleep "$SERVER_BOOT_WAIT_SECONDS"

  wait_for_endpoint "http://localhost:8002/v1" "$ENDPOINT_READY_TIMEOUT_SECONDS"
  wait_for_endpoint "http://localhost:8003/v1" "$ENDPOINT_READY_TIMEOUT_SECONDS"
  if [[ "$NEEDS_SUMM" == "true" ]]; then
    wait_for_endpoint "http://localhost:8004/v1" "$ENDPOINT_READY_TIMEOUT_SECONDS"
    wait_for_endpoint "http://localhost:8005/v1" "$ENDPOINT_READY_TIMEOUT_SECONDS"
  fi

  echo "[3/5] Running inference on math benchmarks..."
  mkdir -p "$NLTK_DATA_DIR"
  export NLTK_DATA="$NLTK_DATA_DIR"
  MODEL_PATH="$REASON_MODEL_PATH" \
  DEFAULT_MODEL="$REASON_MODEL_NAME" \
  SUMM_MODEL_PATH="$SUMM_MODEL_PATH" \
  SUMM_MODEL_NAME="$SUMM_MODEL_NAME" \
  INFER_MODE="$INFER_MODE" \
  PROMPT_TYPE="$PROMPT_TYPE" \
  MAX_PYTHON_TIMES="$MAX_PYTHON_TIMES" \
  MAX_SEARCH_TIMES="$MAX_SEARCH_TIMES" \
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

  echo "Inference complete; stopping reasoning servers to free GPUs 4-7..."
  stop_server REASON_PID
else
  echo "[1-3/5] RESUME_FROM_EVAL=true -> skipping reasoning/summarization launch and inference."
  echo "        Using existing inference outputs under: $OUTPUT_PATH"
fi

if [[ "$USE_LLM" == "true" ]]; then
  echo "[4/5] Bringing up LLM judge for evaluation..."
  if [[ -n "$SUMM_PID" ]]; then
    echo "Stopping summarization servers to free GPUs 0-3 for the judge..."
    stop_server SUMM_PID
    sleep "$SERVER_TEARDOWN_WAIT_SECONDS"
  fi
  setsid env MODEL_PATH="$JUDGE_MODEL_PATH" MODEL_NAME="$JUDGE_MODEL_NAME" \
    bash vllm_scripts/echo_vllm_launch_judge_model_hf_cuda0-3.sh \
    > logs/run_judge_wrapper.log 2>&1 < /dev/null &
  JUDGE_PID=$!
  echo "Waiting $SERVER_BOOT_WAIT_SECONDS seconds for judge warmup..."
  sleep "$SERVER_BOOT_WAIT_SECONDS"
  wait_for_endpoint "$API_BASE_URL" "$JUDGE_ENDPOINT_READY_TIMEOUT_SECONDS"
else
  echo "[4/5] Skipping judge launch because USE_LLM=$USE_LLM"
fi

echo "[5/5] Evaluating outputs..."
OUTPUT_DIR="$OUTPUT_PATH" \
USE_LLM="$USE_LLM" \
API_BASE_URL="$API_BASE_URL" \
MODEL_NAME="$JUDGE_MODEL_NAME" \
PROMPT_TYPE="$PROMPT_TYPE" \
VALIDATOR_PROFILE="$ECHO_VALIDATOR_PROFILE" \
bash echo_evaluate_passk_math_4qa.sh | tee "logs/run_eval_math_4qa_hf${RUN_TAG:+_$RUN_TAG}.log"

echo "Run completed successfully. Outputs: $OUTPUT_PATH"
