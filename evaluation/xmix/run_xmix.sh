#!/bin/bash
# End-to-end orchestration for the cross-checkpoint mixed-prefix experiment.
#
# Pipeline (single bash job, one judge boot at the end):
#   [1] cand1 greedy rollouts on the val set (existing infer.py + vllm servers)
#   [2] cand2 greedy rollouts on the val set
#   [3] splicer.py builds mixed prefixes (HL=cand1 thinks, LL=cand2 selects+tools+results)
#   [4] cand1 + prefix continuation rollouts (xmix/infer_prefix.py)
#   [5] LLM judge: evaluate.py over all three runs with one judge boot
#
# All artifacts land under ${XMIX_ROOT}/<base_run>/c1_<step>_c2_<step>/, with
# per-phase subfolders (cand1/, cand2/, data_mixed/, mix/) and a single
# logs/ folder. Infrastructure is lifted directly from
# evaluation/run_3B_math_4qa_hf_single_job.sh; nothing in evaluation/src is
# modified.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
EVAL_DIR="$( cd "${SCRIPT_DIR}/.." &> /dev/null && pwd )"
REPO_ROOT="$( cd "${EVAL_DIR}/.." &> /dev/null && pwd )"
cd "${EVAL_DIR}"

# ============================ Editable Run Config ============================
# Bing search key + zone used by the search tool inside both rollout passes
# and the prefix-continuation pass. Identical to the training config so the
# tool surface the model sees in eval matches what it saw during RL.
BING_API_KEY="${BING_API_KEY:-9c221824-9a57-4261-b1b7-979959492235}"
BING_ZONE="${BING_ZONE:-serp_api1}"
BING_LOCATION="${BING_LOCATION:-us}"

# Snapshot root: each global_step_<N> subdir contains both the FSDP "actor"
# shards and a pre-merged "hf" directory we can serve directly via vLLM.
BASE_RUN="${BASE_RUN:-echo3BInstruct}"
CKPT_ROOT="${CKPT_ROOT:-/scratch/project/prj-02-llm-reasoning-shakkottai/saratb/ECHO/checkpoint_snapshots/${BASE_RUN}}"

# Two ECHO checkpoints from the same training run. Defaults: step 15 (cand1)
# and step 40 (cand2) of echo3BInstruct, both pre-merged under <step>/hf.
CAND1_STEP="${CAND1_STEP:-15}"
CAND2_STEP="${CAND2_STEP:-40}"
PREFIX_TAG="c1_${CAND1_STEP}_c2_${CAND2_STEP}"

# Per-experiment shared output root; both candidates and the mix live here.
XMIX_ROOT_DEFAULT="${EVAL_DIR}/xmix_runs"
XMIX_ROOT="${XMIX_ROOT:-${XMIX_ROOT_DEFAULT}/${BASE_RUN}/${PREFIX_TAG}}"

# Base HF model used as the merge template + the served alias for the
# reasoning endpoints. Must match the training run.
REASON_BASE_MODEL_PATH="${REASON_BASE_MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
REASON_MODEL_NAME="${REASON_MODEL_NAME:-Qwen2.5-3B-Instruct}"

# Summarization helper kept identical across all three phases; same alias as
# echo_vllm_launch_summarize_model_hf_cuda0-3.sh defaults.
SUMM_MODEL_PATH="${SUMM_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
SUMM_MODEL_NAME="${SUMM_MODEL_NAME:-Qwen2.5-7B-Instruct}"

# Judge model (loaded once at the end to score all three runs in one pass).
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-Qwen/Qwen2.5-72B-Instruct}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-Qwen2.5-72B-Instruct}"
API_BASE_URL="${API_BASE_URL:-http://localhost:8001/v1}"

# Inference knobs: greedy decoding (T=0, top_p=1, top_k=-1, no rep penalty) so
# both candidates produce a single deterministic rollout per question. Matches
# run_math_4qa_hf_greedy.sh.
INFER_MODE="${INFER_MODE:-completion_sds}"
PROMPT_TYPE="${PROMPT_TYPE:-echo}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"     # vLLM convention: -1 disables top-k
MIN_P="${MIN_P:-0.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TURNS="${TURNS:-1}"
COUNTS="${COUNTS:-1000000}"          # cap for safety; > val set size => full set
SAMPLE_TIMEOUT="${SAMPLE_TIMEOUT:-900}"
MAX_CONCURRENT="${MAX_CONCURRENT:-32}"
PYTHON_MAX_CONCURRENT="${PYTHON_MAX_CONCURRENT:-16}"

# Dataset: grpo_mix mirrors the val parquet used during ECHO RL training; rows
# are {question, answer}. The phase-3 splicer rewrites this into a parallel
# JSONL with an extra `prefix` field consumed by phase-4's infer_prefix.py.
DATASET_GROUP="${DATASET_GROUP:-grpo_mix}"
DATA_PATH_BASE="${DATA_PATH_BASE:-${EVAL_DIR}/data}"

# ECHO prompt + budget knobs (consumed by PromptManager when PROMPT_TYPE=echo).
ECHO_SYSTEM_PROMPT_YAML="${ECHO_SYSTEM_PROMPT_YAML:-${REPO_ROOT}/ARPO/verl_arpo_entropy/recipe/echo/config/echo_system_prompts.yaml}"
ECHO_ACTIVE_SYSTEM_PROMPT="${ECHO_ACTIVE_SYSTEM_PROMPT:-1}"
ECHO_TOOL_CALL_LIMIT="${ECHO_TOOL_CALL_LIMIT:-8}"
ECHO_VALIDATOR_PROFILE="${ECHO_VALIDATOR_PROFILE:-c1}"
MAX_PYTHON_TIMES="${MAX_PYTHON_TIMES:-${ECHO_TOOL_CALL_LIMIT}}"
MAX_SEARCH_TIMES="${MAX_SEARCH_TIMES:-${ECHO_TOOL_CALL_LIMIT}}"
export ECHO_SYSTEM_PROMPT_YAML ECHO_ACTIVE_SYSTEM_PROMPT ECHO_TOOL_CALL_LIMIT

# Tool runtime: same conda env name the training run used so the python tool
# subprocess imports match (sympy/numpy/etc.).
CONDA_PATH="${CONDA_PATH:-/scratch/user/saratb_tamu.edu/miniconda3}"
CONDA_ENV="${CONDA_ENV:-evaluation}"
NLTK_DATA_DIR="${CONDA_PATH}/envs/${CONDA_ENV}/nltk_data"
mkdir -p "${NLTK_DATA_DIR}"
export NLTK_DATA="${NLTK_DATA_DIR}"

# Persistent caches (shared across all three phases so the search/python work
# done in cand1 is reused by cand2 / mix wherever the queries match).
SEARCH_CACHE_FILE="${SEARCH_CACHE_FILE:-${XMIX_ROOT}/search_cache.db}"
URL_CACHE_FILE="${URL_CACHE_FILE:-${XMIX_ROOT}/search_url_cache.db}"

# Endpoint topology (matches the existing vLLM launchers).
ENDPOINTS_STR="${ENDPOINTS:-http://localhost:8002/v1 http://localhost:8003/v1}"
SUMM_MODEL_URLS_STR="${SUMM_MODEL_URLS:-http://localhost:8004/v1 http://localhost:8005/v1}"
SERVER_BOOT_WAIT_SECONDS="${SERVER_BOOT_WAIT_SECONDS:-120}"
ENDPOINT_READY_TIMEOUT_SECONDS="${ENDPOINT_READY_TIMEOUT_SECONDS:-900}"
JUDGE_ENDPOINT_READY_TIMEOUT_SECONDS="${JUDGE_ENDPOINT_READY_TIMEOUT_SECONDS:-1200}"
SERVER_TEARDOWN_WAIT_SECONDS="${SERVER_TEARDOWN_WAIT_SECONDS:-20}"

# Splicer needs to import verl.utils.reward_score.deep_research_echo from the
# arpo-lens repo (verl is not pip-installed in the eval env).
VERL_PYTHONPATH="${REPO_ROOT}/ARPO/verl_arpo_entropy"
# -----------------------------------------------------------------------------

# ============================ Layout setup ============================
CAND1_HF="${CKPT_ROOT}/global_step_${CAND1_STEP}/hf"
CAND2_HF="${CKPT_ROOT}/global_step_${CAND2_STEP}/hf"

CAND1_OUT="${XMIX_ROOT}/cand1"
CAND2_OUT="${XMIX_ROOT}/cand2"
MIX_DATA="${XMIX_ROOT}/data_mixed"
MIX_OUT="${XMIX_ROOT}/mix"
LOG_DIR="${XMIX_ROOT}/logs"
SUMMARY_PATH="${XMIX_ROOT}/summary.json"
mkdir -p "${CAND1_OUT}" "${CAND2_OUT}" "${MIX_DATA}" "${MIX_OUT}" "${LOG_DIR}"
chmod -R u+rw "${XMIX_ROOT}" 2>/dev/null || true

# Sanity-check both candidate HF dirs before we boot anything.
for hf_dir in "${CAND1_HF}" "${CAND2_HF}"; do
  if [[ ! -f "${hf_dir}/config.json" ]]; then
    echo "ERROR: Expected pre-merged HF checkpoint at ${hf_dir}/config.json" >&2
    exit 1
  fi
done

# Snapshot the run config so the output folder is fully self-describing.
cat > "${XMIX_ROOT}/run_config.yaml" <<EOF
timestamp: "$(date -Iseconds)"
driver_script: "$(basename "$0")"
base_run: "${BASE_RUN}"
cand1:
  step: "${CAND1_STEP}"
  hf_path: "${CAND1_HF}"
cand2:
  step: "${CAND2_STEP}"
  hf_path: "${CAND2_HF}"
infer:
  mode: "${INFER_MODE}"
  prompt_type: "${PROMPT_TYPE}"
  dataset_group: "${DATASET_GROUP}"
  counts: ${COUNTS}
  turns: "${TURNS}"
  temperature: ${TEMPERATURE}
  top_p: ${TOP_P}
  top_k: ${TOP_K}
  min_p: ${MIN_P}
  repetition_penalty: ${REPETITION_PENALTY}
  max_tokens: ${MAX_TOKENS}
  sample_timeout: ${SAMPLE_TIMEOUT}
  max_python_times: ${MAX_PYTHON_TIMES}
  max_search_times: ${MAX_SEARCH_TIMES}
echo:
  system_prompt_yaml: "${ECHO_SYSTEM_PROMPT_YAML}"
  active_system_prompt: "${ECHO_ACTIVE_SYSTEM_PROMPT}"
  tool_call_limit: "${ECHO_TOOL_CALL_LIMIT}"
  validator_profile: "${ECHO_VALIDATOR_PROFILE}"
judge:
  model_path: "${JUDGE_MODEL_PATH}"
  model_name: "${JUDGE_MODEL_NAME}"
  api_base_url: "${API_BASE_URL}"
EOF

# ============================ Server lifecycle helpers ============================
# Mirror the helpers in run_3B_math_4qa_hf_single_job.sh so a SIGTERM to a
# wrapper PID kills the underlying `vllm serve` workers via setsid + pgrp kill.
REASON_PID=""
SUMM_PID=""
JUDGE_PID=""

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
  echo "Waiting for endpoint: ${endpoint}"
  while (( waited < timeout )); do
    if curl -sS --max-time 3 "${endpoint}/models" >/dev/null 2>&1; then
      echo "Endpoint ready: ${endpoint}"
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  echo "Endpoint did not become ready within ${timeout}s: ${endpoint}" >&2
  return 1
}

start_reasoning_servers() {
  local model_path="$1"
  echo "[servers] reasoning model = ${model_path}"
  setsid env MODEL_PATH="${model_path}" MODEL_NAME="${REASON_MODEL_NAME}" \
    bash vllm_scripts/echo_vllm_launch_reasoning_model_hf_cuda4-7.sh \
    > "${LOG_DIR}/reasoning_wrapper.log" 2>&1 < /dev/null &
  REASON_PID=$!
}

start_summ_servers() {
  echo "[servers] summarization model = ${SUMM_MODEL_PATH}"
  setsid env MODEL_PATH="${SUMM_MODEL_PATH}" MODEL_NAME="${SUMM_MODEL_NAME}" \
    bash vllm_scripts/echo_vllm_launch_summarize_model_hf_cuda0-3.sh \
    > "${LOG_DIR}/summarization_wrapper.log" 2>&1 < /dev/null &
  SUMM_PID=$!
}

start_judge_server() {
  echo "[servers] judge model = ${JUDGE_MODEL_PATH}"
  setsid env MODEL_PATH="${JUDGE_MODEL_PATH}" MODEL_NAME="${JUDGE_MODEL_NAME}" \
    bash vllm_scripts/echo_vllm_launch_judge_model_hf_cuda0-3.sh \
    > "${LOG_DIR}/judge_wrapper.log" 2>&1 < /dev/null &
  JUDGE_PID=$!
}

# ============================ Inference helper ============================
# Single source of truth for the python infer.* CMD; same flags as
# echo_infer_math_4qa_hf.sh so the trainer-time tool surface stays identical.
# Args: <infer_py> <output_path> <data_path>
run_inference() {
  local infer_py="$1"
  local output_path="$2"
  local data_path="$3"

  local cmd=(python -u "${infer_py}")
  cmd+=(--infer_mode "${INFER_MODE}")
  # Endpoints / SUMM URLs are space-separated; rely on word-splitting once.
  # shellcheck disable=SC2206
  local endpoints=( ${ENDPOINTS_STR} )
  # shellcheck disable=SC2206
  local summ_urls=( ${SUMM_MODEL_URLS_STR} )
  cmd+=(--endpoints "${endpoints[@]}")
  cmd+=(--model_path "${REASON_BASE_MODEL_PATH}")
  cmd+=(--default_model "${REASON_MODEL_NAME}")
  cmd+=(--dataset_name "${DATASET_GROUP}")
  cmd+=(--output_path "${output_path}")
  cmd+=(--data_path "${data_path}")
  # TURNS is a space-separated list; pass each value as a separate --turns arg
  # token (argparse with nargs='+').
  # shellcheck disable=SC2206
  local turns_arr=( ${TURNS} )
  cmd+=(--turns "${turns_arr[@]}")
  cmd+=(--prompt_type "${PROMPT_TYPE}")
  cmd+=(--counts "${COUNTS}")
  cmd+=(--max_concurrent_requests "${MAX_CONCURRENT}")
  cmd+=(--max_python_times "${MAX_PYTHON_TIMES}")
  cmd+=(--max_search_times "${MAX_SEARCH_TIMES}")
  cmd+=(--sample_timeout "${SAMPLE_TIMEOUT}")
  cmd+=(--temperature "${TEMPERATURE}")
  cmd+=(--max_tokens "${MAX_TOKENS}")
  cmd+=(--top_p "${TOP_P}")
  cmd+=(--top_k "${TOP_K}")
  cmd+=(--min_p "${MIN_P}")
  cmd+=(--repetition_penalty "${REPETITION_PENALTY}")
  cmd+=(--include_stop_str_in_output true)
  cmd+=(--python_max_concurrent "${PYTHON_MAX_CONCURRENT}")
  cmd+=(--conda_path "${CONDA_PATH}")
  cmd+=(--conda_env "${CONDA_ENV}")
  cmd+=(--bing_api_key "${BING_API_KEY}")
  cmd+=(--bing_zone "${BING_ZONE}")
  cmd+=(--bing_location "${BING_LOCATION}")
  cmd+=(--summ_model_urls "${summ_urls[@]}")
  cmd+=(--summ_model_name "${SUMM_MODEL_NAME}")
  cmd+=(--summ_model_path "${SUMM_MODEL_PATH}")
  cmd+=(--search_cache_file "${SEARCH_CACHE_FILE}")
  cmd+=(--url_cache_file "${URL_CACHE_FILE}")
  echo "Running: ${cmd[*]}"
  "${cmd[@]}"
}

# ============================ Phase 1: cand1 rollouts ============================
echo "================ [1/5] cand1 (step ${CAND1_STEP}) rollouts ================"
start_reasoning_servers "${CAND1_HF}"
start_summ_servers
sleep "${SERVER_BOOT_WAIT_SECONDS}"
wait_for_endpoint "http://localhost:8002/v1" "${ENDPOINT_READY_TIMEOUT_SECONDS}"
wait_for_endpoint "http://localhost:8003/v1" "${ENDPOINT_READY_TIMEOUT_SECONDS}"
wait_for_endpoint "http://localhost:8004/v1" "${ENDPOINT_READY_TIMEOUT_SECONDS}"
wait_for_endpoint "http://localhost:8005/v1" "${ENDPOINT_READY_TIMEOUT_SECONDS}"
run_inference "infer.py" "${CAND1_OUT}" "${DATA_PATH_BASE}" 2>&1 | tee "${LOG_DIR}/cand1_infer.log"
stop_server REASON_PID
sleep "${SERVER_TEARDOWN_WAIT_SECONDS}"

# ============================ Phase 2: cand2 rollouts ============================
echo "================ [2/5] cand2 (step ${CAND2_STEP}) rollouts ================"
start_reasoning_servers "${CAND2_HF}"
sleep "${SERVER_BOOT_WAIT_SECONDS}"
wait_for_endpoint "http://localhost:8002/v1" "${ENDPOINT_READY_TIMEOUT_SECONDS}"
wait_for_endpoint "http://localhost:8003/v1" "${ENDPOINT_READY_TIMEOUT_SECONDS}"
run_inference "infer.py" "${CAND2_OUT}" "${DATA_PATH_BASE}" 2>&1 | tee "${LOG_DIR}/cand2_infer.log"
stop_server REASON_PID
sleep "${SERVER_TEARDOWN_WAIT_SECONDS}"

# ============================ Phase 3: splice ============================
echo "================ [3/5] splice (HL=cand1 thinks, LL=cand2 selects+tools) ================"
PYTHONPATH="${VERL_PYTHONPATH}:${PYTHONPATH:-}" python -u "${SCRIPT_DIR}/splicer.py" \
  --cand1 "${CAND1_OUT}/${DATASET_GROUP}/${DATASET_GROUP}_output_1.json" \
  --cand2 "${CAND2_OUT}/${DATASET_GROUP}/${DATASET_GROUP}_output_1.json" \
  --dataset_name "${DATASET_GROUP}" \
  --out "${MIX_DATA}" 2>&1 | tee "${LOG_DIR}/splice.log"

# ============================ Phase 4: cand1 + prefix continuation ============================
echo "================ [4/5] cand1 (step ${CAND1_STEP}) + prefix continuation ================"
start_reasoning_servers "${CAND1_HF}"
sleep "${SERVER_BOOT_WAIT_SECONDS}"
wait_for_endpoint "http://localhost:8002/v1" "${ENDPOINT_READY_TIMEOUT_SECONDS}"
wait_for_endpoint "http://localhost:8003/v1" "${ENDPOINT_READY_TIMEOUT_SECONDS}"
run_inference "${SCRIPT_DIR}/infer_prefix.py" "${MIX_OUT}" "${MIX_DATA}" 2>&1 | tee "${LOG_DIR}/mix_infer.log"
stop_server REASON_PID
stop_server SUMM_PID
sleep "${SERVER_TEARDOWN_WAIT_SECONDS}"

# ============================ Phase 5: judge + scoring ============================
echo "================ [5/5] LLM-judge scoring (single boot, three runs) ================"
start_judge_server
sleep "${SERVER_BOOT_WAIT_SECONDS}"
wait_for_endpoint "${API_BASE_URL}" "${JUDGE_ENDPOINT_READY_TIMEOUT_SECONDS}"

# echo_evaluate_passk_math_4qa.sh scans <OUTPUT_DIR>/*/*_output_*.json; point
# it at each phase folder in turn so metrics land next to the inference JSONs.
for phase_dir in "${CAND1_OUT}" "${CAND2_OUT}" "${MIX_OUT}"; do
  phase_name="$(basename "${phase_dir}")"
  echo "[scoring] ${phase_name} -> ${phase_dir}"
  OUTPUT_DIR="${phase_dir}" \
  USE_LLM=true \
  API_BASE_URL="${API_BASE_URL}" \
  MODEL_NAME="${JUDGE_MODEL_NAME}" \
  PROMPT_TYPE="${PROMPT_TYPE}" \
  VALIDATOR_PROFILE="${ECHO_VALIDATOR_PROFILE}" \
  bash echo_evaluate_passk_math_4qa.sh 2>&1 | tee -a "${LOG_DIR}/eval.log"
done

stop_server JUDGE_PID

# ============================ Phase 5.5: aggregate summary ============================
# Read each phase's *_metrics_overall.json and stitch a single summary.json so
# the headline numbers (cand1 / cand2 / mix) sit alongside splice diagnostics.
python - <<'PY' "${CAND1_OUT}" "${CAND2_OUT}" "${MIX_OUT}" "${XMIX_ROOT}/splice_summary.json" "${SUMMARY_PATH}"
import json, sys, glob, os

cand1, cand2, mix, splice_summary_path, out_path = sys.argv[1:]

def load_phase(phase_dir):
    overalls = {}
    for path in sorted(glob.glob(os.path.join(phase_dir, "*", "*_metrics_overall.json"))):
        ds = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            overalls[ds] = json.load(f)
    return overalls

summary = {
    "cand1": load_phase(cand1),
    "cand2": load_phase(cand2),
    "mix": load_phase(mix),
}
if os.path.isfile(splice_summary_path):
    with open(splice_summary_path) as f:
        summary["splice"] = json.load(f)

with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Wrote summary: {out_path}")
PY

echo "================ Done. All artifacts under ${XMIX_ROOT} ================"
