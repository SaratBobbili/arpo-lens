#!/bin/bash
# Launch ECHO cross-checkpoint HL/LL ping-pong validation.
# Mirrors the env-setup pattern of recipe/echo/ECHO_2.5_3B_Reasoning_1node_v1_ll_hl.sh.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
VERL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARPO_ROOT="$(dirname "$VERL_ROOT")"
REPO_ROOT="$(dirname "$ARPO_ROOT")"
cd "$VERL_ROOT"
echo "Switched to verl root directory: $VERL_ROOT"

# ============================ Environment Setup ============================
export TMPDIR=/tmp/saratb_xckpt
export RAY_TMPDIR=/tmp/saratb_xckpt
mkdir -p "$TMPDIR"
export VERL_LOGGING_LEVEL=WARN
export VLLM_LOGGING_LEVEL=WARN
export RAY_BACKEND_LOG_LEVEL=warning
export NCCL_DEBUG=WARN
export VLLM_USE_V1=1
# Disable Dynamo / torch.compile to match the recipe's eager mode.
export TORCHDYNAMO_DISABLE=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

export PYTHONPATH="${VERL_ROOT}:$PYTHONPATH"

# ============================ Run identity ============================
RUN_NAME="echo3B-rerun-entropy-coeff-0-penalty-0.1"
CHECKPOINT_ROOT="/scratch/project/prj-02-llm-reasoning-shakkottai/saratb/ECHO/checkpoints/${RUN_NAME}"
HF_CACHE_DIR="/scratch/project/prj-02-llm-reasoning-shakkottai/saratb/ECHO/xckpt_eval/merged_hf"
OUTPUT_DIR="/scratch/project/prj-02-llm-reasoning-shakkottai/saratb/ECHO/xckpt_eval/results/${RUN_NAME}"

# Two HL steps and two LL steps -> 4 combos total. Override on the CLI as
#   bash run_xckpt_eval.sh '[10,25]' '[10,25]'
HL_STEPS="${1:-[10,25]}"
LL_STEPS="${2:-[10,25]}"

# ============================ Brightdata (search tool) ============================
# Same Brightdata account/key the training run used; required by BingSearchTool.
export BRIGHTDATA_API_KEY="${BRIGHTDATA_API_KEY:-9c221824-9a57-4261-b1b7-979959492235}"
BRIGHTDATA_ZONE="${BRIGHTDATA_ZONE:-serp_api1}"
BRIGHTDATA_LOCATION="${BRIGHTDATA_LOCATION:-us}"
BRIGHTDATA_TIMEOUT="${BRIGHTDATA_TIMEOUT:-90}"

# Search cache (shared format-compatible JSON the training run produced).
SEARCH_CACHE_PATH="${ARPO_ROOT}/search_cache/search_cache_xckpt_eval.json"
mkdir -p "$(dirname "$SEARCH_CACHE_PATH")"
# Optionally point at the training cache to bootstrap; uncomment to share entries:
# SEARCH_CACHE_PATH="${ARPO_ROOT}/search_cache/search_cache_echo_0_3B_penalty-0_1.json"

# ============================ Output prep ============================
mkdir -p "$OUTPUT_DIR"
CONFIG_SNAPSHOT_DIR="${OUTPUT_DIR}/config_snapshot"
mkdir -p "$CONFIG_SNAPSHOT_DIR"
cp "${SCRIPT_PATH}" "$CONFIG_SNAPSHOT_DIR/launch_script.sh"
cp -r "${SCRIPT_DIR}/config" "$CONFIG_SNAPSHOT_DIR/config"

# ============================ Launch ============================
python3 -m recipe.echo_xckpt_eval.main \
    --config-path="${SCRIPT_DIR}/config" \
    --config-name=xckpt_eval \
    xckpt.checkpoint_root="${CHECKPOINT_ROOT}" \
    xckpt.run_name="${RUN_NAME}" \
    xckpt.hf_cache_dir="${HF_CACHE_DIR}" \
    xckpt.output_dir="${OUTPUT_DIR}" \
    xckpt.hl_steps="${HL_STEPS}" \
    xckpt.ll_steps="${LL_STEPS}" \
    xckpt.tools.instances.search.params.api_key="${BRIGHTDATA_API_KEY}" \
    xckpt.tools.instances.search.params.zone="${BRIGHTDATA_ZONE}" \
    xckpt.tools.instances.search.params.location="${BRIGHTDATA_LOCATION}" \
    xckpt.tools.instances.search.params.request_timeout="${BRIGHTDATA_TIMEOUT}" \
    xckpt.tools.instances.search.params.cache_file="${SEARCH_CACHE_PATH}" \
    2>&1 | tee "${OUTPUT_DIR}/run.log"
