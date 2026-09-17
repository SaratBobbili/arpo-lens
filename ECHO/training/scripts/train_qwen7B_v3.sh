#!/bin/bash
# ECHO nested-phase RL on Qwen2.5-7B-Instruct (ECHO cold-start SFT, lr 5e-8, 1 epoch).
# Only the values that differ from the base profile live here; everything else
# (lengths, tools, masks, clips, phase knobs) comes from ${BASE_PROFILE}.
#   bash train_qwen7B.sh
#   EXPERIMENT_NAME=echo7BInstruct_r2 bash train_qwen7B.sh hl_lr=5e-7 ll_num_iters=40
# Requires BRIGHTDATA_API_KEY in secrets.sh for live search on a cache miss.
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_PROFILE="training_config/config1.yaml"

PROJECT_NAME="${PROJECT_NAME:-qwen25_7B}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-echo7BInstruct_lr3e8}"
# HF weights resolved as ${SFT_ROOT}/${ACTOR_MODEL_SUBPATH} (SFT_ROOT from secrets.sh).
ACTOR_MODEL_SUBPATH="${ACTOR_MODEL_SUBPATH:-checkpoints/Qwen2.5-7B-Instruct-lr3e8-ep1}"
# Query -> SERP cache is model-independent, so share the largest one for hit rate.
SEARCH_CACHE_FILE="${SEARCH_CACHE_FILE:-search_cache_echo_3B.json}"

bash "${SCRIPT_DIR}/train.sh" "${BASE_PROFILE}" \
    project_name="${PROJECT_NAME}" \
    experiment_name="${EXPERIMENT_NAME}" \
    actor_model_subpath="${ACTOR_MODEL_SUBPATH}" \
    search_cache_file="${SEARCH_CACHE_FILE}" \
    "$@"
