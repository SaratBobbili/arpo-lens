#!/bin/bash
# ECHO nested-phase RL on Qwen2.5-7B-Instruct, warm-started from the SFT checkpoint trained at
# lr 1e-8 for 1 epoch. NOTE: that 1e-8 is the *SFT* learning rate, not the RL one -- the RL
# actor lr lives in ${BASE_PROFILE} (hl_lr / ll_lr) and is the same for all three scripts.
#
# train_qwen7B{,_v2,_v3}.sh form one controlled sweep: they must differ ONLY in the SFT
# checkpoint (and its matching search cache). Keep every algorithmic knob in ${BASE_PROFILE}.
#   bash train_qwen7B.sh
#   EXPERIMENT_NAME=my_run bash train_qwen7B.sh hl_lr=5e-7 ll_num_iters=40
# Requires BRIGHTDATA_API_KEY in secrets.sh for live search on a cache miss.
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Switch arms for all three scripts at once, e.g.
#   BASE_PROFILE=training_config/config1.yaml EXPERIMENT_NAME=my_run bash train_qwen7B.sh
# config_r3.yaml   = response path off, alternating role-masked GRPO   <-- default
# config1.yaml     = response path on (Algorithm 1 round structure + g_resp)
#
# BASE_PROFILE and EXPERIMENT_NAME must be changed TOGETHER. The default name below
# belongs to the config_r3 arm; pointing it at config1 mislabels the run, and since
# resume_mode is auto, reusing any name resumes that directory instead of starting over.
# That is how the response-off r3 arm came to resume global_step_63 of the response-on
# run on 2026-09-20.
BASE_PROFILE="${BASE_PROFILE:-training_config/config_r3.yaml}"

PROJECT_NAME="${PROJECT_NAME:-qwen25_7B}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-echo7B_sft1e8_rl1e-6-r4}"
# HF weights resolved as ${SFT_ROOT}/${ACTOR_MODEL_SUBPATH} (SFT_ROOT from secrets.sh).
ACTOR_MODEL_SUBPATH="${ACTOR_MODEL_SUBPATH:-checkpoints/Qwen2.5-7B-Instruct-lr1e8-ep1}"
# One cache file per concurrent run, on purpose. The cache is a single JSON file, so pointing
# three simultaneous jobs at one path risks read/write contention and corruption. The three caches
# are near-identical in content; do not "consolidate" them.
SEARCH_CACHE_FILE="${SEARCH_CACHE_FILE:-search_cache_final_3B.json}"

bash "${SCRIPT_DIR}/train.sh" "${BASE_PROFILE}" \
    project_name="${PROJECT_NAME}" \
    experiment_name="${EXPERIMENT_NAME}" \
    actor_model_subpath="${ACTOR_MODEL_SUBPATH}" \
    search_cache_file="${SEARCH_CACHE_FILE}" \
    "$@"
