#!/bin/bash
# Usage: bash train.sh training_config/<config>.yaml
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
ECHO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ECHO_TOP="$(cd "$ECHO_ROOT/.." && pwd)"
CONFIG_PATH="${ECHO_ROOT}/config"
export VERL_ROOT="/scratch/user/saratb_tamu.edu/research/arpo-lens/ARPO/verl_arpo_entropy"
ARPO_ROOT="$(dirname "$VERL_ROOT")"
REPO_ROOT="$(dirname "$ARPO_ROOT")"
cd "$ECHO_TOP"

export TMPDIR=/tmp/saratb_ray
export RAY_TMPDIR=/tmp/saratb_ray
mkdir -p "$TMPDIR"

export VERL_LOGGING_LEVEL=WARN
export RAY_BACKEND_LOG_LEVEL=warning
export RAY_memory_usage_threshold=0.8
export NCCL_DEBUG=WARN
export VLLM_USE_V1=1
export TORCHDYNAMO_DISABLE=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export PYTHONPATH="${VERL_ROOT}:${ECHO_TOP}:$PYTHONPATH"

source "${SCRIPT_DIR}/secrets.sh"

LAUNCH_CONFIG_PATH="${ECHO_ROOT}/$1"
VALID_LAUNCH_KEYS=(
    project_name experiment_name nnodes n_gpus_per_node
    train_batch_size gen_batch_size ppo_mini_batch_size max_prompt_length max_response_length prompt_key
    active_system_prompt
    train_files valid_files actor_model_subpath reward_manager
    rollout_n high_level_budget low_level_budget reuse_phase_rollouts enable_multi_turn
    tensor_model_parallel_size gpu_memory_utilization rollout_name rollout_mode
    exclude_tag_tokens_from_phase_masks search_cache_file search_class_path brightdata_timeout tool_call_limit
    rag_server_url similarity_threshold topk soft_fallback rag_request_timeout
    conda_path conda_env brightdata_api_key brightdata_zone brightdata_location wandb_api_key
    output_root sft_root
    total_epochs save_freq test_freq save_best_checkpoint best_checkpoint_metric max_actor_ckpt_to_keep resume_mode
    phase_order high_level_update_repeats low_level_update_repeats
    high_level_advantage_algorithm low_level_advantage_algorithm
    norm_adv_by_std_in_grpo
    skip_training_on_tool_failure
    mask_tool mask_think mask_answer mask_search mask_python
    clip_ratio_low clip_ratio_high clip_ratio_c clip_ratio_low_pos clip_ratio_high_pos clip_ratio_low_neg clip_ratio_high_neg
    hl_kl_loss_coef ll_kl_loss_coef hl_use_aepo_clip ll_use_aepo_clip
    high_level_use_sign_cond_clip low_level_use_sign_cond_clip
    hl_entropy_reg_coeff ll_entropy_reg_coeff hl_entropy_normalization ll_entropy_normalization
    hl_entropy_alpha ll_entropy_alpha
    high_level_rollout_strategy low_level_rollout_strategy
    hl_aepo_enable_dynamic_rollouts ll_aepo_enable_dynamic_rollouts
    hl_aepo_initial_rollouts ll_aepo_initial_rollouts
    hl_aepo_beam_size ll_aepo_beam_size
    hl_aepo_branch_probability ll_aepo_branch_probability
    hl_aepo_entropy_weight ll_aepo_entropy_weight
    hl_aepo_initial_entropy_tokens ll_aepo_initial_entropy_tokens
)
python3 -c 'import sys,yaml; cfg=yaml.safe_load(open(sys.argv[1])) or {}; unknown=sorted(set(cfg)-set(sys.argv[2:])); sys.stderr.write("Unknown or unused launch config keys: " + ", ".join(unknown) + "\n") if unknown else None; sys.exit(1 if unknown else 0)' "${LAUNCH_CONFIG_PATH}" "${VALID_LAUNCH_KEYS[@]}"
# Parse YAML config — all keys are uppercased and exported as shell variables
eval "$(python3 -c 'import yaml,sys,shlex;cfg=yaml.safe_load(open(sys.argv[1]));[print(k.upper()+"="+shlex.quote("null" if v is None else "true" if isinstance(v,bool) and v else "false" if isinstance(v,bool) else str(v))) for k,v in cfg.items()]' "${LAUNCH_CONFIG_PATH}")"

# Construct full paths from roots (defined in secrets.sh) + relative paths from config
TRAIN_FILES="${ARPO_ROOT}/${TRAIN_FILES}"
VALID_FILES="${ARPO_ROOT}/${VALID_FILES}"
ACTOR_MODEL_PATH="${SFT_ROOT}/${ACTOR_MODEL_SUBPATH}"
SEARCH_CACHE_PATH="${ECHO_TOP}/search_cache/${SEARCH_CACHE_FILE}"

SAVE_PATH="${OUTPUT_ROOT}/checkpoints/${EXPERIMENT_NAME}"
ROLLOUT_SAVE_PATH="${SAVE_PATH}/rollout"
mkdir -p "${SAVE_PATH}" "${ROLLOUT_SAVE_PATH}"
wandb login --relogin "${WANDB_API_KEY}"
export WANDB_DIR="${SAVE_PATH}"

ARGS=(
    --config-path="${CONFIG_PATH}"
    --config-name=echo_trainer
    algorithm.adv_estimator=grpo
    algorithm.kl_ctrl.kl_coef=0.0
    algorithm.norm_adv_by_std_in_grpo="${NORM_ADV_BY_STD_IN_GRPO:-false}"
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VALID_FILES}"
    data.prompt_key="${PROMPT_KEY}"
    data.active_system_prompt="${ACTIVE_SYSTEM_PROMPT:-1}"
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.gen_batch_size="${GEN_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    actor_rollout_ref.model.path="${ACTOR_MODEL_PATH}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.use_dynamic_bsz=True
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH)))"
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW:-0.2}"
    actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH:-0.2}"
    actor_rollout_ref.actor.clip_ratio_c="${CLIP_RATIO_C:-3.0}"
    actor_rollout_ref.actor.clip_ratio_low_pos="${CLIP_RATIO_LOW_POS:-0.2}"
    actor_rollout_ref.actor.clip_ratio_high_pos="${CLIP_RATIO_HIGH_POS:-0.2}"
    actor_rollout_ref.actor.clip_ratio_low_neg="${CLIP_RATIO_LOW_NEG:-0.2}"
    actor_rollout_ref.actor.clip_ratio_high_neg="${CLIP_RATIO_HIGH_NEG:-0.2}"
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH)))"
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE}"
    actor_rollout_ref.rollout.name="${ROLLOUT_NAME}"
    actor_rollout_ref.rollout.mode="${ROLLOUT_MODE}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}"
    actor_rollout_ref.rollout.n="${ROLLOUT_N}"
    actor_rollout_ref.rollout.high_level_budget="${HIGH_LEVEL_BUDGET}"
    actor_rollout_ref.rollout.low_level_budget="${LOW_LEVEL_BUDGET}"
    actor_rollout_ref.rollout.reuse_phase_rollouts="${REUSE_PHASE_ROLLOUTS:-false}"
    actor_rollout_ref.rollout.exclude_tag_tokens_from_phase_masks="${EXCLUDE_TAG_TOKENS_FROM_PHASE_MASKS:-true}"
    actor_rollout_ref.rollout.multi_turn.enable="${ENABLE_MULTI_TURN}"
    actor_rollout_ref.rollout.tools.call_limit="${TOOL_CALL_LIMIT}"
    actor_rollout_ref.rollout.tools.tool_instances.python.params.conda_path="${CONDA_PATH}"
    actor_rollout_ref.rollout.tools.tool_instances.python.params.conda_env="${CONDA_ENV}"
    actor_rollout_ref.rollout.tools.tool_instances.search.params.cache_file="${SEARCH_CACHE_PATH}"
    actor_rollout_ref.rollout.tools.tool_instances.search.params.api_key="${BRIGHTDATA_API_KEY}"
    actor_rollout_ref.rollout.tools.tool_instances.search.params.zone="${BRIGHTDATA_ZONE}"
    actor_rollout_ref.rollout.tools.tool_instances.search.params.location="${BRIGHTDATA_LOCATION}"
    actor_rollout_ref.rollout.tools.tool_instances.search.params.request_timeout="${BRIGHTDATA_TIMEOUT}"
    actor_rollout_ref.rollout.tools.tool_instances.search.class_path="${SEARCH_CLASS_PATH}"
    actor_rollout_ref.rollout.mask_categories.tool="${MASK_TOOL}"
    actor_rollout_ref.rollout.mask_categories.think="${MASK_THINK}"
    actor_rollout_ref.rollout.mask_categories.answer="${MASK_ANSWER}"
    actor_rollout_ref.rollout.mask_categories.search="${MASK_SEARCH}"
    actor_rollout_ref.rollout.mask_categories.python="${MASK_PYTHON}"
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH)))"
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    reward_model.reward_manager="${REWARD_MANAGER}"
    "reward_model.phase_order=${PHASE_ORDER}"
    "reward_model.phase_update_repeats.high_level=${HIGH_LEVEL_UPDATE_REPEATS:-1}"
    "reward_model.phase_update_repeats.low_level=${LOW_LEVEL_UPDATE_REPEATS:-1}"
    "reward_model.phase_rewards.high_level.advantage_algorithm=${HIGH_LEVEL_ADVANTAGE_ALGORITHM:-grpo}"
    "reward_model.phase_rewards.low_level.advantage_algorithm=${LOW_LEVEL_ADVANTAGE_ALGORITHM:-grpo}"
    actor_rollout_ref.rollout.tools.skip_training_on_tool_failure="${SKIP_TRAINING_ON_TOOL_FAILURE:-false}"
    "reward_model.phase_rewards.high_level.kl_loss_coef=${HL_KL_LOSS_COEF:-0.0}"
    "reward_model.phase_rewards.low_level.kl_loss_coef=${LL_KL_LOSS_COEF:-0.0}"
    reward_model.phase_rewards.high_level.use_aepo_clip="${HL_USE_AEPO_CLIP:-false}"
    reward_model.phase_rewards.low_level.use_aepo_clip="${LL_USE_AEPO_CLIP:-false}"
    reward_model.phase_rewards.high_level.use_sign_cond_clip="${HIGH_LEVEL_USE_SIGN_COND_CLIP:-false}"
    reward_model.phase_rewards.low_level.use_sign_cond_clip="${LOW_LEVEL_USE_SIGN_COND_CLIP:-false}"
    "reward_model.phase_rewards.high_level.entropy.reg_coeff=${HL_ENTROPY_REG_COEFF:-0.0}"
    "reward_model.phase_rewards.low_level.entropy.reg_coeff=${LL_ENTROPY_REG_COEFF:-0.0}"
    "reward_model.phase_rewards.high_level.entropy.normalization=${HL_ENTROPY_NORMALIZATION:-token_pool}"
    "reward_model.phase_rewards.low_level.entropy.normalization=${LL_ENTROPY_NORMALIZATION:-token_pool}"
    "reward_model.phase_rewards.high_level.entropy.alpha=${HL_ENTROPY_ALPHA:-0.2}"
    "reward_model.phase_rewards.low_level.entropy.alpha=${LL_ENTROPY_ALPHA:-0.2}"
    "actor_rollout_ref.rollout.phase_rollouts.high_level.strategy=${HIGH_LEVEL_ROLLOUT_STRATEGY:-default}"
    "actor_rollout_ref.rollout.phase_rollouts.low_level.strategy=${LOW_LEVEL_ROLLOUT_STRATEGY:-default}"
    actor_rollout_ref.rollout.phase_rollouts.high_level.aepo.enable_dynamic_rollouts="${HL_AEPO_ENABLE_DYNAMIC_ROLLOUTS:-false}"
    actor_rollout_ref.rollout.phase_rollouts.low_level.aepo.enable_dynamic_rollouts="${LL_AEPO_ENABLE_DYNAMIC_ROLLOUTS:-false}"
    "actor_rollout_ref.rollout.phase_rollouts.high_level.aepo.initial_rollouts=${HL_AEPO_INITIAL_ROLLOUTS:-8}"
    "actor_rollout_ref.rollout.phase_rollouts.low_level.aepo.initial_rollouts=${LL_AEPO_INITIAL_ROLLOUTS:-8}"
    "actor_rollout_ref.rollout.phase_rollouts.high_level.aepo.beam_size=${HL_AEPO_BEAM_SIZE:-2}"
    "actor_rollout_ref.rollout.phase_rollouts.low_level.aepo.beam_size=${LL_AEPO_BEAM_SIZE:-2}"
    "actor_rollout_ref.rollout.phase_rollouts.high_level.aepo.branch_probability=${HL_AEPO_BRANCH_PROBABILITY:-0.5}"
    "actor_rollout_ref.rollout.phase_rollouts.low_level.aepo.branch_probability=${LL_AEPO_BRANCH_PROBABILITY:-0.5}"
    "actor_rollout_ref.rollout.phase_rollouts.high_level.aepo.entropy_weight=${HL_AEPO_ENTROPY_WEIGHT:-0.2}"
    "actor_rollout_ref.rollout.phase_rollouts.low_level.aepo.entropy_weight=${LL_AEPO_ENTROPY_WEIGHT:-0.2}"
    "actor_rollout_ref.rollout.phase_rollouts.high_level.aepo.initial_entropy_tokens=${HL_AEPO_INITIAL_ENTROPY_TOKENS:-50}"
    "actor_rollout_ref.rollout.phase_rollouts.low_level.aepo.initial_entropy_tokens=${LL_AEPO_INITIAL_ENTROPY_TOKENS:-50}"
    "custom_reward_function.path=${VERL_ROOT}/verl/utils/reward_score/deep_research_echo.py"
    custom_reward_function.name=compute_score
    trainer.critic_warmup=0
    "trainer.logger=[console, wandb]"
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}"
    trainer.nnodes="${NNODES}"
    trainer.save_freq="${SAVE_FREQ}"
    trainer.test_freq="${TEST_FREQ}"
    trainer.save_best_checkpoint="${SAVE_BEST_CHECKPOINT:-false}"
    trainer.best_checkpoint_metric="${BEST_CHECKPOINT_METRIC:-val-core/reward}"
    trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}"
    trainer.total_epochs="${TOTAL_EPOCHS}"
    trainer.default_local_dir="${SAVE_PATH}"
    trainer.val_before_train=False
    trainer.rollout_data_dir="${ROLLOUT_SAVE_PATH}"
    trainer.resume_mode="${RESUME_MODE}"
    "hydra.run.dir=${SAVE_PATH}/outputs"
)

# Snapshot launch artifacts into the checkpoint folder (self-describing runs).
# Hydra also writes the resolved config under ${SAVE_PATH}/outputs/.hydra/.
CONFIG_SNAPSHOT_DIR="${SAVE_PATH}/training_config"
mkdir -p "${CONFIG_SNAPSHOT_DIR}"
cp "${SCRIPT_PATH}" "${CONFIG_SNAPSHOT_DIR}/launch_script.sh"
cp "${LAUNCH_CONFIG_PATH}" "${CONFIG_SNAPSHOT_DIR}/launch_config.yaml"
cp -r "${CONFIG_PATH}" "${CONFIG_SNAPSHOT_DIR}/config"

RAG_SIDECAR_PID=""
cleanup_rag_sidecar() {
    if [[ -n "${RAG_SIDECAR_PID}" ]]; then
        echo "Stopping RAG sidecar (pid=${RAG_SIDECAR_PID})"
        kill "${RAG_SIDECAR_PID}" 2>/dev/null || true
        wait "${RAG_SIDECAR_PID}" 2>/dev/null || true
    fi
}

if [[ "${SEARCH_CLASS_PATH}" == *RagSearchTool ]]; then
    RAG_SERVER_URL="${RAG_SERVER_URL:-http://127.0.0.1:5003}"
    SIMILARITY_THRESHOLD="${SIMILARITY_THRESHOLD:-0.92}"
    TOPK="${TOPK:-1}"
    SOFT_FALLBACK="${SOFT_FALLBACK:-true}"
    RAG_REQUEST_TIMEOUT="${RAG_REQUEST_TIMEOUT:-30}"
    RAG_READY_TIMEOUT="${RAG_READY_TIMEOUT:-3600}"
    RAG_LOG="${SAVE_PATH}/rag_sidecar.log"
    echo "Starting ECHO RAG sidecar → ${RAG_SERVER_URL} (log: ${RAG_LOG})"
    bash "${SCRIPT_DIR}/rag_launch.sh" >"${RAG_LOG}" 2>&1 &
    RAG_SIDECAR_PID=$!
    trap cleanup_rag_sidecar EXIT

    waited=0
    until curl -sf --max-time 3 "${RAG_SERVER_URL}/stats" >/dev/null; do
        if ! kill -0 "${RAG_SIDECAR_PID}" 2>/dev/null; then
            echo "RAG sidecar exited before becoming ready; see ${RAG_LOG}"
            exit 1
        fi
        if (( waited >= RAG_READY_TIMEOUT )); then
            echo "RAG sidecar /stats not healthy within ${RAG_READY_TIMEOUT}s; see ${RAG_LOG}"
            exit 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "RAG sidecar ready: $(curl -sf --max-time 3 "${RAG_SERVER_URL}/stats")"

    ARGS+=(
        "+actor_rollout_ref.rollout.tools.tool_instances.search.params.rag_server_url=${RAG_SERVER_URL}"
        "+actor_rollout_ref.rollout.tools.tool_instances.search.params.similarity_threshold=${SIMILARITY_THRESHOLD}"
        "+actor_rollout_ref.rollout.tools.tool_instances.search.params.topk=${TOPK}"
        "+actor_rollout_ref.rollout.tools.tool_instances.search.params.soft_fallback=${SOFT_FALLBACK}"
        "+actor_rollout_ref.rollout.tools.tool_instances.search.params.rag_request_timeout=${RAG_REQUEST_TIMEOUT}"
    )
fi

printf '%s\n' "${ARGS[@]:2}" > "${CONFIG_SNAPSHOT_DIR}/launch_hydra_overrides.txt"

python3 -m training.main_echo "${ARGS[@]}" 2>&1 | tee "${SAVE_PATH}/run.log"
