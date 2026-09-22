#!/bin/bash
# Usage: bash train.sh training_config/<config>.yaml [key=value ...]
# Trailing key=value args override the profile; keys must be in VALID_LAUNCH_KEYS.
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

# Login nodes cap nproc at 4096 (limits.d, 2026-09-02) and srun propagates it;
# Ray prestarts one worker per node CPU, so worker threads hit the cap and abort.
ulimit -u "$(ulimit -Hu)"

# stdout is a pipe (tee below), so without this every print() is block-buffered.
export PYTHONUNBUFFERED=1
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
    max_prompt_length max_response_length prompt_key
    active_system_prompt
    train_files valid_files actor_model_subpath reward_manager
    rollout_n enable_multi_turn
    tensor_model_parallel_size gpu_memory_utilization rollout_name rollout_mode
    exclude_tag_tokens_from_phase_masks search_cache_file search_class_path brightdata_timeout tool_call_limit
    rag_server_url similarity_threshold topk soft_fallback rag_request_timeout
    conda_path conda_env brightdata_api_key brightdata_zone brightdata_location wandb_api_key
    output_root sft_root
    save_freq test_freq save_best_checkpoint best_checkpoint_metric max_actor_ckpt_to_keep resume_mode
    checkpoint_contents
    shared_prompt_stream total_epochs
    hl_num_iters ll_num_iters hl_group_size ll_group_size
    hl_ppo_mini_batch_size ll_ppo_mini_batch_size
    hl_ppo_micro_batch_size_per_gpu ll_ppo_micro_batch_size_per_gpu
    hl_lr ll_lr hl_weight_decay ll_weight_decay
    hl_warmup_style ll_warmup_style hl_lr_warmup_steps_ratio ll_lr_warmup_steps_ratio
    high_level_advantage_algorithm low_level_advantage_algorithm
    norm_adv_by_std_in_grpo
    skip_training_on_tool_failure skip_training_on_budget_exhausted budget_exhausted_mode
    mask_tool mask_think mask_answer mask_search mask_python
    clip_ratio_low clip_ratio_high clip_ratio_c clip_ratio_low_pos clip_ratio_high_pos clip_ratio_low_neg clip_ratio_high_neg
    hl_kl_loss_coef ll_kl_loss_coef hl_use_aepo_clip ll_use_aepo_clip
    high_level_use_sign_cond_clip low_level_use_sign_cond_clip
    hl_entropy_reg_coeff ll_entropy_reg_coeff hl_entropy_normalization ll_entropy_normalization
    hl_entropy_alpha ll_entropy_alpha
    hl_entropy_enabled ll_entropy_enabled
    response_enabled response_gradient response_coef response_replay_fraction response_follower_return
    response_exact response_curvature response_group_aligned response_fd_rel hl_optimizer ll_optimizer
    loss_agg_mode
    hl_opefo_enabled ll_opefo_enabled
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

# Per-run overrides from the caller, applied before any path is derived below.
LAUNCH_OVERRIDES=("${@:2}")
for override in "${LAUNCH_OVERRIDES[@]}"; do
    key="${override%%=*}"
    printf '%s\n' "${VALID_LAUNCH_KEYS[@]}" | grep -qxF -- "${key}" || { echo "Unknown launch override key: ${key}" >&2; exit 1; }
    eval "${key^^}=$(printf '%q' "${override#*=}")"
done

# Construct full paths from roots (defined in secrets.sh) + relative paths from config
TRAIN_FILES="${ARPO_ROOT}/${TRAIN_FILES}"
VALID_FILES="${ARPO_ROOT}/${VALID_FILES}"
ACTOR_MODEL_PATH="${SFT_ROOT}/${ACTOR_MODEL_SUBPATH}"
SEARCH_CACHE_PATH="${ECHO_TOP}/search_cache/${SEARCH_CACHE_FILE}"

# An empty EXPERIMENT_NAME collapses SAVE_PATH to ${OUTPUT_ROOT}/checkpoints, dumping a
# run's checkpoints, rollouts and run.log on top of the shared checkpoints root.
[[ -n "${EXPERIMENT_NAME}" && "${EXPERIMENT_NAME}" != "null" ]] || { echo "EXPERIMENT_NAME is empty; refusing to write into ${OUTPUT_ROOT}/checkpoints" >&2; exit 1; }
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
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    actor_rollout_ref.model.path="${ACTOR_MODEL_PATH}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
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
    actor_rollout_ref.rollout.tools.skip_training_on_tool_failure="${SKIP_TRAINING_ON_TOOL_FAILURE:-false}"
    actor_rollout_ref.rollout.tools.skip_training_on_budget_exhausted="${SKIP_TRAINING_ON_BUDGET_EXHAUSTED:-true}"
    actor_rollout_ref.rollout.tools.budget_exhausted_mode="${BUDGET_EXHAUSTED_MODE:-in_group_zero}"
    phases.shared_prompt_stream="${SHARED_PROMPT_STREAM:-true}"
    "phases.high_level.num_iters=${HL_NUM_ITERS}"
    "phases.low_level.num_iters=${LL_NUM_ITERS}"
    "phases.high_level.group_size=${HL_GROUP_SIZE}"
    "phases.low_level.group_size=${LL_GROUP_SIZE}"
    "phases.high_level.ppo_mini_batch_size=${HL_PPO_MINI_BATCH_SIZE}"
    "phases.low_level.ppo_mini_batch_size=${LL_PPO_MINI_BATCH_SIZE}"
    "phases.high_level.ppo_micro_batch_size_per_gpu=${HL_PPO_MICRO_BATCH_SIZE_PER_GPU:-null}"
    "phases.low_level.ppo_micro_batch_size_per_gpu=${LL_PPO_MICRO_BATCH_SIZE_PER_GPU:-null}"
    "phases.high_level.optim.lr=${HL_LR:-1e-6}"
    "phases.low_level.optim.lr=${LL_LR:-1e-6}"
    "phases.high_level.optim.weight_decay=${HL_WEIGHT_DECAY:-0.01}"
    "phases.low_level.optim.weight_decay=${LL_WEIGHT_DECAY:-0.01}"
    "phases.high_level.optim.warmup_style=${HL_WARMUP_STYLE:-constant}"
    "phases.low_level.optim.warmup_style=${LL_WARMUP_STYLE:-constant}"
    "phases.high_level.optim.optimizer=${HL_OPTIMIZER:-adamw}"
    "phases.low_level.optim.optimizer=${LL_OPTIMIZER:-adamw}"
    "phases.high_level.optim.lr_warmup_steps_ratio=${HL_LR_WARMUP_STEPS_RATIO:-0.0}"
    "phases.low_level.optim.lr_warmup_steps_ratio=${LL_LR_WARMUP_STEPS_RATIO:-0.0}"
    "phases.high_level.advantage_algorithm=${HIGH_LEVEL_ADVANTAGE_ALGORITHM:-grpo}"
    "phases.low_level.advantage_algorithm=${LOW_LEVEL_ADVANTAGE_ALGORITHM:-grpo}"
    "phases.high_level.kl_loss_coef=${HL_KL_LOSS_COEF:-0.0}"
    "phases.low_level.kl_loss_coef=${LL_KL_LOSS_COEF:-0.0}"
    phases.high_level.use_aepo_clip="${HL_USE_AEPO_CLIP:-false}"
    phases.low_level.use_aepo_clip="${LL_USE_AEPO_CLIP:-false}"
    phases.high_level.use_sign_cond_clip="${HIGH_LEVEL_USE_SIGN_COND_CLIP:-false}"
    phases.low_level.use_sign_cond_clip="${LOW_LEVEL_USE_SIGN_COND_CLIP:-false}"
    "phases.high_level.entropy.reg_coeff=${HL_ENTROPY_REG_COEFF:-0.0}"
    "phases.low_level.entropy.reg_coeff=${LL_ENTROPY_REG_COEFF:-0.0}"
    "phases.high_level.entropy.normalization=${HL_ENTROPY_NORMALIZATION:-token_pool}"
    "phases.low_level.entropy.normalization=${LL_ENTROPY_NORMALIZATION:-token_pool}"
    "phases.high_level.entropy.alpha=${HL_ENTROPY_ALPHA:-0.2}"
    "phases.low_level.entropy.alpha=${LL_ENTROPY_ALPHA:-0.2}"
    phases.high_level.entropy.enabled="${HL_ENTROPY_ENABLED:-false}"
    phases.low_level.entropy.enabled="${LL_ENTROPY_ENABLED:-false}"
    phases.response.enabled="${RESPONSE_ENABLED:-true}"
    phases.response.gradient="${RESPONSE_GRADIENT:-false}"
    phases.response.follower_return="${RESPONSE_FOLLOWER_RETURN:-null}"
    "actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE:-token-mean}"
    "phases.response.coef=${RESPONSE_COEF:-1.0}"
    "phases.response.replay_fraction=${RESPONSE_REPLAY_FRACTION:-1.0}"
    phases.response.exact="${RESPONSE_EXACT:-false}"
    phases.response.curvature="${RESPONSE_CURVATURE:-false}"
    phases.response.group_aligned="${RESPONSE_GROUP_ALIGNED:-false}"
    "phases.response.fd_rel=${RESPONSE_FD_REL:-2e-2}"
    phases.high_level.opefo.enabled="${HL_OPEFO_ENABLED:-false}"
    phases.low_level.opefo.enabled="${LL_OPEFO_ENABLED:-false}"
    "phases.high_level.rollout.strategy=${HIGH_LEVEL_ROLLOUT_STRATEGY:-default}"
    "phases.low_level.rollout.strategy=${LOW_LEVEL_ROLLOUT_STRATEGY:-default}"
    phases.high_level.rollout.aepo.enable_dynamic_rollouts="${HL_AEPO_ENABLE_DYNAMIC_ROLLOUTS:-false}"
    phases.low_level.rollout.aepo.enable_dynamic_rollouts="${LL_AEPO_ENABLE_DYNAMIC_ROLLOUTS:-false}"
    "phases.high_level.rollout.aepo.initial_rollouts=${HL_AEPO_INITIAL_ROLLOUTS:-8}"
    "phases.low_level.rollout.aepo.initial_rollouts=${LL_AEPO_INITIAL_ROLLOUTS:-8}"
    "phases.high_level.rollout.aepo.beam_size=${HL_AEPO_BEAM_SIZE:-2}"
    "phases.low_level.rollout.aepo.beam_size=${LL_AEPO_BEAM_SIZE:-2}"
    "phases.high_level.rollout.aepo.branch_probability=${HL_AEPO_BRANCH_PROBABILITY:-0.5}"
    "phases.low_level.rollout.aepo.branch_probability=${LL_AEPO_BRANCH_PROBABILITY:-0.5}"
    "phases.high_level.rollout.aepo.entropy_weight=${HL_AEPO_ENTROPY_WEIGHT:-0.2}"
    "phases.low_level.rollout.aepo.entropy_weight=${LL_AEPO_ENTROPY_WEIGHT:-0.2}"
    "phases.high_level.rollout.aepo.initial_entropy_tokens=${HL_AEPO_INITIAL_ENTROPY_TOKENS:-50}"
    "phases.low_level.rollout.aepo.initial_entropy_tokens=${LL_AEPO_INITIAL_ENTROPY_TOKENS:-50}"
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
    trainer.total_epochs="${TOTAL_EPOCHS:-4}"
    trainer.save_best_checkpoint="${SAVE_BEST_CHECKPOINT:-false}"
    trainer.best_checkpoint_metric="${BEST_CHECKPOINT_METRIC:-val-core/reward}"
    trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}"
    "actor_rollout_ref.actor.checkpoint.contents=${CHECKPOINT_CONTENTS:-[model,optimizer,extra]}"
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
printf '%s\n' "${LAUNCH_OVERRIDES[@]}" > "${CONFIG_SNAPSHOT_DIR}/launch_overrides.txt"
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
