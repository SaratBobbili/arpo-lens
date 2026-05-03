SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VERL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARPO_ROOT="$(dirname "$VERL_ROOT")"
REPO_ROOT="$(dirname "$ARPO_ROOT")"
cd "$VERL_ROOT"
echo "Switched to verl root directory: $VERL_ROOT"

export TMPDIR=/tmp/saratb_ray
export RAY_TMPDIR=/tmp/saratb_ray
mkdir -p "$TMPDIR"
# ============================ Environment Setup ============================
# Set basic environment variables
#export PYTHONUNBUFFERED=1
#export HYDRA_FULL_ERROR=1
#export VLLM_ATTENTION_BACKEND=XFORMERS
export VERL_LOGGING_LEVEL=WARN
export RAY_BACKEND_LOG_LEVEL=warning
#export MKL_SERVICE_FORCE_INTEL=1
#export MKL_THREADING_LAYER=GNU
export RAY_memory_usage_threshold=0.8
#export RAY_memory_monitor_refresh_ms=0
export NCCL_DEBUG=WARN
export VLLM_USE_V1=1
# When set, PyTorch runs without Dynamo (no graph capture / torch.compile)
export TORCHDYNAMO_DISABLE=1
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES


# Set Python path
export PYTHONPATH="${VERL_ROOT}:$PYTHONPATH"

# ============================ Basic Configuration ============================
# Experiment name and project
PROJECT_NAME="qwen3B" # Modify experiment group
EXPERIMENT_NAME="echo3B-rerun-hybrid-entropy-1.0-bfp0" # validator profile c1 (plan/reason/answer HL; tool choice + payload LL), phase_order=[low_level, high_level]; LL bad_format_penalty=0 so format pass acts as a positive selection signal instead of a -1 axis that dominates σ_g under GRPO std-norm.

# Configuration file path
CONFIG_PATH="${SCRIPT_DIR}/config" # ECHO recipe config colocated with this launch script
CONFIG_NAME="echo_trainer"

# Distributed training settings
NNODES=1
N_GPUS_PER_NODE=8

# ============================ Data Configuration ============================
# Data parameters
PROMPT_KEY="prompt"                 # Prompt field name
TRAIN_BATCH_SIZE=128                # Training batch size
PPO_MINI_BATCH_SIZE=16              # PPO mini-batch size
MAX_PROMPT_LENGTH=1536              # Maximum prompt length
MAX_RESPONSE_LENGTH=4096            # Maximum response length

# Data file paths
TRAIN_FILES="${ARPO_ROOT}/rl_datasets/train_10k.parquet" # Modify training data path
VALID_FILES="${ARPO_ROOT}/rl_datasets/valid.parquet" # Modify validation data path

# ============================ Model Configuration ============================
# Actor: HF checkpoint dir (LLaMA-Factory SFT writes under arpo_train_sft/checkpoints/...)
CHECKPOINT_DIR="/scratch/project/prj-02-llm-reasoning-shakkottai/saratb/ECHO/sft"
ACTOR_MODEL_PATH="${CHECKPOINT_DIR}/checkpoints/Qwen2.5-3B-Instruct"

# ============================ Rollout Configuration ==========================
# Rollout settings
ROLLOUT_NAME="vllm"                 # Use vllm engine
ROLLOUT_MODE="sync_echo"            # ECHO rollout mode with hierarchical masks
ROLLOUT_N=16                         # Number of responses generated per sample
HIGH_LEVEL_BUDGET=8                 # Number of rollouts used for high-level masked update
ENABLE_MULTI_TURN=False            # Toggle multi-turn tool interaction in rollout
# ============================ Rollout Tools Configuration ==========================
SEARCH_CACHE_PATH="${ARPO_ROOT}/search_cache/search_cache_echo_3B_v2.json" # Per-variant cache for v1 with phase_order=[low_level, high_level]

# ============================ Reward Model Configuration ==========================
# Reward model settings
REWARD_MANAGER="echo"              # Reward manager type
CUSTOM_REWARD_FUNCTION_PATH="${VERL_ROOT}/verl/utils/reward_score/deep_research_echo.py" # Modify reward function path
CUSTOM_REWARD_FUNCTION_NAME="compute_score"
HIGH_LEVEL_REWARD_STRATEGY="scorer" # High-level phase reward strategy: {scorer, entropy, entropy-hybrid}.
LOW_LEVEL_REWARD_STRATEGY="entropy-hybrid"  # Low-level phase reward strategy: {scorer, entropy, entropy-hybrid}.
# Coefficient on the direct entropy regularizer added to the LL actor loss:
#   L_actor = L_GRPO - LL_ENTROPY_REG_COEFF * mean_{m^LL}(H(pi_theta(.|x))).
# The entropy is the full-vocab entropy of the *current* policy (gradient flows
# through pi_theta), and m^LL = low_level_loss_mask ∩ select_loss_mask under
# `entropy-hybrid` (the same mask used for the entropy reward channel). The
# regularizer is gated on LOW_LEVEL_REWARD_STRATEGY ∈ {entropy, entropy-hybrid}
# in the trainer, so this knob is a no-op for `scorer` / `maxentropy_rl`.
# Units of H tracked by LL_ENTROPY_NORMALIZE (see below): when true, H ∈ [0,1];
# when false, H is in raw nats (≤ log(vocab_size) ≈ 11.93 for Qwen2.5). pg_loss
# under GRPO+token-mean is typically O(1e-2..1e-1), so reg_coeff = 1.0 with
# normalize=false dominates pg_loss by ~30-500x and collapses the policy.
LL_ENTROPY_REG_COEFF=1.0

# ---------- LL entropy reward channel knobs ----------
# All knobs below feed the LL `entropy` reward block in echo_trainer.yaml and are
# only consumed when LOW_LEVEL_REWARD_STRATEGY ∈ {entropy, entropy-hybrid}. They
# shape the per-sample LL reward *before* GRPO mean/std-normalization.
LL_ENTROPY_REDUCTION="mean"        # How per-token H is reduced to a per-sample scalar over m^LL ∩ select_loss_mask: {sum, mean}. `mean` is length-invariant; `sum` introduces mask-length variance into σ_g.
LL_ENTROPY_SCALE=1.0               # Scalar multiplier applied to the reduced entropy before clamp/format_gate. Pure rescaling; under GRPO with norm_adv_by_std_in_grpo=True it is absorbed by σ_g and has no effect on advantages.
LL_ENTROPY_NORMALIZE=true        # If true, divides per-token H by log(vocab_size) → H ∈ [0,1]. Applies to BOTH channels: (a) reward channel (no effect on GRPO advantages — σ_g absorbs the scale — only changes logged reward magnitude); (b) regularizer channel (dp_actor divides H by log(vocab_size) before agg_loss, so LL_ENTROPY_REG_COEFF acts on H ∈ [0,1] instead of raw nats; without this the coefficient is silently ~12x larger than expected for Qwen2.5).
LL_FORMAT_GATE=true                # If true, samples failing scorer phase-local format checks have their entropy scalar replaced with LL_BAD_FORMAT_PENALTY (and LL soft-fail samples zeroed); good-format samples keep their entropy.
LL_BAD_FORMAT_PENALTY=-0.1          # Terminal scalar written on format-gate failure. 0.0 makes format pass a *positive selection* signal (good-format ≥ bad-format) without letting the format axis dominate σ_g; -1.0 (the old default) made σ_g essentially the format-vs-good axis and washed out within-good H spread.

# ============================ Training Configuration ============================
# Training parameters
TOTAL_EPOCHS=2                      # Total training epochs
SAVE_FREQ=5                        # Save frequency
TEST_FREQ=5                        # Test frequency

# ============================ Path Configuration ============================
# Save path
CHECKPOINT_DIR="/scratch/project/prj-02-llm-reasoning-shakkottai/saratb/ECHO"
SAVE_PATH="${CHECKPOINT_DIR}/checkpoints/${EXPERIMENT_NAME}" # Modify save path
ROLLOUT_SAVE_PATH="${SAVE_PATH}/rollout"

# ============================ WandB / API Keys ==============================
# WandB settings
WANDB_API_KEY="0986ce441bdc0e809cd73f235d468fa624518fe8" # Modify your wandb key
SEARCH_CLASS_PATH="verl.workers.agent.tools.search_tool.BingSearchTool"
# Bright Data (third-party Bing SERP used by BingSearchTool -> api.brightdata.com/request).
#BRIGHTDATA_API_KEY="" # Bright Data API token; set manually in terminal before launch
BRIGHTDATA_API_KEY="f75663d4-caf9-432a-baf1-dec27a13625a"
BRIGHTDATA_ZONE="serp_api1"                    # Bright Data SERP zone configured in your Bright Data account
BRIGHTDATA_LOCATION="us"                       # Country code passed to Bing via &cc=<code>; also selects the Bright Data proxy geo. "us" routes through US proxies (faster+more reliable from this cluster than "cn", which periodically returns HTTP 200 with empty body under load).
BRIGHTDATA_TIMEOUT=90                        # Per-HTTP-call read timeout (s) to api.brightdata.com. Brightdata SERP tail latency is ~30-60s+ under concurrent rollout load, so 120 absorbs the tail and avoids spurious retries.
# ============================ Preparation ============================
# Login to WandB (if API key is provided)
if [ "$WANDB_API_KEY" != "" ]; then
    wandb login --relogin $WANDB_API_KEY
    export WANDB_DIR=${SAVE_PATH}
fi

# Create save directory
if [ ! -d "$SAVE_PATH" ]; then
    mkdir -p $SAVE_PATH
fi

# Create rollout save directory
if [ ! -d "$ROLLOUT_SAVE_PATH" ]; then
    mkdir -p $ROLLOUT_SAVE_PATH
fi

# ============================ Start Training ============================
python3 -m recipe.echo.main_echo \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VALID_FILES} \
    data.prompt_key=${PROMPT_KEY} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
    actor_rollout_ref.rollout.mode=${ROLLOUT_MODE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.high_level_budget=${HIGH_LEVEL_BUDGET} \
    actor_rollout_ref.rollout.mask_categories.first_select=high \
    actor_rollout_ref.rollout.mask_categories.select=low \
    actor_rollout_ref.rollout.mask_categories.think=high \
    actor_rollout_ref.rollout.mask_categories.answer=high \
    actor_rollout_ref.rollout.mask_categories.search=low \
    actor_rollout_ref.rollout.mask_categories.python=low \
    actor_rollout_ref.rollout.tools.call_limit=3 \
    actor_rollout_ref.rollout.tools.tool_instances.python.params.conda_path=/scratch/user/saratb_tamu.edu/miniconda3 \
    actor_rollout_ref.rollout.tools.tool_instances.python.params.conda_env=arpo \
    actor_rollout_ref.rollout.tools.tool_instances.search.params.cache_file=${SEARCH_CACHE_PATH} \
    actor_rollout_ref.rollout.tools.tool_instances.search.params.api_key=${BRIGHTDATA_API_KEY} \
    actor_rollout_ref.rollout.tools.tool_instances.search.params.zone=${BRIGHTDATA_ZONE} \
    actor_rollout_ref.rollout.tools.tool_instances.search.params.location=${BRIGHTDATA_LOCATION} \
    actor_rollout_ref.rollout.tools.tool_instances.search.params.request_timeout=${BRIGHTDATA_TIMEOUT} \
    actor_rollout_ref.rollout.tools.tool_instances.search.class_path=${SEARCH_CLASS_PATH} \
    actor_rollout_ref.rollout.multi_turn.enable=${ENABLE_MULTI_TURN} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=${REWARD_MANAGER} \
    'reward_model.phase_order=["low_level", "high_level"]' \
    reward_model.phase_rewards.high_level.strategy=${HIGH_LEVEL_REWARD_STRATEGY} \
    reward_model.phase_rewards.low_level.strategy=${LOW_LEVEL_REWARD_STRATEGY} \
    reward_model.phase_rewards.low_level.entropy.reg_coeff=${LL_ENTROPY_REG_COEFF} \
    reward_model.phase_rewards.low_level.entropy.reduction=${LL_ENTROPY_REDUCTION} \
    reward_model.phase_rewards.low_level.entropy.scale=${LL_ENTROPY_SCALE} \
    reward_model.phase_rewards.low_level.entropy.normalize=${LL_ENTROPY_NORMALIZE} \
    reward_model.phase_rewards.low_level.entropy.format_gate=${LL_FORMAT_GATE} \
    reward_model.phase_rewards.low_level.entropy.bad_format_penalty=${LL_BAD_FORMAT_PENALTY} \
    custom_reward_function.path=${CUSTOM_REWARD_FUNCTION_PATH} \
    custom_reward_function.name=${CUSTOM_REWARD_FUNCTION_NAME} \
    trainer.critic_warmup=0 \
    trainer.logger="[console, wandb]" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.max_actor_ckpt_to_keep=null \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.val_before_train=False \
    trainer.rollout_data_dir=${ROLLOUT_SAVE_PATH} \
    hydra.run.dir=${SAVE_PATH}/outputs 2>&1 | tee ${SAVE_PATH}/run.log
