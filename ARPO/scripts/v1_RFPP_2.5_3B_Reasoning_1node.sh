SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"
echo "Switched to parent directory: $PARENT_DIR"

# export TMPDIR=/scratch/user/saratb_tamu.edu/tmp
# export RAY_TMPDIR=/scratch/user/saratb_tamu.edu/tmp/ray
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


ABSOLUTE_PATH="/scratch/user/saratb_tamu.edu/research/arpo-lens/ARPO"
REPO_ROOT="$(dirname "${ABSOLUTE_PATH}")"
# Set Python path
export PYTHONPATH="${ABSOLUTE_PATH}"/verl_arpo_entropy:$PYTHONPATH

# ============================ Basic Configuration ============================
# Experiment name and project
PROJECT_NAME="qwen3B" # Modify experiment group
EXPERIMENT_NAME="rfpp" # Modify experiment name


# Configuration file path
CONFIG_PATH="${ABSOLUTE_PATH}/scripts/config" # Modify the absolute path of the config folder, relative path is not recommended
CONFIG_NAME="ppo_trainer.yaml"

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
TRAIN_FILES="${ABSOLUTE_PATH}/rl_datasets/train_10k.parquet" # Modify training data path
VALID_FILES="${ABSOLUTE_PATH}/rl_datasets/valid.parquet" # Modify validation data path

# ============================ Model Configuration ============================
# Actor: HF checkpoint dir (LLaMA-Factory SFT writes under arpo_train_sft/checkpoints/...)
ACTOR_MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"

# ============================ Rollout Configuration ==========================
# Rollout settings
ROLLOUT_NAME="vllm"                 # Use vllm engine
ROLLOUT_MODE="sync_with_tool"       # Synchronous mode with tool support
ROLLOUT_N=16                         # Number of responses generated per sample
INITIAL_ROLLOUTS=16                 # Initial rollout number
BEAM_SIZE=1                        # Beam size
BRANCH_PROBABILITY=0.0             # Branch probability
Entropy_weight=0.0
# ============================ Rollout Tools Configuration ==========================
SEARCH_CACHE_PATH="${ABSOLUTE_PATH}/search_cache/search_cache_rfpp_3B.json" # Modify

# ============================ Reward Model Configuration ==========================
# Reward model settings
REWARD_MANAGER="naive"              # Reward manager type
CUSTOM_REWARD_FUNCTION_PATH="${ABSOLUTE_PATH}/verl_arpo_entropy/verl/utils/reward_score/deep_research.py" # Modify reward function path
CUSTOM_REWARD_FUNCTION_NAME="compute_score"

# ============================ Training Configuration ============================
# Training parameters
TOTAL_EPOCHS=2                      # Total training epochs
SAVE_FREQ=5                        # Save frequency
TEST_FREQ=5                        # Test frequency
MAX_ACTOR_CKPTS_TO_KEEP=1          # Maximum actor checkpoints to retain; 1 keeps only the latest checkpoint

# ============================ Path Configuration ============================
# Save path
CHECKPOINT_DIR="/scratch/project/prj-02-llm-reasoning-shakkottai/saratb/RFPP/Qwen3B-Instruct"
SAVE_PATH="${CHECKPOINT_DIR}/checkpoints/${EXPERIMENT_NAME}" # Modify save path
ROLLOUT_SAVE_PATH="${SAVE_PATH}/rollout"

# ============================ WandB Configuration ============================
# WandB settings
WANDB_API_KEY="0986ce441bdc0e809cd73f235d468fa624518fe8" # Modify your wandb key
SEARCH_CLASS_PATH="verl.workers.agent.tools.search_tool.BingSearchTool"
BRIGHTDATA_API_KEY="9c221824-9a57-4261-b1b7-979959492235" # Bright Data API token; set manually in terminal before launch
BRIGHTDATA_ZONE="serp_api1"                    # Bright Data SERP zone configured in your Bright Data account
BRIGHTDATA_LOCATION="us"                       # Country code passed to Bing via &cc=<code>; also selects the Bright Data proxy geo. "us" routes through US proxies (faster+more reliable from this cluster than "cn", which periodically returns HTTP 200 with empty body under load).
BRIGHTDATA_TIMEOUT=45                        # Per-HTTP-call read timeout (s) to api.brightdata.com. Brightdata SERP tail latency is ~30-60s+ under concurrent rollout load, so 120 absorbs the tail and avoids spurious retries.
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
python3 -m verl.trainer.main_ppo \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    algorithm.adv_estimator=reinforce_plus_plus \
    algorithm.kl_ctrl.kl_coef=0.0 \
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
    actor_rollout_ref.actor.use_kl_loss=False \
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
    actor_rollout_ref.rollout.initial_rollouts=${INITIAL_ROLLOUTS} \
    actor_rollout_ref.rollout.beam_size=${BEAM_SIZE} \
    actor_rollout_ref.rollout.branch_probability=${BRANCH_PROBABILITY} \
    actor_rollout_ref.rollout.entropy_weight=${Entropy_weight} \
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
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPTS_TO_KEEP} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.val_before_train=False \
    trainer.rollout_data_dir=${ROLLOUT_SAVE_PATH} \
    hydra.run.dir=${SAVE_PATH}/outputs 2>&1 | tee ${SAVE_PATH}/run.log