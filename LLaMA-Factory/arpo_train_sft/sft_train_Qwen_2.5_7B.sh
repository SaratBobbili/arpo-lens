#!/bin/bash
# ARPO cold-start SFT: dongguanting/ARPO-SFT-54K (see arpo_train_sft/dataset_info/dataset_info.json).
# Per-example system text (tool-use + reasoning format) lives in the dataset `system` column; no echo-specific prompt.
export TMPDIR=/scratch/user/saratb_tamu.edu/tmp
export RAY_TMPDIR=/scratch/user/saratb_tamu.edu/tmp/ray
# Overrides model_name_or_path in the YAML config
MODEL_NAME="Qwen/Qwen2.5-7B"

#================== Basic Configuration ==================#
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # List of visible GPUs
LF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${LF_ROOT}/src:${PYTHONPATH}"

# Disable Weights & Biases
export WANDB_DISABLED=false
export WANDB_API_KEY=0986ce441bdc0e809cd73f235d468fa624518fe8
export WANDB_PROJECT="ECHO_sft"

#================== Training Parameter Configuration ==================#
# Distributed training configuration
NNODES=1                 # Total number of nodes
NODE_RANK=0              # Rank of the current node
PROC_PER_NODE=8          # Number of processes per node
MASTER_ADDR="127.0.0.1"  # Address of the master node
MASTER_PORT=29500        # Port of the master node

# Output directory derived from MODEL_NAME
CHECKPOINT_DIR="/scratch/project/prj-02-llm-reasoning-shakkottai/saratb/ECHO/sft"
OUTPUT_DIR="${CHECKPOINT_DIR}/checkpoints/${MODEL_NAME##*/}/"
# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Resume from latest checkpoint if one exists, otherwise start fresh
RESUME_ARG=""
if ls "${OUTPUT_DIR}"/checkpoint-* 1>/dev/null 2>&1; then
    RESUME_ARG="resume_from_checkpoint=false"
fi

# Path to the training script
TRAIN_SCRIPT="${LF_ROOT}/src/llamafactory/launcher.py"

# Path to the training argument configuration file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ARGS="${SCRIPT_DIR}/yaml/qwen.yaml"

# Command to launch training
torchrun --nnodes ${NNODES} \
         --node_rank ${NODE_RANK} \
         --nproc_per_node ${PROC_PER_NODE} \
         --master_addr ${MASTER_ADDR} \
         --master_port ${MASTER_PORT} \
         ${TRAIN_SCRIPT} \
         ${TRAIN_ARGS} \
         model_name_or_path=${MODEL_NAME} \
         output_dir=${OUTPUT_DIR} \
         run_name=${MODEL_NAME##*/} \
         num_train_epochs=1.0 \
         ${RESUME_ARG} 2>&1 | tee ${OUTPUT_DIR}/training.log

# Optionally enable logging redirection
# exec >> ${OUTPUT_DIR}/training.log 2>&1
