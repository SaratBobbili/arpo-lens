#!/bin/bash
#SBATCH -J exact_k8
#SBATCH -p def
#SBATCH -A prj-02-llm-reasoning-shakkottai
#SBATCH -q standard
#SBATCH -N 1
#SBATCH --gres=gpu:H200:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=1857527M
#SBATCH -t 2-00:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err
#
# Exact hypergradient, K = 8.  sbatch scripts/sbatch_exact_k8.sh
#
# Batch rather than an attached shell: `srun --overlap` hands a step only 4 of the node's
# 8 GPUs whatever --gres asks for (cgroup device isolation; CUDA_VISIBLE_DEVICES cannot
# widen it), which halves world_size and breaks the 136-prompt batch the profile derives.
set -euo pipefail

cd /scratch/user/saratb_tamu.edu/research/arpo-lens/ECHO/training

export EXPERIMENT_NAME=echo7B_sft5e8_exact_k8
export BASE_PROFILE=training_config/config_exact_k8.yaml
export SEARCH_CACHE_FILE=search_cache_final_7B_b.json

echo "[${SLURM_JOB_ID}] ${EXPERIMENT_NAME} on $(hostname) $(date -Is)"
nvidia-smi -L

exec bash scripts/train_qwen7B_v2.sh
