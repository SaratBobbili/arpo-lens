#!/bin/bash
#SBATCH -J alt_grpo
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
# ALTERNATING-GRPO baseline.  sbatch scripts/sbatch_alt_grpo.sh
set -euo pipefail

cd /scratch/user/saratb_tamu.edu/research/arpo-lens/ECHO/training

# ALTERNATING-GRPO at ARPO's operating point: 128 prompts, mini 16, group 16
# -> 8 optimizer steps per phase iteration. The high-level iteration re-rolls the
# prompts the low-level phase just adapted on (same prompts, fresh rollouts).
export EXPERIMENT_NAME=echo7B_sft1e6_alt_grpo_dispo_clips_v0
export BASE_PROFILE=training_config/config_alt_grpo.yaml
# SFT warm-start; train_qwen7B_v2.sh passes this as actor_model_subpath= (overrides YAML).
export ACTOR_MODEL_SUBPATH=checkpoints/Qwen2.5-7B-Instruct-lr1e6-ep3

echo "[${SLURM_JOB_ID:-nojob}] ${EXPERIMENT_NAME} on $(hostname) $(date -Is)"
nvidia-smi -L

exec bash scripts/train_qwen7B_v2.sh
