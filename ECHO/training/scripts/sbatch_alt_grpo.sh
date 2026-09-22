set -euo pipefail

cd /scratch/user/saratb_tamu.edu/research/arpo-lens/ECHO/training

# ALTERNATING-GRPO at ARPO's operating point: 128 prompts, mini 16, group 16
# -> 8 optimizer steps per phase iteration. The high-level iteration re-rolls the
# prompts the low-level phase just adapted on (same prompts, fresh rollouts).
export EXPERIMENT_NAME=echo7B_sft5e8_alt_grpo
export BASE_PROFILE=training_config/config_alt_grpo.yaml

echo "[${SLURM_JOB_ID:-nojob}] ${EXPERIMENT_NAME} on $(hostname) $(date -Is)"
nvidia-smi -L

exec bash scripts/train_qwen7B_v2.sh
