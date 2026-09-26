set -euo pipefail

cd /scratch/user/saratb_tamu.edu/research/arpo-lens/ECHO/training

export EXPERIMENT_NAME=echo7B_sft5e8_aho_k8
export BASE_PROFILE=training_config/config_aho_k8.yaml

echo "[${SLURM_JOB_ID}] ${EXPERIMENT_NAME} on $(hostname) $(date -Is)"
nvidia-smi -L

exec bash scripts/train_qwen7B_v2.sh
