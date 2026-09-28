#!/bin/bash
#SBATCH -J aho_k8
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
# ECHO with the AHO response term, K = 8.  sbatch recipe/aho/sbatch_aho_k8.sh
set -euo pipefail

cd /scratch/user/saratb_tamu.edu/research/arpo-lens/ECHO/training

# ECHO with the AHO response term (arXiv:2607.28849, GRPO variant): Algorithm 1's round
# structure (K=8 follower steps on R_L from a common base, fresh leader batch, one leader
# step, discard), with g_resp computed as one surrogate on the leader batch --
#   reinforce each reasoning token after the first tool call with weight coef * A_H * A_L / tau
# -- instead of the adjoint sweep. One optimizer step per phase iteration (mini = prompt
# batch = 128), as C.3 requires; the follower runs with entropy on (tau = ll_entropy_reg_coeff).
# Everything algorithmic lives in the profile; see recipe/aho/profiles/config_aho_k8.yaml.
export EXPERIMENT_NAME=echo7B_sft1e6_echo_aho
export BASE_PROFILE=recipe/aho/profiles/config_aho_k8.yaml
# Same SFT warm-start as recipe/alt_grpo/sbatch_alt_grpo.sh, so the two arms differ in
# the algorithm only.
# train_qwen7B_v2.sh passes this as actor_model_subpath= (overrides the profile's value).
export ACTOR_MODEL_SUBPATH=checkpoints/Qwen2.5-7B-Instruct-lr1e6-ep3
# Own cache file: the wrapper's default (search_cache_final_7B.json) is shared with other
# arms, and concurrent writers to one JSON file corrupt it. A copy of the 7B cache.
export SEARCH_CACHE_FILE=search_cache_final_7B_aho.json

echo "[${SLURM_JOB_ID:-nojob}] ${EXPERIMENT_NAME} on $(hostname) $(date -Is)"
nvidia-smi -L

exec bash scripts/train_qwen7B_v2.sh
