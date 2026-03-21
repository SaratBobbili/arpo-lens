#!/bin/bash
#SBATCH -J superpod 
#SBATCH --partition=2xlong
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH -t 2-00:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err
#SBATCH --mem=200G
 
set -euo pipefail
echo "Job $SLURM_JOB_ID on $(hostname) ($(date))"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# (Optional) activate your env/modules here
# module load cuda/12.1
# source ~/.bashrc && conda activate verl-tool-env

#pip install flash_attn==2.7.4.post1 --no-build-isolation
bash ARPO/scripts/ARPO_1_5B_Reasoning_1node.sh


