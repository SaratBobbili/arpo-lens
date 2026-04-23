#!/bin/bash
set -euo pipefail

# This script serves the main reasoning model on ports 8002/8003.
# Keep MODEL_PATH and MODEL_NAME consistent with infer.py defaults.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
mkdir -p logs

# HuggingFace model id or local checkpoint path for the model being evaluated.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"

# API model alias returned by vLLM; must match --default_model in infer script.
MODEL_NAME="${MODEL_NAME:-Qwen2.5-7B-Instruct}"

# Maximum context length exposed by the served model.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

# Fraction of each GPU memory reserved for vLLM.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"

echo "Serving reasoning model: $MODEL_PATH"
echo "Served model name: $MODEL_NAME"

echo "Starting reasoning instance on GPU 4,5 -> :8002"
CUDA_VISIBLE_DEVICES=4,5 nohup vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor_parallel_size 2 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --port 8002 > logs/reasoning_8002.log 2>&1 &
PID_8002=$!

echo "Starting reasoning instance on GPU 6,7 -> :8003"
CUDA_VISIBLE_DEVICES=6,7 nohup vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor_parallel_size 2 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --port 8003 > logs/reasoning_8003.log 2>&1 &
PID_8003=$!

echo "Reasoning servers started. PIDs: $PID_8002 $PID_8003"
trap "kill $PID_8002 $PID_8003" SIGTERM SIGINT
wait $PID_8002 $PID_8003
