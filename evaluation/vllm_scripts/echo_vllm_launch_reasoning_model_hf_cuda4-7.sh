#!/bin/bash
set -euo pipefail

# This script serves the main reasoning model on two configurable ports.
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

# Port for the first reasoning replica (GPU 4,5).
REASON_PORT_1="${REASON_PORT_1:-8002}"
# Port for the second reasoning replica (GPU 6,7).
REASON_PORT_2="${REASON_PORT_2:-8003}"

echo "Serving reasoning model: $MODEL_PATH"
echo "Served model name: $MODEL_NAME"

echo "Starting reasoning instance on GPU 4,5 -> :$REASON_PORT_1"
CUDA_VISIBLE_DEVICES=4,5 nohup vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor_parallel_size 2 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --port "$REASON_PORT_1" > "logs/reasoning_${REASON_PORT_1}.log" 2>&1 &
PID_1=$!

echo "Starting reasoning instance on GPU 6,7 -> :$REASON_PORT_2"
CUDA_VISIBLE_DEVICES=6,7 nohup vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor_parallel_size 2 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --port "$REASON_PORT_2" > "logs/reasoning_${REASON_PORT_2}.log" 2>&1 &
PID_2=$!

echo "Reasoning servers started. PIDs: $PID_1 $PID_2"
trap "kill $PID_1 $PID_2" SIGTERM SIGINT
wait $PID_1 $PID_2
