#!/bin/bash
set -euo pipefail

# This script serves the SDS summarization helper model on ports 8004/8005.
# Only needed when infer_mode=completion_sds.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
mkdir -p logs

# HuggingFace model id or local checkpoint path used for web-page summarization.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"

# API model alias; must match --summ_model_name in infer script.
MODEL_NAME="${MODEL_NAME:-Qwen2.5-7B-Instruct}"

# Maximum context length exposed by the summarization model.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

# Fraction of each GPU memory reserved for vLLM.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"

echo "Serving summarization model: $MODEL_PATH"
echo "Served model name: $MODEL_NAME"

echo "Starting summarization instance on GPU 0,1 -> :8004"
CUDA_VISIBLE_DEVICES=0,1 nohup vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor_parallel_size 2 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --port 8004 > logs/summarize_8004.log 2>&1 &
PID_8004=$!

echo "Starting summarization instance on GPU 2,3 -> :8005"
CUDA_VISIBLE_DEVICES=2,3 nohup vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor_parallel_size 2 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --port 8005 > logs/summarize_8005.log 2>&1 &
PID_8005=$!

echo "Summarization servers started. PIDs: $PID_8004 $PID_8005"
trap "kill $PID_8004 $PID_8005" SIGTERM SIGINT
wait $PID_8004 $PID_8005
