#!/bin/bash
set -euo pipefail

# Serves the LLM-as-judge model on a single port (default 8001).
# Only invoked by the orchestrator when USE_LLM=true.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
mkdir -p logs

# HuggingFace id or local checkpoint path of the judge model.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4}"

# Served model alias; must match --model_name passed to evaluate.py.
MODEL_NAME="${MODEL_NAME:-Qwen2.5-72B-Instruct}"

# Maximum context length exposed by the judge endpoint.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

# Fraction of each GPU's memory reserved for vLLM.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"

# vLLM quantization flag; default matches the GPTQ-Int4 release.
QUANTIZATION="${QUANTIZATION:-gptq}"

# Tensor-parallel degree across CUDA_DEVICES (must equal the device count).
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"

# GPUs visible to vLLM; pinned to 0-3 so summarization GPUs can be reused.
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"

# TCP port for the judge endpoint; orchestrator points API_BASE_URL here.
PORT="${PORT:-8001}"

echo "Serving judge model: $MODEL_PATH"
echo "Served model name: $MODEL_NAME"

echo "Starting judge instance on GPU $CUDA_DEVICES -> :$PORT"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" nohup vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --quantization "$QUANTIZATION" \
  --port "$PORT" > "logs/judge_${PORT}.log" 2>&1 &
PID=$!

echo "Judge server started. PID: $PID"
trap "kill $PID" SIGTERM SIGINT
wait $PID
