#!/bin/bash

CKPT="/path/to/your/echo/best_checkpoint/hf"        # <-- set this
test -f "$CKPT/config.json" && echo OK || echo "BAD CKPT PATH"

export ECHO_SYSTEM_PROMPT_YAML="$(pwd)/../ARPO/verl_arpo_entropy/recipe/echo/config/echo_system_prompts.yaml"
export ECHO_ACTIVE_SYSTEM_PROMPT=1      # which system_prompt_N (1/2/3)
export ECHO_TOOL_CALL_LIMIT=8           # per-sample combined tool budget

LOG_DIR="logs/eval_$(basename "$CKPT")_math_$(date +%F_%H-%M-%S)"
mkdir -p "$LOG_DIR"

for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((i*2)),$((i*2+1)) nohup vllm serve "$CKPT" \
    --served-model-name "AgentModel" \
    --max-model-len 32768 \
    --tensor_parallel_size 2 \
    --port $((8001+i)) > "$LOG_DIR/vllm_$i.log" 2>&1 &
done

for p in 8001 8002 8003 8004; do
  until curl -s --fail http://localhost:$p/health >/dev/null; do sleep 10; done
  echo "port $p ready"
done

python inference.py \
    --endpoints http://localhost:8001/v1 http://localhost:8002/v1 http://localhost:8003/v1 http://localhost:8004/v1 \
    --model_path "$CKPT" \
    --dataset_name aime24 aime25 math500 gsm8k math \
    --prompt_type echo \
    --output_path "$LOG_DIR" \
    --conda_path "$(conda info --base)" \
    --conda_env evaluation 2>&1 | tee "$LOG_DIR/inference.log"

python evaluate.py --output_path "$LOG_DIR" 2>&1 | tee "$LOG_DIR/evaluation.log"

pkill -f "vllm serve"
