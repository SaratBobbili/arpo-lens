#!/bin/bash

#SBATCH --job-name=AgentEval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --gpus=8
#SBATCH --time=01:30:00
#SBATCH --qos=standard
#SBATCH --output=logs/slurm_%j.log

# ---------------------------------------------------------------------
# Console Colors & Formatting
# ---------------------------------------------------------------------
BOLD='\033[1m'
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

function print_header() {
    echo -e "\n${BLUE}${BOLD}======================================================================${NC}"
    echo -e "${BLUE}${BOLD}>>> $1${NC}"
    echo -e "${BLUE}${BOLD}======================================================================${NC}\n"
}

function print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

function print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

function print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header "Agent Evaluation Job Started"
print_info "Time: $(date '+%Y-%m-%d %H:%M:%S')"
print_info "Slurm Job ID: ${SLURM_JOB_ID}"

if [ "$#" -lt 1 ]; then
    print_error "Usage: sbatch $0 </path/to/hf/checkpoint> [prompt_type]"
    print_error "  prompt_type: echo (default), code_search, math, base"
    exit 1
fi

# ---------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------
print_info "Loading environment modules..."
ml CUDA/12.9.1
source ~/.bashrc
conda activate treehca_env

# ---------------------------------------------------------------------
# Model & ECHO Config
# ---------------------------------------------------------------------
MODEL_PATH="${1%/}"
PROMPT_TYPE="${2:-echo}"

if [ ! -f "${MODEL_PATH}/config.json" ]; then
    print_error "HF checkpoint not found: ${MODEL_PATH}/config.json"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ECHO_SYSTEM_PROMPT_YAML="${SCRIPT_DIR}/../ECHO/training/config/echo_system_prompts.yaml"
ECHO_ACTIVE_SYSTEM_PROMPT="1"
ECHO_TOOL_CALL_LIMIT="8"

if [ "$PROMPT_TYPE" = "echo" ]; then
    export ECHO_SYSTEM_PROMPT_YAML ECHO_ACTIVE_SYSTEM_PROMPT ECHO_TOOL_CALL_LIMIT
fi

print_info "Model Path: ${MODEL_PATH}"
print_info "Prompt Type: ${PROMPT_TYPE}"

# ---------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------
LOG_DIR="logs/eval_$(basename "$MODEL_PATH")_$(date +'%Y-%m-%d_%H-%M-%S')_${SLURM_JOB_ID}"
mkdir -p "${LOG_DIR}"
print_info "Logging Directory: ${LOG_DIR}"

trap 'print_warn "Cleaning up background processes..."; kill $LR_PID $VLLM_INSTANCE1_PID $VLLM_INSTANCE2_PID $VLLM_INSTANCE3_PID $VLLM_INSTANCE4_PID 2>/dev/null' EXIT

# ---------------------------------------------------------------------
# 1. Launch Background Processes
# ---------------------------------------------------------------------
print_header "Launching Background Processes"

RETRIEVER_PORT=8000
VLLM_PORT1=$((RETRIEVER_PORT + 1))
VLLM_PORT2=$((RETRIEVER_PORT + 2))
VLLM_PORT3=$((RETRIEVER_PORT + 3))
VLLM_PORT4=$((RETRIEVER_PORT + 4))

print_info "Starting Local Retriever (Port: ${RETRIEVER_PORT})"
conda run -n retriever_env \
    python rag_server/retrieval_server.py \
    --index_path rag_data/e5_Flat.index \
    --corpus_path rag_data/wiki-18.jsonl \
    --topk 3 \
    --retriever_model intfloat/e5-base-v2 > "${LOG_DIR}/local_retriever.log" 2>&1 &
LR_PID=$!

print_info "Starting vLLM Instance 1 (GPU 0,1 | Port: ${VLLM_PORT1})"
CUDA_VISIBLE_DEVICES=0,1 nohup vllm serve "$MODEL_PATH" \
    --served-model-name "AgentModel" \
    --max-model-len 32768 \
    --tensor_parallel_size 2 \
    --port ${VLLM_PORT1} > "${LOG_DIR}/vllm_instance_1.log" 2>&1 &
VLLM_INSTANCE1_PID=$!

print_info "Starting vLLM Instance 2 (GPU 2,3 | Port: ${VLLM_PORT2})"
CUDA_VISIBLE_DEVICES=2,3 nohup vllm serve "$MODEL_PATH" \
    --served-model-name "AgentModel" \
    --max-model-len 32768 \
    --tensor_parallel_size 2 \
    --port ${VLLM_PORT2} > "${LOG_DIR}/vllm_instance_2.log" 2>&1 &
VLLM_INSTANCE2_PID=$!

print_info "Starting vLLM Instance 3 (GPU 4,5 | Port: ${VLLM_PORT3})"
CUDA_VISIBLE_DEVICES=4,5 nohup vllm serve "$MODEL_PATH" \
    --served-model-name "AgentModel" \
    --max-model-len 32768 \
    --tensor_parallel_size 2 \
    --port ${VLLM_PORT3} > "${LOG_DIR}/vllm_instance_3.log" 2>&1 &
VLLM_INSTANCE3_PID=$!

print_info "Starting vLLM Instance 4 (GPU 6,7 | Port: ${VLLM_PORT4})"
CUDA_VISIBLE_DEVICES=6,7 nohup vllm serve "$MODEL_PATH" \
    --served-model-name "AgentModel" \
    --max-model-len 32768 \
    --tensor_parallel_size 2 \
    --port ${VLLM_PORT4} > "${LOG_DIR}/vllm_instance_4.log" 2>&1 &
VLLM_INSTANCE4_PID=$!

# ---------------------------------------------------------------------
# 2. Health Check Loop
# ---------------------------------------------------------------------
print_header "Running Health Checks"
print_info "Waiting for services to become healthy..."

VLLM_PIDS=($VLLM_INSTANCE1_PID $VLLM_INSTANCE2_PID $VLLM_INSTANCE3_PID $VLLM_INSTANCE4_PID)
VLLM_PORTS=($VLLM_PORT1 $VLLM_PORT2 $VLLM_PORT3 $VLLM_PORT4)

ELAPSED=0
MAX_WAIT=300
while true; do
    ALL_HEALTHY=true

    for port in "${VLLM_PORTS[@]}"; do
        if ! curl -s --fail http://localhost:${port}/health > /dev/null 2>&1; then
            ALL_HEALTHY=false
            break
        fi
    done

    if ! curl -s --fail http://localhost:${RETRIEVER_PORT}/openapi.json > /dev/null 2>&1; then
        ALL_HEALTHY=false
    fi

    if $ALL_HEALTHY; then
        echo ""
        print_info "Success: All processes are healthy and ready!"
        break
    fi

    for pid in "${VLLM_PIDS[@]}" "$LR_PID"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo ""
            print_error "One of the processes died unexpectedly during startup."
            print_error "Check the log files in ${LOG_DIR} for details."
            exit 1
        fi
    done

    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo ""
        print_error "Timed out waiting for processes to become healthy after ${MAX_WAIT} seconds."
        exit 1
    fi

    echo -en "\r\033[K${YELLOW}[WAIT]${NC} Still initializing background processes... (${ELAPSED}/${MAX_WAIT}s elapsed)"
    sleep 10
    ELAPSED=$((ELAPSED + 10))
done

# ---------------------------------------------------------------------
# 3. Inference
# ---------------------------------------------------------------------
print_header "Starting Inference Script"

infer_endpoints=(
    "http://localhost:${VLLM_PORT1}/v1"
    "http://localhost:${VLLM_PORT2}/v1"
    "http://localhost:${VLLM_PORT3}/v1"
    "http://localhost:${VLLM_PORT4}/v1"
)

data_names=(
    "aime24"
    "aime25"
    "math500"
    "gsm8k"
    "math"
    "hotpotqa"
    "2wiki"
    "musique"
    "bamboogle"
)

python inference.py \
    --endpoints "${infer_endpoints[@]}" \
    --model_path "${MODEL_PATH}" \
    --dataset_name "${data_names[@]}" \
    --prompt_type "${PROMPT_TYPE}" \
    --output_path "${LOG_DIR}" \
    --conda_path "$(conda info --base)" \
    --conda_env "treehca_env" 2>&1 | tee "${LOG_DIR}/inference.log"

print_info "Inference completed."

# ---------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------
print_header "Starting Evaluation Script"

python evaluate.py \
    --output_path "${LOG_DIR}" 2>&1 | tee "${LOG_DIR}/evaluation.log"

print_info "Evaluation completed."
print_header "Job ${SLURM_JOB_ID} Finished Successfully"