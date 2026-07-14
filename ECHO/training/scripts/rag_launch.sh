#!/usr/bin/env bash
# ECHO RAG sidecar: indexes search_cache_union_rag.json keys into FAISS; serves /retrieve /add /stats.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
TRAINING_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"
ECHO_ROOT="$( cd "${TRAINING_DIR}/.." && pwd )"

CORPUS_PATH="${CORPUS_PATH:-${ECHO_ROOT}/search_cache/search_cache_union_rag.json}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
RETRIEVAL_METHOD="${RETRIEVAL_METHOD:-e5}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5003}"
BATCH_SIZE="${BATCH_SIZE:-512}"
MAX_LENGTH="${MAX_LENGTH:-256}"

nvcc --version || true

python3 "${TRAINING_DIR}/rag/server.py" \
    --corpus_path "${CORPUS_PATH}" \
    --retriever_model "${RETRIEVER_MODEL}" \
    --retrieval_method "${RETRIEVAL_METHOD}" \
    --batch_size "${BATCH_SIZE}" \
    --max_length "${MAX_LENGTH}" \
    --host "${HOST}" \
    --port "${PORT}"
