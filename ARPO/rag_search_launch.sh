#!/usr/bin/env bash
# Sidecar launcher for the ECHO search-cache RAG server.
# Mirrors rag_launch.sh; loads search_cache_echo_3B.json keys into a FAISS
# IndexFlatIP and serves /retrieve + /add on $PORT.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# === Knobs (override via env) =================================================
# Path to the dict[str,str] cache the trainer uses; we index its keys.
CORPUS_PATH="${CORPUS_PATH:-${SCRIPT_DIR}/search_cache/search_cache_echo_3B.json}"
# E5-base-v2 (matches retrieval_server.py reference; HF auto-download on first run).
RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
# Encoder family name; controls "query: "/"passage: " prefixing inside Encoder.
RETRIEVAL_METHOD="${RETRIEVAL_METHOD:-e5}"
# Bind to loopback for single-node ECHO; switch to 0.0.0.0 only if multi-node.
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5003}"
# Encoder batch size for the one-time corpus pass at startup (~123k entries).
BATCH_SIZE="${BATCH_SIZE:-512}"
# Max tokens per encoded string.
MAX_LENGTH="${MAX_LENGTH:-256}"

nvcc --version || true

python3 "${SCRIPT_DIR}/rag_search_server.py" \
    --corpus_path "${CORPUS_PATH}" \
    --retriever_model "${RETRIEVER_MODEL}" \
    --retrieval_method "${RETRIEVAL_METHOD}" \
    --batch_size "${BATCH_SIZE}" \
    --max_length "${MAX_LENGTH}" \
    --host "${HOST}" \
    --port "${PORT}"
