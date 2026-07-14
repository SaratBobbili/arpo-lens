We will be listing all our tasks to run the ECHO algorithm

**MAIN STEPS**
1. Modify the SFT training data to https://huggingface.co/datasets/dongguanting/ARPO-SFT-54K,
follow the tool trajectory template we want.


2. SFT-fine-tuning stage.

    The SFT fine-tuning stage should enable trajectory stabilization for the model such that the model selects the tools to be used for generating rollouts after the prompt. Then after each reasoning turn, there will be a tool selection rationale that is generated.

## RAG search (`RagSearchTool`)

Semantic lookup over the unioned Bright Data caches under `ECHO/search_cache/`, with optional Bing soft-fallback. Live sidecar + launch live under **ECHO** (`ECHO/training/rag/`, `ECHO/training/scripts/rag_launch.sh`). Plain `BingSearchTool` YAMLs are unchanged and never start the sidecar.

### Deps (manual)

Install into your training env yourself (no install from `train.sh`): `faiss-gpu` (or `faiss-cpu`), `torch`, `transformers`, `fastapi`, `uvicorn`, `pydantic`, `numpy`, `tqdm`. The encoder (`intfloat/e5-base-v2` by default) needs a CUDA GPU.

### 1) Build the union corpus

```bash
python ECHO/training/build_rag_corpus_union.py
# → ECHO/search_cache/search_cache_union_rag.json
```

Naive `dict.update` over sorted `ECHO/search_cache/*.json` (skips the output file). Re-run after caches grow.

### 2) Train (sidecar auto-started)

```bash
cd ECHO/training/scripts
bash train.sh training_config/echo_3B_ll_hl_rag.yaml
```

When `search_class_path` is `RagSearchTool`, `train.sh` backgrounds `rag_launch.sh`, polls `${rag_server_url}/stats` until healthy (default wait up to `RAG_READY_TIMEOUT=3600`s for encode), then trains, and kills the sidecar on exit. Launch knobs in that YAML: `rag_server_url`, `similarity_threshold`, `topk`, `soft_fallback`, `rag_request_timeout`. Non-RAG configs skip all of this.

Give the sidecar a free GPU if training already saturates devices, e.g. `CUDA_VISIBLE_DEVICES=7 bash train.sh training_config/echo_3B_ll_hl_rag.yaml` (or set the same env around a manual launch below). First encode over the full union can take a long time; watch `${SAVE_PATH}/rag_sidecar.log`.

### 3) Optional: manual sidecar smoke

```bash
# optional: pin a GPU
# export CUDA_VISIBLE_DEVICES=0
bash ECHO/training/scripts/rag_launch.sh
curl -s http://127.0.0.1:5003/stats
```

Env knobs for the launcher: `CORPUS_PATH`, `RETRIEVER_MODEL`, `HOST`, `PORT` (default `5003`), `BATCH_SIZE`, `MAX_LENGTH`.
