# tree_hca_eval -- Usage Notes

## Prerequisites

### Conda Environments

Two conda environments are required:

| Environment | Purpose |
|---|---|
| `treehca_env` | Main env: vLLM serving, inference, evaluation, Python tool execution |
| `retriever_env` | RAG retriever server (FAISS + e5 encoder) |

#### Creating `treehca_env`

```bash
conda create -n treehca_env python=3.10 -y
conda activate treehca_env

pip install vllm
pip install transformers openai torch numpy tqdm nltk pyyaml requests langid
```

#### Creating `retriever_env`

```bash
conda create -n retriever_env python=3.10 -y
conda activate retriever_env

pip install torch transformers faiss-gpu datasets
pip install fastapi uvicorn pydantic
pip install tqdm numpy huggingface_hub
```

### RAG Data

Download the Wikipedia index and corpus before first use:

```bash
cd tree_hca_eval
conda activate retriever_env
python rag_server/download.py
```

This creates `rag_data/e5_Flat.index` and `rag_data/wiki-18.jsonl`.

### Datasets

Test data lives under `data/<dataset_name>/test.jsonl`. The following datasets are expected:

- **Math:** aime24, aime25, math500, gsm8k, math
- **QA:** hotpotqa, 2wiki, musique, bamboogle

### CUDA Module

The script loads `CUDA/12.9.1` via `ml`. Ensure this module is available on your cluster.

---

## Running an Evaluation

### Basic Usage (ECHO checkpoint)

```bash
cd tree_hca_eval
sbatch main.sh /path/to/best_checkpoint/hf
```

This uses the `echo` prompt type by default.

### Specifying Prompt Type

```bash
sbatch main.sh /path/to/checkpoint echo          # ECHO <select>/<tool> format (default)
sbatch main.sh /path/to/checkpoint code_search    # search + python (original Search-o1 style)
sbatch main.sh /path/to/checkpoint math           # python only
sbatch main.sh /path/to/checkpoint base           # no tools (pure CoT)
```

### Example with ECHO Checkpoint

```bash
CKPT="/scratch/project/prj-02-llm-reasoning-shakkottai/saratb/ECHO/checkpoints/echo3BInst_hl_scorer_ll_entropy_sign_cond_clip_true_rollout_reuse_true/best_checkpoint/hf"
sbatch main.sh "$CKPT"
```

---

## ECHO-Specific Configuration

When `prompt_type=echo`, the script exports these environment variables:

| Variable | Default | Description |
|---|---|---|
| `ECHO_SYSTEM_PROMPT_YAML` | `../ARPO/.../echo_system_prompts.yaml` | Path to ECHO system prompt YAML |
| `ECHO_ACTIVE_SYSTEM_PROMPT` | `1` | Which `system_prompt_N` to use from the YAML |
| `ECHO_TOOL_CALL_LIMIT` | `8` | Combined per-sample tool budget across all tools |

To override defaults, edit the variables in `main.sh` under the "Model & ECHO Config" section.

---

## GPU Layout

The script requests **8 GPUs** and launches:

| Instance | GPUs | Port |
|---|---|---|
| RAG Retriever | CPU only | 8000 |
| vLLM Instance 1 | 0, 1 | 8001 |
| vLLM Instance 2 | 2, 3 | 8002 |
| vLLM Instance 3 | 4, 5 | 8003 |
| vLLM Instance 4 | 6, 7 | 8004 |

Each vLLM instance uses tensor parallelism across 2 GPUs.

---

## Output Structure

```
tree_hca_eval/logs/eval_<checkpoint_name>_<timestamp>_<slurm_job_id>/
├── local_retriever.log
├── vllm_instance_{1,2,3,4}.log
├── inference.log
├── evaluation.log
└── outputs/
    ├── aime24.json
    ├── aime25.json
    ├── math500.json
    ├── gsm8k.json
    ├── math.json
    ├── hotpotqa.json
    ├── 2wiki.json
    ├── musique.json
    └── bamboogle.json
```

Each `<dataset>.json` contains per-sample results with fields: `instruction`, `input`, `output`, `prediction`, `answer`, `logs`, `timing`.

The evaluator prints per-dataset metrics (EM, F1, math_equal, tool usage) to stdout and `evaluation.log`.

---

## Customizing Datasets

Edit the `data_names` array in `main.sh` to run a subset:

```bash
data_names=(
    "math500"
    "aime24"
)
```

---

## SLURM Parameters

Default SLURM config in `main.sh`:

```
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --qos=standard
```

Increase `--time` for full dataset runs (all 9 datasets with large sample counts).
