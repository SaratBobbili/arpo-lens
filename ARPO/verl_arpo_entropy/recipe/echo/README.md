# ECHO Evaluation Harness (ARPO-Style Pipeline)

This README is a standalone runbook for evaluating ECHO checkpoints with the same evaluation pipeline used by ARPO.

## 0) Scope and Expected Inputs

- **Goal**: run tool-augmented inference on benchmark datasets, then compute pass@k-style metrics.
- **Checkpoint input**: an ECHO actor checkpoint directory, typically:
  - `ARPO/checkpoints/<echo_experiment>/global_step_<step>/actor`
- **Pipeline location**: `evaluation/`

## 1) Environments

Use two environments, same as ARPO:

1. **vLLM serving env** (for reasoning/summarization models)
2. **evaluation env** (for inference scripts + scoring)

### 1.1 vLLM environment

```bash
cd evaluation/vllm_scripts
conda create -n vllm_env python=3.10 -y
conda activate vllm_env
pip install -r requirement.txt
```

### 1.2 Evaluation environment

```bash
cd evaluation
conda create -n evaluation python=3.10 -y
conda activate evaluation
pip install -r requirement.txt
```

## 2) Launch vLLM Services

You need:
- 2 reasoning endpoints (`8002`, `8003`)
- 2 summarization endpoints (`8004`, `8005`)

### 2.1 Reasoning model (ECHO checkpoint)

Edit `evaluation/vllm_scripts/echo_vllm_launch_reasoning_model_cuda4-7.sh`:

- `MODEL_PATH`: set to your ECHO actor checkpoint
- `MODEL_NAME`: keep consistent with `DEFAULT_MODEL` in inference script
- `source ...` / `conda activate ...`: set to your local conda path/env

Run:

```bash
cd evaluation/vllm_scripts
bash echo_vllm_launch_reasoning_model_cuda4-7.sh
```

### 2.2 Summarization model

Pick one script:
- `evaluation/vllm_scripts/echo_vllm_launch_summarize_model_cuda0-3_qwen3_8b.sh`
- `evaluation/vllm_scripts/echo_vllm_launch_summarize_model_cuda0-3_qwen3_14b.sh`
- `evaluation/vllm_scripts/echo_vllm_launch_summarize_model_cuda0-3_qwq_32b.sh`

Edit:
- `MODEL_PATH`
- conda activation lines

Run one script:

```bash
cd evaluation/vllm_scripts
bash echo_vllm_launch_summarize_model_cuda0-3_qwen3_14b.sh
```

## 3) Configure ECHO Inference

Edit `evaluation/echo_infer_local_sds.sh` and set the following:

- `data_names`: datasets to evaluate (for example `hle`, `gaia`)
- `EXP_NAME`: experiment tag
- `MODEL_PATH`: same ECHO actor checkpoint as reasoning vLLM
- `OUTPUT_PATH`: output json path (example: `results/echo3b_hle_gaia/output.json`)
- `CONDA_PATH`, `CONDA_ENV`: python tool runtime env
- `BING_API_KEY`, `BING_ZONE`: Bright Data credentials
- `SUMM_MODEL_PATH`: summarization checkpoint path

Keep defaults unless you intentionally tune evaluation behavior:
- `INFER_MODE=completion_sds`
- `with_tools=true`
- `TURNS="1 2 3 4 5"`

Run:

```bash
cd evaluation
bash echo_infer_local_sds.sh
```

Generated files are written under your `OUTPUT_PATH` directory and logs under `evaluation/logs/`.

## 4) Compute Metrics

### 4.1 Start judge model service

Edit `evaluation/echo_deploy_qwen2.5_72B_instruct.sh`:
- set conda activation
- set `your_model_path` for judge model

Run:

```bash
cd evaluation
bash echo_deploy_qwen2.5_72B_instruct.sh
```

### 4.2 Run dataset scoring

Edit `evaluation/echo_evaluate_passk.sh`:
- set `OUTPUT_DIR` to the directory containing `*_output_*.json` files from inference

Run:

```bash
cd evaluation
bash echo_evaluate_passk.sh
```

## 5) Minimal End-to-End Checklist

1. Train ECHO and pick: `ARPO/checkpoints/<exp>/global_step_<step>/actor`
2. Start reasoning vLLM (`8002`, `8003`)
3. Start summarization vLLM (`8004`, `8005`)
4. Run `evaluation/echo_infer_local_sds.sh`
5. Start judge service (`8001`)
6. Run `evaluation/echo_evaluate_passk.sh`

## 6) Common Failure Checks

- **Endpoint mismatch**: `echo_infer_local_sds.sh` endpoints must match launched ports.
- **Model name mismatch**: `DEFAULT_MODEL` should match served model name.
- **Conda tool runtime errors**: verify `CONDA_PATH` and `CONDA_ENV`.
- **Search failures**: verify Bright Data `BING_API_KEY` and `BING_ZONE`.
- **No score output**: `OUTPUT_DIR` in `echo_evaluate_passk.sh` must point to inference outputs.
