# Evaluation

Math / QA HF eval for ECHO and base models (`run_*_math_4qa_hf_single_job.sh`).

## Pass@8 exploration curves

Collect Pass@8 (OR of correctness across turn rollouts) at several temperatures, then plot Temperature vs Pass@8 Rate with one subplot per dataset. Legend series are `training_experiment+prompt_type` from each run’s `run_config.yaml` (e.g. `base+base`, `sft+echo`).

Existing single-temp / single-turn jobs are unchanged. Exploration runs go through the sweep wrapper.

### 1) Temperature sweep

Drivers accept env overrides: `TEMPERATURE`, `TURNS`, `PROMPT_TYPE`. The wrapper sets those per temperature.

```bash
cd evaluation

# Default: run_7B_math_4qa_hf_single_job.sh, temps 0.0 0.2 0.4 0.6 0.8 1.0
# T=0.0 → TURNS=1 (greedy); other temps → TURNS="1 2 3 4 5 6 7 8"
JOB_SCRIPT=./run_7B_math_4qa_hf_single_job.sh PROMPT_TYPE=base \
  ./sweep_pass_at_k_temps.sh

# SFT / echo example (3B driver)
JOB_SCRIPT=./run_3B_math_4qa_hf_single_job.sh PROMPT_TYPE=echo \
  ./sweep_pass_at_k_temps.sh

# Optional overrides
TEMPERATURES="0.0 0.6 1.0" TURNS="1 2 3 4 5 6 7 8" \
  JOB_SCRIPT=./run_7B_math_4qa_hf_single_job.sh ./sweep_pass_at_k_temps.sh
```

Each temperature writes a separate run under `outputs/hf_math_4qa/.../<run_id>/` with `*_output_{k}.json` and `*_metrics.json` per turn.

### 2) Plot

Requires the `evaluation` conda env (`matplotlib`, `pyyaml`, `tqdm`).

```bash
cd evaluation
conda activate evaluation

python plot_pass_at_k.py \
  --root outputs/hf_math_4qa \
  --out outputs/pass_at_k_exploration.png \
  --summary_json outputs/pass_at_k_summary.json

# Optional: restrict to explicit run dirs; metric = auto | llm_equal | math_equal
python plot_pass_at_k.py --run_dirs path/to/run_a path/to/run_b --metric llm_equal
```

Pass@8 for a run/dataset is the mean over samples of “any turn correct” across whatever `*_output_*_metrics.json` files exist (8 after a sampling temp, 1 after greedy T=0).
