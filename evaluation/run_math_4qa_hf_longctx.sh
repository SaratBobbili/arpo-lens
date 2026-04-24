#!/bin/bash
# Variant of run_math_4qa_hf_single_job.sh that targets the "raise context"
# hypothesis: give the model more time and more tokens per rollout so long
# AIME/MATH traces no longer get truncated or timed out.
#
# Knobs overridden relative to the baseline:
#   MAX_TOKENS      4096 -> 8192   # doubles the per-call generation budget
#   SAMPLE_TIMEOUT  900  -> 1800   # doubles wall-clock budget per sample
#
# Other defaults (turns, temperature, prompt_type, tool budgets) are inherited
# from run_math_4qa_hf_single_job.sh / echo_infer_math_4qa_hf.sh so we only
# change the thing we want to test.
#
# RUN_TAG causes outputs + logs to land under a *_longctx suffix so this run
# does not clobber the baseline outputs.

set -euo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

export RUN_TAG="${RUN_TAG:-longctx}"
export MAX_TOKENS="${MAX_TOKENS:-8192}"
export SAMPLE_TIMEOUT="${SAMPLE_TIMEOUT:-1800}"

exec bash run_math_4qa_hf_single_job.sh
