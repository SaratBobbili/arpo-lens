#!/bin/bash
# Variant of run_math_4qa_hf_single_job.sh that targets the "deterministic
# pass@1" setup used by most paper tables: greedy decoding, single turn.
#
# Knobs overridden relative to the baseline:
#   TEMPERATURE  0.6 -> 0.0   # greedy decoding (no sampling noise across turns)
#   TURNS        "1 2 3" -> "1"  # only one rollout per sample, since greedy
#                                 # makes turns 2/3 identical anyway
#
# All other defaults (max_tokens, sample_timeout, prompt_type, tool budgets)
# are inherited from run_math_4qa_hf_single_job.sh / echo_infer_math_4qa_hf.sh.
#
# RUN_TAG causes outputs + logs to land under a *_greedy suffix so this run
# does not clobber the baseline outputs.

set -euo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

export RUN_TAG="${RUN_TAG:-greedy}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export TURNS="${TURNS:-1}"

exec bash run_math_4qa_hf_single_job.sh
