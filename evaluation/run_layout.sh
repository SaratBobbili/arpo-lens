#!/bin/bash
# Shared helper for run_*_math_4qa_hf_single_job.sh.
#
# Produces a flat, collision-free OUTPUT_PATH per invocation and persists every
# knob into OUTPUT_PATH/run_config.yaml so the folder alone fully identifies
# (a) which checkpoint the run loaded, (b) which training recipe produced it,
# and (c) the exact inference/judge settings used. Drivers no longer need to
# pack this info into path segments.
#
# Expected parent-script variables (read, never written):
#   RAW_ACTOR_CHECKPOINT_PATH  VERL actor shards (<root>/<experiment>/<step>/actor).
#                              Empty => base HF eval (no training run to attribute).
#   ACTOR_MODEL_PATH           Merged HF directory or HF repo id served by vLLM.
#   REASON_BASE_MODEL_PATH     Base HF model used as the merge template.
#   REASON_MODEL_NAME          Served alias for reasoning endpoints.
#   TRAINING_RECIPE_PATH       Optional pointer to the recipe .sh that trained
#                              this checkpoint (for full reproducibility).
#   INFER_MODE, PROMPT_TYPE, DATASET_GROUP, COUNTS, TURNS, TEMPERATURE,
#   MAX_TOKENS, TOP_P, TOP_K, MIN_P, REPETITION_PENALTY, SAMPLE_TIMEOUT,
#   MAX_PYTHON_TIMES, MAX_SEARCH_TIMES,
#   ECHO_SYSTEM_PROMPT_YAML, ECHO_ACTIVE_SYSTEM_PROMPT, ECHO_TOOL_CALL_LIMIT,
#   ECHO_VALIDATOR_PROFILE,
#   USE_LLM, JUDGE_MODEL_PATH, JUDGE_MODEL_NAME, API_BASE_URL,
#   NEEDS_SUMM, SUMM_MODEL_PATH, SUMM_MODEL_NAME
#
# Optional inputs:
#   RESUME_RUN_DIR  Reuse this exact directory instead of minting a new RUN_ID;
#                   required when RESUME_FROM_EVAL=true so the eval stage
#                   overwrites the same run's metrics files.
#
# Outputs set on the parent shell:
#   TRAINING_EXPERIMENT  Basename of the training experiment (e.g. echo7BInstruct),
#                        or "base" for a non-trained HF model.
#   CHECKPOINT_STEP      Basename of the checkpoint step (e.g. global_step_35),
#                        or a slugified model id for base HF evals.
#   RUN_ID               <date>_<pid> (or the basename of RESUME_RUN_DIR on resume).
#   OUTPUT_PATH          outputs/hf_math_4qa/<experiment>/<step>/<run_id>.
#   RUN_LOG_DIR          OUTPUT_PATH/logs; drivers tee infer/eval logs here so
#                        a run folder is self-contained.

_derive_source_tag() {
    # Trained checkpoint layout is <root>/<experiment>/<step>/actor, so experiment
    # and step are the two parents of the raw path.
    if [[ -n "${RAW_ACTOR_CHECKPOINT_PATH:-}" ]]; then
        CHECKPOINT_STEP="$(basename "$(dirname "$RAW_ACTOR_CHECKPOINT_PATH")")"
        TRAINING_EXPERIMENT="$(basename "$(dirname "$(dirname "$RAW_ACTOR_CHECKPOINT_PATH")")")"
    else
        # Base HF eval: slugify the repo id / local dir so it lives in a single
        # path segment alongside the "base" experiment sentinel.
        TRAINING_EXPERIMENT="base"
        CHECKPOINT_STEP="${ACTOR_MODEL_PATH//\//__}"
    fi
}

_init_output_path() {
    if [[ -n "${RESUME_RUN_DIR:-}" ]]; then
        # Resume mode: keep writing into the user-supplied directory so the eval
        # stage overwrites metrics alongside the original inference outputs.
        OUTPUT_PATH="${RESUME_RUN_DIR%/}"
        RUN_ID="$(basename "$OUTPUT_PATH")"
    else
        # PID suffix guards against two launches in the same wall-clock second.
        RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
        OUTPUT_PATH="outputs/hf_math_4qa/${TRAINING_EXPERIMENT}/${CHECKPOINT_STEP}/${RUN_ID}"
    fi
    RUN_LOG_DIR="${OUTPUT_PATH}/logs"
    mkdir -p "$RUN_LOG_DIR"
}

_write_run_config() {
    local cfg="${OUTPUT_PATH}/run_config.yaml"
    cat > "$cfg" <<EOF
run_id: "${RUN_ID}"
timestamp: "$(date -Iseconds)"
driver_script: "$(basename "$0")"

source:
  training_experiment: "${TRAINING_EXPERIMENT}"
  checkpoint_step: "${CHECKPOINT_STEP}"
  raw_checkpoint_path: "${RAW_ACTOR_CHECKPOINT_PATH:-}"
  hf_checkpoint_path: "${ACTOR_MODEL_PATH:-}"
  base_model: "${REASON_BASE_MODEL_PATH:-}"
  served_model_name: "${REASON_MODEL_NAME:-}"
  training_recipe_path: "${TRAINING_RECIPE_PATH:-}"

infer:
  mode: "${INFER_MODE:-}"
  prompt_type: "${PROMPT_TYPE:-}"
  dataset_group: "${DATASET_GROUP:-}"
  counts: ${COUNTS:-0}
  turns: "${TURNS:-1}"
  temperature: ${TEMPERATURE:-0}
  max_tokens: ${MAX_TOKENS:-0}
  top_p: ${TOP_P:-null}
  top_k: ${TOP_K:-null}
  min_p: ${MIN_P:-null}
  repetition_penalty: ${REPETITION_PENALTY:-null}
  sample_timeout: ${SAMPLE_TIMEOUT:-0}
  max_python_times: ${MAX_PYTHON_TIMES:-0}
  max_search_times: ${MAX_SEARCH_TIMES:-0}

echo:
  system_prompt_yaml: "${ECHO_SYSTEM_PROMPT_YAML:-}"
  active_system_prompt: "${ECHO_ACTIVE_SYSTEM_PROMPT:-}"
  tool_call_limit: "${ECHO_TOOL_CALL_LIMIT:-}"
  validator_profile: "${ECHO_VALIDATOR_PROFILE:-}"

judge:
  enabled: ${USE_LLM:-false}
  model_path: "${JUDGE_MODEL_PATH:-}"
  model_name: "${JUDGE_MODEL_NAME:-}"
  api_base_url: "${API_BASE_URL:-}"

summ:
  required: ${NEEDS_SUMM:-false}
  model_path: "${SUMM_MODEL_PATH:-}"
  model_name: "${SUMM_MODEL_NAME:-}"
EOF
    echo "Run config: $cfg"
}

# Single entry point: call once after every driver variable is set.
init_run_layout() {
    _derive_source_tag
    _init_output_path
    _write_run_config
}
