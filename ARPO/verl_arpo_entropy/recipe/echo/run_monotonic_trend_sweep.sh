#!/usr/bin/env bash
# Sweep `plot_monotonic_trend.py` over a fixed list of candidate x-metrics for a
# single ECHO run.log. Selected step set is fixed by the (Δ_LL, Δ_HL) joint
# monotone-subseq DP and is independent of x; sweeping x lets us visually
# identify which metric is also monotone over those steps.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Resolve the script path now (absolute) so the post-`cd` snapshot copy works
# regardless of how the user invoked us (relative or absolute path).
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
VERL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$VERL_ROOT"

# ============================ User-tunable knobs ============================
# Run directory containing run.log (the trainer-produced per-step text dump
# parsed by plot_monotonic_trend.py).
RUN_DIR="${RUN_DIR:-/scratch/project/prj-02-llm-reasoning-shakkottai/saratb/ECHO/checkpoints/echo3BInstruct}"

# Direction of joint Δ-reward monotonicity for step selection:
#   up   = "sustained joint improvement" (Δ_LL ↑ ∧ Δ_HL ↑)
#   down = "sustained joint regression"  (Δ_LL ↓ ∧ Δ_HL ↓)
#   auto = pick whichever of the two yields the longer subsequence
DIRECTION="${DIRECTION:-up}"

# If non-empty, passes --strict (requires < / > rather than ≤ / ≥). Empty by
# default so plateaus in Δ aren't penalized.
STRICT_FLAG="${STRICT_FLAG:-}"

# LL/HL reward channels whose Δ defines the y-axes. Override only if you want
# to compare with a different reward (e.g. raw critic reward instead of the
# pre-gate entropy mean).
LL_Y_METRIC="${LL_Y_METRIC:-low_level/reward/entropy_scalar_mean}"
HL_Y_METRIC="${HL_Y_METRIC:-high_level/reward/f1_mean}"

# Conda python that has matplotlib + pandas. Falls back to system python3 if
# unset (the script will fail loudly if those modules are missing).
PY="${PY:-/scratch/user/saratb_tamu.edu/miniconda3/envs/arpo/bin/python}"

# Candidate x-metrics to sweep — one PNG/CSV per entry. Add or remove freely;
# any metric present in run.log works (run with a bogus name to dump the list).
X_METRICS=(
    # --- policy / actor diagnostics --------------------------------------
    "low_level/actor/entropy_loss"           # LL policy entropy after the LL update (HL has no entropy_loss field; only LL phase tracks policy entropy in this trainer)
    "low_level/actor/grad_norm"              # LL pre-step gradient norm
    "high_level/actor/grad_norm"             # HL pre-step gradient norm
    "low_level/actor/pg_loss"                # LL PPO clipped policy-gradient loss
    "high_level/actor/pg_loss"               # HL PPO clipped policy-gradient loss
    # --- reward / format diagnostics --------------------------------------
    "high_level/reward/format_pass_rate"     # HL valid-format share (the "valid format rate" axis)
    "high_level/reward/bad_format_rate"      # HL bad-format share (= 1 - format_pass_rate up to soft fails)
    "high_level/reward/score_mean"           # HL scorer reward mean (raw, pre-Δ)
    "low_level/reward/bad_format_rate"       # LL bad-format share
    "low_level/reward/no_tool_rate"          # LL no-tool-call share (signal of degenerate policy)
    # --- critic / advantage diagnostics -----------------------------------
    "low_level/critic/rewards/mean"          # LL critic reward mean (post-norm)
    "high_level/critic/rewards/mean"         # HL critic reward mean (post-norm)
)

# Output root: <RUN_DIR-basename>__<direction>[_strict] under SCRIPT_DIR.
RUN_TAG="$(basename "$RUN_DIR")"
EXTRA_TAG="${DIRECTION}"
[[ -n "$STRICT_FLAG" ]] && EXTRA_TAG="${EXTRA_TAG}_strict"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/echo_monotone_plots/${RUN_TAG}__${EXTRA_TAG}}"
mkdir -p "$OUTPUT_DIR"

# Snapshot of the invocation (run config + script copy) so re-running with
# different knobs doesn't quietly overwrite history.
{
    echo "# generated $(date -Iseconds)"
    echo "RUN_DIR=$RUN_DIR"
    echo "DIRECTION=$DIRECTION"
    echo "STRICT_FLAG=$STRICT_FLAG"
    echo "LL_Y_METRIC=$LL_Y_METRIC"
    echo "HL_Y_METRIC=$HL_Y_METRIC"
    echo "X_METRICS=(${X_METRICS[*]})"
} > "${OUTPUT_DIR}/sweep_config.txt"
cp "$SCRIPT_PATH" "${OUTPUT_DIR}/run_monotonic_trend_sweep.sh"

echo "Sweeping ${#X_METRICS[@]} x-metrics for ${RUN_TAG} (direction=${DIRECTION}) -> ${OUTPUT_DIR}"
n_ok=0; n_fail=0
for x in "${X_METRICS[@]}"; do
    echo "--- x-metric: ${x} ---"
    # `|| rc=$?` prevents a single missing metric (or any non-zero exit) from
    # aborting the whole sweep under `set -e`; we just count failures and move on.
    rc=0
    "$PY" -m recipe.echo.plot_monotonic_trend \
        --run-dir "$RUN_DIR" \
        --x-metric "$x" \
        --ll-y-metric "$LL_Y_METRIC" \
        --hl-y-metric "$HL_Y_METRIC" \
        --direction "$DIRECTION" \
        ${STRICT_FLAG} \
        --output-dir "$OUTPUT_DIR" || rc=$?
    if [[ $rc -eq 0 ]]; then n_ok=$((n_ok+1)); else n_fail=$((n_fail+1)); fi
done

echo "done: ${n_ok} ok, ${n_fail} skipped -> ${OUTPUT_DIR}"
