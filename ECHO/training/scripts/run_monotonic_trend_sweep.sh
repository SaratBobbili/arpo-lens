#!/usr/bin/env bash
# ============================================================================
# run_monotonic_trend_sweep.sh
# ----------------------------------------------------------------------------
# Sweep `ECHO/training/analysis/plot_monotonic_trend.py` over a list of candidate x-metrics
# for ONE training run.log. For each x, the python script picks the longest
# chronological step subseq along which (x, Δy¹, Δy², …) are jointly monotone
# in DIRECTION; bash sorts by k descending so the strongest co-mover floats
# to the top of `sweep_summary.tsv`.
#
# Auto-detected layouts (no profile flag — point RUN_DIR and go):
#   ECHO 2-phase    : run.log has both `low_level/` and `high_level/` keys
#                     → y = (LL entropy reward, HL F1)
#   single-phase HL : only `high_level/` keys (e.g. ARPO trainer)
#                     → y = (high_level/reward/f1_mean)
#   single-phase    : bare `actor/` / `critic/` keys (e.g. GRPO)
#                     → y = (critic/rewards/mean)
# Defaults are picked accordingly. Override anything via env vars below.
#
# Usage
# -----
# ECHO sweep:
#   bash ECHO/training/scripts/run_monotonic_trend_sweep.sh
#
# Any non-ECHO checkpoint — script auto-detects the layout:
#   RUN_DIR=/scratch/.../ECHO/checkpoints/<experiment_name> \
#       bash ECHO/training/scripts/run_monotonic_trend_sweep.sh
#
# Override y-metrics manually (e.g. ECHO with raw HL score, not F1):
#   Y_METRICS=("LL:low_level/reward/entropy_scalar_mean" "HL:high_level/reward/score_mean") \
#       bash ECHO/training/scripts/run_monotonic_trend_sweep.sh
#
# Override x-candidate list (space-separated single string):
#   X_METRICS_OVERRIDE="high_level/actor/kl_loss high_level/actor/grad_norm" \
#   RUN_DIR=/scratch/.../ECHO/checkpoints/<experiment_name> \
#       bash ECHO/training/scripts/run_monotonic_trend_sweep.sh
#
# Outputs
# -------
# Written under  ${SCRIPT_DIR}/echo_monotone_plots/<RUN_TAG>__<dir>[_strict]/ :
#   <run>_monotone_<x_slug>.png      — per x: scatter of all (x, Δy_i) +
#                                      highlighted joint-monotone subseq
#   <run>_monotone_<x_slug>.csv      — per-step (x, Δy…, in_monotone)
#   sweep_summary.tsv                — ranking row per x (sorted by k)
#   sweep_config.txt                 — env knobs frozen at sweep time
#   run_monotonic_trend_sweep.sh     — frozen copy of this script
# ============================================================================
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

# Direction of joint monotonicity for step selection. Applied jointly across
# x and every Δy, so 'up' = "x co-moves with sustained joint reward improvement".
#   up   = all axes non-decreasing
#   down = all axes non-increasing
#   auto = pick whichever yields the longer subsequence
DIRECTION="${DIRECTION:-up}"

# If non-empty, passes --strict (requires < / > rather than ≤ / ≥). Empty by
# default so plateaus in Δ aren't penalized.
STRICT_FLAG="${STRICT_FLAG:-}"

# Conda python that has matplotlib + pandas. Override if your env lives elsewhere.
PY="${PY:-/scratch/user/saratb_tamu.edu/miniconda3/envs/arpo/bin/python}"

# ============================ Layout auto-detection =========================
# Cheap prefix scan over run.log; one of ll/hl/bare wins.
RUN_LOG="${RUN_DIR}/run.log"
[[ -f "$RUN_LOG" ]] || { echo "no run.log under $RUN_DIR" >&2; exit 2; }
has_ll=$(grep -cE ' - low_level/'  "$RUN_LOG" || true)
has_hl=$(grep -cE ' - high_level/' "$RUN_LOG" || true)
has_actor=$(grep -cE ' - actor/'   "$RUN_LOG" || true)

if (( has_ll > 0 && has_hl > 0 )); then
    LAYOUT="echo"
elif (( has_hl > 0 )); then
    LAYOUT="single_hl"
elif (( has_actor > 0 )); then
    LAYOUT="single_bare"
else
    echo "could not detect metric layout in $RUN_LOG" >&2
    exit 2
fi

# ============================ Layout-driven defaults ========================
# Y_METRICS  : array of "LABEL:metric_name"; one entry per phase panel.
# X_METRICS  : array of bare metric names; one PNG/CSV per entry.
# Both are env-overridable so a single layout can be tweaked without editing
# the script (e.g. drop one x candidate).

if [[ -z "${Y_METRICS:-}" ]]; then
    case "$LAYOUT" in
        echo)
            # ECHO dual-phase: LL pre-gate entropy reward (the channel feeding
            # the LL critic) + HL F1 (the actual task reward).
            Y_METRICS=(
                "LL:low_level/reward/entropy_scalar_mean"
                "HL:high_level/reward/f1_mean"
            )
            ;;
        single_hl)
            # ARPO and similar: single-phase but uses the HL prefix from the
            # verl base trainer; reward is f1_mean as in ECHO HL.
            Y_METRICS=("HL:high_level/reward/f1_mean")
            ;;
        single_bare)
            # GRPO and similar: unprefixed actor/critic keys.
            Y_METRICS=("actor:critic/rewards/mean")
            ;;
    esac
fi

if [[ -z "${X_METRICS_OVERRIDE:-}" ]]; then
    # Note on aliases: deliberately exclude metrics bit-identical to a y-level
    # (e.g. low_level/critic/rewards/mean ≡ low_level/reward/entropy_scalar_mean
    # for ECHO; critic/rewards/mean ≡ critic/score/mean for bare). Including
    # those would make the joint-monotone selection partly tautological. The
    # python script also prints a warning if |ρ(x, y_level)| > 0.95.
    case "$LAYOUT" in
        echo)
            X_METRICS=(
                # --- policy / actor diagnostics --------------------------------------
                "low_level/actor/entropy_loss"           # LL policy entropy after the LL update (HL has no entropy_loss field; only LL phase tracks policy entropy in this trainer)
                "low_level/actor/grad_norm"              # LL pre-step gradient norm
                "high_level/actor/grad_norm"             # HL pre-step gradient norm
                "low_level/actor/pg_loss"                # LL PPO clipped policy-gradient loss
                "high_level/actor/pg_loss"               # HL PPO clipped policy-gradient loss
                "low_level/actor/kl_loss"                # LL KL-to-reference (low_var_kl); proxy for how far policy drifted from ref this step
                "high_level/actor/kl_loss"               # HL KL-to-reference
                # --- reward / format diagnostics --------------------------------------
                "high_level/reward/format_pass_rate"     # HL valid-format share (the "valid format rate" axis)
                "high_level/reward/bad_format_rate"      # HL bad-format share (= 1 - format_pass_rate up to soft fails)
                "low_level/reward/bad_format_rate"       # LL bad-format share
                "low_level/reward/no_tool_rate"          # LL no-tool-call share (signal of degenerate policy)
                # --- response-length / behaviour diagnostics --------------------------
                "low_level/response_length/mean"         # mean LL response length (proxy for tool-call density)
                "high_level/response_length/mean"        # mean HL response length
            )
            ;;
        single_hl)
            X_METRICS=(
                # All under the HL prefix this trainer uses for its single phase.
                "high_level/actor/entropy_loss"          # policy entropy after the actor update
                "high_level/actor/grad_norm"             # pre-step gradient norm
                "high_level/actor/pg_loss"               # PPO clipped policy-gradient loss
                "high_level/actor/kl_loss"               # KL-to-reference (low_var_kl)
                "high_level/actor/ppo_kl"                # ratio-based PPO KL
                "high_level/actor/pg_clipfrac"           # share of tokens hit by the PPO clip
                "high_level/reward/format_pass_rate"     # valid-format share
                "high_level/reward/no_tool_rate"         # no-tool-call share
                "high_level/response_length/mean"        # mean response length
            )
            ;;
        single_bare)
            X_METRICS=(
                "actor/entropy_loss"
                "actor/grad_norm"
                "actor/pg_loss"
                "actor/kl_loss"
                "actor/ppo_kl"
                "actor/pg_clipfrac"
                "response_length/mean"
                "response_length/clip_ratio"
            )
            ;;
    esac
else
    # X_METRICS_OVERRIDE is a space-separated single string.
    read -ra X_METRICS <<< "$X_METRICS_OVERRIDE"
fi

# Output root encodes everything that affects the result so reruns with
# different knobs land in sibling folders (no silent overwrites).
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
    echo "LAYOUT=$LAYOUT       # auto-detected (echo / single_hl / single_bare)"
    echo "DIRECTION=$DIRECTION"
    echo "STRICT_FLAG=$STRICT_FLAG"
    echo "Y_METRICS=(${Y_METRICS[*]})"
    echo "X_METRICS=(${X_METRICS[*]})"
} > "${OUTPUT_DIR}/sweep_config.txt"
cp "$SCRIPT_PATH" "${OUTPUT_DIR}/run_monotonic_trend_sweep.sh"

# Truncate the summary so reruns don't accumulate stale rankings.
SUMMARY_FILE="${OUTPUT_DIR}/sweep_summary.tsv"
rm -f "$SUMMARY_FILE"

echo "Sweeping ${#X_METRICS[@]} x-metrics for ${RUN_TAG} [layout=${LAYOUT}, direction=${DIRECTION}] -> ${OUTPUT_DIR}"
echo "  y-metrics: ${Y_METRICS[*]}"
n_ok=0; n_fail=0
for x in "${X_METRICS[@]}"; do
    echo "--- x-metric: ${x} ---"
    # `|| rc=$?` prevents a single missing metric (or any non-zero exit) from
    # aborting the whole sweep under `set -e`; we just count failures and move on.
    rc=0
    "$PY" "${SCRIPT_DIR}/../analysis/plot_monotonic_trend.py" \
        --run-dir "$RUN_DIR" \
        --x-metric "$x" \
        --y-metrics "${Y_METRICS[@]}" \
        --direction "$DIRECTION" \
        ${STRICT_FLAG} \
        --output-dir "$OUTPUT_DIR" \
        --summary-file "$SUMMARY_FILE" || rc=$?
    if [[ $rc -eq 0 ]]; then n_ok=$((n_ok+1)); else n_fail=$((n_fail+1)); fi
done

echo "done: ${n_ok} ok, ${n_fail} skipped -> ${OUTPUT_DIR}"
if [[ -f "$SUMMARY_FILE" ]]; then
    echo
    echo "=== ranking by k (longest joint-monotone-${DIRECTION} subseq across (x, Δy…)) ==="
    # Header pass-through, then numeric sort on column 3 (k) descending.
    { head -n1 "$SUMMARY_FILE"; tail -n +2 "$SUMMARY_FILE" | sort -t$'\t' -k3,3nr; } \
        | column -ts $'\t'
fi
