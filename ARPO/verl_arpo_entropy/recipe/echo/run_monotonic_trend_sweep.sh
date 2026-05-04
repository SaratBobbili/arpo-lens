#!/usr/bin/env bash
# ============================================================================
# run_monotonic_trend_sweep.sh
# ----------------------------------------------------------------------------
# Sweep `recipe.echo.plot_monotonic_trend` over a list of candidate x-metrics
# for ONE training run.log. For each x, the python script picks the longest
# chronological step subseq along which (x, Δy¹, Δy², …) are jointly monotone
# in DIRECTION; bash sorts by k descending at the end so the strongest co-mover
# floats to the top.
#
# Profiles
# --------
# Two built-in profiles, selected via the env var `PROFILE`:
#   PROFILE=echo (default)   — ECHO trainer dual-phase: y = (LL entropy reward,
#                              HL F1) and x candidates use `low_level/`,
#                              `high_level/` prefixes.
#   PROFILE=baseline         — single-phase RL (ARPO, GRPO, …): y = critic
#                              reward; x candidates are unprefixed (`actor/...`,
#                              `critic/...`).
# All knobs below are env-overridable; PROFILE just sets sensible defaults
# for X_METRICS / Y_METRICS so a new run typically just needs RUN_DIR.
#
# Usage
# -----
# Default ECHO sweep (the run we pointed at first):
#   bash ARPO/verl_arpo_entropy/recipe/echo/run_monotonic_trend_sweep.sh
#
# ECHO with a different run + direction:
#   RUN_DIR=/scratch/.../checkpoints/echo3B-rerun-hybrid-entropy-1.0-bfp0 \
#   DIRECTION=auto \
#       bash ARPO/verl_arpo_entropy/recipe/echo/run_monotonic_trend_sweep.sh
#
# ARPO/GRPO baseline sweep (single-phase):
#   PROFILE=baseline \
#   RUN_DIR=/scratch/.../GRPO/Qwen3B-Instruct/checkpoints/grpo \
#       bash ARPO/verl_arpo_entropy/recipe/echo/run_monotonic_trend_sweep.sh
#
# Override the y axes manually (e.g. ECHO with raw HL score, not F1):
#   Y_METRICS=("LL:low_level/reward/entropy_scalar_mean" "HL:high_level/reward/score_mean") \
#       bash ARPO/verl_arpo_entropy/recipe/echo/run_monotonic_trend_sweep.sh
#
# Outputs
# -------
# Written under  ${SCRIPT_DIR}/echo_monotone_plots/<RUN_TAG>__<PROFILE>__<dir>[_strict]/ :
#   <run>_monotone_<x_slug>.png      — k-row plot per x-metric
#   <run>_monotone_<x_slug>.csv      — per-step (x, Δy…, in_monotone) table
#   sweep_summary.tsv                — one row per x-metric (sorted ranking)
#   sweep_config.txt                 — env knobs at the time of sweep
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
# Profile-driven defaults; see header for available profiles.
PROFILE="${PROFILE:-echo}"

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

# ============================ Profile defaults =============================
# Y_METRICS  : array of "LABEL:metric_name"; one entry per phase panel.
# X_METRICS  : array of bare metric names (one PNG/CSV per entry).
# Both are exposed as env vars so a single profile can be tweaked without
# editing the script (e.g. drop one x candidate).

if [[ -z "${Y_METRICS:-}" ]]; then
    case "$PROFILE" in
        echo)
            # ECHO dual-phase: LL pre-gate entropy reward (the channel that
            # feeds the LL critic) + HL F1 (the actual task reward).
            Y_METRICS=(
                "LL:low_level/reward/entropy_scalar_mean"
                "HL:high_level/reward/f1_mean"
            )
            ;;
        baseline)
            # Single-phase RL (ARPO, GRPO, …): only one reward channel exists.
            Y_METRICS=(
                "actor:critic/rewards/mean"
            )
            ;;
        *)
            echo "unknown PROFILE=$PROFILE (expected 'echo' or 'baseline'); set Y_METRICS manually." >&2
            exit 2
            ;;
    esac
fi

if [[ -z "${X_METRICS_OVERRIDE:-}" ]]; then
    # Note on aliases: deliberately exclude metrics bit-identical to the y-level
    # (e.g. low_level/critic/rewards/mean ≡ low_level/reward/entropy_scalar_mean
    # for ECHO; critic/rewards/mean ≡ critic/score/mean for baselines). Including
    # those would make the joint-monotone selection partly tautological. The
    # python script also prints a warning if |ρ(x, y_level)| > 0.95.
    case "$PROFILE" in
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
        baseline)
            X_METRICS=(
                # --- policy / actor diagnostics (single-phase) -----------------------
                "actor/entropy_loss"                     # policy entropy after the actor update
                "actor/grad_norm"                        # pre-step gradient norm
                "actor/pg_loss"                          # PPO clipped policy-gradient loss
                "actor/kl_loss"                          # KL-to-reference (low_var_kl)
                "actor/ppo_kl"                           # ratio-based PPO KL
                "actor/pg_clipfrac"                      # share of tokens hit by the PPO clip
                # --- response-length / behaviour diagnostics ---------------------
                "response_length/mean"                   # mean response length
                "response_length/clip_ratio"             # share of responses clipped at max length
            )
            ;;
    esac
else
    # X_METRICS_OVERRIDE is a space-separated list (single env var); split it.
    read -ra X_METRICS <<< "$X_METRICS_OVERRIDE"
fi

# Output root encodes everything that affects the result so reruns with
# different knobs land in sibling folders (no silent overwrites).
RUN_TAG="$(basename "$RUN_DIR")"
EXTRA_TAG="${PROFILE}__${DIRECTION}"
[[ -n "$STRICT_FLAG" ]] && EXTRA_TAG="${EXTRA_TAG}_strict"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/echo_monotone_plots/${RUN_TAG}__${EXTRA_TAG}}"
mkdir -p "$OUTPUT_DIR"

# Snapshot of the invocation (run config + script copy) so re-running with
# different knobs doesn't quietly overwrite history.
{
    echo "# generated $(date -Iseconds)"
    echo "PROFILE=$PROFILE"
    echo "RUN_DIR=$RUN_DIR"
    echo "DIRECTION=$DIRECTION"
    echo "STRICT_FLAG=$STRICT_FLAG"
    echo "Y_METRICS=(${Y_METRICS[*]})"
    echo "X_METRICS=(${X_METRICS[*]})"
} > "${OUTPUT_DIR}/sweep_config.txt"
cp "$SCRIPT_PATH" "${OUTPUT_DIR}/run_monotonic_trend_sweep.sh"

# Truncate the summary so reruns don't accumulate stale rankings.
SUMMARY_FILE="${OUTPUT_DIR}/sweep_summary.tsv"
rm -f "$SUMMARY_FILE"

echo "Sweeping ${#X_METRICS[@]} x-metrics for ${RUN_TAG} [profile=${PROFILE}, direction=${DIRECTION}] -> ${OUTPUT_DIR}"
echo "  y-metrics: ${Y_METRICS[*]}"
n_ok=0; n_fail=0
for x in "${X_METRICS[@]}"; do
    echo "--- x-metric: ${x} ---"
    # `|| rc=$?` prevents a single missing metric (or any non-zero exit) from
    # aborting the whole sweep under `set -e`; we just count failures and move on.
    rc=0
    "$PY" -m recipe.echo.plot_monotonic_trend \
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
