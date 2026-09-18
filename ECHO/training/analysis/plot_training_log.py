# Copyright 2026 ECHO contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Per-step training-trace plotter for the ECHO recipe.

Reads the JSONL traces produced by `RayECHOTrainer._dump_logging_data` under
`{SAVE_PATH}/logging_data/{low_level,high_level,policy}/<metric>.jsonl` and overlays
all enabled metric series on a single PNG. Each enabled toggle below adds one
line to the figure; flip multiple toggles on simultaneously to compare them on
the same axes (x = global step, y = metric value or step-over-step gain).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

# =============================================================================
# Inputs / outputs (edit these for the run you want to plot).
# =============================================================================

# Root directory written by the trainer (matches `${SAVE_PATH}/logging_data`).
# Point this at "${SAVE_PATH}/logging_data" for the run you want to plot.
LOG_DIR = Path(
    "/scratch/project/prj-02-llm-reasoning-shakkottai/saratb/ECHO/checkpoints/echo7BInstruct_lr5e8/logging_data"
)
# Destination PNG for this invocation. Parent dirs are created if missing.
OUTPUT_PNG = Path("./echo_training_plot.png")

# =============================================================================
# Toggles. True -> the corresponding line is added to the overlaid PNG.
# `*_GAIN` reads the `gain` field (Δ vs previous dumped step); the rest read
# the raw `value` field. Toggle independently of each other.
# =============================================================================

# --- Shared policy health (policy/*) -------------------------------------------
PLOT_POLICY_REWARD = True
PLOT_POLICY_REWARD_GAIN = True
PLOT_POLICY_FORMAT_PENALTY = False
PLOT_POLICY_FORMAT_PENALTY_GAIN = False
PLOT_POLICY_IN_GROUP_REWARD_STD = False
PLOT_POLICY_FORMAT_VALID_RATE = False
PLOT_POLICY_F1_MEAN = False
PLOT_POLICY_NO_TOOL_RATE = False
PLOT_POLICY_ADVANTAGE_STD = False
PLOT_POLICY_PPO_KL = False
PLOT_POLICY_PG_CLIPFRAC = False
PLOT_POLICY_GROUP_ZERO_STD_FRAC = False
PLOT_POLICY_BUDGET_EXHAUSTED_RATE = False
PLOT_POLICY_TOOL_CALLS_PER_TRAJ = False
PLOT_POLICY_FAIL_ANSWER_COUNT_0 = False
PLOT_POLICY_FAIL_UNCLOSED_TAG = False
PLOT_POLICY_FAIL_NO_BOXED = False
PLOT_POLICY_FAIL_OTHER = False
PLOT_POLICY_RESPONSE_LENGTH_MEAN = False
PLOT_POLICY_RESPONSE_LENGTH_CLIP = False
PLOT_POLICY_TOOLS_TOTAL_CALLS = False
PLOT_POLICY_TOOLS_SUCCESSFUL_CALLS = False
PLOT_POLICY_ENTROPY = False

# --- Phase-owned optimizer signals --------------------------------------------
PLOT_LL_PG_LOSS = False
PLOT_HL_PG_LOSS = False
PLOT_LL_ENTROPY_REG_LOSS = False
PLOT_HL_ENTROPY_REG_LOSS = False
PLOT_LL_GRAD_NORM = False
PLOT_HL_GRAD_NORM = False
PLOT_LL_OPEFO_LAMBDA = False
PLOT_HL_OPEFO_LAMBDA = False
PLOT_LL_OPEFO_DELTA_H_NET = False
PLOT_HL_OPEFO_DELTA_H_NET = False
PLOT_LL_ENTROPY_PHASE_MASK = False
PLOT_HL_ENTROPY_PHASE_MASK = False

# =============================================================================
# Series registry. (toggle, jsonl_relative_path, label_in_legend, field).
# =============================================================================

SERIES = [
    (PLOT_POLICY_REWARD,                 "policy/reward.jsonl",                      "policy reward",                 "value"),
    (PLOT_POLICY_REWARD_GAIN,            "policy/reward.jsonl",                      "policy reward (gain)",          "gain"),
    (PLOT_POLICY_FORMAT_PENALTY,         "policy/format_penalty.jsonl",              "policy format penalty",         "value"),
    (PLOT_POLICY_FORMAT_PENALTY_GAIN,    "policy/format_penalty.jsonl",              "policy format penalty (gain)",  "gain"),
    (PLOT_POLICY_IN_GROUP_REWARD_STD,    "policy/in_group_reward_std.jsonl",         "policy in_group_reward_std",    "value"),
    (PLOT_POLICY_FORMAT_VALID_RATE,      "policy/format_valid_rate.jsonl",           "policy format_valid_rate",      "value"),
    (PLOT_POLICY_F1_MEAN,                "policy/f1_mean.jsonl",                     "policy f1_mean",                "value"),
    (PLOT_POLICY_NO_TOOL_RATE,           "policy/no_tool_rate.jsonl",                "policy no_tool_rate",           "value"),
    (PLOT_POLICY_ADVANTAGE_STD,          "policy/advantage_std.jsonl",               "policy advantage_std",          "value"),
    (PLOT_POLICY_ENTROPY,                "policy/entropy.jsonl",                     "policy entropy",                "value"),
    (PLOT_POLICY_PPO_KL,                 "policy/ppo_kl.jsonl",                      "policy ppo_kl",                 "value"),
    (PLOT_POLICY_PG_CLIPFRAC,            "policy/pg_clipfrac.jsonl",                 "policy pg_clipfrac",            "value"),
    (PLOT_POLICY_GROUP_ZERO_STD_FRAC,    "policy/group_zero_std_frac.jsonl",         "policy group_zero_std_frac",    "value"),
    (PLOT_POLICY_BUDGET_EXHAUSTED_RATE,  "policy/budget_exhausted_rate.jsonl",       "policy budget_exhausted_rate",  "value"),
    (PLOT_POLICY_TOOL_CALLS_PER_TRAJ,    "policy/tool_calls_per_traj_mean.jsonl",    "policy tool_calls_per_traj",    "value"),
    (PLOT_POLICY_FAIL_ANSWER_COUNT_0,    "policy/fail_answer_count_0.jsonl",         "policy fail answer_count=0",    "value"),
    (PLOT_POLICY_FAIL_UNCLOSED_TAG,      "policy/fail_unclosed_tag.jsonl",           "policy fail unclosed_tag",      "value"),
    (PLOT_POLICY_FAIL_NO_BOXED,          "policy/fail_no_boxed.jsonl",               "policy fail no_boxed",          "value"),
    (PLOT_POLICY_FAIL_OTHER,             "policy/fail_other.jsonl",                  "policy fail other",             "value"),
    (PLOT_POLICY_RESPONSE_LENGTH_MEAN,   "policy/response_length_mean.jsonl",        "policy response_length_mean",   "value"),
    (PLOT_POLICY_RESPONSE_LENGTH_CLIP,   "policy/response_length_clip_ratio.jsonl",  "policy response_length_clip",   "value"),
    (PLOT_POLICY_TOOLS_TOTAL_CALLS,      "policy/tools_total_calls.jsonl",           "policy tools_total_calls",      "value"),
    (PLOT_POLICY_TOOLS_SUCCESSFUL_CALLS, "policy/tools_successful_calls.jsonl",      "policy tools_successful_calls", "value"),
    (PLOT_LL_PG_LOSS,                    "low_level/pg_loss.jsonl",                  "LL pg_loss",                    "value"),
    (PLOT_HL_PG_LOSS,                    "high_level/pg_loss.jsonl",                 "HL pg_loss",                    "value"),
    (PLOT_LL_ENTROPY_REG_LOSS,           "low_level/entropy_reg_loss.jsonl",         "LL entropy_reg_loss",           "value"),
    (PLOT_HL_ENTROPY_REG_LOSS,           "high_level/entropy_reg_loss.jsonl",        "HL entropy_reg_loss",           "value"),
    (PLOT_LL_GRAD_NORM,                  "low_level/grad_norm.jsonl",                "LL grad_norm",                  "value"),
    (PLOT_HL_GRAD_NORM,                  "high_level/grad_norm.jsonl",               "HL grad_norm",                  "value"),
    (PLOT_LL_OPEFO_LAMBDA,               "low_level/opefo_lambda.jsonl",             "LL opefo_lambda",               "value"),
    (PLOT_HL_OPEFO_LAMBDA,               "high_level/opefo_lambda.jsonl",            "HL opefo_lambda",               "value"),
    (PLOT_LL_OPEFO_DELTA_H_NET,          "low_level/opefo_delta_H_net.jsonl",        "LL opefo_delta_H_net",          "value"),
    (PLOT_HL_OPEFO_DELTA_H_NET,          "high_level/opefo_delta_H_net.jsonl",       "HL opefo_delta_H_net",          "value"),
    (PLOT_LL_ENTROPY_PHASE_MASK,         "low_level/entropy_phase_mask.jsonl",       "LL entropy (phase mask)",       "value"),
    (PLOT_HL_ENTROPY_PHASE_MASK,         "high_level/entropy_phase_mask.jsonl",      "HL entropy (phase mask)",       "value"),
]


def _read_series(path: Path, field: str) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    ys: list[float] = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            y = row[field]
            if y is None:
                continue
            steps.append(int(row["step"]))
            ys.append(float(y))
    return steps, ys


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0
    for enabled, rel_path, label, field in SERIES:
        if not enabled:
            continue
        steps, ys = _read_series(LOG_DIR / rel_path, field)
        ax.plot(steps, ys, marker="o", linewidth=1, markersize=3, label=label)
        plotted += 1

    assert plotted > 0, "No PLOT_* toggle is enabled; nothing to draw."

    ax.set_xlabel("global step")
    ax.set_ylabel("metric")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150)
    print(f"Wrote {OUTPUT_PNG} with {plotted} series.")


if __name__ == "__main__":
    main()
