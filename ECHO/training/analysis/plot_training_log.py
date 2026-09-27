# Copyright 2026 ECHO contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Per-step training-trace plotter for the ECHO recipe.

Reads the JSONL traces produced by `RayECHOTrainer._dump_logging_data` under
`{SAVE_PATH}/logging_data/<wandb key>.jsonl` (one file per logged key) and overlays
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

# --- Toggles: one per logged key; paths mirror the wandb keys exactly ---------------
PLOT_REWARD_MEAN = True
PLOT_REWARD_MEAN_GAIN = False
PLOT_F1_MEAN = False
PLOT_FORMAT_VALID_RATE = False
PLOT_IN_GROUP_STD = False
PLOT_IN_GROUP_STD_POST = False
PLOT_GROUP_ZERO_STD_FRAC = False
PLOT_FAIL_ANSWER_COUNT_0 = False
PLOT_FAIL_UNCLOSED_TAG = False
PLOT_FAIL_NO_BOXED = False
PLOT_FAIL_OTHER = False
PLOT_RESPONSE_LENGTH_MEAN = False
PLOT_RESPONSE_LENGTH_CLIP = False
PLOT_TOOL_CALLS_PER_TRAJ = False
PLOT_NO_TOOL_RATE = False
PLOT_BUDGET_EXHAUSTED_RATE = False
PLOT_ENTROPY_REASONING = False
PLOT_ENTROPY_TOOL = False
PLOT_LL_REWARD_MEAN = False
PLOT_HL_REWARD_MEAN = False
PLOT_LL_GATE_PASS_RATE = False
PLOT_HL_GATE_PASS_RATE = False
PLOT_LL_PG_LOSS = False
PLOT_HL_PG_LOSS = False
PLOT_LL_GRAD_NORM = False
PLOT_HL_GRAD_NORM = False
PLOT_LL_PPO_KL = False
PLOT_HL_PPO_KL = False
PLOT_LL_PG_CLIPFRAC = False
PLOT_HL_PG_CLIPFRAC = False
PLOT_LL_ADVANTAGE_STD = False
PLOT_HL_ADVANTAGE_STD = False
PLOT_LL_ENTROPY_REG_LOSS = False
PLOT_HL_ENTROPY_REG_LOSS = False
PLOT_LL_OPEFO_LAMBDA = False
PLOT_HL_OPEFO_LAMBDA = False
PLOT_HL_RESPONSE_NORM = False
PLOT_HL_RESPONSE_RATIO = False
PLOT_HL_RESPONSE_COSINE = False
PLOT_HL_AHO_SURROGATE = False
PLOT_HL_AHO_OMEGA_ABSMEAN = False
PLOT_VAL_CORE_REWARD = True
PLOT_VAL_AUX_F1 = False

# =============================================================================
# Series registry. (toggle, jsonl_relative_path, label_in_legend, field).
# =============================================================================

SERIES = [
    (PLOT_REWARD_MEAN,             "reward/reward_mean.jsonl", 'reward_mean (shared)', "value"),
    (PLOT_REWARD_MEAN_GAIN,        "reward/reward_mean.jsonl", 'reward_mean (gain)', "gain"),
    (PLOT_F1_MEAN,                 "reward/f1_mean.jsonl", 'f1_mean', "value"),
    (PLOT_FORMAT_VALID_RATE,       "reward/format_valid_rate.jsonl", 'format_valid_rate (whole schema)', "value"),
    (PLOT_IN_GROUP_STD,            "reward/in_group_std.jsonl", 'in-group reward std', "value"),
    (PLOT_IN_GROUP_STD_POST,       "reward/in_group_std_post.jsonl", 'in-group reward std (post GRPO adj.)', "value"),
    (PLOT_GROUP_ZERO_STD_FRAC,     "reward/group_zero_std_frac.jsonl", 'zero-std group fraction', "value"),
    (PLOT_FAIL_ANSWER_COUNT_0,     "reward/fail_answer_count_0.jsonl", 'fail answer_count=0', "value"),
    (PLOT_FAIL_UNCLOSED_TAG,       "reward/fail_unclosed_tag.jsonl", 'fail unclosed_tag', "value"),
    (PLOT_FAIL_NO_BOXED,           "reward/fail_no_boxed.jsonl", 'fail no_boxed', "value"),
    (PLOT_FAIL_OTHER,              "reward/fail_other.jsonl", 'fail other', "value"),
    (PLOT_RESPONSE_LENGTH_MEAN,    "rollout/response_length_mean.jsonl", 'response_length_mean', "value"),
    (PLOT_RESPONSE_LENGTH_CLIP,    "rollout/response_length_clip_ratio.jsonl", 'response_length_clip_ratio', "value"),
    (PLOT_TOOL_CALLS_PER_TRAJ,     "rollout/tool_calls_per_traj_mean.jsonl", 'tool_calls_per_traj', "value"),
    (PLOT_NO_TOOL_RATE,            "rollout/no_tool_rate.jsonl", 'no_tool_rate', "value"),
    (PLOT_BUDGET_EXHAUSTED_RATE,   "rollout/budget_exhausted_rate.jsonl", 'budget_exhausted_rate', "value"),
    (PLOT_ENTROPY_REASONING,       "policy/entropy_reasoning.jsonl", 'entropy (think/answer tokens)', "value"),
    (PLOT_ENTROPY_TOOL,            "policy/entropy_tool.jsonl", 'entropy (tool/search/python tokens)', "value"),
    (PLOT_LL_REWARD_MEAN,          "low_level/reward_mean.jsonl", 'LL reward_mean', "value"),
    (PLOT_HL_REWARD_MEAN,          "high_level/reward_mean.jsonl", 'HL reward_mean', "value"),
    (PLOT_LL_GATE_PASS_RATE,       "low_level/gate_pass_rate.jsonl", 'LL gate_pass_rate', "value"),
    (PLOT_HL_GATE_PASS_RATE,       "high_level/gate_pass_rate.jsonl", 'HL gate_pass_rate', "value"),
    (PLOT_LL_PG_LOSS,              "low_level/actor/pg_loss.jsonl", 'LL pg_loss', "value"),
    (PLOT_HL_PG_LOSS,              "high_level/actor/pg_loss.jsonl", 'HL pg_loss', "value"),
    (PLOT_LL_GRAD_NORM,            "low_level/actor/grad_norm.jsonl", 'LL grad_norm', "value"),
    (PLOT_HL_GRAD_NORM,            "high_level/actor/grad_norm.jsonl", 'HL grad_norm', "value"),
    (PLOT_LL_PPO_KL,               "low_level/actor/ppo_kl.jsonl", 'LL ppo_kl', "value"),
    (PLOT_HL_PPO_KL,               "high_level/actor/ppo_kl.jsonl", 'HL ppo_kl', "value"),
    (PLOT_LL_PG_CLIPFRAC,          "low_level/actor/pg_clipfrac.jsonl", 'LL pg_clipfrac', "value"),
    (PLOT_HL_PG_CLIPFRAC,          "high_level/actor/pg_clipfrac.jsonl", 'HL pg_clipfrac', "value"),
    (PLOT_LL_ADVANTAGE_STD,        "low_level/actor/advantage_std.jsonl", 'LL advantage_std (as used)', "value"),
    (PLOT_HL_ADVANTAGE_STD,        "high_level/actor/advantage_std.jsonl", 'HL advantage_std (as used)', "value"),
    (PLOT_LL_ENTROPY_REG_LOSS,     "low_level/actor/entropy_reg_loss.jsonl", 'LL entropy_reg_loss', "value"),
    (PLOT_HL_ENTROPY_REG_LOSS,     "high_level/actor/entropy_reg_loss.jsonl", 'HL entropy_reg_loss', "value"),
    (PLOT_LL_OPEFO_LAMBDA,         "low_level/actor/opefo_lambda.jsonl", 'LL opefo_lambda', "value"),
    (PLOT_HL_OPEFO_LAMBDA,         "high_level/actor/opefo_lambda.jsonl", 'HL opefo_lambda', "value"),
    (PLOT_HL_RESPONSE_NORM,        "high_level/response/norm.jsonl", 'HL ||g_resp||', "value"),
    (PLOT_HL_RESPONSE_RATIO,       "high_level/response/ratio.jsonl", 'HL ||g_resp|| / ||g_dir||', "value"),
    (PLOT_HL_RESPONSE_COSINE,      "high_level/response/cosine.jsonl", 'HL cos(g_resp, g_dir)', "value"),
    (PLOT_HL_AHO_SURROGATE,        "high_level/aho/surrogate.jsonl", 'HL aho surrogate', "value"),
    (PLOT_HL_AHO_OMEGA_ABSMEAN,    "high_level/aho/omega_absmean.jsonl", 'HL aho |omega| mean', "value"),
    (PLOT_VAL_CORE_REWARD,         "val-core/DR_grpo_mix/reward/mean@1.jsonl", 'val-core reward', "value"),
    (PLOT_VAL_AUX_F1,              "val-aux/DR_grpo_mix/f1_score/mean@1.jsonl", 'val-aux f1', "value"),
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
