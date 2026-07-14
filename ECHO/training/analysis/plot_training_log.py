# Copyright 2026 ECHO contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Per-step training-trace plotter for the ECHO recipe.

Reads the JSONL traces produced by `RayECHOTrainer._dump_logging_data` under
`{SAVE_PATH}/logging_data/{low_level,high_level}/<metric>.jsonl` and overlays
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
LOG_DIR = Path(
    "/scratch/project/prj-02-llm-reasoning-shakkottai/saratb/ECHO/checkpoints/echo3B-rerun-hybrid-entropy-1.0-bfp0/logging_data"
)
# Destination PNG for this invocation. Parent dirs are created if missing.
OUTPUT_PNG = Path("./echo_training_plot.png")

# =============================================================================
# Toggles. True -> the corresponding line is added to the overlaid PNG.
# `*_GAIN` reads the `gain` field (Δ vs previous dumped step); the rest read
# the raw `value` field. Toggle independently of each other.
# =============================================================================

# --- Reward channels (the headline gain metric) ---------------------------------
PLOT_LL_REWARD = True              # LL effective_reward_mean (scorer score used by GRPO)
PLOT_LL_REWARD_GAIN = True         # Δ of LL reward vs previous step
PLOT_LL_FORMAT_PENALTY = False     # LL bad_format_rate (share of samples with score < 0)
PLOT_LL_FORMAT_PENALTY_GAIN = False
PLOT_HL_REWARD = False             # HL effective_reward_mean (same scorer)
PLOT_HL_REWARD_GAIN = False
PLOT_HL_FORMAT_PENALTY = False     # HL bad_format_rate
PLOT_HL_FORMAT_PENALTY_GAIN = False

# --- Optimization metrics (raw values; gains usually less interesting here) ----
PLOT_LL_PG_LOSS = False            # LL PPO clipped policy-gradient loss
PLOT_HL_PG_LOSS = False
PLOT_LL_ENTROPY_REG_LOSS = False   # LL differentiable entropy regularizer term (HL has no file)
PLOT_LL_GRAD_NORM = False          # LL pre-step gradient norm
PLOT_HL_GRAD_NORM = False
PLOT_LL_ENTROPY_OLD_POLICY = False # LL diagnostic: old-policy entropy on loss_mask
PLOT_HL_ENTROPY_OLD_POLICY = False

# --- Validity / tool diagnostics ------------------------------------------------
PLOT_LL_FORMAT_VALID_RATE = False  # LL phase: shared format_valid rate
PLOT_HL_FORMAT_VALID_RATE = False  # HL phase: shared format_valid rate
PLOT_LL_TOOLS_TOTAL_CALLS = False
PLOT_HL_TOOLS_TOTAL_CALLS = False
PLOT_LL_TOOLS_SUCCESSFUL_CALLS = False
PLOT_HL_TOOLS_SUCCESSFUL_CALLS = False

# =============================================================================
# Series registry. (toggle, jsonl_relative_path, label_in_legend, field).
# `field` is "value" (raw) or "gain" (Δ vs previous step). One row per toggle.
# =============================================================================

SERIES = [
    (PLOT_LL_REWARD,                 "low_level/reward.jsonl",                  "LL reward",                  "value"),
    (PLOT_LL_REWARD_GAIN,            "low_level/reward.jsonl",                  "LL reward (gain)",           "gain"),
    (PLOT_LL_FORMAT_PENALTY,         "low_level/format_penalty.jsonl",          "LL format penalty",          "value"),
    (PLOT_LL_FORMAT_PENALTY_GAIN,    "low_level/format_penalty.jsonl",          "LL format penalty (gain)",   "gain"),
    (PLOT_HL_REWARD,                 "high_level/reward.jsonl",                 "HL reward",                  "value"),
    (PLOT_HL_REWARD_GAIN,            "high_level/reward.jsonl",                 "HL reward (gain)",           "gain"),
    (PLOT_HL_FORMAT_PENALTY,         "high_level/format_penalty.jsonl",         "HL format penalty",          "value"),
    (PLOT_HL_FORMAT_PENALTY_GAIN,    "high_level/format_penalty.jsonl",         "HL format penalty (gain)",   "gain"),
    (PLOT_LL_PG_LOSS,                "low_level/pg_loss.jsonl",                 "LL pg_loss",                 "value"),
    (PLOT_HL_PG_LOSS,                "high_level/pg_loss.jsonl",                "HL pg_loss",                 "value"),
    (PLOT_LL_ENTROPY_REG_LOSS,       "low_level/entropy_reg_loss.jsonl",        "LL entropy_reg_loss",        "value"),
    (PLOT_LL_GRAD_NORM,              "low_level/grad_norm.jsonl",               "LL grad_norm",               "value"),
    (PLOT_HL_GRAD_NORM,              "high_level/grad_norm.jsonl",              "HL grad_norm",               "value"),
    (PLOT_LL_ENTROPY_OLD_POLICY,     "low_level/entropy_old_policy.jsonl",      "LL entropy_old_policy",      "value"),
    (PLOT_HL_ENTROPY_OLD_POLICY,     "high_level/entropy_old_policy.jsonl",     "HL entropy_old_policy",      "value"),
    (PLOT_LL_FORMAT_VALID_RATE,      "low_level/format_valid_rate.jsonl",       "LL format_valid_rate",       "value"),
    (PLOT_HL_FORMAT_VALID_RATE,      "high_level/format_valid_rate.jsonl",      "HL format_valid_rate",       "value"),
    (PLOT_LL_TOOLS_TOTAL_CALLS,      "low_level/tools_total_calls.jsonl",       "LL tools/total_calls",       "value"),
    (PLOT_HL_TOOLS_TOTAL_CALLS,      "high_level/tools_total_calls.jsonl",      "HL tools/total_calls",       "value"),
    (PLOT_LL_TOOLS_SUCCESSFUL_CALLS, "low_level/tools_successful_calls.jsonl",  "LL tools/successful_calls",  "value"),
    (PLOT_HL_TOOLS_SUCCESSFUL_CALLS, "high_level/tools_successful_calls.jsonl", "HL tools/successful_calls",  "value"),
]


def _read_series(path: Path, field: str) -> tuple[list[int], list[float]]:
    # JSONL has one row per dumped training step with keys `step`, `value`,
    # `gain`. `gain` is null on the very first dump for that key (no prior
    # step), so those rows are dropped from the gain trace.
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
