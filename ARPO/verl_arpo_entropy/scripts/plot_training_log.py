# Copyright 2026 ARPO contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Per-step training-trace plotter for the ARPO recipe.

Reads the JSONL traces produced by `RayPPOTrainer._dump_logging_data` under
`{SAVE_PATH}/logging_data/<metric>.jsonl` and overlays all enabled metric
series on a single PNG. ARPO has no high/low phase split, so the layout is
flat (one file per metric) — unlike the ECHO plotter which groups files
under `low_level/` and `high_level/`. Each enabled toggle below adds one
line to the figure; flip multiple toggles on simultaneously to compare them
on the same axes (x = global step, y = metric value or step-over-step gain).
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
    "/scratch/user/saratb_tamu.edu/research/arpo-lens/ARPO/checkpoints/arpo/logging_data"
)
# Destination PNG for this invocation. Parent dirs are created if missing.
OUTPUT_PNG = Path("./arpo_training_plot.png")

# =============================================================================
# Toggles. True -> the corresponding line is added to the overlaid PNG.
# `*_GAIN` reads the `gain` field (Δ vs previous dumped step); the rest read
# the raw `value` field. Toggle independently of each other.
# =============================================================================

# --- Reward channels (the headline gain metric) ---------------------------------
PLOT_REWARD = True              # f1_mean (task signal; zero on bad-format / wrong)
PLOT_REWARD_GAIN = True         # Δ of reward vs previous step
PLOT_FORMAT_PENALTY = False     # bad_format_rate (share of samples with score == -1)
PLOT_FORMAT_PENALTY_GAIN = False
PLOT_SCORE_MEAN = False         # critic/score/mean (per-sample summed token-level score)
PLOT_SCORE_MEAN_GAIN = False

# --- Optimization metrics (raw values; gains usually less interesting here) ----
PLOT_PG_LOSS = False            # PPO clipped policy-gradient loss
PLOT_ENTROPY_REG_LOSS = False   # Differentiable entropy regularizer term (only present when actor.entropy_coeff != 0)
PLOT_GRAD_NORM = False          # Pre-step gradient norm
PLOT_ENTROPY_OLD_POLICY = False # Diagnostic: old-policy entropy on loss_mask (file: entropy_old_policy.jsonl, key: actor/entropy_loss)

# --- Tool diagnostics -----------------------------------------------------------
PLOT_TOOLS_TOTAL_CALLS = False
PLOT_TOOLS_SUCCESSFUL_CALLS = False

# =============================================================================
# Series registry. (toggle, jsonl_relative_path, label_in_legend, field).
# `field` is "value" (raw) or "gain" (Δ vs previous step). One row per toggle.
# Paths must match `RayPPOTrainer._LOGGING_SPEC` filenames exactly.
# =============================================================================

SERIES = [
    (PLOT_REWARD,                 "reward.jsonl",                 "reward",                  "value"),
    (PLOT_REWARD_GAIN,            "reward.jsonl",                 "reward (gain)",           "gain"),
    (PLOT_FORMAT_PENALTY,         "format_penalty.jsonl",         "format penalty",          "value"),
    (PLOT_FORMAT_PENALTY_GAIN,    "format_penalty.jsonl",         "format penalty (gain)",   "gain"),
    (PLOT_SCORE_MEAN,             "score_mean.jsonl",             "score_mean",              "value"),
    (PLOT_SCORE_MEAN_GAIN,        "score_mean.jsonl",             "score_mean (gain)",       "gain"),
    (PLOT_PG_LOSS,                "pg_loss.jsonl",                "pg_loss",                 "value"),
    (PLOT_ENTROPY_REG_LOSS,       "entropy_reg_loss.jsonl",       "entropy_reg_loss",        "value"),
    (PLOT_GRAD_NORM,              "grad_norm.jsonl",              "grad_norm",               "value"),
    (PLOT_ENTROPY_OLD_POLICY,     "entropy_old_policy.jsonl",     "entropy_old_policy",      "value"),
    (PLOT_TOOLS_TOTAL_CALLS,      "tools_total_calls.jsonl",      "tools/total_calls",       "value"),
    (PLOT_TOOLS_SUCCESSFUL_CALLS, "tools_successful_calls.jsonl", "tools/successful_calls",  "value"),
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
