# Copyright 2026 ECHO contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Δreward monotone-trend extractor for ECHO runs.

For every consecutive pair of logged training steps t-1 → t in `<run>/run.log`
we compute Δr_t for the LL phase (default: low_level/reward/entropy_scalar_mean)
and the HL phase (default: high_level/reward/f1_mean). The script then picks
the longest chronological subsequence of steps i_1 < i_2 < ... < i_k along
which the *triple* (x_t, Δ_LL_t, Δ_HL_t) is jointly monotone in the chosen
direction (default 'up' → all three non-decreasing = "x co-moves with sustained
joint reward improvement"). The selected step set is now x-dependent, so
sweeping `--x-metric` across candidates and comparing the resulting `k`
(subsequence length) directly answers "which axis is most consistent with
joint reward improvement?".

Source of truth is `run.log` (every per-step trainer dump contains
`training/global_step:N.000` and inline `key:value` fields), which is uniform
across older runs (no logging_data/) and newer runs.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Regexes for parsing the trainer's `step:N - k1:v1 - k2:v2 - …` log lines.
# `_STEP_RE` anchors on the canonical training step (avoids matching the
# narrower `step:` prefix that tqdm/progress bars also emit).
_STEP_RE = re.compile(r"training/global_step:([0-9]+)\.000")
# Each kv field is preceded by ` - ` so the leading `step:N` (no leading sep)
# is intentionally skipped — we read the step from `_STEP_RE` instead.
_KV_RE = re.compile(r" - ([\w/]+):(-?[0-9.eE+-]+)")


def load_run_log(run_dir: Path) -> pd.DataFrame:
    """Parse `<run_dir>/run.log` into a DataFrame[step, metric] of floats."""
    log_path = run_dir / "run.log"
    rows: list[dict] = []
    with open(log_path) as f:
        for line in f:
            m = _STEP_RE.search(line)
            if not m:
                continue
            row: dict = {"step": int(m.group(1))}
            for k, v in _KV_RE.findall(line):
                row[k] = float(v)
            rows.append(row)
    # Trainer occasionally emits a step twice (e.g. when both LL and HL phases
    # log into the same line buffer); keep the last (most-complete) snapshot.
    return pd.DataFrame(rows).drop_duplicates("step", keep="last").set_index("step").sort_index()


def longest_monotone_kd(series: list[list[float]], direction: str, strict: bool) -> list[int]:
    """Longest order-preserving subseq where every series in `series` is jointly monotone.

    series:    list of equal-length value arrays; index ordering is chronological.
    direction: "up" / "down" / "auto" (auto picks the longer of joint-up vs joint-down).
    strict:    True → < / > ;   False → ≤ / ≥ (default; allows equal values).
    Returns the selected indices in chronological order. O(n² k) where k=len(series).
    """
    n = len(series[0]) if series else 0

    def run_dp(want_up: bool) -> list[int]:
        if want_up:
            cmp = (lambda a, b: a < b) if strict else (lambda a, b: a <= b)
        else:
            cmp = (lambda a, b: a > b) if strict else (lambda a, b: a >= b)
        # dp[i] = length of longest valid subseq ending at i; prev[i] back-pointer.
        dp = [1] * n
        prev = [-1] * n
        for i in range(n):
            for j in range(i):
                # All k axes must be jointly monotone; short-circuit on first failure.
                if all(cmp(s[j], s[i]) for s in series) and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    prev[i] = j
        if not n:
            return []
        end = max(range(n), key=lambda k: dp[k])
        seq: list[int] = []
        while end != -1:
            seq.append(end)
            end = prev[end]
        return list(reversed(seq))

    if direction == "auto":
        up, down = run_dp(True), run_dp(False)
        return up if len(up) >= len(down) else down
    return run_dp(direction == "up")


def _plot_panel(ax, x_all, y_all, sel, x_label, y_label, title, color):
    ax.scatter(x_all, y_all, color="lightgrey", s=18, zorder=1, label="all steps")
    if sel:
        x_sel = [x_all[i] for i in sel]
        y_sel = [y_all[i] for i in sel]
        ax.plot(x_sel, y_sel, "-", color=color, linewidth=1.0, alpha=0.7, zorder=2)
        ax.scatter(x_sel, y_sel, color=color, s=36, zorder=3, label=f"monotone (k={len(sel)})")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Checkpoint directory containing run.log (e.g. .../checkpoints/echo3BInstruct).")
    p.add_argument("--x-metric", default="low_level/actor/entropy_loss",
                   help="Metric to plot on x-axis of BOTH subplots (one of run.log keys).")
    p.add_argument("--ll-y-metric", default="low_level/reward/entropy_scalar_mean",
                   help="LL reward metric whose Δ is plotted on row 1's y-axis.")
    p.add_argument("--hl-y-metric", default="high_level/reward/f1_mean",
                   help="HL reward metric whose Δ is plotted on row 2's y-axis.")
    p.add_argument("--direction", choices=["up", "down", "auto"], default="up",
                   help="Direction of joint monotonicity required on (Δ_LL, Δ_HL).")
    p.add_argument("--strict", action="store_true",
                   help="Require strict < / > (default: allow equal values).")
    p.add_argument("--output-dir", type=Path, default=Path("./echo_monotone_plots"),
                   help="Where to write the PNG + CSV (created if missing).")
    p.add_argument("--summary-file", type=Path, default=None,
                   help="Optional TSV that we append a single-row ranking summary to "
                        "(x_metric, direction, k, first_step, last_step, sel_steps).")
    args = p.parse_args()

    df = load_run_log(args.run_dir)
    needed = [args.x_metric, args.ll_y_metric, args.hl_y_metric]
    missing = [m for m in needed if m not in df.columns]
    if missing:
        sample = sorted(df.columns)[:25]
        raise SystemExit(f"metrics not found: {missing}\nfirst 25 available: {sample}")

    # Restrict to rows where every required metric is present, then take per-step Δ
    # using `.diff()` on chronologically sorted rows (NaN for the first row).
    df = df[needed].dropna().sort_index()
    # Pearson(x, reward_level) on the *level* series (not Δ). Bit-identical
    # aliases yield ρ=1.0 and the joint-monotone selection becomes partly
    # tautological (x↑ ⇒ y_level↑ ⇒ Δy mostly ≥ 0). We surface the value here
    # and warn loudly above the 0.95 threshold so users notice.
    rho_ll_lvl = df[args.x_metric].corr(df[args.ll_y_metric])
    rho_hl_lvl = df[args.x_metric].corr(df[args.hl_y_metric])

    df_d = pd.DataFrame({
        "x": df[args.x_metric],
        "dy_ll": df[args.ll_y_metric].diff(),
        "dy_hl": df[args.hl_y_metric].diff(),
    }).dropna()

    steps = df_d.index.tolist()
    xs = df_d["x"].tolist()
    ys_ll = df_d["dy_ll"].tolist()
    ys_hl = df_d["dy_hl"].tolist()

    # 3-D joint monotone selection: (x_t, Δ_LL_t, Δ_HL_t) all monotone in the
    # same direction. Selected step set is now x-dependent, so the resulting `k`
    # ranks how well this x co-moves with sustained joint reward improvement.
    sel = longest_monotone_kd([xs, ys_ll, ys_hl], args.direction, args.strict)
    sel_steps = [steps[i] for i in sel]

    print(f"[{args.run_dir.name}] {len(df_d)} consecutive Δ-pairs (steps {steps[0]}..{steps[-1]})")
    print(f"x={args.x_metric} | selected k={len(sel)} joint-monotone-{args.direction} steps: {sel_steps}")
    print(f"  ρ(x, LL reward level) = {rho_ll_lvl:.3f} | ρ(x, HL reward level) = {rho_hl_lvl:.3f}")
    if max(abs(rho_ll_lvl), abs(rho_hl_lvl)) > 0.95:
        print("  WARNING: x is near-collinear with a reward level — selection is partly tautological.")
    if sel:
        x_sel = [xs[i] for i in sel]
        print(f"  x range over selection: [{min(x_sel):.4g}, {max(x_sel):.4g}]")
        print(f"  Δ_LL range over selection: [{min(ys_ll[i] for i in sel):.4g}, {max(ys_ll[i] for i in sel):.4g}]")
        print(f"  Δ_HL range over selection: [{min(ys_hl[i] for i in sel):.4g}, {max(ys_hl[i] for i in sel):.4g}]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    _plot_panel(axes[0], xs, ys_ll, sel, "", f"Δ {args.ll_y_metric}",
                f"{args.run_dir.name} — LL Δreward vs {args.x_metric}", "tab:blue")
    _plot_panel(axes[1], xs, ys_hl, sel, args.x_metric, f"Δ {args.hl_y_metric}",
                f"{args.run_dir.name} — HL Δreward vs {args.x_metric}", "tab:orange")
    for i in sel:
        axes[0].annotate(str(steps[i]), (xs[i], ys_ll[i]),
                         textcoords="offset points", xytext=(4, 4), fontsize=7)
        axes[1].annotate(str(steps[i]), (xs[i], ys_hl[i]),
                         textcoords="offset points", xytext=(4, 4), fontsize=7)
    fig.tight_layout()

    slug = args.x_metric.replace("/", "_")
    out_png = args.output_dir / f"{args.run_dir.name}_monotone_{slug}.png"
    out_csv = out_png.with_suffix(".csv")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    df_d["in_monotone"] = df_d.index.isin(sel_steps)
    df_d.to_csv(out_csv)
    print(f"wrote {out_png}")
    print(f"wrote {out_csv}")

    if args.summary_file is not None:
        # Append a single TSV row; the wrapping shell sorts/prints at the end.
        # rho_ll/rho_hl let the ranking flag tautological x's (|ρ|→1) at a glance.
        first_step = sel_steps[0] if sel_steps else ""
        last_step = sel_steps[-1] if sel_steps else ""
        sel_csv = ",".join(str(s) for s in sel_steps)
        write_header = not args.summary_file.exists()
        args.summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_file, "a") as f:
            if write_header:
                f.write("x_metric\tdirection\tk\trho_ll\trho_hl\tfirst_step\tlast_step\tsel_steps\n")
            f.write(f"{args.x_metric}\t{args.direction}\t{len(sel)}\t{rho_ll_lvl:.3f}\t{rho_hl_lvl:.3f}\t{first_step}\t{last_step}\t{sel_csv}\n")


if __name__ == "__main__":
    main()
