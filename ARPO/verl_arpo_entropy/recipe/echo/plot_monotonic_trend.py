# Copyright 2026 ECHO contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Δreward joint-monotone-trend extractor (ECHO + single-phase baselines).

What
----
For every consecutive pair of logged training steps t-1 → t in `<run>/run.log`
we compute Δr_t for each user-supplied y-metric (one per "phase" panel). The
script then picks the longest chronological subsequence of steps
i_1 < i_2 < ... < i_k along which the (k+1)-tuple
    (x_t, Δr¹_t, Δr²_t, ...)
is jointly monotone in the chosen direction (default 'up' = "x co-moves with
sustained joint reward improvement"). Sweeping `--x-metric` across candidates
and ranking by `k` answers "which axis is most consistent with joint reward
improvement?". A self-correlation guard prints |ρ(x, y_level)| per phase so
near-tautological x's (e.g. x ≡ a reward alias, ρ → ±1) are flagged.

Why run.log
-----------
Source of truth is `<run_dir>/run.log` — every per-step trainer dump contains
`training/global_step:N.000` and inline `key:val` fields. This is uniform
across:
  - older ECHO runs without `logging_data/` (e.g. echo3BInstruct);
  - newer ECHO runs *with* `logging_data/<phase>/<metric>.jsonl` traces;
  - non-ECHO baselines (ARPO, GRPO) that emit a single-phase metric set.
The new JSONL dumps are a strict subset of run.log keys (just under different
naming, e.g. `low_level/reward.jsonl` ↔ `low_level/reward/entropy_scalar_mean`),
so we don't need a second loader.

Usage
-----
ECHO dual-phase (default — LL entropy reward + HL F1):
  python -m recipe.echo.plot_monotonic_trend \\
      --run-dir /scratch/.../checkpoints/echo3BInstruct \\
      --x-metric low_level/actor/entropy_loss

ECHO with custom phase pair (e.g. raw HL score instead of F1):
  python -m recipe.echo.plot_monotonic_trend \\
      --run-dir .../checkpoints/echo3BInstruct \\
      --x-metric high_level/actor/kl_loss \\
      --y-metrics LL:low_level/reward/entropy_scalar_mean \\
                  HL:high_level/reward/score_mean

ARPO/GRPO single-phase baseline:
  python -m recipe.echo.plot_monotonic_trend \\
      --run-dir /scratch/.../GRPO/Qwen3B-Instruct/checkpoints/grpo \\
      --x-metric actor/entropy_loss \\
      --y-metrics actor:critic/rewards/mean

Each `--y-metrics` entry is `LABEL:metric_name`; LABEL is used for panel
titles, summary TSV columns (`rho_<LABEL>`), and step annotations. Number of
labels = number of subplot rows. Run with a bogus metric to dump the list of
keys present in the run.log.
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

_PANEL_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


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


def _parse_y_metric_spec(spec: str) -> tuple[str, str]:
    """Split a `LABEL:metric_name` CLI entry. Errors out on a bare metric (no label)."""
    if ":" not in spec:
        raise SystemExit(f"--y-metrics entry must be 'LABEL:metric_name', got: {spec!r}")
    label, metric = spec.split(":", 1)
    if not label or not metric:
        raise SystemExit(f"--y-metrics entry has empty label or metric: {spec!r}")
    return label, metric


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Checkpoint directory containing run.log (e.g. .../checkpoints/echo3BInstruct).")
    p.add_argument("--x-metric", default="low_level/actor/entropy_loss",
                   help="Metric to plot on the x-axis of every subplot (one of run.log keys).")
    p.add_argument("--y-metrics", nargs="+",
                   default=["LL:low_level/reward/entropy_scalar_mean",
                            "HL:high_level/reward/f1_mean"],
                   help="One or more LABEL:metric_name pairs. Each defines a Δ-reward panel "
                        "and a DP axis. Default = ECHO dual-phase. For ARPO/GRPO baselines: "
                        "`--y-metrics actor:critic/rewards/mean`.")
    p.add_argument("--direction", choices=["up", "down", "auto"], default="up",
                   help="Direction of joint monotonicity required across (x, Δy¹, Δy², …).")
    p.add_argument("--strict", action="store_true",
                   help="Require strict < / > (default: allow equal values).")
    p.add_argument("--output-dir", type=Path, default=Path("./echo_monotone_plots"),
                   help="Where to write the PNG + CSV (created if missing).")
    p.add_argument("--summary-file", type=Path, default=None,
                   help="Optional TSV that we append a single-row ranking summary to "
                        "(x_metric, direction, k, rho_<LABEL>..., first_step, last_step, sel_steps).")
    args = p.parse_args()

    y_specs = [_parse_y_metric_spec(s) for s in args.y_metrics]
    y_labels = [lbl for lbl, _ in y_specs]
    y_metrics = [m for _, m in y_specs]

    df = load_run_log(args.run_dir)
    needed = [args.x_metric] + y_metrics
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
    rhos = {lbl: df[args.x_metric].corr(df[m]) for lbl, m in y_specs}

    df_d = pd.DataFrame({"x": df[args.x_metric],
                         **{f"dy_{lbl}": df[m].diff() for lbl, m in y_specs}}).dropna()

    steps = df_d.index.tolist()
    xs = df_d["x"].tolist()
    dys = {lbl: df_d[f"dy_{lbl}"].tolist() for lbl in y_labels}

    # (k+1)-D joint monotone selection: (x_t, Δy¹_t, …, Δy^K_t) all monotone in
    # the same direction. Selected step set is x-dependent, so `k` ranks how
    # well this x co-moves with sustained joint reward improvement.
    sel = longest_monotone_kd([xs, *dys.values()], args.direction, args.strict)
    sel_steps = [steps[i] for i in sel]

    print(f"[{args.run_dir.name}] {len(df_d)} consecutive Δ-pairs (steps {steps[0]}..{steps[-1]})")
    print(f"x={args.x_metric} | y={y_labels} | k={len(sel)} joint-monotone-{args.direction} steps: {sel_steps}")
    rho_str = " | ".join(f"ρ(x, {lbl} level)={r:.3f}" for lbl, r in rhos.items())
    print(f"  {rho_str}")
    if any(abs(r) > 0.95 for r in rhos.values()):
        print("  WARNING: x is near-collinear with a reward level — selection is partly tautological.")
    if sel:
        x_sel = [xs[i] for i in sel]
        print(f"  x range over selection: [{min(x_sel):.4g}, {max(x_sel):.4g}]")
        for lbl in y_labels:
            ys = dys[lbl]
            print(f"  Δ_{lbl} range over selection: [{min(ys[i] for i in sel):.4g}, {max(ys[i] for i in sel):.4g}]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_panels = len(y_labels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for idx, lbl in enumerate(y_labels):
        # Only the bottom subplot gets the x label (sharex hides others' tick labels).
        x_label = args.x_metric if idx == n_panels - 1 else ""
        color = _PANEL_COLORS[idx % len(_PANEL_COLORS)]
        _plot_panel(axes[idx], xs, dys[lbl], sel, x_label,
                    f"Δ {y_specs[idx][1]}",
                    f"{args.run_dir.name} — {lbl} Δreward vs {args.x_metric}", color)
        for i in sel:
            axes[idx].annotate(str(steps[i]), (xs[i], dys[lbl][i]),
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
        # rho_<label> per phase lets the ranking flag tautological x's (|ρ|→1) at a glance.
        first_step = sel_steps[0] if sel_steps else ""
        last_step = sel_steps[-1] if sel_steps else ""
        sel_csv = ",".join(str(s) for s in sel_steps)
        rho_cols = [f"rho_{lbl}" for lbl in y_labels]
        rho_vals = [f"{rhos[lbl]:.3f}" for lbl in y_labels]
        write_header = not args.summary_file.exists()
        args.summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_file, "a") as f:
            if write_header:
                f.write("\t".join(["x_metric", "direction", "k", *rho_cols,
                                   "first_step", "last_step", "sel_steps"]) + "\n")
            f.write("\t".join([args.x_metric, args.direction, str(len(sel)), *rho_vals,
                               str(first_step), str(last_step), sel_csv]) + "\n")


if __name__ == "__main__":
    main()
