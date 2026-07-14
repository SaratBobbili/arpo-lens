from __future__ import annotations

import argparse
import base64
import io
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

RUN_ROOT_DEFAULT = Path(
    "/scratch/project/prj-02-llm-reasoning-shakkottai/saratb/ECHO/checkpoints"
)
RUN_PREFIX = "echo3BInst"
OUTPUT_HTML_DEFAULT = (
    Path(__file__).resolve().parent / "reports" / "echo3BInst_report.html"
)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"step:(\d+)\s*-\s*(.*)")
KV_RE = re.compile(r"([A-Za-z0-9_@./-]+):(-?\d+(?:\.\d+)?)")


@dataclass
class RunData:
    name: str
    run_dir: Path
    group: str
    config: dict
    series: dict[str, tuple[np.ndarray, np.ndarray]]
    summary: dict[str, float]


def _nested_get(obj: dict, path: Iterable[str], default=None):
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _first_existing(run_dir: Path) -> Path | None:
    run_log = run_dir / "run.log"
    if run_log.exists():
        return run_log
    candidates = sorted(
        run_dir.glob("wandb/run-*/files/output.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _parse_step_metrics(log_path: Path) -> dict[str, dict[int, float]]:
    metrics: dict[str, dict[int, float]] = defaultdict(dict)
    total = log_path.stat().st_size
    with open(log_path, "rb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=f"parse {log_path.parent.name}",
        leave=False,
    ) as pbar:
        while True:
            raw = f.readline()
            if not raw:
                break
            pbar.update(len(raw))
            if b"step:" not in raw:
                continue
            line = ANSI_RE.sub("", raw.decode("utf-8", errors="ignore"))
            m = STEP_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            payload = m.group(2)
            for key, val in KV_RE.findall(payload):
                metrics[key][step] = float(val)
    return metrics


def _series_from_map(step_map: dict[int, float] | None) -> tuple[np.ndarray, np.ndarray]:
    if not step_map:
        return np.array([]), np.array([])
    xs = np.array(sorted(step_map), dtype=float)
    ys = np.array([step_map[int(x)] for x in xs], dtype=float)
    return xs, ys


def _pick_metric(
    metrics: dict[str, dict[int, float]], keys: list[str], prefix: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    for key in keys:
        if key in metrics:
            return _series_from_map(metrics[key])
    if prefix is not None:
        found = sorted([k for k in metrics if k.startswith(prefix)])
        if found:
            return _series_from_map(metrics[found[0]])
    return np.array([]), np.array([])


def _extract_series(metrics: dict[str, dict[int, float]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    out["val_core"] = _pick_metric(metrics, [], prefix="val-core/")
    out["hl_score"] = _pick_metric(
        metrics,
        ["high_level/reward/f1_mean", "high_level/reward/score_mean"],
    )
    out["ll_score"] = _pick_metric(
        metrics,
        [
            "low_level/reward/f1_mean",
            "low_level/reward/score_mean",
            "low_level/reward/effective_reward_mean",
            # Fallback for pre-advantage_algorithm runs that logged entropy-as-reward.
            "low_level/reward/entropy_scalar_mean_good",
            "low_level/reward/entropy_scalar_mean",
        ],
    )
    out["hl_resp_len"] = _pick_metric(metrics, ["high_level/response_length/mean"])
    out["ll_resp_len"] = _pick_metric(metrics, ["low_level/response_length/mean"])
    out["hl_no_tool_rate"] = _pick_metric(metrics, ["high_level/reward/no_tool_rate"])
    out["ll_no_tool_rate"] = _pick_metric(metrics, ["low_level/reward/no_tool_rate"])
    out["ll_entropy_old_policy"] = _pick_metric(
        metrics, ["low_level/actor/entropy_old_policy"]
    )
    return out


def _assign_group(name: str) -> str:
    if "multiple_updates_per_phase" in name:
        return "LL→HL GRPO: multi-update schedule"
    if "ll_grpo_hl_dapo" in name:
        return "LL→HL with HL DAPO variant"
    if name.startswith("echo3BInst_hl_ll_entropy_grpo"):
        return "HL/LL entropy GRPO ablations"
    if name.startswith("echo3BInst_hl_ll_entropy_hybrid"):
        return "HL/LL legacy entropy-as-reward hybrid ablations"
    if "entropy_hybrid" in name:
        return "LL→HL legacy entropy-as-reward hybrid sweeps"
    if "ll_hl" in name:
        return "LL→HL GRPO baselines"
    return "Other"


def _load_config(run_dir: Path) -> dict:
    path = run_dir / "outputs" / ".hydra" / "config.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _config_signature(cfg: dict) -> dict[str, str]:
    hl = _nested_get(cfg, ["reward_model", "phase_rewards", "high_level"], {}) or {}
    ll = _nested_get(cfg, ["reward_model", "phase_rewards", "low_level"], {}) or {}
    return {
        "norm_adv": str(_nested_get(cfg, ["algorithm", "norm_adv_by_std_in_grpo"])),
        "phase_order": str(_nested_get(cfg, ["reward_model", "phase_order"])),
        "phase_repeat_hl": str(
            _nested_get(cfg, ["reward_model", "phase_update_repeats", "high_level"])
        ),
        "phase_repeat_ll": str(
            _nested_get(cfg, ["reward_model", "phase_update_repeats", "low_level"])
        ),
        "hl_adv": str(hl.get("advantage_algorithm", hl.get("algorithm"))),
        "ll_adv": str(ll.get("advantage_algorithm", ll.get("algorithm"))),
        "hl_kl_coef": str(hl.get("kl_loss_coef")),
        "ll_kl_coef": str(ll.get("kl_loss_coef")),
        "ll_entropy_reg": str(_nested_get(ll, ["entropy", "reg_coeff"])),
        "mask_tool": str(
            _nested_get(cfg, ["actor_rollout_ref", "rollout", "mask_categories", "tool"])
        ),
    }


def _tail_slope(xs: np.ndarray, ys: np.ndarray, frac: float = 0.35) -> float:
    if len(xs) < 3:
        return float("nan")
    n = max(3, int(math.ceil(len(xs) * frac)))
    xt = xs[-n:]
    yt = ys[-n:]
    if np.allclose(xt, xt[0]):
        return 0.0
    return float(np.polyfit(xt, yt, 1)[0])


def _basic_stats(xs: np.ndarray, ys: np.ndarray, prefix: str) -> dict[str, float]:
    if len(xs) == 0:
        return {}
    out = {
        f"{prefix}_start": float(ys[0]),
        f"{prefix}_final": float(ys[-1]),
        f"{prefix}_best": float(np.max(ys)),
        f"{prefix}_mean": float(np.mean(ys)),
        f"{prefix}_tail_slope": _tail_slope(xs, ys),
    }
    return out


def _summarize_series(series: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in [
        "val_core",
        "hl_score",
        "ll_score",
        "hl_resp_len",
        "ll_resp_len",
        "hl_no_tool_rate",
        "ll_no_tool_rate",
        "ll_entropy_old_policy",
    ]:
        xs, ys = series[key]
        out.update(_basic_stats(xs, ys, key))
    return out


def _curve_mean(curves: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    valid = [(x, y) for x, y in curves if len(x) > 1]
    if not valid:
        return np.array([]), np.array([])
    grid = np.unique(np.concatenate([x for x, _ in valid]))
    arr = np.full((len(valid), len(grid)), np.nan, dtype=float)
    for i, (x, y) in enumerate(valid):
        y_interp = np.interp(grid, x, y, left=np.nan, right=np.nan)
        y_interp[grid < x[0]] = np.nan
        y_interp[grid > x[-1]] = np.nan
        arr[i] = y_interp
    return grid, np.nanmean(arr, axis=0)


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _plot_global_dashboard(runs: list[RunData]) -> dict[str, str]:
    figures: dict[str, str] = {}
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10, 4.6))
    for r in runs:
        x, y = r.series["val_core"]
        if len(x):
            ax.plot(x, y, alpha=0.35, linewidth=1.1, label=r.name)
    ax.set_title("val-core reward by run")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Reward")
    if len(runs) <= 10:
        ax.legend(fontsize=7, ncol=2)
    figures["val_core"] = _fig_to_base64(fig)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    for r in runs:
        xh, yh = r.series["hl_score"]
        xl, yl = r.series["ll_score"]
        if len(xh):
            ax1.plot(xh, yh, alpha=0.35, linewidth=1.1)
        if len(xl):
            ax2.plot(xl, yl, alpha=0.35, linewidth=1.1)
    ax1.set_title("High-level score by run")
    ax1.set_ylabel("HL score")
    ax2.set_title("Low-level score by run")
    ax2.set_ylabel("LL score")
    ax2.set_xlabel("Global step")
    figures["hl_ll_scores"] = _fig_to_base64(fig)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    for r in runs:
        xh, yh = r.series["hl_resp_len"]
        xl, yl = r.series["ll_resp_len"]
        if len(xh):
            ax1.plot(xh, yh, alpha=0.35, linewidth=1.1)
        if len(xl):
            ax2.plot(xl, yl, alpha=0.35, linewidth=1.1)
    ax1.set_title("High-level response length mean")
    ax1.set_ylabel("Tokens")
    ax2.set_title("Low-level response length mean")
    ax2.set_ylabel("Tokens")
    ax2.set_xlabel("Global step")
    figures["resp_len"] = _fig_to_base64(fig)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    for r in runs:
        xh, yh = r.series["hl_no_tool_rate"]
        xl, yl = r.series["ll_no_tool_rate"]
        if len(xh):
            ax1.plot(xh, yh, alpha=0.35, linewidth=1.1)
        if len(xl):
            ax2.plot(xl, yl, alpha=0.35, linewidth=1.1)
    ax1.set_title("High-level no-tool rate")
    ax1.set_ylabel("Rate")
    ax2.set_title("Low-level no-tool rate")
    ax2.set_ylabel("Rate")
    ax2.set_xlabel("Global step")
    figures["no_tool_rate"] = _fig_to_base64(fig)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    for r in runs:
        x, y = r.series["ll_entropy_old_policy"]
        if len(x):
            ax.plot(x, y, alpha=0.35, linewidth=1.1)
    ax.set_title("Low-level entropy_old_policy")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Entropy")
    figures["ll_entropy"] = _fig_to_base64(fig)

    return figures


def _plot_group_disp_style(group: str, runs: list[RunData]) -> str:
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    for r in runs:
        xv, yv = r.series["val_core"]
        xh, yh = r.series["hl_score"]
        xe, ye = r.series["ll_entropy_old_policy"]
        xn, yn = r.series["ll_no_tool_rate"]
        if len(xv):
            ax_top.plot(xv, yv, color="#1f77b4", alpha=0.22, linewidth=1.0)
        if len(xh):
            ax_top.plot(xh, yh, color="#2ca02c", alpha=0.22, linewidth=1.0, linestyle="--")
        if len(xe):
            ax_bottom.plot(xe, ye, color="#2ca02c", alpha=0.22, linewidth=1.0)
        if len(xn):
            ax_bottom.plot(xn, yn, color="#ff7f0e", alpha=0.22, linewidth=1.0, linestyle="--")

    xmv, ymv = _curve_mean([r.series["val_core"] for r in runs])
    xmh, ymh = _curve_mean([r.series["hl_score"] for r in runs])
    xme, yme = _curve_mean([r.series["ll_entropy_old_policy"] for r in runs])
    xmn, ymn = _curve_mean([r.series["ll_no_tool_rate"] for r in runs])

    if len(xmv):
        ax_top.plot(xmv, ymv, color="#1f77b4", linewidth=2.4, label="val-core (group mean)")
    if len(xmh):
        ax_top.plot(
            xmh, ymh, color="#2ca02c", linewidth=2.4, linestyle="--", label="HL score (group mean)"
        )
    if len(xme):
        ax_bottom.plot(
            xme, yme, color="#2ca02c", linewidth=2.4, label="LL entropy_old_policy (group mean)"
        )
    if len(xmn):
        ax_bottom.plot(
            xmn, ymn, color="#ff7f0e", linewidth=2.4, linestyle="--", label="LL no-tool rate (group mean)"
        )

    ax_top.set_title(f"{group}: consolidation vs exploration")
    ax_top.set_ylabel("Performance")
    ax_top.legend(loc="best", fontsize=8)
    ax_bottom.set_title("Exploration diagnostics")
    ax_bottom.set_xlabel("Global step")
    ax_bottom.set_ylabel("Entropy / Rate")
    ax_bottom.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _insights_placeholder() -> str:
    return """
    <ul>
      <li>[Fill in insights for this section]</li>
    </ul>
    """


def _common_hyperparams(runs: list[RunData]) -> list[tuple[str, str]]:
    counters: dict[str, Counter] = defaultdict(Counter)
    for r in runs:
        sig = _config_signature(r.config)
        for k, v in sig.items():
            counters[k][v] += 1
    rows: list[tuple[str, str]] = []
    for k in [
        "norm_adv",
        "phase_order",
        "phase_repeat_hl",
        "phase_repeat_ll",
        "hl_adv",
        "ll_adv",
        "hl_kl_coef",
        "ll_kl_coef",
        "ll_entropy_reg",
        "mask_tool",
    ]:
        if counters[k]:
            top, cnt = counters[k].most_common(1)[0]
            rows.append((k, f"{top} ({cnt}/{len(runs)} runs)"))
    return rows


def _build_html(runs: list[RunData]) -> str:
    group_map: dict[str, list[RunData]] = defaultdict(list)
    for r in runs:
        group_map[r.group].append(r)
    groups = sorted(group_map)
    dash = _plot_global_dashboard(runs)

    sections = []
    for g in groups:
        rs = sorted(group_map[g], key=lambda x: x.name)
        img = _plot_group_disp_style(g, rs)
        run_list = "".join(f"<li><code>{r.name}</code></li>" for r in rs)
        sections.append(
            f"""
            <section>
              <h2>{g}</h2>
              <p>Runs in this regime:</p>
              <ul>{run_list}</ul>
              <img src="data:image/png;base64,{img}" alt="{g} plot" />
              <h3>Insights</h3>
              {_insights_placeholder()}
            </section>
            """
        )

    hp_rows = "".join(f"<tr><td><code>{k}</code></td><td>{v}</td></tr>" for k, v in _common_hyperparams(runs))
    section_html = "\n".join(sections)
    generated = json.dumps([r.name for r in runs])

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>echo3BInst ECHO report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; line-height: 1.4; color: #111; }}
    h1, h2, h3 {{ margin-bottom: 8px; }}
    section {{ margin-top: 30px; padding-top: 8px; border-top: 1px solid #ddd; }}
    img {{ max-width: 100%; border: 1px solid #e5e5e5; border-radius: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    code {{ background: #f6f8fa; padding: 1px 4px; border-radius: 4px; }}
    .small {{ color: #555; font-size: 0.92em; }}
  </style>
</head>
<body>
  <h1>echo3BInst Experiment Report</h1>
  <p class="small">Generated from {len(runs)} runs under <code>{RUN_ROOT_DEFAULT}</code></p>
  <p class="small">Runs: <code>{generated}</code></p>

  <section>
    <h2>Introduction: ECHO objective (compact)</h2>
    <p>At each phase <code>p ∈ {{LL, HL}}</code>, the optimization signal can be read as:</p>
    <p><code>J_p(θ) = E[A_p · log π_θ(a|s)] + λ_ent,p · H_p(π_θ) - λ_kl,p · KL(π_θ || π_ref)</code></p>
    <p>Reward is always scorer F1 (+ format gate). Per-phase <code>advantage_algorithm</code> chooses how A_p is formed (<code>grpo</code> / <code>entropy</code> / <code>aepo</code>). LL explores under its phase mask; HL consolidates under its own.</p>
    <h3>Insights</h3>
    {_insights_placeholder()}
  </section>

  <section>
    <h2>Key hyperparameters (dominant settings)</h2>
    <table>
      <thead><tr><th>Hyperparameter</th><th>Most common value</th></tr></thead>
      <tbody>{hp_rows}</tbody>
    </table>
    <h3>Insights</h3>
    {_insights_placeholder()}
  </section>

  <section>
    <h2>Global dashboards (W&amp;B-style time series)</h2>
    <h3>val-core reward</h3>
    <img src="data:image/png;base64,{dash["val_core"]}" alt="val-core plot"/>
    <h3>High-level and low-level scores</h3>
    <img src="data:image/png;base64,{dash["hl_ll_scores"]}" alt="hl-ll score plot"/>
    <h3>Response length mean (HL / LL)</h3>
    <img src="data:image/png;base64,{dash["resp_len"]}" alt="response length plot"/>
    <h3>No-tool rate (HL / LL)</h3>
    <img src="data:image/png;base64,{dash["no_tool_rate"]}" alt="no-tool rate plot"/>
    <h3>LL entropy_old_policy</h3>
    <img src="data:image/png;base64,{dash["ll_entropy"]}" alt="ll entropy plot"/>
    <h3>Insights</h3>
    {_insights_placeholder()}
  </section>

  {section_html}

  <section>
    <h2>Failure-focused synthesis</h2>
    <ul>
      <li><b>Late regression:</b> runs with large best→final val-core drop indicate consolidation is not being retained.</li>
      <li><b>Exploration collapse:</b> LL entropy tail drop or very high LL no-tool rate suggests reduced structured exploration.</li>
      <li><b>Weak HL consolidation:</b> flat/negative HL tail slope despite active LL exploration points to ineffective transfer from exploration to correctness.</li>
    </ul>
    <h3>Insights</h3>
    {_insights_placeholder()}
  </section>
</body>
</html>
"""


def build_report(run_root: Path, output_html: Path) -> None:
    run_dirs = sorted([p for p in run_root.glob(f"{RUN_PREFIX}*") if p.is_dir()])
    runs: list[RunData] = []
    for run_dir in tqdm(run_dirs, desc="runs", leave=False):
        log_path = _first_existing(run_dir)
        if log_path is None:
            continue
        metrics = _parse_step_metrics(log_path)
        series = _extract_series(metrics)
        cfg = _load_config(run_dir)
        summary = _summarize_series(series)
        runs.append(
            RunData(
                name=run_dir.name,
                run_dir=run_dir,
                group=_assign_group(run_dir.name),
                config=cfg,
                series=series,
                summary=summary,
            )
        )

    if not runs:
        raise RuntimeError("No runs parsed. Check run root and prefix.")
    output_html.parent.mkdir(parents=True, exist_ok=True)
    html = _build_html(runs)
    output_html.write_text(html)
    print(f"Wrote report: {output_html}")
    print(f"Parsed runs: {len(runs)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT_DEFAULT)
    parser.add_argument("--output-html", type=Path, default=OUTPUT_HTML_DEFAULT)
    args = parser.parse_args()
    build_report(args.run_root, args.output_html)


if __name__ == "__main__":
    main()
