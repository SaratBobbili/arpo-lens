#!/usr/bin/env python3
"""Post-hoc Pass@k (OR across turns) and Exploration Ability plots."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

METRICS_RE = re.compile(r"^(.+)_output_(\d+)_metrics\.json$")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "hf_math_4qa",
        help="Root containing run dirs with run_config.yaml",
    )
    p.add_argument(
        "--run_dirs",
        nargs="*",
        default=None,
        help="Optional explicit run directories (overrides discovery under --root)",
    )
    p.add_argument(
        "--metric",
        default="auto",
        help="Correctness key: llm_equal | math_equal | auto (prefer llm_equal)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "pass_at_k_exploration.png",
        help="Output figure path",
    )
    p.add_argument(
        "--summary_json",
        type=Path,
        default=None,
        help="Optional path to write aggregated Pass@k points as JSON",
    )
    return p.parse_args()


def discover_run_dirs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.rglob("run_config.yaml"))


def load_run_config(run_dir: Path) -> dict:
    with open(run_dir / "run_config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def series_label(cfg: dict) -> str:
    exp = cfg.get("source", {}).get("training_experiment", "unknown")
    prompt = cfg.get("infer", {}).get("prompt_type", "unknown")
    return f"{exp}+{prompt}"


def list_turn_metrics(dataset_dir: Path, dataset: str) -> list[Path]:
    files = []
    for path in dataset_dir.glob(f"{dataset}_output_*_metrics.json"):
        m = METRICS_RE.match(path.name)
        if m and m.group(1) == dataset:
            files.append((int(m.group(2)), path))
    files.sort(key=lambda x: x[0])
    return [p for _, p in files]


def resolve_metric(sample_metrics: dict, metric: str) -> str:
    if metric != "auto":
        return metric
    if "llm_equal" in sample_metrics:
        return "llm_equal"
    return "math_equal"


def pass_at_k(metric_paths: list[Path], metric: str) -> float:
    turns = []
    for path in metric_paths:
        with open(path, encoding="utf-8") as f:
            turns.append(json.load(f))
    n = len(turns[0])
    for t in turns[1:]:
        if len(t) != n:
            raise ValueError(f"Turn length mismatch under {metric_paths[0].parent}")
    key = resolve_metric(turns[0][0]["metrics"], metric)
    correct = 0
    for i in range(n):
        if any(bool(turns[k][i]["metrics"].get(key, 0)) for k in range(len(turns))):
            correct += 1
    return correct / n if n else 0.0


def collect_points(run_dirs: list[Path], metric: str) -> dict:
    """
    Returns:
      points[series][dataset][temperature] = (pass_at_k, run_id)
    Keeps the latest run_id when duplicates collide on series+dataset+temp.
    """
    points = defaultdict(lambda: defaultdict(dict))
    for run_dir in tqdm(run_dirs, desc="Runs"):
        cfg_path = run_dir / "run_config.yaml"
        if not cfg_path.exists():
            continue
        cfg = load_run_config(run_dir)
        temp = float(cfg.get("infer", {}).get("temperature", 0.0))
        run_id = str(cfg.get("run_id", run_dir.name))
        label = series_label(cfg)
        for dataset_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            if dataset_dir.name == "logs":
                continue
            dataset = dataset_dir.name
            metric_files = list_turn_metrics(dataset_dir, dataset)
            if not metric_files:
                continue
            rate = pass_at_k(metric_files, metric)
            prev = points[label][dataset].get(temp)
            if prev is None or run_id >= prev[1]:
                points[label][dataset][temp] = (rate, run_id)
    return points


def plot_exploration(points: dict, out_path: Path):
    datasets = sorted({d for series in points.values() for d in series})
    if not datasets:
        raise SystemExit("No Pass@k points found")

    n = len(datasets)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle("Exploration Ability", fontsize=14)

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    series_names = sorted(points.keys())

    for idx, dataset in enumerate(datasets):
        ax = axes[idx // ncols][idx % ncols]
        for s_i, series in enumerate(series_names):
            temp_map = points[series].get(dataset, {})
            if not temp_map:
                continue
            temps = sorted(temp_map.keys())
            rates = [temp_map[t][0] for t in temps]
            ax.plot(
                temps,
                rates,
                marker=markers[s_i % len(markers)],
                color=colors[s_i % len(colors)],
                label=series,
                linewidth=1.5,
            )
        ax.set_title(dataset)
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Pass@8 Rate")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True, linestyle="-", alpha=0.3)

    for j in range(idx + 1, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if not handles:
        for ax_row in axes:
            for ax in ax_row:
                h, l = ax.get_legend_handles_labels()
                if h:
                    handles, labels = h, l
                    break
            if handles:
                break
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False)
        fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    else:
        fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def points_to_json(points: dict) -> dict:
    out = {}
    for series, datasets in points.items():
        out[series] = {}
        for dataset, temps in datasets.items():
            out[series][dataset] = {
                str(t): {"pass_at_k": rate, "run_id": run_id}
                for t, (rate, run_id) in sorted(temps.items())
            }
    return out


def main():
    args = parse_args()
    if args.run_dirs:
        run_dirs = [Path(p) for p in args.run_dirs]
    else:
        run_dirs = discover_run_dirs(args.root)
    if not run_dirs:
        raise SystemExit(f"No runs found under {args.root}")

    points = collect_points(run_dirs, args.metric)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(points_to_json(points), f, indent=2)
        print(f"Wrote {args.summary_json}")
    plot_exploration(points, args.out)


if __name__ == "__main__":
    main()
