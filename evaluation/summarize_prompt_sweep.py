#!/usr/bin/env python3
"""Rank ARPO system-prompt eval runs by accuracy on math_all datasets."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import yaml
from tqdm import tqdm

METRICS_RE = re.compile(r"^(.+)_output_(\d+)_metrics\.json$")
DEFAULT_PROMPTS = frozenset(
    {"base", "math", "search", "code_search", "gemini", "react", "claude"}
)
DEFAULT_MODEL_NEEDLES = ("Qwen2.5-7B-ARPO", "dongguanting/Qwen2.5-7B-ARPO")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "hf_math_4qa",
    )
    p.add_argument("--run_dirs", nargs="*", default=None)
    p.add_argument(
        "--metric",
        default="auto",
        help="llm_equal | math_equal | auto (prefer llm_equal)",
    )
    p.add_argument(
        "--model_needles",
        nargs="*",
        default=list(DEFAULT_MODEL_NEEDLES),
        help="Keep runs whose served name or HF path contains any needle",
    )
    p.add_argument(
        "--prompt_types",
        nargs="*",
        default=sorted(DEFAULT_PROMPTS),
        help="Keep these infer.prompt_type values",
    )
    p.add_argument("--out", type=Path, default=None, help="JSON summary path")
    p.add_argument("--csv", type=Path, default=None, help="CSV summary path")
    return p.parse_args()


def discover_run_dirs(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.rglob("run_config.yaml"))


def load_run_config(run_dir: Path) -> dict:
    with open(run_dir / "run_config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def accuracy(metric_paths: list[Path], metric: str) -> tuple[float, str]:
    turns = []
    for path in metric_paths:
        with open(path, encoding="utf-8") as f:
            turns.append(json.load(f))
    n = len(turns[0])
    for t in turns[1:]:
        if len(t) != n:
            raise ValueError(f"Turn length mismatch under {metric_paths[0].parent}")
    key = resolve_metric(turns[0][0]["metrics"], metric)
    # Single-turn mean correctness; multi-turn = mean of OR across turns (Pass@k).
    correct = 0
    for i in range(n):
        if any(bool(turns[k][i]["metrics"].get(key, 0)) for k in range(len(turns))):
            correct += 1
    return (correct / n if n else 0.0), key


def model_matches(cfg: dict, needles: list[str]) -> bool:
    if not needles:
        return True
    src = cfg.get("source", {})
    blob = " ".join(
        str(src.get(k, "") or "")
        for k in (
            "served_model_name",
            "hf_checkpoint_path",
            "raw_checkpoint_path",
            "checkpoint_step",
            "training_experiment",
        )
    )
    return any(n in blob for n in needles)


def collect_rows(
    run_dirs: list[Path],
    metric: str,
    needles: list[str],
    prompt_types: set[str],
) -> list[dict]:
    # prompt_type -> dataset -> (rate, run_id, used_metric, run_dir)
    best: dict[str, dict[str, tuple]] = defaultdict(dict)
    for run_dir in tqdm(run_dirs, desc="Runs"):
        cfg_path = run_dir / "run_config.yaml"
        if not cfg_path.exists():
            continue
        cfg = load_run_config(run_dir)
        if not model_matches(cfg, needles):
            continue
        prompt = cfg.get("infer", {}).get("prompt_type")
        if prompt not in prompt_types:
            continue
        run_id = str(cfg.get("run_id", run_dir.name))
        for dataset_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            if dataset_dir.name == "logs":
                continue
            dataset = dataset_dir.name
            metric_files = list_turn_metrics(dataset_dir, dataset)
            if not metric_files:
                continue
            rate, used = accuracy(metric_files, metric)
            prev = best[prompt].get(dataset)
            if prev is None or run_id >= prev[1]:
                best[prompt][dataset] = (rate, run_id, used, str(run_dir))

    rows = []
    for prompt, datasets in best.items():
        if not datasets:
            continue
        rates = {d: v[0] for d, v in datasets.items()}
        mean = sum(rates.values()) / len(rates)
        sample_metric = next(iter(datasets.values()))[2]
        sample_run = next(iter(datasets.values()))[3]
        rows.append(
            {
                "prompt_type": prompt,
                "mean": mean,
                "metric": sample_metric,
                "run_dir": sample_run,
                "datasets": rates,
            }
        )
    rows.sort(key=lambda r: r["mean"], reverse=True)
    return rows


def print_table(rows: list[dict]):
    if not rows:
        print("No matching runs.")
        return
    datasets = sorted({d for r in rows for d in r["datasets"]})
    headers = ["prompt_type", "mean"] + datasets
    widths = {h: len(h) for h in headers}
    for r in rows:
        widths["prompt_type"] = max(widths["prompt_type"], len(r["prompt_type"]))
        widths["mean"] = max(widths["mean"], len(f"{r['mean']:.4f}"))
        for d in datasets:
            val = r["datasets"].get(d)
            s = f"{val:.4f}" if val is not None else "-"
            widths[d] = max(widths[d], len(s))

    def fmt_row(cells):
        return "  ".join(str(c).ljust(widths[h]) for h, c in zip(headers, cells))

    print(fmt_row(headers))
    print(fmt_row(["-" * widths[h] for h in headers]))
    for r in rows:
        cells = [r["prompt_type"], f"{r['mean']:.4f}"]
        for d in datasets:
            v = r["datasets"].get(d)
            cells.append(f"{v:.4f}" if v is not None else "-")
        print(fmt_row(cells))
    print()
    print(f"Best prompt_type: {rows[0]['prompt_type']}  mean={rows[0]['mean']:.4f}")


def main():
    args = parse_args()
    if args.run_dirs:
        run_dirs = [Path(p) for p in args.run_dirs]
    else:
        run_dirs = discover_run_dirs(args.root)
    if not run_dirs:
        raise SystemExit(f"No runs found under {args.root}")

    rows = collect_rows(
        run_dirs,
        args.metric,
        args.model_needles,
        set(args.prompt_types),
    )
    print_table(rows)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"Wrote {args.out}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        datasets = sorted({d for r in rows for d in r["datasets"]})
        fieldnames = ["prompt_type", "mean", "metric"] + datasets
        with open(args.csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                row = {
                    "prompt_type": r["prompt_type"],
                    "mean": r["mean"],
                    "metric": r["metric"],
                }
                for d in datasets:
                    row[d] = r["datasets"].get(d)
                w.writerow(row)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
