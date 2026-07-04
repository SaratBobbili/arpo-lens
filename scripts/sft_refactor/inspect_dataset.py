#!/usr/bin/env python3
"""
Full-dataset pattern scan for dongguanting/ARPO-SFT-54K.

Must run on all rows before split_trajectories.py.

Usage:
    python scripts/sft_refactor/inspect_dataset.py --output-dir scripts/sft_refactor/output/inspect
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import load_dataset
from tqdm import tqdm

from trajectory_parse import (
    count_tags,
    extract_assistant,
    parse_segments,
    pattern_id_from_segments,
    structural_flags,
    tag_sequence,
    think_follows,
)


def infer_conversion_rule(pattern_id: str, flags: list[str], counts: dict) -> dict:
    drop_flags = {
        "empty_trajectory",
        "unpaired_think",
        "unpaired_search",
        "unpaired_python",
        "unpaired_result",
        "unpaired_answer",
        "multiple_answer",
        "first_tag_not_think",
        "tool_before_think",
    }
    if any(f in flags for f in drop_flags):
        return {
            "action": "drop",
            "needs_gpt_split": False,
            "needs_tool_before_answer": False,
            "strip_stray": False,
        }

    has_tools = counts.get("search", 0) + counts.get("python", 0) > 0
    has_stray = "has_stray_text" in flags
    missing_boxed = "missing_boxed" in flags

    if not has_tools:
        if has_stray or missing_boxed:
            return {
                "action": "gpt_split",
                "needs_gpt_split": True,
                "needs_tool_before_answer": True,
                "strip_stray": has_stray,
            }
        return {
            "action": "pass_through",
            "needs_gpt_split": False,
            "needs_tool_before_answer": False,
            "strip_stray": False,
        }

    if has_stray:
        return {
            "action": "gpt_split",
            "needs_gpt_split": True,
            "needs_tool_before_answer": False,
            "strip_stray": True,
        }

    return {
        "action": "gpt_split",
        "needs_gpt_split": True,
        "needs_tool_before_answer": False,
        "strip_stray": False,
    }


def analyze_row(idx: int, assistant: str) -> dict:
    segments = parse_segments(assistant)
    pid = pattern_id_from_segments(segments)
    counts = count_tags(segments)
    flags = structural_flags(segments, assistant)
    rule = infer_conversion_rule(pid, flags, counts)
    return {
        "idx": idx,
        "pattern_id": pid,
        "tag_sequence": tag_sequence(segments),
        "flags": flags,
        "think_count": counts.get("think", 0),
        "search_count": counts.get("search", 0),
        "python_count": counts.get("python", 0),
        "result_count": counts.get("result", 0),
        "answer_count": counts.get("answer", 0),
        "stray_count": counts.get("stray", 0),
        "think_follows": think_follows(segments),
        "conversion_rule": rule,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="dongguanting/ARPO-SFT-54K")
    p.add_argument("--split", default="train")
    p.add_argument("--output-dir", default="scripts/sft_refactor/output/inspect")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--samples-per-pattern", type=int, default=5)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    samples_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split)
    n = len(ds) if args.limit is None else min(args.limit, len(ds))

    rows_meta = []
    pattern_indices: dict[str, list[int]] = defaultdict(list)
    pattern_rules: dict[str, dict] = {}
    flag_counter: Counter = Counter()

    for idx in tqdm(range(n), desc="Inspecting"):
        ex = ds[idx]
        assistant = extract_assistant(ex)
        meta = analyze_row(idx, assistant)
        rows_meta.append(meta)
        pattern_indices[meta["pattern_id"]].append(idx)
        pattern_rules[meta["pattern_id"]] = meta["conversion_rule"]
        for f in meta["flags"]:
            flag_counter[f] += 1

    taxonomy = {}
    for pid, indices in pattern_indices.items():
        taxonomy[pid] = {
            "count": len(indices),
            "pct": round(100.0 * len(indices) / n, 3),
            "example_indices": indices[: args.samples_per_pattern],
            "conversion_rule": pattern_rules[pid],
        }

    action_counts = Counter(r["conversion_rule"]["action"] for r in rows_meta)

    report = {
        "dataset": args.dataset,
        "split": args.split,
        "total_rows": n,
        "unique_patterns": len(taxonomy),
        "action_counts": dict(action_counts),
        "flag_counts": dict(flag_counter),
        "top_patterns": sorted(
            [{"pattern_id": k, "count": v["count"], "pct": v["pct"]} for k, v in taxonomy.items()],
            key=lambda x: -x["count"],
        )[:30],
    }

    conversion_rules = {pid: info["conversion_rule"] for pid, info in taxonomy.items()}

    with open(os.path.join(args.output_dir, "inspect_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.output_dir, "pattern_taxonomy.json"), "w") as f:
        json.dump(taxonomy, f, indent=2)
    with open(os.path.join(args.output_dir, "conversion_rules.json"), "w") as f:
        json.dump(conversion_rules, f, indent=2)
    with open(os.path.join(args.output_dir, "inspect_index.jsonl"), "w") as f:
        for r in rows_meta:
            f.write(json.dumps({
                "idx": r["idx"],
                "pattern_id": r["pattern_id"],
                "flags": r["flags"],
                "think_count": r["think_count"],
                "search_count": r["search_count"],
                "python_count": r["python_count"],
                "conversion_action": r["conversion_rule"]["action"],
            }) + "\n")

    for pid, indices in tqdm(pattern_indices.items(), desc="Writing samples"):
        safe = pid.replace("→", "_").replace("+", "_")[:120]
        sample_path = os.path.join(samples_dir, f"{safe}.jsonl")
        with open(sample_path, "w") as f:
            for idx in indices[: args.samples_per_pattern]:
                ex = ds[idx]
                f.write(json.dumps({
                    "idx": idx,
                    "pattern_id": pid,
                    "assistant": extract_assistant(ex),
                    "meta": rows_meta[idx],
                }, ensure_ascii=False) + "\n")

    print(f"Inspected {n} rows → {args.output_dir}")
    print(f"  Unique patterns: {len(taxonomy)}")
    print(f"  Actions: {dict(action_counts)}")


if __name__ == "__main__":
    main()
