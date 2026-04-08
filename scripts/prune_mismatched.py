#!/usr/bin/env python3
"""
Remove rows whose planning <select> tools don't match actual tool usage.

Reuses the verification logic from verify_select_blocks.py.

Usage:
    python scripts/prune_mismatched.py input.jsonl -o output.jsonl
    python scripts/prune_mismatched.py input.jsonl -o output.parquet
"""

import argparse
import json

import pandas as pd
from tqdm import tqdm

from verify_select_blocks import extract_planning_tools, extract_used_tools


def prune(input_path: str, output_path: str) -> None:
    with open(input_path) as f:
        lines = [l for l in f if l.strip()]

    kept, dropped, no_plan = [], 0, 0
    for line in tqdm(lines, desc="Pruning"):
        row = json.loads(line)
        conv = row.get("conversations", [])
        assistant = next((t["value"] for t in conv if t["from"] in ("gpt", "assistant")), "")

        planned = extract_planning_tools(assistant)
        if planned is None:
            no_plan += 1
            continue

        used = extract_used_tools(assistant)
        if planned == used:
            kept.append(row)
        else:
            dropped += 1

    print(f"Kept: {len(kept)}  |  Dropped (mismatch): {dropped}  |  Dropped (no plan): {no_plan}")

    if output_path.endswith(".parquet"):
        pd.DataFrame(kept).to_parquet(output_path, index=False)
    else:
        with open(output_path, "w") as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved → {output_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jsonl", help="Input JSONL file")
    p.add_argument("-o", "--output", required=True,
                   help="Output path (.jsonl or .parquet)")
    prune(p.parse_args().jsonl, p.parse_args().output)


if __name__ == "__main__":
    main()
