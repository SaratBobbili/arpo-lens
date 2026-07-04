#!/usr/bin/env python3
"""
Drop rows failing verify_format checks.

Usage:
    python scripts/sft_refactor/prune_invalid.py input.jsonl -o output.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from tqdm import tqdm

from trajectory_parse import extract_assistant
from verify_format import verify_text


def prune(input_path: str, output_path: str) -> None:
    kept = []
    dropped = 0
    reasons: dict[str, int] = {}

    with open(input_path) as f:
        for line in tqdm(f, desc="Pruning"):
            if not line.strip():
                continue
            row = json.loads(line)
            text = extract_assistant(row)
            ok, reason = verify_text(text)
            if ok:
                kept.append(row)
            else:
                dropped += 1
                reasons[reason] = reasons.get(reason, 0) + 1

    print(f"Kept: {len(kept)}  Dropped: {dropped}")
    if reasons:
        print("Drop reasons:", dict(sorted(reasons.items(), key=lambda x: -x[1])))

    if output_path.endswith(".parquet"):
        pd.DataFrame(kept).to_parquet(output_path, index=False)
    else:
        with open(output_path, "w") as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved → {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl")
    p.add_argument("-o", "--output", required=True)
    prune(p.parse_args().jsonl, p.parse_args().output)


if __name__ == "__main__":
    main()
