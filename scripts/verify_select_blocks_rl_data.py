#!/usr/bin/env python3
"""
Verify RL rollout JSONL: each line must have a non-empty `output` field, and the
planning <select> block (before the first <redacted_thinking>) must list exactly
the tools invoked via <search>, <python>, or <answer> in that trajectory.

Usage:
    python scripts/verify_select_blocks_rl_data.py rollout/35.jsonl
    python scripts/verify_select_blocks_rl_data.py rollout/35.jsonl --verbose
"""

import argparse
import json
import re

from tqdm import tqdm

_TOOL_IN_SELECT = re.compile(r'<tool>\s*"([^"]+)"\s*</tool>', re.IGNORECASE)
_TOOL_INVOCATION = re.compile(r'<(search|python|answer)[\s>]', re.IGNORECASE)
TAG_TO_TOOL = {"search": "search", "python": "python", "answer": "no-tool"}


def extract_planning_tools(text: str) -> set[str] | None:
    first_think = text.find("<redacted_thinking>")
    prefix = text[:first_think] if first_think != -1 else text
    m = re.search(r"<select>(.*?)</select>", prefix, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    return set(_TOOL_IN_SELECT.findall(m.group(1)))


def extract_used_tools(text: str) -> set[str]:
    tags = _TOOL_INVOCATION.findall(text)
    return {TAG_TO_TOOL[t.lower()] for t in tags}


def verify(path: str, verbose: bool = False) -> None:
    with open(path) as f:
        lines = [ln for ln in f if ln.strip()]

    total = len(lines)
    ok = mismatch = no_plan = missing_out = 0
    mismatches: list[dict] = []

    for i, line in enumerate(tqdm(lines, desc="Verifying")):
        row = json.loads(line)
        text = row.get("output")
        if not text:
            missing_out += 1
            if verbose:
                print(f"[{i}] MISSING_OR_EMPTY_OUTPUT")
            continue

        planned = extract_planning_tools(text)
        used = extract_used_tools(text)

        if planned is None:
            no_plan += 1
            if verbose:
                print(f"[{i}] NO_PLANNING_BLOCK  (no <select>...</select> before first <redacted_thinking>)")
            continue

        if planned == used:
            ok += 1
        else:
            mismatch += 1
            rec = {
                "line": i,
                "planned": sorted(planned),
                "used": sorted(used),
                "missing": sorted(used - planned),
                "extra": sorted(planned - used),
            }
            mismatches.append(rec)
            if verbose:
                print(
                    f"[{i}] MISMATCH  planned={sorted(planned)}  used={sorted(used)}  "
                    f"missing={sorted(used - planned)}  extra={sorted(planned - used)}"
                )

    print(f"\n{'='*60}")
    print(
        f"Total: {total}  |  OK: {ok}  |  Mismatch: {mismatch}  "
        f"|  No planning block: {no_plan}  |  Missing/empty output: {missing_out}"
    )
    checked = ok + mismatch
    if checked and mismatch:
        print(f"Mismatch rate: {mismatch}/{checked} = {mismatch / checked:.1%}")
    print(f"{'='*60}")

    if mismatches and not verbose:
        print("\nFirst 10 mismatches (use --verbose for full detail):")
        for rec in mismatches[:10]:
            print(
                f"  line {rec['line']:>5}: planned={rec['planned']}  used={rec['used']}  "
                f"missing={rec['missing']}  extra={rec['extra']}"
            )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jsonl", help="Path to RL rollout JSONL (expects `output` per line)")
    p.add_argument("--verbose", "-v", action="store_true", help="Print every issue")
    args = p.parse_args()
    verify(args.jsonl, args.verbose)


if __name__ == "__main__":
    main()
