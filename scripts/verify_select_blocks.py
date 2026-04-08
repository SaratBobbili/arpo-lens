#!/usr/bin/env python3
"""
Verify that the planning <select> block (before the first <think>) lists
exactly the set of tools actually invoked in each trajectory.

Usage:
    python scripts/verify_select_blocks.py test.jsonl
    python scripts/verify_select_blocks.py test.jsonl --verbose
"""

import argparse
import json
import re
import sys
from tqdm import tqdm

# Extracts all <tool> "name" </tool> entries inside a <select> block
_TOOL_IN_SELECT = re.compile(r'<tool>\s*"([^"]+)"\s*</tool>', re.IGNORECASE)

# Matches top-level tool invocation tags (not inside <select> or <tool>)
_TOOL_INVOCATION = re.compile(r'<(search|python|answer)[\s>]', re.IGNORECASE)

TAG_TO_TOOL = {"search": "search", "python": "python", "answer": "no-tool"}


def extract_planning_tools(text: str) -> set[str] | None:
    """Return the set of tools declared in the planning <select> block, or None if missing."""
    first_think = text.find("<think>")
    # Planning block is the <select>...</select> that appears before the first <think>
    prefix = text[:first_think] if first_think != -1 else text
    m = re.search(r'<select>(.*?)</select>', prefix, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    return set(_TOOL_IN_SELECT.findall(m.group(1)))


def extract_used_tools(text: str) -> set[str]:
    """Return the set of tools actually invoked in the trajectory."""
    tags = _TOOL_INVOCATION.findall(text)
    return {TAG_TO_TOOL[t.lower()] for t in tags}


def verify(path: str, verbose: bool = False) -> None:
    with open(path) as f:
        lines = [l for l in f if l.strip()]

    total = len(lines)
    ok, mismatch, no_plan = 0, 0, 0
    mismatches: list[dict] = []

    for i, line in enumerate(tqdm(lines, desc="Verifying")):
        row = json.loads(line)
        conv = row.get("conversations", [])
        assistant = next((t["value"] for t in conv if t["from"] in ("gpt", "assistant")), "")

        planned = extract_planning_tools(assistant)
        used = extract_used_tools(assistant)

        if planned is None:
            no_plan += 1
            if verbose:
                q = next((t["value"] for t in conv if t["from"] in ("human", "user")), "")
                print(f"[{i}] NO PLANNING BLOCK  q={q[:80]}...")
            continue

        if planned == used:
            ok += 1
        else:
            mismatch += 1
            rec = {"line": i, "planned": sorted(planned), "used": sorted(used),
                   "missing": sorted(used - planned), "extra": sorted(planned - used)}
            mismatches.append(rec)
            if verbose:
                q = next((t["value"] for t in conv if t["from"] in ("human", "user")), "")
                print(f"[{i}] MISMATCH  planned={sorted(planned)}  used={sorted(used)}  "
                      f"missing={sorted(used - planned)}  extra={sorted(planned - used)}")
                print(f"     q={q[:100]}")

    print(f"\n{'='*60}")
    print(f"Total: {total}  |  OK: {ok}  |  Mismatch: {mismatch}  |  No planning block: {no_plan}")
    if mismatch:
        print(f"\nMismatch rate: {mismatch}/{ok + mismatch} = {mismatch / (ok + mismatch):.1%}")
    print(f"{'='*60}")

    if mismatches and not verbose:
        print("\nFirst 10 mismatches (use --verbose for full detail):")
        for rec in mismatches[:10]:
            print(f"  line {rec['line']:>5}: planned={rec['planned']}  used={rec['used']}  "
                  f"missing={rec['missing']}  extra={rec['extra']}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jsonl", help="Path to JSONL file to verify")
    p.add_argument("--verbose", "-v", action="store_true", help="Print every mismatch")
    args = p.parse_args()
    verify(args.jsonl, args.verbose)


if __name__ == "__main__":
    main()
