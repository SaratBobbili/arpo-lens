#!/usr/bin/env python3
"""
Structural validation for refactored SFT trajectories.

Usage:
    python scripts/sft_refactor/verify_format.py output.jsonl
    python scripts/sft_refactor/verify_format.py output.jsonl --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tqdm import tqdm

from trajectory_parse import extract_assistant, parse_segments

_ALLOWED = {"think", "tool", "search", "python", "result", "answer"}


def verify_text(text: str) -> tuple[bool, str]:
    if not text.strip():
        return False, "empty"

    segments = parse_segments(text)
    kinds = [s.kind for s in segments]
    if any(s.kind == "stray" for s in segments):
        return False, "stray_text"

    if not kinds:
        return False, "no_tags"

    if kinds[0] != "think":
        return False, "first_tag_not_think"

    if kinds.count("answer") != 1:
        return False, f"answer_count={kinds.count('answer')}"

    if "\\boxed{" not in text:
        return False, "missing_boxed"

    for k in kinds:
        if k not in _ALLOWED:
            return False, f"disallowed_tag:{k}"

    # tool must precede search, python, or answer
    for i, k in enumerate(kinds):
        if k != "tool":
            continue
        if i + 1 >= len(kinds) or kinds[i + 1] not in ("search", "python", "answer"):
            return False, "tool_not_before_call"

    # search/python -> result
    for i, k in enumerate(kinds):
        if k in ("search", "python") and (i + 1 >= len(kinds) or kinds[i + 1] != "result"):
            return False, f"{k}_not_followed_by_result"

    # no search/python before first think
    first_think = kinds.index("think")
    if any(k in ("search", "python") for k in kinds[:first_think]):
        return False, "tool_before_think"

    return True, "ok"


def verify(path: str, verbose: bool = False) -> None:
    ok, bad = 0, 0
    reasons: dict[str, int] = {}

    with open(path) as f:
        lines = [ln for ln in f if ln.strip()]

    for i, line in enumerate(tqdm(lines, desc="Verifying")):
        row = json.loads(line)
        text = extract_assistant(row)
        valid, reason = verify_text(text)
        if valid:
            ok += 1
        else:
            bad += 1
            reasons[reason] = reasons.get(reason, 0) + 1
            if verbose:
                print(f"[{i}] {reason}")

    print(f"\nTotal: {len(lines)}  OK: {ok}  Bad: {bad}")
    if reasons:
        print("Reasons:", dict(sorted(reasons.items(), key=lambda x: -x[1])))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl")
    p.add_argument("--verbose", "-v", action="store_true")
    verify(p.parse_args().jsonl, p.parse_args().verbose)


if __name__ == "__main__":
    main()
