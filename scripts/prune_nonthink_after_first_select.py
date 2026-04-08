#!/usr/bin/env python3
"""
Prune rows where the first planning/select step is not followed by <think>.

This is intended for datasets produced by scripts/add_select_rationales.py.
Rows are kept only when the first trajectory action after the first
<select>...</select> block is <think>.

Usage:
    python scripts/prune_nonthink_after_first_select.py input.jsonl -o output.jsonl
    python scripts/prune_nonthink_after_first_select.py input.jsonl -o output.parquet
"""

import argparse
import json
import re

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_: dict):
        return iterable

from verify_think_before_tool import extract_assistant_text

_FIRST_SELECT_RE = re.compile(r"<select>.*?</select>", re.IGNORECASE | re.DOTALL)
_ACTION_RE = re.compile(r"<(select|think|search|python|answer)[\s>]", re.IGNORECASE)
_TOOL_IN_SELECT_RE = re.compile(r'<tool>\s*"([^"]+)"\s*</tool>', re.IGNORECASE)
_TOOL_INVOCATION_RE = re.compile(r"<(search|python|answer)[\s>]", re.IGNORECASE)
_TAG_TO_TOOL = {"search": "search", "python": "python", "answer": "no-tool"}


def follows_think_after_first_select(text: str) -> tuple[bool, str]:
    first_action = _ACTION_RE.search(text)
    if first_action is None:
        return False, "no_action_tag"
    if first_action.group(1).lower() != "select":
        return False, "first_action_not_select"

    first_select = _FIRST_SELECT_RE.search(text)
    if first_select is None:
        return False, "no_first_select"

    next_action = _ACTION_RE.search(text, pos=first_select.end())
    if next_action is None:
        return False, "no_action_after_first_select"

    if next_action.group(1).lower() != "think":
        return False, "non_think_after_first_select"

    return True, "ok"


def extract_tools_declared_in_first_select(text: str) -> set[str] | None:
    first_select = _FIRST_SELECT_RE.search(text)
    if first_select is None:
        return None
    return set(_TOOL_IN_SELECT_RE.findall(first_select.group(0)))


def extract_tools_used(text: str) -> set[str]:
    tags = _TOOL_INVOCATION_RE.findall(text)
    return {_TAG_TO_TOOL[tag.lower()] for tag in tags}


def first_select_tools_match_usage(text: str) -> tuple[bool, str]:
    planned = extract_tools_declared_in_first_select(text)
    if planned is None:
        return False, "no_first_select"

    used = extract_tools_used(text)
    if planned != used:
        return False, "tool_set_mismatch"

    return True, "ok"


def prune(input_path: str, output_path: str) -> None:
    kept: list[dict] = []
    total = 0
    dropped_missing_assistant = 0
    dropped_no_action_tag = 0
    dropped_first_action_not_select = 0
    dropped_no_first_select = 0
    dropped_no_action_after_first_select = 0
    dropped_non_think_after_first_select = 0
    dropped_tool_set_mismatch = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Pruning"):
            if not line.strip():
                continue

            total += 1
            row = json.loads(line)
            assistant = extract_assistant_text(row)
            if not assistant:
                dropped_missing_assistant += 1
                continue

            ok, reason = follows_think_after_first_select(assistant)
            if not ok:
                if reason == "no_first_select":
                    dropped_no_first_select += 1
                elif reason == "no_action_tag":
                    dropped_no_action_tag += 1
                elif reason == "first_action_not_select":
                    dropped_first_action_not_select += 1
                elif reason == "no_action_after_first_select":
                    dropped_no_action_after_first_select += 1
                else:
                    dropped_non_think_after_first_select += 1
                continue

            tools_ok, tools_reason = first_select_tools_match_usage(assistant)
            if not tools_ok:
                if tools_reason == "tool_set_mismatch":
                    dropped_tool_set_mismatch += 1
                else:
                    dropped_no_first_select += 1
                continue

            kept.append(row)

    if output_path.endswith(".parquet"):
        import pandas as pd

        pd.DataFrame(kept).to_parquet(output_path, index=False)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    dropped_total = (
        dropped_missing_assistant
        + dropped_no_action_tag
        + dropped_first_action_not_select
        + dropped_no_first_select
        + dropped_no_action_after_first_select
        + dropped_non_think_after_first_select
        + dropped_tool_set_mismatch
    )
    print(
        f"Total: {total}  |  Kept: {len(kept)}  |  Dropped: {dropped_total}\n"
        f"  - Missing assistant text: {dropped_missing_assistant}\n"
        f"  - No trajectory action tag: {dropped_no_action_tag}\n"
        f"  - First action is not <select>: {dropped_first_action_not_select}\n"
        f"  - No first <select>: {dropped_no_first_select}\n"
        f"  - No action after first <select>: {dropped_no_action_after_first_select}\n"
        f"  - First action after first <select> is not <think>: {dropped_non_think_after_first_select}\n"
        f"  - Tools used do not match first <select> tool set: {dropped_tool_set_mismatch}"
    )
    print(f"Saved -> {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jsonl", help="Input JSONL file")
    p.add_argument("-o", "--output", required=True, help="Output path (.jsonl or .parquet)")
    args = p.parse_args()
    prune(args.jsonl, args.output)


if __name__ == "__main__":
    main()
