#!/usr/bin/env python3
"""
Verify that trajectories do not invoke tools before the first think step.

Checks each JSONL row and flags:
- `<search>` / `<python>` appears but no `<think>` exists.
- first `<search>` / `<python>` appears before first `<think>`.

Supports rows with either:
- `assistant` (preprocessed SFT format), or
- `conversations` with `from: gpt|assistant` turns.

Usage:
    python scripts/verify_think_before_tool.py LLaMA-Factory/arpo_train_sft/dataset/echo_sft.jsonl
    python scripts/verify_think_before_tool.py ... --verbose
"""

import argparse
import json
import re

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_: dict):
        return iterable

_TOOL_RE = re.compile(r"<(search|python)[\s>]", re.IGNORECASE)
_THINK_RE = re.compile(r"<think[\s>]", re.IGNORECASE)


def extract_assistant_text(row: dict) -> str:
    assistant = row.get("assistant")
    if isinstance(assistant, str):
        return assistant

    conversations = row.get("conversations")
    if isinstance(conversations, list):
        for turn in conversations:
            role = turn.get("from") or turn.get("role")
            if role in ("gpt", "assistant"):
                value = turn.get("value") or turn.get("content") or ""
                if isinstance(value, str):
                    return value
    return ""


def _snippet(text: str, idx: int, radius: int = 40) -> str:
    return text[max(0, idx - radius) : idx + 120].replace("\n", " ")


def verify(path: str, verbose: bool = False, max_examples: int = 20) -> None:
    total = 0
    clean = 0
    missing_assistant = 0
    no_think_with_tool = 0
    tool_before_think = 0
    examples: list[dict] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(tqdm(f, desc="Verifying")):
            if not line.strip():
                continue

            total += 1
            row = json.loads(line)
            text = extract_assistant_text(row)
            if not text:
                missing_assistant += 1
                if verbose:
                    print(f"[{line_idx}] MISSING_ASSISTANT_TEXT")
                continue

            tool_match = _TOOL_RE.search(text)
            if tool_match is None:
                clean += 1
                continue

            think_match = _THINK_RE.search(text)
            if think_match is None:
                no_think_with_tool += 1
                if len(examples) < max_examples:
                    examples.append(
                        {
                            "line": line_idx,
                            "kind": "NO_THINK_WITH_TOOL",
                            "first_tool": tool_match.group(1).lower(),
                            "snippet": _snippet(text, tool_match.start()),
                        }
                    )
                continue

            if tool_match.start() < think_match.start():
                tool_before_think += 1
                if len(examples) < max_examples:
                    examples.append(
                        {
                            "line": line_idx,
                            "kind": "TOOL_BEFORE_THINK",
                            "first_tool": tool_match.group(1).lower(),
                            "tool_pos": tool_match.start(),
                            "think_pos": think_match.start(),
                            "snippet": _snippet(text, tool_match.start()),
                        }
                    )
                continue

            clean += 1

    flagged = no_think_with_tool + tool_before_think
    print(f"\n{'=' * 72}")
    print(
        f"Total checked: {total}  |  Clean: {clean}  |  Flagged: {flagged}  "
        f"|  Missing assistant text: {missing_assistant}"
    )
    print(
        f"  - No <think> but has tool call: {no_think_with_tool}\n"
        f"  - Tool call before first <think>: {tool_before_think}"
    )
    if total:
        print(f"Flag rate: {flagged}/{total} = {flagged / total:.1%}")
    print(f"{'=' * 72}")

    if examples:
        print(f"\nFirst {len(examples)} flagged examples:")
        for ex in examples:
            if ex["kind"] == "TOOL_BEFORE_THINK":
                print(
                    f"  line {ex['line']:>6}  {ex['kind']}  tool={ex['first_tool']}  "
                    f"tool_pos={ex['tool_pos']}  think_pos={ex['think_pos']}"
                )
            else:
                print(f"  line {ex['line']:>6}  {ex['kind']}  tool={ex['first_tool']}")
            print(f"      snippet: {ex['snippet']}")

    if verbose and not examples:
        print("No flagged examples found.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jsonl", help="Path to JSONL file to verify")
    p.add_argument("--verbose", "-v", action="store_true", help="Print per-row missing-assistant notes")
    p.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="How many flagged examples/snippets to print",
    )
    args = p.parse_args()
    verify(args.jsonl, args.verbose, args.max_examples)


if __name__ == "__main__":
    main()
