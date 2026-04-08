#!/usr/bin/env python3
"""
Offline preprocessor for dongguanting/ARPO-SFT-54K: canonical trajectory layout

    reasoning (think) -> <tool_rationale>...</tool_rationale> -> <search|python>...</> -> ...

No tool calls: explicit <no_tool_needed /> after reasoning (before <answer> when present).

Outputs Parquet (and optional JSONL) with system, user, assistant columns for SFT loaders.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import List, Tuple

# Marker for trajectories that complete without search/python tools (reasoning-only path).
NO_TOOL_MARKER = "<no_tool_needed />"

# Inserts before each first-time tool invocation.
_TOOL_OPEN_RE = re.compile(r"<(search|python)>", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")
_TRAJECTORY_TAG_RE = re.compile(r"<(think|search|python|answer)[\s>]", re.IGNORECASE)
def _pop_rationale(prefix: str) -> Tuple[str, str]:
    """Split segment before a tool into kept reasoning and rationale for tool choice."""
    prefix = prefix.rstrip()
    if not prefix:
        return "", "Use the tool to retrieve facts or run computation as required."
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(prefix) if p.strip()]
    if len(parts) >= 2:
        return " ".join(parts[:-1]), parts[-1]
    return "", parts[0]


def _find_matching_close(text: str, start: int, tag: str) -> int:
    close = f"</{tag}>"
    j = text.lower().find(close.lower(), start)
    return j + len(close) if j != -1 else len(text)


def insert_tool_rationales(assistant: str) -> str:
    """Insert <tool_rationale>...</tool_rationale> before each <search> / <python> block."""
    if not assistant:
        return assistant

    out: List[str] = []
    pos = 0
    while pos < len(assistant):
        m = _TOOL_OPEN_RE.search(assistant, pos)
        if not m:
            out.append(assistant[pos:])
            break
        prefix = assistant[pos : m.start()]
        tool_lc = m.group(1).lower()
        # Skip if this tool already follows a tool_rationale (re-run safe).
        tail = assistant[m.start() :]
        if prefix.rstrip().endswith("</tool_rationale>"):
            end_close = _find_matching_close(assistant, m.start(), tool_lc)
            out.append(assistant[pos:end_close])
            pos = end_close
            continue

        main, rationale = _pop_rationale(prefix)
        out.append(main)
        if main and not main.endswith("\n"):
            out.append("\n")
        out.append("<tool_rationale>\n")
        out.append(rationale)
        out.append("\n</tool_rationale>\n")
        end_close = _find_matching_close(assistant, m.start(), tool_lc)
        out.append(assistant[m.start() : end_close])
        pos = end_close
    return "".join(out)


def has_tool_invocation(text: str) -> bool:
    return bool(_TOOL_OPEN_RE.search(text))


def insert_no_tool_marker(assistant: str) -> str:
    """Append NO_TOOL_MARKER when no <search>/<python> appears (reasoning-only trajectory)."""
    if has_tool_invocation(assistant):
        return assistant
    if NO_TOOL_MARKER in assistant:
        return assistant
    lo = assistant.lower()
    if "<answer>" in lo:
        idx = lo.index("<answer>")
        return assistant[:idx].rstrip() + "\n" + NO_TOOL_MARKER + "\n" + assistant[idx:]
    return assistant.rstrip() + "\n" + NO_TOOL_MARKER + "\n"


def transform_assistant(assistant: str) -> str:
    """Full pipeline: tool rationales, then no-tool marker when applicable."""
    t = insert_tool_rationales(assistant)
    t = insert_no_tool_marker(t)
    return t


def conversations_to_messages(
    conversations: list,
) -> Tuple[str, str, str]:
    """Returns (system_hint, user_text, assistant_text) from ShareGPT-style rows."""
    user_text = ""
    assistant_text = ""
    for turn in conversations:
        role = turn.get("from") or turn.get("role")
        val = turn.get("value") or turn.get("content") or ""
        if role in ("human", "user"):
            user_text = val
        elif role in ("gpt", "assistant"):
            assistant_text = val
    return user_text, assistant_text


def starts_with_think_reasoning(assistant: str) -> bool:
    """True when the first trajectory action tag is <think>."""
    first = _TRAJECTORY_TAG_RE.search(assistant)
    return bool(first and first.group(1).lower() == "think")


def build_records(example: dict, system_key: str = "system") -> dict:
    system = example.get(system_key) or ""
    conv = example["conversations"]
    user_text, assistant_orig = conversations_to_messages(conv)
    assistant_new = transform_assistant(assistant_orig)
    return {
        "system": system,
        "user": user_text,
        "assistant": assistant_new,
        "assistant_original": assistant_orig,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="dongguanting/ARPO-SFT-54K",
        help="HF dataset id.",
    )
    parser.add_argument("--split", default="train", help="Split name.")
    parser.add_argument(
        "--output",
        default="arpo_sft_trajectories_v2.parquet",
        help="Output Parquet path.",
    )
    parser.add_argument(
        "--jsonl",
        default="arpo_sft_trajectories_v2.jsonl",
        help="Optional JSONL path (same rows).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N rows (debug).",
    )
    parser.add_argument(
        "--drop-original",
        action="store_true",
        help="Do not keep assistant_original in the table.",
    )
    parser.add_argument(
        "--keep-non-think-first",
        action="store_true",
        help="Keep rows even when the first trajectory tag is not <think>.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream the HF split (lower RAM; tqdm has no total).",
    )
    args = parser.parse_args()

    from datasets import load_dataset
    from tqdm.auto import tqdm

    rows = []
    dropped_non_think_first = 0
    if args.streaming:
        ds = load_dataset(args.dataset, split=args.split, streaming=True)
        n = 0
        _pbar_kw = {"desc": "trajectories"}
        if args.limit is not None:
            _pbar_kw["total"] = args.limit
        for ex in tqdm(ds, **_pbar_kw):
            if args.limit is not None and n >= args.limit:
                break
            _user_text, assistant_orig = conversations_to_messages(ex["conversations"])
            if (not args.keep_non_think_first) and (not starts_with_think_reasoning(assistant_orig)):
                dropped_non_think_first += 1
                n += 1
                continue
            rec = build_records(ex)
            if args.drop_original:
                del rec["assistant_original"]
            rows.append(rec)
            n += 1
    else:
        ds = load_dataset(args.dataset, split=args.split)
        if args.limit is not None:
            ds = ds.select(range(min(args.limit, len(ds))))
        for i in tqdm(range(len(ds)), desc="trajectories"):
            ex = ds[i]
            _user_text, assistant_orig = conversations_to_messages(ex["conversations"])
            if (not args.keep_non_think_first) and (not starts_with_think_reasoning(assistant_orig)):
                dropped_non_think_first += 1
                continue
            rec = build_records(ex)
            if args.drop_original:
                del rec["assistant_original"]
            rows.append(rec)

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_parquet(args.output, index=False)
    if args.jsonl:
        with open(args.jsonl, "w", encoding="utf-8") as f:
            for rec in rows:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"Wrote {len(rows)} rows to {args.output}. "
        f"Dropped non-think-first rows: {dropped_non_think_first}."
    )


if __name__ == "__main__":
    main()
