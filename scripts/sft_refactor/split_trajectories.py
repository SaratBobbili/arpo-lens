#!/usr/bin/env python3
"""
GPT-based think/tool split for ARPO-SFT-54K trajectories.

Requires inspect_dataset.py outputs (inspect_index.jsonl + conversion_rules.json).

Usage:
    export OPENAI_API_KEY="sk-..."
    python scripts/sft_refactor/split_trajectories.py \
        --inspect-dir scripts/sft_refactor/output/inspect \
        --output scripts/sft_refactor/output/echo_sft_v3.parquet \
        --jsonl scripts/sft_refactor/output/echo_sft_v3.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from constants import GPT_SYSTEM, NEW_SYSTEM_PROMPT
from trajectory_parse import (
    extract_assistant,
    parse_segments,
    reassemble_after_split,
    split_units,
)


def _pop_rationale(text: str) -> tuple[str, str]:
    """Heuristic fallback: last sentence -> tool rationale."""
    text = text.rstrip()
    if not text:
        return "", "Proceed with the next step."
    parts = re.split(r"(?<=[.!?。！？])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 2:
        return " ".join(parts[:-1]), parts[-1]
    return "", parts[0]


def build_split_prompt(question: str, units: List[dict]) -> str:
    lines = [
        "Split each think block below into general reasoning (thinking) and tool-selection rationale (tool_rationale).",
        "thinking: step reasoning only — no 'I will search', no query/code planning.",
        "tool_rationale: 1-2 sentences on why the upcoming tool or direct answer is appropriate.",
        f"\nQuestion: {question}\n",
    ]
    for i, u in enumerate(units):
        lines.append(f"Block {i} (next action: {u['next_kind']}):")
        lines.append(u["think_content"][-2000:])
        lines.append("")
    lines.append(
        f'Return a JSON array of {len(units)} objects: {{"thinking": "...", "tool_rationale": "..."}}'
    )
    return "\n".join(lines)


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return raw


async def gpt_split(
    client,
    model: str,
    question: str,
    units: List[dict],
    sem: asyncio.Semaphore,
    heuristic_only: bool = False,
    retries: int = 3,
) -> List[dict]:
    if not units:
        return []
    if heuristic_only or client is None:
        return [
            {"thinking": t, "tool_rationale": r}
            for t, r in (_pop_rationale(u["think_content"]) for u in units)
        ]
    prompt = build_split_prompt(question, units)

    for attempt in range(retries):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": GPT_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=4096,
                )
            raw = _strip_fences(resp.choices[0].message.content)
            splits = json.loads(raw)
            while len(splits) < len(units):
                u = units[len(splits)]
                thinking, rationale = _pop_rationale(u["think_content"])
                splits.append({"thinking": thinking, "tool_rationale": rationale})
            return splits[: len(units)]
        except Exception:
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)

    return [
        {"thinking": t, "tool_rationale": r}
        for t, r in (_pop_rationale(u["think_content"]) for u in units)
    ]


def transform_pass_through(assistant: str, rule: dict) -> str:
    segments = parse_segments(assistant)
    if not rule.get("strip_stray"):
        return assistant
    return reassemble_after_split(segments, {})


def transform_gpt_result(segments, units: List[dict], splits: List[dict]) -> str:
    split_map = {}
    for u, sp in zip(units, splits):
        split_map[u["think_idx"]] = {
            "thinking": sp["thinking"],
            "tool_rationale": sp["tool_rationale"],
        }
    return reassemble_after_split(segments, split_map)


async def run(args: argparse.Namespace) -> None:
    with open(os.path.join(args.inspect_dir, "inspect_index.jsonl")) as f:
        index = [json.loads(ln) for ln in f if ln.strip()]
    with open(os.path.join(args.inspect_dir, "conversion_rules.json")) as f:
        rules = json.load(f)

    client = None
    if not args.heuristic_only:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=args.api_key or os.environ.get("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(args.concurrency)

    ds = load_dataset(args.dataset, split=args.split)
    if args.limit:
        index = index[: args.limit]

    ckpt_path = args.output + ".ckpt.jsonl"
    done: dict[int, dict] = {}
    if os.path.exists(ckpt_path) and not args.fresh:
        with open(ckpt_path) as f:
            for line in f:
                rec = json.loads(line)
                done[rec["idx"]] = rec

    if args.fresh and os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ckpt_f = open(ckpt_path, "a")
    ckpt_lock = asyncio.Lock()
    pbar = tqdm(total=len(index), initial=len(done), desc="Splitting")
    stats = {"pass_through": 0, "gpt_split": 0, "drop": 0}

    async def process(meta: dict) -> None:
        idx = meta["idx"]
        if idx in done:
            return

        rule = rules.get(meta["pattern_id"], {"action": "gpt_split"})
        action = rule.get("action", "gpt_split")
        ex = ds[idx]
        conv = ex["conversations"]
        q = next((t["value"] for t in conv if t["from"] in ("human", "user")), "")
        assistant = extract_assistant(ex)

        if action == "drop":
            async with ckpt_lock:
                stats["drop"] += 1
                pbar.update(1)
            return

        segments = parse_segments(assistant)

        if action == "pass_through":
            new_assistant = transform_pass_through(assistant, rule)
            async with ckpt_lock:
                stats["pass_through"] += 1
        else:
            units = split_units(segments)
            splits = await gpt_split(
                client, args.model, q, units, sem, args.heuristic_only
            )
            new_assistant = transform_gpt_result(segments, units, splits)
            async with ckpt_lock:
                stats["gpt_split"] += 1

        rec = {
            "idx": idx,
            "system": NEW_SYSTEM_PROMPT,
            "conversations": [
                {"from": "human", "value": q},
                {"from": "gpt", "value": new_assistant},
            ],
        }
        async with ckpt_lock:
            ckpt_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ckpt_f.flush()
            pbar.update(1)

    await asyncio.gather(*(process(m) for m in index))

    pbar.close()
    ckpt_f.close()

    results = []
    with open(ckpt_path) as f:
        for line in f:
            results.append(json.loads(line))

    results.sort(key=lambda r: r["idx"])
    for r in results:
        r.pop("idx", None)

    df = pd.DataFrame(results)
    df.to_parquet(args.output, index=False)
    print(f"Saved {len(df)} rows → {args.output}")
    print(f"Stats: {stats}")

    if args.jsonl:
        with open(args.jsonl, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved JSONL → {args.jsonl}")

    if os.path.exists(ckpt_path) and not args.keep_ckpt:
        os.remove(ckpt_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inspect-dir", required=True)
    p.add_argument("--dataset", default="dongguanting/ARPO-SFT-54K")
    p.add_argument("--split", default="train")
    p.add_argument("--output", default="scripts/sft_refactor/output/echo_sft_v3.parquet")
    p.add_argument("--jsonl", default=None)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--api-key", default=None)
    p.add_argument("--concurrency", type=int, default=50)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--keep-ckpt", action="store_true")
    p.add_argument(
        "--heuristic-only",
        action="store_true",
        help="Skip GPT API; use last-sentence heuristic split (for pipeline testing)",
    )
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
