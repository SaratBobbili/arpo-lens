#!/usr/bin/env python3
"""
ARPO-SFT-54K trajectory refactor CLI.

Subcommands:
    inspect   full-dataset pattern scan -> taxonomy + conversion rules
    split     two-pass GPT think/tool split -> refactored trajectories
    verify    structural validation of a JSONL
    prune     drop rows failing verification
    export    swap in ECHO system prompt -> SFT-ready ShareGPT jsonl

Examples:
    python scripts/sft_refactor/refactor.py inspect --output-dir output/inspect
    python scripts/sft_refactor/refactor.py split --inspect-dir output/inspect \\
        --output output/echo_sft_v3.parquet --jsonl output/echo_sft_v3.jsonl
    python scripts/sft_refactor/refactor.py verify output/echo_sft_v3.jsonl
    python scripts/sft_refactor/refactor.py prune output/echo_sft_v3.jsonl -o clean.jsonl
    python scripts/sft_refactor/refactor.py export output/echo_sft_v3_clean.jsonl -o output/echo_sft_v3_final.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from tqdm import tqdm

from constants import ECHO_SYSTEM_PROMPT, GPT_REVIEW_SYSTEM, GPT_SYSTEM, NEW_SYSTEM_PROMPT
from trajectory import (
    action_units,
    count_tags,
    extract_assistant,
    extract_question,
    normalize,
    parse_segments,
    pattern_id_from_segments,
    reassemble,
    structural_flags,
    tag_sequence,
    think_blocks,
    think_follows,
    verify_text,
)

# Only truly malformed rows are dropped. Tool-first / tool-before-think rows are salvaged
# (leading <think> prepended); missing-answer rows are salvaged when a trailing <think> holds
# a \boxed{} conclusion (promoted to <answer>).
UNPAIRED_FLAGS = {
    "unpaired_think",
    "unpaired_search",
    "unpaired_python",
    "unpaired_result",
    "unpaired_answer",
}

# Hardcoded API credentials live in secrets/credentials.json (git-ignored); used as CLI defaults.
_SECRETS_FILE = Path(__file__).resolve().parent / "secrets" / "credentials.json"
CREDS = json.loads(_SECRETS_FILE.read_text()) if _SECRETS_FILE.exists() else {}


def load_data(dataset: str, split: str):
    from datasets import load_dataset

    return load_dataset(dataset, split=split)


# --------------------------------------------------------------------------- #
# inspect
# --------------------------------------------------------------------------- #

def infer_conversion_rule(flags: list[str]) -> dict:
    """Drop only unrecoverable rows; everything else (incl. salvageable) goes through GPT split."""
    # A trailing boxed conclusion is promoted to <answer>; the stray tag fragments it leaves behind
    # (which trip unpaired_* / missing_answer) are stripped during reassembly, so salvage it.
    if "trailing_boxed" not in flags and (
        "empty_trajectory" in flags
        or "multiple_answer" in flags
        or "missing_answer" in flags
        or any(f in flags for f in UNPAIRED_FLAGS)
    ):
        return {"action": "drop", "strip_stray": False}
    return {"action": "gpt_split", "strip_stray": "has_stray_text" in flags}


def analyze_row(idx: int, assistant: str) -> dict:
    segments = parse_segments(assistant)
    counts = count_tags(segments)
    flags = structural_flags(segments, assistant)
    return {
        "idx": idx,
        "pattern_id": pattern_id_from_segments(segments),
        "tag_sequence": tag_sequence(segments),
        "flags": flags,
        "think_count": counts.get("think", 0),
        "search_count": counts.get("search", 0),
        "python_count": counts.get("python", 0),
        "result_count": counts.get("result", 0),
        "answer_count": counts.get("answer", 0),
        "stray_count": counts.get("stray", 0),
        "think_follows": think_follows(segments),
        "conversion_rule": infer_conversion_rule(flags),
    }


def cmd_inspect(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    samples_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    ds = load_data(args.dataset, args.split)
    n = len(ds) if args.limit is None else min(args.limit, len(ds))

    rows_meta = []
    pattern_indices: dict[str, list[int]] = defaultdict(list)
    pattern_rules: dict[str, dict] = {}
    flag_counter: Counter = Counter()

    for idx in tqdm(range(n), desc="Inspecting"):
        meta = analyze_row(idx, extract_assistant(ds[idx]))
        rows_meta.append(meta)
        pattern_indices[meta["pattern_id"]].append(idx)
        pattern_rules[meta["pattern_id"]] = meta["conversion_rule"]
        for f in meta["flags"]:
            flag_counter[f] += 1

    taxonomy = {
        pid: {
            "count": len(indices),
            "pct": round(100.0 * len(indices) / n, 3),
            "example_indices": indices[: args.samples_per_pattern],
            "conversion_rule": pattern_rules[pid],
        }
        for pid, indices in pattern_indices.items()
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
        with open(os.path.join(samples_dir, f"{safe}.jsonl"), "w") as f:
            for idx in indices[: args.samples_per_pattern]:
                f.write(json.dumps({
                    "idx": idx,
                    "pattern_id": pid,
                    "assistant": extract_assistant(ds[idx]),
                    "meta": rows_meta[idx],
                }, ensure_ascii=False) + "\n")

    print(f"Inspected {n} rows → {args.output_dir}")
    print(f"  Unique patterns: {len(taxonomy)}")
    print(f"  Actions: {dict(action_counts)}")


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #

def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return raw


def _loads_lenient(raw: str) -> dict:
    """Parse model JSON; on failure treat every stray backslash as literal LaTeX (\\frac, \\boxed,
    \\() and retry. Escapes all backslashes except before a quote, so \\b/\\f/\\n are not misread."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(re.sub(r'\\(?=[^"])', r"\\\\", raw))


def _last_sentence(text: str) -> str:
    parts = [p.strip() for p in re.split(r"(?<=[.!?。！？])\s+", text.rstrip()) if p.strip()]
    return parts[-1] if parts else ""


def heuristic_split(thinks: List[str], units: List[dict]) -> tuple[List[str], List[str]]:
    """No-API fallback: keep thinks; last sentence of nearest think for first action, else default."""
    tools: List[str] = []
    prev_think: Optional[str] = None
    for u in units:
        cur = u["think"]
        if cur and cur != prev_think:
            prev_think = cur
            rationale = _last_sentence(cur) or "Take this step to make progress."
        elif u["kind"] == "answer":
            rationale = "All necessary information has been gathered; ready to answer directly."
        else:
            rationale = "Continue with the next tool call to make progress."
        tools.append(rationale)
    thinks_out = [t if t.strip() else "Let me work through this problem step by step." for t in thinks]
    return thinks_out, tools


def build_split_prompt(question: str, segments) -> str:
    """Ordered, interleaved view so each tool rationale can follow the think block just above it."""
    merged = normalize(segments)
    lines = [f"Question:\n{question}\n", "Trajectory in order:"]
    ti = aj = 0
    for s in merged:
        if s.kind == "think":
            body = s.content.strip()
            lines.append(
                f"[think {ti}] {body[-1500:]}" if body
                else f"[think {ti}] (no source — write the opening reasoning that frames the problem)"
            )
            ti += 1
        elif s.kind in ("search", "python", "answer"):
            label = "final answer" if s.kind == "answer" else f"{s.kind} call"
            lines.append(f"[action {aj}] ({label}) {s.content[:1500]}")
            aj += 1
        elif s.kind == "result":
            lines.append(f"    (result) {s.content[:400].strip()}")
    lines.append(
        f'\nReturn JSON: {{"thinks": [{ti} strings], "tools": [{aj} strings]}}. '
        "thinks[i] rewrites [think i]; tools[j] justifies [action j] and must read as a natural "
        "continuation of the think block just above it."
    )
    return "\n".join(lines)


def build_review_prompt(question: str, assembled: str) -> str:
    return (
        f"Question:\n{question}\n\nAssembled trajectory:\n{assembled}\n\n"
        'Return {"ok": true} if coherent, else revised {"thinks": [...], "tools": [...]} '
        "with the same counts."
    )


def _fit(values: List[str], n: int, filler: List[str]) -> List[str]:
    values = list(values)[:n]
    while len(values) < n:
        i = len(values)
        values.append(filler[i] if i < len(filler) else "")
    return values


async def _gpt_json(client, model, system, prompt, timeout, label="", retries=3):
    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=4096,
                timeout=timeout,
            )
            return _loads_lenient(_strip_fences(resp.choices[0].message.content))
        except Exception as e:
            tqdm.write(f"[gpt:{label}] attempt {attempt + 1}/{retries} failed: {type(e).__name__}: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
    tqdm.write(f"[gpt:{label}] gave up after {retries} attempts -> heuristic fallback")
    return None


async def split_row(client, args, segments, question: str) -> Optional[str]:
    units, think_count = action_units(segments)
    if not units:
        return None
    thinks_src = think_blocks(segments)

    if args.heuristic_only or client is None:
        thinks, tools = heuristic_split(thinks_src, units)
        return reassemble(segments, thinks, tools)

    _, heur_tools = heuristic_split(thinks_src, units)
    data = await _gpt_json(
        client, args.model, GPT_SYSTEM, build_split_prompt(question, segments), args.timeout, "breakdown"
    )
    if data:
        thinks = _fit(data.get("thinks", []), think_count, thinks_src)
        tools = _fit(data.get("tools", []), len(units), heur_tools)
    else:
        thinks, tools = list(thinks_src), heur_tools

    assembled = reassemble(segments, thinks, tools)

    if args.coherence_review:
        review = await _gpt_json(
            client, args.model, GPT_REVIEW_SYSTEM, build_review_prompt(question, assembled),
            args.timeout, "review",
        )
        if review and not review.get("ok"):
            r_thinks = _fit(review.get("thinks", thinks), think_count, thinks)
            r_tools = _fit(review.get("tools", tools), len(units), tools)
            revised = reassemble(segments, r_thinks, r_tools)
            if verify_text(revised)[0]:
                assembled = revised

    return assembled


async def run_split(args: argparse.Namespace) -> None:
    with open(os.path.join(args.inspect_dir, "inspect_index.jsonl")) as f:
        index = [json.loads(ln) for ln in f if ln.strip()]

    if args.heuristic_only:
        args.coherence_review = False

    client = None
    if not args.heuristic_only:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=args.base_url or None,
            timeout=args.timeout,
        )
        print(f"GPT client: model={args.model} base_url={args.base_url or 'default'} "
              f"timeout={args.timeout}s concurrency={args.concurrency}")

    ds = load_data(args.dataset, args.split)
    if args.limit:
        index = index[: args.limit]

    ckpt_path = args.output + ".ckpt.jsonl"
    if args.fresh and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    done: set[int] = set()
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            for line in f:
                done.add(json.loads(line)["idx"])

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ckpt_f = open(ckpt_path, "a")
    lock = asyncio.Lock()
    pbar = tqdm(total=len(index), initial=len(done), desc="Splitting")
    stats = {"gpt_split": 0, "drop": 0, "failed": 0}

    async def process(meta: dict) -> None:
        idx = meta["idx"]
        if idx in done:
            return
        if meta.get("conversion_action", "gpt_split") == "drop":
            async with lock:
                stats["drop"] += 1
                pbar.update(1)
            return

        ex = ds[idx]
        question = extract_question(ex)
        segments = parse_segments(extract_assistant(ex))
        new_assistant = await split_row(client, args, segments, question)

        async with lock:
            if new_assistant is None:
                stats["failed"] += 1
            else:
                stats["gpt_split"] += 1
                rec = {
                    "idx": idx,
                    "system": NEW_SYSTEM_PROMPT,
                    "conversations": [
                        {"from": "human", "value": question},
                        {"from": "gpt", "value": new_assistant},
                    ],
                }
                ckpt_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ckpt_f.flush()
            pbar.update(1)

    # Bounded worker pool: only `concurrency` rows in flight at once (prep is lazy, each row is
    # checkpointed as it finishes), so progress is steady and never stalls on task-creation flood.
    cursor = 0

    async def worker() -> None:
        nonlocal cursor
        while True:
            async with lock:
                if cursor >= len(index):
                    return
                meta = index[cursor]
                cursor += 1
            await process(meta)

    await asyncio.gather(*(worker() for _ in range(max(1, args.concurrency))))
    pbar.close()
    ckpt_f.close()

    with open(ckpt_path) as f:
        results = [json.loads(line) for line in f if line.strip()]
    results.sort(key=lambda r: r["idx"])
    for r in results:
        r.pop("idx", None)

    pd.DataFrame(results).to_parquet(args.output, index=False)
    print(f"Saved {len(results)} rows → {args.output}")
    print(f"Stats: {stats}")

    if args.jsonl:
        with open(args.jsonl, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved JSONL → {args.jsonl}")

    if os.path.exists(ckpt_path) and not args.keep_ckpt:
        os.remove(ckpt_path)


def cmd_split(args: argparse.Namespace) -> None:
    asyncio.run(run_split(args))


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #

def cmd_verify(args: argparse.Namespace) -> None:
    with open(args.jsonl) as f:
        lines = [ln for ln in f if ln.strip()]

    ok, bad = 0, 0
    reasons: dict[str, int] = {}
    for i, line in enumerate(tqdm(lines, desc="Verifying")):
        valid, reason = verify_text(extract_assistant(json.loads(line)))
        if valid:
            ok += 1
        else:
            bad += 1
            reasons[reason] = reasons.get(reason, 0) + 1
            if args.verbose:
                print(f"[{i}] {reason}")

    print(f"\nTotal: {len(lines)}  OK: {ok}  Bad: {bad}")
    if reasons:
        print("Reasons:", dict(sorted(reasons.items(), key=lambda x: -x[1])))


# --------------------------------------------------------------------------- #
# prune
# --------------------------------------------------------------------------- #

def cmd_prune(args: argparse.Namespace) -> None:
    kept = []
    dropped = 0
    reasons: dict[str, int] = {}
    with open(args.jsonl) as f:
        for line in tqdm(f, desc="Pruning"):
            if not line.strip():
                continue
            row = json.loads(line)
            ok, reason = verify_text(extract_assistant(row))
            if ok:
                kept.append(row)
            else:
                dropped += 1
                reasons[reason] = reasons.get(reason, 0) + 1

    print(f"Kept: {len(kept)}  Dropped: {dropped}")
    if reasons:
        print("Drop reasons:", dict(sorted(reasons.items(), key=lambda x: -x[1])))

    if args.output.endswith(".parquet"):
        pd.DataFrame(kept).to_parquet(args.output, index=False)
    else:
        with open(args.output, "w") as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved → {args.output}")


def cmd_export(args: argparse.Namespace) -> None:
    rows = []
    with open(args.jsonl) as f:
        for line in tqdm(f, desc="Exporting"):
            if not line.strip():
                continue
            row = json.loads(line)
            row["system"] = ECHO_SYSTEM_PROMPT
            rows.append(row)

    if args.output.endswith(".parquet"):
        pd.DataFrame(rows).to_parquet(args.output, index=False)
    else:
        with open(args.output, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Exported {len(rows)} rows → {args.output}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _add_dataset_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", default="dongguanting/ARPO-SFT-54K")
    p.add_argument("--split", default="train")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="full-dataset pattern scan")
    _add_dataset_args(pi)
    pi.add_argument("--output-dir", default="scripts/sft_refactor/output/inspect")
    pi.add_argument("--limit", type=int, default=None)
    pi.add_argument("--samples-per-pattern", type=int, default=5)
    pi.set_defaults(func=cmd_inspect)

    ps = sub.add_parser("split", help="two-pass GPT think/tool split")
    _add_dataset_args(ps)
    ps.add_argument("--inspect-dir", required=True)
    ps.add_argument("--output", default="scripts/sft_refactor/output/echo_sft_v3.parquet")
    ps.add_argument("--jsonl", default=None)
    ps.add_argument("--model", default=CREDS.get("model", "gpt-oss"))
    ps.add_argument("--api-key", default=CREDS.get("api_key"))
    ps.add_argument("--base-url", default=CREDS.get("base_url"))
    ps.add_argument("--timeout", type=float, default=120.0, help="per-request timeout (s)")
    ps.add_argument("--concurrency", type=int, default=50)
    ps.add_argument("--limit", type=int, default=None)
    ps.add_argument("--fresh", action="store_true")
    ps.add_argument("--keep-ckpt", action="store_true")
    ps.add_argument("--heuristic-only", action="store_true", help="skip GPT API (pipeline testing)")
    ps.add_argument("--coherence-review", dest="coherence_review", action="store_true", default=True)
    ps.add_argument("--no-coherence-review", dest="coherence_review", action="store_false")
    ps.set_defaults(func=cmd_split)

    pv = sub.add_parser("verify", help="structural validation of a JSONL")
    pv.add_argument("jsonl")
    pv.add_argument("--verbose", "-v", action="store_true")
    pv.set_defaults(func=cmd_verify)

    pp = sub.add_parser("prune", help="drop rows failing verification")
    pp.add_argument("jsonl")
    pp.add_argument("-o", "--output", required=True)
    pp.set_defaults(func=cmd_prune)

    pe = sub.add_parser("export", help="swap in ECHO system prompt -> SFT-ready ShareGPT jsonl")
    pe.add_argument("jsonl", nargs="?", default="scripts/sft_refactor/output/echo_sft_v3_clean.jsonl")
    pe.add_argument("-o", "--output", default="scripts/sft_refactor/output/echo_sft_v3_final.jsonl")
    pe.set_defaults(func=cmd_export)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
