#!/usr/bin/env python3
"""
Augment ARPO-SFT-54K with <select> tool-selection rationale blocks via GPT.

Parses each assistant trajectory to find </think> -> <search|python|answer>
transitions and asks GPT to generate contextual rationales, inserting
<select> rationale <tool> "tool-name" </tool> </select> blocks.

Input format  (HuggingFace):  { conversations: [{from, value}], system: str }
Output format (same schema):  system prompt replaced, assistant augmented with <select> blocks.

Usage:
    export OPENAI_API_KEY="sk-..."
    python scripts/add_select_rationales.py --model gpt-4o-mini --concurrency 50

    # Test on 5 rows first
    python scripts/add_select_rationales.py --model gpt-4o-mini --limit 5 --jsonl test.jsonl
"""

import argparse
import asyncio
import json
import os
import re
from typing import List

import pandas as pd
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm import tqdm

# ── System prompt: adds planning <select> (multi-tool) before first <think> ─────
NEW_SYSTEM_PROMPT = (
    'You are a helpful assistant that can solve the given question step by step '
    'with the help of the wikipedia search tool and python interpreter tool. '
    'Given a question, you need to first think about the reasoning process in the mind '
    'and then provide the answer. Before you begin thinking, you must decide the set of tools '
    'you would be using for your reasoning process in a planning <select> block that may list '
    'multiple <tool> tags. You can invoke the wikipedia search tool '
    'to search for factual information about specific topics if needed OR python interpreter tool '
    'to calculate a math problem OR directly reason with no tool usage. If you decide to use a tool, '
    'you MUST reason about the tool selection separately as a part of the tool reasoning process. '
    'The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags '
    'respectively. The tool selection rationale and tool calls are enclosed within '
    '<select> this is the tool selection reasoning <tool> "tool-name" </tool> </select> where '
    '"tool-name" can be "search", "python", or "no-tool". If the tool chosen is either "search" or '
    '"python", the tool call and result are enclosed within <search> </search> or <python> </python> '
    'and <result> </result> tags respectively. For example, '
    '<select> This question requires factual lookup and computation. '
    '<tool> "search" </tool> <tool> "python" </tool> </select> '
    '<think> This is the reasoning process. '
    '</think> <select> This is the tool selection reasoning process. <tool> "search" </tool> </select> '
    '<search> search query here </search> <result> search result here </result> '
    '<think> This is the reasoning process. </think> '
    '<select> This is the tool selection reasoning process. <tool> "python" </tool> </select> '
    '<python> python code here </python> <result> python interpreter result here </result> '
    '<think> This is the reasoning process. </think> '
    '<select> This is the tool selection reasoning process. <tool> "no-tool" </tool> </select> '
    '<answer> The final answer is \\\\[ \\\\boxed{answer here} \\\\] </answer>. '
    'In the last part of the answer, the final exact answer is enclosed within \\\\boxed{} with latex format.'
)

# ── Regex: </think> OR </result> followed (optionally with whitespace) by <search|python|answer>
_TRANSITION = re.compile(r'(</(?:think|result)>)(\s*)(<(?:search|python|answer))', re.IGNORECASE)

# Maps closing tag to its opening tag + offset past the '>'
_OPEN_TAGS = {"think": ("<think>", 7), "result": ("<result>", 8)}


# ─────────────────────────── trajectory parsing ──────────────────────────────

def find_transitions(text: str) -> List[dict]:
    """Locate all </think|result> -> <tool|answer> transition points in a trajectory."""
    out = []
    for m in _TRANSITION.finditer(text):
        tag = re.match(r'<(\w+)', m.group(3)).group(1).lower()
        tool = "no-tool" if tag == "answer" else tag

        close_tag = re.match(r'</(\w+)>', m.group(1)).group(1).lower()
        open_str, offset = _OPEN_TAGS[close_tag]

        close_pos = m.start(1)
        open_pos = text.rfind(open_str, 0, close_pos)
        ctx = ""
        if open_pos != -1:
            ctx = text[open_pos + offset : close_pos].strip()

        out.append({
            "start": m.end(1),      # right after </think> or </result>
            "end":   m.start(3),    # right before <search|python|answer>
            "tool":  tool,
            "ctx":   ctx[-400:],    # last 400 chars — keeps the GPT prompt compact
        })
    return out


def stitch(text: str, transitions: List[dict], rationales: List[str]) -> str:
    """Insert per-step <select> blocks at each transition (reverse order preserves positions)."""
    result = text
    for t, r in reversed(list(zip(transitions, rationales))):
        block = f' <select> {r} <tool> "{t["tool"]}" </tool> </select> '
        result = result[:t["start"]] + block + result[t["end"]:]
    return result


def _assemble(
    text: str,
    transitions: List[dict],
    tools_used: List[str],
    rationales: List[str],
) -> str:
    """Insert planning <select> block (before first <think>) + per-step <select> blocks."""
    # Per-step blocks (rationales[1:] correspond to transitions)
    result = stitch(text, transitions, rationales[1:])

    # Planning block: multi-tool <tool> tags
    tool_tags = " ".join(f'<tool> "{t}" </tool>' for t in tools_used)
    planning = f'<select> {rationales[0]} {tool_tags} </select> '
    first_think = result.find("<think>")
    if first_think != -1:
        result = result[:first_think] + planning + result[first_think:]
    else:
        result = planning + result
    return result


# ──────────────────────────── GPT interaction ────────────────────────────────

GPT_SYSTEM = (
    "You generate tool-selection rationales for agentic reasoning trajectories. "
    "Return ONLY a valid JSON array of strings — no markdown fences, no extra text."
)


def build_rationale_prompt(question: str, transitions: List[dict], tools_used: List[str]) -> str:
    """Prompt GPT to produce a planning rationale (element 0) + one per-transition rationale."""
    n = len(transitions) + 1
    lines = [
        "A reasoning trajectory answers the question below. The model first plans which "
        "tools it will use (Transition 0), then at each subsequent point it stops thinking "
        "and invokes a specific tool (or answers directly). "
        "For each transition write a rationale explaining WHY that tool was chosen.\n",
        "Guidelines:",
        "- Transition 0 (planning): 1-2 sentences on why the listed tools are needed for this question.",
        '- For "search" or "python": 1-2 sentences on why this tool is useful right now.',
        '- For "no-tool": one short decisive sentence (e.g. "Sufficient information to answer.").'
        " Do NOT summarize prior results or restate the answer.\n",
        f"Question: {question}\n",
        f'Transition 0 (planning) → tools: {json.dumps(tools_used)}',
        "  Given the question, explain why this set of tools is appropriate.\n",
    ]
    for i, t in enumerate(transitions, 1):
        lines.append(f"Transition {i} → \"{t['tool']}\"")
        lines.append(f"  Preceding reasoning (truncated): ...{t['ctx']}\n")
    lines.append(
        f"Return a JSON array of exactly {n} rationale strings, in order "
        f"(element 0 = planning, elements 1..{n - 1} = per-step)."
    )
    return "\n".join(lines)


def _strip_fences(raw: str) -> str:
    """Remove optional ```json ... ``` wrappers from GPT output."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r'^```\w*\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
    return raw


async def augment_one(
    client: AsyncOpenAI,
    model: str,
    question: str,
    assistant: str,
    sem: asyncio.Semaphore,
    retries: int = 3,
) -> str:
    """Augment a single trajectory with planning + per-step <select> blocks."""
    transitions = find_transitions(assistant)
    if not transitions:
        return assistant

    tools_used = sorted(set(t["tool"] for t in transitions))
    n_expected = len(transitions) + 1  # 1 planning + N per-step
    prompt = build_rationale_prompt(question, transitions, tools_used)

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
                    max_tokens=1024,
                )
            rationales = json.loads(_strip_fences(resp.choices[0].message.content))

            # Pad if GPT returned fewer than expected
            while len(rationales) < n_expected:
                idx = len(rationales)
                if idx == 0:
                    rationales.append(f"Planning to use {', '.join(tools_used)}.")
                else:
                    rationales.append(
                        f"Selecting {transitions[idx - 1]['tool']} for this reasoning step."
                    )
            return _assemble(assistant, transitions, tools_used, rationales[:n_expected])
        except Exception:
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)

    # All retries failed — generic fallback so we never drop a row
    fallback = [f"Planning to use {', '.join(tools_used)}."]
    fallback += [f"Selecting {t['tool']} for this reasoning step." for t in transitions]
    return _assemble(assistant, transitions, tools_used, fallback)


# ────────────────────────────── main pipeline ────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    client = AsyncOpenAI(api_key=args.api_key or os.environ.get("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(args.concurrency)

    print("Loading dataset …")
    ds = load_dataset(args.dataset, split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    n = len(ds)

    # ── checkpoint / resume ──
    ckpt_path = args.output + ".ckpt.jsonl"
    done: dict[int, dict] = {}
    if os.path.exists(ckpt_path) and not args.fresh:
        with open(ckpt_path) as f:
            for line in f:
                rec = json.loads(line)
                done[rec["idx"]] = rec
        print(f"Resuming: {len(done)}/{n} already processed")

    ckpt_lock = asyncio.Lock()
    ckpt_f = open(ckpt_path, "a")
    pbar = tqdm(total=n, initial=len(done), desc="Augmenting")

    async def process(idx: int) -> dict:
        if idx in done:
            return done[idx]

        ex = ds[idx]
        conv = ex["conversations"]
        q = next((t["value"] for t in conv if t["from"] in ("human", "user")), "")
        a = next((t["value"] for t in conv if t["from"] in ("gpt", "assistant")), "")

        aug = await augment_one(client, args.model, q, a, sem)

        rec = {
            "idx": idx,
            "system": NEW_SYSTEM_PROMPT,
            "conversations": [
                {"from": "human", "value": q},
                {"from": "gpt", "value": aug},
            ],
        }
        async with ckpt_lock:
            ckpt_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        pbar.update(1)
        return rec

    results = await asyncio.gather(*(process(i) for i in range(n)))
    pbar.close()
    ckpt_f.close()

    # Sort by original index, drop idx field
    results.sort(key=lambda r: r["idx"])
    for r in results:
        r.pop("idx")

    df = pd.DataFrame(results)
    df.to_parquet(args.output, index=False)
    print(f"Saved {len(df)} rows → {args.output}")

    if args.jsonl:
        with open(args.jsonl, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved JSONL  → {args.jsonl}")

    os.remove(ckpt_path)
    print("Done.")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # data
    p.add_argument("--dataset", default="dongguanting/ARPO-SFT-54K",
                    help="HuggingFace dataset id or local path")
    p.add_argument("--split", default="train")
    p.add_argument("--output", default="arpo_sft_with_select.parquet",
                    help="Output parquet file")
    p.add_argument("--jsonl", default=None,
                    help="Also write a JSONL copy (for inspection)")
    # model
    p.add_argument("--model", default="gpt-4o-mini",
                    help="OpenAI model for rationale generation")
    p.add_argument("--api-key", default=None,
                    help="OpenAI API key (or set OPENAI_API_KEY env var)")
    # throughput
    p.add_argument("--concurrency", type=int, default=50,
                    help="Max concurrent API calls")
    # debug
    p.add_argument("--limit", type=int, default=None,
                    help="Process only the first N rows")
    p.add_argument("--fresh", action="store_true",
                    help="Ignore existing checkpoint and start from scratch")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
