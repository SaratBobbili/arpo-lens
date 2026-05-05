"""Build mixed prefixes from two ECHO candidate rollouts.

For each example shared by `cand1` and `cand2` (matched by question text), this
script produces a prefix that splices:
  - cand2's first `<select>` block (planning select; declares the allowed tools),
  - then K rounds of {cand1 `<think>`} + {cand2 non-initial `<select>` group}, where
    a "non-initial select group" is the `<select>` plus the immediately following
    `<search>`/`<python>` payload and `<result>` (when present),
  - K = min(L1, L2), L_i = number of non-initial selects in candidate i.

Output: a `test.jsonl` per dataset with rows `{question, answer, prefix}` and a
sibling `splice_stats.jsonl` with per-example diagnostics.

Block parsing reuses `verl.utils.reward_score.deep_research_echo.get_ordered_blocks`
(the same parser used at training time), so HL/LL semantics stay aligned.
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

# `verl` is not installed; arpo-lens repo's verl_arpo_entropy must be on PYTHONPATH.
# The orchestration script (run_xmix.sh) handles this; we just import.
from verl.utils.reward_score.deep_research_echo import get_ordered_blocks


def _split_blocks(text):
    """Return (first_select, ll_groups, think_blocks).

    `first_select` is the first `<select>...</select>` block (planning select).
    `ll_groups` is a list of contiguous block lists, one per non-initial
    `<select>`. Each group is `[select]`, `[select, tool_payload]`, or
    `[select, tool_payload, result]`.
    `think_blocks` is the ordered list of `<think>...</think>` blocks.

    Each block is the 4-tuple emitted by `get_ordered_blocks`:
    `(tag, start, end, content)`. `start`/`end` are character offsets into
    `text`; `text[start:end]` is the full `<tag>...</tag>` raw span.
    """
    # Drop unclosed blocks (end == -1) up front: their `_span` slice would
    # silently chop the last character of the response and the validator
    # would reject them anyway, so they can't enter the splice.
    blocks = [b for b in get_ordered_blocks(text) if b[2] != -1]
    first_select = next((b for b in blocks if b[0] == "select"), None)
    think_blocks = [b for b in blocks if b[0] == "think"]

    ll_groups = []
    seen_first = False
    i = 0
    while i < len(blocks):
        tag = blocks[i][0]
        if tag != "select":
            i += 1
            continue
        if not seen_first:
            seen_first = True
            i += 1
            continue
        group = [blocks[i]]
        j = i + 1
        if j < len(blocks) and blocks[j][0] in ("search", "python"):
            group.append(blocks[j])
            j += 1
            if j < len(blocks) and blocks[j][0] == "result":
                group.append(blocks[j])
                j += 1
        ll_groups.append(group)
        i = j
    return first_select, ll_groups, think_blocks


def _span(text, block):
    # block = (tag, start, end, content); end == -1 marks an unclosed tag (we
    # exclude such candidates upstream). Slicing the inclusive raw span keeps
    # the original whitespace + attribute formatting intact for the validator.
    return text[block[1] : block[2]]


def build_prefix(r1: str, r2: str):
    """Build the mixed prefix string from cand1 and cand2 responses.

    Returns (prefix_text, stats_dict). `prefix_text` is empty (and stats flag
    `valid=False`) when either candidate fails to expose a planning <select>
    or has no non-initial selects, since the splice would be ill-defined.
    """
    _, ll1, thk1 = _split_blocks(r1)
    fs2, ll2, _ = _split_blocks(r2)

    stats = {
        "L1": len(ll1),
        "L2": len(ll2),
        "K": 0,
        "thinks_in_cand1": len(thk1),
        "thinks_used": 0,
        "valid": False,
        "reason": "",
    }

    if fs2 is None:
        stats["reason"] = "cand2 has no planning <select>"
        return "", stats
    if not ll1 or not ll2:
        # Either side has zero non-initial selects; nothing to splice.
        stats["reason"] = "cand1 or cand2 has no non-initial selects"
        return "", stats

    K = min(len(ll1), len(ll2))
    stats["K"] = K
    stats["thinks_used"] = min(K, len(thk1))
    stats["valid"] = True

    parts = [_span(r2, fs2)]
    for k in range(K):
        if k < len(thk1):
            parts.append(_span(r1, thk1[k]))
        for blk in ll2[k]:
            parts.append(_span(r2, blk))
    return "\n".join(parts), stats


def _index_by_question(records):
    """Map question -> record. Drops samples whose question collides; this
    only happens on the rare duplicate val row and we just keep the first."""
    out = {}
    for rec in records:
        q = rec["input"]
        if q not in out:
            out[q] = rec
    return out


def _read_output_json(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand1", required=True, help="cand1 inference output JSON (e.g. .../grpo_mix/grpo_mix_output_1.json)")
    ap.add_argument("--cand2", required=True, help="cand2 inference output JSON")
    ap.add_argument("--dataset_name", required=True, help="Dataset folder name (e.g. grpo_mix); test.jsonl is written under <out>/<dataset_name>/")
    ap.add_argument("--out", required=True, help="Output data root: <out>/<dataset_name>/test.jsonl + <out>/splice_stats.jsonl")
    args = ap.parse_args()

    cand1 = _read_output_json(args.cand1)
    cand2 = _read_output_json(args.cand2)
    by_q2 = _index_by_question(cand2)

    # Iterate cand1's order so the spliced JSONL has a stable, deterministic
    # row order matching the val set the user already saw at training time.
    out_dir = Path(args.out) / args.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    test_path = out_dir / "test.jsonl"
    stats_path = Path(args.out) / "splice_stats.jsonl"

    n_emitted = 0
    n_skipped = 0
    n_valid = 0
    with open(test_path, "w") as ftest, open(stats_path, "w") as fstats:
        for rec in tqdm(cand1, desc="splicing"):
            q = rec["input"]
            ans = rec["answer"]
            r1 = rec.get("output", "") or ""
            mate = by_q2.get(q)
            if mate is None:
                n_skipped += 1
                fstats.write(json.dumps({"question": q, "valid": False, "reason": "no cand2 match"}) + "\n")
                continue
            r2 = mate.get("output", "") or ""
            prefix, stats = build_prefix(r1, r2)
            stats.update({"question": q})
            fstats.write(json.dumps(stats) + "\n")
            if not stats["valid"]:
                n_skipped += 1
                continue
            ftest.write(json.dumps({"question": q, "answer": ans, "prefix": prefix}) + "\n")
            n_emitted += 1
            n_valid += 1

    summary = {
        "n_cand1": len(cand1),
        "n_cand2": len(cand2),
        "n_emitted": n_emitted,
        "n_skipped": n_skipped,
        "n_valid": n_valid,
        "test_jsonl": str(test_path),
        "stats_jsonl": str(stats_path),
    }
    print(json.dumps(summary, indent=2))
    summary_path = Path(args.out) / "splice_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
