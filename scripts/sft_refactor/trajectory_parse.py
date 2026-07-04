"""Parse ARPO-SFT assistant trajectories into ordered segments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

TAG_ALIASES = {
    "redacted_thinking": "think",
    "think": "think",
    "search": "search",
    "python": "python",
    "result": "result",
    "answer": "answer",
    "tool": "tool",
}

_OPEN_RE = re.compile(
    r"<(redacted_thinking|think|search|python|result|answer|tool)\s*>",
    re.IGNORECASE,
)


@dataclass
class Segment:
    kind: str  # think | search | python | result | answer | tool | stray | other
    content: str
    start: int
    end: int
    raw_tag: str = ""


def _close_tag(raw: str) -> str:
    return f"</{raw}>"


def parse_segments(text: str) -> List[Segment]:
    """Split assistant text into ordered segments."""
    if not text:
        return []

    segments: List[Segment] = []
    pos = 0
    n = len(text)

    while pos < n:
        m = _OPEN_RE.search(text, pos)
        if not m:
            tail = text[pos:n]
            if tail.strip():
                segments.append(Segment("stray", tail, pos, n))
            break

        if m.start() > pos:
            stray = text[pos : m.start()]
            if stray.strip():
                segments.append(Segment("stray", stray, pos, m.start()))

        raw_tag = m.group(1).lower()
        kind = TAG_ALIASES.get(raw_tag, "other")
        content_start = m.end()
        close = _close_tag(raw_tag)
        close_idx = text.lower().find(close.lower(), content_start)
        if close_idx == -1:
            segments.append(Segment(kind, text[content_start:n], m.start(), n, raw_tag))
            break

        content_end = close_idx
        segments.append(
            Segment(kind, text[content_start:content_end], m.start(), close_idx + len(close), raw_tag)
        )
        pos = close_idx + len(close)

    return segments


def tag_sequence(segments: List[Segment]) -> List[str]:
    return [s.kind for s in segments if s.kind != "stray"]


def pattern_id_from_segments(segments: List[Segment]) -> str:
    """Canonical pattern signature, e.g. think→search→result→think→answer."""
    kinds = tag_sequence(segments)
    if not kinds:
        return "empty"
    base = "→".join(kinds)
    stray = [s for s in segments if s.kind == "stray"]
    if stray:
        base += "+stray"
    return base


def count_tags(segments: List[Segment]) -> dict:
    counts = {k: 0 for k in ("think", "search", "python", "result", "answer", "tool", "stray", "other")}
    for s in segments:
        counts[s.kind] = counts.get(s.kind, 0) + 1
    return counts


def think_follows(segments: List[Segment]) -> List[str]:
    """For each think segment, what tag follows it (or 'none' / 'stray')."""
    out = []
    for i, s in enumerate(segments):
        if s.kind != "think":
            continue
        nxt = "none"
        for j in range(i + 1, len(segments)):
            if segments[j].kind == "stray":
                continue
            nxt = segments[j].kind
            break
        out.append(nxt)
    return out


def structural_flags(segments: List[Segment], text: str) -> List[str]:
    flags: List[str] = []
    kinds = tag_sequence(segments)
    counts = count_tags(segments)

    if not kinds:
        flags.append("empty_trajectory")
        return flags

    if kinds[0] not in ("think",):
        flags.append("first_tag_not_think")

    if counts["search"] + counts["python"] > 0:
        first_tool = next((k for k in kinds if k in ("search", "python")), None)
        first_think = next((i for i, k in enumerate(kinds) if k == "think"), None)
        first_tool_i = next((i for i, k in enumerate(kinds) if k in ("search", "python")), None)
        if first_tool and first_think is not None and first_tool_i is not None and first_tool_i < first_think:
            flags.append("tool_before_think")

    open_think = text.lower().count("<think>")
    close_think = text.lower().count("</think>")
    if open_think != close_think:
        flags.append("unpaired_think")

    for tag in ("search", "python", "result", "answer"):
        lo = tag.lower()
        if text.lower().count(f"<{lo}>") != text.lower().count(f"</{lo}>"):
            flags.append(f"unpaired_{tag}")

    if counts["answer"] == 0:
        flags.append("missing_answer")
    elif counts["answer"] > 1:
        flags.append("multiple_answer")

    if counts["answer"] >= 1 and "\\boxed{" not in text:
        flags.append("missing_boxed")

    # search/python must be followed by result
    kinds_full = [s.kind for s in segments]
    for i, k in enumerate(kinds_full):
        if k not in ("search", "python"):
            continue
        rest = kinds_full[i + 1 :]
        rest_nonstray = [x for x in rest if x != "stray"]
        if not rest_nonstray or rest_nonstray[0] != "result":
            flags.append(f"missing_result_after_{k}")

    if counts["stray"] > 0:
        flags.append("has_stray_text")

    return flags


def split_units(segments: List[Segment]) -> List[dict]:
    """Think blocks whose next non-stray tag is search, python, or answer."""
    units = []
    for i, s in enumerate(segments):
        if s.kind != "think":
            continue
        nxt_kind = None
        nxt_idx = None
        for j in range(i + 1, len(segments)):
            if segments[j].kind == "stray":
                continue
            nxt_kind = segments[j].kind
            nxt_idx = j
            break
        if nxt_kind not in ("search", "python", "answer"):
            continue
        units.append({
            "think_idx": i,
            "next_idx": nxt_idx,
            "next_kind": nxt_kind,
            "think_content": s.content,
        })
    return units


def extract_assistant(row: dict) -> str:
    if isinstance(row.get("assistant"), str):
        return row["assistant"]
    for turn in row.get("conversations") or []:
        role = turn.get("from") or turn.get("role")
        if role in ("gpt", "assistant"):
            return turn.get("value") or turn.get("content") or ""
    return ""


def reassemble_after_split(segments: List[Segment], splits: dict[int, dict]) -> str:
    """Insert <tool> blocks after split think segments. splits: think_idx -> {thinking, tool_rationale}."""
    parts: List[str] = []
    for i, s in enumerate(segments):
        if s.kind == "stray":
            continue
        if s.kind == "think" and i in splits:
            sp = splits[i]
            parts.append(f"<think>{sp['thinking']}</think>")
            parts.append(f"<tool>{sp['tool_rationale']}</tool>")
        elif s.kind == "think":
            parts.append(f"<think>{s.content}</think>")
        elif s.kind == "search":
            parts.append(f"<search>{s.content}</search>")
        elif s.kind == "python":
            parts.append(f"<python>{s.content}</python>")
        elif s.kind == "result":
            parts.append(f"<result>{s.content}</result>")
        elif s.kind == "answer":
            parts.append(f"<answer>{s.content}</answer>")
        elif s.kind == "tool":
            parts.append(f"<tool>{s.content}</tool>")
    return "".join(parts)
