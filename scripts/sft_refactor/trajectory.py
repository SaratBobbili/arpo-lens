"""Parse, transform, and validate ARPO-SFT assistant trajectories."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from constants import ALLOWED_TAGS, TAG_ALIASES

_OPEN_RE = re.compile(
    r"<(" + "|".join(TAG_ALIASES) + r")\s*>",
    re.IGNORECASE,
)

_TRUNC = 2000


@dataclass
class Segment:
    kind: str  # think | search | python | result | answer | tool | stray | other
    content: str
    start: int
    end: int
    raw_tag: str = ""


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
        close = f"</{raw_tag}>"
        close_idx = text.lower().find(close.lower(), content_start)
        if close_idx == -1:
            segments.append(Segment(kind, text[content_start:n], m.start(), n, raw_tag))
            break

        segments.append(
            Segment(kind, text[content_start:close_idx], m.start(), close_idx + len(close), raw_tag)
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
    if any(s.kind == "stray" for s in segments):
        base += "+stray"
    return base


def count_tags(segments: List[Segment]) -> dict:
    counts = {k: 0 for k in ("think", "search", "python", "result", "answer", "tool", "stray", "other")}
    for s in segments:
        counts[s.kind] = counts.get(s.kind, 0) + 1
    return counts


def think_follows(segments: List[Segment]) -> List[str]:
    """For each think segment, what tag follows it (or 'none')."""
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

    if kinds[0] != "think":
        flags.append("first_tag_not_think")

    if counts["search"] + counts["python"] > 0:
        first_think = next((i for i, k in enumerate(kinds) if k == "think"), None)
        first_tool_i = next((i for i, k in enumerate(kinds) if k in ("search", "python")), None)
        if first_think is not None and first_tool_i is not None and first_tool_i < first_think:
            flags.append("tool_before_think")

    if text.lower().count("<think>") != text.lower().count("</think>"):
        flags.append("unpaired_think")

    for tag in ("search", "python", "result", "answer"):
        if text.lower().count(f"<{tag}>") != text.lower().count(f"</{tag}>"):
            flags.append(f"unpaired_{tag}")

    if counts["answer"] == 0:
        flags.append("missing_answer")
        if _trailing_boxed_think(merge_consecutive_thinks(segments)) is not None:
            flags.append("trailing_boxed")
    elif counts["answer"] > 1:
        flags.append("multiple_answer")

    if counts["answer"] >= 1 and "\\boxed{" not in text:
        flags.append("missing_boxed")

    kinds_full = [s.kind for s in segments]
    for i, k in enumerate(kinds_full):
        if k not in ("search", "python"):
            continue
        rest_nonstray = [x for x in kinds_full[i + 1 :] if x != "stray"]
        if not rest_nonstray or rest_nonstray[0] != "result":
            flags.append(f"missing_result_after_{k}")

    if counts["stray"] > 0:
        flags.append("has_stray_text")

    return flags


def extract_assistant(row: dict) -> str:
    if isinstance(row.get("assistant"), str):
        return row["assistant"]
    for turn in row.get("conversations") or []:
        role = turn.get("from") or turn.get("role")
        if role in ("gpt", "assistant"):
            return turn.get("value") or turn.get("content") or ""
    return ""


def extract_question(row: dict) -> str:
    for turn in row.get("conversations") or []:
        role = turn.get("from") or turn.get("role")
        if role in ("human", "user"):
            return turn.get("value") or turn.get("content") or ""
    return ""


def merge_consecutive_thinks(segments: List[Segment]) -> List[Segment]:
    """Collapse adjacent think segments and drop stray/other so every think precedes an action."""
    merged: List[Segment] = []
    for s in segments:
        if s.kind in ("stray", "other"):
            continue
        if s.kind == "think" and merged and merged[-1].kind == "think":
            prev = merged[-1]
            merged[-1] = Segment(
                "think", prev.content.rstrip() + "\n\n" + s.content.lstrip(), prev.start, s.end
            )
            continue
        merged.append(s)
    return merged


# Delimiter fragments the parser leaves stranded in a trailing boxed think; stripped before split.
_STRIP_FRAGMENTS = ("</answer>", "<answer>", "</result>", "<result>", "</think>", "<think>")
# Tool call / result content leaking into the trailing think — unsafe to salvage, keep dropped.
_LEAK_MARKERS = ("<result>", "</result>", "<search>", "</search>", "<python>", "</python>")


def _strip_tag_fragments(text: str) -> str:
    for f in _STRIP_FRAGMENTS:
        text = re.sub(re.escape(f), "", text, flags=re.IGNORECASE)
    return text


def _trailing_boxed_think(segs: List[Segment]) -> Segment | None:
    """The trailing <think> holding a \\boxed{} conclusion (no <answer>, no leaked tool content)."""
    if any(s.kind == "answer" for s in segs):
        return None
    last = next((s for s in reversed(segs) if s.kind != "stray"), None)
    if last is None or last.kind != "think" or "\\boxed{" not in last.content:
        return None
    if any(m in last.content.lower() for m in _LEAK_MARKERS):
        return None
    return last


def _split_boxed(text: str) -> tuple[str, str]:
    """Split trailing reasoning into (reasoning, answer) at the paragraph/sentence holding the last \\boxed."""
    bi = text.rfind("\\boxed{")
    para = text.rfind("\n\n", 0, bi)
    if para != -1:
        cut = para
    else:
        sent = max(text.rfind(". ", 0, bi), text.rfind(".\n", 0, bi), text.rfind("。", 0, bi))
        cut = sent + 1 if sent != -1 else 0
    return text[:cut].strip(), text[cut:].strip()


def _promote_trailing_answer(segs: List[Segment]) -> List[Segment]:
    """No <answer> but a trailing <think> holding \\boxed{}: strip stray tag fragments, split think + answer."""
    last = _trailing_boxed_think(segs)
    if last is None:
        return segs
    reasoning, answer = _split_boxed(_strip_tag_fragments(last.content))
    out = [s for s in segs if s is not last]
    if reasoning:
        out.append(Segment("think", reasoning, last.start, last.end))
    out.append(Segment("answer", answer, last.start, last.end))
    return out


def normalize(segments: List[Segment]) -> List[Segment]:
    """Merge thinks, promote a trailing boxed think into an answer, and prepend a leading think if needed."""
    merged = _promote_trailing_answer(merge_consecutive_thinks(segments))
    if merged and merged[0].kind != "think":
        merged.insert(0, Segment("think", "", 0, 0))
    return merged


def action_units(segments: List[Segment]) -> tuple[List[dict], int]:
    """Ordered actions (search/python/answer) with nearest preceding think; plus post-merge think count."""
    merged = normalize(segments)
    think_count = sum(1 for s in merged if s.kind == "think")
    units: List[dict] = []
    last_think = ""
    for s in merged:
        if s.kind == "think":
            last_think = s.content
        elif s.kind in ("search", "python", "answer"):
            units.append(
                {"kind": s.kind, "content": s.content[:_TRUNC], "think": last_think[-_TRUNC:]}
            )
    return units, think_count


def think_blocks(segments: List[Segment]) -> List[str]:
    return [s.content for s in normalize(segments) if s.kind == "think"]


def reassemble(segments: List[Segment], clean_thinks: List[str], tool_rationales: List[str]) -> str:
    """Emit <think> from clean_thinks and a <tool> before every search/python/answer."""
    merged = normalize(segments)
    parts: List[str] = []
    ti = ai = 0
    for s in merged:
        if s.kind == "think":
            content = clean_thinks[ti] if ti < len(clean_thinks) else s.content
            ti += 1
            parts.append(f"<think>{content}</think>")
        elif s.kind in ("search", "python", "answer"):
            rationale = tool_rationales[ai] if ai < len(tool_rationales) else ""
            ai += 1
            parts.append(f"<tool>{rationale}</tool>")
            parts.append(f"<{s.kind}>{s.content}</{s.kind}>")
        elif s.kind == "result":
            parts.append(f"<result>{s.content}</result>")
    return "".join(parts)


def verify_text(text: str) -> tuple[bool, str]:
    """Enforce the target invariants: think→tool→call/answer, call preceded by tool + followed by result."""
    if not text.strip():
        return False, "empty"

    segments = parse_segments(text)
    if any(s.kind == "stray" for s in segments):
        return False, "stray_text"

    kinds = [s.kind for s in segments]
    if not kinds:
        return False, "no_tags"

    for k in kinds:
        if k not in ALLOWED_TAGS:
            return False, f"disallowed_tag:{k}"

    if kinds[0] != "think":
        return False, "first_tag_not_think"

    if kinds.count("answer") != 1:
        return False, f"answer_count={kinds.count('answer')}"

    if "\\boxed{" not in text:
        return False, "missing_boxed"

    for i, k in enumerate(kinds):
        if k == "think" and (i + 1 >= len(kinds) or kinds[i + 1] != "tool"):
            return False, "think_not_before_tool"
        if k == "tool" and (i + 1 >= len(kinds) or kinds[i + 1] not in ("search", "python", "answer")):
            return False, "tool_not_before_call"
        if k in ("search", "python"):
            if i == 0 or kinds[i - 1] != "tool":
                return False, f"{k}_not_after_tool"
            if i + 1 >= len(kinds) or kinds[i + 1] != "result":
                return False, f"{k}_not_before_result"

    ai = kinds.index("answer")
    if ai == 0 or kinds[ai - 1] != "tool":
        return False, "answer_not_after_tool"

    return True, "ok"
