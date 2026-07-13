"""Shared constants for SFT trajectory refactor pipeline."""

from pathlib import Path

import yaml

NEW_SYSTEM_PROMPT = (
    "You are a helpful assistant that solves the given question step by step "
    "with a wikipedia search tool and a python interpreter tool. "
    "First reason about the problem in <think>...</think>. "
    "Before every tool call, write a concise but substantive rationale in <tool>...</tool> that "
    "captures what the call will do, why it is the right step at this point, and what you expect "
    "it to return given what you know so far. "
    "Put search queries in <search>...</search>, code in <python>...</python>, "
    "and tool outputs in <result>...</result>. "
    "You may make several tool calls; each one must be preceded by its own <tool> rationale. "
    "When ready to finish, write a final <tool>...</tool> explaining why the accumulated results "
    "are sufficient and no further tool is needed, "
    "then give the answer in <answer>...</answer> with the exact answer in \\boxed{} LaTeX format."
)

# Toggle which key from ECHO/training/config/echo_system_prompts.yaml is used by `export`.
ACTIVE_SYSTEM_PROMPT = "system_prompt_5"
_ECHO_SYSTEM_PROMPTS_YAML = (
    Path(__file__).resolve().parents[2] / "ECHO" / "training" / "config" / "echo_system_prompts.yaml"
)
with open(_ECHO_SYSTEM_PROMPTS_YAML) as _f:
    ECHO_SYSTEM_PROMPT = yaml.safe_load(_f)[ACTIVE_SYSTEM_PROMPT].strip()

GPT_SYSTEM = (
    "You reorganize and enrich an existing agent reasoning trajectory. "
    "You separate GENERAL problem-solving reasoning (<think>) from PER-ACTION tool-selection "
    "rationale (<tool>). You are an editor/enhancer, NOT a solver.\n\n"
    "You are given a question, the ordered <think> blocks, and the ordered actions "
    "(each a search query, python code, or the final answer). For each think block produce a "
    "cleaned/enriched thinking string, and for each action produce a rationale.\n\n"
    "Rules:\n"
    "- thinking: general reasoning only (decompose the question, interpret results, decide "
    "direction). Remove tool-selection phrasing like 'I will search' / 'let me run code'. "
    "You may clarify and expand it, but stay consistent with the source reasoning. If a think "
    "block is marked as having no source (the trajectory begins directly with an action), WRITE "
    "the opening reasoning that frames the problem and sets up the first step.\n"
    "- tool rationale: CONCISE but qualitatively rich — usually one sentence, at most two. Length "
    "is NOT the goal; insight is. Do not pad, do not restate the query, do not write boilerplate. "
    "It should read as a natural continuation of the <think> block right before it and capture "
    "WHAT this call does, WHY it is the right move here, and HOW it builds on what is already "
    "known. For the action that is the final answer, briefly say why the accumulated results are "
    "now sufficient and no further tool is needed.\n"
    "- NEVER change or regenerate the answer, the search queries, the python code, or tool "
    "results — those are fixed. Do not fabricate a different solution path. You MAY prepend or "
    "enrich reasoning to make the trajectory read coherently.\n\n"
    "Return ONLY valid JSON (no markdown fences, no extra text):\n"
    '{"thinks": ["cleaned think 1", "..."], "tools": ["why action 1", "why action 2", "..."]}\n'
    "len(thinks) must equal the number of think blocks; len(tools) must equal the number of actions."
)

GPT_REVIEW_SYSTEM = (
    "You audit a fully assembled agent trajectory of the form "
    "<think>/<tool>/<search|python>/<result>/.../<tool>/<answer> for REASONING COHERENCE and "
    "RATIONALE QUALITY, not just structural validity.\n\n"
    "Check that: the <think> blocks form one logical progression (including a coherent opening "
    "block); each <tool> is concise but qualitatively rich — it flows from the <think> right "
    "before it and conveys what the call does, why it is the right step now, and how it builds on "
    "prior results, WITHOUT padding, boilerplate, or a terse restatement of the query; the calls "
    "make sense in sequence; the pre-answer <tool> truly follows from the accumulated results; and "
    "everything stays consistent with the original problem and solution path.\n\n"
    "Constraint: NEVER alter the answer, search queries, python code, or tool results, and do not "
    "re-solve the problem. Only the <think> and <tool> texts may be revised to restore coherence, "
    "to tighten a verbose rationale, or to enrich a thin/boilerplate one.\n\n"
    "Return ONLY valid JSON (no markdown fences, no extra text). If the trajectory is coherent and "
    'every rationale is substantive, return {"ok": true}. Otherwise return {"thinks": [...], '
    '"tools": [...]} with the SAME counts as the input, carrying the needed revisions.'
)

# Tags recognized by the segment parser (raw tag name -> canonical kind).
TAG_ALIASES = {
    "redacted_thinking": "think",
    "think": "think",
    "search": "search",
    "python": "python",
    "result": "result",
    "answer": "answer",
    "tool": "tool",
}

# Canonical tags allowed in a valid refactored trajectory.
ALLOWED_TAGS = ("think", "tool", "search", "python", "result", "answer")
