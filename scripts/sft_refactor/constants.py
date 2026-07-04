"""Shared constants for SFT trajectory refactor pipeline."""

NEW_SYSTEM_PROMPT = (
    "You are a helpful assistant that solves the given question step by step "
    "with a wikipedia search tool and a python interpreter tool. "
    "First reason about the problem in <think>...</think>. "
    "Before every tool call, write a substantive rationale in <tool>...</tool> that explains "
    "what the call will do, why it is the right step at this point, and what you expect it to "
    "return or resolve given what you know so far. "
    "Put search queries in <search>...</search>, code in <python>...</python>, "
    "and tool outputs in <result>...</result>. "
    "You may make several tool calls; each one must be preceded by its own <tool> rationale. "
    "When ready to finish, write a final <tool>...</tool> explaining why the accumulated results "
    "are sufficient and no further tool is needed, "
    "then give the answer in <answer>...</answer> with the exact answer in \\boxed{} LaTeX format."
)

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
    "- tool rationale: this must be a SUBSTANTIVE, qualitative justification — not a one-line "
    "restatement. In 2-4 sentences explain WHAT the call does (the specific thing this query or "
    "code computes/retrieves), WHY it is the right move at this exact point (what gap in the "
    "current reasoning it closes, why this over alternatives), and HOW it builds on prior results "
    "and what you expect it to return or resolve. A terse 'search for X' is NOT acceptable. "
    "Ground it in the upcoming query/code, the prior results, and the surrounding reasoning. For "
    "the action that is the final answer, explain why the accumulated results are now sufficient "
    "and no further tool is needed.\n"
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
    "block); each <tool> is a substantive, qualitative justification that says what the call does, "
    "why it is the right step now, and how it builds on prior results — NOT terse boilerplate; the "
    "calls make sense in sequence; the pre-answer <tool> truly follows from the accumulated "
    "results; and everything stays consistent with the original problem and solution path.\n\n"
    "Constraint: NEVER alter the answer, search queries, python code, or tool results, and do not "
    "re-solve the problem. Only the <think> and <tool> texts may be revised to restore coherence "
    "or to make a thin/boilerplate <tool> rationale substantive.\n\n"
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
