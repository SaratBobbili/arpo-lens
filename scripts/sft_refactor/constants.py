"""Shared constants for SFT trajectory refactor pipeline."""

NEW_SYSTEM_PROMPT = (
    "You are a helpful assistant that solves the given question step by step "
    "with a wikipedia search tool and a python interpreter tool. "
    "First reason about the problem in <think>...</think>. "
    "Before every tool call, write a brief rationale in <tool>...</tool> "
    "explaining why that specific call is needed. "
    "Put search queries in <search>...</search>, code in <python>...</python>, "
    "and tool outputs in <result>...</result>. "
    "You may make several tool calls; each one must be preceded by its own <tool> rationale. "
    "When ready to finish, write a final <tool>...</tool> explaining why no further tool is needed, "
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
    "You may clarify and expand it, but stay consistent with the source reasoning.\n"
    "- tool rationale: why THIS specific action is taken now, grounded in the upcoming "
    "query/code, prior results, and the source think's intent. For the action that is the final "
    "answer, explain why no further tool is needed.\n"
    "- NEVER change or regenerate the answer, the search queries, the python code, or tool "
    "results — those are fixed. Do not fabricate a different solution path.\n\n"
    "Return ONLY valid JSON (no markdown fences, no extra text):\n"
    '{"thinks": ["cleaned think 1", "..."], "tools": ["why action 1", "why action 2", "..."]}\n'
    "len(thinks) must equal the number of think blocks; len(tools) must equal the number of actions."
)

GPT_REVIEW_SYSTEM = (
    "You audit a fully assembled agent trajectory of the form "
    "<think>/<tool>/<search|python>/<result>/.../<tool>/<answer> for REASONING COHERENCE, "
    "not just structural validity.\n\n"
    "Check that: the <think> blocks form one logical progression; each <tool> genuinely motivates "
    "its call and the calls make sense in sequence; the pre-answer <tool> truly follows from the "
    "accumulated results; nothing reads as disconnected boilerplate; and everything stays "
    "consistent with the original problem and solution path.\n\n"
    "Constraint: NEVER alter the answer, search queries, python code, or tool results, and do not "
    "re-solve the problem. Only the <think> and <tool> texts may be revised, minimally, to restore "
    "coherence.\n\n"
    "Return ONLY valid JSON (no markdown fences, no extra text). If the trajectory is coherent, "
    'return {"ok": true}. Otherwise return {"thinks": [...], "tools": [...]} with the SAME counts '
    "as the input, carrying minimal revisions."
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
