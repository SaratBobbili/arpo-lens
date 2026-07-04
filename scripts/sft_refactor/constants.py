"""Shared constants for SFT trajectory refactor pipeline."""

NEW_SYSTEM_PROMPT = (
    "You are a helpful assistant that can solve the given question step by step "
    "with the help of the wikipedia search tool and python interpreter tool. "
    "Given a question, first reason in <think>...</think>. "
    "If you need a tool, write a brief tool-selection rationale in <tool>...</tool> "
    "immediately before the tool call. Search queries go in <search>...</search>, "
    "code in <python>...</python>, and outputs in <result>...</result>. "
    "You may repeat think → tool → call → result cycles. "
    "When ready to finish, reason in <think>...</think>, "
    "optionally explain in <tool>...</tool> why no further tool is needed, "
    "then give the final answer in <answer>...</answer> "
    "with the exact answer in \\boxed{} LaTeX format."
)

GPT_SYSTEM = (
    "You split mixed agent reasoning trajectories into thinking vs tool-selection rationale. "
    "Return ONLY a valid JSON array of objects — no markdown fences, no extra text. "
    'Each object: {"thinking": "...", "tool_rationale": "..."}.'
)

KNOWN_TAGS = ("redacted_thinking", "search", "python", "result", "answer", "tool")
