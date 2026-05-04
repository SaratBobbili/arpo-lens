"""
Pure-Python ping-pong rollout state machine for ECHO validator profile c1.

Profile c1 phase ownership of close tags:
  HL = {first_select, think, answer}     -> </select> (first only), </think>, </answer>
  LL = {select (after first), search, python} -> </select> (subsequent), </search>, </python>
"result" is excluded from both phases (tool output, mask 0 in training).

Per-sample turn:
  1. Active engine generates with stop=ALL_CLOSE_TAGS, include_stop_str_in_output=True.
     With this combo, vLLM stops at the *first* close tag emitted, so each turn
     yields at most one close tag and the new tokens always end with that tag
     (or hit max_tokens with no stop).
  2. If the close tag is owned by the active phase, the new tokens (text + ids)
     are appended to the canonical response history. If LL closed </search> or
     </python>, the corresponding tool is run and "<result>...</result>" is
     appended with mask 0. Phase is then handed to the other side.
  3. If not owned (or finish_reason=length with no stop), discard the new tokens
     entirely (no progress) and hand off to the other phase. Two consecutive
     no-progress turns terminate the sample.
  4. Termination conditions:
       (a) HL produced a closed </answer>            -> done="answer"
       (b) total response_len >= max_response_length -> done="length"
       (c) turn count >= max_turns                   -> done="turn_cap"
       (d) tool call cap reached on a required tool  -> done="tool_cap"
       (e) two consecutive no-progress turns         -> done="no_progress"

The state machine is engine-free here; the caller (shard_worker) owns the two
vLLM engines and invokes `prepare_turn_request` / `apply_turn_output` per
sample per step.
"""

from dataclasses import dataclass, field
from typing import Optional

# Phase tags align with the trainer's reward_model.phase_order entries.
PHASE_HL = "high_level"
PHASE_LL = "low_level"

# Close tags vLLM stops on every turn, regardless of the active phase. The
# strip-back / no-progress logic below decides whether the matched stop is
# owned by the active phase.
ALL_CLOSE_TAGS = ("</select>", "</think>", "</answer>", "</search>", "</python>")

# Owned-close map for profile c1. </select> is special-cased at runtime via the
# select_close_count field (first close goes to HL, subsequent ones to LL).
_OWNED_CLOSE = {
    PHASE_HL: {"</think>", "</answer>"},
    PHASE_LL: {"</search>", "</python>"},
}


def is_owned(close_tag: str, phase: str, select_close_count: int) -> bool:
    """Return True iff `close_tag` belongs to `phase` under c1 ownership.

    `select_close_count` is the number of properly-closed <select> blocks
    accumulated in the response *before* this turn; the first </select>
    overall is HL (first_select), every subsequent one is LL (select).
    """
    if close_tag == "</select>":
        return (phase == PHASE_HL and select_close_count == 0) or (
            phase == PHASE_LL and select_close_count > 0
        )
    return close_tag in _OWNED_CLOSE.get(phase, set())


def other_phase(phase: str) -> str:
    return PHASE_LL if phase == PHASE_HL else PHASE_HL


@dataclass
class SampleState:
    """Mutable per-sample state across ping-pong turns.

    `prompt_ids`     : initial chat-template token ids (frozen).
    `response_ids`   : accumulated response tokens (HL+LL+result), canonical history.
    `result_mask`    : parallel mask: 1 for model-generated tokens, 0 for tool
                       result tokens. Kept for parity / debugging; scoring uses
                       only the decoded response text.
    `phase`          : currently-active phase.
    `turn`           : turn counter (HL+LL combined).
    `select_close_count`: number of properly-closed <select> blocks accepted.
    `tool_calls`     : tool calls already consumed (vs `tool_call_limit`).
    `no_progress`    : consecutive no-progress turns (HL or LL).
    `done`           : None until terminated; set to a short reason string.
    """

    sample_id: int
    prompt_ids: list[int]
    response_ids: list[int] = field(default_factory=list)
    result_mask: list[int] = field(default_factory=list)
    phase: str = PHASE_HL
    turn: int = 0
    select_close_count: int = 0
    tool_calls: int = 0
    no_progress: int = 0
    done: Optional[str] = None

    def context_ids(self) -> list[int]:
        return self.prompt_ids + self.response_ids

    def remaining_response(self, max_response_length: int) -> int:
        return max_response_length - len(self.response_ids)


@dataclass
class TurnOutcome:
    """What happened in one turn for one sample, as decided by `apply_turn_output`."""

    accepted: bool                 # True iff new tokens were kept and phase advanced.
    closed_tag: Optional[str]      # The accepted close tag (if accepted).
    tool_request: Optional[tuple]  # (tool_tag, content) when LL closed search/python; else None.
    finished: bool                 # True iff the sample is now terminated.


def apply_turn_output(
    state: SampleState,
    new_token_ids: list[int],
    new_text: str,
    finish_reason: str,
    stop_reason: Optional[str],
    max_response_length: int,
    no_progress_terminate: int,
) -> TurnOutcome:
    """Update `state` in place after a single engine turn and decide next action.

    Returns a TurnOutcome describing whether tokens were accepted, whether a
    tool needs to be dispatched, and whether the sample is now finished.
    The caller is responsible for actually running the tool (LL only) and for
    appending its <result> tokens via `append_tool_result`.
    """
    state.turn += 1

    closed = stop_reason if (finish_reason == "stop" and stop_reason in ALL_CLOSE_TAGS) else None
    owned = closed is not None and is_owned(closed, state.phase, state.select_close_count)

    if not owned:
        state.no_progress += 1
        if state.no_progress >= no_progress_terminate:
            state.done = "no_progress"
            return TurnOutcome(accepted=False, closed_tag=None, tool_request=None, finished=True)
        state.phase = other_phase(state.phase)
        return TurnOutcome(accepted=False, closed_tag=closed, tool_request=None, finished=False)

    # Accept and append the new tokens.
    state.response_ids.extend(new_token_ids)
    state.result_mask.extend([1] * len(new_token_ids))
    state.no_progress = 0
    if closed == "</select>":
        state.select_close_count += 1

    # Hard length cap (training-side response_length). Treat any overflow as
    # terminated; we keep the tokens up to the cap as the final response.
    if len(state.response_ids) >= max_response_length:
        state.response_ids = state.response_ids[:max_response_length]
        state.result_mask = state.result_mask[:max_response_length]
        state.done = "length"
        return TurnOutcome(accepted=True, closed_tag=closed, tool_request=None, finished=True)

    # HL closed answer -> rollout terminator.
    if closed == "</answer>" and state.phase == PHASE_HL:
        state.done = "answer"
        return TurnOutcome(accepted=True, closed_tag=closed, tool_request=None, finished=True)

    # LL closed a tool tag: caller must extract content from the *response* text
    # (since the close tag we just accepted was the latest in the response) and
    # run the tool. We pass the close tag back so the caller can find the
    # matching open and pull content.
    tool_request = None
    if state.phase == PHASE_LL and closed in ("</search>", "</python>"):
        tag = closed[2:-1]  # "search" / "python"
        tool_request = (tag, _extract_last_block(new_text, tag))

    # Switch phase.
    state.phase = other_phase(state.phase)
    return TurnOutcome(accepted=True, closed_tag=closed, tool_request=tool_request, finished=False)


def _extract_last_block(text: str, tag: str) -> str:
    """Return the content of the last <tag>...</tag> block in `text`.

    Mirrors `vLLMRolloutECHO._extract_content` semantics: if the tag is not
    properly bracketed, returns "" (caller treats this as an empty tool input,
    which the BingSearchTool / PythonTool handle gracefully).
    """
    open_t, close_t = f"<{tag}>", f"</{tag}>"
    end = text.rfind(close_t)
    if end == -1:
        return ""
    start = text.rfind(open_t, 0, end)
    if start == -1:
        return ""
    return text[start + len(open_t):end].strip()


def append_tool_result(
    state: SampleState,
    encoded_result_ids: list[int],
    max_response_length: int,
) -> bool:
    """Append "<result>...</result>" tokens (already encoded by the caller).

    Returns True iff the sample is still active afterwards (False -> finished
    by length cap and `state.done` is set).
    """
    state.response_ids.extend(encoded_result_ids)
    state.result_mask.extend([0] * len(encoded_result_ids))
    state.tool_calls += 1
    if len(state.response_ids) >= max_response_length:
        state.response_ids = state.response_ids[:max_response_length]
        state.result_mask = state.result_mask[:max_response_length]
        state.done = "length"
        return False
    return True


def hit_tool_cap(state: SampleState, tool_call_limit: int) -> bool:
    return state.tool_calls >= tool_call_limit


def hit_turn_cap(state: SampleState, max_turns: int) -> bool:
    return state.turn >= max_turns
