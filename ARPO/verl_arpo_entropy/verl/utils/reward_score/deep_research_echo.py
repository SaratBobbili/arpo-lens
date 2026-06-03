import re
import string
from typing import Union, List, Dict, Any, Optional, Tuple
from collections import Counter

VALID_ECHO_TOOLS = {"search", "python", "no-tool"}

# Supported mask_categories -> validator_profile signatures. The profile is the
# single source of truth derived from `actor_rollout_ref.rollout.mask_categories`
# at startup; each profile routes a fixed subset of format checks to HL vs LL.
# "result" is always excluded (never masked).
VALIDATOR_PROFILE_SIGNATURES = {
    # c1: planning/reasoning/answer HL; tool choice + tool payload LL.
    "c1": {"first_select": "high", "select": "low",  "think": "high", "answer": "high", "search": "low",  "python": "low"},
    # c2: same as c1 but non-initial <select> is pulled up to HL.
    "c2": {"first_select": "high", "select": "high", "think": "high", "answer": "high", "search": "low",  "python": "low"},
    # c3: same as c1 but <search>/<python> payloads are HL (prevents LL from
    # entropy-hacking via junk tool-code tokens).
    "c3": {"first_select": "high", "select": "low",  "think": "high", "answer": "high", "search": "high", "python": "high"},
    # c4: only <think>/<answer> are HL; first_select + select + search + python
    # are all LL. Isolates HL to reasoning + final answer and hands the entire
    # tool-planning + tool-payload stack to LL.
    "c4": {"first_select": "low",  "select": "low",  "think": "high", "answer": "high", "search": "low",  "python": "low"},
    # c5: every category is HL. Used by the maxentropy_rl single-phase setup
    # where the LL phase is skipped (rollout budget = 0) and HL owns the full
    # set of format checks.
    "c5": {"first_select": "high", "select": "high", "think": "high", "answer": "high", "search": "high", "python": "high"},
}


def resolve_validator_profile(mask_categories):
    """Match `mask_categories` (dict-like) against the known profile signatures.
    Returns the profile id ("c1"/"c2"/"c3"); raises if no profile matches."""
    observed = {k: str(mask_categories[k]) for k in VALIDATOR_PROFILE_SIGNATURES["c1"]}
    for profile, signature in VALIDATOR_PROFILE_SIGNATURES.items():
        if observed == signature:
            return profile
    raise ValueError(
        f"mask_categories {observed} does not match any supported validator profile. "
        f"Supported profiles: {VALIDATOR_PROFILE_SIGNATURES}"
    )


# ---------------------------------------------------------------------------
# Tag parsing helpers
# ---------------------------------------------------------------------------

def find_tag_blocks(text, tag):
    """Find all <tag>content</tag> occurrences, returning (start, end, content) tuples.
    Returns end=-1 for unclosed tags."""
    blocks = []
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    pos = 0
    while True:
        start = text.find(open_tag, pos)
        if start == -1:
            break
        content_start = start + len(open_tag)
        end = text.find(close_tag, content_start)
        if end == -1:
            blocks.append((start, -1, None))
            break
        blocks.append((start, end + len(close_tag), text[content_start:end]))
        pos = end + len(close_tag)
    return blocks


def extract_tools_from_select(select_content):
    """Extract tool names from <tool>"name"</tool> tags within a select block."""
    tools = []
    pos = 0
    while True:
        start = select_content.find("<tool>", pos)
        if start == -1:
            break
        end = select_content.find("</tool>", start)
        if end == -1:
            break
        tools.append(select_content[start + 6:end].strip().strip('"').strip("'"))
        pos = end + 7
    return tools


def get_ordered_blocks(text):
    """Get all echo-relevant tag blocks sorted by start position."""
    blocks = []
    for tag in ("select", "think", "search", "python", "result", "answer"):
        for start, end, content in find_tag_blocks(text, tag):
            blocks.append((tag, start, end, content))
    blocks.sort(key=lambda b: b[1])
    return blocks


# ---------------------------------------------------------------------------
# Per-check helpers. Each returns (ok, reason) and is attributed to one
# "primary tag" category; the profile routes it to HL or LL.
# ---------------------------------------------------------------------------

def _check_all_closed(blocks):
    for tag, start, end, content in blocks:
        if end == -1:
            return False, f"<{tag}> is not closed"
    return True, None


def _check_first_select(blocks):
    """Primary tag: first_select. First block is planning <select> with a
    non-empty, VALID_ECHO_TOOLS-subset tool list."""
    if blocks[0][0] != "select":
        return False, f"must start with <select>, found <{blocks[0][0]}>"
    allowed = set(extract_tools_from_select(blocks[0][3]))
    if not allowed:
        return False, "first <select> declares no tools"
    invalid = allowed - VALID_ECHO_TOOLS
    if invalid:
        return False, f"unknown tools in planning <select>: {invalid}"
    return True, None


def _check_think_followup(blocks):
    """Primary tag: think. Every <think> must be immediately followed by <select>."""
    for i, (tag, start, end, content) in enumerate(blocks):
        if tag != "think":
            continue
        if i + 1 >= len(blocks):
            return False, "<think> at end with no following <select>"
        if blocks[i + 1][0] != "select":
            return False, f"<think> must be followed by <select>, found <{blocks[i + 1][0]}>"
    return True, None


def _check_answer_boxed(blocks):
    """Primary tag: answer. Exactly one <answer> block containing \\boxed{}."""
    answers = [b for b in blocks if b[0] == "answer"]
    if len(answers) != 1:
        return False, f"expected 1 <answer>, found {len(answers)}"
    if '\\boxed{' not in answers[0][3] or '}' not in answers[0][3]:
        return False, "answer missing \\boxed{}"
    return True, None


def _check_step_select(blocks):
    """Primary tag: select (non-initial). Step-select tool is a subset of the
    planning set and the block that follows matches the selected tool
    (search/python -> tool+result, no-tool -> think|answer)."""
    allowed_tools = set(extract_tools_from_select(blocks[0][3]))
    for i, (tag, start, end, content) in enumerate(blocks):
        if tag != "select" or i == 0:
            continue
        selected = extract_tools_from_select(content)
        if not selected:
            return False, "step <select> declares no tool"
        selected_set = set(selected)
        if not selected_set.issubset(allowed_tools):
            return False, f"tool(s) {selected_set - allowed_tools} not in allowed set {allowed_tools}"
        tool = selected[0]
        if i + 1 >= len(blocks):
            if tool == "no-tool":
                continue
            return False, f"<select> chose '{tool}' but nothing follows"
        next_tag = blocks[i + 1][0]
        if tool == "search":
            if next_tag != "search":
                return False, f"<select> chose 'search' but next is <{next_tag}>"
            if i + 2 >= len(blocks) or blocks[i + 2][0] != "result":
                return False, "<search> not followed by <result>"
        elif tool == "python":
            if next_tag != "python":
                return False, f"<select> chose 'python' but next is <{next_tag}>"
            if i + 2 >= len(blocks) or blocks[i + 2][0] != "result":
                return False, "<python> not followed by <result>"
        elif tool == "no-tool":
            if next_tag not in ("think", "answer"):
                return False, f"<select> chose 'no-tool' but next is <{next_tag}>"
    return True, None


def _check_tool_ordering(blocks):
    """Primary tag: search / python. <search>/<python> preceded by a <select>
    that picked them, and every <result> preceded by <search> or <python>."""
    for i, (tag, start, end, content) in enumerate(blocks):
        if tag in ("search", "python"):
            if i == 0 or blocks[i - 1][0] != "select":
                return False, f"<{tag}> not preceded by <select>"
            prev_tools = extract_tools_from_select(blocks[i - 1][3])
            if tag not in prev_tools:
                return False, f"<{tag}> preceded by <select> that didn't choose '{tag}'"
        elif tag == "result":
            if i == 0 or blocks[i - 1][0] not in ("search", "python"):
                return False, "<result> not preceded by <search> or <python>"
    return True, None


# ---------------------------------------------------------------------------
# Profile-aware validators. Routing table (HL = owned by high_level phase):
#   first_select                  -> HL for c1, c2, c3, c5; LL for c4
#   think, answer                 -> HL for c1, c2, c3, c4, c5
#   select (non-initial)          -> HL for c2, c5;          LL for c1, c3, c4
#   search/python (tool payloads) -> HL for c3, c5;          LL for c1, c2, c4
#   c5 is the all-HL profile used by maxentropy_rl: HL runs every check; LL
#   has no checks (single-phase configuration).
# ---------------------------------------------------------------------------

def validate_high_level(text, profile="c1"):
    blocks = get_ordered_blocks(text)
    if not blocks:
        return False, "no tags found"
    ok, reason = _check_all_closed(blocks)
    if not ok:
        return False, reason
    # HL always owns think/answer; first_select is HL for c1/c2/c3/c5 but LL for
    # c4 (c4 moves the entire tool-planning stack, including the initial plan,
    # to LL).
    hl_checks = [_check_think_followup, _check_answer_boxed]
    if profile in ("c1", "c2", "c3", "c5"):
        hl_checks.insert(0, _check_first_select)
    if profile in ("c2", "c5"):
        hl_checks.append(_check_step_select)
    if profile in ("c3", "c5"):
        hl_checks.append(_check_tool_ordering)
    for check in hl_checks:
        ok, reason = check(blocks)
        if not ok:
            return False, reason
    return True, "high-level format is correct"


def validate_low_level(text, profile="c1"):
    blocks = get_ordered_blocks(text)
    if not blocks:
        return False, "no tags found"
    ok, reason = _check_all_closed(blocks)
    if not ok:
        return False, reason
    # c5: HL owns every format check; LL is vacuously valid so logging
    # metrics (low_level_valid) stay sane even though the LL phase never
    # consumes this verdict.
    if profile == "c5":
        return True, "low-level format is correct"
    # c4 owns first_select fully at LL; for c1/c2/c3 HL already validates it,
    # but LL still needs a minimal guard because _check_step_select and
    # _check_tool_ordering both derive `allowed_tools` from blocks[0].
    if profile == "c4":
        ok, reason = _check_first_select(blocks)
        if not ok:
            return False, reason
    else:
        if not blocks or blocks[0][0] != "select":
            return False, "missing planning <select> needed to derive allowed tools"
        if not set(extract_tools_from_select(blocks[0][3])):
            return False, "planning <select> declares no tools"
    if profile in ("c1", "c3", "c4"):
        ok, reason = _check_step_select(blocks)
        if not ok:
            return False, reason
    if profile in ("c1", "c2", "c4"):
        ok, reason = _check_tool_ordering(blocks)
        if not ok:
            return False, reason
    return True, "low-level format is correct"


# ---------------------------------------------------------------------------
# Combined validation
# ---------------------------------------------------------------------------

def validate_format_echo(text, profile="c1"):
    """Run both high-level and low-level validation, return
    (is_valid, reason, high_level_valid, low_level_valid)."""
    high_valid, high_reason = validate_high_level(text, profile)
    low_valid, low_reason = validate_low_level(text, profile)

    if high_valid and low_valid:
        return True, "format is correct", True, True

    reasons = []
    if not high_valid:
        reasons.append(f"high-level: {high_reason}")
    if not low_valid:
        reasons.append(f"low-level: {low_reason}")
    return False, "; ".join(reasons), high_valid, low_valid


# ---------------------------------------------------------------------------
# Answer extraction and scoring helpers (unchanged from deep_research.py)
# ---------------------------------------------------------------------------

def extract_answer(text):
    """Extract content from <answer>...</answer>."""
    text = text.strip()
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    return match.group(1)


def remove_boxed(s):
    """Remove the LaTeX \\boxed{} wrapper; return None when malformed."""
    text = s.strip()
    if text.startswith("\\boxed "):
        value = text[len("\\boxed ") :].strip()
        return value if value else None

    if not text.startswith("\\boxed{") or not text.endswith("}"):
        return None

    return text[len("\\boxed{") : -1].strip() or None


def last_boxed_only_string(string):
    """Extract the last \\boxed{} content from the string."""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def normalize_answer(s):
    """Normalize by removing articles, whitespace, punctuation and lowering."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_f1_score(prediction, ground_truths):
    """Token-level F1 between prediction and ground truth(s)."""
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    final_metric = {"f1": 0, "precision": 0, "recall": 0}

    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue
        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        final_metric["precision"] = max(precision, final_metric["precision"])
        final_metric["recall"] = max(recall, final_metric["recall"])
        final_metric["f1"] = max(f1, final_metric["f1"])

    return final_metric['f1']


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    result = {
        "score": 0,
        "reason": "",
        "answer": "",
        "f1_score": 0,
        "high_level_valid": False,
        "low_level_valid": False,
        # True when the LL phase sees a format-valid rollout that invoked no
        # tool. Consumed by entropy reward overrides on the LL phase.
        "no_tool_calls": False,
    }

    response = solution_str
    # Validator profile is derived from mask_categories at trainer init and
    # forwarded via extra_info; defaults to c1 for standalone scoring calls.
    profile = extra_info.get("validator_profile", "c1") if extra_info else "c1"
    # Both validators always run so high_level_valid / low_level_valid are available
    # for logging regardless of which phase is gating the -1 verdict.
    hl_valid, hl_reason = validate_high_level(response, profile)
    ll_valid, ll_reason = validate_low_level(response, profile)
    result["high_level_valid"] = hl_valid
    result["low_level_valid"] = ll_valid

    # Phase-aware gating: high_level uses only validate_high_level, low_level uses only
    # validate_low_level, and missing/other falls back to the combined verdict.
    phase = extra_info.get("phase") if extra_info else None
    if phase == "high_level":
        phase_valid, phase_reason = hl_valid, hl_reason
    elif phase == "low_level":
        phase_valid, phase_reason = ll_valid, ll_reason
    else:
        phase_valid = hl_valid and ll_valid
        combined = [r for ok, r in ((hl_valid, f"high-level: {hl_reason}"), (ll_valid, f"low-level: {ll_reason}")) if not ok]
        phase_reason = "; ".join(combined) if combined else "format is correct"

    if not phase_valid:
        print(f"--------bad format ({phase or 'combined'}): {phase_reason}--------\nsolution_str: {solution_str[:200]}, ground_truth: {ground_truth}")
        result["score"] = -1
        result["reason"] = f"bad format: {phase_reason}"
        return result

    # LL phase short-circuits before answer scoring; entropy overrides use the
    # format verdict and no_tool_calls flag from above.
    if phase == "low_level":
        # A format-valid rollout that never invoked <search>/<python> is a soft
        # fail: non-initial <select> tokens would otherwise soak up LL entropy
        # reward and reinforce the no-tool degenerate strategy. Flag the sample
        # so the format gate zeros its dense reward; score stays >= 0 so the
        # hard penalty path is not triggered.
        has_tool_call = any(
            end != -1
            for tag in ("search", "python")
            for _, end, _ in find_tag_blocks(response, tag)
        )
        if not has_tool_call:
            result["no_tool_calls"] = True
            result["reason"] = "low-level: no tool call invoked"
        else:
            result["reason"] = "low-level format is correct"
        return result

    # Strip EOS token if present
    if extra_info and "tokenizer" in extra_info and extra_info["tokenizer"].eos_token and response.endswith(extra_info["tokenizer"].eos_token):
        response = response[:-len(extra_info["tokenizer"].eos_token)]

    answer_part = extract_answer(response)
    if answer_part is None:
        result["score"] = -1
        result["reason"] = "cannot extract answer"
        return result

    boxed = last_boxed_only_string(answer_part)
    if boxed is None:
        result["score"] = -1
        result["reason"] = "no \\boxed{} in answer"
        return result

    answer = remove_boxed(boxed)
    if answer is None:
        result["score"] = -1
        result["reason"] = "malformed \\boxed{} in answer"
        return result
    result["answer"] = answer

    f1_score = get_f1_score(answer, ground_truth)
    result["f1_score"] = f1_score
    print(f"f1_score: {f1_score}, answer: {answer}, ground_truth: {ground_truth}")

    if f1_score > 0 and "</search>" in response and "</python>" in response:
        result["score"] = f1_score + 0.1
        result["reason"] = f"correct answer and calling search and python at the same time, get score: {f1_score + 0.1}"
    elif f1_score > 0:
        result["score"] = f1_score
        result["reason"] = f"correct answer, get f1 score: {f1_score}"
    else:
        result["score"] = 0
        result["reason"] = f"wrong answer but good format: {answer}"

    return result


# ---------------------------------------------------------------------------
# Manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Valid echo-format response with search tool
    test_good = (
        '<select> Need search for this question. <tool> "no-tool" </tool> <tool> "search" </tool> </select> '
        '<think> I need to find info about the Mazda R360 transmission. </think> '
        '<select> Using search to find details. <tool> "search" </tool> </select> '
        '<search>Mazda R360 transmission type japan</search>'
        '<result>The Mazda R360 had a KRBB manual transmission.</result> '
        '<think> The transmission is called KRBB. </think> '
        '<select> Sufficient information. <tool> "no-tool" </tool> </select> '
        '<answer>\\boxed{KRBB}</answer>'
    )
    print("=== Good format ===")
    print(compute_score("test", test_good, "KRBB"))

    # Bad: tool consistency violation (python not in allowed set)
    test_bad_tool = (
        '<select> Need search. <tool> "no-tool" </tool> <tool> "search" </tool> </select> '
        '<think> Let me compute. </think> '
        '<select> Using python. <tool> "python" </tool> </select> '
        '<python>print(42)</python>'
        '<result>42</result> '
        '<think> Done. </think> '
        '<select> Done. <tool> "no-tool" </tool> </select> '
        '<answer>\\boxed{42}</answer>'
    )
    print("\n=== Bad tool consistency ===")
    print(compute_score("test", test_bad_tool, "42"))

    # Bad: think not followed by select
    test_bad_structure = (
        '<select> Tools. <tool> "no-tool" </tool> </select> '
        '<think> Reasoning. </think> '
        '<answer>\\boxed{42}</answer>'
    )
    print("\n=== Bad structure (think not followed by select) ===")
    print(compute_score("test", test_bad_structure, "42"))

    # Valid: consecutive tool calls (result -> select without think)
    test_consecutive = (
        '<select> Need search. <tool> "no-tool" </tool> <tool> "search" </tool> </select> '
        '<think> Need two searches. </think> '
        '<select> First search. <tool> "search" </tool> </select> '
        '<search>query 1</search><result>result 1</result>'
        '<select> Second search. <tool> "search" </tool> </select> '
        '<search>query 2</search><result>result 2</result>'
        '<think> Got all info. </think> '
        '<select> Done. <tool> "no-tool" </tool> </select> '
        '<answer>\\boxed{answer}</answer>'
    )
    print("\n=== Consecutive tool calls (no think between) ===")
    print(compute_score("test", test_consecutive, "answer"))
