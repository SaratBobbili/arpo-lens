import re
import string
from typing import Union, List, Dict, Any, Optional, Tuple
from collections import Counter

VALID_ECHO_TOOLS = {"search", "python", "no-tool"}


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
# High-level policy validation: <select> structure and tool consistency
# ---------------------------------------------------------------------------

def validate_high_level(text):
    """Checks:
    1. Response starts with <select> declaring allowed tools from VALID_ECHO_TOOLS
    2. Every <think> is immediately followed by <select>
    3. ALL non-initial <select> blocks only use tools from the allowed set
    """
    blocks = get_ordered_blocks(text)
    if not blocks:
        return False, "no tags found"

    for tag, start, end, content in blocks:
        if end == -1:
            return False, f"<{tag}> is not closed"

    # First block must be the planning <select>
    if blocks[0][0] != "select":
        return False, f"must start with <select>, found <{blocks[0][0]}>"

    allowed_tools = set(extract_tools_from_select(blocks[0][3]))
    if not allowed_tools:
        return False, "first <select> declares no tools"
    invalid = allowed_tools - VALID_ECHO_TOOLS
    if invalid:
        return False, f"unknown tools in planning <select>: {invalid}"

    # Every non-initial <select> must only reference tools from the allowed set
    for i, (tag, start, end, content) in enumerate(blocks):
        if tag != "select" or i == 0:
            continue
        selected = set(extract_tools_from_select(content))
        if not selected:
            return False, "step <select> declares no tool"
        if not selected.issubset(allowed_tools):
            return False, f"tool(s) {selected - allowed_tools} not in allowed set {allowed_tools}"

    # Every <think> must be immediately followed by <select>
    for i, (tag, start, end, content) in enumerate(blocks):
        if tag != "think":
            continue
        if i + 1 >= len(blocks):
            return False, "<think> at end with no following <select>"
        if blocks[i + 1][0] != "select":
            return False, f"<think> must be followed by <select>, found <{blocks[i + 1][0]}>"

    return True, "high-level format is correct"


# ---------------------------------------------------------------------------
# Low-level policy validation: tool call formatting
# ---------------------------------------------------------------------------

def validate_low_level(text):
    """Checks:
    1. After step-<select> choosing "search": <search>...</search> <result>...</result>
    2. After step-<select> choosing "python": <python>...</python> <result>...</result>
    3. After step-<select> choosing "no-tool": <think> or <answer>
    4. Every <search>/<python> is preceded by a matching <select>
    5. Every <result> is preceded by <search> or <python>
    6. Exactly one <answer> containing \\boxed{}
    """
    blocks = get_ordered_blocks(text)

    # Forward: each step-select is followed by the correct block
    for i, (tag, start, end, content) in enumerate(blocks):
        if tag != "select" or i == 0:
            continue

        selected = extract_tools_from_select(content)
        if not selected:
            continue
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

    # Reverse: tool blocks must be preceded by matching select
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

    # Answer block
    answer_blocks = [b for b in blocks if b[0] == "answer"]
    if len(answer_blocks) != 1:
        return False, f"expected 1 <answer>, found {len(answer_blocks)}"
    if '\\boxed{' not in answer_blocks[0][3] or '}' not in answer_blocks[0][3]:
        return False, "answer missing \\boxed{}"

    return True, "low-level format is correct"


# ---------------------------------------------------------------------------
# Combined validation
# ---------------------------------------------------------------------------

def validate_format_echo(text):
    """Run both high-level and low-level validation, return
    (is_valid, reason, high_level_valid, low_level_valid)."""
    high_valid, high_reason = validate_high_level(text)
    low_valid, low_reason = validate_low_level(text)

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
    """Remove the LaTeX \\boxed{} wrapper."""
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"
    assert s[:len(left)] == left
    assert s[-1] == "}"
    return s[len(left):-1]


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
    }

    response = solution_str
    valid, reason, hl_valid, ll_valid = validate_format_echo(response)
    result["high_level_valid"] = hl_valid
    result["low_level_valid"] = ll_valid

    if not valid:
        print(f"--------bad format: {reason}--------\nsolution_str: {solution_str[:200]}, ground_truth: {ground_truth}")
        result["score"] = -1
        result["reason"] = f"bad format: {reason}"
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
    result["answer"] = answer

    f1_score = get_f1_score(answer, ground_truth)
    result["f1_score"] = f1_score
    print(f"f1_score: {f1_score}, answer: {answer}, ground_truth: {ground_truth}")

    if f1_score > 0 and "</search>" in response and "</python>" in response:
        result["score"] = f1_score + 0.1
        result["reason"] = f"correct with multi-tool, score: {f1_score + 0.1}"
    elif f1_score > 0:
        result["score"] = f1_score
        result["reason"] = f"correct, f1: {f1_score}"
    else:
        result["score"] = 0
        result["reason"] = f"wrong answer, good format: {answer}"

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
