import re
import string
from collections import Counter

ALLOWED_TAGS = ("think", "tool", "search", "python", "result", "answer")


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


def get_ordered_blocks(text):
    """Get all prompt-5 tag blocks sorted by start position."""
    blocks = []
    for tag in ALLOWED_TAGS:
        for start, end, content in find_tag_blocks(text, tag):
            blocks.append((tag, start, end, content))
    blocks.sort(key=lambda b: b[1])
    return blocks


def validate_format(text):
    """Prompt-5 invariants (same as scripts/sft_refactor/trajectory.verify_text):
    think→tool→(search|python|answer); search/python→result; answer after tool + \\boxed{}.
    """
    if not text or not text.strip():
        return False, "empty"

    blocks = get_ordered_blocks(text)
    if not blocks:
        return False, "no_tags"

    for tag, start, end, content in blocks:
        if end == -1:
            return False, f"<{tag}> is not closed"

    kinds = [b[0] for b in blocks]
    if kinds[0] != "think":
        return False, "first_tag_not_think"

    if kinds.count("answer") != 1:
        return False, f"answer_count={kinds.count('answer')}"

    answers = [b for b in blocks if b[0] == "answer"]
    if "\\boxed{" not in answers[0][3] or "}" not in answers[0][3]:
        return False, "answer missing \\boxed{}"

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

    return True, "format is correct"


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
        return None
    return string[idx:right_brace_idx + 1]


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

    return final_metric["f1"]


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Shared phase-agnostic score: −1 on format/answer/boxed fail; else F1 (+ multi-tool bonus)."""
    result = {
        "score": 0,
        "reason": "",
        "answer": "",
        "f1_score": 0,
        "format_valid": False,
        "no_tool_calls": False,
    }

    response = solution_str
    valid, reason = validate_format(response)
    result["format_valid"] = valid
    if not valid:
        print(f"--------bad format: {reason}--------\nsolution_str: {solution_str[:200]}, ground_truth: {ground_truth}")
        result["score"] = -1
        result["reason"] = f"bad format: {reason}"
        return result

    has_tool_call = any(
        end != -1
        for tag in ("search", "python")
        for _, end, _ in find_tag_blocks(response, tag)
    )
    if not has_tool_call:
        result["no_tool_calls"] = True

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

    if result["no_tool_calls"]:
        result["reason"] = f"{result['reason']} (no tool call invoked)"

    return result


if __name__ == "__main__":
    test_good = (
        '<think> Need search for Mazda R360 transmission. </think> '
        '<tool> Search wikipedia for the transmission type used in Japan. </tool> '
        '<search>Mazda R360 transmission type japan</search>'
        '<result>The Mazda R360 had a KRBB manual transmission.</result> '
        '<think> The transmission is called KRBB. </think> '
        '<tool> Accumulated result answers the question; no further tool needed. </tool> '
        '<answer>\\boxed{KRBB}</answer>'
    )
    print("=== Good format ===")
    print(compute_score("test", test_good, "KRBB"))
    assert compute_score("test", test_good, "KRBB")["score"] > 0

    test_think_answer = (
        '<think> Direct answer from prior knowledge. </think> '
        '<tool> No tool needed; the answer is known. </tool> '
        '<answer>\\boxed{42}</answer>'
    )
    print("\n=== Think→tool→answer (no tool call) ===")
    r = compute_score("test", test_think_answer, "42")
    print(r)
    assert r["score"] > 0 and r["no_tool_calls"]

    test_bad_structure = (
        '<think> Reasoning. </think> '
        '<answer>\\boxed{42}</answer>'
    )
    print("\n=== Bad structure (think not followed by tool) ===")
    r = compute_score("test", test_bad_structure, "42")
    print(r)
    assert r["score"] == -1

    test_multi = (
        '<think> Need both search and python. </think> '
        '<tool> Search for the formula. </tool> '
        '<search>query 1</search><result>result 1</result>'
        '<think> Now compute with python. </think> '
        '<tool> Run python to get the numeric answer. </tool> '
        '<python>print(12)</python><result>12</result>'
        '<think> Got all info. </think> '
        '<tool> Enough evidence to answer. </tool> '
        '<answer>\\boxed{12}</answer>'
    )
    print("\n=== Multi tool (search+python bonus) ===")
    r = compute_score("test", test_multi, "12")
    print(r)
    assert r["score"] > 1.0

    test_legacy_select = (
        '<select> Need search. <tool> "search" </tool> </select> '
        '<think> Reasoning. </think> '
        '<answer>\\boxed{42}</answer>'
    )
    print("\n=== Legacy select schema rejected ===")
    r = compute_score("test", test_legacy_select, "42")
    print(r)
    assert r["score"] == -1

    print("\nAll smoke checks passed.")
