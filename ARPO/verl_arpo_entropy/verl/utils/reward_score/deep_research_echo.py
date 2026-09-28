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


# Which phase owns each tag. Mirrors rollout.mask_categories (the loss masks): the
# leader trains <think>/<answer> tokens, the follower <tool>/<search>/<python> tokens.
# <result> is environment-authored, so no phase is ever gated on it. The scorer's format
# gate fires only on violations in the tags of the phase being scored: the leader is not
# penalised for an unclosed </tool>, the follower not for a missing \boxed{}. Passing
# extra_info["mask_categories"] (tag -> "high" | "low" | "both" | "none") overrides the
# defaults so the gate tracks whatever the run's masks say. Scoring is otherwise
# identical for both phases and for every algorithm family (alt_grpo, hypergradient,
# aho): F1 of the boxed answer, or -1 on a phase-owned schema failure.
_DEFAULT_TAG_OWNERS = {"think": "high", "answer": "high", "tool": "low", "search": "low", "python": "low"}
_ENV_TAGS = ("result",)


def tag_owner(tag, mask_categories=None):
    """Phase that owns ``tag``: "high", "low", "both" or "none". ``None`` = the whole response."""
    if tag is None:
        return "both"
    if tag in _ENV_TAGS:
        return "none"
    owners = dict(_DEFAULT_TAG_OWNERS)
    for key, level in (mask_categories or {}).items():
        if key in owners and level in ("high", "low", "both", "none"):
            owners[key] = level
    return owners.get(tag, "both")


def format_failures(text):
    """Every prompt-5 schema violation in ``text`` as ``(offending_tag, reason)`` pairs.

    Phase-first rules (2026-09-27). Each tag is checked on its own row, and a violation is
    charged to that tag; a phase fails only on rows of the tags it owns:

        think          closed; preceded by the start of the response or a <result>
        answer         closed; preceded by a <tool>; exactly one; contains \\boxed{}
        tool           closed; preceded by a <think> or a <result> (calls may chain)
        search/python  closed; preceded by a <tool>; followed by a <result> (call served)
        result         environment text, never charged

    "X must be followed by Y" is the same fact as "Y must be preceded by X" charged to Y,
    so no forward check is needed except the served-call one. Unclosed blocks stay in the
    sequence for the predecessor checks, so an unclosed <tool> before a good <answer> is one
    follower-owned failure, not a cascade onto the leader. Empty / tag-less text is charged
    to ``None`` (the whole response) and fails both phases.
    """
    if not text or not text.strip():
        return [(None, "empty")]

    blocks = get_ordered_blocks(text)
    if not blocks:
        return [(None, "no_tags")]

    failures = []
    for tag, start, end, content in blocks:
        if end == -1:
            failures.append((tag, f"<{tag}> is not closed"))

    kinds = [b[0] for b in blocks]
    for i, k in enumerate(kinds):
        prev_k = kinds[i - 1] if i > 0 else None
        next_k = kinds[i + 1] if i + 1 < len(kinds) else None
        if k == "think" and prev_k not in (None, "result"):
            failures.append((k, "think_not_after_result"))
        elif k == "tool" and prev_k not in ("think", "result"):
            failures.append((k, "tool_not_after_think_or_result"))
        elif k in ("search", "python"):
            if prev_k != "tool":
                failures.append((k, f"{k}_not_after_tool"))
            if next_k != "result":
                failures.append((k, f"{k}_not_before_result"))
        elif k == "answer" and prev_k != "tool":
            failures.append((k, "answer_not_after_tool"))

    n_answers = kinds.count("answer")
    if n_answers != 1:
        failures.append(("answer", f"answer_count={n_answers}"))
    else:
        answer = next(b for b in blocks if b[0] == "answer")
        # content is None for an unclosed <answer>, already charged above.
        if answer[3] is not None and ("\\boxed{" not in answer[3] or "}" not in answer[3]):
            failures.append(("answer", "answer missing \\boxed{}"))

    return failures


def validate_format(text):
    """Whole-schema check: (valid, first failure reason). Kept for callers that want one bit."""
    failures = format_failures(text)
    if failures:
        return False, failures[0][1]
    return True, "format is correct"


def phase_format_check(text, level, mask_categories=None):
    """Format gate for one phase.

    Returns ``(schema_valid, phase_valid, reason, issues)``: ``schema_valid`` is the
    whole-response bit, ``phase_valid`` is False only when a violation is charged to a
    tag that ``level`` ("high" or "low") owns, ``reason`` is that violation (or the
    generic "format is correct"), and ``issues`` lists every violation as
    ``"tag/owner: reason"`` for the rollout dump.
    """
    failures = format_failures(text)
    owned = [
        (tag, reason)
        for tag, reason in failures
        if tag_owner(tag, mask_categories) in (level, "both")
    ]
    issues = "; ".join(f"{tag or '*'}/{tag_owner(tag, mask_categories)}: {reason}" for tag, reason in failures)
    reason = owned[0][1] if owned else "format is correct"
    return not failures, not owned, reason, issues


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
    """Task return for one phase: F1 of the boxed answer (+ multi-tool bonus), or -1 on a
    schema failure in the tags that phase owns (see ``format_failures`` / ``tag_owner``).

    ``extra_info["phase"]`` selects the phase: "low_level" scores under the follower's rows
    (tool/search/python), anything else under the leader's (think/answer). Violations in
    the other phase's tags never gate this score; they are recorded in ``format_issues``.
    A follower-scored trajectory with no scoreable answer is a leader failure, so it gets
    F1 = 0 there rather than -1.
    """
    result = {
        "score": 0,
        "reason": "",
        "answer": "",
        "f1_score": 0,
        "format_valid": False,
        "phase_format_valid": False,
        "format_issues": "",
        "no_tool_calls": False,
    }

    response = solution_str
    info = extra_info or {}
    level = "low" if str(info.get("phase", "high_level")) == "low_level" else "high"
    schema_valid, phase_valid, reason, issues = phase_format_check(response, level, info.get("mask_categories"))
    result["format_valid"] = schema_valid
    result["phase_format_valid"] = phase_valid
    result["format_issues"] = issues
    if not phase_valid:
        print(f"--------bad format ({level}): {reason}--------\nsolution_str: {solution_str[:200]}, ground_truth: {ground_truth}")
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

    # An unextractable answer is a leader-owned failure: -1 for the leader, 0 (no answer,
    # so no F1) for the follower, whose own rows already passed above.
    no_answer_score = -1 if level == "high" else 0
    answer_part = extract_answer(response)
    if answer_part is None:
        result["score"] = no_answer_score
        result["reason"] = "cannot extract answer"
        return result

    boxed = last_boxed_only_string(answer_part)
    if boxed is None:
        result["score"] = no_answer_score
        result["reason"] = "no \\boxed{} in answer"
        return result

    answer = remove_boxed(boxed)
    if answer is None:
        result["score"] = no_answer_score
        result["reason"] = "malformed \\boxed{} in answer"
        return result
    result["answer"] = answer

    f1_score = get_f1_score(answer, ground_truth)
    result["f1_score"] = f1_score
    print(f"f1_score: {f1_score}, answer: {answer}, ground_truth: {ground_truth}")

    # No tool-usage bonus: ARPO's +0.1 for "search and python in the same trajectory"
    # is hackable (the policy learns to fire every tool on every question, exhausting
    # the call budget), so the task return is the answer F1 alone.
    if f1_score > 0:
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
    print("\n=== Multi tool (no bonus: score is F1 alone) ===")
    r = compute_score("test", test_multi, "12")
    print(r)
    assert r["score"] == 1.0

    test_legacy_select = (
        '<select> Need search. <tool> "search" </tool> </select> '
        '<think> Reasoning. </think> '
        '<answer>\\boxed{42}</answer>'
    )
    print("\n=== Legacy select schema rejected ===")
    r = compute_score("test", test_legacy_select, "42")
    print(r)
    assert r["score"] == -1

    LOW = {"phase": "low_level"}
    test_unclosed_tool = test_good.replace("no further tool needed. </tool>", "no further tool needed.")
    print("\n=== Unclosed </tool>: follower-owned, leader still scores F1 ===")
    r = compute_score("test", test_unclosed_tool, "KRBB")
    print(r)
    assert r["score"] == 1.0 and not r["format_valid"] and r["phase_format_valid"]
    r = compute_score("test", test_unclosed_tool, "KRBB", LOW)
    print(r)
    assert r["score"] == -1 and not r["phase_format_valid"]

    test_no_boxed = test_good.replace("\\boxed{KRBB}", "KRBB")
    print("\n=== Missing \\boxed{}: leader-owned, follower scores 0 (no answer to grade) ===")
    r = compute_score("test", test_no_boxed, "KRBB", LOW)
    print(r)
    assert r["score"] == 0 and not r["format_valid"] and r["phase_format_valid"]
    assert compute_score("test", test_no_boxed, "KRBB")["score"] == -1

    test_chained = (
        '<think> Two lookups. </think><tool> first </tool><search> a </search><result> 1 </result>'
        '<tool> second, chained without a think </tool><search> b </search><result> 2 </result>'
        '<think> Done. </think><tool> enough </tool><answer>\\boxed{KRBB}</answer>'
    )
    print("\n=== result -> tool chaining is legal for both phases ===")
    assert format_failures(test_chained) == []
    assert compute_score("test", test_chained, "KRBB")["score"] == 1.0
    assert compute_score("test", test_chained, "KRBB", LOW)["score"] == 1.0

    print("\nAll smoke checks passed.")
