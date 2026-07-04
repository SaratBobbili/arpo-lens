"""Self-tests for the corrected math-eval signal. Run from evaluation/:
    python -m src.test_equivalence
"""

from .math_equivalence import is_equiv, math_answers_equal, numeric_equal, symbolic_equal
from .llm_evaluator_sds import parse_judge_verdict


def run():
    # acc is int(is_equiv(prediction, reference)); these were the false positives.
    assert is_equiv("-216", "216") is False
    assert is_equiv("60\\pi", "\\pi") is False

    # former false negatives now pass strict acc equality
    assert is_equiv("\\dfrac{1}{2}", "\\frac{1}{2}") is True
    assert is_equiv("\\left( 3, \\frac{\\pi}{2} \\right)", "\\left(3,\\frac{\\pi}{2}\\right)") is True
    assert is_equiv("0.5", "\\frac{1}{2}") is True
    assert is_equiv("1,000", "1000") is True

    # numeric equivalence
    assert numeric_equal("0.5", "1/2") is True
    assert numeric_equal("0.5", "\\frac{1}{2}") is True

    # symbolic equivalence (commuted expression) -> math_equal but not acc
    assert symbolic_equal("x + 11", "11 + x") is True
    assert is_equiv("x + 11", "11 + x") is False
    assert math_answers_equal("x + 11", "11 + x") is True

    # malformed / degenerate inputs must never raise (root-cause regression)
    for bad in ["", "\\frac", "3\\frac", "\\sqrt", "5\\sqrt", "{", "\\frac{1}"]:
        is_equiv(bad, "3")
        numeric_equal(bad, "3")
        math_answers_equal(bad, "3")
        math_answers_equal("3", bad)
    assert math_answers_equal("3", "\\frac") is False
    assert numeric_equal(None, "3") is False
    assert math_answers_equal(None, None) is True
    assert math_answers_equal(None, "3") is False

    # deterministic judge: verdict must agree with the text
    assert parse_judge_verdict("Incorrect\n\nThe Predicted Answer does not match.") is False
    assert parse_judge_verdict("Correct") is True
    assert parse_judge_verdict("<judgment>Incorrect</judgment>") is False
    assert parse_judge_verdict("The answer is not correct") is False

    print("All equivalence/judge self-tests passed.")


if __name__ == "__main__":
    run()
