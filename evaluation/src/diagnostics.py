import sys
import os
sys.path.append(os.getcwd())

import re
from typing import Dict, List, Any

# Categories reported side by side. `arithmetic` is the fallback bucket.
CATEGORIES = ["diagram", "geometry", "number_theory", "probability_counting", "rate_word", "arithmetic"]

_GEOMETRY_KW = ("triangle", "circle", "angle", "polygon", "hexagon", "pentagon",
                "square", "rectangle", "parallelogram", "rhombus", "trapezoid",
                "perimeter", "radius", "diameter", "circumference", "vertex",
                "vertices", "diagonal", "quadrilateral", "sphere", "cylinder",
                "cone", "coordinate", "hypotenuse", "tangent", "chord", "arc")
_NUMBER_THEORY_KW = ("divisor", "divisible", "prime", "modulo", "remainder",
                     "factor", "gcd", "lcm", "congruent", "integer solution",
                     "base-", "digits", "multiple of")
_PROB_KW = ("probability", "combination", "permutation", "how many ways",
            "number of ways", "choose", "expected value", "dice", "coin",
            "randomly", "at random")
_RATE_KW = ("per hour", "per minute", "per second", "miles per", "km/h",
            "speed", "rate of", "how long", "how many hours", "how many minutes",
            "average speed", "travel", "fills", "work together", "faster")

_PY_ERROR_TOKENS = ("Traceback (most recent call last)", "SyntaxError",
                    "NameError", "TypeError", "IndexError", "ValueError",
                    "KeyError", "ZeroDivisionError", "AttributeError",
                    "Execution Timeout", "Tool execute failed")

_PY_RESULT_RE = re.compile(r"<python>.*?</python>\s*<result>(.*?)</result>", re.DOTALL)


def classify_question(text: str) -> str:
    """Heuristically bucket a math question by the reasoning it requires."""
    if not text:
        return "arithmetic"
    low = text.lower()
    if "[asy]" in low or "\\begin{asy}" in low or "asymptote" in low:
        return "diagram"
    if any(k in low for k in _GEOMETRY_KW):
        return "geometry"
    if any(k in low for k in _NUMBER_THEORY_KW):
        return "number_theory"
    if any(k in low for k in _PROB_KW):
        return "probability_counting"
    if any(k in low for k in _RATE_KW):
        return "rate_word"
    return "arithmetic"


def detect_python_error(output: str) -> bool:
    """True if any python-triggered <result> block carries an error signature."""
    if not output:
        return False
    for result in _PY_RESULT_RE.findall(output):
        if any(tok in result for tok in _PY_ERROR_TOKENS):
            return True
    return False


def _m(row: Dict[str, Any], key: str, default=0):
    return row.get("metrics", {}).get(key, default)


def bucket_rows(rows: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """Bucket eval rows by the failure modes described in the plan. A category
    'failure' means wrong by the trustworthy math_equal metric."""
    buckets = {
        "acc0_matheq1": [],
        "acc0_llmeq1": [],
        "acc1_matheq0_llmeq0": [],
        "python_exception": [],
        "diagram_fail": [],
        "geometry_fail": [],
        "word_rate_fail": [],
    }
    for i, row in enumerate(rows):
        acc = _m(row, "acc")
        math_equal = _m(row, "math_equal")
        llm_equal = _m(row, "llm_equal")
        output = row.get("output", "") or ""
        category = row.get("metrics", {}).get("category") or classify_question(row.get("input", ""))

        if acc == 0 and math_equal == 1:
            buckets["acc0_matheq1"].append(i)
        if acc == 0 and llm_equal == 1:
            buckets["acc0_llmeq1"].append(i)
        if acc == 1 and math_equal == 0 and llm_equal == 0:
            buckets["acc1_matheq0_llmeq0"].append(i)
        if detect_python_error(output):
            buckets["python_exception"].append(i)
        if category == "diagram" and math_equal == 0:
            buckets["diagram_fail"].append(i)
        if category == "geometry" and math_equal == 0:
            buckets["geometry_fail"].append(i)
        if category == "rate_word" and math_equal == 0:
            buckets["word_rate_fail"].append(i)
    return buckets
