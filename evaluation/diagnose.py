#!/usr/bin/env python
"""Reproducible offline diagnostic for math eval rollouts.

Re-scores an existing *_metrics.json with the corrected metrics (acc,
math_equal, deterministic llm_equal), buckets the failure modes, reports a
per-category breakdown, and optionally puts an ARPO run side by side with echo.
No re-inference: everything is recomputed from the stored output/prediction/
answer/llm_response fields.
"""

import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Any

from src.math_equivalence import is_equiv, math_answers_equal
from src.llm_evaluator_sds import parse_judge_verdict
from src.utils import extract_answer
from src.diagnostics import classify_question, detect_python_error, bucket_rows, CATEGORIES


def _load(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rescore(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Recompute acc/math_equal/llm_equal/python_error/category in place."""
    for row in rows:
        answer = row.get("answer", "")
        prediction = (row.get("prediction") or "").strip()
        output = row.get("output", "") or ""
        if not prediction and output:
            prediction = (extract_answer(output) or "").strip()

        metrics = row.setdefault("metrics", {})
        metrics["acc"] = int(is_equiv(prediction, answer))
        metrics["math_equal"] = int(math_answers_equal(prediction, answer))

        llm_response = metrics.get("llm_response")
        if llm_response and llm_response != "Error":
            metrics["llm_equal"] = int(parse_judge_verdict(llm_response))
        metrics["python_error"] = int(detect_python_error(output))
        metrics["category"] = classify_question(row.get("input", ""))
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows) or 1

    def rate(key):
        return sum(r["metrics"].get(key, 0) for r in rows) / n

    have_llm = any("llm_equal" in r["metrics"] for r in rows)
    have_fmt = any("echo_format_valid" in r["metrics"] for r in rows)

    cat = defaultdict(lambda: {"count": 0, "acc": 0, "math_equal": 0, "llm_equal": 0})
    for r in rows:
        m = r["metrics"]
        s = cat[m.get("category", "arithmetic")]
        s["count"] += 1
        s["acc"] += m.get("acc", 0)
        s["math_equal"] += m.get("math_equal", 0)
        s["llm_equal"] += m.get("llm_equal", 0)

    return {
        "num_samples": len(rows),
        "acc": rate("acc"),
        "math_equal": rate("math_equal"),
        "llm_equal": rate("llm_equal") if have_llm else None,
        "format_pass_rate": rate("echo_format_valid") if have_fmt else None,
        "python_exception_rate": rate("python_error"),
        "category_metrics": {
            c: {
                "count": s["count"],
                "acc": s["acc"] / s["count"],
                "math_equal": s["math_equal"] / s["count"],
                "llm_equal": s["llm_equal"] / s["count"],
            }
            for c, s in cat.items()
        },
    }


def _fmt(v):
    return "  n/a" if v is None else f"{v:.4f}"


def _print_side_by_side(echo: Dict[str, Any], arpo: Dict[str, Any] = None):
    cols = [("echo", echo)] + ([("arpo", arpo)] if arpo else [])
    header = f"{'metric':<24}" + "".join(f"{name:>10}" for name, _ in cols)
    print("\n===== Overall (side by side) =====")
    print(header)
    for key in ["num_samples", "acc", "math_equal", "llm_equal", "format_pass_rate", "python_exception_rate"]:
        cells = []
        for _, s in cols:
            v = s.get(key)
            cells.append(f"{v:>10}" if key == "num_samples" else f"{_fmt(v):>10}")
        print(f"{key:<24}" + "".join(cells))

    print("\n===== Category accuracy (math_equal) =====")
    print(f"{'category':<22}" + "".join(f"{name:>10}" for name, _ in cols) + f"{'n(echo)':>10}")
    for c in CATEGORIES:
        cells = []
        for _, s in cols:
            cm = s["category_metrics"].get(c)
            cells.append(f"{_fmt(cm['math_equal']) if cm else '  n/a':>10}")
        necho = echo["category_metrics"].get(c, {}).get("count", 0)
        print(f"{c:<22}" + "".join(cells) + f"{necho:>10}")


def _print_buckets(rows, buckets, n_examples):
    print("\n===== Failure buckets (echo) =====")
    for name, idxs in buckets.items():
        examples = idxs[:n_examples]
        preview = ", ".join(str(rows[i].get("input", "") or "")[:40].replace("\n", " ") for i in examples)
        print(f"{name:<24} count={len(idxs):<6} e.g. [{preview}]")


def main():
    parser = argparse.ArgumentParser(description="Math eval diagnostic / bucketing")
    parser.add_argument("--echo", required=True, help="Path to echo *_metrics.json")
    parser.add_argument("--arpo", default=None, help="Optional path to ARPO *_metrics.json for side-by-side")
    parser.add_argument("--output", default=None, help="Where to write the JSON report (default: <echo>_diagnostic.json)")
    parser.add_argument("--examples", type=int, default=5, help="Example ids to show per bucket")
    args = parser.parse_args()

    echo_rows = rescore(_load(args.echo))
    echo_summary = summarize(echo_rows)
    buckets = bucket_rows(echo_rows)

    arpo_summary = None
    if args.arpo:
        arpo_rows = rescore(_load(args.arpo))
        arpo_summary = summarize(arpo_rows)

    _print_side_by_side(echo_summary, arpo_summary)
    _print_buckets(echo_rows, buckets, args.examples)

    report = {
        "echo": echo_summary,
        "arpo": arpo_summary,
        "buckets": {name: {"count": len(idxs), "indices": idxs} for name, idxs in buckets.items()},
    }
    out_path = args.output or (os.path.splitext(args.echo)[0] + "_diagnostic.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
