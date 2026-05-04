"""
Score ping-pong rollouts and aggregate val-core F1 / bad-format rate per combo.

Reuses `verl.utils.reward_score.deep_research_echo.compute_score` (the same
function the trainer's ECHORewardManager calls during validation), with
phase=None so the combined-format gate matches the trainer's val path. The
validator profile c1 is forwarded via extra_info so per-check HL/LL attribution
is the same as during training.

Bad-format rollouts (compute_score score == -1) are excluded from the F1 mean
and reported as a separate `bad_format_rate`. This mirrors the spec:
"Computes the val-core F1 for each combination, dropping bad-format (-1)
rollouts from the F1 mean and reporting the bad-format rate alongside."
"""

import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from verl.utils.reward_score.deep_research_echo import compute_score


def _iter_jsonl(paths):
    for p in paths:
        with open(p, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def score_combo(jsonl_paths, validator_profile: str = "c1") -> dict:
    """Score every JSONL row and return aggregate metrics for one combo.

    Returns:
        dict with keys:
          - per_data_source: {data_source: {f1_mean, bad_format_rate, n_total, n_good}}
          - overall: {f1_mean, bad_format_rate, n_total, n_good}
          - rows: list of per-sample {sample_id, rep, data_source, score, f1, bad_format, terminator, ...}
    """
    rows = []
    for r in tqdm(list(_iter_jsonl(jsonl_paths)), desc="scoring", leave=False):
        result = compute_score(
            data_source=r["data_source"],
            solution_str=r["response_text"],
            ground_truth=r["ground_truth"],
            extra_info={"validator_profile": validator_profile},
        )
        bad_format = (result["score"] == -1)
        rows.append({
            "sample_id": r["sample_id"],
            "rep": r["rep"],
            "data_source": r["data_source"],
            "score": result["score"],
            "f1": result["f1_score"],
            "bad_format": bad_format,
            "high_level_valid": result["high_level_valid"],
            "low_level_valid": result["low_level_valid"],
            "terminator": r["terminator"],
            "turns": r["turns"],
            "tool_calls": r["tool_calls"],
            "answer": result["answer"],
        })

    per_ds = defaultdict(lambda: {"f1_sum": 0.0, "n_good": 0, "n_total": 0, "n_bad": 0})
    for row in rows:
        ds = row["data_source"]
        per_ds[ds]["n_total"] += 1
        if row["bad_format"]:
            per_ds[ds]["n_bad"] += 1
        else:
            per_ds[ds]["f1_sum"] += float(row["f1"])
            per_ds[ds]["n_good"] += 1

    summary_per_ds = {}
    overall = {"f1_sum": 0.0, "n_good": 0, "n_total": 0, "n_bad": 0}
    for ds, agg in per_ds.items():
        summary_per_ds[ds] = {
            "f1_mean": (agg["f1_sum"] / agg["n_good"]) if agg["n_good"] > 0 else 0.0,
            "bad_format_rate": agg["n_bad"] / max(1, agg["n_total"]),
            "n_total": agg["n_total"],
            "n_good": agg["n_good"],
        }
        for k in overall:
            overall[k] += agg[k]
    overall = {
        "f1_mean": (overall["f1_sum"] / overall["n_good"]) if overall["n_good"] > 0 else 0.0,
        "bad_format_rate": overall["n_bad"] / max(1, overall["n_total"]),
        "n_total": overall["n_total"],
        "n_good": overall["n_good"],
    }

    return {
        "per_data_source": summary_per_ds,
        "overall": overall,
        "rows": rows,
    }


def write_summary(summary: dict, out_dir: str, combo_label: str) -> None:
    """Write per-combo scored rows JSONL and summary JSON."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{combo_label}.scored.jsonl", "w") as f:
        for row in summary["rows"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(out / f"{combo_label}.summary.json", "w") as f:
        json.dump({k: v for k, v in summary.items() if k != "rows"}, f, indent=2)
