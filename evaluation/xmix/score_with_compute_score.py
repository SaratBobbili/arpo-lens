import argparse
import glob
import json
import os
import sys

from tqdm import tqdm


def _load_compute_score():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    verl_path = os.path.join(repo_root, "ARPO", "verl_arpo_entropy")
    if verl_path not in sys.path:
        sys.path.insert(0, verl_path)
    from verl.utils.reward_score.deep_research_echo import compute_score

    return compute_score


def _score_file(path: str, validator_profile: str, compute_score):
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    scored_rows = []
    reward_rows = []
    f1_scores = []
    format_ok = 0
    hl_ok = 0
    ll_ok = 0
    bad_format = 0

    for row in tqdm(rows, desc=f"scoring {os.path.basename(path)}", leave=False):
        result = compute_score(
            data_source="grpo_mix",
            solution_str=row.get("output", "") or "",
            ground_truth=row.get("answer", ""),
            extra_info={"validator_profile": validator_profile},
        )
        hl_valid = int(result.get("high_level_valid", False))
        ll_valid = int(result.get("low_level_valid", False))
        fmt_valid = int(hl_valid and ll_valid)
        bad_fmt = int(result.get("score", 0) == -1)
        f1 = float(result.get("f1_score", 0.0) or 0.0)

        row_copy = row.copy()
        reward = float(result.get("score", 0.0))
        row_copy["metrics"] = {
            "score": reward,
            "f1": f1,
            "echo_format_valid": fmt_valid,
            "echo_high_level_valid": hl_valid,
            "echo_low_level_valid": ll_valid,
            "bad_format": bad_fmt,
            "reason": result.get("reason", ""),
            "answer": result.get("answer", ""),
        }
        scored_rows.append(row_copy)
        reward_rows.append(
            {
                "question": row.get("input", ""),
                "answer": row.get("answer", ""),
                "prediction": row.get("prediction", ""),
                "reward": reward,
                "f1": f1,
                "bad_format": bad_fmt,
                "reason": result.get("reason", ""),
            }
        )

        f1_scores.append(f1)
        format_ok += fmt_valid
        hl_ok += hl_valid
        ll_ok += ll_valid
        bad_format += bad_fmt

    n = len(rows) if rows else 1
    overall = {
        "f1": sum(f1_scores) / n,
        "echo_format_pass_rate": format_ok / n,
        "echo_high_level_pass_rate": hl_ok / n,
        "echo_low_level_pass_rate": ll_ok / n,
        "bad_format_rate": bad_format / n,
        "num_samples": len(rows),
        "validator_profile": validator_profile,
    }

    base, _ = os.path.splitext(path)
    rewards_path = f"{base}_rewards.json"
    metrics_path = f"{base}_metrics.json"
    overall_path = f"{base}_metrics_overall.json"
    with open(rewards_path, "w", encoding="utf-8") as f:
        json.dump(reward_rows, f, indent=2, ensure_ascii=False)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(scored_rows, f, indent=2, ensure_ascii=False)
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print(f"[done] {path}")
    print(json.dumps(overall, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--validator_profile", default="c1")
    args = parser.parse_args()

    compute_score = _load_compute_score()
    pattern = os.path.join(args.output_dir, "*", "*_output_*.json")
    files = sorted(glob.glob(pattern))
    files = [p for p in files if p.endswith(".json") and "_metrics" not in os.path.basename(p)]
    if not files:
        raise ValueError(f"No output json files found under: {args.output_dir}")

    for path in tqdm(files, desc="output files"):
        _score_file(path, args.validator_profile, compute_score)


if __name__ == "__main__":
    main()
