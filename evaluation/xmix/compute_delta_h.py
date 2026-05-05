import argparse
import json
import os
from statistics import mean, pstdev

from tqdm import tqdm


def _load_rewards(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", required=True)
    args = parser.parse_args()

    metrics_dir = args.metrics_dir
    high_a_low_b = _load_rewards(os.path.join(metrics_dir, "HIGH_A_LOW_B_rewards.json"))
    high_b_low_a = _load_rewards(os.path.join(metrics_dir, "HIGH_B_LOW_A_rewards.json"))
    high_a_low_a = _load_rewards(os.path.join(metrics_dir, "HIGH_A_LOW_A_rewards.json"))
    high_b_low_b = _load_rewards(os.path.join(metrics_dir, "HIGH_B_LOW_B_rewards.json"))

    n = len(high_a_low_b)
    if not (len(high_b_low_a) == n and len(high_a_low_a) == n and len(high_b_low_b) == n):
        raise ValueError("All four reward files must have the same number of examples.")

    per_example = []
    deltas = []
    for i in tqdm(range(n), desc="computing delta_h"):
        s_hb_lb = float(high_b_low_b[i]["reward"])
        s_ha_lb = float(high_a_low_b[i]["reward"])
        s_hb_la = float(high_b_low_a[i]["reward"])
        s_ha_la = float(high_a_low_a[i]["reward"])
        delta_h = s_hb_lb - s_ha_lb - s_hb_la + s_ha_la
        deltas.append(delta_h)
        per_example.append(
            {
                "index": i,
                "question": high_a_low_a[i].get("question", ""),
                "delta_h": delta_h,
                "score_high_b_low_b": s_hb_lb,
                "score_high_a_low_b": s_ha_lb,
                "score_high_b_low_a": s_hb_la,
                "score_high_a_low_a": s_ha_la,
            }
        )

    out = {
        "num_examples": n,
        "delta_h_avg": mean(deltas) if deltas else 0.0,
        "delta_h_std": pstdev(deltas) if len(deltas) > 1 else 0.0,
        "per_example": per_example,
    }

    out_path = os.path.join(metrics_dir, "delta_h.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote Delta_H metrics: {out_path}")
    print(json.dumps({k: out[k] for k in ("num_examples", "delta_h_avg", "delta_h_std")}, indent=2))


if __name__ == "__main__":
    main()
