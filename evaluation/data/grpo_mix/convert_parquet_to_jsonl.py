"""One-time conversion of grpo_mix_test.parquet -> test.jsonl.

The parquet mirrors the RL training validation set (ARPO/rl_datasets/valid.parquet):
180 mixed math+qa rows with schema {data_source, question, ability, reward_model,
extra_info}. evaluation/src/data_loader.py expects the default per-dataset jsonl
schema {"question": ..., "answer": ...}, so we extract reward_model.ground_truth
as the answer and drop the rest. We do NOT split by `ability`: the training
scorer computes a uniform F1 on the boxed answer for both math and qa rows, and
the eval pipeline's LLM-as-judge is task-agnostic, so a single mixed jsonl flows
through inference + evaluation as one dataset.
"""
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
SRC = HERE / "grpo_mix_test.parquet"
DST = HERE / "test.jsonl"


def main():
    df = pd.read_parquet(SRC)
    with DST.open("w", encoding="utf-8") as f:
        for row in tqdm(df.itertuples(index=False), total=len(df), desc="grpo_mix -> jsonl"):
            record = {
                "question": row.question,
                "answer": row.reward_model["ground_truth"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(df)} rows to {DST}")


if __name__ == "__main__":
    main()
