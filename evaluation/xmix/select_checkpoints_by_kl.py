import argparse
import glob
import json
import os
import re


STEP_RE = re.compile(r"step:(\d+)")
KV_RE = re.compile(r"([A-Za-z0-9_./@-]+):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def _parse_wandb_output_log(path: str):
    high_kl_by_step = {}
    low_kl_by_step = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = STEP_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            kvs = {k: float(v) for k, v in KV_RE.findall(line)}
            high_key = "high_level/actor/kl_loss"
            low_key = "low_level/actor/kl_loss"
            if high_key in kvs:
                high_kl_by_step[step] = kvs[high_key]
            if low_key in kvs:
                low_kl_by_step[step] = kvs[low_key]
    return high_kl_by_step, low_kl_by_step


def _snapshot_steps(ckpt_root: str):
    steps = []
    for path in sorted(glob.glob(os.path.join(ckpt_root, "global_step_*", "hf", "config.json"))):
        step_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
        step = int(step_dir.split("_")[-1])
        steps.append(step)
    return steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_root", required=True)
    parser.add_argument("--train_checkpoint_dir", required=True)
    parser.add_argument("--out_json", default="")
    args = parser.parse_args()

    output_logs = sorted(
        glob.glob(os.path.join(args.train_checkpoint_dir, "wandb", "run-*", "files", "output.log"))
    )
    if not output_logs:
        raise ValueError(f"Missing wandb output.log under: {args.train_checkpoint_dir}/wandb/run-*/files/output.log")
    output_log = output_logs[-1]
    if not os.path.isfile(output_log):
        raise ValueError(f"Missing wandb output.log: {output_log}")

    high_kl_by_step, low_kl_by_step = _parse_wandb_output_log(output_log)
    steps = _snapshot_steps(args.ckpt_root)
    if len(steps) < 2:
        raise ValueError(f"Need at least 2 snapshot checkpoints in {args.ckpt_root}, found {len(steps)}")

    high_candidates = [(s, high_kl_by_step[s]) for s in steps if s in high_kl_by_step]
    low_candidates = [(s, low_kl_by_step[s]) for s in steps if s in low_kl_by_step]
    if len(high_candidates) < 2:
        raise ValueError("Need at least 2 high-level checkpoints with kl_loss in wandb log.")
    if len(low_candidates) < 2:
        raise ValueError("Need at least 2 low-level checkpoints with kl_loss in wandb log.")

    low_sorted = sorted(low_candidates, key=lambda x: x[1])
    high_sorted = sorted(high_candidates, key=lambda x: x[1])
    low_a_step, low_a_kl = low_sorted[0]
    low_b_step, low_b_kl = low_sorted[-1]
    high_a_step, high_a_kl = high_sorted[0]
    high_b_step, high_b_kl = high_sorted[-1]
    if low_a_step == low_b_step:
        raise ValueError("Low pair collapsed to the same step.")
    if high_a_step == high_b_step:
        raise ValueError("High pair collapsed to the same step.")

    payload = {
        "LOW_A_STEP": low_a_step,
        "LOW_A_KL_LOSS": low_a_kl,
        "LOW_B_STEP": low_b_step,
        "LOW_B_KL_LOSS": low_b_kl,
        "HIGH_A_STEP": high_a_step,
        "HIGH_A_KL_LOSS": high_a_kl,
        "HIGH_B_STEP": high_b_step,
        "HIGH_B_KL_LOSS": high_b_kl,
    }
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    for k, v in payload.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()
