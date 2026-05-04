"""
Top-level driver for ECHO cross-checkpoint HL/LL ping-pong validation.

Per combo (HL_step, LL_step) in cartesian product of cfg.xckpt.hl_steps x ll_steps:
  1. Lazily merge the FSDP shards under <checkpoint_root>/global_step_<step>/actor
     into HF format under <hf_cache_dir>/<run_name>/global_step_<step> if missing.
  2. Slice the val parquet into `dp` contiguous shards.
  3. Spawn `dp` shard worker subprocesses in parallel, each pinned to a 2-GPU
     pair via CUDA-isolated engine subprocesses; wait for all to finish.
  4. Score the union of per-shard JSONLs via deep_research_echo.compute_score
     and dump the per-combo summary JSON + scored JSONL.

After all combos: render the F1 / bad-format comparison plot.

All paths/sampling/tool/engine knobs flow from xckpt_eval.yaml; no hardcoded
behavior lives in this file.
"""

import os
import subprocess
import sys
from itertools import product
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from recipe.echo_xckpt_eval.aggregate import score_combo, write_summary
from recipe.echo_xckpt_eval.checkpoint_utils import merge_fsdp_to_hf, resolve_step_paths
from recipe.echo_xckpt_eval.plot import plot_summary


def _shard_ranges(n_rows: int, dp: int) -> list[tuple[int, int]]:
    """Split [0, n_rows) into `dp` contiguous chunks; sizes differ by <=1."""
    base, rem = divmod(n_rows, dp)
    out, cursor = [], 0
    for i in range(dp):
        size = base + (1 if i < rem else 0)
        out.append((cursor, cursor + size))
        cursor += size
    return out


def _resolve_gpus(cfg) -> list[int]:
    if cfg.xckpt.engine.gpus is not None:
        gpus = list(cfg.xckpt.engine.gpus)
    else:
        gpus = list(range(2 * cfg.xckpt.engine.dp))
    assert len(gpus) >= 2 * cfg.xckpt.engine.dp, (
        f"Need at least 2*dp = {2 * cfg.xckpt.engine.dp} GPUs; got {gpus}"
    )
    return gpus


def _ensure_merged(cfg, step: int) -> str:
    actor, target = resolve_step_paths(
        cfg.xckpt.checkpoint_root, cfg.xckpt.hf_cache_dir, cfg.xckpt.run_name, step
    )
    return merge_fsdp_to_hf(actor, target)


def _launch_combo(cfg, hl_step: int, ll_step: int, hl_path: str, ll_path: str) -> list[str]:
    """Run one (HL, LL) combo across all DP shards in parallel; return the per-shard
    JSONL output paths once all finish."""
    n_rows = len(pd.read_parquet(cfg.xckpt.data.val_parquet))
    ranges = _shard_ranges(n_rows, cfg.xckpt.engine.dp)
    gpus = _resolve_gpus(cfg)

    combo_label = f"hl{hl_step}_ll{ll_step}"
    rollouts_dir = Path(cfg.xckpt.output_dir) / "rollouts" / combo_label
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    shard_cfg_dir = Path(cfg.xckpt.output_dir) / "shard_configs" / combo_label
    shard_cfg_dir.mkdir(parents=True, exist_ok=True)

    shard_paths, procs = [], []
    for shard_idx, (lo, hi) in enumerate(ranges):
        hl_gpu, ll_gpu = gpus[2 * shard_idx], gpus[2 * shard_idx + 1]
        out_path = str(rollouts_dir / f"shard_{shard_idx}.jsonl")
        shard_paths.append(out_path)

        # Build a per-shard OmegaConf snapshot embedding the shard slice / GPUs / paths.
        shard_cfg = OmegaConf.create({"xckpt": OmegaConf.to_container(cfg.xckpt, resolve=True)})
        shard_cfg.xckpt.shard = OmegaConf.create({
            "idx": shard_idx,
            "lo": lo,
            "hi": hi,
            "hl_path": hl_path,
            "ll_path": ll_path,
            "hl_gpu": hl_gpu,
            "ll_gpu": ll_gpu,
            "output_path": out_path,
        })
        shard_cfg_path = str(shard_cfg_dir / f"shard_{shard_idx}.yaml")
        with open(shard_cfg_path, "w") as f:
            OmegaConf.save(shard_cfg, f)

        # PYTHONPATH already includes verl root via the launch script; keep the
        # env explicit so the subprocess imports resolve identically.
        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)  # let the engine subprocess set it itself.
        log_path = str(rollouts_dir / f"shard_{shard_idx}.log")
        log_f = open(log_path, "w")
        proc = subprocess.Popen(
            [sys.executable, "-m", "recipe.echo_xckpt_eval.shard_worker", "--config", shard_cfg_path],
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        procs.append((proc, log_f, log_path))

    # Wait for all shard workers; surface a clear error if any exits non-zero.
    failures = []
    for proc, log_f, log_path in procs:
        rc = proc.wait()
        log_f.close()
        if rc != 0:
            failures.append((log_path, rc))
    if failures:
        msgs = "\n".join(f"  - {p} (rc={rc})" for p, rc in failures)
        raise RuntimeError(f"Shard worker(s) failed for combo {combo_label}:\n{msgs}")

    return shard_paths


def _validator_profile(cfg) -> str:
    from verl.utils.reward_score.deep_research_echo import resolve_validator_profile
    return resolve_validator_profile(OmegaConf.to_container(cfg.xckpt.rollout.mask_categories, resolve=True))


@hydra.main(config_path="config", config_name="xckpt_eval", version_base=None)
def main(cfg):
    OmegaConf.resolve(cfg)
    out_dir = Path(cfg.xckpt.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_root = out_dir / "summaries"
    summary_root.mkdir(exist_ok=True)
    print("Resolved config:")
    print(OmegaConf.to_yaml(cfg))

    profile = _validator_profile(cfg)
    print(f"Resolved validator profile: {profile}")

    # Pre-merge the (up to 4) unique step ids so each combo just looks up paths.
    step_paths: dict[int, str] = {}
    for step in tqdm(sorted(set(list(cfg.xckpt.hl_steps) + list(cfg.xckpt.ll_steps))), desc="merging FSDP -> HF"):
        step_paths[step] = _ensure_merged(cfg, step)

    combos = list(product(cfg.xckpt.hl_steps, cfg.xckpt.ll_steps))
    for hl_step, ll_step in tqdm(combos, desc="combos"):
        combo_label = f"hl{hl_step}_ll{ll_step}"
        shard_paths = _launch_combo(cfg, hl_step, ll_step, step_paths[hl_step], step_paths[ll_step])
        summary = score_combo(shard_paths, validator_profile=profile)
        write_summary(summary, str(summary_root), combo_label)
        ov = summary["overall"]
        print(f"[{combo_label}] f1_mean={ov['f1_mean']:.4f}  bad_format_rate={ov['bad_format_rate']:.4f}  n_total={ov['n_total']}  n_good={ov['n_good']}")

    plot_path = plot_summary(
        str(summary_root),
        combos=[tuple(c) for c in combos],
        filename=cfg.xckpt.plot.filename,
        title=cfg.xckpt.plot.title,
    )
    print(f"Plot written: {plot_path}")


if __name__ == "__main__":
    main()
