"""
FSDP -> HF merge cache for the cross-checkpoint pipeline.

ECHO trainer saves each checkpoint as `global_step_<n>/actor/` containing FSDP
shards (`model_world_size_<W>_rank_<r>.pt`) plus the HF config/tokenizer files.
vLLM consumes only HF-format directories, so this module merges shards once per
(run_name, step) into `<hf_cache_dir>/<run_name>/global_step_<step>/` and
returns that path on subsequent calls.
"""

import os
import sys
from pathlib import Path


def _ensure_scripts_on_path():
    # `scripts/` is not a package; reach FSDPModelMerger by adding the verl root
    # (the dir containing `scripts/`) to sys.path and importing the module.
    verl_root = Path(__file__).resolve().parents[2]
    scripts_dir = verl_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def merge_fsdp_to_hf(actor_dir: str, target_dir: str) -> str:
    """Merge an FSDP-sharded actor directory into an HF model directory.

    Idempotent: if `target_dir/config.json` already exists we treat it as cached
    and return immediately. Returns `target_dir`.
    """
    target = Path(target_dir)
    if (target / "config.json").exists():
        return str(target)

    _ensure_scripts_on_path()
    from model_merger import FSDPModelMerger, ModelMergerConfig

    target.mkdir(parents=True, exist_ok=True)
    cfg = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=str(actor_dir),
        # Actor dir already holds config.json/tokenizer files saved by FSDPCheckpointManager,
        # so use it as the HF config source (avoids needing a separate base model path).
        hf_model_config_path=str(actor_dir),
        target_dir=str(target),
    )
    FSDPModelMerger(cfg).merge_and_save()
    return str(target)


def resolve_step_paths(checkpoint_root: str, hf_cache_dir: str, run_name: str, step: int) -> tuple[str, str]:
    """Return (actor_fsdp_dir, hf_target_dir) for `global_step_<step>`."""
    actor = os.path.join(checkpoint_root, f"global_step_{step}", "actor")
    target = os.path.join(hf_cache_dir, run_name, f"global_step_{step}")
    return actor, target
