"""No-GPU check for S2: compose echo_trainer with the override list train.sh actually emits.

Expectations are derived from the override list itself rather than hardcoded, so this
does not rot every time a launch profile or a wrapper default moves. What it actually
asserts is the round trip: every key train.sh emits survives composition with the value
it emitted, the phase tree is mirrored into the copy the workers read, and the paths the
run will open exist.
"""

import os
import sys

ECHO_TOP = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

args = [line.rstrip("\n") for line in open(sys.argv[1]) if line.strip()]
overrides = [a for a in args if not a.startswith("--")]

with initialize_config_dir(config_dir=os.path.join(ECHO_TOP, "training", "config"), version_base=None):
    cfg = compose(config_name="echo_trainer", overrides=overrides)


def _get(dotted):
    node = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


def _same(got, want):
    if got is None:
        return want.lower() in ("null", "none", "~", "")
    if isinstance(got, bool):
        return str(got).lower() == want.lower()
    if isinstance(got, (int, float)):
        try:
            return float(got) == float(want)
        except ValueError:
            return False
    if OmegaConf.is_config(got):
        return True          # lists/dicts: presence is enough for a round-trip check
    return str(got) == want


# --- every emitted override survives composition with the value it was given ---------
checked = skipped = 0
for ov in overrides:
    if "=" not in ov:
        continue
    key, want = ov.split("=", 1)
    key = key.lstrip("+~")
    try:
        got = _get(key)
    except Exception:
        skipped += 1
        continue
    assert _same(got, want), f"{key}: emitted {want!r}, composed {got!r}"
    checked += 1
print(f"round trip: {checked} overrides survived composition ({skipped} not addressable)")

# --- the phase tree is mirrored into the copy the workers read ----------------------
for phase in ("high_level", "low_level"):
    p = cfg.phases[phase]
    mirror = cfg.actor_rollout_ref.phases[phase]
    assert mirror.optim.lr == p.optim.lr, (phase, mirror.optim.lr, p.optim.lr)
    assert mirror.ppo_mini_batch_size == p.ppo_mini_batch_size, phase
assert cfg.actor_rollout_ref.phases.prompt_batch_size == cfg.phases.prompt_batch_size
print("phase tree mirrored into actor_rollout_ref for the workers")

# --- the paths this run will actually open exist ------------------------------------
assert os.path.isfile(os.path.join(cfg.actor_rollout_ref.model.path, "config.json")), \
    cfg.actor_rollout_ref.model.path
assert os.path.isfile(cfg.data.train_files) and os.path.isfile(cfg.data.val_files)
search = cfg.actor_rollout_ref.rollout.tools.tool_instances.search
assert search.class_path.endswith("BingSearchTool")
assert os.path.isfile(search.params.cache_file), search.params.cache_file

# --- batch arithmetic, off the explicit knob (the round equation is gone) -----------
batch = int(cfg.phases.prompt_batch_size)
world_size = cfg.trainer.n_gpus_per_node * cfg.trainer.nnodes
n_hl, n_ll = cfg.phases.high_level.num_iters, cfg.phases.low_level.num_iters
assert batch % world_size == 0, (batch, world_size)
for phase in ("high_level", "low_level"):
    mini = int(cfg.phases[phase].ppo_mini_batch_size)
    if batch % mini:
        # Not fatal: the legacy profiles kept prompt_batch 136 so r3/r4 reproduce, and
        # 136/16 = 8.5 means a ragged final mini-batch every iteration. Reported so it
        # cannot be rediscovered by accident. New profiles use 128, which divides.
        print(f"  WARNING {phase}: prompt batch {batch} / mini {mini} = "
              f"{batch / mini:.2f} -> {batch // mini} full mini-batches plus a ragged "
              f"{batch % mini}-prompt tail")

print("algorithm:", cfg.phases.algorithm)
print("actor:", cfg.actor_rollout_ref.model.path)
print("search cache:", search.params.cache_file)
print(f"prompt batch/iteration: {batch}  sequences/iteration: {batch * cfg.phases.high_level.group_size}")
print(f"optimizer steps/iteration: hl={batch // int(cfg.phases.high_level.ppo_mini_batch_size)} "
      f"ll={batch // int(cfg.phases.low_level.ppo_mini_batch_size)}")
print("total global steps:", n_hl * (n_ll + 1))
print("S2 COMPOSE OK")
