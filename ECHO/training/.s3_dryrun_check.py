"""No-GPU check for S3: derived per-phase batch sizes, step accounting, cycling iterators."""

import os
import sys

ECHO_TOP = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VERL_ROOT = os.environ["VERL_ROOT"]
sys.path[:0] = [VERL_ROOT, ECHO_TOP]

import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import Dataset

from training.echo_ray_trainer import RayECHOTrainer, _PhaseDataloaders

OVERRIDES = [
    "algorithm.adv_estimator=grpo",
    "actor_rollout_ref.rollout.n=16",
    "trainer.n_gpus_per_node=8",
    "trainer.nnodes=1",
    "trainer.save_freq=5",
    "trainer.test_freq=5",
    "phases.shared_prompt_stream=false",
    "phases.high_level.num_iters=79",
    "phases.low_level.num_iters=79",
    "phases.high_level.group_size=16",
    "phases.low_level.group_size=16",
    "phases.low_level.rollout.strategy=aepo",
]


class PromptDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "input_ids": torch.zeros(4, dtype=torch.long),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "position_ids": torch.arange(4, dtype=torch.long),
            "raw_prompt_ids": [1, 2, 3, 4],
            "index": idx,
        }


with initialize_config_dir(config_dir=os.path.join(ECHO_TOP, "training", "config"), version_base=None):
    cfg = compose(config_name="echo_trainer", overrides=OVERRIDES)

from verl.utils.dataset.rl_dataset import collate_fn

trainer = object.__new__(RayECHOTrainer)
trainer.config = cfg
trainer.tokenizer = None
trainer.processor = None
trainer._create_dataloader(PromptDataset(10000), PromptDataset(64), collate_fn, None)

assert isinstance(trainer.train_dataloader, _PhaseDataloaders)
assert trainer.total_training_steps == 79 * 80, trainer.total_training_steps
for phase in ("low_level", "high_level"):
    bs = trainer._phase_dataloaders[phase].batch_size
    assert bs == (10000 // 79) // 8 * 8, (phase, bs)
    assert bs % 8 == 0, (phase, bs)

# Independent streams: same index must not be handed to both phases in lockstep.
ll_first = trainer._next_batch_dict("low_level")["index"].tolist()
hl_first = trainer._next_batch_dict("high_level")["index"].tolist()
assert ll_first != hl_first

# One full LL pass per outer cycle, then wrap-around without raising StopIteration.
seen = 1
for _ in range(2 * int(cfg.phases.low_level.num_iters)):
    batch = trainer._next_batch_dict("low_level")
    assert batch["input_ids"].shape[0] == trainer._phase_dataloaders["low_level"].batch_size
    seen += 1

state = trainer.train_dataloader.state_dict()
assert set(state) == {"low_level", "high_level"}
trainer.train_dataloader.load_state_dict(state)

print("total_training_steps:", trainer.total_training_steps)
print("ll batch:", trainer._phase_dataloaders["low_level"].batch_size,
      "batches/pass:", len(trainer._phase_dataloaders["low_level"]))
print("hl batch:", trainer._phase_dataloaders["high_level"].batch_size,
      "batches/pass:", len(trainer._phase_dataloaders["high_level"]))
print("ll iterations served (with wrap-around):", seen)
print("S3 DRYRUN OK")
