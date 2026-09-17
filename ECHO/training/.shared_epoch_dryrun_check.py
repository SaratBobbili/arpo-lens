"""No-GPU check for shared-epoch nested RL: 288 steps, batch 136, shared cursor."""

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
    "trainer.total_epochs=4",
    "trainer.save_freq=72",
    "trainer.test_freq=72",
    "phases.shared_prompt_stream=true",
    "phases.high_level.num_iters=8",
    "phases.low_level.num_iters=8",
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

assert trainer._shared_prompt_stream is True
assert isinstance(trainer.train_dataloader, _PhaseDataloaders)
assert trainer.total_training_steps == 4 * 8 * 9 == 288, trainer.total_training_steps
assert trainer._phase_dataloaders["low_level"] is trainer._phase_dataloaders["high_level"]
bs = trainer._phase_dataloaders["low_level"].batch_size
assert bs == (10000 // 72) // 8 * 8 == 136, bs
assert bs % 8 == 0, bs

# Shared cursor: consecutive phase pulls advance the same stream.
ll_first = trainer._next_batch_dict("low_level")["index"].tolist()
hl_next = trainer._next_batch_dict("high_level")["index"].tolist()
assert ll_first != hl_next

state = trainer.train_dataloader.state_dict()
assert set(state) == {"low_level", "high_level"}
trainer.train_dataloader.load_state_dict(state)

# Worker horizon mirror: shared scales by E.
assert int(cfg.actor_rollout_ref.total_epochs) == 4
assert bool(cfg.actor_rollout_ref.phases.get("shared_prompt_stream", False)) is True

print("total_training_steps:", trainer.total_training_steps)
print("shared batch:", bs, "batches/pass:", len(trainer._phase_dataloaders["low_level"]))
print("SHARED EPOCH DRYRUN OK")
