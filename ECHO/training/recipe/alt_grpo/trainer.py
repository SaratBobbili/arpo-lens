"""ALTERNATING-GRPO recipe trainer: the loop, and nothing else.

Role-masked GRPO over two phases on the shared PhaseTrainerBase machinery. This file
owns the cycle structure, which prompts each phase iteration draws, and the step
accounting. It never reads phases.response.

Cycle: num_iters[low_level] low-level iterations, then num_iters[high_level]
high-level iterations. The counts are per-phase iteration counts per cycle, NOT a
nesting: 1/1 is the April-May 2026 loop (one low-level iteration, then one high-level
iteration on the same 128 prompts, re-rolled from the updated policy), which is what
config_alt_grpo.yaml sets. High-level iteration j re-rolls the prompts low-level
iteration j adapted on; if there are more high-level than low-level iterations, the
extra ones draw fresh prompts.

Epochs are passes over the prompt stream: the low-level iterations consume it (the
high-level ones re-roll), so cycles per epoch = batches per pass // num_iters[low_level]
(78 // 1 = 78 at 128 prompts over 10k). save_freq / test_freq count cycles, and a
checkpoint lands on the last high-level iteration of a cycle. The per-phase LR
scheduler horizons follow the same arithmetic and are handed to the workers through
phases.<phase>.optim.total_training_steps.
"""
from pprint import pprint

from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from ...core.phase_trainer import PhaseTrainerBase


class AltGRPOTrainer(PhaseTrainerBase):
    """ALTERNATING-GRPO: the recipe loop over PhaseTrainerBase."""

    def _cycle_shape(self) -> tuple:
        n_ll = int(self._phase_cfg("low_level").num_iters)
        n_hl = int(self._phase_cfg("high_level").num_iters)
        assert n_ll >= 1 and n_hl >= 1, (
            f"alt_grpo needs at least one iteration per phase per cycle, got "
            f"low_level={n_ll}, high_level={n_hl}"
        )
        return n_ll, n_hl

    def _next_batch_dict(self, phase_name: str):
        """ALTERNATING-GRPO: high-level iteration j re-rolls the prompts low-level
        iteration j adapted on, rather than pulling new ones.

        Each low-level iteration pulls its own batch and appends it to the cycle's list;
        the j-th high-level iteration returns the j-th entry (or pulls a fresh batch when
        the cycle has more high-level than low-level iterations). The rollout happens
        downstream in _phase_rollout_to_scored_batch, so the generations are fresh from
        the updated policy even though the prompts are the same.
        """
        if phase_name == "low_level":
            batch_dict = self._pull_batch_dict(phase_name)
            self._cycle_batches.append(batch_dict)
            return batch_dict
        j = self._hl_index
        self._hl_index += 1
        if j < len(self._cycle_batches):
            return self._cycle_batches[j]
        return self._pull_batch_dict(phase_name)

    def fit(self):
        """Alternating GRPO: cycles of N_LL low-level then N_HL high-level iterations,
        `_cycles_per_epoch` cycles per epoch, `trainer.total_epochs` epochs (shared
        prompt stream) or one pass (legacy per-phase streams)."""
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        resolved_config = OmegaConf.to_container(self.config, resolve=True)
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=resolved_config,
        )
        logger.log_hparams(resolved_config)

        self.global_steps = 0
        self._best_metric_value = float("-inf")
        self._best_metric_step = -1
        self._best_metric_key = None
        self._last_val_metrics = None
        self._current_epoch = 0

        self._init_logging_data()

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # Checkpoints only land on the high-level iteration that closes a cycle, so the
        # resumed step count maps back to a whole number of completed cycles.
        # we start from step 1
        self.global_steps += 1

        self._run_cycles(logger, progress_bar)

        pprint(f"Final validation metrics: {self._last_val_metrics}")
        progress_bar.close()

    def _run_cycles(self, logger, progress_bar) -> None:
        """Every cycle from the (possibly resumed) global step to the end of training."""
        n_ll, n_hl = self._cycle_shape()
        cycles_done = (self.global_steps - 1) // self._cycle_len
        start_epoch = cycles_done // self._cycles_per_epoch
        for epoch in range(start_epoch, self._epochs):
            self._current_epoch = epoch
            first = cycles_done % self._cycles_per_epoch if epoch == start_epoch else 0
            for cycle in range(first, self._cycles_per_epoch):
                self._run_cycle(cycle, n_ll, n_hl, logger, progress_bar)

    def _run_cycle(self, cycle: int, n_ll: int, n_hl: int, logger, progress_bar) -> None:
        """One cycle: N_LL low-level iterations, then N_HL high-level iterations. The last
        high-level iteration closes the cycle (validation / checkpoint gate)."""
        self._cycle_batches = []
        self._hl_index = 0
        for _ in range(n_ll):
            self._run_phase_iteration("low_level", cycle, end_of_cycle=False, logger=logger, progress_bar=progress_bar)
        for j in range(n_hl):
            self._run_phase_iteration(
                "high_level", cycle, end_of_cycle=(j == n_hl - 1), logger=logger, progress_bar=progress_bar
            )

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        super()._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        self._set_cycle_accounting(batches_per_pass=len(self._phase_dataloaders["low_level"]))

    def _set_cycle_accounting(self, batches_per_pass: int) -> None:
        """Cycles per epoch, total steps, and the workers' LR-scheduler horizons, all from
        the cycle shape and the prompt stream length. Runs before init_workers, so the
        horizons written into actor_rollout_ref.phases reach the workers (verl does the
        same for actor.optim.total_training_steps)."""
        n_ll, n_hl = self._cycle_shape()
        self._cycle_len = n_ll + n_hl
        self._cycles_per_epoch = int(batches_per_pass) // n_ll
        assert self._cycles_per_epoch >= 1, (
            f"the prompt stream has {batches_per_pass} batches per pass but a cycle needs "
            f"{n_ll} low-level iterations"
        )
        self._epochs = int(self.config.trainer.total_epochs) if self._shared_prompt_stream else 1
        total_cycles = self._epochs * self._cycles_per_epoch
        self.total_training_steps = total_cycles * self._cycle_len
        horizons = {"high_level": total_cycles * n_hl, "low_level": total_cycles * n_ll}
        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            for phase, steps in horizons.items():
                self.config.actor_rollout_ref.phases[phase].optim.total_training_steps = steps
        print(
            f"[alt_grpo] cycle = {n_ll} low-level + {n_hl} high-level iterations; "
            f"{self._cycles_per_epoch} cycles/epoch x {self._epochs} epochs = {total_cycles} cycles; "
            f"scheduler horizons {horizons}"
        )
        print(f"Total training steps: {self.total_training_steps}")

    def _hl_updates_done(self, cycle: int) -> int:
        """Cycles completed once this one closes; save_freq / test_freq count cycles."""
        return self._current_epoch * self._cycles_per_epoch + cycle + 1
