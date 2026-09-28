"""ALTERNATING-GRPO recipe trainer: the loop, and nothing else.

Role-masked GRPO over two phases on the shared PhaseTrainerBase machinery. This file
owns the cycle structure, which prompts each phase iteration draws, and the step
accounting. It never reads phases.response.

Cycle (kept as it was for this commit; the per-phase update-count semantics are the
next change): num_iters[low_level] low-level iterations, then ONE high-level
iteration, for num_iters[high_level] cycles per epoch. The high-level iteration
re-rolls the prompts the last low-level iteration adapted on (same prompts, fresh
rollouts from the updated policy).
"""
from pprint import pprint

from tqdm import tqdm

from ...core.phase_trainer import PhaseTrainerBase


class AltGRPOTrainer(PhaseTrainerBase):
    """ALTERNATING-GRPO: the recipe loop over PhaseTrainerBase."""

    def _next_batch_dict(self, phase_name: str):
        """ALTERNATING-GRPO: the high-level iteration re-rolls the prompts the low-level
        phase just adapted on, rather than pulling new ones.

        Each low-level iteration pulls and caches its own batch; the high-level iteration
        returns the cached one. The rollout happens downstream in
        _phase_rollout_to_scored_batch, so the generations are fresh from the updated
        policy even though the prompts are the same.
        """
        if phase_name == "high_level" and getattr(self, "_cycle_batch_dict", None) is not None:
            return self._cycle_batch_dict
        batch_dict = self._pull_batch_dict(phase_name)
        self._cycle_batch_dict = batch_dict
        return batch_dict

    def fit(self):
        """Nested-phase GRPO: `num_iters[low_level]` low-level iterations inside each of
        `num_iters[high_level]` outer cycles, each cycle closed by one high-level iteration.

        Shared mode wraps that cycle loop in `trainer.total_epochs` outer epochs.
        """
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

        num_hl_iters = int(self._phase_cfg("high_level").num_iters)
        num_ll_iters = int(self._phase_cfg("low_level").num_iters)
        cycle_len = num_ll_iters + 1
        steps_per_epoch = num_hl_iters * cycle_len

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # Checkpoints only land on the high-level iteration that closes a cycle, so the
        # resumed step count maps back to a whole number of completed outer cycles.
        # we start from step 1
        self.global_steps += 1

        if self._shared_prompt_stream:
            total_epochs = int(self.config.trainer.total_epochs)
            start_epoch = (self.global_steps - 1) // steps_per_epoch
            start_cycle = ((self.global_steps - 1) % steps_per_epoch) // cycle_len
            for epoch in range(start_epoch, total_epochs):
                self._current_epoch = epoch
                cycle_start = start_cycle if epoch == start_epoch else 0
                for hl_cycle in range(cycle_start, num_hl_iters):
                    self._run_cycle(hl_cycle, num_ll_iters, logger, progress_bar)
        else:
            start_cycle = (self.global_steps - 1) // cycle_len
            for hl_cycle in range(start_cycle, num_hl_iters):
                self._run_cycle(hl_cycle, num_ll_iters, logger, progress_bar)


        pprint(f"Final validation metrics: {self._last_val_metrics}")
        progress_bar.close()

    def _run_cycle(self, hl_cycle: int, num_ll_iters: int, logger, progress_bar) -> None:
        """One outer cycle: N_LL low-level iterations, then the high-level iteration."""
        for _ in range(num_ll_iters):
            self._run_phase_iteration("low_level", hl_cycle, end_of_cycle=False, logger=logger, progress_bar=progress_bar)
        self._run_phase_iteration("high_level", hl_cycle, end_of_cycle=True, logger=logger, progress_bar=progress_bar)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        super()._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        num_hl_iters = int(self._phase_cfg("high_level").num_iters)
        num_ll_iters = int(self._phase_cfg("low_level").num_iters)
        if self._shared_prompt_stream:
            total_epochs = int(self.config.trainer.total_epochs)
            self.total_training_steps = total_epochs * num_hl_iters * (num_ll_iters + 1)
        else:
            self.total_training_steps = num_hl_iters * (num_ll_iters + 1)
        print(f"Total training steps: {self.total_training_steps}")

    def _hl_updates_done(self, hl_cycle: int) -> int:
        if self._shared_prompt_stream:
            n_hl = int(self._phase_cfg("high_level").num_iters)
            return self._current_epoch * n_hl + hl_cycle + 1
        else:
            return hl_cycle + 1
