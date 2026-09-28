"""ECHO recipe trainer: Algorithm 1 (hypergradient) with the adjoint / exact estimators.

Owns the Algorithm-1 round loop (N_LL follower iterations then one leader iteration
per round, fresh query groups every iteration, snapshot / restore at the round
boundaries, the final response), the response-term configuration and its validation.
The shared two-phase machinery is core/phase_trainer.py (PhaseTrainerBase). The AHO
estimator is its own recipe (training/recipe/aho) and is not selectable here.
"""
import os
import shutil

from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl.utils.metric import reduce_metrics

from . import response as echo_response
from ...core.phase_trainer import PhaseTrainerBase


class EchoTrainer(PhaseTrainerBase):
    """PhaseTrainerBase plus the Algorithm-1 round structure and the response term."""

    def _response_cfg(self):
        return self.config.phases.get("response", None)

    def _response_enabled(self) -> bool:
        """Algorithm 1's round structure: reset/discard, r_valid, one-step, final response."""
        cfg = self._response_cfg()
        return bool(cfg.get("enabled", False)) if cfg is not None else False

    def _response_gradient_enabled(self) -> bool:
        """The g_resp term itself. Gated separately from the structure above.

        The mask composition that once annihilated the terminal seed was fixed in 1596bcc
        (the unmasked per-trajectory scalar of v10 Eq. (30) now rides alongside the
        pre-masked token advantages), so g_resp is no longer identically zero. It is still
        the sampling half of Eq. (35) only, and measured at 0.1-3% of the direct gradient
        with `coef=1.0` -- see the config comment on phases.response.gradient. Keeping this
        separate lets the round structure run without paying 2K extra full-batch passes.
        """
        cfg = self._response_cfg()
        if cfg is None or not self._response_enabled():
            return False
        return bool(cfg.get("gradient", False))



    def _response_exact_enabled(self) -> bool:
        """The exact K=1 hypergradient rather than the sampling-half-only estimator.

        On: both halves of Eq. (35) -- the fixed-data Hessian term by central differences
        on the gradient, and the sampling score term -- each carrying the follower's true
        optimizer Jacobian D_0. Nothing is left as a tuning scalar, so `coef` is ignored.
        K > 1 runs the score-corrected reverse sweep of Eq. (54) rather than truncating the
        adjoint to g_fol; at K = 1 the chain is empty and the question does not arise.
        """
        cfg = self._response_cfg()
        if cfg is None or not self._response_gradient_enabled():
            return False
        return bool(cfg.get("exact", False))

    def _response_group_aligned_enabled(self) -> bool:
        """Replay the follower record in query-GROUP units rather than token-budget blocks.

        OFF by default. Eq. (49) pairs each group's gradient with its own score, but making
        that partition agree across ranks is unsolved here: excision fragments the uids so
        group counts differ rank to rank, and padding the counts leaves group SIZES
        differing, so a globally agreed chunk count is not locally achievable. The result
        was a NCCL desync -- some ranks in the FSDP all-gather, others already in the
        per-group dot() all-reduce. And even once synced, FSDP reduce-scatters, so p.grad
        is the DP average over eight different groups anyway. Off until that is solved.
        """
        cfg = self._response_cfg()
        if cfg is None or not self._response_exact_enabled():
            return False
        return bool(cfg.get("group_aligned", False))

    def _response_curvature_enabled(self) -> bool:
        """The parametric half of H_yx,s, i.e. the central-difference Hessian term.

        Off by default: FSDP1's bf16 all-gather rounds away most of a globally sized
        perturbation, so the difference comes out 20-70x short and hard-thresholded on
        |g_j|/|w_j|. See phases.response.curvature. Only meaningful with `exact`.
        """
        cfg = self._response_cfg()
        if cfg is None or not self._response_exact_enabled():
            return False
        return bool(cfg.get("curvature", False))

    def _validate_response_config(self) -> None:
        """Algorithm 1 takes one update per fresh group; the code must too.

        C.3: "if a group is reused, each clipped optimizer epoch and its fixed behavior
        snapshot is a separate substep of the adaptation map." With several mini-batches
        per iteration the true adaptation map would be K x n_minibatch substeps, and the
        stashed-record-to-substep correspondence the reverse sweep relies on would break.
        Requiring one step per iteration keeps K = phases.low_level.num_iters exactly.
        """
        cfg = self._response_cfg()
        assert cfg is None or cfg.get("follower_return", None) is None, (
            "phases.response.follower_return is retired (2026-09-27): scoring is phase-owned "
            "and algorithm-independent, so the follower is scored by the task return under "
            "its own schema rows on every family. Remove the key."
        )
        assert cfg is None or str(cfg.get("estimator", "adjoint")).lower() == "adjoint", (
            "recipe/echo runs the adjoint / exact estimators; "
            "phases.response.estimator=aho runs under recipe/aho."
        )
        if not self._response_enabled():
            return
        ppo_epochs = int(self.config.actor_rollout_ref.actor.get("ppo_epochs", 1))
        assert ppo_epochs == 1, (
            f"phases.response.enabled requires actor.ppo_epochs=1, got {ppo_epochs}. Each "
            "optimizer epoch over a reused group is a separate substep of the adaptation map."
        )
        for phase_name in self._PHASE_NAMES:
            prompt_batch = self._phase_prompt_batch_sizes[phase_name]
            mini = int(self._phase_cfg(phase_name).ppo_mini_batch_size)
            assert mini == prompt_batch, (
                f"phases.response.enabled requires one optimizer step per {phase_name} "
                f"iteration: set phases.{phase_name}.ppo_mini_batch_size to that phase's "
                f"prompt batch ({prompt_batch}), got {mini}."
            )
        if self._response_exact_enabled():
            k = int(self._phase_cfg("low_level").num_iters)
            curvature = self._response_curvature_enabled()
            passes = 4 if curvature else 2
            if k > 1:
                # Not an error, and no longer an approximation. The K steps run in reverse
                # with the score-corrected adjoint of Eq. (54): with grad^fix out, Eq. (51)
                # makes each adjoint step a scalar dot plus one backward, and shared routing
                # (S_x = S_y) lets one accumulation serve both the adjoint update and the
                # leader contribution. Costs K weight snapshots on the host.
                print(
                    f"[echo] phases.response.exact with K={k}: reverse sweep over {k} steps "
                    f"with the Eq. (54) adjoint run, not truncated; {passes} passes per "
                    f"stashed block and {k} host weight snapshots per round."
                )
            if not curvature:
                print(
                    "[echo] phases.response.curvature=false: computing the DISTRIBUTIONAL "
                    "half of g_resp only (score term), at "
                    f"{passes} passes per stashed block. This is not the exact "
                    "hypergradient at any K -- per-step w_s, D_s and the c baseline are "
                    "kept, the central-difference Hessian term is not."
                )
            follower_optim = self._phase_cfg("low_level").optim
            optimizer_name = str(follower_optim.get("optimizer", "adamw")).lower()
            assert optimizer_name == "sgd", (
                "phases.response.exact needs phases.low_level.optim.optimizer=sgd. AdamW's "
                "first step from reset moments is -lr*g/(|g|+eps); its EXACT Jacobian is "
                "lr*eps/(|g|+eps)^2 ~ 4e-12, so differentiating through it correctly gives "
                "a response term of zero. SGD gives D_0 = lr*I."
            )
            assert float(follower_optim.get("weight_decay", 0.0)) == 0.0, (
                "phases.response.exact needs phases.low_level.optim.weight_decay=0. SGD's "
                "coupled decay adds lambda*w to the gradient, and w depends on x, so a "
                "non-zero decay puts an extra lambda*I into d ghat/dx that this path does "
                "not model."
            )

    def _run_final_response(self, num_ll_iters: int, logger, progress_bar) -> None:
        """Algorithm 1 line 16: return x_Nout *and* the response it induces.

        The trained artifact the algorithm defines is the pair (x_Nout, xi_K(x_Nout)), not
        the leader alone. Every checkpoint written during training holds w_core + x_t,
        because the round discards its adapted follower before stepping; so after the loop
        we adapt y_init to the final commitment under the same K-step protocol and save
        that separately. The plain actor checkpoints are left untouched.
        """
        if not self._response_enabled():
            return

        print(f"[final_response] adapting y_init to x_Nout for K={num_ll_iters} follower steps")
        self.actor_rollout_wg.snapshot_leader_weights()
        self.actor_rollout_wg.reset_follower_optimizer()
        if self._response_gradient_enabled():
            self.actor_rollout_wg.clear_follower_records()

        # This is an adaptation, not training. Running it through _run_phase_iteration
        # would advance global_steps and the progress bar and -- via update_actor --
        # step the follower LR scheduler, all after training has finished. Save and
        # restore the counters around it so the K adaptation steps leave no trace on the
        # training record. The follower LR schedule is restored too: these steps consume
        # schedule the run's own accounting never allotted them.
        saved_steps = self.global_steps
        saved_lr_state = self.actor_rollout_wg.follower_scheduler_state()[0]
        for _ in range(num_ll_iters):
            # hl_cycle=-1 marks these steps as the post-training response in the logs;
            # with end_of_cycle=False it is only used as a metric label.
            self._run_phase_iteration(
                "low_level", -1, end_of_cycle=False, logger=logger, progress_bar=progress_bar,
            )
        self.global_steps = saved_steps
        self.actor_rollout_wg.restore_follower_scheduler_state(saved_lr_state)

        run_dir = self.config.trainer.default_local_dir
        dst = os.path.join(run_dir, "final_response")
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        os.makedirs(dst, exist_ok=True)
        self.actor_rollout_wg.save_checkpoint(
            local_path=os.path.join(dst, "actor"),
            hdfs_path=None,
            global_step=self.global_steps,
            max_ckpt_to_keep=None,
        )
        with open(os.path.join(dst, "README.txt"), "w") as f:
            f.write(
                "ECHO Algorithm 1 line 16: the adapted reasoning-tool pair.\n"
                f"x from global_step_{self.global_steps}, then K={num_ll_iters} follower\n"
                "GRPO steps from y_init under the same protocol used during training.\n"
            )

        # Put the leader back, so anything downstream sees x_Nout rather than x + y_K.
        self.actor_rollout_wg.restore_leader_weights()
        self.actor_rollout_wg.clear_follower_records()
        print(f"[final_response] wrote {dst}")

    # --- PhaseTrainerBase hooks ------------------------------------------------------

    def _validate_extra(self) -> None:
        self._validate_response_config()


    def _phase_meta_info(self, phase_name: str) -> dict:
        response_cfg = self._response_cfg()
        meta = {
            "response_enabled": self._response_gradient_enabled(),
            # Line 14 rides with the structure flag, not the gradient flag.
            "discard_follower": self._response_enabled(),
        }
        if response_cfg is not None:
            meta["response_coef"] = float(response_cfg.get("coef", 1.0))
            meta["response_replay_fraction"] = float(response_cfg.get("replay_fraction", 1.0))
            meta["response_exact"] = self._response_exact_enabled()
            meta["response_curvature"] = self._response_curvature_enabled()
            meta["response_group_aligned"] = self._response_group_aligned_enabled()
            meta["response_fd_rel"] = float(response_cfg.get("fd_rel", 2e-2))
        return meta


    def _next_batch_dict(self, phase_name: str):
        """Algorithm 1 draws distinct query groups for the follower and the leader
        (C.3/C.4), so every iteration pulls its own batch."""
        return self._pull_batch_dict(phase_name)

    def _open_round(self) -> None:
        """Algorithm 1 line 2: commit to x_t, clone y_0 = y_init, fresh optimizer state.

        y_init = 0, so the clone is implicit: snapshotting w_core + x_t is what makes
        the follower coordinate recoverable at line 14. See the block comment on
        EchoWorker.snapshot_leader_weights.
        """
        if not self._response_enabled():
            return
        self.actor_rollout_wg.snapshot_leader_weights()
        self.actor_rollout_wg.reset_follower_optimizer()
        if self._response_gradient_enabled():
            self.actor_rollout_wg.clear_follower_records()

    def _close_cycle(self) -> None:
        if self._response_gradient_enabled():
            self.actor_rollout_wg.clear_follower_records()

    # --- the Algorithm-1 round loop (PhaseTrainerBase leaves fit to the recipe) -------

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

        self._run_final_response(num_ll_iters, logger, progress_bar)

        pprint(f"Final validation metrics: {self._last_val_metrics}")
        progress_bar.close()

    def _run_cycle(self, hl_cycle: int, num_ll_iters: int, logger, progress_bar) -> None:
        """One outer cycle: N_LL low-level iterations, then the high-level iteration."""
        self._open_round()
        for _ in range(num_ll_iters):
            self._run_phase_iteration("low_level", hl_cycle, end_of_cycle=False, logger=logger, progress_bar=progress_bar)
        # The leader iteration restores W_x inside the worker between its last backward
        # and its optimizer step, so the adapted follower is discarded there (line 14).
        self._run_phase_iteration("high_level", hl_cycle, end_of_cycle=True, logger=logger, progress_bar=progress_bar)
        self._close_cycle()

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
