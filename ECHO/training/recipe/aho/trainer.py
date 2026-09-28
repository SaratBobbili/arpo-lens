"""AHO recipe trainer: Algorithm 1's round structure with the AHO response surrogate.

Owns the Algorithm-1 round loop (N_LL follower iterations then one leader iteration
per round, fresh query groups every iteration, snapshot / restore at the round
boundaries, the final response), the AHO configuration (tau, gamma), its validation,
and the leader-batch preparation (A_L and the token weights). The shared two-phase
machinery is core/phase_trainer.py (PhaseTrainerBase). This recipe never imports
training/recipe/echo; the adjoint / exact estimators live there.
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


class AhoTrainer(PhaseTrainerBase):
    """PhaseTrainerBase plus the Algorithm-1 round structure and the AHO response term."""

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


    def _aho_cfg(self):
        cfg = self._response_cfg()
        return cfg.get("aho", None) if cfg is not None else None

    def _aho_gamma(self) -> float:
        aho = self._aho_cfg()
        return float(aho.get("gamma", 1.0)) if aho is not None else 1.0

    def _aho_derived_tau(self) -> float:
        """The follower's entropy temperature as its loss actually applies it.

        lambda_ent H_tool (phases.low_level.entropy.reg_coeff, only when enabled) plus the
        follower's KL coefficient when actor.use_kl_loss: a KL to a fixed reference is
        entropy plus x-independent shaping, so it adds to the temperature.
        """
        low = self._phase_cfg("low_level")
        tau = 0.0
        if self._uses_entropy_regularizer(low):
            tau += self._entropy_reg_coeff(low)
        actor_cfg = self.config.actor_rollout_ref.actor
        if bool(actor_cfg.get("use_kl_loss", False)):
            tau += float(low.get("kl_loss_coef", actor_cfg.get("kl_loss_coef", 0.0)))
        return tau

    def _aho_tau(self) -> float:
        """tau for the 1/tau factor: derived from the follower's regularisers unless
        phases.response.aho.tau is set explicitly."""
        aho = self._aho_cfg()
        explicit = aho.get("tau", None) if aho is not None else None
        if explicit is None:
            return self._aho_derived_tau()
        return float(explicit)

    def _validate_aho_config(self) -> None:
        """What the AHO estimator's derivation needs from the rest of the config."""
        cfg = self._response_cfg()
        assert self._response_gradient_enabled(), (
            "phases.response.estimator=aho selects how g_resp is computed; it needs "
            "phases.response.gradient=true (and enabled=true)."
        )
        for key in ("exact", "curvature", "group_aligned"):
            assert not bool(cfg.get(key, False)), (
                f"phases.response.{key} belongs to the adjoint estimator and is not read "
                "under estimator=aho; unset it so the run's config says what runs."
            )
        high = self._phase_cfg("high_level")
        low = self._phase_cfg("low_level")
        assert str(high.get("advantage_algorithm", "grpo")) == "grpo", (
            "estimator=aho multiplies the UNMODULATED task advantage (scalar_advantages); "
            "phases.high_level.advantage_algorithm must be grpo."
        )
        assert str(low.get("advantage_algorithm", "grpo")) == "grpo", (
            "estimator=aho substitutes the follower's GRPO outcome advantage A_L for the "
            "value function; phases.low_level.advantage_algorithm must be grpo (the yaml "
            "default is `entropy`, which makes A_L = alpha * H_norm and the derivation false)."
        )
        for name, phase_cfg in (("high_level", high), ("low_level", low)):
            opefo = phase_cfg.get("opefo", None)
            assert not (opefo is not None and bool(opefo.get("enabled", False))), (
                f"estimator=aho needs phases.{name}.opefo.enabled=false."
            )
        tau = self._aho_tau()
        assert tau > 0.0, (
            "estimator=aho needs tau > 0. Either give the follower an entropy term "
            "(phases.low_level.entropy.enabled=true with reg_coeff>0) / KL term, or set "
            "phases.response.aho.tau explicitly."
        )
        derived = self._aho_derived_tau()
        if derived <= 0.0:
            print(
                f"[echo] estimator=aho with tau={tau} set explicitly while the follower has no "
                "entropy or KL term. The Boltzmann premise is then nominal: tau is a scale knob "
                "next to phases.response.coef, not the follower's temperature."
            )
        elif abs(derived - tau) > 1e-12:
            print(
                f"[echo] estimator=aho: phases.response.aho.tau={tau} overrides the follower's "
                f"own temperature {derived} (entropy reg_coeff + KL coef)."
            )
        print(
            f"[echo] estimator=aho: tau={tau}, gamma={self._aho_gamma()}. One extra leader-batch "
            "pass per round; no follower records, weight snapshots or replay passes."
        )


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
        assert self._response_enabled() and self._response_gradient_enabled(), (
            "recipe/aho needs phases.response.enabled=true and phases.response.gradient=true; "
            "a response-off run is recipe/echo's r3 profile."
        )
        assert cfg is not None and str(cfg.get("estimator", "aho")).lower() == "aho", (
            "recipe/aho runs the AHO estimator only; set phases.response.estimator=aho "
            "(the adjoint / exact estimators run under recipe/echo)."
        )
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
        self._validate_aho_config()

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
        print(f"[final_response] wrote {dst}")

    # --- PhaseTrainerBase hooks ------------------------------------------------------

    def _validate_extra(self) -> None:
        self._validate_response_config()

    def _rollout_meta_info(self, phase_name: str) -> dict:
        """Scoring is the same on every family; the only extra is AHO's second pass, which
        scores the leader batch under the follower's schema rows as well (for A_L)."""
        return {
            "compute_follower_score": phase_name == "high_level",
        }

    def _phase_meta_info(self, phase_name: str) -> dict:
        response_cfg = self._response_cfg()
        meta = {
            "response_enabled": self._response_gradient_enabled(),
            # Line 14 rides with the structure flag, not the gradient flag.
            "discard_follower": self._response_enabled(),
        }
        if response_cfg is not None:
            meta["response_coef"] = float(response_cfg.get("coef", 1.0))
            meta["aho_tau"] = self._aho_tau()
            meta["aho_gamma"] = self._aho_gamma()
        return meta

    def _after_advantage(self, phase_batch, phase_name: str) -> dict:
        """AHO estimator, leader phase: put A_L and the per-token weights on the batch.

        Runs after _apply_tool_failure_before_grpo and compute_advantage, so an excised
        sample already carries its fresh uid (a singleton group, hence A_L = 0) and a
        demoted one already carries its follower-side format penalty. Everything the actor's
        surrogate needs is then a plain batch tensor that survives micro-batching.
        """
        if phase_name != "high_level":
            return {}
        follower_scores = phase_batch.non_tensor_batch.get("follower_score")
        assert follower_scores is not None, (
            "estimator=aho needs R_L on the leader batch: reward_manager=echo scores it when "
            "meta_info.compute_follower_score is set (see ECHORewardManager)."
        )
        uid = phase_batch.non_tensor_batch["uid"]
        norm_by_std = bool(self.config.algorithm.get("norm_adv_by_std_in_grpo", True))
        follower_adv = echo_response.follower_group_advantage(follower_scores, uid, norm_by_std)

        responses = phase_batch.batch["responses"]
        response_length = responses.shape[-1]
        device = responses.device
        high = phase_batch.batch["high_level_loss_mask"][:, -response_length:]
        low = phase_batch.batch["low_level_loss_mask"][:, -response_length:]
        follower_adv = follower_adv.to(device)
        weights = echo_response.aho_token_weights(high, low, follower_adv, gamma=self._aho_gamma())
        phase_batch.batch["follower_scalar_advantages"] = follower_adv
        phase_batch.batch["aho_token_weights"] = weights

        with torch.no_grad():
            high_f = high.to(torch.float32)
            low_f = low.to(torch.float32)
            n_high = torch.clamp(high_f.sum(), min=1.0)
            metrics = {
                "high_level/aho/follower_adv_zero_frac": float((follower_adv.reshape(-1) == 0).float().mean().item()),
                "high_level/aho/no_tool_traj_frac": float((low_f.sum(dim=-1) == 0).float().mean().item()),
                "high_level/aho/weighted_reasoning_frac": float((((weights != 0) & (high_f > 0)).sum() / n_high).item()),
            }
            task_adv = phase_batch.batch.get("scalar_advantages", None)
            if task_adv is not None:
                product = task_adv.reshape(-1).to(torch.float32) * follower_adv.reshape(-1)
                metrics["high_level/aho/adv_product_mean"] = float(product.mean().item())
        return metrics

    def _next_batch_dict(self, phase_name: str):
        """Algorithm 1 draws distinct query groups for the follower and the leader
        (C.3/C.4), so every iteration pulls its own batch."""
        return self._pull_batch_dict(phase_name)

    def _open_round(self) -> None:
        """Algorithm 1 line 2: commit to x_t, clone y_0 = y_init, fresh optimizer state.

        y_init = 0, so the clone is implicit: snapshotting w_core + x_t is what makes
        the follower coordinate recoverable at line 14. See the block comment on
        AhoWorker.snapshot_leader_weights.
        """
        if not self._response_enabled():
            return
        self.actor_rollout_wg.snapshot_leader_weights()
        self.actor_rollout_wg.reset_follower_optimizer()

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
