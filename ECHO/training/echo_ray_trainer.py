"""HYPERGRADIENT trainer: ECHO Algorithm 1 on top of ALTERNATING-GRPO.

Everything here exists for the response term. The shared two-phase GRPO loop lives
in RayAlternatingGRPOTrainer (alt_ray_trainer.py); this class fills its hooks.
"""
import os
import shutil

from pprint import pprint

import numpy as np
import torch

from verl.utils.metric import reduce_metrics

from . import echo_response
from .alt_ray_trainer import RayAlternatingGRPOTrainer


class RayECHOTrainer(RayAlternatingGRPOTrainer):
    """ALTERNATING-GRPO plus the Algorithm-1 round structure and the response term."""

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

    def _response_estimator(self) -> str:
        """Which estimator computes g_resp: ``adjoint`` (Algorithm 1's K-step reverse sweep,
        the default) or ``aho`` (arXiv:2607.28849's Hessian-free Boltzmann sensitivity, GRPO
        variant -- see echo_response's module docstring)."""
        cfg = self._response_cfg()
        estimator = str(cfg.get("estimator", "adjoint")).lower() if cfg is not None else "adjoint"
        assert estimator in ("adjoint", "aho"), (
            f"phases.response.estimator must be 'adjoint' or 'aho', got {estimator!r}"
        )
        return estimator

    def _response_aho_enabled(self) -> bool:
        return self._response_gradient_enabled() and self._response_estimator() == "aho"

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
        assert self._follower_return_enabled(), (
            "estimator=aho contracts A_L, the follower's advantage on R_L, against A_H. With "
            "phases.response.follower_return off the follower optimised R_H and A_L is not "
            "the advantage of the objective whose optimum the Boltzmann identity describes."
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

    def _follower_return_enabled(self) -> bool:
        """Whether the follower phase is scored by R_L (Eq. 2) rather than the task return.

        v10 Eq. (32): the first surrogate adapts the tool policy with the TOOL-level return,
        while both leader surrogates use the task return. This is split out from
        `phases.response.enabled` because that one flag also gates the round structure, the
        follower discard and the response term -- four changes in a single switch, which
        makes any A/B against a plain-GRPO run uninterpretable. Null inherits `enabled`, so
        existing launch configs behave exactly as before.
        """
        cfg = self._response_cfg()
        if cfg is None:
            return False
        value = cfg.get("follower_return", None)
        if value is None:
            return self._response_enabled()
        return bool(value)

    def _validate_response_config(self) -> None:
        """Algorithm 1 takes one update per fresh group; the code must too.

        C.3: "if a group is reused, each clipped optimizer epoch and its fixed behavior
        snapshot is a separate substep of the adaptation map." With several mini-batches
        per iteration the true adaptation map would be K x n_minibatch substeps, and the
        stashed-record-to-substep correspondence the reverse sweep relies on would break.
        Requiring one step per iteration keeps K = phases.low_level.num_iters exactly.
        """
        if self._response_gradient_enabled() and not self._follower_return_enabled():
            # Eq. (33)'s g_grp_L is built from the follower's OWN advantages. Scoring the
            # follower phase by the task return instead makes the replayed surrogate a
            # different object from the update it is supposed to stand for.
            print(
                "[echo] WARNING: phases.response.gradient is on but phases.response."
                "follower_return is off, so the response term contracts against task-return "
                "advantages rather than R_L. This is not Eq. (33); the run is measuring "
                "something else."
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
        if self._response_estimator() == "aho":
            # Asserts exact/curvature/group_aligned off, so the adjoint-only checks below
            # (SGD follower, zero weight decay) are skipped: nothing is differentiated
            # through the optimizer here.
            self._validate_aho_config()
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

    # --- RayAlternatingGRPOTrainer hooks -----------------------------------------

    def _validate_extra(self) -> None:
        self._validate_response_config()

    def _rollout_meta_info(self, phase_name: str) -> dict:
        """Algorithm 1 scores u_L by r_valid (Eq. 2) and u_H by r_task (Eq. 3). Off, both
        phases share the task score, which is the pre-Algorithm-1 behavior."""
        return {
            "use_follower_return": self._follower_return_enabled(),
            # AHO: the leader batch is scored by R_L too (alongside the task return).
            "compute_follower_score": self._response_aho_enabled() and phase_name == "high_level",
        }

    def _scorer_metric_kwargs(self, phase_name: str) -> dict:
        return {"follower_return": self._follower_return_enabled() and phase_name == "low_level"}

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
            meta["response_estimator"] = self._response_estimator()
            if self._response_aho_enabled():
                meta["aho_tau"] = self._aho_tau()
                meta["aho_gamma"] = self._aho_gamma()
        return meta

    def _after_advantage(self, phase_batch, phase_name: str) -> dict:
        """AHO estimator, leader phase: put A_L and the per-token weights on the batch.

        Runs after _apply_tool_failure_before_grpo and compute_advantage, so an excised
        sample already carries its fresh uid (a singleton group, hence A_L = 0) and a
        demoted one already has R_L = 0 through the format gate. Everything the actor's
        surrogate needs is then a plain batch tensor that survives micro-batching.
        """
        if not (self._response_aho_enabled() and phase_name == "high_level"):
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
                "policy/aho_follower_adv_zero_frac": float((follower_adv.reshape(-1) == 0).float().mean().item()),
                "policy/aho_no_tool_traj_frac": float((low_f.sum(dim=-1) == 0).float().mean().item()),
                "policy/aho_weighted_reasoning_frac": float((((weights != 0) & (high_f > 0)).sum() / n_high).item()),
            }
            task_adv = phase_batch.batch.get("scalar_advantages", None)
            if task_adv is not None:
                product = task_adv.reshape(-1).to(torch.float32) * follower_adv.reshape(-1)
                metrics["policy/aho_adv_product_mean"] = float(product.mean().item())
        return metrics

    def _next_batch_dict(self, phase_name: str):
        """Algorithm 1 draws distinct query groups for the follower and the leader
        (C.3/C.4), so every iteration pulls its own batch."""
        return self._pull_batch_dict(phase_name)

    def _open_round(self) -> None:
        """Algorithm 1 line 2: commit to x_t, clone y_0 = y_init, fresh optimizer state.

        y_init = 0, so the clone is implicit: snapshotting w_core + x_t is what makes
        the follower coordinate recoverable at line 14. See the block comment on
        EchoActorRolloutRefWorker.snapshot_leader_weights.
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

    def _after_fit(self, num_ll_iters: int, logger, progress_bar) -> None:
        self._run_final_response(num_ll_iters, logger, progress_bar)
