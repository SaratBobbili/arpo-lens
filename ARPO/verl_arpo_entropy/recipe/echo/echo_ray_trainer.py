# Copyright 2026 ECHO contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
ECHO trainer entrypoint.

This module keeps ECHO behavior identical to ARPO PPO while hosting recipe-local
trainer code that can diverge later.
"""

import uuid
from copy import deepcopy
import math
from pprint import pprint

import numpy as np
import ray
import torch
from tqdm import tqdm

from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, ResourcePoolManager, Role, RayPPOTrainer, _timer, apply_kl_penalty, compute_advantage
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.metric import reduce_metrics

from .echo_core_algos import agg_loss


class RayECHOTrainer(RayPPOTrainer):
    """ECHO trainer with ARPO-identical PPO training loop."""

    @staticmethod
    def _prefix_metrics(metrics_dict: dict, prefix: str) -> dict:
        return {f"{prefix}{key}": value for key, value in metrics_dict.items()}

    def _phase_reward_cfg(self, phase_name: str):
        # Per-phase reward config block (strategy + strategy-specific params).
        return self.config.reward_model.phase_rewards[phase_name]

    def _build_entropy_reward(self, entropys: torch.Tensor, phase_mask: torch.Tensor, entropy_cfg) -> torch.Tensor:
        scale = float(entropy_cfg.scale)
        entropy_reward = entropys.to(torch.float32)
        if bool(entropy_cfg.normalize):
            vocab_size = self.tokenizer.vocab_size
            entropy_reward = entropy_reward / math.log(vocab_size)
        entropy_reward = entropy_reward * scale
        if entropy_cfg.clamp_min is not None or entropy_cfg.clamp_max is not None:
            entropy_reward = torch.clamp(entropy_reward, min=entropy_cfg.clamp_min, max=entropy_cfg.clamp_max)
        return entropy_reward * phase_mask.to(torch.float32)

    def _apply_format_gate(self, phase_batch: DataProto, reward_tensor: torch.Tensor, penalty: float):
        # Bad-format rollouts: dense penalty over the phase mask so the
        # post-aggregation per-sample scalar is `penalty` under
        # score_aggregation=mean and `penalty * denom` under
        # score_aggregation=sum.
        # Soft-fail (valid format, no tool call): dense reward zeroed with no
        # terminal penalty.
        # Format validity comes from the scorer: ECHORewardManager emits the
        # scorer scalar at the last valid response token (zeros elsewhere) and
        # deep_research_echo.compute_score emits exactly -1 iff the response
        # fails format / parse checks, so summing over tokens recovers the
        # per-sample verdict without decoding again. The scorer also emits a
        # per-sample `no_tool_calls` flag via reward_extra_info for the LL
        # soft-fail case (valid format but no <search>/<python> invocation).
        scorer_tensor, reward_extra = compute_reward(phase_batch, self.reward_fn)
        per_sample_score = scorer_tensor.to(reward_tensor.device).sum(dim=-1)
        bad = per_sample_score < 0.0
        bad_rate = bad.float().mean().item()

        no_tool_flags = reward_extra.get("no_tool_calls", [False] * reward_tensor.size(0))
        no_tool = torch.tensor(no_tool_flags, dtype=torch.bool, device=reward_tensor.device)
        no_tool_rate = no_tool.float().mean().item()

        gated = reward_tensor.clone()
        # Soft-fail: zero the dense entropy reward so no-tool rollouts contribute
        # no LL gradient from their step-<select> tokens.
        gated[no_tool] = 0.0
        # Hard-fail: broadcast `penalty` across every phase-mask token of each
        # bad-format row. response_mask is 0/1 and already restricted to
        # phase-local positions upstream, so tokens outside the phase are left
        # at 0 and the write implicitly overrides the dense entropy reward on
        # those rows.
        phase_mask = phase_batch.batch["response_mask"].to(gated.dtype)
        gated[bad] = penalty * phase_mask[bad]
        return gated, bad_rate, no_tool_rate

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

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

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    total_rollout_budget = int(self.config.actor_rollout_ref.rollout.n)
                    high_level_budget = int(self.config.actor_rollout_ref.rollout.get("high_level_budget", total_rollout_budget))
                    assert 0 <= high_level_budget <= total_rollout_budget, (
                        f"Invalid high_level_budget={high_level_budget}. "
                        f"Must satisfy 0 <= high_level_budget <= rollout.n({total_rollout_budget})."
                    )
                    low_level_budget = total_rollout_budget - high_level_budget
                    metrics["training/high_level_rollout_budget"] = high_level_budget
                    metrics["training/low_level_rollout_budget"] = low_level_budget

                    phase_specs = []
                    if high_level_budget > 0:
                        phase_specs.append(("high_level", high_level_budget, "high_level_loss_mask"))
                    if low_level_budget > 0:
                        phase_specs.append(("low_level", low_level_budget, "low_level_loss_mask"))
                    assert phase_specs, "At least one hierarchical phase must have positive rollout budget."

                    norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                    last_phase_batch = None

                    for phase_name, phase_rollout_n, phase_mask_key in phase_specs:
                        phase_prefix = f"{phase_name}/"
                        phase_reward_cfg = self._phase_reward_cfg(phase_name)
                        phase_strategy = phase_reward_cfg.strategy
                        # batch.batch is an empty TensorDict (all tensor keys were popped
                        # into gen_batch). deepcopy would call consolidate() on that empty
                        # TensorDict, which crashes. Build a fresh DataProto instead.
                        phase_batch = DataProto(
                            batch=TensorDict({}, batch_size=batch.batch.batch_size),
                            non_tensor_batch=deepcopy(batch.non_tensor_batch),
                            meta_info=deepcopy(batch.meta_info),
                        )
                        # Phase tag consumed by ECHORewardManager -> deep_research_echo.compute_score
                        # to gate the -1 format verdict on the phase-local validator only.
                        phase_batch.meta_info["phase"] = phase_name
                        phase_reward_extra_infos_dict = {}

                        with _timer(f"{phase_name}_gen", timing_raw):
                            phase_gen_batch = deepcopy(gen_batch)
                            if phase_gen_batch.meta_info is None:
                                phase_gen_batch.meta_info = {}
                            phase_gen_batch.meta_info["rollout_n_override"] = phase_rollout_n
                            if not self.async_rollout_mode:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(phase_gen_batch)
                            else:
                                self.async_rollout_manager.wake_up()
                                gen_batch_output = self.async_rollout_manager.generate_sequences(phase_gen_batch)
                                self.async_rollout_manager.sleep()

                            if gen_batch_output.meta_info and "metrics" in gen_batch_output.meta_info:
                                metrics.update(self._prefix_metrics(gen_batch_output.meta_info["metrics"], phase_prefix))

                        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                            with _timer(f"{phase_name}_gen_max", timing_raw):
                                if phase_strategy == "scorer":
                                    gen_baseline_batch = deepcopy(phase_gen_batch)
                                    gen_baseline_batch.meta_info["do_sample"] = False
                                    gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                                    phase_batch = phase_batch.union(gen_baseline_output)
                                    reward_baseline_tensor = self.reward_fn(phase_batch).sum(dim=-1)
                                    phase_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                                    phase_batch.batch["reward_baselines"] = reward_baseline_tensor
                                else:
                                    # Entropy-strategy rewards do not use the scorer-based REMAX baseline.
                                    phase_batch.batch["reward_baselines"] = torch.zeros(
                                        phase_gen_batch.batch["input_ids"].size(0),
                                        dtype=torch.float32,
                                        device=phase_gen_batch.batch["input_ids"].device,
                                    )

                        phase_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(phase_batch.batch))], dtype=object)
                        phase_batch = phase_batch.repeat(repeat_times=phase_rollout_n, interleave=True)
                        phase_batch = phase_batch.union(gen_batch_output)
                        if phase_mask_key not in phase_batch.batch:
                            raise KeyError(f"Missing '{phase_mask_key}' in rollout batch; ensure rollout.mode=sync_echo.")
                        phase_batch.batch["loss_mask"] = phase_batch.batch[phase_mask_key]
                        # Restrict response_mask to phase-local tokens. GRPO's
                        # compute_advantage broadcasts scores.unsqueeze(-1) * response_mask,
                        # so this isolates the non-zero advantage support to this phase's
                        # tokens. Actor PPO aggregation is independently driven by
                        # loss_mask (set above) which the dp_actor picks up whenever
                        # loss_mask is present on the batch.
                        phase_batch.batch["response_mask"] = phase_batch.batch[phase_mask_key]

                        if self.config.trainer.balance_batch:
                            phase_balance_metrics = {}
                            self._balance_batch(phase_batch, metrics=phase_balance_metrics)
                            metrics.update(self._prefix_metrics(phase_balance_metrics, phase_prefix))

                        phase_batch.meta_info["global_token_num"] = torch.sum(phase_batch.batch["attention_mask"], dim=-1).tolist()

                        with _timer(f"{phase_name}_reward", timing_raw):
                            reward_tensor = None
                            future_reward = None
                            if phase_strategy == "scorer":
                                if self.use_rm:
                                    reward_tensor = self.rm_wg.compute_rm_score(phase_batch)
                                    phase_batch = phase_batch.union(reward_tensor)

                                if self.config.reward_model.launch_reward_fn_async:
                                    future_reward = compute_reward_async.remote(phase_batch, self.config, self.tokenizer)
                                else:
                                    reward_tensor, phase_reward_extra_infos_dict = compute_reward(phase_batch, self.reward_fn)

                        with _timer(f"{phase_name}_old_log_prob", timing_raw):
                            phase_batch.meta_info["calculate_entropy"] = phase_strategy == "entropy"
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(phase_batch)
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropys = old_log_prob.batch.pop("entropys", None)
                            if entropys is not None:
                                entropy_loss = agg_loss(loss_mat=entropys, loss_mask=phase_batch.batch["loss_mask"], loss_agg_mode=loss_agg_mode)
                                metrics[f"{phase_prefix}actor/entropy_loss"] = entropy_loss.detach().item()
                            phase_batch = phase_batch.union(old_log_prob)

                            if "rollout_log_probs" in phase_batch.batch.keys():
                                rollout_old_log_probs = phase_batch.batch["rollout_log_probs"]
                                actor_old_log_probs = phase_batch.batch["old_log_probs"]
                                attention_mask = phase_batch.batch["attention_mask"]
                                responses = phase_batch.batch["responses"]
                                response_length = responses.size(1)
                                response_mask = attention_mask[:, -response_length:]

                                rollout_probs = torch.exp(rollout_old_log_probs)
                                actor_probs = torch.exp(actor_old_log_probs)
                                rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                                rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                                metrics[f"{phase_prefix}training/rollout_probs_diff_max"] = torch.max(rollout_probs_diff).detach().item()
                                metrics[f"{phase_prefix}training/rollout_probs_diff_mean"] = torch.mean(rollout_probs_diff).detach().item()
                                metrics[f"{phase_prefix}training/rollout_probs_diff_std"] = torch.std(rollout_probs_diff).detach().item()

                        if phase_strategy == "entropy":
                            if entropys is None:
                                raise RuntimeError(f"{phase_name} phase uses entropy reward but compute_log_prob did not return entropys.")
                            phase_mask = phase_batch.batch[phase_mask_key]
                            phase_batch.batch[f"{phase_name}_token_entropy"] = entropys.to(torch.float32) * phase_mask.to(torch.float32)
                            reward_tensor = self._build_entropy_reward(
                                entropys=entropys,
                                phase_mask=phase_mask,
                                entropy_cfg=phase_reward_cfg.entropy,
                            )
                            if bool(phase_reward_cfg.entropy.get("format_gate", False)):
                                reward_tensor, bad_format_rate, no_tool_rate = self._apply_format_gate(
                                    phase_batch=phase_batch,
                                    reward_tensor=reward_tensor,
                                    penalty=float(phase_reward_cfg.entropy.bad_format_penalty),
                                )
                                metrics[f"{phase_prefix}reward/bad_format_rate"] = bad_format_rate
                                metrics[f"{phase_prefix}reward/no_tool_rate"] = no_tool_rate
                            metrics[f"{phase_prefix}reward/entropy_reward_mean"] = agg_loss(
                                loss_mat=reward_tensor,
                                loss_mask=phase_batch.batch["loss_mask"],
                                loss_agg_mode=loss_agg_mode,
                            ).detach().item()

                        if self.use_reference_policy:
                            with _timer(f"{phase_name}_ref", timing_raw):
                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(phase_batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(phase_batch)
                                phase_batch = phase_batch.union(ref_log_prob)

                        if self.use_critic:
                            with _timer(f"{phase_name}_values", timing_raw):
                                values = self.critic_wg.compute_values(phase_batch)
                                phase_batch = phase_batch.union(values)

                        with _timer(f"{phase_name}_adv", timing_raw):
                            if phase_strategy == "scorer" and self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, phase_reward_extra_infos_dict = ray.get(future_reward)
                            if reward_tensor is None:
                                raise RuntimeError(f"{phase_name} reward_tensor was not initialized.")
                            phase_batch.batch["token_level_scores"] = reward_tensor
                            if phase_reward_extra_infos_dict:
                                phase_batch.non_tensor_batch.update({k: np.array(v) for k, v in phase_reward_extra_infos_dict.items()})

                            if self.config.algorithm.use_kl_in_reward:
                                phase_batch, kl_metrics = apply_kl_penalty(
                                    phase_batch,
                                    kl_ctrl=self.kl_ctrl_in_reward,
                                    kl_penalty=self.config.algorithm.kl_penalty,
                                )
                                metrics.update(self._prefix_metrics(kl_metrics, phase_prefix))
                            else:
                                phase_batch.batch["token_level_rewards"] = phase_batch.batch["token_level_scores"]

                            # GRPO reduces token_level_rewards to a per-sample scalar via
                            # sum(dim=-1). For "mean" aggregation we pre-divide rewards by
                            # the per-sample phase-mask token count so the same sum recovers
                            # a mask-aware mean, avoiding any edits to the shared GRPO
                            # implementation. Rewards are already zero outside the phase
                            # mask (scorer: sparse last-token scalar; entropy: masked in
                            # _build_entropy_reward), so scaling by the mask denom is safe.
                            score_aggregation = phase_reward_cfg.score_aggregation
                            assert score_aggregation in ("sum", "mean"), (
                                f"reward_model.phase_rewards.{phase_name}.score_aggregation must be "
                                f"'sum' or 'mean', got {score_aggregation!r}."
                            )
                            if score_aggregation == "mean":
                                tlr = phase_batch.batch["token_level_rewards"]
                                denom = phase_batch.batch["response_mask"].to(tlr.dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
                                phase_batch.batch["token_level_rewards"] = tlr / denom

                            phase_batch = compute_advantage(
                                phase_batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=phase_rollout_n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                                use_pf_ppo=self.config.algorithm.use_pf_ppo,
                                pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                                pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            )

                        if self.use_critic:
                            with _timer(f"{phase_name}_update_critic", timing_raw):
                                critic_output = self.critic_wg.update_critic(phase_batch)
                            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                            metrics.update(self._prefix_metrics(critic_output_metrics, phase_prefix))

                        if self.config.trainer.critic_warmup <= self.global_steps:
                            with _timer(f"{phase_name}_update_actor", timing_raw):
                                phase_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                actor_output = self.actor_rollout_wg.update_actor(phase_batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(self._prefix_metrics(actor_output_metrics, phase_prefix))

                        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                        if rollout_data_dir:
                            with _timer(f"{phase_name}_dump_rollout_generations", timing_raw):
                                inputs = self.tokenizer.batch_decode(phase_batch.batch["prompts"], skip_special_tokens=True)
                                outputs = self.tokenizer.batch_decode(phase_batch.batch["responses"], skip_special_tokens=True)
                                scores = phase_batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                                self._dump_generations(
                                    inputs=inputs,
                                    outputs=outputs,
                                    scores=scores,
                                    reward_extra_infos_dict=phase_reward_extra_infos_dict,
                                    dump_path=rollout_data_dir,
                                )

                        last_phase_batch = phase_batch

                    batch = last_phase_batch

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

