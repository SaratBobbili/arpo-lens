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

import json
import os
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

from verl.utils.reward_score.deep_research_echo import resolve_validator_profile

from .echo_core_algos import agg_loss


class RayECHOTrainer(RayPPOTrainer):
    """ECHO trainer with ARPO-identical PPO training loop."""

    # Per-phase JSONL dump spec consumed by `_dump_logging_data`. Each entry is
    # (filename_under_logging_data/<phase>/, metric_suffix appended to "<phase>/"
    # to look up in the per-step `metrics` dict). LL "reward" reads the pre-gate
    # entropy mean over good-format ∧ has-tool only (clean entropy axis); HL
    # "reward" reads f1_mean (clean task axis, ignores -1 by construction).
    # `entropy_reg_loss` is only populated for entropy/entropy-hybrid phases, so
    # it is intentionally absent from the high_level spec.
    _LOGGING_SPEC = {
        "low_level": [
            ("reward.jsonl", "reward/entropy_scalar_mean_good"),
            ("format_penalty.jsonl", "reward/bad_format_rate"),
            ("pg_loss.jsonl", "actor/pg_loss"),
            ("entropy_reg_loss.jsonl", "actor/entropy_reg_loss"),
            ("grad_norm.jsonl", "actor/grad_norm"),
            ("entropy_old_policy.jsonl", "actor/entropy_old_policy"),
            ("high_level_valid_rate.jsonl", "reward/high_level_valid_rate"),
            ("low_level_valid_rate.jsonl", "reward/low_level_valid_rate"),
            ("tools_total_calls.jsonl", "tools/total_calls"),
            ("tools_successful_calls.jsonl", "tools/successful_calls"),
        ],
        "high_level": [
            ("reward.jsonl", "reward/f1_mean"),
            ("format_penalty.jsonl", "reward/bad_format_rate"),
            ("pg_loss.jsonl", "actor/pg_loss"),
            ("grad_norm.jsonl", "actor/grad_norm"),
            ("entropy_old_policy.jsonl", "actor/entropy_old_policy"),
            ("high_level_valid_rate.jsonl", "reward/high_level_valid_rate"),
            ("low_level_valid_rate.jsonl", "reward/low_level_valid_rate"),
            ("tools_total_calls.jsonl", "tools/total_calls"),
            ("tools_successful_calls.jsonl", "tools/successful_calls"),
        ],
    }

    @staticmethod
    def _prefix_metrics(metrics_dict: dict, prefix: str) -> dict:
        return {f"{prefix}{key}": value for key, value in metrics_dict.items()}

    def _phase_reward_cfg(self, phase_name: str):
        # Per-phase reward config block (strategy + strategy-specific params).
        return self.config.reward_model.phase_rewards[phase_name]

    def _init_logging_data(self) -> None:
        # One JSONL per (phase, metric) under {default_local_dir}/logging_data/.
        # Read by recipe/echo/plot_training_log.py for offline per-step plots.
        self._logging_data_root = os.path.join(self.config.trainer.default_local_dir, "logging_data")
        self._prev_logged_values: dict[str, float] = {}
        for phase in self._LOGGING_SPEC:
            os.makedirs(os.path.join(self._logging_data_root, phase), exist_ok=True)

    def _dump_logging_data(self, metrics: dict) -> None:
        # Append one line per metric file: {"step", "value", "gain"}. `gain` is
        # the difference vs the previous dumped step for the same key, or null
        # on the first dump. Keys absent from `metrics` (e.g. entropy_reg_loss
        # on a scorer phase) are skipped without erroring.
        for phase, specs in self._LOGGING_SPEC.items():
            phase_dir = os.path.join(self._logging_data_root, phase)
            for filename, metric_suffix in specs:
                full_key = f"{phase}/{metric_suffix}"
                if full_key not in metrics:
                    continue
                value = float(metrics[full_key])
                prev = self._prev_logged_values.get(full_key)
                gain = None if prev is None else value - prev
                self._prev_logged_values[full_key] = value
                with open(os.path.join(phase_dir, filename), "a") as f:
                    f.write(json.dumps({"step": self.global_steps, "value": value, "gain": gain}) + "\n")

    @staticmethod
    def _build_scorer_metrics(reward_extra_info: dict) -> dict:
        metrics: dict = {}
        if not reward_extra_info:
            return metrics

        if "score" in reward_extra_info:
            scores = torch.tensor(reward_extra_info["score"], dtype=torch.float32)
            metrics["reward/score_mean"] = scores.mean().item()
            metrics["reward/bad_format_rate"] = (scores < 0.0).to(torch.float32).mean().item()
            metrics["reward/format_pass_rate"] = (scores >= 0.0).to(torch.float32).mean().item()

        if "f1_score" in reward_extra_info:
            f1_scores = torch.tensor(reward_extra_info["f1_score"], dtype=torch.float32)
            metrics["reward/f1_mean"] = f1_scores.mean().item()

        if "no_tool_calls" in reward_extra_info:
            no_tool = torch.tensor(reward_extra_info["no_tool_calls"], dtype=torch.float32)
            metrics["reward/no_tool_rate"] = no_tool.mean().item()

        if "high_level_valid" in reward_extra_info:
            hl_valid = torch.tensor(reward_extra_info["high_level_valid"], dtype=torch.float32)
            metrics["reward/high_level_valid_rate"] = hl_valid.mean().item()

        if "low_level_valid" in reward_extra_info:
            ll_valid = torch.tensor(reward_extra_info["low_level_valid"], dtype=torch.float32)
            metrics["reward/low_level_valid_rate"] = ll_valid.mean().item()

        return metrics

    def _build_entropy_scalar_reward(self, entropys: torch.Tensor, phase_batch: DataProto, phase_mask_key: str, entropy_cfg, extra_mask_key: str | None = None):
        """Sparse entropy reward shaped like the scorer's output.

        Reduces per-token entropy over phase-mask tokens to one per-sample
        scalar (sum or mean per `entropy_cfg.reduction`), optionally
        normalized by log(vocab_size) and scaled/clamped, then writes that
        scalar at the last valid response token. When `format_gate` is on,
        calls the scorer and overrides bad-format samples with
        `bad_format_penalty` and LL soft-fail samples (flagged via
        `no_tool_calls`) with zero. GRPO's `sum(dim=-1)` recovers the same
        per-sample score as the scorer path, so no dense layout or post-hoc
        aggregation is needed.

        When `extra_mask_key` is provided (e.g. `entropy-hybrid` passes
        `"select_loss_mask"`), the reduction is further restricted to the
        intersection `phase_mask * extra_mask`, and the mean-reduction
        denominator uses the intersected count so the per-sample scalar is
        the average entropy over the restricted span. The sparse write
        position is still the last valid response token (from the phase's
        attention mask), not the last restricted token, so GRPO's
        `sum(dim=-1)` over the phase `response_mask` still sees the scalar.
        """
        # Phase-local (optionally restricted) entropy reduced per sample to a single scalar.
        phase_mask = phase_batch.batch[phase_mask_key].to(torch.float32)
        if extra_mask_key is not None:
            entropy_mask = phase_mask * phase_batch.batch[extra_mask_key].to(torch.float32)
        else:
            entropy_mask = phase_mask
        ent = entropys.to(torch.float32)
        if bool(entropy_cfg.normalize):
            ent = ent / math.log(self.tokenizer.vocab_size)
        reduction = entropy_cfg.reduction
        assert reduction in ("sum", "mean"), f"entropy.reduction must be 'sum' or 'mean', got {reduction!r}."
        masked = ent * entropy_mask
        if reduction == "sum":
            per_sample = masked.sum(dim=-1)
        else:
            # clamp_min(1.0) guards empty (phase ∩ extra) masks without affecting the ratio when denom >= 1.
            denom = entropy_mask.sum(dim=-1).clamp_min(1.0)
            per_sample = masked.sum(dim=-1) / denom
        per_sample = per_sample * float(entropy_cfg.scale)
        if entropy_cfg.clamp_min is not None or entropy_cfg.clamp_max is not None:
            per_sample = torch.clamp(per_sample, min=entropy_cfg.clamp_min, max=entropy_cfg.clamp_max)

        metrics: dict = {}
        if bool(entropy_cfg.get("format_gate", False)):
            # Scorer emits -1 at the last valid response token iff the phase-local
            # validator fails, and flags `no_tool_calls` via reward_extra_info for
            # the LL soft-fail (valid format, no <search>/<python> invoked).
            scorer_tensor, reward_extra = compute_reward(phase_batch, self.reward_fn)
            scorer_per_sample = scorer_tensor.to(per_sample.device).sum(dim=-1)
            bad = scorer_per_sample < 0.0
            no_tool_flags = reward_extra.get("no_tool_calls", [False] * per_sample.size(0))
            no_tool = torch.tensor(no_tool_flags, dtype=torch.bool, device=per_sample.device)
            metrics["reward/bad_format_rate"] = bad.float().mean().item()
            metrics["reward/no_tool_rate"] = no_tool.float().mean().item()
            # Pre-gate mean over good-format ∧ has-tool samples only. Isolates the
            # entropy-reward axis from the format-penalty / no-tool axes so the
            # downstream JSONL trace has a clean per-step reward signal that
            # doesn't drift just because the bad-format share moves.
            keep = (~bad) & (~no_tool)
            metrics["reward/entropy_scalar_mean_good"] = (
                per_sample[keep].mean().detach().item() if keep.any() else 0.0
            )
            # Hoist HL/LL validator pass rates from the same scorer extras so
            # the entropy path exposes the same validity diagnostics that
            # `_build_scorer_metrics` emits for scorer phases.
            if "high_level_valid" in reward_extra:
                metrics["reward/high_level_valid_rate"] = (
                    torch.tensor(reward_extra["high_level_valid"], dtype=torch.float32).mean().item()
                )
            if "low_level_valid" in reward_extra:
                metrics["reward/low_level_valid_rate"] = (
                    torch.tensor(reward_extra["low_level_valid"], dtype=torch.float32).mean().item()
                )
            penalty = float(entropy_cfg.bad_format_penalty)
            per_sample = torch.where(bad, torch.full_like(per_sample, penalty), per_sample)
            # `no_tool_calls` is only set by the scorer when phase_valid is True,
            # so it's already mutually exclusive with `bad`; the where is safe.
            per_sample = torch.where(no_tool, torch.zeros_like(per_sample), per_sample)

        metrics["reward/entropy_scalar_mean"] = per_sample.mean().detach().item()

        # Sparse write at the last valid response token, matching ECHORewardManager's placement.
        response_length = phase_mask.size(-1)
        resp_attn = phase_batch.batch["attention_mask"][:, -response_length:]
        last_idx = (resp_attn.sum(dim=-1).long() - 1).clamp_min(0)
        reward_tensor = torch.zeros_like(phase_mask)
        reward_tensor[torch.arange(reward_tensor.size(0), device=reward_tensor.device), last_idx] = per_sample
        return reward_tensor, metrics

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

        # Resolve validator profile once from the rollout mask_categories so the
        # format validator's HL/LL routing matches the phase mask layout.
        # Fails fast if mask_categories does not match any supported profile.
        self._validator_profile = resolve_validator_profile(
            self.config.actor_rollout_ref.rollout.mask_categories
        )

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

                    # Phase metadata keyed by phase name so `phase_order` from config
                    # selects which phase's GRPO pipeline runs first. Each entry is
                    # (rollout_budget, loss_mask_key); phases with zero budget are
                    # skipped while preserving the requested order.
                    phase_registry = {
                        "high_level": (high_level_budget, "high_level_loss_mask"),
                        "low_level": (low_level_budget, "low_level_loss_mask"),
                    }
                    phase_order = list(self.config.reward_model.phase_order)
                    assert set(phase_order) == set(phase_registry.keys()), (
                        f"reward_model.phase_order must be a permutation of {sorted(phase_registry)}, got {phase_order}."
                    )
                    phase_specs = [
                        (name, phase_registry[name][0], phase_registry[name][1])
                        for name in phase_order
                        if phase_registry[name][0] > 0
                    ]
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
                        # Validator profile (derived from mask_categories at trainer init)
                        # routes per-check HL/LL attribution inside compute_score.
                        phase_batch.meta_info["validator_profile"] = self._validator_profile
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
                            entropy_reward_tensor = None
                            if phase_strategy in ("scorer", "maxentropy_rl"):
                                if self.use_rm:
                                    reward_tensor = self.rm_wg.compute_rm_score(phase_batch)
                                    phase_batch = phase_batch.union(reward_tensor)

                                if self.config.reward_model.launch_reward_fn_async:
                                    future_reward = compute_reward_async.remote(phase_batch, self.config, self.tokenizer)
                                else:
                                    reward_tensor, phase_reward_extra_infos_dict = compute_reward(phase_batch, self.reward_fn)

                        with _timer(f"{phase_name}_old_log_prob", timing_raw):
                            phase_batch.meta_info["calculate_entropy"] = phase_strategy in ("entropy", "entropy-hybrid", "maxentropy_rl")
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(phase_batch)
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropys = old_log_prob.batch.pop("entropys", None)
                            if entropys is not None:
                                # Diagnostic: old-policy entropy aggregated over the GRPO loss
                                # mask. Distinct from the differentiable `actor/entropy_reg_loss`
                                # logged by `update_policy` (current-policy entropy reduced over
                                # m^phase, the term that actually enters the gradient).
                                entropy_loss = agg_loss(loss_mat=entropys, loss_mask=phase_batch.batch["loss_mask"], loss_agg_mode=loss_agg_mode)
                                metrics[f"{phase_prefix}actor/entropy_old_policy"] = entropy_loss.detach().item()
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

                        if phase_strategy in ("entropy", "entropy-hybrid"):
                            if entropys is None:
                                raise RuntimeError(f"{phase_name} phase uses entropy reward but compute_log_prob did not return entropys.")
                            # `entropy-hybrid` restricts entropy to tokens inside <select>...</select>
                            # (emitted by rollout as `select_loss_mask`), intersected with the phase
                            # mask. Users who want to exclude the initial <select> block from this
                            # computation can move `first_select` to the other phase via
                            # `mask_categories`; the intersection will drop it automatically.
                            extra_mask_key = "select_loss_mask" if phase_strategy == "entropy-hybrid" else None
                            phase_mask_f = phase_batch.batch[phase_mask_key].to(torch.float32)
                            entropy_mask_f = (
                                phase_mask_f * phase_batch.batch[extra_mask_key].to(torch.float32)
                                if extra_mask_key is not None
                                else phase_mask_f
                            )
                            # Diagnostic mirrors the mask actually used for the reduction.
                            phase_batch.batch[f"{phase_name}_token_entropy"] = entropys.to(torch.float32) * entropy_mask_f
                            # Persist the same m^phase used for the entropy *reward* as the mask
                            # consumed by the entropy *regularizer* in update_policy. Identity
                            # of masks across the two channels is intentional: both channels are
                            # gated on the same scoring strategy, so they share m^phase.
                            phase_batch.batch["entropy_reg_loss_mask"] = entropy_mask_f
                            reward_tensor, entropy_metrics = self._build_entropy_scalar_reward(
                                entropys=entropys,
                                phase_batch=phase_batch,
                                phase_mask_key=phase_mask_key,
                                entropy_cfg=phase_reward_cfg.entropy,
                                extra_mask_key=extra_mask_key,
                            )
                            metrics.update(self._prefix_metrics(entropy_metrics, phase_prefix))
                        elif phase_strategy == "maxentropy_rl":
                            if entropys is None:
                                raise RuntimeError(f"{phase_name} phase uses maxentropy_rl reward but compute_log_prob did not return entropys.")
                            me_cfg = phase_reward_cfg.max_entropy
                            # Reduction support set: phase mask intersected with the rollout-emitted
                            # tool-portion mask (default `tool_loss_mask` -> first_select+select+search+python).
                            extra_mask_key = me_cfg.mask_key
                            phase_mask_f = phase_batch.batch[phase_mask_key].to(torch.float32)
                            entropy_mask_f = phase_mask_f * phase_batch.batch[extra_mask_key].to(torch.float32)
                            phase_batch.batch[f"{phase_name}_token_entropy"] = entropys.to(torch.float32) * entropy_mask_f
                            # Reuse `_build_entropy_scalar_reward` by mapping `alpha` -> `scale`
                            # and disabling the format gate; the scorer's -1 on bad format is
                            # preserved unchanged in `reward_tensor` (combined later in `_adv`),
                            # so the entropy term is added unconditionally.
                            me_entropy_cfg = OmegaConf.create({
                                "reduction": me_cfg.reduction,
                                "normalize": me_cfg.normalize,
                                "scale": me_cfg.alpha,
                                "clamp_min": None,
                                "clamp_max": None,
                                "format_gate": False,
                            })
                            entropy_reward_tensor, entropy_metrics = self._build_entropy_scalar_reward(
                                entropys=entropys,
                                phase_batch=phase_batch,
                                phase_mask_key=phase_mask_key,
                                entropy_cfg=me_entropy_cfg,
                                extra_mask_key=extra_mask_key,
                            )
                            metrics.update(self._prefix_metrics(entropy_metrics, phase_prefix))

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
                            if phase_strategy in ("scorer", "maxentropy_rl") and self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, phase_reward_extra_infos_dict = ray.get(future_reward)
                            if reward_tensor is None:
                                raise RuntimeError(f"{phase_name} reward_tensor was not initialized.")
                            if phase_strategy == "maxentropy_rl":
                                # r_i = scorer_i + alpha * ent_i; both terms are sparse scalars at
                                # the last valid response token, so addition stays sparse and
                                # GRPO's sum(dim=-1) recovers r_i directly.
                                reward_tensor = reward_tensor + entropy_reward_tensor.to(reward_tensor.device)
                            phase_batch.batch["token_level_scores"] = reward_tensor
                            if phase_reward_extra_infos_dict:
                                phase_batch.non_tensor_batch.update({k: np.array(v) for k, v in phase_reward_extra_infos_dict.items()})
                                if phase_strategy in ("scorer", "maxentropy_rl"):
                                    metrics.update(
                                        self._prefix_metrics(
                                            self._build_scorer_metrics(phase_reward_extra_infos_dict),
                                            phase_prefix,
                                        )
                                    )

                            if self.config.algorithm.use_kl_in_reward:
                                phase_batch, kl_metrics = apply_kl_penalty(
                                    phase_batch,
                                    kl_ctrl=self.kl_ctrl_in_reward,
                                    kl_penalty=self.config.algorithm.kl_penalty,
                                )
                                metrics.update(self._prefix_metrics(kl_metrics, phase_prefix))
                            else:
                                phase_batch.batch["token_level_rewards"] = phase_batch.batch["token_level_scores"]

                            # Both strategies emit a sparse scalar at the last valid response
                            # token, so GRPO's sum(dim=-1) directly recovers the per-sample
                            # score with no further aggregation needed here.
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
                            metrics.update(
                                self._prefix_metrics(
                                    compute_data_metrics(batch=phase_batch, use_critic=self.use_critic),
                                    phase_prefix,
                                )
                            )

                        if self.use_critic:
                            with _timer(f"{phase_name}_update_critic", timing_raw):
                                critic_output = self.critic_wg.update_critic(phase_batch)
                            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                            metrics.update(self._prefix_metrics(critic_output_metrics, phase_prefix))

                        if self.config.trainer.critic_warmup <= self.global_steps:
                            with _timer(f"{phase_name}_update_actor", timing_raw):
                                phase_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                # Direct entropy regularizer is gated on the phase's scoring
                                # strategy: only enabled when the phase uses an entropy-based
                                # reward (entropy / entropy-hybrid), so the regularizer auto-
                                # disables for `scorer` and `maxentropy_rl` (where entropy
                                # already enters via the reward channel). When enabled, the
                                # mask m^phase is `entropy_reg_loss_mask`, populated above to
                                # mirror the reward-side intersection (phase_mask, optionally
                                # ∩ select_loss_mask for entropy-hybrid).
                                if phase_strategy in ("entropy", "entropy-hybrid"):
                                    phase_batch.meta_info["entropy_coeff_override"] = float(phase_reward_cfg.entropy.get("reg_coeff", 0.0))
                                    phase_batch.meta_info["entropy_loss_mask_key"] = "entropy_reg_loss_mask"
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
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)
                self._dump_logging_data(metrics)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

