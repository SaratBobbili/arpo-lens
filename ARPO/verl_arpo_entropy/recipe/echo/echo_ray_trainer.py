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
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, ResourcePoolManager, Role, RayPPOTrainer, _timer
from .echo_core_algos import agg_loss, apply_kl_penalty, compute_advantage, filter_informative_groups
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.metric import reduce_metrics

from verl.utils.reward_score.deep_research_echo import resolve_validator_profile


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

    @staticmethod
    def _copy_tool_metrics_between_phases(metrics: dict, source_phase: str, target_phase: str) -> None:
        source_prefix = f"{source_phase}/tools/"
        target_prefix = f"{target_phase}/tools/"
        for key, value in list(metrics.items()):
            if key.startswith(source_prefix):
                metrics[f"{target_prefix}{key[len(source_prefix):]}"] = value

    def _collect_logging_only_phase_metrics(
        self,
        source_batch: DataProto,
        target_phase_name: str,
        target_phase_mask_key: str,
    ) -> dict:
        """Compute phase metrics without actor/critic updates or extra rollouts."""
        phase_prefix = f"{target_phase_name}/"
        phase_reward_cfg = self._phase_reward_cfg(target_phase_name)
        phase_strategy = phase_reward_cfg.strategy
        phase_metrics: dict = {}

        phase_batch = deepcopy(source_batch)
        if phase_batch.meta_info is None:
            phase_batch.meta_info = {}
        phase_batch.meta_info["phase"] = target_phase_name
        phase_batch.batch["loss_mask"] = phase_batch.batch[target_phase_mask_key]
        phase_batch.batch["response_mask"] = phase_batch.batch[target_phase_mask_key]

        if phase_strategy in ("scorer", "maxentropy_rl"):
            _, reward_extra_infos_dict = compute_reward(phase_batch, self.reward_fn)
            phase_metrics.update(
                self._prefix_metrics(
                    self._build_scorer_metrics(reward_extra_infos_dict),
                    phase_prefix,
                )
            )

        if phase_strategy in ("entropy", "entropy-hybrid"):
            phase_batch.meta_info["calculate_entropy"] = True
            old_log_prob = self.actor_rollout_wg.compute_log_prob(phase_batch)
            entropys = old_log_prob.batch.get("entropys")
            if entropys is None:
                raise RuntimeError(
                    f"{target_phase_name} logging-only entropy metrics require entropys from compute_log_prob."
                )
            phase_mask_f = phase_batch.batch[target_phase_mask_key].to(torch.float32)
            entropy_mask_f = phase_mask_f
            if phase_strategy == "entropy-hybrid":
                entropy_mask_f = entropy_mask_f * phase_batch.batch["select_loss_mask"].to(torch.float32)
            entropy_mask_f = entropy_mask_f * phase_batch.batch["non_border_loss_mask"].to(torch.float32)
            _, entropy_metrics = self._build_entropy_scalar_reward(
                entropys=entropys,
                phase_batch=phase_batch,
                phase_mask_key=target_phase_mask_key,
                entropy_cfg=phase_reward_cfg.entropy,
                entropy_mask=entropy_mask_f,
            )
            phase_metrics.update(self._prefix_metrics(entropy_metrics, phase_prefix))

        return phase_metrics

    def _build_entropy_scalar_reward(self, entropys: torch.Tensor, phase_batch: DataProto, phase_mask_key: str, entropy_cfg, entropy_mask: torch.Tensor):
        """Sparse entropy reward shaped like the scorer's output.

        Reduces per-token entropy over `entropy_mask` tokens to one per-sample
        scalar (sum or mean per `entropy_cfg.reduction`), optionally
        normalized by log(vocab_size) and scaled/clamped, then writes that
        scalar at the last valid response token. When `format_gate` is on,
        calls the scorer and overrides bad-format samples with
        `bad_format_penalty` and LL soft-fail samples (flagged via
        `no_tool_calls`) with zero. GRPO's `sum(dim=-1)` recovers the same
        per-sample score as the scorer path, so no dense layout or post-hoc
        aggregation is needed.

        `entropy_mask` is precomputed by the caller (trainer) and is the
        single source of truth for which tokens enter the reduction. The
        caller is responsible for intersecting the phase mask with any
        strategy-specific masks (`select_loss_mask` for entropy-hybrid,
        `tool_loss_mask` for maxentropy_rl) and the global tag-border
        exclusion (`non_border_loss_mask`). The mean-reduction denominator
        uses `entropy_mask.sum(dim=-1)` so the per-sample scalar is the
        average entropy over exactly the restricted span. The sparse write
        position is the last valid response token (from `phase_mask_key`'s
        attention mask), not the last restricted token, so GRPO's
        `sum(dim=-1)` over the phase `response_mask` still sees the scalar.
        """
        ent = entropys.to(torch.float32)
        if bool(entropy_cfg.normalize):
            ent = ent / math.log(self.tokenizer.vocab_size)
        reduction = entropy_cfg.reduction
        assert reduction in ("sum", "mean"), f"entropy.reduction must be 'sum' or 'mean', got {reduction!r}."
        masked = ent * entropy_mask
        if reduction == "sum":
            per_sample = masked.sum(dim=-1)
        else:
            # clamp_min(1.0) guards empty masks without affecting the ratio when denom >= 1.
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
        # Use the *phase* mask (not entropy_mask) for the write template so the scalar
        # lands at the phase's last response token even when entropy_mask is empty
        # (e.g. all-border samples) or zero on that final token.
        phase_mask = phase_batch.batch[phase_mask_key].to(torch.float32)
        response_length = phase_mask.size(-1)
        resp_attn = phase_batch.batch["attention_mask"][:, -response_length:]
        last_idx = (resp_attn.sum(dim=-1).long() - 1).clamp_min(0)
        reward_tensor = torch.zeros_like(phase_mask)
        reward_tensor[torch.arange(reward_tensor.size(0), device=reward_tensor.device), last_idx] = per_sample
        return reward_tensor, metrics

    def _echo_rollout_tools_cfg(self):
        return self.config.actor_rollout_ref.rollout.tools

    def _apply_tool_failure_phase_masks(self, phase_batch: DataProto, phase_mask_key: str) -> None:
        if not bool(self._echo_rollout_tools_cfg().get("skip_training_on_tool_failure", False)):
            return
        flags = phase_batch.non_tensor_batch.get("tool_rollout_failed")
        if flags is None or not np.any(flags):
            return
        dev = phase_batch.batch[phase_mask_key].device
        keep = (~torch.tensor(flags.astype(np.bool_), device=dev)).float().unsqueeze(-1)
        phase_batch.batch[phase_mask_key] = phase_batch.batch[phase_mask_key] * keep

    def _apply_tool_failure_before_grpo(self, phase_batch: DataProto) -> None:
        if not bool(self._echo_rollout_tools_cfg().get("skip_training_on_tool_failure", False)):
            return
        flags = phase_batch.non_tensor_batch.get("tool_rollout_failed")
        if flags is None or not np.any(flags):
            return
        dev = phase_batch.batch["token_level_rewards"].device
        failed = torch.tensor(flags.astype(np.bool_), device=dev)
        phase_batch.batch["token_level_rewards"][failed] = 0
        phase_batch.batch["token_level_scores"][failed] = 0
        uids = phase_batch.non_tensor_batch["uid"].copy()
        for i in np.flatnonzero(flags):
            uids[i] = str(uuid.uuid4())
        phase_batch.non_tensor_batch["uid"] = uids

    def _validate_phase_reward_configs(self, phase_specs) -> None:
        for phase_name, _, _ in phase_specs:
            cfg = self._phase_reward_cfg(phase_name)
            algo = cfg.get("algorithm", "grpo")
            assert algo in ("grpo", "dapo"), f"{phase_name}.algorithm must be grpo or dapo, got {algo!r}"
            if algo == "dapo":
                assert bool(cfg.filter_groups.enable), f"{phase_name} algorithm=dapo requires filter_groups.enable=true"
                assert cfg.filter_groups.metric, f"{phase_name} algorithm=dapo requires filter_groups.metric"

    @staticmethod
    def _phase_algorithm(phase_reward_cfg) -> str:
        return phase_reward_cfg.get("algorithm", "grpo")

    def _next_batch_dict(self, data_iter):
        try:
            return next(data_iter), data_iter
        except StopIteration:
            data_iter = iter(self.train_dataloader)
            return next(data_iter), data_iter

    @staticmethod
    def _pop_gen_batch(batch: DataProto) -> DataProto:
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        return batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

    def _phase_rollout_to_scored_batch(
        self,
        gen_batch: DataProto,
        phase_name: str,
        phase_rollout_n: int,
        phase_mask_key: str,
        timing_raw: dict,
        metrics: dict,
    ) -> tuple[DataProto, dict]:
        from omegaconf import OmegaConf

        phase_prefix = f"{phase_name}/"
        phase_reward_cfg = self._phase_reward_cfg(phase_name)
        phase_strategy = phase_reward_cfg.strategy
        phase_reward_extra_infos_dict: dict = {}

        phase_batch = DataProto(
            batch=TensorDict({}, batch_size=gen_batch.batch.batch_size),
            non_tensor_batch=deepcopy(gen_batch.non_tensor_batch),
            meta_info=deepcopy(gen_batch.meta_info) if gen_batch.meta_info else {},
        )
        phase_batch.meta_info["phase"] = phase_name
        phase_batch.meta_info["validator_profile"] = self._validator_profile

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
                    phase_batch.batch["reward_baselines"] = torch.zeros(
                        phase_gen_batch.batch["input_ids"].size(0),
                        dtype=torch.float32,
                        device=phase_gen_batch.batch["input_ids"].device,
                    )

        phase_batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(phase_batch.batch))], dtype=object
        )
        phase_batch = phase_batch.repeat(repeat_times=phase_rollout_n, interleave=True)
        phase_batch = phase_batch.union(gen_batch_output)
        if phase_mask_key not in phase_batch.batch:
            raise KeyError(f"Missing '{phase_mask_key}' in rollout batch; ensure rollout.mode=sync_echo.")
        phase_batch.batch["loss_mask"] = phase_batch.batch[phase_mask_key]
        phase_batch.batch["response_mask"] = phase_batch.batch[phase_mask_key]

        if self.config.trainer.balance_batch:
            phase_balance_metrics = {}
            self._balance_batch(phase_batch, metrics=phase_balance_metrics)
            metrics.update(self._prefix_metrics(phase_balance_metrics, phase_prefix))

        self._apply_tool_failure_phase_masks(phase_batch, phase_mask_key)
        phase_batch.meta_info["global_token_num"] = torch.sum(phase_batch.batch["attention_mask"], dim=-1).tolist()

        reward_tensor = None
        future_reward = None
        entropy_reward_tensor = None
        with _timer(f"{phase_name}_reward", timing_raw):
            if phase_strategy in ("scorer", "maxentropy_rl"):
                if self.use_rm:
                    reward_tensor = self.rm_wg.compute_rm_score(phase_batch)
                    phase_batch = phase_batch.union(reward_tensor)
                if self.config.reward_model.launch_reward_fn_async:
                    future_reward = compute_reward_async.remote(phase_batch, self.config, self.tokenizer)
                else:
                    reward_tensor, phase_reward_extra_infos_dict = compute_reward(phase_batch, self.reward_fn)

        entropys = None
        with _timer(f"{phase_name}_old_log_prob", timing_raw):
            phase_batch.meta_info["calculate_entropy"] = phase_strategy in (
                "entropy",
                "entropy-hybrid",
                "maxentropy_rl",
            )
            old_log_prob = self.actor_rollout_wg.compute_log_prob(phase_batch)
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropys = old_log_prob.batch.pop("entropys", None)
            if entropys is not None:
                entropy_loss = agg_loss(
                    loss_mat=entropys, loss_mask=phase_batch.batch["loss_mask"], loss_agg_mode=loss_agg_mode
                )
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
            phase_mask_f = phase_batch.batch[phase_mask_key].to(torch.float32)
            entropy_mask_f = phase_mask_f
            if phase_strategy == "entropy-hybrid":
                entropy_mask_f = entropy_mask_f * phase_batch.batch["select_loss_mask"].to(torch.float32)
            entropy_mask_f = entropy_mask_f * phase_batch.batch["non_border_loss_mask"].to(torch.float32)
            phase_batch.batch[f"{phase_name}_token_entropy"] = entropys.to(torch.float32) * entropy_mask_f
            phase_batch.batch["entropy_reg_loss_mask"] = entropy_mask_f
            reward_tensor, entropy_metrics = self._build_entropy_scalar_reward(
                entropys=entropys,
                phase_batch=phase_batch,
                phase_mask_key=phase_mask_key,
                entropy_cfg=phase_reward_cfg.entropy,
                entropy_mask=entropy_mask_f,
            )
            metrics.update(self._prefix_metrics(entropy_metrics, phase_prefix))
        elif phase_strategy == "maxentropy_rl":
            if entropys is None:
                raise RuntimeError(f"{phase_name} phase uses maxentropy_rl reward but compute_log_prob did not return entropys.")
            me_cfg = phase_reward_cfg.max_entropy
            phase_mask_f = phase_batch.batch[phase_mask_key].to(torch.float32)
            entropy_mask_f = phase_mask_f * phase_batch.batch[me_cfg.mask_key].to(torch.float32)
            entropy_mask_f = entropy_mask_f * phase_batch.batch["non_border_loss_mask"].to(torch.float32)
            phase_batch.batch[f"{phase_name}_token_entropy"] = entropys.to(torch.float32) * entropy_mask_f
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
                entropy_mask=entropy_mask_f,
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

        if phase_strategy in ("scorer", "maxentropy_rl") and self.config.reward_model.launch_reward_fn_async:
            reward_tensor, phase_reward_extra_infos_dict = ray.get(future_reward)
        if reward_tensor is None:
            raise RuntimeError(f"{phase_name} reward_tensor was not initialized.")
        if phase_strategy == "maxentropy_rl":
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

        return phase_batch, phase_reward_extra_infos_dict

    def _collect_phase_batch_dapo(self, data_iter, phase_name, phase_rollout_n, phase_mask_key, timing_raw, metrics):
        phase_prefix = f"{phase_name}/"
        phase_reward_cfg = self._phase_reward_cfg(phase_name)
        metric_name = phase_reward_cfg.filter_groups.metric
        max_num_gen_batches = int(phase_reward_cfg.filter_groups.max_num_gen_batches)
        prompt_bsz = self.config.data.train_batch_size

        accumulator = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        phase_reward_extra: dict = {}

        while num_prompt_in_batch < prompt_bsz:
            batch_dict, data_iter = self._next_batch_dict(data_iter)
            batch = DataProto.from_single_dict(batch_dict)
            gen_batch = self._pop_gen_batch(batch)
            num_gen_batches += 1
            new_batch, phase_reward_extra = self._phase_rollout_to_scored_batch(
                gen_batch, phase_name, phase_rollout_n, phase_mask_key, timing_raw, metrics
            )
            new_batch, num_kept = filter_informative_groups(new_batch, metric_name)
            num_prompt_in_batch += num_kept
            accumulator = new_batch if accumulator is None else DataProto.concat([accumulator, new_batch])

            if num_prompt_in_batch < prompt_bsz:
                print(f"{phase_name} {num_prompt_in_batch=} < {prompt_bsz=}")
                if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                    print(f"{phase_name} {num_gen_batches=}. Keep generating...")
                    continue
                raise ValueError(
                    f"{phase_name} {num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. "
                    "Check data difficulty or set max_num_gen_batches=0 for no upper limit."
                )

        traj_bsz = prompt_bsz * phase_rollout_n
        phase_batch = accumulator[:traj_bsz]
        metrics[f"{phase_prefix}training/filter_groups_kept_prompts"] = num_prompt_in_batch
        metrics[f"{phase_prefix}training/num_gen_batches"] = num_gen_batches
        return phase_batch, phase_reward_extra, data_iter

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
            data_iter = iter(self.train_dataloader)
            while self.global_steps <= self.total_training_steps:
                metrics = {}
                timing_raw = {}
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
                    self._validate_phase_reward_configs(phase_specs)

                    step_gen_batch = None
                    if any(self._phase_algorithm(self._phase_reward_cfg(n)) == "grpo" for n, _, _ in phase_specs):
                        batch_dict, data_iter = self._next_batch_dict(data_iter)
                        step_gen_batch = self._pop_gen_batch(DataProto.from_single_dict(batch_dict))

                    norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                    last_phase_batch = None

                    for phase_name, phase_rollout_n, phase_mask_key in phase_specs:
                        phase_prefix = f"{phase_name}/"
                        phase_reward_cfg = self._phase_reward_cfg(phase_name)
                        phase_strategy = phase_reward_cfg.strategy

                        if self._phase_algorithm(phase_reward_cfg) == "dapo":
                            phase_batch, phase_reward_extra_infos_dict, data_iter = self._collect_phase_batch_dapo(
                                data_iter, phase_name, phase_rollout_n, phase_mask_key, timing_raw, metrics
                            )
                        else:
                            phase_batch, phase_reward_extra_infos_dict = self._phase_rollout_to_scored_batch(
                                step_gen_batch, phase_name, phase_rollout_n, phase_mask_key, timing_raw, metrics
                            )
                            metrics[f"{phase_prefix}training/num_gen_batches"] = 1

                        with _timer(f"{phase_name}_adv", timing_raw):
                            self._apply_tool_failure_before_grpo(phase_batch)

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
                                    # Mirror the reward-channel `normalize` flag onto the
                                    # regularizer so a single yaml knob sets the units of H
                                    # for both channels. When true, dp_actor divides per-token
                                    # H by log(vocab_size) before agg_loss, so reg_coeff acts
                                    # on H ∈ [0,1] (matching the reward path). When false, the
                                    # regularizer keeps raw nats (~log(vocab_size) ≈ 12x scale).
                                    if bool(phase_reward_cfg.entropy.get("normalize", False)):
                                        phase_batch.meta_info["entropy_loss_normalizer"] = math.log(self.tokenizer.vocab_size)
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

                    # When one phase consumes the whole rollout budget, keep logging
                    # for the zero-budget phase by re-scoring the same rollouts
                    # under that phase's reward semantics. No extra rollouts or
                    # actor/critic updates are run for the zero-budget phase.
                    if bool(self.config.trainer.get("log_zero_budget_phase_metrics", True)) and len(phase_specs) == 1:
                        active_phase_name = phase_specs[0][0]
                        for skipped_phase_name, (skipped_budget, skipped_mask_key) in phase_registry.items():
                            if skipped_budget != 0:
                                continue
                            metrics.update(
                                self._collect_logging_only_phase_metrics(
                                    source_batch=last_phase_batch,
                                    target_phase_name=skipped_phase_name,
                                    target_phase_mask_key=skipped_mask_key,
                                )
                            )
                            self._copy_tool_metrics_between_phases(
                                metrics=metrics,
                                source_phase=active_phase_name,
                                target_phase=skipped_phase_name,
                            )

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

