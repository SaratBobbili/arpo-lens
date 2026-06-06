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
import shutil
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

    # Per-phase JSONL dump spec consumed by `_dump_logging_data`.
    _LOGGING_SPEC = {
        "low_level": [
            ("reward.jsonl", "reward/effective_reward_mean"),
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
            ("reward.jsonl", "reward/effective_reward_mean"),
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
    }

    @staticmethod
    def _prefix_metrics(metrics_dict: dict, prefix: str) -> dict:
        return {f"{prefix}{key}": value for key, value in metrics_dict.items()}

    def _phase_reward_cfg(self, phase_name: str):
        # Per-phase reward config block (strategy + strategy-specific params).
        return self.config.reward_model.phase_rewards[phase_name]

    @staticmethod
    def _entropy_reg_coeff(phase_reward_cfg) -> float:
        return float(phase_reward_cfg.entropy.get("reg_coeff", 0.0))

    @staticmethod
    def _uses_entropy_regularizer(phase_strategy: str, phase_reward_cfg) -> bool:
        if phase_strategy in ("entropy", "entropy-hybrid"):
            return True
        if phase_strategy == "scorer":
            return RayECHOTrainer._entropy_reg_coeff(phase_reward_cfg) > 0.0
        return False

    @staticmethod
    def _needs_policy_entropy(phase_strategy: str, phase_reward_cfg) -> bool:
        if phase_strategy in ("entropy", "entropy-hybrid", "maxentropy_rl"):
            return True
        if phase_strategy == "scorer":
            return RayECHOTrainer._entropy_reg_coeff(phase_reward_cfg) > 0.0
        return False

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
            metrics["reward/effective_reward_mean"] = scores.mean().item()
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

        if phase_strategy == "scorer":
            _, reward_extra_infos_dict = compute_reward(phase_batch, self.reward_fn)
            phase_metrics.update(
                self._prefix_metrics(
                    self._build_scorer_metrics(reward_extra_infos_dict),
                    phase_prefix,
                )
            )
        elif phase_strategy == "maxentropy_rl":
            from omegaconf import OmegaConf

            scorer_tensor, reward_extra_infos_dict = compute_reward(phase_batch, self.reward_fn)
            phase_metrics.update(
                self._prefix_metrics(
                    self._build_scorer_metrics(reward_extra_infos_dict),
                    phase_prefix,
                )
            )
            phase_batch.meta_info["calculate_entropy"] = True
            old_log_prob = self.actor_rollout_wg.compute_log_prob(phase_batch)
            entropys = old_log_prob.batch.get("entropys")
            if entropys is None:
                raise RuntimeError(
                    f"{target_phase_name} logging-only maxentropy_rl metrics require entropys from compute_log_prob."
                )
            me_cfg = phase_reward_cfg.max_entropy
            phase_mask_f = phase_batch.batch[target_phase_mask_key].to(torch.float32)
            entropy_mask_f = phase_mask_f * phase_batch.batch[me_cfg.mask_key].to(torch.float32)
            entropy_mask_f = entropy_mask_f * phase_batch.batch["non_border_loss_mask"].to(torch.float32)
            me_entropy_cfg = OmegaConf.create(
                {
                    "reduction": me_cfg.reduction,
                    "normalize": me_cfg.normalize,
                    "scale": me_cfg.alpha,
                    "clamp_min": None,
                    "clamp_max": None,
                    "bad_format_penalty": 0,
                    "no_tool_penalty": None,
                }
            )
            entropy_reward_tensor, entropy_metrics = self._build_entropy_scalar_reward(
                entropys=entropys,
                phase_batch=phase_batch,
                phase_mask_key=target_phase_mask_key,
                entropy_cfg=me_entropy_cfg,
                reward_entropy_mask=entropy_mask_f,
            )
            phase_metrics.update(self._prefix_metrics(entropy_metrics, phase_prefix))
            combined_tensor = scorer_tensor + entropy_reward_tensor.to(scorer_tensor.device)
            phase_metrics[f"{phase_prefix}reward/effective_reward_mean"] = (
                combined_tensor.to(torch.float32).sum(dim=-1).mean().detach().item()
            )
        elif phase_strategy in ("entropy", "entropy-hybrid"):
            phase_batch.meta_info["calculate_entropy"] = True
            old_log_prob = self.actor_rollout_wg.compute_log_prob(phase_batch)
            entropys = old_log_prob.batch.get("entropys")
            if entropys is None:
                raise RuntimeError(
                    f"{target_phase_name} logging-only entropy metrics require entropys from compute_log_prob."
                )
            _, entropy_metrics = self._build_entropy_scalar_reward(
                entropys=entropys,
                phase_batch=phase_batch,
                phase_mask_key=target_phase_mask_key,
                entropy_cfg=phase_reward_cfg.entropy,
            )
            phase_metrics.update(self._prefix_metrics(entropy_metrics, phase_prefix))

        return phase_metrics

    def _reduce_masked_entropy(self, entropys: torch.Tensor, entropy_mask: torch.Tensor, entropy_cfg) -> torch.Tensor:
        ent = entropys.to(torch.float32)
        if bool(entropy_cfg.normalize):
            ent = ent / math.log(self.tokenizer.vocab_size)
        reduction = entropy_cfg.reduction
        assert reduction in ("sum", "mean"), f"entropy.reduction must be 'sum' or 'mean', got {reduction!r}."
        masked = ent * entropy_mask
        if reduction == "sum":
            return masked.sum(dim=-1)
        denom = entropy_mask.sum(dim=-1).clamp_min(1.0)
        return masked.sum(dim=-1) / denom

    def _apply_entropy_band_score(
        self,
        h_bar: torch.Tensor,
        h_init: torch.Tensor,
        entropy_cfg,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        band_cfg = entropy_cfg.band
        eps_fallback = float(band_cfg.get("epsilon", 0.2))
        eps_low = float(band_cfg.get("epsilon_low", eps_fallback))
        eps_high = float(band_cfg.get("epsilon_high", eps_fallback))
        h_floor = float(band_cfg.get("h_floor", 1e-4))
        h_init = h_init.clamp_min(h_floor)
        h_low = (1.0 - eps_low) * h_init
        h_high = (1.0 + eps_high) * h_init
        score = torch.full_like(h_bar, scale)
        below = h_bar < h_low
        above = h_bar > h_high
        score = torch.where(below, scale * (h_bar / h_low), score)
        score = torch.where(above, scale * (h_high / h_bar.clamp_min(h_floor)), score)
        return score, h_init, h_low, h_high

    def _build_entropy_scalar_reward(
        self,
        entropys: torch.Tensor,
        phase_batch: DataProto,
        phase_mask_key: str,
        entropy_cfg,
        reward_entropy_mask: torch.Tensor | None = None,
    ):
        """Sparse entropy reward shaped like the scorer's output.

        Reduces per-token entropy to one per-sample scalar (sum or mean per
        `entropy_cfg.reduction`), optionally normalized by log(vocab_size).
        For `entropy` / `entropy-hybrid`, H_bar and H_init always use
        m^phase ∩ non_border_loss_mask (strategy masks apply only to the actor
        regularizer). `maxentropy_rl` may pass `reward_entropy_mask` for its
        additive entropy leg. With `entropy.band.enable`, global steps
        1..warmup_steps use scale*H_bar (no band) and capture mean phase
        entropy as frozen H_init; later steps band against that frozen anchor.
        """
        phase_entropy_mask = (
            phase_batch.batch[phase_mask_key].to(torch.float32)
            * phase_batch.batch["non_border_loss_mask"].to(torch.float32)
        )
        if reward_entropy_mask is not None:
            h_bar = self._reduce_masked_entropy(entropys, reward_entropy_mask, entropy_cfg)
        else:
            h_bar = self._reduce_masked_entropy(entropys, phase_entropy_mask, entropy_cfg)
        scale = float(entropy_cfg.scale)
        band_cfg = entropy_cfg.get("band")
        band_enable = bool(band_cfg.get("enable", False)) if band_cfg is not None else False
        band_warmup_active = False
        if band_enable:
            assert reward_entropy_mask is None, "entropy.band uses phase-level H_bar only"
            warmup_steps = int(band_cfg.get("warmup_steps", 1))
            if self.global_steps <= warmup_steps:
                per_sample = h_bar * scale
                band_warmup_active = True
                self._frozen_h_init_ref[phase_mask_key] = h_bar.mean().item()
            else:
                frozen = self._frozen_h_init_ref[phase_mask_key]
                assert frozen is not None, (
                    f"entropy.band missing frozen H_init for {phase_mask_key} at global_step={self.global_steps}"
                )
                h_init = torch.full_like(h_bar, frozen)
                per_sample, h_init, h_low, h_high = self._apply_entropy_band_score(
                    h_bar, h_init, entropy_cfg, scale
                )
        else:
            per_sample = h_bar * scale
            if entropy_cfg.clamp_min is not None or entropy_cfg.clamp_max is not None:
                per_sample = torch.clamp(per_sample, min=entropy_cfg.clamp_min, max=entropy_cfg.clamp_max)

        metrics: dict = {}
        metrics["reward/entropy_h_bar_mean"] = h_bar.mean().detach().item()
        if band_enable:
            metrics["reward/entropy_h_phase_mean"] = h_bar.mean().detach().item()
            metrics["reward/entropy_h_init_frozen_mean"] = self._frozen_h_init_ref[phase_mask_key]
            metrics["reward/entropy_band_warmup_active"] = float(band_warmup_active)
            if not band_warmup_active:
                in_band = (h_bar >= h_low) & (h_bar <= h_high)
                metrics["reward/entropy_h_init_mean"] = h_init.mean().detach().item()
                metrics["reward/entropy_h_low_mean"] = h_low.mean().detach().item()
                metrics["reward/entropy_h_high_mean"] = h_high.mean().detach().item()
                metrics["reward/entropy_in_band_rate"] = in_band.float().mean().detach().item()
        bad_format_penalty = float(entropy_cfg.get("bad_format_penalty", 0))
        no_tool_penalty_cfg = entropy_cfg.get("no_tool_penalty", None)
        no_tool_penalty_active = no_tool_penalty_cfg is not None
        reward_override_active = bad_format_penalty != 0 or no_tool_penalty_active
        entropy_scalar_mean_good = per_sample.mean().detach().item()
        if reward_override_active:
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
            entropy_scalar_mean_good = per_sample[keep].mean().detach().item() if keep.any() else 0.0
            if "high_level_valid" in reward_extra:
                metrics["reward/high_level_valid_rate"] = (
                    torch.tensor(reward_extra["high_level_valid"], dtype=torch.float32).mean().item()
                )
            if "low_level_valid" in reward_extra:
                metrics["reward/low_level_valid_rate"] = (
                    torch.tensor(reward_extra["low_level_valid"], dtype=torch.float32).mean().item()
                )
            if bad_format_penalty != 0:
                per_sample = torch.where(
                    bad, torch.full_like(per_sample, bad_format_penalty), per_sample
                )
            if no_tool_penalty_active:
                no_tool_penalty = float(no_tool_penalty_cfg)
                per_sample = torch.where(
                    no_tool, torch.full_like(per_sample, no_tool_penalty), per_sample
                )

        metrics["reward/entropy_scalar_mean_good"] = entropy_scalar_mean_good
        metrics["reward/entropy_scalar_mean"] = per_sample.mean().detach().item()
        metrics["reward/effective_reward_mean"] = per_sample.mean().detach().item()

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
            band = cfg.entropy.get("band")
            if band is not None and bool(band.get("enable", False)):
                warmup_steps = int(band.get("warmup_steps", 1))
                assert warmup_steps >= 1, (
                    f"{phase_name} entropy.band.enable requires warmup_steps >= 1, got {warmup_steps}"
                )

    @staticmethod
    def _phase_algorithm(phase_reward_cfg) -> str:
        return phase_reward_cfg.get("algorithm", "grpo")

    @staticmethod
    def _canonical_best_metric_selector(selector: str) -> str:
        normalized = str(selector).strip().lower().replace("_", "-")
        if normalized in ("val-core/reward", "reward", "val-core_reward"):
            return "val-core/reward"
        if normalized in ("val-aux/f1-score", "f1-score", "f1", "val-aux/f1_score"):
            return "val-aux/f1-score"
        raise ValueError(
            "trainer.best_checkpoint_metric must be one of {'val-core/reward', 'val-aux/f1-score'}."
        )

    def _resolve_best_metric_from_val(self, val_metrics: dict) -> tuple[str, float]:
        selector = self._canonical_best_metric_selector(self.config.trainer.best_checkpoint_metric)
        if selector == "val-core/reward":
            matched = [(k, v) for k, v in val_metrics.items() if k.startswith("val-core/") and "/reward/" in k]
        else:
            matched = [(k, v) for k, v in val_metrics.items() if k.startswith("val-aux/") and "/f1_score/" in k]
        if not matched:
            raise ValueError(
                f"Could not resolve metric '{selector}' from validation metrics keys: {list(val_metrics.keys())[:20]}"
            )
        metric_key, metric_value = max(matched, key=lambda kv: float(kv[1]))
        return metric_key, float(metric_value)

    def _sync_best_checkpoint_dir(self) -> None:
        src_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        dst_dir = os.path.join(self.config.trainer.default_local_dir, "best_checkpoint")
        if os.path.lexists(dst_dir):
            if os.path.islink(dst_dir) or os.path.isfile(dst_dir):
                os.unlink(dst_dir)
            else:
                shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        best_txt = os.path.join(self.config.trainer.default_local_dir, "best_checkpoint.txt")
        with open(best_txt, "w") as f:
            f.write(str(self.global_steps))

    def _phase_update_repeats(self) -> dict[str, int]:
        repeats_cfg = self.config.reward_model.get("phase_update_repeats", {})
        phase_names = ("high_level", "low_level")
        assert set(repeats_cfg.keys()) == set(phase_names), (
            f"reward_model.phase_update_repeats must have keys {phase_names}, got {list(repeats_cfg.keys())}."
        )
        repeats = {name: int(repeats_cfg[name]) for name in phase_names}
        for name, repeat in repeats.items():
            assert repeat > 0, f"reward_model.phase_update_repeats.{name} must be > 0, got {repeat}."
        return repeats

    def _expand_phase_specs_with_repeats(self, phase_specs) -> list[tuple[str, int, str]]:
        repeats = self._phase_update_repeats()
        expanded = []
        for phase_name, phase_rollout_n, phase_mask_key in phase_specs:
            expanded.extend([(phase_name, phase_rollout_n, phase_mask_key)] * repeats[phase_name])
        return expanded

    def _next_batch_dict(self, data_iter):
        try:
            return next(data_iter), data_iter
        except StopIteration:
            data_iter = iter(self.train_dataloader)
            return next(data_iter), data_iter

    @staticmethod
    def _pop_gen_batch(batch: DataProto) -> tuple[DataProto, DataProto]:
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
        return gen_batch, batch

    def _phase_rollout_to_scored_batch(
        self,
        gen_batch: DataProto,
        prompt_batch: DataProto,
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
            non_tensor_batch=deepcopy(prompt_batch.non_tensor_batch),
            meta_info=deepcopy(prompt_batch.meta_info) if prompt_batch.meta_info else {},
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
            phase_batch.meta_info["calculate_entropy"] = self._needs_policy_entropy(
                phase_strategy, phase_reward_cfg
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
            reg_entropy_mask_f = phase_mask_f * phase_batch.batch["non_border_loss_mask"].to(torch.float32)
            if phase_strategy == "entropy-hybrid":
                reg_entropy_mask_f = reg_entropy_mask_f * phase_batch.batch["select_loss_mask"].to(torch.float32)
            phase_batch.batch[f"{phase_name}_token_entropy"] = entropys.to(torch.float32) * reg_entropy_mask_f
            phase_batch.batch["entropy_reg_loss_mask"] = reg_entropy_mask_f
            reward_tensor, entropy_metrics = self._build_entropy_scalar_reward(
                entropys=entropys,
                phase_batch=phase_batch,
                phase_mask_key=phase_mask_key,
                entropy_cfg=phase_reward_cfg.entropy,
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
                "bad_format_penalty": 0,
                "no_tool_penalty": None,
            })
            entropy_reward_tensor, entropy_metrics = self._build_entropy_scalar_reward(
                entropys=entropys,
                phase_batch=phase_batch,
                phase_mask_key=phase_mask_key,
                entropy_cfg=me_entropy_cfg,
                reward_entropy_mask=entropy_mask_f,
            )
            metrics.update(self._prefix_metrics(entropy_metrics, phase_prefix))
        elif phase_strategy == "scorer" and self._entropy_reg_coeff(phase_reward_cfg) > 0.0:
            if entropys is None:
                raise RuntimeError(
                    f"{phase_name} phase uses scorer with entropy.reg_coeff > 0 but compute_log_prob did not return entropys."
                )
            phase_mask_f = phase_batch.batch[phase_mask_key].to(torch.float32)
            entropy_mask_f = phase_mask_f * phase_batch.batch["non_border_loss_mask"].to(torch.float32)
            phase_batch.batch["entropy_reg_loss_mask"] = entropy_mask_f

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
        metrics[f"{phase_prefix}reward/effective_reward_mean"] = (
            reward_tensor.to(torch.float32).sum(dim=-1).mean().detach().item()
        )
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
            gen_batch, prompt_batch = self._pop_gen_batch(batch)
            num_gen_batches += 1
            new_batch, phase_reward_extra = self._phase_rollout_to_scored_batch(
                gen_batch, prompt_batch, phase_name, phase_rollout_n, phase_mask_key, timing_raw, metrics
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
        self._frozen_h_init_ref: dict[str, float] = {}
        self._best_metric_value = float("-inf")
        self._best_metric_step = -1
        self._best_metric_key = None

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
                total_rollout_budget = int(self.config.actor_rollout_ref.rollout.n)
                high_level_budget = int(self.config.actor_rollout_ref.rollout.get("high_level_budget", total_rollout_budget))
                assert 0 <= high_level_budget <= total_rollout_budget, (
                    f"Invalid high_level_budget={high_level_budget}. "
                    f"Must satisfy 0 <= high_level_budget <= rollout.n({total_rollout_budget})."
                )
                low_level_budget = total_rollout_budget - high_level_budget

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
                repeated_phase_specs = self._expand_phase_specs_with_repeats(phase_specs)

                norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                for phase_name, phase_rollout_n, phase_mask_key in repeated_phase_specs:
                    metrics = {
                        "training/high_level_rollout_budget": high_level_budget,
                        "training/low_level_rollout_budget": low_level_budget,
                    }
                    timing_raw = {}
                    is_last_step = self.global_steps >= self.total_training_steps
                    saved_checkpoint_this_step = False

                    with _timer("step", timing_raw):
                        phase_prefix = f"{phase_name}/"
                        phase_reward_cfg = self._phase_reward_cfg(phase_name)
                        phase_strategy = phase_reward_cfg.strategy

                        if self._phase_algorithm(phase_reward_cfg) == "dapo":
                            phase_batch, phase_reward_extra_infos_dict, data_iter = self._collect_phase_batch_dapo(
                                data_iter, phase_name, phase_rollout_n, phase_mask_key, timing_raw, metrics
                            )
                        else:
                            batch_dict, data_iter = self._next_batch_dict(data_iter)
                            step_gen_batch, step_prompt_batch = self._pop_gen_batch(DataProto.from_single_dict(batch_dict))
                            phase_batch, phase_reward_extra_infos_dict = self._phase_rollout_to_scored_batch(
                                step_gen_batch, step_prompt_batch, phase_name, phase_rollout_n, phase_mask_key, timing_raw, metrics
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
                                phase_batch.meta_info["kl_loss_coef_override"] = float(
                                    phase_reward_cfg.get(
                                        "kl_loss_coef",
                                        self.config.actor_rollout_ref.actor.kl_loss_coef,
                                    )
                                )
                                if self._uses_entropy_regularizer(phase_strategy, phase_reward_cfg):
                                    phase_batch.meta_info["entropy_coeff_override"] = self._entropy_reg_coeff(phase_reward_cfg)
                                    phase_batch.meta_info["entropy_loss_mask_key"] = "entropy_reg_loss_mask"
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

                        if bool(self.config.trainer.get("log_zero_budget_phase_metrics", True)) and len(phase_specs) == 1:
                            active_phase_name = phase_specs[0][0]
                            for skipped_phase_name, (skipped_budget, skipped_mask_key) in phase_registry.items():
                                if skipped_budget != 0:
                                    continue
                                metrics.update(
                                    self._collect_logging_only_phase_metrics(
                                        source_batch=phase_batch,
                                        target_phase_name=skipped_phase_name,
                                        target_phase_mask_key=skipped_mask_key,
                                    )
                                )
                                self._copy_tool_metrics_between_phases(
                                    metrics=metrics,
                                    source_phase=active_phase_name,
                                    target_phase=skipped_phase_name,
                                )

                        if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                            with _timer("testing", timing_raw):
                                val_metrics: dict = self._validate()
                                if is_last_step:
                                    last_val_metrics = val_metrics
                            metrics.update(val_metrics)

                            if bool(self.config.trainer.get("save_best_checkpoint", False)):
                                current_metric_key, current_metric_value = self._resolve_best_metric_from_val(val_metrics)
                                selector = self._canonical_best_metric_selector(self.config.trainer.best_checkpoint_metric)
                                metrics["training/best_checkpoint_metric_selector_id"] = (
                                    0.0 if selector == "val-core/reward" else 1.0
                                )
                                metrics["training/best_checkpoint_metric_current"] = current_metric_value
                                metrics["training/best_checkpoint_metric_best"] = self._best_metric_value
                                metrics["training/best_checkpoint_metric_improved"] = float(
                                    current_metric_value > self._best_metric_value
                                )
                                if current_metric_value > self._best_metric_value:
                                    self._best_metric_value = current_metric_value
                                    self._best_metric_step = self.global_steps
                                    self._best_metric_key = current_metric_key
                                    if not saved_checkpoint_this_step:
                                        with _timer("save_checkpoint", timing_raw):
                                            self._save_checkpoint()
                                        saved_checkpoint_this_step = True
                                    with _timer("save_best_checkpoint", timing_raw):
                                        self._sync_best_checkpoint_dir()
                                    print(
                                        f"[best_checkpoint] updated: selector={selector}, "
                                        f"resolved_key={current_metric_key}, value={current_metric_value}, step={self.global_steps}"
                                    )
                                metrics["training/best_checkpoint_metric_best"] = self._best_metric_value
                                metrics["training/best_checkpoint_metric_best_step"] = float(self._best_metric_step)

                        if (
                            not saved_checkpoint_this_step
                            and self.config.trainer.save_freq > 0
                            and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0)
                        ):
                            with _timer("save_checkpoint", timing_raw):
                                self._save_checkpoint()

                    metrics.update(
                        {
                            "training/global_step": self.global_steps,
                            "training/epoch": epoch,
                        }
                    )
                    metrics.update(compute_timing_metrics(batch=phase_batch, timing_raw=timing_raw))
                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    metrics.update(compute_throughout_metrics(batch=phase_batch, timing_raw=timing_raw, n_gpus=n_gpus))

                    logger.log(data=metrics, step=self.global_steps)
                    self._dump_logging_data(metrics)

                    progress_bar.update(1)
                    self.global_steps += 1
                    if is_last_step:
                        pprint(f"Final validation metrics: {last_val_metrics}")
                        progress_bar.close()
                        return

