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

import gc
import json
import logging
import os
import shutil
import subprocess
import sys
import uuid
from copy import deepcopy
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


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_VERL_TO_HF_SCRIPT = os.path.join(_REPO_ROOT, "ARPO", "merge_ckpt", "convert_checkpoint_from_verl_to_hf.py")

logger = logging.getLogger(__name__)


class RayECHOTrainer(RayPPOTrainer):
    """ECHO trainer with ARPO-identical PPO training loop."""

    # Per-phase JSONL dump spec consumed by `_dump_logging_data`.
    _LOGGING_SPEC = {
        "low_level": [
            ("reward.jsonl", "reward/effective_reward_mean"),
            ("format_penalty.jsonl", "reward/bad_format_rate"),
            ("in_group_reward_std.jsonl", "reward/in_group_reward_std"),
            ("pg_loss.jsonl", "actor/pg_loss"),
            ("entropy_reg_loss.jsonl", "actor/entropy_reg_loss"),
            ("grad_norm.jsonl", "actor/grad_norm"),
            ("entropy_old_policy.jsonl", "actor/entropy_old_policy"),
            ("format_valid_rate.jsonl", "reward/format_valid_rate"),
            ("tools_total_calls.jsonl", "tools/total_calls"),
            ("tools_successful_calls.jsonl", "tools/successful_calls"),
        ],
        "high_level": [
            ("reward.jsonl", "reward/effective_reward_mean"),
            ("format_penalty.jsonl", "reward/bad_format_rate"),
            ("in_group_reward_std.jsonl", "reward/in_group_reward_std"),
            ("pg_loss.jsonl", "actor/pg_loss"),
            ("entropy_reg_loss.jsonl", "actor/entropy_reg_loss"),
            ("grad_norm.jsonl", "actor/grad_norm"),
            ("entropy_old_policy.jsonl", "actor/entropy_old_policy"),
            ("format_valid_rate.jsonl", "reward/format_valid_rate"),
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

    def _phase_rollout_cfg(self, phase_name: str):
        return self.config.actor_rollout_ref.rollout.phase_rollouts[phase_name]

    @staticmethod
    def _entropy_reg_coeff(phase_reward_cfg) -> float:
        return float(phase_reward_cfg.entropy.get("reg_coeff", 0.0))

    @staticmethod
    def _uses_entropy_regularizer(phase_reward_cfg) -> bool:
        return RayECHOTrainer._entropy_reg_coeff(phase_reward_cfg) > 0.0

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

        if "format_valid" in reward_extra_info:
            fmt_valid = torch.tensor(reward_extra_info["format_valid"], dtype=torch.float32)
            metrics["reward/format_valid_rate"] = fmt_valid.mean().item()

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

        _, reward_extra_infos_dict = compute_reward(phase_batch, self.reward_fn)
        phase_metrics.update(
            self._prefix_metrics(
                self._build_scorer_metrics(reward_extra_infos_dict),
                phase_prefix,
            )
        )

        return phase_metrics

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
        allowed_sign_cond = {"scorer", "entropy", "aepo"}
        for phase_name, _, _ in phase_specs:
            cfg = self._phase_reward_cfg(phase_name)
            algo = cfg.get("algorithm", "grpo")
            assert algo in ("grpo", "dapo"), f"{phase_name}.algorithm must be grpo or dapo, got {algo!r}"
            if algo == "dapo":
                assert bool(cfg.filter_groups.enable), f"{phase_name} algorithm=dapo requires filter_groups.enable=true"
                assert cfg.filter_groups.metric, f"{phase_name} algorithm=dapo requires filter_groups.metric"
            if bool(cfg.get("use_sign_cond_clip", False)):
                sign_cond_strategy = str(cfg.get("sign_cond_strategy", "scorer"))
                assert sign_cond_strategy in allowed_sign_cond, (
                    f"{phase_name}.sign_cond_strategy must be one of {sorted(allowed_sign_cond)}, got {sign_cond_strategy!r}"
                )

    def _validate_phase_rollout_configs(self, phase_specs) -> None:
        allowed_rollout = {"default", "aepo"}
        for phase_name, _, _ in phase_specs:
            cfg = self._phase_rollout_cfg(phase_name)
            strategy = str(cfg.get("strategy", "default"))
            assert strategy in allowed_rollout, (
                f"{phase_name} rollout strategy must be one of {sorted(allowed_rollout)}, got {strategy!r}"
            )
            if strategy == "aepo":
                assert cfg.get("aepo") is not None, f"{phase_name} rollout strategy=aepo requires phase_rollouts.{phase_name}.aepo block"

    @staticmethod
    def _rollout_strategies_uniform(phase_specs, rollout_cfg_getter) -> bool:
        strategies = {rollout_cfg_getter(name).get("strategy", "default") for name, _, _ in phase_specs}
        return len(strategies) <= 1

    @staticmethod
    def _phase_algorithm(phase_reward_cfg) -> str:
        return phase_reward_cfg.get("algorithm", "grpo")

    @staticmethod
    def _compute_in_group_reward_std(phase_batch: DataProto) -> float:
        seq_rewards = phase_batch.batch["token_level_rewards"].sum(dim=-1)
        uids = phase_batch.non_tensor_batch["uid"]
        uid_to_indices = {}
        for i, uid in enumerate(uids):
            uid_to_indices.setdefault(uid, []).append(i)
        group_stds = []
        for indices in uid_to_indices.values():
            if len(indices) > 1:
                group_stds.append(seq_rewards[indices].std().item())
        return sum(group_stds) / len(group_stds) if group_stds else 0.0

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
        run_dir = self.config.trainer.default_local_dir
        src_actor = os.path.join(run_dir, f"global_step_{self.global_steps}", "actor")
        best_dir = os.path.join(run_dir, "best_checkpoint")
        dst_hf = os.path.join(best_dir, "hf")

        legacy_actor = os.path.join(best_dir, "actor")
        if os.path.isdir(legacy_actor):
            shutil.rmtree(legacy_actor)
        legacy_data = os.path.join(best_dir, "data.pt")
        if os.path.isfile(legacy_data):
            os.remove(legacy_data)
        if os.path.isdir(dst_hf):
            shutil.rmtree(dst_hf)
        os.makedirs(best_dir, exist_ok=True)

        print(f"[best_checkpoint] merging FSDP actor to HF: {src_actor} -> {dst_hf}")
        subprocess.run(
            [
                sys.executable,
                _VERL_TO_HF_SCRIPT,
                "merge",
                "--backend",
                "fsdp",
                "--local_dir",
                src_actor,
                "--target_dir",
                dst_hf,
            ],
            check=True,
        )

        best_txt = os.path.join(run_dir, "best_checkpoint.txt")
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
        return_gen_output: bool = False,
    ):
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
        phase_batch.meta_info["mask_categories"] = dict(self.config.actor_rollout_ref.rollout.mask_categories)

        with _timer(f"{phase_name}_gen", timing_raw):
            phase_gen_batch = deepcopy(gen_batch)
            if phase_gen_batch.meta_info is None:
                phase_gen_batch.meta_info = {}
            phase_gen_batch.meta_info["rollout_n_override"] = phase_rollout_n
            phase_rollout_cfg = self._phase_rollout_cfg(phase_name)
            phase_gen_batch.meta_info["rollout_strategy"] = phase_rollout_cfg.strategy
            if phase_rollout_cfg.strategy == "aepo":
                from omegaconf import OmegaConf

                phase_gen_batch.meta_info["rollout_aepo_cfg"] = OmegaConf.to_container(
                    phase_rollout_cfg.aepo, resolve=True
                )
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

        del phase_gen_batch

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

        future_reward = None
        with _timer(f"{phase_name}_reward", timing_raw):
            if self.use_rm:
                reward_tensor = self.rm_wg.compute_rm_score(phase_batch)
                phase_batch = phase_batch.union(reward_tensor)
            if self.config.reward_model.launch_reward_fn_async:
                future_reward = compute_reward_async.remote(phase_batch, self.config, self.tokenizer)
            else:
                reward_tensor, phase_reward_extra_infos_dict = compute_reward(phase_batch, self.reward_fn)

        with _timer(f"{phase_name}_old_log_prob", timing_raw):
            # Always compute old-policy entropy for logging; it's a free byproduct of compute_log_prob.
            phase_batch.meta_info["calculate_entropy"] = True
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

        if self._uses_entropy_regularizer(phase_reward_cfg):
            phase_batch.batch["entropy_reg_loss_mask"] = phase_batch.batch[phase_mask_key].to(torch.float32)

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

        if self.config.reward_model.launch_reward_fn_async:
            reward_tensor, phase_reward_extra_infos_dict = ray.get(future_reward)
        metrics[f"{phase_prefix}reward/effective_reward_mean"] = (
            reward_tensor.to(torch.float32).sum(dim=-1).mean().detach().item()
        )
        phase_batch.batch["token_level_scores"] = reward_tensor
        if phase_reward_extra_infos_dict:
            phase_batch.non_tensor_batch.update({k: np.array(v) for k, v in phase_reward_extra_infos_dict.items()})
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

        if return_gen_output:
            return phase_batch, phase_reward_extra_infos_dict, gen_batch_output
        return phase_batch, phase_reward_extra_infos_dict

    def _phase_from_cached_rollout(
        self,
        gen_batch_output: DataProto,
        prompt_batch: DataProto,
        phase_name: str,
        phase_rollout_n: int,
        phase_mask_key: str,
        timing_raw: dict,
        metrics: dict,
    ) -> tuple[DataProto, dict]:
        """Reuse rollout from a preceding phase, recomputing rewards with this phase's strategy."""
        phase_prefix = f"{phase_name}/"
        phase_reward_cfg = self._phase_reward_cfg(phase_name)
        phase_strategy = phase_reward_cfg.strategy
        phase_reward_extra_infos_dict: dict = {}

        num_prompts = gen_batch_output.batch.batch_size[0] // phase_rollout_n
        phase_batch = DataProto(
            batch=TensorDict({}, batch_size=[num_prompts]),
            non_tensor_batch=deepcopy(prompt_batch.non_tensor_batch),
            meta_info=deepcopy(prompt_batch.meta_info) if prompt_batch.meta_info else {},
        )
        phase_batch.meta_info["phase"] = phase_name
        phase_batch.meta_info["mask_categories"] = dict(self.config.actor_rollout_ref.rollout.mask_categories)

        phase_batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(num_prompts)], dtype=object
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

        future_reward = None
        with _timer(f"{phase_name}_reward", timing_raw):
            if self.use_rm:
                reward_tensor = self.rm_wg.compute_rm_score(phase_batch)
                phase_batch = phase_batch.union(reward_tensor)
            if self.config.reward_model.launch_reward_fn_async:
                future_reward = compute_reward_async.remote(phase_batch, self.config, self.tokenizer)
            else:
                reward_tensor, phase_reward_extra_infos_dict = compute_reward(phase_batch, self.reward_fn)

        with _timer(f"{phase_name}_old_log_prob", timing_raw):
            phase_batch.meta_info["calculate_entropy"] = True
            old_log_prob = self.actor_rollout_wg.compute_log_prob(phase_batch)
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropys = old_log_prob.batch.pop("entropys", None)
            if entropys is not None:
                entropy_loss = agg_loss(
                    loss_mat=entropys, loss_mask=phase_batch.batch["loss_mask"], loss_agg_mode=loss_agg_mode
                )
                metrics[f"{phase_prefix}actor/entropy_old_policy"] = entropy_loss.detach().item()
            phase_batch = phase_batch.union(old_log_prob)

        if self._uses_entropy_regularizer(phase_reward_cfg):
            phase_batch.batch["entropy_reg_loss_mask"] = phase_batch.batch[phase_mask_key].to(torch.float32)

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

        if self.config.reward_model.launch_reward_fn_async:
            reward_tensor, phase_reward_extra_infos_dict = ray.get(future_reward)
        metrics[f"{phase_prefix}reward/effective_reward_mean"] = (
            reward_tensor.to(torch.float32).sum(dim=-1).mean().detach().item()
        )
        phase_batch.batch["token_level_scores"] = reward_tensor
        if phase_reward_extra_infos_dict:
            phase_batch.non_tensor_batch.update({k: np.array(v) for k, v in phase_reward_extra_infos_dict.items()})
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
                high_level_budget = int(self.config.actor_rollout_ref.rollout.get("high_level_budget", 0))
                low_level_budget = int(self.config.actor_rollout_ref.rollout.get("low_level_budget", 0))

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
                self._validate_phase_rollout_configs(phase_specs)
                repeated_phase_specs = self._expand_phase_specs_with_repeats(phase_specs)

                norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                reuse_phase_rollouts = bool(self.config.actor_rollout_ref.rollout.get("reuse_phase_rollouts", False))
                if reuse_phase_rollouts and not self._rollout_strategies_uniform(phase_specs, self._phase_rollout_cfg):
                    logger.warning(
                        "reuse_phase_rollouts disabled: active phases use different rollout strategies; "
                        "regenerating rollouts per phase."
                    )
                    reuse_phase_rollouts = False

                batch_dict, data_iter = self._next_batch_dict(data_iter)
                step_gen_batch, step_prompt_batch = self._pop_gen_batch(DataProto.from_single_dict(batch_dict))
                cached_gen_batch_output = None
                cached_rollout_n = None

                for phase_idx, (phase_name, phase_rollout_n, phase_mask_key) in enumerate(repeated_phase_specs):
                    is_last_phase_in_cycle = (phase_idx == len(repeated_phase_specs) - 1)
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
                        elif reuse_phase_rollouts and cached_gen_batch_output is not None:
                            phase_rollout_n = cached_rollout_n
                            phase_batch, phase_reward_extra_infos_dict = self._phase_from_cached_rollout(
                                cached_gen_batch_output, step_prompt_batch, phase_name,
                                phase_rollout_n, phase_mask_key, timing_raw, metrics
                            )
                            del cached_gen_batch_output
                            cached_gen_batch_output = None
                            metrics[f"{phase_prefix}training/num_gen_batches"] = 0
                        else:
                            if reuse_phase_rollouts:
                                phase_batch, phase_reward_extra_infos_dict, gen_output = self._phase_rollout_to_scored_batch(
                                    step_gen_batch, step_prompt_batch, phase_name, phase_rollout_n, phase_mask_key, timing_raw, metrics,
                                    return_gen_output=True,
                                )
                                cached_gen_batch_output = gen_output
                                del gen_output
                                cached_rollout_n = phase_rollout_n
                            else:
                                phase_batch, phase_reward_extra_infos_dict = self._phase_rollout_to_scored_batch(
                                    step_gen_batch, step_prompt_batch, phase_name, phase_rollout_n, phase_mask_key, timing_raw, metrics,
                                )
                            metrics[f"{phase_prefix}training/num_gen_batches"] = 1

                        gc.collect()
                        torch.cuda.empty_cache()

                        with _timer(f"{phase_name}_adv", timing_raw):
                            metrics[f"{phase_prefix}reward/in_group_reward_std"] = self._compute_in_group_reward_std(phase_batch)
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
                                phase_batch.meta_info["use_aepo_clip_override"] = bool(
                                    phase_reward_cfg.get("use_aepo_clip", False)
                                )
                                phase_batch.meta_info["use_sign_cond_clip_override"] = bool(
                                    phase_reward_cfg.get("use_sign_cond_clip", False)
                                )
                                phase_batch.meta_info["sign_cond_strategy"] = str(
                                    phase_reward_cfg.get("sign_cond_strategy", "scorer")
                                )
                                phase_batch.meta_info["phase_strategy"] = phase_strategy
                                phase_batch.meta_info["entropy_normalization"] = str(
                                    phase_reward_cfg.entropy.get("normalization", "token_pool")
                                )
                                phase_batch.meta_info["entropy_alpha"] = float(
                                    phase_reward_cfg.entropy.get("alpha", 0.2)
                                )
                                if self._uses_entropy_regularizer(phase_reward_cfg):
                                    phase_batch.meta_info["entropy_coeff_override"] = self._entropy_reg_coeff(phase_reward_cfg)
                                    phase_batch.meta_info["entropy_loss_mask_key"] = "entropy_reg_loss_mask"
                                phase_batch.meta_info["skip_lr_step"] = not is_last_phase_in_cycle
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

