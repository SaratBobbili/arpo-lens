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
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, ResourcePoolManager, Role, RayPPOTrainer, _timer
from .echo_core_algos import agg_loss, apply_kl_penalty, compute_advantage
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.metric import reduce_metrics


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_VERL_TO_HF_SCRIPT = os.path.join(_REPO_ROOT, "ARPO", "merge_ckpt", "convert_checkpoint_from_verl_to_hf.py")

logger = logging.getLogger(__name__)


class _PhaseDataloaders:
    """Per-phase prompt loaders behind the single state_dict API the base checkpointer uses."""

    def __init__(self, loaders: dict):
        self.loaders = loaders

    def state_dict(self) -> dict:
        return {phase: loader.state_dict() for phase, loader in self.loaders.items()}

    def load_state_dict(self, state_dict: dict) -> None:
        for phase, loader in self.loaders.items():
            loader.load_state_dict(state_dict[phase])


class RayECHOTrainer(RayPPOTrainer):
    """ECHO trainer with ARPO-identical PPO training loop."""

    # Inner phase first: one outer cycle is num_iters[low_level] LL iterations then one HL iteration.
    _PHASE_NAMES = ("low_level", "high_level")
    _PHASE_MASK_KEYS = {"low_level": "low_level_loss_mask", "high_level": "high_level_loss_mask"}

    # Phase-owned optimizer signals only (prefixed high_level/|low_level/).
    _PHASE_OWNED_KEYS = frozenset({
        "actor/lr",
        "actor/pg_loss",
        "actor/grad_norm",
        "actor/entropy_reg_loss",
        "actor/entropy_reg_coef",
        "actor/opefo_lambda",
        "actor/opefo_delta_H_net",
        "actor/opefo_pos_mag",
        "actor/opefo_neg_mag",
        # Response term (Eqs. 15-16). response_to_direct_ratio is the empirical content of
        # Proposition 1: how far the frozen-response stationary point sits from x*.
        "actor/response_norm",
        "actor/direct_norm",
        "actor/response_to_direct_ratio",
        "actor/response_direct_cosine",
        "actor/response_c_mean",
        "actor/response_c_std",
        "actor/response_blocks",
        "actor/follower_direction_pg_loss",
        # Post-response allocator state. vLLM's wake_up() maps physical pages, so what
        # matters for the next generation is what is left RESERVED here, not allocated.
        "actor/response_mem_reserved_gb",
        "actor/response_mem_allocated_gb",
    })

    _ACTOR_TO_POLICY = {
        "actor/ppo_kl": "policy/ppo_kl",
        "actor/pg_clipfrac": "policy/pg_clipfrac",
    }

    # WARNING: these come from the rollout's meta_info, and DataProto.concat keeps meta_info from
    # rank 0 only (verl/protocol.py:710), so they report a SINGLE data-parallel shard, not the
    # batch. Treat them as relative trends only. policy/tool_calls_per_traj_mean and
    # policy/budget_exhausted_rate are computed from per-sample arrays and are batch-correct.
    _TOOL_TO_POLICY = {
        "tools/total_calls": "policy/tools_total_calls",
        "tools/successful_calls": "policy/tools_successful_calls",
    }

    # JSONL dump spec: phase-owned under {phase}/, shared health under policy/.
    _LOGGING_SPEC = {
        "low_level": [
            ("pg_loss.jsonl", "actor/pg_loss"),
            ("entropy_reg_loss.jsonl", "actor/entropy_reg_loss"),
            ("grad_norm.jsonl", "actor/grad_norm"),
            ("opefo_lambda.jsonl", "actor/opefo_lambda"),
            ("opefo_delta_H_net.jsonl", "actor/opefo_delta_H_net"),
            ("entropy_phase_mask.jsonl", "actor/entropy_phase_mask"),
        ],
        "high_level": [
            ("pg_loss.jsonl", "actor/pg_loss"),
            ("entropy_reg_loss.jsonl", "actor/entropy_reg_loss"),
            ("grad_norm.jsonl", "actor/grad_norm"),
            ("opefo_lambda.jsonl", "actor/opefo_lambda"),
            ("opefo_delta_H_net.jsonl", "actor/opefo_delta_H_net"),
            ("entropy_phase_mask.jsonl", "actor/entropy_phase_mask"),
            ("response_norm.jsonl", "actor/response_norm"),
            ("response_to_direct_ratio.jsonl", "actor/response_to_direct_ratio"),
            ("response_direct_cosine.jsonl", "actor/response_direct_cosine"),
            ("response_c_mean.jsonl", "actor/response_c_mean"),
            ("response_c_std.jsonl", "actor/response_c_std"),
            ("response_mem_reserved_gb.jsonl", "actor/response_mem_reserved_gb"),
        ],
        "policy": [
            ("reward.jsonl", "reward_mean"),
            ("format_penalty.jsonl", "bad_format_rate"),
            ("in_group_reward_std.jsonl", "in_group_reward_std"),
            ("format_valid_rate.jsonl", "format_valid_rate"),
            ("f1_mean.jsonl", "f1_mean"),
            ("no_tool_rate.jsonl", "no_tool_rate"),
            ("advantage_std.jsonl", "advantage_std"),
            ("ppo_kl.jsonl", "ppo_kl"),
            ("pg_clipfrac.jsonl", "pg_clipfrac"),
            ("response_length_mean.jsonl", "response_length_mean"),
            ("response_length_clip_ratio.jsonl", "response_length_clip_ratio"),
            ("tools_total_calls.jsonl", "tools_total_calls"),
            ("tools_successful_calls.jsonl", "tools_successful_calls"),
            ("entropy.jsonl", "entropy"),
            ("group_zero_std_frac.jsonl", "group_zero_std_frac"),
            ("budget_exhausted_rate.jsonl", "budget_exhausted_rate"),
            ("tool_calls_per_traj_mean.jsonl", "tool_calls_per_traj_mean"),
            ("fail_answer_count_0.jsonl", "fail_answer_count_0"),
            ("fail_unclosed_tag.jsonl", "fail_unclosed_tag"),
            ("fail_no_boxed.jsonl", "fail_no_boxed"),
            ("fail_other.jsonl", "fail_other"),
        ],
    }

    @staticmethod
    def _prefix_phase_owned(metrics_dict: dict, phase_prefix: str) -> dict:
        return {
            f"{phase_prefix}{key}": value
            for key, value in metrics_dict.items()
            if key in RayECHOTrainer._PHASE_OWNED_KEYS
        }

    @staticmethod
    def _remap_to_policy(metrics_dict: dict, mapping: dict) -> dict:
        return {dst: metrics_dict[src] for src, dst in mapping.items() if src in metrics_dict}

    def _phase_cfg(self, phase_name: str):
        return self.config.phases[phase_name]

    def _response_cfg(self):
        return self.config.phases.get("response", None)

    def _response_enabled(self) -> bool:
        """Algorithm 1's round structure: reset/discard, r_valid, one-step, final response."""
        cfg = self._response_cfg()
        return bool(cfg.get("enabled", False)) if cfg is not None else False

    def _response_gradient_enabled(self) -> bool:
        """The g_resp term itself. Gated separately from the structure above.

        The terminal seed is currently annihilated by a mask composition (see the config
        comment on phases.response.gradient), so g_resp is identically zero and the sweep
        costs 2K extra full-batch passes per round to add nothing. Keeping this separate
        lets the round structure run without paying for that.
        """
        cfg = self._response_cfg()
        if cfg is None or not self._response_enabled():
            return False
        return bool(cfg.get("gradient", False))

    def _phase_rollout_cfg(self, phase_name: str):
        return self.config.phases[phase_name].rollout

    @staticmethod
    def _entropy_reg_coeff(phase_cfg) -> float:
        return float(phase_cfg.entropy.get("reg_coeff", 0.0))

    @staticmethod
    def _uses_entropy_regularizer(phase_cfg) -> bool:
        # lambda_ent H_tool in u_L (Eq. 2) is flag-gated and off by default, so out of the
        # box each role is driven purely by its own return. reg_coeff alone no longer
        # switches it on.
        if not bool(phase_cfg.entropy.get("enabled", False)):
            return False
        return RayECHOTrainer._entropy_reg_coeff(phase_cfg) > 0.0

    def _init_logging_data(self) -> None:
        # One JSONL per (phase, metric) under {default_local_dir}/logging_data/.
        # Read by training/analysis/plot_training_log.py for offline per-step plots.
        self._logging_data_root = os.path.join(self.config.trainer.default_local_dir, "logging_data")
        self._prev_logged_values: dict[str, float] = {}
        for phase in self._LOGGING_SPEC:
            os.makedirs(os.path.join(self._logging_data_root, phase), exist_ok=True)

    def _dump_logging_data(self, metrics: dict) -> None:
        # Append one line per metric file: {"step", "value", "gain"}. `gain` is
        # the difference vs the previous dumped step for the same key, or null
        # on the first dump. Keys absent from `metrics` (e.g. entropy_reg_loss
        # when reg_coeff=0) are skipped without erroring.
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
            metrics["policy/reward_mean"] = scores.mean().item()
            metrics["policy/bad_format_rate"] = (scores < 0.0).to(torch.float32).mean().item()

        if "f1_score" in reward_extra_info:
            f1_scores = torch.tensor(reward_extra_info["f1_score"], dtype=torch.float32)
            metrics["policy/f1_mean"] = f1_scores.mean().item()

        if "no_tool_calls" in reward_extra_info:
            no_tool = torch.tensor(reward_extra_info["no_tool_calls"], dtype=torch.float32)
            metrics["policy/no_tool_rate"] = no_tool.mean().item()

        if "format_valid" in reward_extra_info:
            fmt_valid = torch.tensor(reward_extra_info["format_valid"], dtype=torch.float32)
            metrics["policy/format_valid_rate"] = fmt_valid.mean().item()

        # A single bad_format_rate scalar hid a reward bug for six runs: ~86% of step-1 format
        # failures were tool-budget truncations scored -1, not real schema errors. Bucket the
        # scorer's own reason string so the next one is visible in one glance.
        reasons = reward_extra_info.get("reason")
        if reasons:
            total = float(len(reasons))
            buckets = {"answer_count_0": 0, "unclosed_tag": 0, "no_boxed": 0, "other": 0}
            for raw in reasons:
                reason = str(raw)
                if not reason.startswith("bad format: ") and "cannot extract answer" not in reason and "boxed" not in reason:
                    continue
                detail = reason[len("bad format: "):] if reason.startswith("bad format: ") else reason
                if detail.startswith("answer_count=0"):
                    buckets["answer_count_0"] += 1
                elif "is not closed" in detail:
                    buckets["unclosed_tag"] += 1
                elif "boxed" in detail:
                    buckets["no_boxed"] += 1
                else:
                    buckets["other"] += 1
            for name, count in buckets.items():
                metrics[f"policy/fail_{name}"] = count / total

        return metrics

    @staticmethod
    def _rollout_behavior_metrics(phase_batch: DataProto) -> dict:
        """Tool-use metrics computed from per-sample arrays, not from the rollout's counters.

        tools/* counters ride in meta_info, and DataProto.concat keeps meta_info from rank 0
        only (verl/protocol.py:710), so every tools/* number in the logs is a single shard.
        These are the world-size-correct versions.
        """
        out: dict = {}
        calls = phase_batch.non_tensor_batch.get("tool_calls_made")
        if calls is not None:
            calls_t = torch.tensor(np.asarray(calls, dtype=np.float32))
            out["policy/tool_calls_per_traj_mean"] = calls_t.mean().item()
        exhausted = phase_batch.non_tensor_batch.get("tool_budget_exhausted")
        if exhausted is not None:
            out["policy/budget_exhausted_rate"] = float(np.asarray(exhausted, dtype=np.float32).mean())
        return out

    @staticmethod
    def _policy_from_data_metrics(data_metrics: dict, phase_batch: DataProto) -> dict:
        out: dict = {}
        if "response_length/mean" in data_metrics:
            out["policy/response_length_mean"] = data_metrics["response_length/mean"]
        if "response_length/clip_ratio" in data_metrics:
            out["policy/response_length_clip_ratio"] = data_metrics["response_length/clip_ratio"]
        advantages = phase_batch.batch["advantages"]
        max_response_length = phase_batch.batch["responses"].shape[-1]
        response_mask = phase_batch.batch["attention_mask"][:, -max_response_length:].bool()
        valid_adv = torch.masked_select(advantages, response_mask)
        out["policy/advantage_std"] = (
            torch.std(valid_adv).detach().item() if valid_adv.numel() > 1 else 0.0
        )
        return out

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
        masked = phase_batch.batch[phase_mask_key] * keep
        phase_batch.batch[phase_mask_key] = masked
        # loss_mask / response_mask were aliased to the same tensor object at rollout-to-batch
        # time. Rebinding phase_mask_key above leaves them pointing at the unmasked original, so
        # zeroing the phase mask would silently not reach the loss. Rebind them too.
        for alias in ("loss_mask", "response_mask"):
            if alias in phase_batch.batch.keys():
                phase_batch.batch[alias] = masked

    def _budget_exhausted_and_invalid(self, phase_batch: DataProto) -> np.ndarray | None:
        """Samples that spent the tool budget and still failed the format gate.

        A rollout that exhausts its tool budget now gets a masked notice and a final turn to
        answer (see vllm_rollout_echo). One that answers legally is real data and is scored
        normally -- a wrong answer earns 0, which is the honest signal. This is only the residual
        safety net for the ones that still produce nothing scoreable, so they contribute 0 instead
        of the -1 that used to teach the policy to stop calling tools.
        """
        if not bool(self._echo_rollout_tools_cfg().get("skip_training_on_budget_exhausted", True)):
            return None
        exhausted = phase_batch.non_tensor_batch.get("tool_budget_exhausted")
        if exhausted is None:
            return None
        fmt_valid = phase_batch.non_tensor_batch.get("format_valid")
        if fmt_valid is None:
            return None
        flags = np.asarray(exhausted, dtype=np.bool_) & ~np.asarray(fmt_valid, dtype=np.bool_)
        return flags if np.any(flags) else None

    def _apply_tool_failure_before_grpo(self, phase_batch: DataProto) -> None:
        flags = None
        if bool(self._echo_rollout_tools_cfg().get("skip_training_on_tool_failure", False)):
            tool_failed = phase_batch.non_tensor_batch.get("tool_rollout_failed")
            if tool_failed is not None and np.any(tool_failed):
                flags = np.asarray(tool_failed, dtype=np.bool_)

        budget_flags = self._budget_exhausted_and_invalid(phase_batch)
        if budget_flags is not None:
            flags = budget_flags if flags is None else (flags | budget_flags)

        if flags is None or not np.any(flags):
            return

        dev = phase_batch.batch["token_level_rewards"].device
        failed = torch.tensor(flags, device=dev)
        phase_batch.batch["token_level_rewards"][failed] = 0
        phase_batch.batch["token_level_scores"][failed] = 0
        # Fresh uid per dropped sample: a singleton group has mean 0, so its advantage is 0 and it
        # cannot shift the baseline of the group it came from.
        uids = phase_batch.non_tensor_batch["uid"].copy()
        for i in np.flatnonzero(flags):
            uids[i] = str(uuid.uuid4())
        phase_batch.non_tensor_batch["uid"] = uids

    def _validate_phase_reward_configs(self) -> None:
        allowed_adv = {"grpo", "entropy", "aepo"}
        for phase_name in self._PHASE_NAMES:
            cfg = self._phase_cfg(phase_name)
            adv_algo = str(cfg.get("advantage_algorithm", "grpo"))
            assert adv_algo in allowed_adv, (
                f"phases.{phase_name}.advantage_algorithm must be one of {sorted(allowed_adv)}, got {adv_algo!r}"
            )
            assert int(cfg.group_size) >= 1, (
                f"phases.{phase_name}.group_size must be >= 1, got {cfg.group_size}."
            )

    def _validate_phase_rollout_configs(self) -> None:
        allowed_rollout = {"default", "aepo"}
        for phase_name in self._PHASE_NAMES:
            cfg = self._phase_rollout_cfg(phase_name)
            strategy = str(cfg.get("strategy", "default"))
            assert strategy in allowed_rollout, (
                f"phases.{phase_name}.rollout.strategy must be one of {sorted(allowed_rollout)}, got {strategy!r}"
            )
            if strategy == "aepo":
                assert cfg.get("aepo") is not None, f"phases.{phase_name}.rollout.strategy=aepo requires an aepo block"

    def _validate_response_config(self) -> None:
        """Algorithm 1 takes one update per fresh group; the code must too.

        C.3: "if a group is reused, each clipped optimizer epoch and its fixed behavior
        snapshot is a separate substep of the adaptation map." With several mini-batches
        per iteration the true adaptation map would be K x n_minibatch substeps, and the
        stashed-record-to-substep correspondence the reverse sweep relies on would break.
        Requiring one step per iteration keeps K = phases.low_level.num_iters exactly.
        """
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
        if not group_stds:
            return 0.0, 0.0
        # A group whose members all score the same yields advantage 0 for every member, i.e. it
        # contributes no gradient. The fraction of such groups is the direct read on whether the
        # GRPO signal is still alive.
        zero_std_frac = sum(1 for std in group_stds if std < 1e-6) / len(group_stds)
        return sum(group_stds) / len(group_stds), zero_std_frac

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

    def _phase_sampler(self, seed: int):
        if not self.config.data.shuffle:
            return SequentialSampler(data_source=self.train_dataset)
        generator = torch.Generator()
        generator.manual_seed(seed)
        return RandomSampler(data_source=self.train_dataset, generator=generator)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """Build prompt stream(s); batch size floored to a multiple of world_size.

        Shared mode: one dataloader sized by steps_per_epoch = N_HL*(N_LL+1).
        Legacy: one dataloader per phase sized by that phase's num_iters.
        """
        from verl.trainer.main_ppo import create_rl_dataset

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self._validate_phase_reward_configs()
        self._validate_phase_rollout_configs()

        dataset_size = len(self.train_dataset)
        world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        num_workers = self.config.data.get("dataloader_num_workers", 8)
        base_seed = self.config.data.get("seed", 1)

        num_hl_iters = int(self._phase_cfg("high_level").num_iters)
        num_ll_iters = int(self._phase_cfg("low_level").num_iters)
        self._shared_prompt_stream = bool(self.config.phases.get("shared_prompt_stream", False))
        self._phase_dataloaders = {}
        self._phase_iters = {}
        # Prompt batch per phase iteration; the response term requires the phase's
        # ppo_mini_batch_size to equal it, so one iteration is one optimizer step.
        self._phase_prompt_batch_sizes = {}

        if self._shared_prompt_stream:
            steps_per_epoch = num_hl_iters * (num_ll_iters + 1)
            assert 1 <= steps_per_epoch <= dataset_size, (
                f"shared steps_per_epoch=N_HL*(N_LL+1) must be in [1, {dataset_size}], got {steps_per_epoch}."
            )
            batch_size = (dataset_size // steps_per_epoch) // world_size * world_size
            assert batch_size >= world_size, (
                f"shared steps_per_epoch={steps_per_epoch} leaves {dataset_size // steps_per_epoch} prompts per step, "
                f"fewer than one per rank ({world_size}); lower N_HL/N_LL."
            )
            shared_loader = StatefulDataLoader(
                dataset=self.train_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                drop_last=True,
                collate_fn=collate_fn,
                sampler=self._phase_sampler(seed=base_seed),
            )
            for phase_name in self._PHASE_NAMES:
                self._phase_dataloaders[phase_name] = shared_loader
                self._phase_prompt_batch_sizes[phase_name] = batch_size
            print(
                f"[shared] steps_per_epoch={steps_per_epoch}, prompt_batch_size={batch_size}, "
                f"batches_per_dataset_pass={len(shared_loader)}"
            )
        else:
            for seed_offset, phase_name in enumerate(self._PHASE_NAMES):
                num_iters = int(self._phase_cfg(phase_name).num_iters)
                assert 1 <= num_iters <= dataset_size, (
                    f"phases.{phase_name}.num_iters must be in [1, {dataset_size}], got {num_iters}."
                )
                batch_size = (dataset_size // num_iters) // world_size * world_size
                assert batch_size >= world_size, (
                    f"phases.{phase_name}.num_iters={num_iters} leaves {dataset_size // num_iters} prompts per iteration, "
                    f"fewer than one per rank ({world_size}); lower num_iters."
                )
                self._phase_prompt_batch_sizes[phase_name] = batch_size
                self._phase_dataloaders[phase_name] = StatefulDataLoader(
                    dataset=self.train_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    drop_last=True,
                    collate_fn=collate_fn,
                    sampler=self._phase_sampler(seed=base_seed + seed_offset),
                )
                print(
                    f"[{phase_name}] num_iters={num_iters}, prompt_batch_size={batch_size}, "
                    f"batches_per_dataset_pass={len(self._phase_dataloaders[phase_name])}"
                )

        # Needs the derived prompt batch sizes, so it runs after the loaders are built.
        self._validate_response_config()

        # The base checkpointer saves/loads `self.train_dataloader.state_dict()`.
        self.train_dataloader = _PhaseDataloaders(self._phase_dataloaders)

        val_batch_size = self.config.data.val_batch_size
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        if self._shared_prompt_stream:
            total_epochs = int(self.config.trainer.total_epochs)
            self.total_training_steps = total_epochs * num_hl_iters * (num_ll_iters + 1)
        else:
            self.total_training_steps = num_hl_iters * (num_ll_iters + 1)
        print(f"Total training steps: {self.total_training_steps}")

    def _next_batch_dict(self, phase_name: str):
        iter_key = "_shared" if self._shared_prompt_stream else phase_name
        data_iter = self._phase_iters.get(iter_key)
        if data_iter is None:
            data_iter = iter(self._phase_dataloaders[phase_name])
            self._phase_iters[iter_key] = data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(self._phase_dataloaders[phase_name])
            self._phase_iters[iter_key] = data_iter
            return next(data_iter)

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
    ):
        phase_cfg = self._phase_cfg(phase_name)
        advantage_algorithm = str(phase_cfg.get("advantage_algorithm", "grpo"))
        phase_reward_extra_infos_dict: dict = {}

        phase_batch = DataProto(
            batch=TensorDict({}, batch_size=gen_batch.batch.batch_size),
            non_tensor_batch=deepcopy(prompt_batch.non_tensor_batch),
            meta_info=deepcopy(prompt_batch.meta_info) if prompt_batch.meta_info else {},
        )
        phase_batch.meta_info["phase"] = phase_name
        phase_batch.meta_info["mask_categories"] = dict(self.config.actor_rollout_ref.rollout.mask_categories)
        # Algorithm 1 scores u_L by r_valid (Eq. 2) and u_H by r_task (Eq. 3). Off, both
        # phases share the task score, which is the pre-Algorithm-1 behavior.
        phase_batch.meta_info["use_follower_return"] = self._response_enabled()

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
                metrics.update(self._remap_to_policy(gen_batch_output.meta_info["metrics"], self._TOOL_TO_POLICY))

        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            with _timer(f"{phase_name}_gen_max", timing_raw):
                if advantage_algorithm == "grpo":
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
            self._balance_batch(phase_batch, metrics={})

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
            entropys = old_log_prob.batch["entropys"]
            attention_mask = phase_batch.batch["attention_mask"]
            responses = phase_batch.batch["responses"]
            response_length = responses.size(1)
            full_response_mask = attention_mask[:, -response_length:]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            # policy/entropy is the whole-policy number over every response token, including the
            # <result> spans the loss never sees. The phase-mask version below is the entropy of
            # what this phase actually optimizes; report both so the two are never confused.
            metrics["policy/entropy"] = agg_loss(
                loss_mat=entropys, loss_mask=full_response_mask, loss_agg_mode=loss_agg_mode
            ).detach().item()
            metrics[f"{phase_name}/actor/entropy_phase_mask"] = agg_loss(
                loss_mat=entropys, loss_mask=phase_batch.batch[phase_mask_key], loss_agg_mode=loss_agg_mode
            ).detach().item()
            old_log_prob.batch.pop("entropys")
            phase_batch = phase_batch.union(old_log_prob)

            # NOTE: the ECHO rollout never emits rollout_log_probs (vllm_rollout_echo sets
            # self.logprobs = 0), so the old policy/rollout_probs_diff_mean metric could never
            # fire. It was removed rather than left as a silent no-op in the dashboard; re-add it
            # together with vLLM logprobs if the vLLM/actor mismatch check is ever needed.

        if self._uses_entropy_regularizer(phase_cfg):
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
        metrics["policy/reward_mean"] = (
            reward_tensor.to(torch.float32).sum(dim=-1).mean().detach().item()
        )
        phase_batch.batch["token_level_scores"] = reward_tensor
        if phase_reward_extra_infos_dict:
            phase_batch.non_tensor_batch.update({k: np.array(v) for k, v in phase_reward_extra_infos_dict.items()})
            metrics.update(self._build_scorer_metrics(phase_reward_extra_infos_dict))
        metrics.update(self._rollout_behavior_metrics(phase_batch))

        if self.config.algorithm.use_kl_in_reward:
            phase_batch, _kl_metrics = apply_kl_penalty(
                phase_batch,
                kl_ctrl=self.kl_ctrl_in_reward,
                kl_penalty=self.config.algorithm.kl_penalty,
            )
        else:
            phase_batch.batch["token_level_rewards"] = phase_batch.batch["token_level_scores"]

        return phase_batch, phase_reward_extra_infos_dict

    def _run_phase_iteration(self, phase_name: str, hl_cycle: int, end_of_cycle: bool, logger, progress_bar) -> None:
        """One GRPO iteration for `phase_name`: fresh prompt chunk -> rollout -> reward -> advantage -> update.

        Rollouts always come from the policy as left by the previous iteration's optimizer step,
        because the prompt chunk is only fetched here and `generate_sequences` resyncs FSDP -> vLLM.
        """
        phase_prefix = f"{phase_name}/"
        phase_cfg = self._phase_cfg(phase_name)
        phase_rollout_n = int(phase_cfg.group_size)
        phase_mask_key = self._PHASE_MASK_KEYS[phase_name]
        metrics = {}
        timing_raw = {}
        is_last_step = self.global_steps >= self.total_training_steps
        saved_checkpoint_this_step = False

        with _timer("step", timing_raw):
            batch_dict = self._next_batch_dict(phase_name)
            gen_batch, prompt_batch = self._pop_gen_batch(DataProto.from_single_dict(batch_dict))
            phase_batch, phase_reward_extra_infos_dict = self._phase_rollout_to_scored_batch(
                gen_batch, prompt_batch, phase_name, phase_rollout_n, phase_mask_key, timing_raw, metrics,
            )
            del gen_batch, prompt_batch
            gc.collect()
            torch.cuda.empty_cache()

            with _timer(f"{phase_name}_adv", timing_raw):
                in_group_std, zero_std_frac = self._compute_in_group_reward_std(phase_batch)
                metrics["policy/in_group_reward_std"] = in_group_std
                metrics["policy/group_zero_std_frac"] = zero_std_frac
                self._apply_tool_failure_before_grpo(phase_batch)

                phase_batch = compute_advantage(
                    phase_batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=phase_rollout_n,
                    norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                    multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                    use_pf_ppo=self.config.algorithm.use_pf_ppo,
                    pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                    pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                )
                data_metrics = compute_data_metrics(batch=phase_batch, use_critic=self.use_critic)
                metrics.update(self._policy_from_data_metrics(data_metrics, phase_batch))

            if self.use_critic:
                with _timer(f"{phase_name}_update_critic", timing_raw):
                    self.critic_wg.update_critic(phase_batch)

            if self.config.trainer.critic_warmup <= self.global_steps:
                with _timer(f"{phase_name}_update_actor", timing_raw):
                    phase_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    phase_batch.meta_info["kl_loss_coef_override"] = float(
                        phase_cfg.get("kl_loss_coef", self.config.actor_rollout_ref.actor.kl_loss_coef)
                    )
                    phase_batch.meta_info["use_aepo_clip_override"] = bool(phase_cfg.get("use_aepo_clip", False))
                    phase_batch.meta_info["use_sign_cond_clip_override"] = bool(
                        phase_cfg.get("use_sign_cond_clip", False)
                    )
                    phase_batch.meta_info["advantage_algorithm"] = str(phase_cfg.get("advantage_algorithm", "grpo"))
                    phase_batch.meta_info["entropy_normalization"] = str(
                        phase_cfg.entropy.get("normalization", "token_pool")
                    )
                    phase_batch.meta_info["entropy_alpha"] = float(phase_cfg.entropy.get("alpha", 0.2))
                    opefo_cfg = phase_cfg.get("opefo", None)
                    phase_batch.meta_info["opefo_enabled"] = bool(
                        opefo_cfg.get("enabled", False) if opefo_cfg is not None else False
                    )
                    response_cfg = self._response_cfg()
                    phase_batch.meta_info["response_enabled"] = self._response_gradient_enabled()
                    if response_cfg is not None:
                        phase_batch.meta_info["response_coef"] = float(response_cfg.get("coef", 1.0))
                        phase_batch.meta_info["response_replay_fraction"] = float(
                            response_cfg.get("replay_fraction", 1.0)
                        )
                    if self._uses_entropy_regularizer(phase_cfg):
                        phase_batch.meta_info["entropy_coeff_override"] = self._entropy_reg_coeff(phase_cfg)
                        phase_batch.meta_info["entropy_loss_mask_key"] = "entropy_reg_loss_mask"
                    actor_output = self.actor_rollout_wg.update_actor(phase_batch)
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(self._prefix_phase_owned(actor_output_metrics, phase_prefix))
                metrics.update(self._remap_to_policy(actor_output_metrics, self._ACTOR_TO_POLICY))
                metrics.update({k: v for k, v in actor_output_metrics.items() if k.startswith("perf/")})

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

            # Validation and checkpointing only after HL updates (end_of_cycle).
            # Shared + legacy: save/test_freq count HL updates (shared spans epochs).
            if end_of_cycle:
                if self._shared_prompt_stream:
                    n_hl = int(self._phase_cfg("high_level").num_iters)
                    hl_done = self._current_epoch * n_hl + hl_cycle + 1
                else:
                    hl_done = hl_cycle + 1
                test_due = self.config.trainer.test_freq > 0 and hl_done % self.config.trainer.test_freq == 0
                save_due = self.config.trainer.save_freq > 0 and hl_done % self.config.trainer.save_freq == 0

                if self.val_reward_fn is not None and (is_last_step or test_due):
                    with _timer("testing", timing_raw):
                        val_metrics: dict = self._validate()
                        self._last_val_metrics = val_metrics
                    metrics.update(val_metrics)
                    # These measure the BARE LEADER, not the adapted pair: the round has
                    # already discarded y_K and stepped x, so the weights under test are
                    # w_core + x_{t+1} with no follower adaptation. Algorithm 1's object is
                    # the pair (x, xi_K(x)), so val-core/* is a leader-only proxy for it --
                    # and best-checkpoint selection below inherits that. Flagged rather
                    # than silently reported, since the two are easy to conflate.
                    metrics["training/val_evaluates_adapted_pair"] = 0.0

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

                if not saved_checkpoint_this_step and (is_last_step or save_due):
                    with _timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

        metrics.update(
            {
                "training/global_step": self.global_steps,
                "training/hl_cycle": hl_cycle,
            }
        )
        if self._shared_prompt_stream:
            metrics["training/epoch"] = self._current_epoch
        timing_metrics = compute_timing_metrics(batch=phase_batch, timing_raw=timing_raw)
        if "timing_s/step" in timing_metrics:
            metrics["timing_s/step"] = timing_metrics["timing_s/step"]
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=phase_batch, timing_raw=timing_raw, n_gpus=n_gpus))

        logger.log(data=metrics, step=self.global_steps)
        self._dump_logging_data(metrics)

        progress_bar.update(1)
        self.global_steps += 1

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

        def _open_round() -> None:
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

        def _run_cycle(hl_cycle: int) -> None:
            _open_round()
            for _ in range(num_ll_iters):
                self._run_phase_iteration("low_level", hl_cycle, end_of_cycle=False, logger=logger, progress_bar=progress_bar)
            # The leader iteration restores W_x inside the worker between its last backward
            # and its optimizer step, so the adapted follower is discarded there (line 14).
            self._run_phase_iteration("high_level", hl_cycle, end_of_cycle=True, logger=logger, progress_bar=progress_bar)
            if self._response_gradient_enabled():
                self.actor_rollout_wg.clear_follower_records()

        if self._shared_prompt_stream:
            total_epochs = int(self.config.trainer.total_epochs)
            start_epoch = (self.global_steps - 1) // steps_per_epoch
            start_cycle = ((self.global_steps - 1) % steps_per_epoch) // cycle_len
            for epoch in range(start_epoch, total_epochs):
                self._current_epoch = epoch
                cycle_start = start_cycle if epoch == start_epoch else 0
                for hl_cycle in range(cycle_start, num_hl_iters):
                    _run_cycle(hl_cycle)
        else:
            start_cycle = (self.global_steps - 1) // cycle_len
            for hl_cycle in range(start_cycle, num_hl_iters):
                _run_cycle(hl_cycle)

        self._run_final_response(num_ll_iters, logger, progress_bar)

        pprint(f"Final validation metrics: {self._last_val_metrics}")
        progress_bar.close()

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

