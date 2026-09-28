"""Shared two-phase trainer base: everything a role-masked two-phase GRPO run needs
EXCEPT the algorithm loop.

Owns the metric layout, logging_data dumps, prompt streams and samplers, the
rollout -> reward -> advantage -> update body of one phase iteration, tool-failure and
budget-exhausted masking, and best-checkpoint sync. It has no fit(), no cycle
structure, no step accounting and no knowledge of phases.response: each recipe under
training/recipe/<name>/trainer.py subclasses this and owns those (see the extension
points at the bottom of the class).
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
from .core_algos import agg_loss, apply_kl_penalty, compute_advantage
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.metric import reduce_metrics


# core/ sits three levels below the repo root (arpo-lens/ECHO/training/core).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VERL_TO_HF_SCRIPT = os.path.join(_REPO_ROOT, "ARPO", "merge_ckpt", "convert_checkpoint_from_verl_to_hf.py")



class _PhaseDataloaders:
    """Per-phase prompt loaders behind the single state_dict API the base checkpointer uses."""

    def __init__(self, loaders: dict):
        self.loaders = loaders

    def state_dict(self) -> dict:
        return {phase: loader.state_dict() for phase, loader in self.loaders.items()}

    def load_state_dict(self, state_dict: dict) -> None:
        for phase, loader in self.loaders.items():
            loader.load_state_dict(state_dict[phase])


class PhaseTrainerBase(RayPPOTrainer):
    """Two-phase role-masked GRPO machinery. Each phase owns its optimizer and batch
    sizes; the recipe subclass owns the loop that orders the phase iterations."""

    # Order is load-bearing: in legacy (non-shared) mode the per-phase sampler seed
    # offset in _create_dataloader is the index in this tuple.
    _PHASE_NAMES = ("low_level", "high_level")
    _PHASE_MASK_KEYS = {"low_level": "low_level_loss_mask", "high_level": "high_level_loss_mask"}

    # ---- metric layout (2026-09-27) --------------------------------------------------
    # Keys are namespaced by what the number is computed on, not by which phase's step it
    # landed in. The policy is one object and every rollout is a full trajectory from it,
    # so trajectory- and policy-level numbers are ONE dense series each, logged every step
    # with train/phase (0 follower, 1 leader) as the marker:
    #   reward/*   reward_mean, f1_mean, whole-schema format_valid_rate, fail_* buckets,
    #              in-group std / zero-std fraction (raw and _post the GRPO adjustment)
    #   rollout/*  response length, tool calls, budget exhaustion, no-tool rate, infra excision
    #   policy/*   entropy of the pre-update policy on the two token populations
    # Only numbers whose value depends on the phase's mask, gate or advantage rule carry
    # the phase prefix:
    #   {phase}/reward_mean, {phase}/gate_pass_rate, {phase}/budget_*   (gate-dependent)
    #   {phase}/actor/*      loss, grad norm, clip stats, advantage stats from the update
    #   high_level/response/* and high_level/aho/*   leader-only estimator diagnostics
    #                                                (emitted by the echo / aho recipes)
    # val-core/*, val-aux/*, perf/*, timing_s/* and train/* are shared as before.
    _ACTOR_DROPPED = frozenset({"entropy_reg_coef", "kl_coef", "aho_tau"})
    _RESPONSE_RENAMES = {
        "response_norm": "norm",
        "direct_norm": "direct_norm",
        "response_to_direct_ratio": "ratio",
        "response_direct_cosine": "cosine",
    }

    @classmethod
    def _route_actor_metrics(cls, actor_metrics: dict, phase_name: str) -> dict:
        """Route the actor's ``actor/*`` numbers into the layout above."""
        out: dict = {}
        for key, value in actor_metrics.items():
            if key.startswith("perf/"):
                out[key] = value
                continue
            if not key.startswith("actor/"):
                continue
            name = key[len("actor/"):]
            if name in cls._ACTOR_DROPPED:
                continue
            if name.startswith("aho_"):
                out[f"{phase_name}/aho/{name[len('aho_'):]}"] = value
            elif name in cls._RESPONSE_RENAMES:
                out[f"{phase_name}/response/{cls._RESPONSE_RENAMES[name]}"] = value
            elif name.startswith(("response_", "follower_", "hvp_")):
                sub = name[len("response_"):] if name.startswith("response_") else name
                out[f"{phase_name}/response/{sub}"] = value
            else:
                out[f"{phase_name}/actor/{name}"] = value
        return out

    def _phase_cfg(self, phase_name: str):
        return self.config.phases[phase_name]

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
        return PhaseTrainerBase._entropy_reg_coeff(phase_cfg) > 0.0

    def _init_logging_data(self) -> None:
        # One JSONL per logged metric under {default_local_dir}/logging_data/<key>.jsonl,
        # mirroring the wandb key exactly. Read by training/analysis/plot_training_log.py.
        self._logging_data_root = os.path.join(self.config.trainer.default_local_dir, "logging_data")
        self._prev_logged_values: dict[str, float] = {}
        os.makedirs(self._logging_data_root, exist_ok=True)

    def _dump_logging_data(self, metrics: dict) -> None:
        # Append one line per metric file: {"step", "value", "gain"}. `gain` is the
        # difference vs the previous dumped step for the same key, or null on the first
        # dump. perf/ and timing keys are skipped; everything else is written.
        for key, raw in metrics.items():
            if key.startswith(("perf/", "timing_s/")):
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            path = os.path.join(self._logging_data_root, f"{key}.jsonl")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            prev = self._prev_logged_values.get(key)
            gain = None if prev is None else value - prev
            self._prev_logged_values[key] = value
            with open(path, "a") as f:
                f.write(json.dumps({"step": self.global_steps, "value": value, "gain": gain}) + "\n")

    @staticmethod
    def _build_scorer_metrics(reward_extra_info: dict, phase_name: str) -> dict:
        metrics: dict = {}
        if not reward_extra_info:
            return metrics

        if "f1_score" in reward_extra_info:
            f1_scores = torch.tensor(reward_extra_info["f1_score"], dtype=torch.float32)
            metrics["reward/f1_mean"] = f1_scores.mean().item()

        if "format_valid" in reward_extra_info:
            # Whole-schema bit, the same quantity on either phase's step.
            fmt_valid = torch.tensor(reward_extra_info["format_valid"], dtype=torch.float32)
            metrics["reward/format_valid_rate"] = fmt_valid.mean().item()

        if "phase_format_valid" in reward_extra_info:
            # The gate this phase's scorer applied (its own tags only), hence phase-prefixed.
            phase_valid = torch.tensor(reward_extra_info["phase_format_valid"], dtype=torch.float32)
            metrics[f"{phase_name}/gate_pass_rate"] = phase_valid.mean().item()

        if "no_tool_calls" in reward_extra_info:
            no_tool = torch.tensor(reward_extra_info["no_tool_calls"], dtype=torch.float32)
            metrics["rollout/no_tool_rate"] = no_tool.mean().item()

        if "follower_score" in reward_extra_info:
            # The leader batch scored under the follower's schema rows (AHO estimator, A_L).
            follower_scores = torch.tensor(reward_extra_info["follower_score"], dtype=torch.float32)
            metrics[f"{phase_name}/aho/follower_score_mean"] = follower_scores.mean().item()

        # A single bad_format_rate scalar hid a reward bug for six runs: ~86% of step-1 format
        # failures were tool-budget truncations scored -1, not real schema errors. Bucket the
        # whole-schema violations (format_issues, every violation with its owner) so the next
        # one is visible in one glance; one bucket per sample, by priority.
        issues = reward_extra_info.get("format_issues")
        if issues is None:
            issues = reward_extra_info.get("reason")
        if issues:
            total = float(len(issues))
            buckets = {"answer_count_0": 0, "unclosed_tag": 0, "no_boxed": 0, "other": 0}
            for raw in issues:
                text = str(raw)
                if not text or text == "format is correct" or not (
                    "bad format" in text or ": " in text or "boxed" in text or "cannot extract answer" in text
                ):
                    continue
                if "answer_count=0" in text:
                    buckets["answer_count_0"] += 1
                elif "is not closed" in text:
                    buckets["unclosed_tag"] += 1
                elif "boxed" in text:
                    buckets["no_boxed"] += 1
                else:
                    buckets["other"] += 1
            for name, count in buckets.items():
                metrics[f"reward/fail_{name}"] = count / total

        return metrics

    @staticmethod
    def _rollout_behavior_metrics(phase_batch: DataProto) -> dict:
        """Tool-use metrics computed from per-sample arrays, not from the rollout's counters.

        tools/* counters ride in meta_info, and DataProto.concat keeps meta_info from rank 0
        only (verl/protocol.py:710), so they are not logged at all; these are the
        world-size-correct versions.
        """
        out: dict = {}
        calls = phase_batch.non_tensor_batch.get("tool_calls_made")
        if calls is not None:
            calls_t = torch.tensor(np.asarray(calls, dtype=np.float32))
            out["rollout/tool_calls_per_traj_mean"] = calls_t.mean().item()
        exhausted = phase_batch.non_tensor_batch.get("tool_budget_exhausted")
        if exhausted is not None:
            out["rollout/budget_exhausted_rate"] = float(np.asarray(exhausted, dtype=np.float32).mean())
        return out

    @staticmethod
    def _policy_from_data_metrics(data_metrics: dict, phase_batch: DataProto) -> dict:
        # advantage stats are NOT read here: in entropy/aepo mode the actor substitutes or
        # rescales the batch advantages, so the driver-side tensor is not what was used.
        # They come from inside the update instead ({phase}/actor/advantage_*).
        out: dict = {}
        if "response_length/mean" in data_metrics:
            out["rollout/response_length_mean"] = data_metrics["response_length/mean"]
        if "response_length/clip_ratio" in data_metrics:
            out["rollout/response_length_clip_ratio"] = data_metrics["response_length/clip_ratio"]
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

    def _budget_exhausted_mode(self) -> str:
        """How to treat a rollout that spent its tool budget and still failed the format gate.

        ``in_group_zero`` (default) scores it 0 but leaves it in its prompt group, so the group
        baseline drops and the siblings that did answer gain a positive advantage against it.
        ``excise`` is the pre-2026-09-19 behaviour: 0 plus a fresh uid, i.e. a singleton group,
        whose mean is pinned to 0 by ``compute_grpo_outcome_advantage``, so the sample's
        advantage is exactly 0 and it contributes no gradient at all. ``none`` leaves the
        scorer's -1 format penalty untouched.
        """
        cfg = self._echo_rollout_tools_cfg()
        mode = cfg.get("budget_exhausted_mode", None)
        if mode is None:
            # Deprecated boolean, honoured so a resume from an older launch config still runs.
            return "excise" if bool(cfg.get("skip_training_on_budget_exhausted", True)) else "none"
        return str(mode)

    def _budget_exhausted_and_invalid(self, phase_batch: DataProto) -> np.ndarray | None:
        """Samples that spent the tool budget and still failed the format gate.

        A rollout that exhausts its tool budget gets a masked notice and a final turn to answer
        (see vllm_rollout_echo). One that answers legally is real data and is scored normally --
        a wrong answer earns 0, which is the honest signal. This selects only the residual: the
        ones that still produced nothing scoreable after that final turn.
        """
        if self._budget_exhausted_mode() == "none":
            return None
        exhausted = phase_batch.non_tensor_batch.get("tool_budget_exhausted")
        if exhausted is None:
            return None
        # "Nothing scoreable" for THIS phase: the phase-owned gate, not the whole schema.
        fmt_valid = phase_batch.non_tensor_batch.get("phase_format_valid")
        if fmt_valid is None:
            fmt_valid = phase_batch.non_tensor_batch.get("format_valid")
        if fmt_valid is None:
            return None
        flags = np.asarray(exhausted, dtype=np.bool_) & ~np.asarray(fmt_valid, dtype=np.bool_)
        return flags if np.any(flags) else None

    def _apply_tool_failure_before_grpo(self, phase_batch: DataProto) -> dict:
        """Neutralise the two kinds of unusable rollout before GRPO forms its baselines.

        They are not the same failure and must not be handled the same way:

        * ``tool_rollout_failed`` is infrastructure -- a retry budget spent on a search API that
          timed out. The policy did nothing wrong, so the sample is *excised*: fresh uid, hence
          a singleton group with mean 0, hence advantage 0 and no gradient.
        * Budget-exhausted-and-unanswered is the policy's own doing. Excising that too (the
          behaviour up to 2026-09-19) made the single most common failure mode invisible to the
          optimizer: every trajectory that spent all three calls and never emitted <answer> was
          deleted from its group, so "keep calling tools" was never compared against "answer
          now". Nothing opposed the drift, which made it an absorbing state --
          echo7B_sft1e8_rl1e-6-r3 entered it around step 89 and never left
          (budget_exhausted_rate 0.99, format_valid_rate 0.03, reward_mean -0.97).

        Under ``in_group_zero`` the sample is scored 0 but *keeps its uid*, so it stays in the
        group. With norm_adv_by_std_in_grpo=false the advantage is ``r_i - mean``, so the sample
        earns a negative advantage and its answering siblings earn a positive one. The pressure
        is group-relative and therefore self-scaling: it fades as a group stops exhausting its
        budget, unlike the flat -1 this replaced.
        """
        mode = self._budget_exhausted_mode()
        out: dict = {}

        infra = None
        if bool(self._echo_rollout_tools_cfg().get("skip_training_on_tool_failure", False)):
            tool_failed = phase_batch.non_tensor_batch.get("tool_rollout_failed")
            if tool_failed is not None and np.any(tool_failed):
                infra = np.asarray(tool_failed, dtype=np.bool_)

        demote = self._budget_exhausted_and_invalid(phase_batch)
        # Both counted before the mode folds anything together, so each number means the same
        # thing in every mode: infra failures excised, and rollouts that hit the budget failure.
        phase = (phase_batch.meta_info or {}).get("phase") or "policy"
        out["rollout/tool_failure_excised_rate"] = float(infra.mean()) if infra is not None else 0.0
        out[f"{phase}/budget_failed_rate"] = float(demote.mean()) if demote is not None else 0.0

        excise = infra
        if demote is not None and mode == "excise":
            excise = demote if excise is None else (excise | demote)
            demote = None
        if demote is not None and excise is not None:
            # An infrastructure failure is not the policy's fault even when the budget also ran
            # out, so excision wins the overlap and the sample stays out of the baseline.
            demote = demote & ~excise
            if not np.any(demote):
                demote = None

        # Of the above, how much stayed in its group and so actually carries counter-pressure.
        # Zero in every mode but in_group_zero; the gap against the rate above is the bug closed.
        out[f"{phase}/budget_demoted_rate"] = float(demote.mean()) if demote is not None else 0.0

        if excise is None and demote is None:
            return out

        if excise is None:
            zeroed = demote
        elif demote is None:
            zeroed = excise
        else:
            zeroed = excise | demote

        dev = phase_batch.batch["token_level_rewards"].device
        failed = torch.tensor(zeroed, device=dev)
        phase_batch.batch["token_level_rewards"][failed] = 0
        phase_batch.batch["token_level_scores"][failed] = 0

        if excise is not None:
            # Fresh uid per excised sample: a singleton group has mean 0, so its advantage is 0
            # and it cannot shift the baseline of the group it came from. Demoted samples are
            # deliberately NOT given one -- shifting that baseline is the entire point.
            uids = phase_batch.non_tensor_batch["uid"].copy()
            for i in np.flatnonzero(excise):
                uids[i] = str(uuid.uuid4())
            phase_batch.non_tensor_batch["uid"] = uids
        return out

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

    def _validate_budget_exhausted_config(self) -> None:
        allowed = {"in_group_zero", "excise", "none"}
        mode = self._budget_exhausted_mode()
        assert mode in allowed, (
            f"actor_rollout_ref.rollout.tools.budget_exhausted_mode must be one of "
            f"{sorted(allowed)}, got {mode!r}."
        )
        if self._echo_rollout_tools_cfg().get("budget_exhausted_mode", None) is None:
            print(
                "[echo] actor_rollout_ref.rollout.tools.budget_exhausted_mode is unset; falling "
                f"back to the deprecated skip_training_on_budget_exhausted flag -> {mode!r}. "
                "Set budget_exhausted_mode explicitly (default: in_group_zero)."
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

        Shared mode: one dataloader sized by phases.prompt_batch_size.
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
        self._validate_budget_exhausted_config()
        self._validate_phase_rollout_configs()

        dataset_size = len(self.train_dataset)
        world_size = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        num_workers = self.config.data.get("dataloader_num_workers", 8)
        base_seed = self.config.data.get("seed", 1)

        self._shared_prompt_stream = bool(self.config.phases.get("shared_prompt_stream", False))
        self._phase_dataloaders = {}
        self._phase_iters = {}
        # Prompt batch per phase iteration; the response term requires the phase's
        # ppo_mini_batch_size to equal it, so one iteration is one optimizer step.
        self._phase_prompt_batch_sizes = {}

        if self._shared_prompt_stream:
            batch_size = int(self.config.phases.prompt_batch_size)
            assert batch_size % world_size == 0 and batch_size <= dataset_size, (
                f"phases.prompt_batch_size={batch_size} must be a multiple of the rank "
                f"count ({world_size}) and at most the dataset size ({dataset_size})."
            )
            batches_per_pass = dataset_size // batch_size
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
                f"[shared] batches_per_pass={batches_per_pass}, prompt_batch_size={batch_size}, "
                f"batches_per_dataset_pass={len(shared_loader)}"
            )
        else:
            for seed_offset, phase_name in enumerate(self._PHASE_NAMES):
                batch_size = int(self.config.phases.prompt_batch_size)
                assert batch_size % world_size == 0 and batch_size <= dataset_size, (
                    f"phases.prompt_batch_size={batch_size} must be a multiple of the "
                    f"rank count ({world_size}) and at most {dataset_size}."
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
                    f"[{phase_name}] prompt_batch_size={batch_size}, "
                    f"batches_per_dataset_pass={len(self._phase_dataloaders[phase_name])}"
                )

        # Needs the derived prompt batch sizes, so it runs after the loaders are built.
        self._validate_extra()

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



    def _pull_batch_dict(self, phase_name: str):
        """One batch off this phase's loader, restarting the iterator at exhaustion."""
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
        phase_batch.meta_info.update(self._rollout_meta_info(phase_name))

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
            # The rollout's tools/* counters ride in meta_info, which DataProto.concat keeps
            # from rank 0 only; they are not logged. See _rollout_behavior_metrics.

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
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            # Entropy of the ONE pre-update policy on its two token populations, logged on
            # every step whichever phase is training: reasoning = think/answer tokens,
            # tool = tool/search/python tokens. The all-token mean is not logged: it is
            # dominated by the injected <result> spans (retrieved text, ~1.2 nats vs ~0.1)
            # and tracks search-cache coverage rather than the policy.
            for pop, mask_key in (("reasoning", "high_level_loss_mask"), ("tool", "low_level_loss_mask")):
                if mask_key in phase_batch.batch:
                    metrics[f"policy/entropy_{pop}"] = agg_loss(
                        loss_mat=entropys, loss_mask=phase_batch.batch[mask_key], loss_agg_mode=loss_agg_mode
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
        reward_mean = reward_tensor.to(torch.float32).sum(dim=-1).mean().detach().item()
        metrics["reward/reward_mean"] = reward_mean          # one dense series across phases
        metrics[f"{phase_name}/reward_mean"] = reward_mean   # bookkeeping copy under the phase
        phase_batch.batch["token_level_scores"] = reward_tensor
        if phase_reward_extra_infos_dict:
            phase_batch.non_tensor_batch.update({k: np.array(v) for k, v in phase_reward_extra_infos_dict.items()})
            metrics.update(self._build_scorer_metrics(phase_reward_extra_infos_dict, phase_name))
            if "phase_format_valid" in phase_reward_extra_infos_dict:
                # As a batch tensor so it survives micro-batching: the actor's entropy
                # advantage sets H_t = -1 on the tokens of a sample that failed this phase's
                # schema rows before normalising (see core/phase_actor.py).
                phase_batch.batch["phase_format_valid"] = torch.tensor(
                    np.asarray(phase_reward_extra_infos_dict["phase_format_valid"], dtype=np.float32)
                )
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
                metrics["reward/in_group_std"] = in_group_std
                metrics["reward/group_zero_std_frac"] = zero_std_frac
                metrics.update(self._apply_tool_failure_before_grpo(phase_batch))
                # The pair above is measured on the raw scores; this one is measured on what
                # GRPO actually sees. They diverge exactly where the reward adjustment bites,
                # so *_post is the number that says whether a group still carries gradient.
                post_std, post_zero_std_frac = self._compute_in_group_reward_std(phase_batch)
                metrics["reward/in_group_std_post"] = post_std
                metrics["reward/group_zero_std_frac_post"] = post_zero_std_frac

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
                # HYPERGRADIENT seam: the subclass may derive extra per-trajectory / per-token
                # tensors from the scored, advantaged batch before it reaches the actor.
                metrics.update(self._after_advantage(phase_batch, phase_name))
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
                    phase_batch.meta_info.update(self._phase_meta_info(phase_name))
                    if self._uses_entropy_regularizer(phase_cfg):
                        phase_batch.meta_info["entropy_coeff_override"] = self._entropy_reg_coeff(phase_cfg)
                        phase_batch.meta_info["entropy_loss_mask_key"] = "entropy_reg_loss_mask"
                    actor_output = self.actor_rollout_wg.update_actor(phase_batch)
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(self._route_actor_metrics(actor_output_metrics, phase_name))

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
                hl_done = self._hl_updates_done(hl_cycle)
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

                    if bool(self.config.trainer.get("save_best_checkpoint", False)):
                        current_metric_key, current_metric_value = self._resolve_best_metric_from_val(val_metrics)
                        selector = self._canonical_best_metric_selector(self.config.trainer.best_checkpoint_metric)
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
                        metrics["train/best_checkpoint_value"] = self._best_metric_value
                        metrics["train/best_checkpoint_step"] = float(self._best_metric_step)

                if not saved_checkpoint_this_step and (is_last_step or save_due):
                    with _timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

        metrics.update(
            {
                "train/global_step": self.global_steps,
                "train/hl_cycle": hl_cycle,
                # Which phase this step trained; the shared series above are one policy's
                # trajectory and this is the only marker of the alternation.
                "train/phase": 1.0 if phase_name == "high_level" else 0.0,
            }
        )
        if self._shared_prompt_stream:
            metrics["train/epoch"] = self._current_epoch
        timing_metrics = compute_timing_metrics(batch=phase_batch, timing_raw=timing_raw)
        if "timing_s/step" in timing_metrics:
            metrics["timing_s/step"] = timing_metrics["timing_s/step"]
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=phase_batch, timing_raw=timing_raw, n_gpus=n_gpus))

        logger.log(data=metrics, step=self.global_steps)
        self._dump_logging_data(metrics)

        progress_bar.update(1)
        self.global_steps += 1




    # --- recipe extension points ---------------------------------------------------
    # The algorithm loop lives ONLY in recipes: fit(), the cycle structure, which
    # prompts a phase iteration draws, and the step accounting. Nothing in this class
    # may reference phases.response or assume a cycle shape.

    def fit(self):
        """Owned by the recipe (training/recipe/<name>/trainer.py).

        verl's RayPPOTrainer.fit is the single-phase PPO loop and must never run here.
        A recipe's fit must, before its first _run_phase_iteration, set
        self.global_steps, self._best_metric_value / _step / _key,
        self._last_val_metrics and self._current_epoch, and call
        self._init_logging_data(); _run_phase_iteration reads all of them. The recipe's
        _create_dataloader override must set self.total_training_steps.
        """
        raise NotImplementedError("each recipe owns its training loop (fit)")

    def _next_batch_dict(self, phase_name: str):
        """Which prompt batch the next `phase_name` iteration rolls out. Recipe-owned:
        alt_grpo re-rolls the low-level prompts for the high-level step, Algorithm 1
        draws distinct query groups. _pull_batch_dict draws from the stream."""
        raise NotImplementedError("each recipe decides which prompts a phase iteration draws")

    def _hl_updates_done(self, hl_cycle: int) -> int:
        """How many high-level updates will have completed once the current one
        finishes; save_freq / test_freq count these. Depends on the cycle shape."""
        raise NotImplementedError("each recipe owns its step accounting")

    def _validate_extra(self) -> None:
        """Extra config validation, run once the prompt batch sizes are known."""

    def _rollout_meta_info(self, phase_name: str) -> dict:
        """meta_info added to a phase batch before generation."""
        return {}

    def _phase_meta_info(self, phase_name: str) -> dict:
        """meta_info added to a phase batch before update_actor."""
        return {}

    def _after_advantage(self, phase_batch: DataProto, phase_name: str) -> dict:
        """Called right after compute_advantage, before the critic/actor update. May add
        batch tensors in place; returns driver-side (batch-correct) metrics."""
        return {}

