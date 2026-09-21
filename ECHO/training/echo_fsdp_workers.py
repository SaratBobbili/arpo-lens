import itertools
from contextlib import nullcontext

import psutil
import torch
import torch.distributed as dist
from codetiming import Timer
from omegaconf import open_dict

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, load_fsdp_optimizer, offload_fsdp_model_to_cpu, offload_fsdp_optimizer
from verl.utils.import_utils import import_external_libs
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.device import get_torch_device
from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from verl.workers.actor import DataParallelPPOActor
from verl.workers.fsdp_workers import ActorRolloutRefWorker, logger

from .echo_dp_actor import DataParallelECHOActor

PHASE_NAMES = ("high_level", "low_level")


def _build_phase_optimizer(parameters, optim_config, total_steps):
    """One AdamW + LR schedule for a single phase, mirroring the ARPO actor optimizer."""
    # "sgd" exists for the follower phase under the exact K=1 hypergradient. AdamW's
    # first step from reset moments is -lr*g/(|g|+eps), whose exact Jacobian is the
    # diagonal lr*eps/(|g|+eps)^2 -- around 4e-12 at lr=1e-6, eps=1e-8, |g|~5e-2, because
    # m_hat and sqrt(v_hat) cancel to a sign. Differentiating through it correctly gives
    # a response term that is exactly zero to twelve digits. Plain SGD gives D_0 = lr*I,
    # which is both exact and non-degenerate. See echo_response.follower_step_jacobian.
    optimizer_name = str(optim_config.get("optimizer", "adamw")).lower()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            parameters,
            lr=optim_config.lr,
            momentum=float(optim_config.get("momentum", 0.0)),
            weight_decay=optim_config.get("weight_decay", 0.0),
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=optim_config.lr,
            betas=optim_config.get("betas", (0.9, 0.999)),
            weight_decay=optim_config.get("weight_decay", 1e-2),
        )
    else:
        raise ValueError(f"unknown phase optimizer {optimizer_name!r}; use 'adamw' or 'sgd'.")

    num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
    if num_warmup_steps < 0:
        num_warmup_steps = int(optim_config.get("lr_warmup_steps_ratio", 0.0) * total_steps)

    warmup_style = optim_config.get("warmup_style", "constant")
    if warmup_style == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps)
    elif warmup_style == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=optim_config.get("min_lr_ratio", 0.0),
            num_cycles=optim_config.get("num_cycles", 0.5),
        )
    else:
        raise NotImplementedError(f"Warmup style {warmup_style} is not supported")

    return optimizer, scheduler, num_warmup_steps


def _normalize_phase_batch_sizes(phases_cfg, dp_size):
    """Prompt-space mini batches -> per-rank sequence counts, using each phase's own group_size."""
    batch_sizes = {}
    for phase in PHASE_NAMES:
        phase_cfg = phases_cfg[phase]
        mini_batch_size = int(phase_cfg.ppo_mini_batch_size) * int(phase_cfg.group_size) // dp_size
        assert mini_batch_size > 0, (
            f"phases.{phase}: ppo_mini_batch_size {phase_cfg.ppo_mini_batch_size} x group_size "
            f"{phase_cfg.group_size} is smaller than the dp size {dp_size}"
        )
        micro_batch_size_per_gpu = phase_cfg.ppo_micro_batch_size_per_gpu
        if micro_batch_size_per_gpu is not None:
            micro_batch_size_per_gpu = int(micro_batch_size_per_gpu)
            assert mini_batch_size % micro_batch_size_per_gpu == 0, (
                f"phases.{phase}: normalized ppo_mini_batch_size {mini_batch_size} is not divisible by "
                f"ppo_micro_batch_size_per_gpu {micro_batch_size_per_gpu}"
            )
        batch_sizes[phase] = (mini_batch_size, micro_batch_size_per_gpu)
    return batch_sizes


class _PhaseStateShim:
    """Presents both phases' optimizers (or schedulers) as one state_dict-able object.

    FSDPCheckpointManager owns a single optimizer and a single scheduler. It calls
    state_dict() inside the FSDP SHARDED_STATE_DICT context, so delegation must happen
    there rather than through a synthetic param_groups wrapper.
    """

    def __init__(self, members):
        self._members = members

    def state_dict(self):
        return {phase: member.state_dict() for phase, member in self._members.items()}

    def load_state_dict(self, state_dict):
        missing = [phase for phase in self._members if phase not in state_dict]
        if missing:
            raise KeyError(f"checkpoint is missing phases {missing}; got {list(state_dict)}")
        for phase, member in self._members.items():
            member.load_state_dict(state_dict[phase])


class EchoActorRolloutRefWorker(ActorRolloutRefWorker):
    def __init__(self, config, role):
        super().__init__(config, role)
        if not self._is_actor:
            return
        # The parent normalized config.actor.ppo_mini_batch_size with the global
        # rollout.n; ECHO ignores it in favour of per-phase sizes.
        self.phase_batch_sizes = _normalize_phase_batch_sizes(
            self.config.phases,
            self.device_mesh.size() // self.ulysses_sequence_parallel_size,
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        import_external_libs(self.config.model.get("external_lib", None))

        from omegaconf import OmegaConf

        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))

        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            fsdp_config = self.config.actor.fsdp_config if self._is_actor else OmegaConf.create()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            (
                self.actor_module_fsdp,
                _,
                _,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
            )

            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_actor:
                # Two AdamWs over the same FSDP parameters: independent moments and LR
                # schedules. LL runs N_LL times per outer cycle, HL once.
                n_hl = int(self.config.phases.high_level.num_iters)
                n_ll = int(self.config.phases.low_level.num_iters)
                if bool(self.config.phases.get("shared_prompt_stream", False)):
                    e = int(self.config.total_epochs)
                    phase_total_steps = {"high_level": e * n_hl, "low_level": e * n_hl * n_ll}
                else:
                    phase_total_steps = {"high_level": n_hl, "low_level": n_hl * n_ll}
                self.phase_optims = {}
                for phase in PHASE_NAMES:
                    total_steps = phase_total_steps[phase]
                    optimizer, scheduler, num_warmup_steps = _build_phase_optimizer(
                        self.actor_module_fsdp.parameters(),
                        self.config.phases[phase].optim,
                        total_steps,
                    )
                    self.phase_optims[phase] = (optimizer, scheduler)
                    if self.rank == 0:
                        print(f"[{phase}] Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                for optimizer, _ in self.phase_optims.values():
                    offload_fsdp_optimizer(optimizer=optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
                self.config.actor.use_fused_kernels = use_fused_kernels
            self.actor = DataParallelECHOActor(
                config=self.config.actor,
                actor_module=self.actor_module_fsdp,
                phase_optims=self.phase_optims,
                phase_batch_sizes=self.phase_batch_sizes,
                restore_leader_weights=self._restore_leader_weights_local,
                snapshot_adapted_weights=self._snapshot_adapted_weights_local,
                restore_adapted_weights=self._restore_adapted_weights_local,
            )

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout(
                trust_remote_code=self.config.model.get("trust_remote_code", False)
            )

        if self._is_ref:
            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=_PhaseStateShim({phase: opt for phase, (opt, _) in self.phase_optims.items()}),
                lr_scheduler=_PhaseStateShim({phase: sched for phase, (_, sched) in self.phase_optims.items()}),
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_contents=self.config.actor.checkpoint.contents,
            )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        is_lora = data.meta_info.pop("is_lora", False)
        calculate_entropy = data.meta_info.pop("calculate_entropy", True)
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        data = data.to(get_torch_device().current_device())
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            with adapter_ctx:
                log_probs, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=calculate_entropy)

            if calculate_entropy and self.config.rollout.log_prob_use_dynamic_bsz:
                select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
                rearrange_batch = data.select(batch_keys=select_keys).batch
                max_token_len = self.config.rollout.log_prob_max_token_len_per_gpu * self.actor.ulysses_sequence_parallel_size
                _, indices = rearrange_micro_batches(batch=rearrange_batch, max_token_len=max_token_len)
                indices = list(itertools.chain.from_iterable(indices))
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long, device=entropys.device)
                entropys = entropys[revert_indices]

            tensor_payload = {"old_log_probs": log_probs}
            if calculate_entropy:
                tensor_payload["entropys"] = entropys
            output = DataProto.from_dict(
                tensors=tensor_payload,
                meta_info={"temperature": self.config.rollout.temperature},
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        data = data.to(get_torch_device().current_device())

        assert self._is_actor
        actor_optimizer, actor_lr_scheduler = self.phase_optims[data.meta_info["phase"]]
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=actor_optimizer, device_id=get_torch_device().current_device())

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr
            actor_lr_scheduler.step()

            output = DataProto(meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        return output

    # --- ECHO Algorithm 1 round boundaries -------------------------------------------
    #
    # w(x, y) = w_core + P_H x + P_L y with P_H = P_L = I (C.1 permits overlapping
    # ranges). Choosing y_init = 0 makes the follower coordinate x-independent, as Eq. (8)
    # requires, while the realized base stays w_core + x_t. The whole clone/reset/discard
    # subsystem of lines 2 and 14 then collapses to one snapshot per round:
    #
    #   snapshot_leader_weights()   W_x := live_w = w_core + x_t     (line 2, y_0 = 0)
    #   reset_follower_optimizer()                                   (line 2)
    #   ... K follower steps mutate live_w -> w_core + x_t + y_s ...
    #   ... leader phase evaluates the gradient at (x_t, y_K) ...
    #   restore_leader_weights()    live_w := W_x, discarding y_K    (line 14)
    #   ... leader optimizer step applies that gradient to x ...     (line 13)
    #
    # The snapshot lives on CPU: a ~2 s copy each way against a multi-minute round, and it
    # keeps the only added GPU residency down to the response gradient buffer.

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def snapshot_leader_weights(self):
        assert self._is_actor
        params = [p for p in self.actor_module_fsdp.parameters() if p.requires_grad]
        self._leader_weight_snapshot = [p.data.detach().to("cpu", copy=True) for p in params]

    def _restore_leader_weights_local(self):
        """Plain (non-RPC) restore, called by the actor between backward and step."""
        assert getattr(self, "_leader_weight_snapshot", None) is not None, (
            "restore_leader_weights() without a snapshot: the round must open with "
            "snapshot_leader_weights()."
        )
        params = [p for p in self.actor_module_fsdp.parameters() if p.requires_grad]
        assert len(params) == len(self._leader_weight_snapshot)
        for p, saved in zip(params, self._leader_weight_snapshot):
            # copy_ in place, never rebind p.data: FSDP1's flat_param._local_shard aliases
            # this storage, and rebinding would leave the shard pointing at stale memory.
            #
            # Copy host->device directly rather than via saved.to(device), which would
            # allocate a full-size GPU temporary per parameter. non_blocking is wrong here
            # too: the snapshot is pageable, so it buys nothing and would let temporaries
            # pile up until the queued copies drain.
            p.data.copy_(saved)

    def _snapshot_adapted_weights_local(self):
        """W_y := live_w = w_core + x_t + y_K, the adapted point.

        The exact response path evaluates grad_x U_H and g_fol at the adapted point but
        both halves of Eq. (35) at w_0, so the leader iteration has to visit w_1, w_0 and
        w_1 again before its single step. Holding w_1 on the CPU costs one host copy and
        keeps the GPU to a single full-size gradient buffer; holding it on the GPU instead
        is a second multi-GiB resident allocation, which is what fragments the allocator
        into vLLM's wake_up failure.
        """
        assert self._is_actor
        params = [p for p in self.actor_module_fsdp.parameters() if p.requires_grad]
        self._adapted_weight_snapshot = [p.data.detach().to("cpu", copy=True) for p in params]

    def _restore_adapted_weights_local(self):
        assert getattr(self, "_adapted_weight_snapshot", None) is not None, (
            "restore_adapted_weights() without a snapshot."
        )
        params = [p for p in self.actor_module_fsdp.parameters() if p.requires_grad]
        assert len(params) == len(self._adapted_weight_snapshot)
        for p, saved in zip(params, self._adapted_weight_snapshot):
            p.data.copy_(saved)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def restore_leader_weights(self):
        assert self._is_actor
        loaded = False
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
            loaded = True
        self._restore_leader_weights_local()
        if loaded:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset_follower_optimizer(self):
        """Algorithm 1 line 2's "fresh optimizer state" for the adapted tool clone.

        v10 C.3: the algorithm "resets y to y_init and the optimizer state to its initial
        value". Only the AdamW moments are follower state. The LR schedule is a
        hyperparameter schedule spanning the run, so it is deliberately left running --
        harmless under the shipped constant/no-warmup schedule, and a stated choice rather
        than an oversight under cosine or warmup.
        """
        assert self._is_actor
        optimizer, _ = self.phase_optims["low_level"]
        optimizer.state.clear()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def follower_scheduler_state(self):
        """Snapshot the follower LR schedule, so an off-the-record adaptation can undo it."""
        assert self._is_actor
        _, scheduler = self.phase_optims["low_level"]
        return scheduler.state_dict()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def restore_follower_scheduler_state(self, state):
        assert self._is_actor
        _, scheduler = self.phase_optims["low_level"]
        scheduler.load_state_dict(state)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def clear_follower_records(self):
        """Drop the stashed follower-phase rollouts (Algorithm 1 line 14)."""
        assert self._is_actor
        self.actor.clear_follower_records()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.load_checkpoint(local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            for optimizer, _ in self.phase_optims.values():
                offload_fsdp_optimizer(optimizer=optimizer)
