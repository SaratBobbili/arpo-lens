"""AHO recipe worker: Algorithm 1 leader-weight snapshots on top of the shared phase worker.

The shared two-phase FSDP worker is PhaseWorkerBase (core/phase_workers.py); this
class builds the AhoActor, states the recipe's scheduler horizons, and adds the RPCs
the round structure needs (snapshot / restore W_x, reset the follower optimizer).
AHO keeps no follower records and no adapted-weight snapshot.
"""
import torch
import torch.distributed as dist

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.workers.fsdp_workers import logger

from ...core.phase_workers import PhaseWorkerBase
from .actor import AhoActor


class AhoWorker(PhaseWorkerBase):
    def _build_actor(self):
        return AhoActor(
            config=self.config.actor,
            actor_module=self.actor_module_fsdp,
            phase_optims=self.phase_optims,
            phase_batch_sizes=self.phase_batch_sizes,
            restore_leader_weights=self._restore_leader_weights_local,
        )

    def _phase_total_steps(self) -> dict:
        """LR-scheduler horizons for this recipe's cycle: LL runs N_LL times per outer
        cycle, HL once, for N_HL cycles per epoch."""
        n_hl = int(self.config.phases.high_level.num_iters)
        n_ll = int(self.config.phases.low_level.num_iters)
        if bool(self.config.phases.get("shared_prompt_stream", False)):
            e = int(self.config.total_epochs)
            return {"high_level": e * n_hl, "low_level": e * n_hl * n_ll}
        return {"high_level": n_hl, "low_level": n_hl * n_ll}

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

