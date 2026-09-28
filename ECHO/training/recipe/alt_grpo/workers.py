"""ALTERNATING-GRPO recipe worker: the shared phase worker with the base actor.

No response term, no weight snapshots. Builds PhaseActorBase directly and states this
recipe's scheduler horizons.
"""
from ...core.phase_actor import PhaseActorBase
from ...core.phase_workers import PhaseWorkerBase


class AltGRPOWorker(PhaseWorkerBase):
    def _build_actor(self):
        return PhaseActorBase(
            config=self.config.actor,
            actor_module=self.actor_module_fsdp,
            phase_optims=self.phase_optims,
            phase_batch_sizes=self.phase_batch_sizes,
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
