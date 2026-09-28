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
        """LR-scheduler horizons, computed by AltGRPOTrainer._set_cycle_accounting from
        the cycle shape and the prompt stream length (the worker cannot see the dataset)
        and handed over in phases.<phase>.optim.total_training_steps."""
        horizons = {}
        for phase in ("high_level", "low_level"):
            steps = self.config.phases[phase].optim.get("total_training_steps", None)
            assert steps is not None and int(steps) > 0, (
                f"phases.{phase}.optim.total_training_steps is unset: AltGRPOTrainer."
                "_set_cycle_accounting must run before the workers are built"
            )
            horizons[phase] = int(steps)
        return horizons
