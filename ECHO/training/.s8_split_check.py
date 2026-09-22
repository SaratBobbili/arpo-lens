"""No-GPU check for the ALTERNATING-GRPO / HYPERGRADIENT code split.

Covers what can be checked without a GPU or a real run:
  1. both class trees import and the inheritance runs the right way;
  2. the ALTERNATING-GRPO classes define every hook the HYPERGRADIENT ones override,
     and none of them reference phases.response;
  3. phases.algorithm picks the tree, and is validated;
  4. phases.prompt_batch_size replaces the N_HL*(N_LL+1) round equation, and the batch
     arithmetic works out for both profiles;
  5. the ALTERNATING-GRPO high-level iteration reuses the low-level prompt batch while
     HYPERGRADIENT pulls a fresh one.
"""

import ast
import os
import sys

ECHO_TOP = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VERL_ROOT = os.environ["VERL_ROOT"]
sys.path[:0] = [VERL_ROOT, ECHO_TOP]

from hydra import compose, initialize_config_dir

from training.alt_dp_actor import DataParallelPhaseActor
from training.alt_fsdp_workers import PhaseActorRolloutRefWorker
from training.alt_ray_trainer import RayAlternatingGRPOTrainer
from training.echo_dp_actor import DataParallelECHOActor
from training.echo_fsdp_workers import EchoActorRolloutRefWorker
from training.echo_ray_trainer import RayECHOTrainer
from training.main_echo import _hypergradient

CONFIG_DIR = os.path.join(ECHO_TOP, "training", "config")
TRAINING = os.path.join(ECHO_TOP, "training")


def _compose(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="echo_trainer", overrides=["algorithm.adv_estimator=grpo", *overrides])


# --- 1. inheritance direction --------------------------------------------------------
print("=== 1. inheritance ===")
for sub, base in ((RayECHOTrainer, RayAlternatingGRPOTrainer),
                  (DataParallelECHOActor, DataParallelPhaseActor),
                  (EchoActorRolloutRefWorker, PhaseActorRolloutRefWorker)):
    assert issubclass(sub, base), f"{sub.__name__} must subclass {base.__name__}"
    assert not issubclass(base, sub), f"{base.__name__} must not know about {sub.__name__}"
    print(f"  {sub.__name__} <- {base.__name__}")

# --- 2. hooks are defined in the base and overridden in the subclass -----------------
print("=== 2. hooks ===")
TRAINER_HOOKS = ["_validate_extra", "_rollout_meta_info", "_scorer_metric_kwargs",
                 "_phase_meta_info", "_open_round", "_close_cycle", "_after_fit",
                 "_next_batch_dict"]
ACTOR_HOOKS = ["_update_begin", "_extra_select_keys", "_on_batch_selected",
               "_compute_response_gradient", "_should_zero_grad",
               "_on_response_grad_consumed", "_before_optimizer_step",
               "_after_optimizer_step", "_after_update"]
for base, sub, hooks in ((RayAlternatingGRPOTrainer, RayECHOTrainer, TRAINER_HOOKS),
                         (DataParallelPhaseActor, DataParallelECHOActor, ACTOR_HOOKS)):
    for h in hooks:
        assert h in vars(base), f"{base.__name__} is missing hook {h}"
        assert h in vars(sub), f"{sub.__name__} does not override {h}"
    print(f"  {base.__name__}: {len(hooks)} hooks, all overridden by {sub.__name__}")
assert "_build_actor" in vars(PhaseActorRolloutRefWorker)
assert "_build_actor" in vars(EchoActorRolloutRefWorker)
print("  PhaseActorRolloutRefWorker._build_actor overridden by EchoActorRolloutRefWorker")

# the ALTERNATING-GRPO tree must not reach into the response path
BANNED = ("phases.response", "_response_cfg(", "_response_enabled(", "_follower_return_enabled(",
          "snapshot_leader_weights", "_follower_records", "_follower_step_weights",
          "_exact_response_gradient", "_hessian_vector_product")
for fname in ("alt_ray_trainer.py", "alt_dp_actor.py", "alt_fsdp_workers.py"):
    src = open(os.path.join(TRAINING, fname)).read()
    code = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("#"))
    hits = [b for b in BANNED if b in code]
    assert not hits, f"{fname} references the response path: {hits}"
    print(f"  {fname}: clean of {len(BANNED)} response-path symbols")

# --- 3. phases.algorithm selects the tree -------------------------------------------
print("=== 3. phases.algorithm ===")
cfg = _compose([])
assert cfg.phases.algorithm == "hypergradient", cfg.phases.algorithm
assert _hypergradient(cfg) is True
assert _hypergradient(_compose(["phases.algorithm=alternating"])) is False
try:
    _hypergradient(_compose(["phases.algorithm=nonsense"]))
except AssertionError as e:
    assert "phases.algorithm must be one of" in str(e), str(e)
    print("  rejects an unknown algorithm:", str(e).split(",")[0])
else:
    raise SystemExit("FAIL: phases.algorithm=nonsense was accepted")
print("  default hypergradient; alternating selects the base tree")

# --- 4. the round equation is gone; prompt_batch_size drives the batch --------------
print("=== 4. batch arithmetic ===")
alt_src = open(os.path.join(TRAINING, "alt_ray_trainer.py")).read()
# The round equation may still count optimizer steps (total_training_steps); what it must
# no longer do is size the prompt batch.
batch_assigns = [l.strip() for l in alt_src.split("\n") if "batch_size = " in l and "mini" not in l
                 and "micro" not in l and "val_batch_size" not in l]
for l in batch_assigns:
    assert "num_iters" not in l and "steps_per_epoch" not in l, \
        f"prompt batch still derived from the round structure: {l}"
assert any("self.config.phases.prompt_batch_size" in l for l in batch_assigns), \
    f"no batch_size assignment reads the explicit knob; saw {batch_assigns}"
print(f"  batch_size assignments read the explicit knob only ({len(batch_assigns)} sites)")
assert int(cfg.phases.prompt_batch_size) == 128
assert int(cfg.actor_rollout_ref.phases.prompt_batch_size) == 128, "worker mirror missing the key"

WORLD, GROUP = 8, 16
for label, batch, mini in (("HYPERGRADIENT", 128, 128), ("ALTERNATING-GRPO", 128, 16)):
    assert batch % WORLD == 0
    steps = batch // mini
    assert batch % mini == 0, f"{label}: {batch}/{mini} leaves a ragged mini-batch"
    seq_per_rank = mini * GROUP // WORLD
    print(f"  {label:<17} batch={batch} mini={mini} -> {steps} optimizer step(s), "
          f"{seq_per_rank} seq/rank/step")
    if label == "HYPERGRADIENT":
        assert steps == 1, "C.3 needs exactly one optimizer step per phase iteration"
    else:
        assert steps == 8 and seq_per_rank == 32, "should match ARPO's 128/16/16"

# --- 5. prompt reuse differs between the two ----------------------------------------
print("=== 5. prompt reuse ===")

class _FakeAlt(RayAlternatingGRPOTrainer):
    def __init__(self):
        self._shared_prompt_stream = True
        self._phase_iters = {}
        self._phase_dataloaders = {"low_level": None, "high_level": None}
        self._cycle_batch_dict = None
        self._n = 0
    def _pull_batch_dict(self, phase_name):
        self._n += 1
        return {"batch": self._n}

class _FakeEcho(RayECHOTrainer):
    __init__ = _FakeAlt.__init__
    _pull_batch_dict = _FakeAlt._pull_batch_dict

a = _FakeAlt()
ll1, ll2, hl = a._next_batch_dict("low_level"), a._next_batch_dict("low_level"), a._next_batch_dict("high_level")
assert ll1 != ll2, "each low-level iteration must pull its own batch"
assert hl == ll2, f"ALTERNATING-GRPO high-level should reuse the last low-level batch, got {hl} vs {ll2}"
print(f"  ALTERNATING-GRPO: LL={ll1['batch']}, LL={ll2['batch']}, HL={hl['batch']}  (HL reuses the last LL)")

e = _FakeEcho()
ell1, ell2, ehl = e._next_batch_dict("low_level"), e._next_batch_dict("low_level"), e._next_batch_dict("high_level")
assert len({ell1["batch"], ell2["batch"], ehl["batch"]}) == 3, \
    "HYPERGRADIENT must draw distinct query groups (C.3/C.4)"
print(f"  HYPERGRADIENT   : LL={ell1['batch']}, LL={ell2['batch']}, HL={ehl['batch']}  (all distinct)")

print("\nALL CHECKS PASSED")
