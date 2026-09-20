"""No-GPU check for budget-exhaustion handling before GRPO forms its baselines.

Covers, for each actor_rollout_ref.rollout.tools.budget_exhausted_mode:
  1. in_group_zero keeps the sample's uid, so its group baseline drops and the siblings that
     answered gain a positive advantage while it gets a negative one;
  2. excise reproduces the pre-2026-09-19 behaviour exactly -- singleton uid, advantage 0;
  3. none leaves the scorer's -1 in place;
  4. an infrastructure tool failure is still excised and wins the overlap with a budget failure;
  5. the three rates reported to wandb mean the same thing in every mode.

Run: conda activate arpo && PYTHONPATH=$VERL_ROOT:$ECHO_TOP python3 ECHO/training/.budget_exhausted_check.py
"""
import numpy as np, torch
from omegaconf import OmegaConf
from ECHO.training.echo_ray_trainer import RayECHOTrainer
from ECHO.training.echo_core_algos import compute_grpo_outcome_advantage
from verl import DataProto


def make_batch(scores, exhausted, fmt_valid, tool_failed=None, uid="q0"):
    n = len(scores)
    T = 4
    tlr = torch.zeros(n, T)
    tlr[:, -1] = torch.tensor(scores, dtype=torch.float32)
    b = DataProto.from_single_dict({
        "token_level_rewards": tlr,
        "token_level_scores": tlr.clone(),
        "responses": torch.ones(n, T, dtype=torch.long),
        "attention_mask": torch.ones(n, T, dtype=torch.long),
    })
    b.non_tensor_batch["uid"] = np.array([uid] * n, dtype=object)
    b.non_tensor_batch["tool_budget_exhausted"] = np.array(exhausted)
    b.non_tensor_batch["format_valid"] = np.array(fmt_valid)
    b.non_tensor_batch["tool_rollout_failed"] = np.array(
        tool_failed if tool_failed is not None else [False] * n)
    return b


def trainer(mode, skip_tool_failure=False):
    t = RayECHOTrainer.__new__(RayECHOTrainer)
    t.config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"tools": {
        "budget_exhausted_mode": mode,
        "skip_training_on_tool_failure": skip_tool_failure,
        "skip_training_on_budget_exhausted": True,
    }}}})
    return t


def advantages(b):
    resp_mask = torch.ones(b.batch["token_level_rewards"].shape)
    _, _, scalar = compute_grpo_outcome_advantage(
        b.batch["token_level_rewards"], resp_mask, b.non_tensor_batch["uid"],
        norm_adv_by_std_in_grpo=False)
    return scalar.squeeze(-1).numpy()


# Group of 4: two answered (f1 1.0, 0.5), two spent the budget and never answered (-1).
SC, EX, FV = [1.0, 0.5, -1.0, -1.0], [False, False, True, True], [True, True, False, False]

print("=== mode=excise (old behaviour) ===")
t = trainer("excise"); b = make_batch(SC, EX, FV)
m = t._apply_tool_failure_before_grpo(b)
adv = advantages(b)
print(" uids distinct:", len(set(b.non_tensor_batch['uid'])), " rewards:", b.batch['token_level_rewards'].sum(-1).tolist())
print(" advantages:", adv.round(4).tolist(), " metrics:", {k: round(v, 3) for k, v in m.items()})
assert np.allclose(adv[2:], 0.0), "old behaviour must give the failures zero advantage"
assert np.allclose(adv[:2], [0.25, -0.25]), adv[:2]

print("=== mode=in_group_zero (the fix) ===")
t = trainer("in_group_zero"); b = make_batch(SC, EX, FV)
m = t._apply_tool_failure_before_grpo(b)
adv = advantages(b)
print(" uids distinct:", len(set(b.non_tensor_batch['uid'])), " rewards:", b.batch['token_level_rewards'].sum(-1).tolist())
print(" advantages:", adv.round(4).tolist(), " metrics:", {k: round(v, 3) for k, v in m.items()})
assert len(set(b.non_tensor_batch["uid"])) == 1, "demoted samples must stay in their group"
assert np.allclose(adv, [0.625, 0.125, -0.375, -0.375]), adv
assert (adv[2:] < 0).all(), "the failure mode must now carry NEGATIVE advantage"
assert (adv[:2] > 0).all(), "answering siblings must gain against it"

print("=== mode=none ===")
t = trainer("none"); b = make_batch(SC, EX, FV)
m = t._apply_tool_failure_before_grpo(b)
adv = advantages(b)
print(" rewards:", b.batch['token_level_rewards'].sum(-1).tolist(), " advantages:", adv.round(4).tolist())
assert np.allclose(b.batch["token_level_rewards"].sum(-1).numpy(), SC), "none must not touch rewards"

print("=== infra failure still excised, and wins the overlap ===")
t = trainer("in_group_zero", skip_tool_failure=True)
b = make_batch(SC, EX, FV, tool_failed=[False, False, True, False])
m = t._apply_tool_failure_before_grpo(b)
adv = advantages(b)
print(" uids distinct:", len(set(b.non_tensor_batch['uid'])), " advantages:", adv.round(4).tolist())
print(" metrics:", {k: round(v, 3) for k, v in m.items()})
assert len(set(b.non_tensor_batch["uid"])) == 2, "the infra-failed sample must be excised"
assert np.allclose(adv[2], 0.0), "excised sample gets no gradient"
assert adv[3] < 0, "the budget-exhausted one is still demoted"
assert m["policy/budget_demoted_rate"] == 0.25 and m["policy/tool_failure_excised_rate"] == 0.25
assert m["policy/budget_exhausted_invalid_rate"] == 0.5

print("=== excise mode must NOT report budget failures as infra failures ===")
t = trainer("excise"); b = make_batch(SC, EX, FV)
m = t._apply_tool_failure_before_grpo(b)
print(" metrics:", {k: round(v, 3) for k, v in m.items()})
assert m["policy/tool_failure_excised_rate"] == 0.0, m
assert m["policy/budget_exhausted_invalid_rate"] == 0.5 and m["policy/budget_demoted_rate"] == 0.0

print("=== all-exhausted group (collapse state) still degenerate, as GRPO must be ===")
t = trainer("in_group_zero")
b = make_batch([-1.0] * 4, [True] * 4, [False] * 4)
t._apply_tool_failure_before_grpo(b)
print(" advantages:", advantages(b).round(4).tolist())

print("\nALL ASSERTIONS PASSED")
