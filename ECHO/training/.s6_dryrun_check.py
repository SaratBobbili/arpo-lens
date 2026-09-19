"""No-GPU check for the response-aware leader update (ECHO Algorithm 1).

Covers what can be checked without a GPU or a real run:
  1. the phases.response block composes, defaults on, and train.sh overrides reach it;
  2. lambda_ent H_tool is flag-gated and off by default, so u_L is pure R_L;
  3. the one-optimizer-step-per-iteration rule is enforced (App. C.3);
  4. R_L scores tool validity off the per-sample counters, not the rank-0 tools/* metrics;
  5. the gradient algebra in echo_response reproduces sum_b c_b S_b exactly.
"""

import os
import sys

ECHO_TOP = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VERL_ROOT = os.environ["VERL_ROOT"]
sys.path[:0] = [VERL_ROOT, ECHO_TOP]

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from torch import nn

from training import echo_response
from training.echo_ray_trainer import RayECHOTrainer
from verl.utils.reward_score.deep_research_echo import compute_tool_score

CONFIG_DIR = os.path.join(ECHO_TOP, "training", "config")


def _compose(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="echo_trainer", overrides=["algorithm.adv_estimator=grpo", *overrides])


# --- 1. the response block exists, is on by default, and is overridable --------------
cfg = _compose([])
assert cfg.phases.response.enabled is True, "the Algorithm 1 round structure should default on"
# The g_resp term is gated SEPARATELY and defaults off: its terminal seed is annihilated
# by a mask composition, so the sweep costs 2K extra full-batch passes to add zero.
assert cfg.phases.response.gradient is False, "the response gradient should default off"
assert float(cfg.phases.response.coef) == 1.0
assert float(cfg.phases.response.replay_fraction) == 1.0
# Worker groups are built from the actor_rollout_ref subtree alone, so the mirror must carry it.
assert cfg.actor_rollout_ref.phases.response.enabled is True
assert cfg.actor_rollout_ref.phases.response.gradient is False


class _GateProbe(RayECHOTrainer):
    def __init__(self, cfg):
        self.config = cfg


# The gradient flag is subordinate: structure off must force the sweep off too, so no
# configuration can collect records without the round boundaries that give them meaning.
for structure, gradient, expect_struct, expect_grad in [
    (True, False, True, False),
    (True, True, True, True),
    (False, True, False, False),
    (False, False, False, False),
]:
    probe = _GateProbe(_compose([
        f"phases.response.enabled={str(structure).lower()}",
        f"phases.response.gradient={str(gradient).lower()}",
    ]))
    assert probe._response_enabled() is expect_struct, (structure, gradient)
    assert probe._response_gradient_enabled() is expect_grad, (structure, gradient)

cfg = _compose(["phases.response.enabled=false", "phases.response.coef=0.25", "phases.response.replay_fraction=0.5"])
assert cfg.phases.response.enabled is False
assert float(cfg.phases.response.coef) == 0.25
assert float(cfg.phases.response.replay_fraction) == 0.5

# --- 2. lambda_ent H_tool is gated off by default -------------------------------------

cfg = _compose([])
for phase in ("high_level", "low_level"):
    assert cfg.phases[phase].entropy.enabled is False, f"{phase} entropy gate should default off"
    assert float(cfg.phases[phase].kl_loss_coef) == 0.0, "utilities carry no KL term"

# reg_coeff alone must no longer switch the regularizer on.
cfg = _compose(["phases.low_level.entropy.reg_coeff=0.1"])
assert not RayECHOTrainer._uses_entropy_regularizer(cfg.phases.low_level), "reg_coeff must not bypass the gate"
cfg = _compose(["phases.low_level.entropy.reg_coeff=0.1", "phases.low_level.entropy.enabled=true"])
assert RayECHOTrainer._uses_entropy_regularizer(cfg.phases.low_level)
cfg = _compose(["phases.low_level.entropy.enabled=true"])
assert not RayECHOTrainer._uses_entropy_regularizer(cfg.phases.low_level), "gate on, coeff 0 -> still off"

# --- 3. one optimizer step per phase iteration ----------------------------------------
class _Probe(RayECHOTrainer):
    def __init__(self, cfg, prompt_batches):
        self.config = cfg
        self._phase_prompt_batch_sizes = prompt_batches


ok = _Probe(_compose([]), {"low_level": 136, "high_level": 136})
ok.config.phases.low_level.ppo_mini_batch_size = 136
ok.config.phases.high_level.ppo_mini_batch_size = 136
ok._validate_response_config()

bad = _Probe(_compose([]), {"low_level": 136, "high_level": 136})
bad.config.phases.low_level.ppo_mini_batch_size = 16
bad.config.phases.high_level.ppo_mini_batch_size = 136
try:
    bad._validate_response_config()
    raise SystemExit("FAIL: multi-step follower iteration was accepted")
except AssertionError as e:
    assert "one optimizer step per low_level" in str(e), str(e)

epochs = _Probe(_compose(["actor_rollout_ref.actor.ppo_epochs=2"]), {"low_level": 8, "high_level": 8})
epochs.config.phases.low_level.ppo_mini_batch_size = 8
epochs.config.phases.high_level.ppo_mini_batch_size = 8
try:
    epochs._validate_response_config()
    raise SystemExit("FAIL: ppo_epochs=2 was accepted")
except AssertionError as e:
    assert "ppo_epochs=1" in str(e), str(e)

# Disabled -> no constraint at all, so the A/B fallback keeps today's batching.
off = _Probe(_compose(["phases.response.enabled=false"]), {"low_level": 136, "high_level": 136})
off.config.phases.low_level.ppo_mini_batch_size = 16
off._validate_response_config()

# --- 4. R_L: tool validity, gated on the schema ---------------------------------------
GOOD = (
    "<think> Need a lookup. </think><tool> use search </tool><search> q </search>"
    "<result> r </result><think> Done. </think><tool> no more tools </tool>"
    "<answer>\\boxed{42}</answer>"
)
r = compute_tool_score("t", GOOD, "42", {"tool_calls_made": 4, "tool_calls_succeeded": 3})
assert r["format_valid"] and abs(r["score"] - 0.75) < 1e-9, r

r = compute_tool_score("t", GOOD, "42", {"tool_calls_made": 2, "tool_calls_succeeded": 2})
assert r["score"] == 1.0, r

r = compute_tool_score("t", GOOD, "42", {"tool_calls_made": 0, "tool_calls_succeeded": 0})
assert r["score"] == 0.0 and r["no_tool_calls"], r

r = compute_tool_score("t", "no schema here at all", "42", {"tool_calls_made": 4, "tool_calls_succeeded": 4})
assert r["score"] == 0.0 and not r["format_valid"], r

# Missing counters must not crash the scorer; they read as zero.
r = compute_tool_score("t", GOOD, "42", {})
assert r["score"] == 0.0 and r["no_tool_calls"], r

# R_L must be independent of the task answer: same trajectory, wrong ground truth.
assert compute_tool_score("t", GOOD, "not-42", {"tool_calls_made": 4, "tool_calls_succeeded": 3})["score"] == 0.75

# --- 4b. KNOWN DEFECT: the terminal seed g_fol is annihilated -------------------------
#
# v10 Eq. (30) defines the advantage as a per-trajectory SCALAR indexed (i, b, j); the
# role mask I_p only selects which tokens Eq. (31) sums over, and Eq. (32)'s two leader
# surrogates share that same scalar. verl instead folds the mask into the token advantage
# (scores.unsqueeze(-1) * response_mask), which is equivalent for a single-role loss and
# wrong the moment two role masks must come off one advantage.
#
# This asserts the CURRENT BROKEN behaviour on purpose, so that repairing it fails here
# and forces this check to be rewritten into the positive assertion below.
from training.echo_core_algos import compute_grpo_outcome_advantage

m_H = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
m_L = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
rewards = torch.zeros(2, 4)
rewards[0, -1], rewards[1, -1] = 1.0, 0.0
uids = np.array(["q0", "q0"])

adv, _ = compute_grpo_outcome_advantage(
    token_level_rewards=rewards, response_mask=m_H, index=uids, norm_adv_by_std_in_grpo=False
)
seed = float((adv * m_L).abs().sum())
expected = float((torch.tensor([[0.5], [-0.5]]) * m_L).abs().sum())
assert expected == 2.0
assert seed == 0.0, (
    "g_fol is no longer annihilated -- the advantage/mask repair (Change 1) has landed. "
    "Replace this block with: assert seed == expected."
)
print(f"KNOWN DEFECT confirmed: g_fol seed = {seed} (should be {expected}); g_resp is identically zero")

# --- 5. the gradient algebra reproduces sum_b c_b S_b ---------------------------------
torch.manual_seed(0)
model = nn.Linear(6, 3, bias=False)
params = echo_response.trainable_params(model)


def _grad_of(loss):
    model.zero_grad(set_to_none=False)
    loss.backward()
    return echo_response.clone_grads(params)


blocks = [torch.randn(4, 6) for _ in range(3)]
g_fol = [torch.randn_like(p) for p in params]

# Reference: accumulate c_b * S_b by hand, with c_b from an explicit gradient dot.
coeffs, scores = [], []
for x in blocks:
    logits = model(x)
    g_b = _grad_of(logits.pow(2).sum())
    coeffs.append(sum(float((a * b).sum()) for a, b in zip(g_b, g_fol)))
    scores.append(model(x).sum())

model.zero_grad(set_to_none=False)
for c, s in zip(coeffs, scores):
    (c * s).backward(retain_graph=True)
reference = echo_response.clone_grads(params)

# Same thing through the module's primitives, two passes as the actor does it.
replayed = []
for x in blocks:
    logits = model(x)
    _grad_of(logits.pow(2).sum())
    replayed.append(echo_response.dot(echo_response.grads(params), g_fol))

assert max(abs(a - b) for a, b in zip(coeffs, replayed)) < 1e-5, (coeffs, replayed)

model.zero_grad(set_to_none=False)
for c, x in zip(replayed, blocks):
    (c * model(x).sum()).backward()
got = echo_response.clone_grads(params)
err = max(float((a - b).abs().max()) for a, b in zip(reference, got))
assert err < 1e-4, f"response gradient mismatch: {err}"

# dot() is the global inner product; norm() its induced norm.
assert abs(echo_response.norm(g_fol) ** 2 - echo_response.dot(g_fol, g_fol)) < 1e-4

# adam_precond is lr before any step, and lr/(sqrt(v_hat)+eps) after one.
opt = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-8)
pre = echo_response.adam_precond(opt, params)
assert all(torch.allclose(p, torch.full_like(p, 1e-6)) for p in pre), "empty state should give lr"

model.zero_grad(set_to_none=False)
model(blocks[0]).sum().backward()
opt.step()
post = echo_response.adam_precond(opt, params)
assert all(torch.isfinite(p).all() and (p > 0).all() for p in post)
assert not torch.allclose(post[0], pre[0]), "preconditioner should react to the first step"

# The in-place fused form must agree with the reference AND must not touch the
# optimizer's own state -- exp_avg_sq is already fp32, so a .to(float32) that forgets
# copy=True aliases it and the in-place ops silently destroy the second moment.
state_before = [opt.state[p]["exp_avg_sq"].clone() for p in params]
fused = [torch.ones_like(p) for p in params]
echo_response.scale_by_adam_precond_(fused, opt, params)
assert all(torch.equal(a, opt.state[p]["exp_avg_sq"]) for a, p in zip(state_before, params)), \
    "scale_by_adam_precond_ mutated the follower optimizer state"
err = max(float((a - b).abs().max()) for a, b in zip(fused, post))
assert err < 1e-10, f"fused preconditioner disagrees with reference: {err}"

# copy_grads_into_ must reuse the buffer's storage -- allocating a second full-size
# gradient buffer mid-step is what fragmented the allocator and OOMed vLLM's wake_up.
model.zero_grad(set_to_none=False)
model(blocks[0]).sum().backward()
reuse = [torch.zeros_like(p) for p in params]
ptrs = [b.data_ptr() for b in reuse]
echo_response.copy_grads_into_(reuse, params)
assert [b.data_ptr() for b in reuse] == ptrs, "copy_grads_into_ reallocated instead of reusing"
assert all(torch.equal(a, b) for a, b in zip(reuse, echo_response.clone_grads(params)))

# --- 6. the shipped launch profile is consistent with the new defaults ----------------
import re

import yaml

ECHO_ROOT = os.path.join(ECHO_TOP, "training")
PROFILE_DIR = os.path.join(ECHO_ROOT, "training_config")
profiles = {
    name: yaml.safe_load(open(os.path.join(PROFILE_DIR, name)))
    for name in sorted(os.listdir(PROFILE_DIR))
    if name.endswith(".yaml")
}
assert profiles, "no launch profiles found"

scripts = {
    script: open(os.path.join(ECHO_ROOT, "scripts", script)).read()
    for script in ("train.sh", ".train_dry.sh")
}
for script, text in scripts.items():
    allow = re.search(r"VALID_LAUNCH_KEYS=\((.*?)\n\)", text, re.S).group(1).split()
    for key in ("response_enabled", "response_coef", "response_replay_fraction",
                "hl_entropy_enabled", "ll_entropy_enabled"):
        assert key in allow, f"{script}: {key} missing from VALID_LAUNCH_KEYS"
        env = key.upper()
        assert env in text, f"{script}: ${env} never reaches a hydra override"
    for name, profile in profiles.items():
        unknown = sorted(k for k in profile if k not in allow)
        assert not unknown, f"{script}: {name} keys not in VALID_LAUNCH_KEYS: {unknown}"

# EVERY profile must satisfy the one-step rule for the batch its own knobs derive --
# response_enabled defaults on, so a stale ppo_mini_batch_size is a startup failure.
for name, profile in profiles.items():
    if not profile.get("response_enabled", True):
        continue
    n_hl, n_ll = int(profile["hl_num_iters"]), int(profile["ll_num_iters"])
    world = int(profile["n_gpus_per_node"]) * int(profile["nnodes"])
    assert "train_10k" in profile["train_files"], f"{name}: update dataset_size below for {profile['train_files']}"
    dataset_size = 10000
    if profile.get("shared_prompt_stream", True):
        steps_per_epoch = n_hl * (n_ll + 1)
        prompt_batch = {p: (dataset_size // steps_per_epoch) // world * world for p in ("high_level", "low_level")}
    else:
        prompt_batch = {
            "high_level": (dataset_size // n_hl) // world * world,
            "low_level": (dataset_size // n_ll) // world * world,
        }

    probe = _Probe(_compose([]), prompt_batch)
    probe.config.phases.high_level.ppo_mini_batch_size = int(profile["hl_ppo_mini_batch_size"])
    probe.config.phases.low_level.ppo_mini_batch_size = int(profile["ll_ppo_mini_batch_size"])
    try:
        probe._validate_response_config()
    except AssertionError as e:
        raise SystemExit(f"FAIL: {name} would abort at startup: {e}")

    assert float(profile["hl_kl_loss_coef"]) == 0.0 and float(profile["ll_kl_loss_coef"]) == 0.0, \
        f"{name}: utilities carry no KL term"
    assert profile["hl_entropy_enabled"] is False and profile["ll_entropy_enabled"] is False, \
        f"{name}: lambda_ent H_tool should be off by default"
    print(f"  {name}: prompt_batch={prompt_batch['low_level']}, K={n_ll}, rounds={n_hl}")
print("s6 response checks passed")
