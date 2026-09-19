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

import torch
from hydra import compose, initialize_config_dir
from torch import nn

from training import echo_response
from verl.utils.reward_score.deep_research_echo import compute_tool_score

CONFIG_DIR = os.path.join(ECHO_TOP, "training", "config")


def _compose(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="echo_trainer", overrides=["algorithm.adv_estimator=grpo", *overrides])


# --- 1. the response block exists, is on by default, and is overridable --------------
cfg = _compose([])
assert cfg.phases.response.enabled is True, "response should default on: it is the algorithm under test"
assert float(cfg.phases.response.coef) == 1.0
assert float(cfg.phases.response.replay_fraction) == 1.0
# Worker groups are built from the actor_rollout_ref subtree alone, so the mirror must carry it.
assert cfg.actor_rollout_ref.phases.response.enabled is True

cfg = _compose(["phases.response.enabled=false", "phases.response.coef=0.25", "phases.response.replay_fraction=0.5"])
assert cfg.phases.response.enabled is False
assert float(cfg.phases.response.coef) == 0.25
assert float(cfg.phases.response.replay_fraction) == 0.5

# --- 2. lambda_ent H_tool is gated off by default -------------------------------------
from training.echo_ray_trainer import RayECHOTrainer

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

# --- 6. the shipped launch profile is consistent with the new defaults ----------------
import re

import yaml

ECHO_ROOT = os.path.join(ECHO_TOP, "training")
PROFILE = os.path.join(ECHO_ROOT, "training_config", "echo_3B_ll_hl_grpo.yaml")
profile = yaml.safe_load(open(PROFILE))

for script in ("train.sh", ".train_dry.sh"):
    text = open(os.path.join(ECHO_ROOT, "scripts", script)).read()
    allow = re.search(r"VALID_LAUNCH_KEYS=\((.*?)\n\)", text, re.S).group(1).split()
    unknown = sorted(k for k in profile if k not in allow)
    assert not unknown, f"{script}: launch profile keys not in VALID_LAUNCH_KEYS: {unknown}"
    for key in ("response_enabled", "response_coef", "response_replay_fraction",
                "hl_entropy_enabled", "ll_entropy_enabled"):
        assert key in allow, f"{script}: {key} missing from VALID_LAUNCH_KEYS"
        env = key.upper()
        assert env in text, f"{script}: ${env} never reaches a hydra override"

# The profile must satisfy the one-step rule for the batch its own knobs derive.
n_hl, n_ll = int(profile["hl_num_iters"]), int(profile["ll_num_iters"])
world = int(profile["n_gpus_per_node"]) * int(profile["nnodes"])
dataset_size = 10000  # train_10k.parquet, as named by the profile's train_files
steps_per_epoch = n_hl * (n_ll + 1)
prompt_batch = (dataset_size // steps_per_epoch) // world * world

probe = _Probe(_compose([]), {"low_level": prompt_batch, "high_level": prompt_batch})
probe.config.phases.high_level.ppo_mini_batch_size = int(profile["hl_ppo_mini_batch_size"])
probe.config.phases.low_level.ppo_mini_batch_size = int(profile["ll_ppo_mini_batch_size"])
probe._validate_response_config()

assert float(profile["hl_kl_loss_coef"]) == 0.0 and float(profile["ll_kl_loss_coef"]) == 0.0, "utilities carry no KL"
assert profile["response_enabled"] is True
assert profile["hl_entropy_enabled"] is False and profile["ll_entropy_enabled"] is False

print(f"launch profile ok: prompt_batch={prompt_batch}, K={n_ll}, rounds={n_hl}")
print("s6 response checks passed")
