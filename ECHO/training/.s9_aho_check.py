"""No-GPU check for the AHO response-term estimator (phases.response.estimator: aho).

Run:  VERL_ROOT=<verl> python .s9_aho_check.py

arXiv:2607.28849, GRPO variant, as implemented in recipe/aho/{response,actor,trainer}.py. Covers what can be checked without a GPU or a real run:
  1. the per-token weights omega_m = c_m * A_L against a per-trajectory Python loop on
     hand-built mask rows (result spans, tag tokens, no-tool rows, pre-first-tool tokens,
     post-last-tool tokens, unequal call lengths, gamma < 1, padding);
  2. the surrogate's gradient against autograd of the enumerated sum, under both
     aggregation modes, including the sign, 1/tau and coef scaling and the equality with
     the code's own g_fol aggregation for the h-sum;
  3. A_L: zero-std groups and singleton (excised) uids reproduce
     compute_grpo_outcome_advantage under both norm_adv_by_std settings, and A_L = 0
     rows yield zero weights;
  4. config and wiring: the aho block composes, train.sh's keys reach it, tau is derived
     or overridden, every startup assert fires on the config it guards, aho mode stashes
     no follower records / snapshots, and the adjoint mode is untouched.
"""

import inspect
import os
import sys

ECHO_TOP = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VERL_ROOT = os.environ["VERL_ROOT"]
sys.path[:0] = [VERL_ROOT, ECHO_TOP]

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from training.recipe.aho import response as echo_response
from training.core.core_algos import agg_loss, compute_grpo_outcome_advantage
from training.recipe.aho.actor import AhoActor as DataParallelECHOActor
from training.recipe.aho.trainer import AhoTrainer as RayECHOTrainer

CONFIG_DIR = os.path.join(ECHO_TOP, "training", "config")
torch.manual_seed(0)


def _compose(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="echo_trainer", overrides=["algorithm.adv_estimator=grpo", *overrides])


# --- 1. omega against a brute-force loop --------------------------------------------
print("=== 1. token weights ===")

# Row legend: H = reasoning (high mask), L = tool (low mask), R = <result> span (neither),
# T = tag token (neither), P = padding (neither). Positions are response positions.
ROWS = [
    "HHTLLLTRRRHHTLTRHHH",      # two calls of unequal length (3 then 1), trailing reasoning
    "HHHHHHHH",                 # no tool call at all
    "LLLHHH",                   # tool call first, no reasoning before it
    "HHLHHLHHLHHPPPP",          # three single-token calls, padded
    "TRRRHHLHTRH",              # result before any policy token, single call
]
T = max(len(r) for r in ROWS)
high = torch.zeros(len(ROWS), T)
low = torch.zeros(len(ROWS), T)
for i, row in enumerate(ROWS):
    for p, ch in enumerate(row):
        if ch == "H":
            high[i, p] = 1.0
        elif ch == "L":
            low[i, p] = 1.0
A_L = torch.tensor([[0.7], [0.3], [-0.5], [1.0], [-0.2]])


def loop_weights(gamma):
    out = torch.zeros(len(ROWS), T)
    for i, row in enumerate(ROWS):
        for p, ch in enumerate(row):
            if ch != "H":
                continue
            j = sum(1 for q in range(p) if row[q] == "L")
            c = gamma ** j if j >= 1 else 0.0
            out[i, p] = c * A_L[i, 0]
    return out


for gamma in (1.0, 0.9):
    got = echo_response.aho_token_weights(high, low, A_L, gamma=gamma)
    want = loop_weights(gamma)
    assert torch.allclose(got, want, atol=1e-6), (gamma, got, want)
    print(f"  gamma={gamma}: weights match the loop on {len(ROWS)} rows")

w1 = echo_response.aho_token_weights(high, low, A_L, gamma=1.0)
assert torch.all(w1[1] == 0), "a no-tool trajectory must carry no response weight"
assert torch.all(w1[2, :3] == 0) and torch.allclose(w1[2, 3:6], torch.full((3,), -0.5)), \
    "reasoning after a 3-token call gets c_m = 1 (counted once); nothing before the first call"
assert torch.allclose(w1[0, 16:19], torch.full((3,), 0.7)), \
    "post-last-tool reasoning is a terminal transition and gets c_m = 1 like any other"
assert torch.allclose(w1[0, 10:12], torch.full((2,), 0.7)), "between the calls: c_m = 1, not the tool-token count"
assert torch.all(w1[3, 11:] == 0) and torch.all(w1[4, :4] == 0), "padding and result/tag spans are 0"
assert torch.all((w1 != 0) <= (high > 0)), "weights live on the reasoning mask only"
w9 = echo_response.aho_token_weights(high, low, A_L, gamma=0.9)
assert torch.allclose(w9[0, 10:12], torch.full((2,), 0.9 ** 3 * 0.7)), "gamma < 1: gamma^{j(m)}, not a partial sum"
print("  conventions: pre-first-tool 0, every later token c_m = gamma^j(m), no-tool row 0, off-mask 0")

# --- 2. the surrogate's gradient ----------------------------------------------------
print("=== 2. surrogate gradient ===")
B = len(ROWS)
logits = torch.randn(B, T, dtype=torch.float64, requires_grad=True)
A_H = torch.tensor([[1.2], [-0.4], [0.9], [-1.1], [0.3]], dtype=torch.float64)
coef, tau = 0.5, 0.02
n_tool = low.sum(dim=-1)

for mode in ("token-mean", "seq-mean-token-mean", "seq-mean-token-sum"):
    log_prob = torch.log_softmax(logits, dim=-1)
    weights = echo_response.aho_token_weights(high, low, A_L, gamma=1.0).to(torch.float64)
    loss, diag = echo_response.aho_response_surrogate(
        log_prob, A_H, weights, high, low, mode, coef=coef, tau=tau
    )
    (got,) = torch.autograd.grad(loss, logits)

    # Enumerated: S_j = sum_m A_H^j omega_m log pi(m), aggregated like the code's g_fol.
    log_prob = torch.log_softmax(logits, dim=-1)
    S = (A_H * weights * log_prob * high.to(torch.float64)).sum(dim=-1)
    if mode == "token-mean":
        agg = S.sum() / n_tool.sum()
    elif mode == "seq-mean-token-mean":
        agg = (S / torch.clamp(n_tool, min=1.0)).mean()
    else:
        agg = S.mean()
    ref_loss = -(coef / tau) * agg
    (want,) = torch.autograd.grad(ref_loss, logits)
    rel = (got - want).norm() / want.norm()
    assert rel < 1e-12, (mode, rel)
    assert abs(loss.item() - ref_loss.item()) < 1e-12
    print(f"  {mode}: rel err {rel:.1e}, surrogate {loss.item():+.4f}")

# The h-sum normaliser is the SAME object the code applies to g_fol: agg_loss over the
# tool mask. Check on the token-mean case that a per-tool-token constant aggregates
# identically through both paths (this is the "like with like" claim).
const = torch.ones(B, T, dtype=torch.float64)
via_agg = agg_loss(loss_mat=const, loss_mask=low.to(torch.float64), loss_agg_mode="token-mean")
via_ours = (const * low.to(torch.float64)).sum() / n_tool.sum()
# masked_mean guards its denominator with a small epsilon; equal up to that.
assert abs(via_agg.item() - via_ours.item()) < 1e-6, (via_agg.item(), via_ours.item())
print("  token-mean normaliser equals agg_loss over the tool mask (up to masked_mean's epsilon)")

# Sign and scaling: doubling coef doubles, doubling tau halves, and the descent direction
# of the surrogate INCREASES sum_m A_H omega_m log pi(m) (it is -g_resp in loss form).
log_prob = torch.log_softmax(logits, dim=-1)
l1, _ = echo_response.aho_response_surrogate(log_prob, A_H, weights, high, low, "token-mean", coef=1.0, tau=0.1)
l2, _ = echo_response.aho_response_surrogate(log_prob, A_H, weights, high, low, "token-mean", coef=2.0, tau=0.1)
l3, _ = echo_response.aho_response_surrogate(log_prob, A_H, weights, high, low, "token-mean", coef=1.0, tau=0.2)
assert abs(l2.item() - 2 * l1.item()) < 1e-12 and abs(l3.item() - 0.5 * l1.item()) < 1e-12
(g,) = torch.autograd.grad(l1, logits)
score = (A_H * weights * torch.log_softmax(logits, dim=-1) * high.to(torch.float64)).sum()
(gs,) = torch.autograd.grad(score, logits)
assert torch.dot(g.flatten(), gs.flatten()) < 0, "loss gradient must point against the score"
print("  coef and 1/tau scale linearly; descent on the surrogate ascends the weighted score")

try:
    echo_response.aho_response_surrogate(log_prob, A_H, weights, high, low, "token-mean", coef=1.0, tau=0.0)
    raise SystemExit("FAIL: tau=0 was accepted")
except AssertionError:
    print("  tau=0 rejected")

# --- 3. A_L through the follower's own advantage builder -----------------------------
print("=== 3. follower group advantage ===")
scores = [1.0, 1.0, 1.0, 0.0, 0.5, 1.0, 0.0]
uids = np.array(["g0", "g0", "g0", "g1", "g1", "g1", "excised"], dtype=object)
for norm in (False, True):
    got = echo_response.follower_group_advantage(scores, uids, norm)
    tl = torch.tensor(scores).unsqueeze(-1)
    _, _, want = compute_grpo_outcome_advantage(
        token_level_rewards=tl, response_mask=torch.ones_like(tl), index=uids, norm_adv_by_std_in_grpo=norm
    )
    assert torch.allclose(got, want.to(torch.float32)), (norm, got, want)
    assert torch.all(got[:3] == 0), "a zero-std group has A_L = 0"
    assert got[6].item() == 0.0, "an excised singleton has A_L = 0"
    assert got.shape == (7, 1)
    print(f"  norm_adv_by_std={norm}: matches compute_grpo_outcome_advantage; zero-std group and singleton are 0")
zero_rows = echo_response.aho_token_weights(high[:3], low[:3], torch.zeros(3, 1), gamma=1.0)
assert torch.all(zero_rows == 0), "A_L = 0 must give zero weights"
print("  A_L = 0 rows carry no weight")

# --- 4. config and wiring -----------------------------------------------------------
print("=== 4. config and wiring ===")
cfg = _compose([])
assert str(cfg.phases.response.estimator) == "adjoint", "the adjoint estimator must stay the default"
assert cfg.phases.response.aho.tau is None and float(cfg.phases.response.aho.gamma) == 1.0
assert str(cfg.actor_rollout_ref.phases.response.estimator) == "adjoint", "the worker mirror must carry it"

# train.sh must know the keys and map them.
train_sh = open(os.path.join(ECHO_TOP, "training", "scripts", "train.sh")).read()
for key in ("response_estimator", "aho_tau", "aho_gamma"):
    assert key in train_sh, f"train.sh VALID_LAUNCH_KEYS lacks {key}"
for hydra_key in ("phases.response.estimator=", "phases.response.aho.tau=", "phases.response.aho.gamma="):
    assert hydra_key in train_sh, f"train.sh does not map {hydra_key}"
print("  yaml defaults and train.sh mapping present")


class _Probe(RayECHOTrainer):
    def __init__(self, cfg, prompt_batches=None):
        self.config = cfg
        self._phase_prompt_batch_sizes = prompt_batches or {"low_level": 128, "high_level": 128}


AHO_BASE = [
    "phases.response.estimator=aho",
    "phases.response.gradient=true",
    "phases.low_level.advantage_algorithm=grpo",
    "phases.high_level.advantage_algorithm=grpo",
    "phases.low_level.entropy.enabled=true",
    "phases.low_level.entropy.reg_coeff=0.01",
    "phases.low_level.ppo_mini_batch_size=128",
    "phases.high_level.ppo_mini_batch_size=128",
    "actor_rollout_ref.actor.use_kl_loss=true",
    "phases.low_level.kl_loss_coef=0.0",
]
ok = _Probe(_compose(AHO_BASE))
ok._validate_response_config()
assert abs(ok._aho_tau() - 0.01) < 1e-12, ok._aho_tau()
print("  tau derived from the follower's entropy coefficient")

kl = _Probe(_compose(AHO_BASE + ["phases.low_level.kl_loss_coef=0.005"]))
assert abs(kl._aho_tau() - 0.015) < 1e-12, "KL coefficient adds to the temperature"
explicit = _Probe(_compose(AHO_BASE + ["phases.response.aho.tau=0.1"]))
assert abs(explicit._aho_tau() - 0.1) < 1e-12
explicit._validate_response_config()
nominal = _Probe(_compose(AHO_BASE + ["phases.low_level.entropy.enabled=false", "phases.response.aho.tau=0.05"]))
nominal._validate_response_config()
assert nominal._aho_derived_tau() == 0.0 and nominal._aho_tau() == 0.05
print("  KL adds to tau; explicit tau overrides; nominal tau allowed with a warning")

meta = ok._phase_meta_info("high_level")
assert abs(meta["aho_tau"] - 0.01) < 1e-12 and meta["aho_gamma"] == 1.0
assert "response_estimator" not in meta and "response_exact" not in meta
assert ok._rollout_meta_info("high_level")["compute_follower_score"] is True
assert ok._rollout_meta_info("low_level")["compute_follower_score"] is False
print("  meta_info carries tau and gamma (no estimator switch); R_L is scored on the leader batch only")


def _must_fail(overrides, needle):
    probe = _Probe(_compose(AHO_BASE + overrides))
    try:
        probe._validate_response_config()
    except AssertionError as e:
        assert needle in str(e), str(e)
        return
    raise SystemExit(f"FAIL: accepted {overrides}")


_must_fail(["phases.low_level.entropy.enabled=false"], "tau > 0")
_must_fail(["phases.low_level.advantage_algorithm=entropy"], "low_level.advantage_algorithm")
_must_fail(["phases.high_level.advantage_algorithm=aepo"], "high_level.advantage_algorithm")
_must_fail(["phases.response.follower_return=false"], "retired")
_must_fail(["phases.response.exact=true"], "phases.response.exact")
_must_fail(["phases.response.curvature=true"], "phases.response.curvature")
_must_fail(["phases.low_level.opefo.enabled=true"], "opefo")
_must_fail(["phases.response.gradient=false"], "gradient=true")
_must_fail(["phases.low_level.ppo_mini_batch_size=16"], "one optimizer step per low_level")
# This recipe IS the AHO estimator: the other estimators and a response-off run are
# routing errors, not modes.
_must_fail(["phases.response.estimator=adjoint"], "AHO estimator only")
_must_fail(["phases.response.estimator=bogus"], "AHO estimator only")
_must_fail(["phases.response.enabled=false"], "response.enabled=true")
print("  every startup assert fires on the config it guards")

# The follower SGD requirement is adjoint-only: aho on AdamW must validate.
adamw = _Probe(_compose(AHO_BASE + ["phases.low_level.optim.optimizer=adamw", "phases.low_level.optim.weight_decay=0.01"]))
adamw._validate_response_config()
print("  AdamW follower accepted under aho (nothing is differentiated through the optimizer)")

# Actor wiring: the aho recipe actor has no records, no snapshots and no g_fol pass at all.
begin = inspect.getsource(DataParallelECHOActor._update_begin)
assert "stash_record" not in begin, "the aho actor stashes nothing"
crg = inspect.getsource(DataParallelECHOActor._compute_response_gradient)
assert "_aho_response_gradient(" in crg and "_leader_follower_direction" not in crg, \
    "aho computes g_resp from the leader batch alone (no g_fol pass)"
keys = inspect.getsource(DataParallelECHOActor._extra_select_keys)
assert "follower_scalar_advantages" in keys and "aho_token_weights" in keys
assert "_aho_response_gradient" in vars(DataParallelECHOActor)
assert "_after_advantage" in vars(RayECHOTrainer)
print("  actor: aho stashes nothing, skips g_fol, selects A_L and omega; trainer overrides _after_advantage")

# The base tree still knows nothing about the response path: no identifier, attribute or
# config access mentions the estimator. Metric NAMES (strings) are allowed there, exactly
# as the base's _route_actor_metrics already routes "actor/response_*" and "actor/aho_*".
import ast

for fname in ("core/phase_trainer.py", "core/phase_actor.py"):
    tree = ast.parse(open(os.path.join(ECHO_TOP, "training", fname)).read())
    idents = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            idents.add(node.id)
        elif isinstance(node, ast.Attribute):
            idents.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            idents.add(node.name)
        elif isinstance(node, ast.arg):
            idents.add(node.arg)
    offenders = sorted(i for i in idents if "aho" in i.lower() or "response_estimator" in i.lower())
    assert not offenders, f"{fname} references the estimator path: {offenders}"
print("  ALTERNATING-GRPO tree carries only metric names, no estimator logic")

print("all AHO checks passed")
