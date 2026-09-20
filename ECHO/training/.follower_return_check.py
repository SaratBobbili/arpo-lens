"""No-GPU check for the phases.response.follower_return split and the reward metric split.

Covers:
  1. follower_return=null inherits phases.response.enabled (the historical coupling);
  2. it can be set true/false independently of enabled, in both directions;
  3. the LOW-LEVEL scorer metric is policy/follower_return_mean when R_L is in use and
     policy/reward_mean otherwise, so one wandb key never carries two reward functions;
  4. bad_format_rate is not emitted for R_L, which is never negative;
  5. gradient=true with follower_return=false warns (Eq. 33 needs R_L advantages).

Run: conda activate arpo && PYTHONPATH=$VERL_ROOT:$ECHO_TOP python3 ECHO/training/.follower_return_check.py
"""
import io, contextlib
from omegaconf import OmegaConf
from ECHO.training.echo_ray_trainer import RayECHOTrainer

BASE = OmegaConf.load("ECHO/training/config/echo_trainer.yaml")


def T(**response):
    t = RayECHOTrainer.__new__(RayECHOTrainer)
    cfg = OmegaConf.merge(BASE, OmegaConf.create({"phases": {"response": response}}))
    t.config = cfg
    return t


print("=== 1/2. follower_return resolution ===")
cases = [
    (dict(enabled=True,  follower_return=None),  True,  "null inherits enabled=true"),
    (dict(enabled=False, follower_return=None),  False, "null inherits enabled=false"),
    (dict(enabled=True,  follower_return=False), False, "Alg-1 structure, task return both phases"),
    (dict(enabled=False, follower_return=True),  True,  "paper reward split, no round structure"),
]
for resp, want, label in cases:
    got = T(**resp)._follower_return_enabled()
    print(f"  enabled={str(resp['enabled']):<5} follower_return={str(resp['follower_return']):<5} -> {got!s:<5} ({label})")
    assert got is want, (resp, got, want)

# The split must not disturb the other three things `enabled` gates.
for resp in (dict(enabled=True, follower_return=False), dict(enabled=True, follower_return=None)):
    t = T(**resp)
    assert t._response_enabled() is True, resp

print("=== 3/4. scorer metric routing ===")
extra = {"score": [1.0, 0.5, -1.0, 0.0], "format_valid": [True, True, False, True],
         "f1_score": [1.0, 0.5, 0.0, 0.0], "reason": ["", "", "bad format: answer_count=0", ""]}
task = RayECHOTrainer._build_scorer_metrics(extra, follower_return=False)
foll = RayECHOTrainer._build_scorer_metrics(extra, follower_return=True)
print("  task-return keys :", sorted(k for k in task if "reward" in k or "return" in k or "bad_format" in k))
print("  R_L keys         :", sorted(k for k in foll if "reward" in k or "return" in k or "bad_format" in k))
assert "policy/reward_mean" in task and "policy/bad_format_rate" in task
assert "policy/follower_return_mean" not in task
assert "policy/follower_return_mean" in foll
assert "policy/reward_mean" not in foll, "R_L must never land on policy/reward_mean"
assert "policy/bad_format_rate" not in foll, "bad_format_rate is meaningless for a [0,1] return"
# Shared health metrics survive in both, so the format gate stays readable either way.
for k in ("policy/format_valid_rate", "policy/fail_answer_count_0"):
    assert k in task and k in foll, k
assert abs(foll["policy/follower_return_mean"] - task["policy/reward_mean"]) < 1e-9

print("=== 5. inconsistent-config warning ===")
buf = io.StringIO()
t = T(enabled=True, follower_return=False, gradient=True)
t._phase_prompt_batch_sizes = {"low_level": 136, "high_level": 136}
with contextlib.redirect_stdout(buf):
    try:
        t._validate_response_config()
    except AssertionError:
        pass  # the one-step-per-group asserts are exercised by .s6_dryrun_check.py
out = buf.getvalue()
assert "not Eq. (33)" in out, out
print("  warned:", out.strip().split("WARNING:")[1].strip()[:80] + "...")

buf = io.StringIO()
t = T(enabled=True, follower_return=None, gradient=True)
t._phase_prompt_batch_sizes = {"low_level": 136, "high_level": 136}
with contextlib.redirect_stdout(buf):
    try:
        t._validate_response_config()
    except AssertionError:
        pass
assert "not Eq. (33)" not in buf.getvalue(), "must not warn on the consistent default"
print("  silent on the consistent default")

print("\nALL ASSERTIONS PASSED")
