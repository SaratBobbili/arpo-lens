# ECHO Logging Cheatsheet

A non-technical guide to the metrics ECHO logs to wandb / the JSONL files
under `<save_path>/logging_data/`. Use this as a quick lookup for "what is
this number actually telling me?".

The metrics fall into four buckets:

1. **Validation** — quality on the held-out set, run every `test_freq` steps.
2. **Per-phase training** — what's happening in each step's HL / LL update.
3. **Rollout & tools** — how the model is using tools during training rollouts.
4. **System** — throughput, timing, batch balancing.

Anything prefixed with `high_level/` is for the HL phase, `low_level/` is for
the LL phase. The same metric usually exists in both with identical meaning.

---

## 1. Validation metrics — the ones to actually trust

Validation runs **greedy** (1 deterministic answer per prompt), so these are
the cleanest "how good is the current checkpoint?" numbers.

| Metric | What it means |
|---|---|
| `val-core/<dataset>/reward/mean@1` | Average task score on the val set (one greedy answer per prompt). For ECHO this is F1, with `-1` for any malformed response. **Primary quality dial.** |
| `val-aux/<dataset>/f1_score/mean@1` | Same idea but averaged over only the F1 component (no `-1` penalty for bad format — bad-format samples just contribute 0). Slightly more lenient than `val-core/.../reward`. |
| `val-aux/<dataset>/high_level_valid/mean@1` | Fraction of greedy responses whose high-level structure (`<select>` / `<think>` / `<answer>` / `\boxed{}`) is well-formed. |
| `val-aux/<dataset>/low_level_valid/mean@1` | Same for low-level structure (tool selection + tool payloads). |
| `val-aux/<dataset>/no_tool_calls/mean@1` | Fraction of valid responses that never called `<search>` or `<python>` (degenerate "answer-from-prior" behavior). Want this low. |

> Why validation looks higher than training rewards: training rolls out at
> temperature 1 with 8–16 noisy samples per prompt; validation is a single
> greedy answer. Greedy almost never breaks format, so the validation mean
> is essentially the model's "best shot".

---

## 2. Per-phase training metrics

These are computed fresh every step on the rollouts that drive that phase's
update. They tell you what the model is doing **right now**, not what it
will do at eval time.

### 2.1 Reward / task signal

| Metric | What it means |
|---|---|
| `high_level/reward/score_mean` | Average HL phase reward across all rollouts (F1 on good answers, `-1` on format failures). The number that GRPO is optimizing on the HL update. |
| `high_level/reward/f1_mean` | Same as above but with `-1`s replaced by 0 — answers F1 averaged over **all** HL rollouts including format failures. Pulls down vs. validation because of T=1 sampling noise. |
| `high_level/reward/format_pass_rate` | Fraction of HL rollouts that passed all format checks. Want this climbing toward 1.0. |
| `high_level/reward/bad_format_rate` | `1 − format_pass_rate`. |
| `high_level/reward/no_tool_rate` | Fraction of HL rollouts that produced a valid answer without ever calling a tool. |
| `low_level/reward/entropy_scalar_mean_good` | Average per-sample entropy reward on **good-format, tool-using** LL rollouts. The clean "how diverse is the LL policy on the samples that count?" signal. |
| `low_level/reward/entropy_scalar_mean` | Same but over all LL rollouts (includes the `bad_format_penalty` and the zeros from no-tool soft-fails). Use this if you want the raw mean reward GRPO sees; use the `_mean_good` variant to track entropy in isolation. |
| `low_level/reward/bad_format_rate` | LL-side format failure rate (defined by the LL validator). |
| `low_level/reward/no_tool_rate` | Fraction of valid LL rollouts that never invoked a tool. Should drift toward 0 as training pushes the model to actually use tools. |

### 2.2 Format validity

| Metric | What it means |
|---|---|
| `high_level/reward/high_level_valid_rate` | Fraction of HL rollouts that pass HL-side format checks. |
| `high_level/reward/low_level_valid_rate` | Fraction of HL rollouts that *also* pass LL-side checks (diagnostic — HL only gates on the HL side). |
| `low_level/reward/high_level_valid_rate` | Same diagnostic in reverse — fraction of LL rollouts that also have valid HL structure. |
| `low_level/reward/low_level_valid_rate` | Fraction of LL rollouts that pass LL-side checks. |

### 2.3 Actor / optimizer health

| Metric | What it means |
|---|---|
| `<phase>/actor/pg_loss` | The policy-gradient loss the optimizer minimized this step. Sign and magnitude tell you whether the update is moving the policy toward the good rollouts. |
| `<phase>/actor/grad_norm` | Gradient norm before clipping. Sudden spikes ≈ instability; flatlining near zero ≈ no learning signal. |
| `<phase>/actor/entropy_old_policy` | How "spread out" the model's distribution is over its phase-mask tokens, measured before the update. Higher = more exploration. |
| `low_level/actor/entropy_reg_loss` | The entropy term added to the LL actor loss when the entropy regularizer is on. Compare its magnitude to `pg_loss` — if `entropy_reg_loss` × `reg_coeff` dominates, the regularizer is driving everything. |
| `<phase>/training/rollout_probs_diff_mean` | How much the rollout engine (vLLM) and the actor disagree per token, on average. Should stay small; if it grows, rollouts and training are drifting apart. |
| `<phase>/training/rollout_probs_diff_max` | Worst-case version of the above. Useful for catching tokenizer / templating bugs. |

### 2.4 Critic (only if `use_critic=True`)

| Metric | What it means |
|---|---|
| `<phase>/critic/vf_loss` | Value-function regression loss. |
| `<phase>/critic/vf_explained_var` | Fraction of return variance the value function explains. Want this rising toward 1; near 0 means the value function isn't learning anything. |

---

## 3. Rollout & tool usage

| Metric | What it means |
|---|---|
| `<phase>/tools/total_calls` | Total `<search>` + `<python>` calls across this phase's rollouts in the step. |
| `<phase>/tools/successful_calls` | Of those, how many returned a usable result (no API error / no exception). |
| `<phase>/training/rollout_probs_diff_*` | See above — also a rollout-side signal. |
| `training/high_level_rollout_budget` | How many rollouts/prompt are routed to the HL update this step. Set by `high_level_budget` in the launch script. |
| `training/low_level_rollout_budget` | Mirror of above for LL (= `rollout.n − high_level_budget`). |

---

## 4. System / throughput

| Metric | What it means |
|---|---|
| `training/global_step` | Current optimizer step. |
| `training/epoch` | Current epoch index. |
| `timing/<phase>_gen` | Wall time spent generating rollouts for that phase. |
| `timing/<phase>_reward` | Wall time spent scoring those rollouts. |
| `timing/<phase>_update_actor` | Wall time spent on the actor update. |
| `timing/testing` | Wall time of the periodic validation pass. |
| `perf/total_num_tokens` | Total tokens processed in the step (prompts + responses). |
| `perf/throughput` | Tokens / second / GPU. |
| `<phase>/training/balance_*` | Diagnostics from the per-DP-rank batch balancer (max/min token count, etc.). Mostly useful for spotting load imbalance. |

---

## TL;DR — which metrics to watch

- **Is the model getting better?** → `val-core/<dataset>/reward/mean@1`.
- **Is the format collapsing?** → `<phase>/reward/format_pass_rate` (want ↑) and `<phase>/reward/bad_format_rate` (want ↓).
- **Is the LL phase actually using tools?** → `low_level/reward/no_tool_rate` (want ↓) and `low_level/tools/total_calls` (want > 0 and stable).
- **Is the entropy regularizer doing anything?** → compare `low_level/actor/entropy_reg_loss × reg_coeff` to `low_level/actor/pg_loss`, and watch `<phase>/actor/entropy_old_policy` over time.
- **Is training stable?** → `<phase>/actor/grad_norm` and `<phase>/training/rollout_probs_diff_mean`.
