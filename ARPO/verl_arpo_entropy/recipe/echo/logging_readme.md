# ECHO / ARPO Logging Cheatsheet

A non-technical guide to the metrics that get logged to wandb and the JSONL
files under `<save_path>/logging_data/`. Use this as a quick lookup for "what
is this number actually telling me?".

This file covers **two trainers**:

- **ECHO** (`recipe/echo/`) — two-phase trainer that runs separate HL / LL
  GRPO updates per step. Metrics are split by phase using `high_level/`
  and `low_level/` prefixes.
- **ARPO baseline** (`verl/trainer/ppo/`) — single-phase GRPO. Metrics are
  flat (no phase prefix). ECHO is a strict superset of ARPO's logging plus
  the per-phase split, so the bulk of this doc is written for ECHO and
  ARPO-specific notes live in section 5.

The metrics fall into four buckets:

1. **Validation** — quality on the held-out set, run every `test_freq` steps.
2. **Per-phase training** — what's happening in each step's HL / LL update
   (for ARPO, drop the phase prefix).
3. **Rollout & tools** — how the model is using tools during training rollouts.
4. **System** — throughput, timing, batch balancing.

Anything prefixed with `high_level/` is for the ECHO HL phase, `low_level/`
is for the ECHO LL phase. The same metric usually exists in both with
identical meaning.

---

## 1. Validation metrics — the ones to actually trust

Validation runs **greedy** (1 deterministic answer per prompt), so these are
the cleanest "how good is the current checkpoint?" numbers.

| Metric | What it means |
|---|---|
| `val-core/<dataset>/reward/mean@1` | Average task score on the val set (one greedy answer per prompt). For ECHO this is F1, with `-1` for any malformed response. **Primary quality dial.** |
| `val-aux/<dataset>/f1_score/mean@1` | Same idea but averaged over only the F1 component (no `-1` penalty for bad format — bad-format samples just contribute 0). Slightly more lenient than `val-core/.../reward`. |
| `val-aux/<dataset>/high_level_valid/mean@1` | **ECHO only.** Fraction of greedy responses whose high-level structure (`<select>` / `<think>` / `<answer>` / `\boxed{}`) is well-formed. |
| `val-aux/<dataset>/low_level_valid/mean@1` | **ECHO only.** Same for low-level structure (tool selection + tool payloads). |
| `val-aux/<dataset>/no_tool_calls/mean@1` | **ECHO only.** Fraction of valid responses that never called `<search>` or `<python>` (degenerate "answer-from-prior" behavior). Want this low. |

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
| `<phase>/reward/effective_reward_mean` | Mean per-sample reward actually used by GRPO for that phase/strategy. This is the source for `<save_path>/logging_data/<phase>/reward.jsonl`. |
| `<phase>/reward/score_mean` | Mean scorer output for scorer-based paths (`scorer`, scorer leg of `maxentropy_rl`). |
| `<phase>/reward/f1_mean` | Mean F1 component from scorer output (zeros on non-matching answers). |
| `high_level/reward/format_pass_rate` | Fraction of HL rollouts that passed all format checks. Want this climbing toward 1.0. |
| `high_level/reward/bad_format_rate` | `1 − format_pass_rate`. |
| `high_level/reward/no_tool_rate` | Fraction of HL rollouts that produced a valid answer without ever calling a tool. |
| `<phase>/reward/entropy_scalar_mean_good` | Entropy reward mean over good-format and tool-using samples when entropy overrides are active; otherwise equals `entropy_scalar_mean`. |
| `<phase>/reward/entropy_scalar_mean` | Entropy-channel reward mean over all samples (for `entropy`, `entropy-hybrid`, and entropy leg of `maxentropy_rl`). |
| `<phase>/reward/entropy_h_bar_mean` | Mean `H_bar` over the phase mask (`m^phase ∩ non_border_loss_mask`); same span as frozen `H_init`. |
| `<phase>/reward/entropy_h_phase_mean` | Alias of `entropy_h_bar_mean` when band is enabled (kept for W&B continuity). |
| `<phase>/reward/entropy_h_init_frozen_mean` | Frozen band anchor captured during warmup (flat after step 1). |
| `<phase>/reward/entropy_in_band_rate` | Fraction of samples with `H_bar` inside the band (logged after warmup only). Target 30–70%. |
| `<phase>/reward/entropy_band_warmup_active` | 1 on warmup steps (band off, ref captured); 0 once band scoring is on. |
| `low_level/reward/bad_format_rate` | LL-side format failure rate (defined by the LL validator). |
| `low_level/reward/no_tool_rate` | Fraction of valid LL rollouts that never invoked a tool. Should drift toward 0 as training pushes the model to actually use tools. |

### 2.2 Format validity (ECHO only)

| Metric | What it means |
|---|---|
| `high_level/reward/high_level_valid_rate` | Fraction of HL rollouts that pass HL-side format checks. |
| `high_level/reward/low_level_valid_rate` | Fraction of HL rollouts that *also* pass LL-side checks (diagnostic — HL only gates on the HL side). |
| `low_level/reward/high_level_valid_rate` | Same diagnostic in reverse — fraction of LL rollouts that also have valid HL structure. |
| `low_level/reward/low_level_valid_rate` | Fraction of LL rollouts that pass LL-side checks. |

ARPO doesn't log these because its scorer applies one combined format check
(`bad_format_rate` already covers it).

### 2.3 Actor / optimizer health

| Metric | What it means |
|---|---|
| `<phase>/actor/pg_loss` | The policy-gradient loss the optimizer minimized this step. Sign and magnitude tell you whether the update is moving the policy toward the good rollouts. |
| `<phase>/actor/grad_norm` | Gradient norm before clipping. Sudden spikes ≈ instability; flatlining near zero ≈ no learning signal. |
| `<phase>/actor/entropy_old_policy` | How "spread out" the model's distribution is over its phase-mask tokens, measured before the update. Higher = more exploration. **ARPO calls this `actor/entropy_loss`.** |
| `<phase>/actor/kl_loss` / `<phase>/actor/ppo_kl` | KL between the current policy and either the reference (`kl_loss`, gated on `kl_loss_coef`) or the old policy used for the rollout (`ppo_kl`). If KL blows up the policy is moving too fast per step. |
| `<phase>/actor/pg_clipfrac` | Share of tokens where the PPO ratio hit the clip range. Healthy is small but non-zero; near 1.0 means the trust region is too tight. |
| `<phase>/actor/lr` | Current learning rate (after warmup / schedule). |
| `<phase>/actor/entropy_reg_loss` | **ECHO only.** The entropy term added to actor loss when the entropy regularizer is on for that phase. Compare `entropy_reg_loss × reg_coeff` to `pg_loss`. |
| `<phase>/training/rollout_probs_diff_mean` | How much the rollout engine (vLLM) and the actor disagree per token, on average. Should stay small; if it grows, rollouts and training are drifting apart. |
| `<phase>/training/rollout_probs_diff_max` | Worst-case version of the above. Useful for catching tokenizer / templating bugs. |

### 2.4 Critic / sample-level reward distribution

`<phase>/critic/score/*` and friends always exist (they describe the
distribution of per-sample scores across the batch). The value-function
metrics only show up when a learned critic is enabled (`use_critic=True`).

| Metric | What it means |
|---|---|
| `<phase>/critic/score/{mean,max,min}` | Distribution of per-sample reward scores in this step's batch. `min` near `-1` ⇒ format failures still present; `max` ⇒ best F1 the model achieved this step. |
| `<phase>/critic/rewards/{mean,max,min}` | Same idea but at the token level (after reward shaping / KL penalty). |
| `<phase>/critic/advantages/{mean,max,min}` | After GRPO normalization. Symmetric around 0 with reasonable spread = healthy learning signal. All zeros = no advantage signal getting through. |
| `<phase>/critic/returns/{mean,max,min}` | Bootstrapped returns. With GAE off (default GRPO) these match advantages. |
| `<phase>/critic/vf_loss` | Value-function regression loss (only with `use_critic=True`). |
| `<phase>/critic/vf_explained_var` | Fraction of return variance the value function explains. Want this rising toward 1; near 0 means the value function isn't learning anything. |
| `<phase>/response_length/{mean,max,min,clip_ratio}` | Generated-response length stats. `clip_ratio` ⇒ fraction hitting `max_response_length`; if this climbs the model is rambling and getting truncated. |
| `<phase>/prompt_length/{mean,max,min,clip_ratio}` | Same for prompts. `clip_ratio > 0` is bad — prompts are getting truncated. |
| `<phase>/global_seqlen/{min,max,minmax_diff,balanced_min,balanced_max,mean}` | Per-DP-rank token counts before / after balancing. Big `minmax_diff` ⇒ load imbalance; the balancer should bring `balanced_*` close together. |

---

## 3. Rollout & tool usage

| Metric | What it means |
|---|---|
| `<phase>/tools/total_calls` | Total `<search>` + `<python>` calls across this phase's rollouts in the step. |
| `<phase>/tools/successful_calls` | Of those, how many returned a usable result (no API error / no exception). |
| `<phase>/tools/failed_calls` | `total_calls − successful_calls`. |
| `<phase>/tools/{python,search}/calls` | Per-tool call counts. |
| `<phase>/tools/{python,search}/success_rate` | Per-tool success rate. |
| `<phase>/tools/{python,search}/avg_time` | Per-tool average wall time per call. Useful for spotting a slow / flaky search backend. |
| `<phase>/tools/total_execution_time` / `tools/avg_execution_time` / `tools/max_execution_time` | Aggregate tool-call wall times for the step. |
| `<phase>/tools/total_retries` / `tools/max_retries` | Retry stats from the tool harness. Climbing retries ⇒ flaky tool API. |
| `<phase>/tools/call_limit_reached_count` | How many rollouts hit `tools.call_limit`. If high, either the limit is too low or the model is looping. |
| `<phase>/training/rollout_probs_diff_*` | See above — also a rollout-side signal. |
| `training/high_level_rollout_budget` | **ECHO only.** How many rollouts/prompt are routed to the HL update this step. Set by `high_level_budget` in the launch script. |
| `training/low_level_rollout_budget` | **ECHO only.** Mirror of above for LL (= `rollout.n − high_level_budget`). |

---

## 4. System / throughput

| Metric | What it means |
|---|---|
| `training/global_step` | Current optimizer step. |
| `training/epoch` | Current epoch index. |
| `timing_s/high_level_gen` / `timing_s/low_level_gen` | Wall time spent generating rollouts for that phase (ECHO embeds the phase in the timer name). ARPO uses `timing_s/gen`. |
| `timing_s/<phase>_reward` | Wall time spent scoring those rollouts (e.g. `timing_s/high_level_reward`). ARPO: `timing_s/reward`. |
| `timing_s/<phase>_update_actor` | Wall time on the actor update. ARPO: `timing_s/update_actor`. |
| `timing_s/<phase>_old_log_prob` / `<phase>_ref` / `<phase>_adv` | Per-stage wall times inside the step. ARPO drops the phase part of the name. |
| `timing_s/step` | Total wall time of one training step. |
| `timing_s/testing` | Wall time of the periodic validation pass. |
| `timing_per_token_ms/<stage>` | Same stages but normalized by tokens processed. Apples-to-apples comparison across batches with different sequence lengths. |
| `perf/total_num_tokens` | Total tokens processed in the step (prompts + responses). |
| `perf/throughput` | Tokens / second / GPU. |
| `perf/time_per_step` | Wall time per step (alias of `timing_s/step` exposed under `perf/`). |
| `perf/mfu/actor` | Model FLOPs Utilization for the actor update. Higher = more efficient use of the GPU. |
| `perf/max_memory_allocated_gb` / `max_memory_reserved_gb` | Peak GPU memory. Watch for OOM headroom. |
| `perf/cpu_memory_used_gb` | Peak CPU RSS. |
| `<phase>/training/balance_*` | Diagnostics from the per-DP-rank batch balancer (max/min token count, etc.). Mostly useful for spotting load imbalance. |

---

## 5. ARPO baseline — what's different

The ARPO baseline (`verl/trainer/ppo/ray_trainer.py`,
`verl/utils/reward_score/deep_research.py`) is a single-phase GRPO trainer.
Compared to ECHO:

- **No phase split.** All `high_level/<x>` and `low_level/<x>` keys collapse
  to a single flat `<x>`. So ECHO `high_level/reward/f1_mean` ↔ ARPO
  `reward/f1_mean`; ECHO `high_level/actor/pg_loss` ↔ ARPO `actor/pg_loss`.
  Timing is the one exception: ECHO embeds the phase in the timer *name*
  rather than as a prefix, so ECHO `timing_s/high_level_gen` ↔ ARPO
  `timing_s/gen`, ECHO `timing_s/low_level_update_actor` ↔ ARPO
  `timing_s/update_actor`, etc.
- **No phase-budget knobs.** `training/high_level_rollout_budget` and
  `training/low_level_rollout_budget` don't exist; every rollout goes
  through the one update. The total count is just `rollout.n`.
- **One combined format check, no validity sub-axes.** The ARPO scorer
  reports `score` and `f1_score` only — there is no `high_level_valid`,
  `low_level_valid`, or `no_tool_calls`. Consequently ARPO logs only
  `reward/f1_mean` + `reward/bad_format_rate` (no
  `format_pass_rate`, `high_level_valid_rate`, `low_level_valid_rate`,
  `no_tool_rate`).
- **Scorer scale.** Both ARPO and ECHO add a **+0.1 multi-tool bonus** when
  the answer is correct and both `</search>` and `</python>` are present, so
  `score` can exceed `1.0` (up to `1.1`) per sample.
- **No entropy reward channel.** ARPO has no
  `reward/entropy_scalar_mean*`. There's only the standard
  `actor/entropy_loss` (the ECHO equivalent of which is renamed —
  see below).
- **Entropy regularizer naming.** ARPO drives the entropy regularizer via
  `actor.entropy_coeff` directly inside `update_policy`; the resulting
  term is logged under `actor/entropy_loss` (it overloads the old-policy
  entropy diagnostic). ECHO splits these cleanly: `actor/entropy_old_policy`
  is the diagnostic, `actor/entropy_reg_loss` is the gradient term.
- **Validation metrics are identical in shape** (both go through the same
  `_validate` / `process_validation_metrics` machinery), but the *set* of
  `val-aux/<dataset>/<var>/mean@1` keys is smaller for ARPO because the
  scorer returns fewer numeric fields. ARPO emits `f1_score`, `score`,
  `reward`. ECHO additionally emits `high_level_valid`, `low_level_valid`,
  `no_tool_calls`.

The launch-script-level JSONL dumps under `logging_data/` reflect the same
split: ARPO writes one set of files at the top level (`reward.jsonl`,
`format_penalty.jsonl`, `score_mean.jsonl`, `pg_loss.jsonl`,
`grad_norm.jsonl`, `entropy_old_policy.jsonl`, `entropy_reg_loss.jsonl`,
`tools_total_calls.jsonl`, `tools_successful_calls.jsonl`); ECHO writes the
same set under `logging_data/high_level/` *and* `logging_data/low_level/`,
where `reward.jsonl` reads `<phase>/reward/effective_reward_mean` so files stay
valid across scorer, entropy, entropy-hybrid, and maxentropy_rl.

## 6. ECHO ↔ ARPO term map

When sanity-checking that a number from ECHO and a number from ARPO mean
the same thing, use this table. "Same axis" means the underlying
quantity is identical; only the prefix / name differs. "ECHO-only" rows
have no ARPO counterpart at all.

| Concept | ARPO key | ECHO key |
|---|---|---|
| Mean F1 across rollouts (zeros for bad format) | `reward/f1_mean` | `high_level/reward/f1_mean` |
| Format failure rate (`score < 0`) | `reward/bad_format_rate` | `high_level/reward/bad_format_rate` (and `low_level/reward/bad_format_rate`) |
| Per-sample reward distribution (mean / max / min) | `critic/score/{mean,max,min}` | `<phase>/critic/score/{mean,max,min}` |
| Per-sample reward, raw scalar reported by reward manager | (not separately logged; collapsed into `critic/score/mean`) | `<phase>/reward/score_mean` |
| Old-policy entropy on the loss mask (diagnostic) | `actor/entropy_loss` | `<phase>/actor/entropy_old_policy` |
| Differentiable entropy regularizer term in the actor loss | `actor/entropy_loss` (overloaded with above when `entropy_coeff != 0`) | `low_level/actor/entropy_reg_loss` (clean separation) |
| Policy-gradient loss | `actor/pg_loss` | `<phase>/actor/pg_loss` |
| Gradient norm | `actor/grad_norm` | `<phase>/actor/grad_norm` |
| KL between current and reference policy | `actor/kl_loss` | `<phase>/actor/kl_loss` |
| KL between current and rollout (old) policy | `actor/ppo_kl` | `<phase>/actor/ppo_kl` |
| PPO clip fraction | `actor/pg_clipfrac` | `<phase>/actor/pg_clipfrac` |
| GRPO advantage stats | `critic/advantages/{mean,max,min}` | `<phase>/critic/advantages/{mean,max,min}` |
| Response / prompt length stats | `response_length/*`, `prompt_length/*` | `<phase>/response_length/*`, `<phase>/prompt_length/*` |
| Rollout-vs-actor probability disagreement | `training/rollout_probs_diff_*` | `<phase>/training/rollout_probs_diff_*` |
| Tool call counts (search + python) | `tools/total_calls`, `tools/successful_calls`, … | `<phase>/tools/total_calls`, `<phase>/tools/successful_calls`, … |
| Per-stage wall time | `timing_s/{gen,reward,old_log_prob,ref,adv,update_actor,step}` | `timing_s/{<phase>_gen, <phase>_reward, <phase>_old_log_prob, <phase>_ref, <phase>_adv, <phase>_update_actor, step}` |
| Throughput / MFU / GPU memory | `perf/{throughput, mfu/actor, max_memory_allocated_gb, …}` | same (not phase-prefixed; reported once per step) |
| Validation core score | `val-core/<dataset>/reward/mean@1` | `val-core/<dataset>/reward/mean@1` |
| Validation F1 (no `-1` penalty) | `val-aux/<dataset>/f1_score/mean@1` | `val-aux/<dataset>/f1_score/mean@1` |
| HL / LL format pass rates on training rollouts | — (combined) | `<phase>/reward/{format_pass_rate, high_level_valid_rate, low_level_valid_rate}` (ECHO-only) |
| LL "valid format but no tool used" rate | — | `low_level/reward/no_tool_rate` (ECHO-only) |
| LL entropy reward axis (clean, on good-format∧tool-using samples) | — | `low_level/reward/entropy_scalar_mean_good` (ECHO-only) |
| Per-phase rollout budgets | — (single budget = `rollout.n`) | `training/{high_level,low_level}_rollout_budget` (ECHO-only) |
| Multi-tool +0.1 bonus baked into score | yes (shifts `score` up to `1.1`) | yes (same rule as ARPO) |

## TL;DR — which metrics to watch

- **Is the model getting better?** → `val-core/<dataset>/reward/mean@1`.
- **Is the format collapsing?** → `<phase>/reward/format_pass_rate` (ECHO; want ↑) and `<phase>/reward/bad_format_rate` (both; want ↓).
- **Is the LL phase actually using tools?** (ECHO) → `low_level/reward/no_tool_rate` (want ↓) and `low_level/tools/total_calls` (want > 0 and stable).
- **Is the entropy regularizer doing anything?** → ECHO: compare `low_level/actor/entropy_reg_loss × reg_coeff` to `low_level/actor/pg_loss`, and watch `<phase>/actor/entropy_old_policy`. ARPO: just watch `actor/entropy_loss` (it doubles as both the diagnostic and the regularizer term).
- **Is training stable?** → `<phase>/actor/grad_norm` and `<phase>/training/rollout_probs_diff_mean` (drop `<phase>/` for ARPO).
