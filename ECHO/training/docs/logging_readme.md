# ECHO / ARPO Logging Cheatsheet

Metrics logged to wandb and JSONL under `<save_path>/logging_data/`.

**ECHO** (`ECHO/training/`) — nested two-phase trainer over a **shared** π_θ.
One outer cycle is `phases.low_level.num_iters` LL iterations then one HL
iteration. Each iteration is its own global step.

Namespaces:

| Prefix | Meaning |
|---|---|
| `policy/*` | Whole-policy health (no phase prefix). One curve for π_θ. |
| `high_level/` / `low_level/` | Phase-owned optimizer / update only. |
| `val-core/*` / `val-aux/*` | Held-out quality (global). |
| `perf/*`, `timing_s/step` | Throughput (unprefixed). |

**ARPO** (`verl/trainer/ppo/`) — single-phase GRPO; flat keys (no phase /
`policy/` split). See section 5.

JSONL dumps: `logging_data/policy/*.jsonl` for health;
`logging_data/{high_level,low_level}/*.jsonl` for phase-owned keepers.

---

## 1. Validation — primary quality dial

Validation is **greedy** (1 answer per prompt), every `test_freq` outer cycles
(always on the closing HL iteration).

| Metric | What it means |
|---|---|
| `val-core/<dataset>/reward/mean@1` | Mean task score (F1; `-1` on bad format). **Primary quality dial.** |
| `val-aux/<dataset>/f1_score/mean@1` | F1 only (no `-1`; bad format → 0). |
| `val-aux/<dataset>/format_valid/mean@1` | Fraction passing system_prompt_1 format gate. |
| `val-aux/<dataset>/no_tool_calls/mean@1` | Fraction that never called `<search>` / `<python>`. Want ↓. |

Training rollouts are temperature-1 with many samples; val is greedy, so
val reward usually sits above training `policy/reward_mean`.

---

## 2. Whole-policy health (`policy/*`)

Emitted every training step from whichever phase ran. Shared namespace → one
wandb curve for π_θ.

| Metric | What it means |
|---|---|
| `policy/reward_mean` | Mean scorer reward on this step's rollouts. |
| `policy/f1_mean` | Mean F1 (zeros on non-matching / format-fail). |
| `policy/bad_format_rate` | Fraction with `score < 0`. Want ↓. |
| `policy/format_valid_rate` | Mean scorer `format_valid`. |
| `policy/no_tool_rate` | Fraction with no tool calls. Want ↓. |
| `policy/in_group_reward_std` | Within-prompt reward std (GRPO signal). ≈0 ⇒ dead groups. |
| `policy/advantage_std` | Post-normalize advantage spread on response tokens. |
| `policy/entropy` | Mean Shannon H on the **full response mask** (attention on response tokens — not phase `loss_mask`). True policy peakedness. |
| `policy/ppo_kl` | KL vs rollout (old) policy. Blow-up ⇒ step too large. |
| `policy/pg_clipfrac` | PPO clip hit rate (0 on pure OPEFO). |
| `policy/response_length_mean` | Mean generated length. |
| `policy/response_length_clip_ratio` | Fraction hitting `max_response_length`. |
| `policy/group_zero_std_frac` | Fraction of GRPO groups whose members all score the same. Those contribute **zero gradient** — the sharpest read on a dying signal. |
| `policy/tool_calls_per_traj_mean` | Mean `<search>`+`<python>` calls per trajectory. Batch-correct. |
| `policy/budget_exhausted_rate` | Fraction that spent the whole `tools.call_limit`. Batch-correct. |
| `policy/fail_answer_count_0` | Format failures with no `<answer>` at all. This was 0.43 at step 1 of the Sept-2026 runs — the tool-budget bug. |
| `policy/fail_unclosed_tag` | Format failures from an unclosed tag. |
| `policy/fail_no_boxed` | Format failures from a missing/malformed `\boxed{}`. |
| `policy/fail_other` | Remaining format failures (ordering violations etc.). |
| `policy/tools_total_calls` | `<search>` + `<python>` calls this step. **Rank-0 only** — see caveat below. |
| `policy/tools_successful_calls` | Successful tool returns. **Rank-0 only.** |

> **`tools/*` counters report a single data-parallel shard.** They ride in the rollout's
> `meta_info`, and `DataProto.concat` keeps `meta_info` from rank 0 only
> (`verl/protocol.py:710`), so with 8 GPUs they under-report by roughly 8x. At step 1 of
> `echo7BInstruct_lr5e8` the logged `tools/call_limit_reached_count` was 127 while the rollout dump
> held 1728 — which is a large part of why the tool-budget bug went unnoticed for six runs. Use
> them for relative trends only. `policy/tool_calls_per_traj_mean` and
> `policy/budget_exhausted_rate` are computed from per-sample arrays and are batch-correct.
>
> `policy/rollout_probs_diff_mean` was removed: the ECHO rollout sets `self.logprobs = 0` and never
> emits `rollout_log_probs`, so it could never fire. Re-add it together with vLLM logprobs if the
> vLLM/actor mismatch check is ever needed.
>
> **Ground truth for anything about reward or format is `<run>/rollout/<step>.jsonl`**, which holds
> per-sample `score`, `reason`, `format_valid` and `f1_score`.

---

## 3. Phase-owned optimizer (`high_level/` \| `low_level/`)

Only signals that differ by phase (independent AdamW / schedule / OPEFO /
entropy reg). Prefixed so HL and LL stay separable.

| Metric | What it means |
|---|---|
| `<phase>/actor/pg_loss` | Policy-gradient scalar minimized this step. |
| `<phase>/actor/grad_norm` | Grad norm before clip. Spikes ⇒ instability; ~0 ⇒ no signal. |
| `<phase>/actor/lr` | That phase's AdamW / schedule LR. |
| `<phase>/actor/entropy_reg_loss` | Entropy regularizer term when `reg_coeff > 0`. Compare `× reg_coeff` to `pg_loss`. |
| `<phase>/actor/entropy_phase_mask` | Mean Shannon H over **that phase's loss mask** — the entropy of the tokens this phase actually optimizes (HL: think+answer, LL: tool+search+python). Always emitted; compare against the whole-policy `policy/entropy`. |
| `<phase>/actor/opefo_lambda` | OPEFO λ* ∈ (−1,1) when `opefo.enabled`. |
| `<phase>/actor/opefo_delta_H_net` | Masked sum of Theorem-1 ΔH. Toward 0 when balanced. |
| `<phase>/actor/opefo_pos_mag` / `opefo_neg_mag` | Positive / \|negative\| ΔH masses for λ*. |

---

## 4. System / throughput

| Metric | What it means |
|---|---|
| `training/global_step` | Optimizer step (every LL or HL iteration). Ends at `N_HL * (N_LL + 1)` (× epochs in shared mode). |
| `training/hl_cycle` | Outer cycle index (0-based). `test_freq` / `save_freq` count these. |
| `timing_s/step` | Wall time of one training step. |
| `perf/throughput` | Tokens / second / GPU. |
| `perf/mfu/actor` | Model FLOPs Utilization on the actor update. |
| `perf/max_memory_allocated_gb` / `max_memory_reserved_gb` | Peak GPU memory. |
| `perf/cpu_memory_used_gb` | Peak CPU RSS. |

---

## 5. ARPO baseline — what's different

ARPO is single-phase GRPO. Flat keys map roughly as:

| ECHO | ARPO |
|---|---|
| `policy/reward_mean`, `policy/f1_mean`, … | `reward/*` / `critic/score/*` (broader legacy set) |
| `policy/entropy` (full response mask), plus `<phase>/actor/entropy_phase_mask` (phase mask) | `actor/entropy_loss` (often on `loss_mask` / `response_mask`) |
| `policy/ppo_kl`, `policy/pg_clipfrac` | `actor/ppo_kl`, `actor/pg_clipfrac` |
| `<phase>/actor/pg_loss`, `grad_norm`, `lr` | `actor/pg_loss`, `grad_norm`, `lr` (one optimizer) |
| `val-core/...` | same |

ECHO-only: nested LL/HL loop, `training/hl_cycle`, per-phase OPEFO /
`entropy_reg_loss`, and the `policy/*` vs phase-owned split.

Scorer: both use F1 (+ optional +0.1 multi-tool bonus when correct and both
tools appear), so `score` can reach `1.1`.

---

## TL;DR — which metrics to watch

- **Getting better?** → `val-core/<dataset>/reward/mean@1`
- **Format / tools sane?** → `policy/bad_format_rate` ↓, `policy/no_tool_rate` ↓, `policy/tool_calls_per_traj_mean` > 0
- **Format failing — why?** → the `policy/fail_*` breakdown, not `bad_format_rate` alone. A single scalar hid the tool-budget bug for six runs.
- **Reward / GRPO alive?** → `policy/reward_mean` ↑, `policy/in_group_reward_std` and `policy/advantage_std` not ≈ 0, `policy/group_zero_std_frac` low
- **Policy peakedness?** → `policy/entropy` (full response mask) vs `<phase>/actor/entropy_phase_mask` (the tokens that phase actually optimizes). They differ; say which one you mean.
- **Update stable?** → `policy/ppo_kl`, `policy/pg_clipfrac`; phase `grad_norm`
- **Entropy reg / OPEFO?** → `<phase>/actor/entropy_reg_loss` when `reg_coeff > 0`; `<phase>/actor/opefo_{lambda,delta_H_net,pos_mag,neg_mag}` when enabled
