# ECHO / ARPO Logging Cheatsheet

Metrics logged to wandb and JSONL under `<save_path>/logging_data/`.

**ECHO** (`ECHO/training/`) — nested two-phase trainer over a **shared** π_θ.
One outer cycle is `phases.low_level.num_iters` LL iterations then one HL
iteration. Each iteration is its own global step.

Namespaces (2026-09-27 layout). Keys are grouped by **what the number is computed
on**, not by which phase's step it landed in: the policy is one object and every
rollout is a full trajectory from it, so trajectory- and policy-level numbers are one
dense series each, logged every step, with `train/phase` (0 follower, 1 leader) as
the only marker of the alternation.

| Prefix | Meaning |
|---|---|
| `reward/*` | Scorer outputs on this step's rollouts, one series across phases. |
| `rollout/*` | Trajectory shape: lengths, tool calls, budget, infra excision. |
| `policy/entropy_*` | Pre-update entropy of π_θ on the two token populations. |
| `<phase>/…` | Only numbers whose value depends on that phase's gate, mask or advantage rule. |
| `high_level/response/*`, `high_level/aho/*` | Leader-only estimator diagnostics. |
| `val-core/*` / `val-aux/*` | Held-out quality (leader steps only). ARPO's split. |
| `train/*`, `perf/*`, `timing_s/step` | Bookkeeping and throughput. |

**ARPO** (`verl/trainer/ppo/`) — single-phase GRPO; flat keys. See section 5.

JSONL dumps: `logging_data/<key>.jsonl`, one file per logged key, mirroring the
wandb key exactly (`perf/*` and timing excluded).

---

## 1. Validation — primary quality dial

Validation is **greedy** (1 answer per prompt), every `test_freq` outer cycles
(always on the closing HL iteration), so it measures the bare leader after its step.

| Metric | What it means |
|---|---|
| `val-core/<dataset>/reward/mean@1` | Mean task score under the leader's schema rows (F1; `-1` on a leader-owned format failure; +0.1 multi-tool bonus). **Primary quality dial**, and the default best-checkpoint selector. |
| `val-aux/<dataset>/f1_score/mean@1` | Raw F1 (no `-1`, no bonus; bad format → 0). The number that lines up with external tables. |
| `val-aux/<dataset>/format_valid/mean@1` | Whole-schema format pass rate. |
| `val-aux/<dataset>/no_tool_calls/mean@1` | Fraction that never called `<search>` / `<python>`. Want ↓. |

Training rollouts are temperature-1 with many samples; val is greedy, so
val reward usually sits above training `reward/reward_mean`.

---

## 2. Shared series (`reward/*`, `rollout/*`, `policy/*`)

Emitted every training step from whichever phase ran.

| Metric | What it means |
|---|---|
| `reward/reward_mean` | Mean scorer reward on this step's rollouts (F1, `-1` on a phase-owned format failure, +0.1 bonus). Differs between phases only through the gate. |
| `reward/f1_mean` | Mean raw F1. Bookkeeping; `val-aux/f1_score` is the one to compare. |
| `reward/format_valid_rate` | Whole-schema format pass rate — the same quantity on either phase's step. |
| `reward/fail_answer_count_0` / `fail_unclosed_tag` / `fail_no_boxed` / `fail_other` | Whole-schema violation buckets from the scorer's `format_issues`, one bucket per sample by priority. This split is what exposed the tool-budget bug. |
| `reward/in_group_std` / `in_group_std_post` | Within-prompt reward std before / after the GRPO reward adjustment. ≈0 ⇒ dead groups. |
| `reward/group_zero_std_frac` / `_post` | Fraction of groups whose members all score the same — those carry **zero gradient**. |
| `rollout/response_length_mean` / `response_length_clip_ratio` | Generated length; fraction hitting `max_response_length` (inflated by EOS-terminated samples). |
| `rollout/tool_calls_per_traj_mean` | Mean `<search>`+`<python>` calls per trajectory. Batch-correct (per-sample arrays). |
| `rollout/no_tool_rate` | Fraction with no tool call. |
| `rollout/budget_exhausted_rate` | Fraction that spent the whole `tools.call_limit`. Batch-correct. |
| `rollout/tool_failure_excised_rate` | Infra failures (search API retries spent) excised from GRPO. |
| `policy/entropy_reasoning` | Mean Shannon H of the pre-update policy on think/answer tokens. |
| `policy/entropy_tool` | Same on tool/search/python tokens. Both logged every step; together they are the one policy's entropy trajectory. |

> The all-token entropy is no longer logged: the injected `<result>` spans (retrieved
> text, ~1.2 nats vs ~0.1 for the model's own tokens) dominated it, so it tracked
> search-cache coverage rather than the policy. The rollout's `tools/*` counters are not
> logged either: they ride in `meta_info`, which `DataProto.concat` keeps from rank 0 only.
>
> **Ground truth for anything about reward or format is `<run>/rollout/<step>.jsonl`**, which holds
> per-sample `score`, `reason`, `format_valid`, `phase_format_valid`, `format_issues` and `f1_score`.

---

## 3. Phase-prefixed (`high_level/` \| `low_level/`)

Only numbers whose value depends on which mask the loss ran under, which schema rows
gated the score, or which advantage rule applied.

| Metric | What it means |
|---|---|
| `<phase>/reward_mean` | Bookkeeping copy of `reward/reward_mean` under the phase that produced it. |
| `<phase>/gate_pass_rate` | Fraction passing **this phase's** schema rows (leader: think/answer; follower: tool/search/python). |
| `<phase>/budget_failed_rate` / `budget_demoted_rate` | Budget-exhausted samples that also failed this phase's gate; of those, how many stayed in their group under `in_group_zero`. |
| `<phase>/actor/pg_loss` | Policy-gradient scalar minimized this step. |
| `<phase>/actor/grad_norm` | Total grad norm **before** clipping (`clip_grad_norm_`'s return). > `grad_clip` ⇒ the step was rescaled. |
| `<phase>/actor/step_skipped` | 1 when the norm was non-finite and the optimizer step was skipped. |
| `<phase>/actor/lr` | That phase's AdamW / schedule LR. |
| `<phase>/actor/ppo_kl`, `pg_clipfrac`, `pg_clipfrac_lower` | Update stability on this phase's tokens. |
| `<phase>/actor/advantage_mean` / `advantage_std` | Measured **inside the update** on the advantages actually used (after the entropy substitution / aepo rescaling), on the phase mask. |
| `<phase>/actor/advantage_penalty_frac` | Entropy mode only: fraction of masked tokens carrying the `H_t = -1` format penalty. |
| `<phase>/actor/entropy_reg_loss` | Entropy regularizer term when enabled (before the coefficient). |
| `<phase>/actor/kl_loss` | KL-to-reference loss when `use_kl_loss`. |
| `<phase>/actor/opefo_lambda`, `opefo_delta_H_net`, `opefo_pos_mag`, `opefo_neg_mag` | OPEFO diagnostics when enabled. |
| `high_level/response/norm`, `direct_norm`, `ratio`, `cosine` | Response term vs direct gradient: ‖g_resp‖, ‖g_dir‖, their ratio, and cos(g_resp, g_dir). The empirical content of Proposition 1. |
| `high_level/response/blocks`, `c_*`, `mem_*`, `follower_*`, `hvp_*`, `steps_K`, `groups` | Estimator internals (adjoint / AHO). |
| `high_level/aho/surrogate` | AHO: the response surrogate, mean over micro-batches. |
| `high_level/aho/omega_absmean` / `omega_absmax` | AHO: size of the per-token weight ω on reasoning tokens. |
| `high_level/aho/weighted_frac`, `weighted_reasoning_frac`, `no_tool_traj_frac` | AHO: coverage of the weights over reasoning tokens / trajectories. |
| `high_level/aho/adv_product_mean` | AHO: mean of `A_H * A_L` over the leader batch — the sign the response term pushes with. |
| `high_level/aho/follower_adv_zero_frac` | AHO: fraction of leader trajectories with `A_L` exactly 0. |
| `high_level/aho/follower_score_mean` | AHO: the leader batch scored under the follower's schema rows (what `A_L` is built from). |

---

## 4. System / bookkeeping

| Metric | What it means |
|---|---|
| `train/global_step` | Optimizer step (every LL or HL iteration). |
| `train/phase` | 0 on a follower step, 1 on a leader step. |
| `train/hl_cycle` | Outer cycle index (0-based). `test_freq` / `save_freq` count these. |
| `train/epoch` | Epoch over the shared prompt stream. |
| `train/best_checkpoint_value` / `best_checkpoint_step` | Best val selector value so far and the step that set it. |
| `timing_s/step` | Wall time of one training step. |
| `perf/throughput`, `perf/mfu/actor`, `perf/*memory*` | Throughput and memory. |

---

## 5. ARPO baseline — what's different

ARPO is single-phase GRPO. Flat keys map roughly as:

| ECHO | ARPO |
|---|---|
| `reward/reward_mean`, `reward/f1_mean`, … | `reward/*` / `critic/score/*` (broader legacy set) |
| `policy/entropy_reasoning` / `policy/entropy_tool` | `actor/entropy_loss` (on `loss_mask` / `response_mask`) |
| `<phase>/actor/ppo_kl`, `pg_clipfrac` | `actor/ppo_kl`, `actor/pg_clipfrac` |
| `<phase>/actor/pg_loss`, `grad_norm`, `lr` | `actor/pg_loss`, `grad_norm`, `lr` (one optimizer) |
| `val-core/...`, `val-aux/...` | same |

ECHO-only: nested LL/HL loop, `train/phase`, `train/hl_cycle`, per-phase OPEFO /
`entropy_reg_loss`, and the shared-vs-phase split.

Scorer: both use F1 with `-1` on a format failure and the +0.1 multi-tool bonus, but
ECHO's gate is phase-owned (each phase fails only on its own tags) and its schema is
prompt-5, so the two `val-core` curves are not directly comparable; compare `val-aux/f1_score`.

---

## TL;DR — which metrics to watch

- **Getting better?** → `val-core/<dataset>/reward/mean@1`; `val-aux/<dataset>/f1_score/mean@1` for the external comparison
- **Format / tools sane?** → `reward/format_valid_rate` ↑, `<phase>/gate_pass_rate` ↑, `rollout/no_tool_rate` ↓, `rollout/tool_calls_per_traj_mean` > 0
- **Format failing — why?** → the `reward/fail_*` breakdown, then `<run>/rollout/<step>.jsonl` `format_issues`.
- **Reward / GRPO alive?** → `reward/reward_mean` ↑, `reward/in_group_std_post` not ≈ 0, `reward/group_zero_std_frac_post` low, `<phase>/actor/advantage_std` not ≈ 0
- **Policy peakedness?** → `policy/entropy_reasoning` and `policy/entropy_tool`, both every step.
- **Update stable?** → `<phase>/actor/ppo_kl`, `pg_clipfrac`, `grad_norm` (pre-clip), `step_skipped`
- **Response term doing anything?** → `high_level/response/ratio` (O(1) or smaller) and `high_level/response/cosine`
