# ECHO 3B Training Notes

Launch config template: `training_config/echo_3B_ll_hl.yaml`.

---

## Shared hyperparameters

Defaults in `echo_3B_ll_hl.yaml` (common to recent 3B Inst runs unless overridden per experiment):

| Key | Value |
|---|---|
| `project_name` | `qwen3BInst` |
| `phase_order` | `["high_level", "low_level"]` |
| `high_level_budget` / `rollout_n` | 8 / 16 |
| `high_level_algorithm` / `low_level_algorithm` | `grpo` / `grpo` |
| `high_level_reward_strategy` | `scorer` |
| `high_level_update_repeats` / `low_level_update_repeats` | 1 / 1 |
| `low_level_reward_strategy` | `scorer` |
| `norm_adv_by_std_in_grpo` | `False` |
| `high_level_filter_groups_enable` / `low_level_filter_groups_enable` | `True` / `True` |
| `mask_*` | `first_select=low, select=low, think=high, answer=high, search=low, python=low` |
| `hl_kl_loss_coef` / `ll_kl_loss_coef` | `0.001` / `0.0` |
| `ll_entropy_reg_coeff` | `0.01` |
| `ll_entropy_reduction` / `ll_entropy_normalize` | `mean` / `true` |
| `ll_bad_format_penalty` / `ll_no_tool_penalty` | `-0.1` / `-0.05` |
| `use_sign_cond_clip` | `True` (pos clip 0.2, neg clip 0.1) |
| `save_best_checkpoint` | `true` (`best_checkpoint_metric: val-core/reward`) |

---

## Experiment naming convention

Suffixes in `experiment_name` map to overrides in the launch yaml:

| Suffix | Hyperparameter |
|---|---|
| `hl_kl_true` | `hl_kl_loss_coef: 0.001` |
| `ll_kl_false` | `ll_kl_loss_coef: 0.0` |
| `reg_on` | `ll_entropy_reg_coeff: 0.01` |
| `no_tool_neg` | `ll_no_tool_penalty: -0.05` |
| `hl_2` | `high_level_update_repeats: 2` |
| `ll_scorer` | `low_level_reward_strategy: scorer` |

---

## Checkpoint storage

Root (from `secrets.sh` `OUTPUT_ROOT`):

```
/scratch/project/prj-02-llm-reasoning-shakkottai/saratb/ECHO/checkpoints/<experiment_name>/
```

Each run directory snapshots its launch config at `training_config/launch_config.yaml` (also `launch_script.sh`, `launch_hydra_overrides.txt`, resolved hydra config under `outputs/.hydra/`).

---

## 1. `echo3BInst_hl_ll_entropy_grpo_hl_kl_true_ll_kl_false_reg_on_ll_scorer`

### Configuration

Overrides relative to shared defaults:

| Key | Value |
|---|---|
| `high_level_update_repeats` | 1 |
| `low_level_reward_strategy` | `scorer` |

All other knobs match `echo_3B_ll_hl.yaml`.

### Checkpoint path

```
/scratch/project/prj-02-llm-reasoning-shakkottai/saratb/ECHO/checkpoints/echo3BInst_hl_ll_entropy_grpo_hl_kl_true_ll_kl_false_reg_on_ll_scorer/
```

Best step (`best_checkpoint.txt`): **25**.

### Observed curves

Metric: `val-core/DR_grpo_mix/reward/mean@1`.

| Step | val reward | F1 | HL format valid |
|---|---|---|---|
| 25 | 0.369 | 0.452 | — |
| 50 | 0.231 | 0.369 | 0.861 |
| 55 | 0.354 | 0.437 | 0.917 |
| 60 | 0.250 | 0.405 | 0.844 |
| 70 | 0.279 | 0.407 | 0.872 |

Val reward is volatile (dips at steps 50 and 60) but recovers. HL `response_length/mean` stays roughly 900–1100. Search tool success rate is often low (0–13%) throughout training.

### Diagnosis

Both phases use `scorer` reward, so HL and LL updates both optimize task quality (F1 / format). Temporary val dips are not sustained: LL updates pull the policy back after HL-phase drift.

`ll_entropy_reg_coeff: 0.01` adds an entropy regularizer on the actor loss without replacing the LL reward signal. `ll_no_tool_penalty` and `ll_bad_format_penalty` are present in the yaml but only affect LL training when `low_level_reward_strategy` is `entropy`; with `scorer` they are inactive.

Best checkpoint at step 25; training continued to step 70 without catastrophic collapse.

---

## 2. `echo3BInst_hl_ll_entropy_grpo_hl_kl_true_ll_kl_false_reg_on_no_tool_neg_hl_2`

### Configuration

Overrides relative to shared defaults:

| Key | Value |
|---|---|
| `high_level_update_repeats` | **2** |
| `low_level_reward_strategy` | **entropy** |

All other knobs match `echo_3B_ll_hl.yaml` (including `ll_no_tool_penalty: -0.05`, `ll_bad_format_penalty: -0.1`, `ll_entropy_reg_coeff: 0.01`).

### Checkpoint path

```
/scratch/project/prj-02-llm-reasoning-shakkottai/saratb/ECHO/checkpoints/echo3BInst_hl_ll_entropy_grpo_hl_kl_true_ll_kl_false_reg_on_no_tool_neg_hl_2/
```

Best step (`best_checkpoint.txt`): **45**.

### Observed curves

Metric: `val-core/DR_grpo_mix/reward/mean@1`.

| Step | val reward | F1 | HL valid | LL valid | HL resp len | HL clip ratio |
|---|---|---|---|---|---|---|
| 45 | 0.353 | 0.436 | 0.917 | 0.933 | — | — |
| 50 | 0.340 | 0.451 | 0.894 | — | 1233 | 7.3% |
| 55 | 0.273 | 0.417 | 0.856 | 0.883 | 1390 | 12.0% |
| 60 | 0.157 | 0.363 | 0.794 | — | — | — |
| 70 | 0.047 | 0.331 | 0.717 | — | 1450 | 8.9% |
| 75 | -0.096 | 0.287 | 0.617 | 0.628 | — | — |

Val reward peaks at step 45 then collapses monotonically. Inflection at step 55: reward drops 0.340 → 0.273 while HL response length jumps 1233 → 1390 and clip ratio 7% → 12%.

Training-side signals during collapse:
- HL `response_length/mean`: 1014 (step 40) → 1758 (step 74)
- HL `response_length/clip_ratio`: up to 15% (responses hitting 4096-token cap)
- HL `actor/pg_clipfrac`: 0.12–0.17 around steps 53–56
- HL `actor/kl_loss`: spikes to 0.033 at steps 53, 71
- LL `reward/entropy_scalar_mean_good` ≈ 0.004–0.005 (flat; core LL reward signal is tiny)
- LL `reward/entropy_scalar_mean` ≈ -0.02 (pulled negative by format/no-tool overrides)
- `val-aux/DR_grpo_mix/no_tool_calls/mean@1`: 0.0 throughout (not a no-tool degeneracy)

### Diagnosis

**LL entropy reward decoupled from task quality.** `low_level_reward_strategy: entropy` optimizes normalized entropy on tool tokens, not F1. Overrides (`bad_format: -0.1`, `no_tool: -0.05`) provide weak negative signal but the dominant LL gradient pushes entropy, not answer correctness. LL updates do not anchor task quality while HL drifts.

**Double HL updates accelerate drift.** `high_level_update_repeats: 2` runs HL rollout + policy update twice per step before LL. HL response lengths grow, more rollouts hit the 4096 cap and truncate, and HL policy updates are larger (high `pg_clipfrac`, `kl_loss` spikes). HL still uses `scorer` reward, but twice the gradient per step outpaces what LL entropy regularization can correct.

**Format validity collapse drives val reward down.** `high_level_valid` falls 0.917 → 0.617; `low_level_valid` falls 0.933 → 0.628. F1 also drops (0.451 → 0.287) but format breakage is the sharper signal. Longer truncated responses produce more format failures, which directly lowers val score.

**Search failures add noise.** Search success rate is often 0–5% during LL steps. `BRIGHTDATA_API_KEY` is empty in `secrets.sh`, so live search mostly misses. With entropy LL reward there is no direct task penalty for failed tool calls, which amplifies instability but is secondary to the entropy + double-HL combination above.

Best checkpoint at step 45 (val 0.353). Live policy at step 75 is -0.096.

---

## Recommendations

| Setting | Suggested value | Reason |
|---|---|---|
| `low_level_reward_strategy` | `scorer` | Task reward on both phases; avoids entropy-only LL drift |
| `high_level_update_repeats` | `1` | `×2` correlated with HL length blow-up and format collapse |
| `ll_entropy_reg_coeff` | `0.01` (keep) | Safe alongside `scorer` LL reward; do not pair with `entropy` LL strategy |
| Training duration | Stop at best checkpoint | Collapsing run peaked step 45; continued training degraded to -0.096 |
| Search API | Restore `BRIGHTDATA_API_KEY` or expand cache | Chronic search failures add noise to tool-use training |

Current `echo_3B_ll_hl.yaml` matches the recommended settings (`ll_scorer`, `hl_repeats=1`).
