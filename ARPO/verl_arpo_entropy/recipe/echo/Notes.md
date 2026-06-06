# ECHO 3B Training Notes

Launch template: `training_config/echo_3B_ll_hl.yaml`.

## Goal of this note

Distill what each experiment actually taught us, so future runs change only the knobs that matter.

## Reward-plumbing note

- HL and LL now share the same reward-strategy config surface (`scorer`, `entropy`, `entropy-hybrid`, `maxentropy_rl`) in launch/config plumbing.
- Format validators remain phase-specific by design: HL and LL still check different structure rules before scoring.
- LL `scorer` uses full answer/F1 scoring after LL format validation; no-tool behavior remains a separate signal for phase penalties.

## Common setup

- Optimizer family: GRPO for both phases unless stated otherwise.
- Training order: `high_level -> low_level`.
- Main model-selection metric: `val-core/DR_grpo_mix/reward/mean@1`.
- Checkpoint root:

```
/scratch/project/prj-02-llm-reasoning-shakkottai/saratb/ECHO/checkpoints/<experiment_name>/
```

## Lens for interpretation

We can reason about updates with a simplified objective:

```math
\mathcal{J}_{HL} \approx \mathbb{E}[R_{\mathrm{task}}] - \beta_{HL}\,\mathrm{KL}(\pi_{HL}\|\pi_{\mathrm{ref}})
```

```math
\mathcal{J}_{LL}^{\mathrm{scorer}} \approx \mathbb{E}[R_{\mathrm{task}}] + \lambda_H H(\pi_{LL})
```

```math
\mathcal{J}_{LL}^{\mathrm{entropy}} \approx \mathbb{E}[R_{\mathrm{entropy}} + p_{\mathrm{format}} + p_{\mathrm{tool}}] + \lambda_H H(\pi_{LL})
```

When LL uses `scorer`, both phases optimize task reward.  
When LL uses `entropy` or `entropy-hybrid`, optimization pressure shifts toward distributional properties unless task reward is explicitly mixed in.

---

## 1) `echo3BInst_hl_ll_entropy_grpo_hl_kl_true_ll_kl_false_reg_on_ll_scorer`

**Config delta**: `low_level_reward_strategy=scorer`, `high_level_update_repeats=1`.

**Behavior**:
- Best checkpoint around step 25; later training is noisy but not catastrophic.
- Reward dips recover instead of cascading.

**Main insight**:
- This remains the most stable baseline because HL and LL both optimize task quality.

---

## 2) `echo3BInst_hl_ll_entropy_grpo_hl_kl_true_ll_kl_false_reg_on_no_tool_neg_hl_2`

**Config delta**: `high_level_update_repeats=2`, `low_level_reward_strategy=entropy`.

**Behavior**:
- Peaks early, then degrades strongly with continued training.
- HL lengths grow and clipping/instability increase.

**Main insight**:
- `HL x2` plus entropy-driven LL is an unstable combination: HL drifts faster than LL can correct.

---

## 3) `echo3BInst_hl_ll_entropy_grpo_hl_kl_true_ll_kl_false_reg_on_band_first_select`

**Observed config (from launch yaml)**:
- `low_level_reward_strategy=entropy`
- `hl_kl_loss_coef=0.0`, `ll_kl_loss_coef=0.0`
- `ll_entropy_reg_coeff=0.001`
- `ll_no_tool_penalty=-0.05`

**Outcome**:
- Best val reward: **0.427 @ step 15**
- Final val reward: **0.318 @ step 115**

**Trajectory summary**:
- Fast early gain, then long noisy decay.
- Search success drops heavily over training (LL search success from ~0.70 early to ~0.04 late).
- Format-valid rates in training remain high, so the main issue is reward alignment, not pure formatting.

**Main insight**:
- Name says `hl_kl_true`, but run actually used `hl_kl_loss_coef=0.0`.  
  This run should be treated as a **KL-off entropy LL** run, not as KL-on.

---

## 4) `echo3BInst_hl_ll_entropy_grpo_hl_kl_false_ll_kl_false_reg_on_no_tool_neg`

**Observed config**:
- `low_level_reward_strategy=entropy`
- `hl_kl_loss_coef=0.0`, `ll_kl_loss_coef=0.0`
- `ll_entropy_reg_coeff=0.001`
- `ll_no_tool_penalty=-0.05`

**Outcome**:
- Best val reward: **0.465 @ step 60**
- Final val reward: **0.254 @ step 120**

**Trajectory summary**:
- Good mid-training peak, then clear post-peak decay.
- Search reliability falls sharply in later steps.
- Training valid rates improve over time, but reward still declines.

**Main insight**:
- Negative no-tool penalty helps less than expected when LL objective is still entropy-dominant.

---

## 5) `echo3BInst_hl_ll_entropy_grpo_hl_kl_true_ll_kl_false_reg_on_no_tool_zero`

**Observed config**:
- `low_level_reward_strategy=entropy`
- `hl_kl_loss_coef=0.001`, `ll_kl_loss_coef=0.0`
- `ll_no_tool_penalty=0.0`
- `ll_entropy_reg_coeff=0.001`

**Outcome**:
- Best val reward: **0.448 @ step 50**
- Final val reward: **0.275 @ step 130**

**Trajectory summary**:
- Similar shape to `no_tool_neg`: rise, then prolonged decay.
- Zeroing no-tool penalty does not prevent late instability.

**Main insight**:
- Moving `ll_no_tool_penalty` from `-0.05` to `0.0` changes behavior less than expected; LL reward type matters more than this penalty.

---

## 6) `echo3BInst_hl_ll_entropy_hybrid_grpo_norm_off_reg_on_kl_off`

**Observed config**:
- `low_level_reward_strategy=entropy-hybrid`
- `hl_kl_loss_coef=0.0`, `ll_kl_loss_coef=0.0`
- `ll_entropy_reg_coeff=0.01`

**Outcome**:
- Best val reward: **0.451 @ step 30**
- Final val reward: **-0.859 @ step 135**

**Trajectory summary**:
- Catastrophic late collapse after an initially strong peak.
- Search success trends to near zero in late training.

**Main insight**:
- **KL-off + entropy-hybrid + long training is unsafe** here; this is the clearest failure mode in the set.

---

## 7) `echo3BInst_hl_ll_entropy_hybrid_grpo_norm_off_reg_off_kl_true`

**Observed config**:
- `low_level_reward_strategy=entropy-hybrid`
- `hl_kl_loss_coef=0.001`, `ll_kl_loss_coef=0.001`
- `ll_entropy_reg_coeff=0.0`

**Outcome**:
- Best val reward: **0.482 @ step 115**
- Final val reward: **0.449 @ step 130**

**Trajectory summary**:
- High and sustained reward in late training.
- Much better post-peak retention than pure entropy runs.

**Main insight**:
- Turning KL on in both phases stabilizes entropy-hybrid behavior even without entropy regularization.

---

## 8) `echo3BInst_hl_ll_entropy_hybrid_grpo_norm_off_reg_on_kl_true`

**Observed config**:
- `low_level_reward_strategy=entropy-hybrid`
- `hl_kl_loss_coef=0.001`, `ll_kl_loss_coef=0.001`
- `ll_entropy_reg_coeff=0.01`

**Outcome**:
- Best val reward: **0.488 @ step 105** (best overall among listed runs)
- Final val reward: **0.435 @ step 135**

**Trajectory summary**:
- Strong late-stage performance with moderate post-peak drop.
- Significantly more stable than KL-off hybrid.

**Main insight**:
- This is currently the best tradeoff of peak quality + stability in the entropy-hybrid family.

---

## 9) `echo3BInst_hl_scorer_ll_hybrid_band_frozen_hinit_kl_on`

**Config delta** (launch: `training_config/echo_3B_ll_hl_dispo_band.yaml`):
- `high_level_reward_strategy=scorer`, `low_level_reward_strategy=entropy-hybrid`
- `ll_entropy_band_enable=true`, `ll_entropy_band_warmup_steps=1`
- `ll_entropy_band_epsilon_low=0.2`, `ll_entropy_band_epsilon_high=0.4`
- `hl_kl_loss_coef=0.001`, `ll_kl_loss_coef=0.001`, `ll_entropy_reg_coeff=0.01`
- Frozen `H_init` = mean phase policy entropy captured on global step 1 (not post-`</select>` token)

**Hypothesis**:
- Step 1 warmup + frozen phase entropy anchor fixes broken band (`entropy_in_band_rate` ~1% on old mismatched-mask anchor).
- Band on phase-level `H_bar` (same mask as `H_init`); actor reg stays on select mask for hybrid.

**W&B watch list**:
- End of warmup: `entropy_h_bar_mean` ≈ `entropy_h_init_frozen_mean`
- Step 2+: `entropy_in_band_rate` near 1.0 initially, then target 30–70% as policy moves
- `low_level/actor/entropy_reg_loss` still reflects select-only entropy (unchanged actor geometry)
- `val-core/DR_grpo_mix/reward/mean@1` ≥ 0.43 sustained
- Compare vs `ll_scorer`, `hybrid_kl_on` (#8), old `band_first_select`

**Note**: frozen ref is not persisted across checkpoint resume.

**Outcome**: *(pending run)*

---

## Cross-experiment takeaways

1. **LL reward choice dominates small penalties**  
   `scorer` or `entropy-hybrid + KL` is robust; pure `entropy` is prone to delayed drift.

2. **KL-on is the key stabilizer in hybrid runs**  
   Hybrid with KL off collapsed; hybrid with KL on delivered the top two final-quality trajectories.

3. **Do not trust run names without launch yaml**  
   At least one run (`band_first_select`) has suffix semantics that do not match realized config.

4. **Late training can destroy good checkpoints**  
   Multiple runs peak around steps 30-60 and degrade afterward; early stopping by val reward is mandatory.

5. **Search reliability is a recurring bottleneck**  
   Many runs show search success decaying toward zero in later phases, adding noise to tool-centric learning.

---

## Recommended next-run priorities

- Keep a **stable baseline** with `low_level_reward_strategy=scorer`.
- For entropy-hybrid experiments, keep **both KL terms on** (`hl_kl_loss_coef=0.001`, `ll_kl_loss_coef=0.001`).
- Avoid long training without aggressive model selection; checkpoint quality often peaks early.
- Enforce config-name consistency checks at launch so run names always reflect actual overrides.
