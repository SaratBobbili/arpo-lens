# ECHO training notes

Launch template: `training_config/echo_3B_ll_hl_grpo.yaml` via `scripts/train.sh`.

## Training loop

Nested GRPO. One outer cycle = `phases.low_level.num_iters` low-level iterations
followed by one high-level iteration; the run is `phases.high_level.num_iters`
outer cycles, so `total_training_steps = N_HL * (N_LL + 1)`.

Each iteration fetches its own prompt chunk, rolls out, scores, computes GRPO
advantages and takes its phase's optimizer step(s). The chunk is fetched *inside*
the iteration, so every rollout comes from the policy left by the previous step
(`generate_sequences` resyncs FSDP → vLLM).

Validation, best-checkpoint sync and `_save_checkpoint` only run on the
high-level iteration that closes a cycle; `test_freq` / `save_freq` count outer
cycles, never low-level iterations. Resume therefore always lands on a cycle
boundary.

## Prompt streams and batch sizes

Each phase has its own shuffled `StatefulDataLoader` over the same
`train_dataset` (different sampler seed), cycling with reshuffle on exhaustion.
Batch size is **derived, not configured**:

```
batch_size = (len(train_dataset) // num_iters) // world_size * world_size
```

with `drop_last=True`. The floor to a multiple of `world_size` is required
because every worker-group call chunks the batch across ranks. So
`N_LL` inner iterations ≈ one LL pass over the dataset per outer cycle, and
`N_HL` outer iterations ≈ one HL pass over the whole run. GPU packing is a
config choice: pick `num_iters`, `group_size` and the PPO batch splits so one
iteration fits. There is no prompt chunking / rollout-side accumulation.

## Per-phase optimizers

Two `AdamW` instances (plus schedulers) over the **same** FSDP parameters, with
independent moments and LR schedules, built in `echo_fsdp_workers`. Routing is by
`meta_info['phase']`. Scheduler horizons are derived: HL `= N_HL`,
LL `= N_HL * N_LL`. Both are checkpointed through `_PhaseStateShim`; a
checkpoint missing a phase fails loud on resume. `actor_rollout_ref.actor.optim`
and `actor_rollout_ref.actor.ppo_*` are inherited from `ppo_trainer.yaml` and
unused by ECHO.

## Current API (source of truth)

- **Reward** is always scorer-based (`deep_research_echo.compute_score`): format gate → F1 (+ multi-tool bonus). No per-phase reward strategy channel.
- **Everything a phase owns** lives under `phases.{high_level,low_level}`: `num_iters`, `group_size` (GRPO trajectories per prompt), `ppo_mini_batch_size` / `ppo_micro_batch_size_per_gpu` (normalized with *that phase's* `group_size`), `optim.*`, `rollout.{strategy,aepo}`, and the loss knobs `advantage_algorithm`, `kl_loss_coef`, `use_sign_cond_clip`, `use_aepo_clip`, `entropy.*`.
- **Per-phase advantage:** `phases.*.advantage_algorithm ∈ {grpo, entropy, aepo}`.
- **Per-phase rollout sampling:** `phases.*.rollout.strategy ∈ {default, aepo}`.
- **Schema:** `data.active_system_prompt=1` (think / tool / search|python / result / answer+boxed).
- HL vs LL differ only by phase loss mask (`actor_rollout_ref.rollout.mask_categories`); scoring/format is shared.
- Sign-cond clip (`use_sign_cond_clip`) uses the assigned advantage sign; optional AEPO ratio clip composes independently.
- Both phases always run: `1 <= num_iters <= len(train_dataset)` and `group_size >= 1` are asserted. No skip path.
- No DAPO-style grouping filters or validator-profile configs.

See `config/echo_trainer.yaml` and `docs/logging_readme.md` for knobs and metrics.
