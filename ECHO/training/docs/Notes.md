# ECHO training notes

Launch template: `training_config/echo_3B_ll_hl.yaml` via `scripts/train.sh`.

## Current API (source of truth)

- **Reward** is always scorer-based (`deep_research_echo.compute_score`): format gate → F1 (+ multi-tool bonus). No per-phase reward strategy channel.
- **Per-phase advantage:** `reward_model.phase_rewards.*.advantage_algorithm ∈ {grpo, entropy, aepo}`.
- **Per-phase rollout sampling:** `actor_rollout_ref.rollout.phase_rollouts.*.strategy ∈ {default, aepo}`.
- **Schema:** `data.active_system_prompt=5` (think / tool / search|python / result / answer+boxed).
- HL vs LL differ only by phase loss mask (`mask_categories`); scoring/format is shared.
- Sign-cond clip (`use_sign_cond_clip`) uses the assigned advantage sign; optional AEPO ratio clip composes independently.
- No DAPO-style grouping filters or validator-profile configs.

See `config/echo_trainer.yaml` and `docs/logging_readme.md` for knobs and metrics.
