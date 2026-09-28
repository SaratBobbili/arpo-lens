# ECHO training notes

Launch template: `recipe/echo/profiles/echo_3B_ll_hl_grpo.yaml` via `scripts/train.sh`.

Layout (since 2026-09-27): `core/` holds the shared, algorithm-free code (`phase_trainer.py`
`PhaseTrainerBase`, `phase_actor.py` `PhaseActorBase`, `phase_workers.py` `PhaseWorkerBase`,
`core_algos.py`, `reward_manager.py`). Each algorithm is a recipe under `recipe/<name>/` with
its own `main.py`, `trainer.py` (the loop: fit, cycle structure, which prompts a phase
iteration draws, step accounting), `workers.py`, `actor.py` and `response.py` where it has a
response term, plus `profiles/*.yaml` and its `sbatch_*.sh`. Recipes: `alt_grpo` (alternating
role-masked GRPO), `echo` (Algorithm 1, adjoint / exact estimators; also the response-off r3
profile), `aho` (Algorithm 1 round structure with the AHO surrogate; a full copy, it never
imports `recipe/echo`). A profile's `recipe:` key picks `python3 -m training.recipe.<name>.main`
in `scripts/train.sh`; the hydra base stays in `config/`. No loop code is shared between recipes.
The alt_grpo cycle is `ll_num_iters` low-level then `hl_num_iters` high-level iterations (1/1 in its
profile, the April-May 2026 schema; the high-level iteration re-rolls the low-level prompts), with
cycles per epoch = batches per pass // `ll_num_iters` and save/test_freq counting cycles.

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
independent moments and LR schedules, built in `core/phase_workers.py` (`_build_phase_optimizer`). Routing is by
`meta_info['phase']`. Scheduler horizons are derived: HL `= N_HL`,
LL `= N_HL * N_LL`. Both are checkpointed through `_PhaseStateShim`; a
checkpoint missing a phase fails loud on resume. `actor_rollout_ref.actor.optim`
and `actor_rollout_ref.actor.ppo_*` are inherited from `ppo_trainer.yaml` and
unused by ECHO.

## Response-term estimators (`phases.response.estimator`)

`gradient: true` adds `g_resp` to the leader step; the estimator decides how it is built.

- **`adjoint`** (default): ECHO Algorithm 1's K-step reverse sweep over stashed follower
  records (Eqs. 10-11, App. C.4). Costs 1 + 2K extra full-batch passes and K host weight
  snapshots per round; `exact`, `curvature`, `group_aligned`, `fd_rel`, `replay_fraction`
  and the SGD follower requirement all belong to it.
- **`aho`**: arXiv:2607.28849 (Approximate Hypergradient Optimization), GRPO variant. An
  entropy-regularised follower's optimum is Boltzmann, so its sensitivity to the leader is
  `(grad_x Q - grad_x V) / tau` with no Hessian. In ECHO the leader enters the follower's
  problem through the reasoning segments between tool tokens (the transition kernel), and
  with the follower's own group advantage `A_L` standing in for the value and the sum
  over follower steps telescoped (deterministic LLM transitions keep `U`, so each segment
  is counted once, not once per tool token before it) the whole term is one surrogate on
  the leader batch: *reinforce each reasoning token after the first tool call with weight*
  `coef * A_H * A_L / tau`. One extra leader-batch pass; no records,
  snapshots, replay or HVP; the follower may stay on AdamW. Needs the follower scored by
  `R_L` (`follower_return`), `grpo` advantages in both phases and a temperature
  (`ll_entropy_enabled: true` with `ll_entropy_reg_coeff = tau`, or `aho_tau`). The
  reasoning before the first tool call gets weight 0; the reasoning after the last call is
  weighted (R_L is gated on the whole response's format). It targets the exact response
  `xi(x)`, evaluated at `y_K`, so its accuracy is `||y_K - xi(x)||`. Derivation and the
  named approximations: `recipe/aho/response.py` module docstring. Launch profile:
  `recipe/aho/profiles/config_aho_k8.yaml`.

Both leave `p.grad = -g_resp` before the direct pass, so `high_level/actor/response_norm`,
`response_to_direct_ratio` and `response_direct_cosine` mean the same thing under either.

## Current API (source of truth)

- **Metrics layout (2026-09-27).** Keys are namespaced by what the number is computed on, not by which phase's step it landed in. `reward/*` (reward_mean, f1_mean, whole-schema format_valid_rate, fail_* buckets, in-group std), `rollout/*` (lengths, tool calls, budget exhaustion, no-tool rate, infra excision) and `policy/entropy_{reasoning,tool}` (pre-update entropy on the think/answer vs tool/search/python token populations) are one dense series each, logged every step with `train/phase` (0 follower, 1 leader) as the marker. Only gate-, mask- or advantage-dependent numbers carry a phase prefix: `{phase}/reward_mean`, `{phase}/gate_pass_rate`, `{phase}/budget_*`, `{phase}/actor/*` (pg_loss, grad_norm pre-clip, step_skipped, ppo_kl, clipfracs, entropy_reg_loss, advantage_{mean,std,penalty_frac} measured on the advantages the update actually used), `high_level/response/*` (norm, direct_norm, ratio, cosine, ...) and `high_level/aho/*`. `val-core/*`, `val-aux/*` (ARPO's split), `perf/*`, `train/*` are shared. Dropped: all-token `policy/entropy`, the rank-0 `tools_*` counters, driver-side `advantage_std`, `bad_format_rate`, constant coefs, and the best-checkpoint bookkeeping beyond `train/best_checkpoint_{value,step}`. `logging_data/<key>.jsonl` mirrors every key.
- **Reward** is always scorer-based (`deep_research_echo.compute_score`), the same for both phases and every algorithm family: phase-owned format gate → F1 (+ multi-tool bonus), or -1. The gate (2026-09-27) checks each tag on its own row and a phase fails only on rows of the tags it trains per `rollout.mask_categories`: leader think (closed; after start or a result) and answer (closed; after a tool; exactly one; `\boxed{}`); follower tool (closed; after a think or a result, so calls may chain) and search/python (closed; after a tool; followed by a result). `<result>` gates nobody. A follower-scored trajectory with no answer gets 0, not -1. `format_valid` stays the whole-schema bit for metrics; `phase_format_valid` is what the score used and rides to the actor as a batch tensor; `format_issues` lists every violation with its owner. `phases.response.follower_return` and the tool-validity R_L are retired. Advantage side: grpo carries the -1 in the score; entropy mode sets H_t = -1 on a failed sample's tokens before normalising; aepo keeps the -1 in the score and lets entropy modulate.
- **Everything a phase owns** lives under `phases.{high_level,low_level}`: `num_iters`, `group_size` (GRPO trajectories per prompt), `ppo_mini_batch_size` / `ppo_micro_batch_size_per_gpu` (normalized with *that phase's* `group_size`), `optim.*`, `rollout.{strategy,aepo}`, and the loss knobs `advantage_algorithm`, `kl_loss_coef`, `use_sign_cond_clip`, `use_aepo_clip`, `entropy.*`.
- **Per-phase advantage:** `phases.*.advantage_algorithm ∈ {grpo, entropy, aepo}`.
- **Per-phase rollout sampling:** `phases.*.rollout.strategy ∈ {default, aepo}`.
- **Schema:** `data.active_system_prompt=1` (think / tool / search|python / result / answer+boxed).
- HL vs LL differ only by phase loss mask (`actor_rollout_ref.rollout.mask_categories`); scoring/format is shared.
- Sign-cond clip (`use_sign_cond_clip`) uses the assigned advantage sign; optional AEPO ratio clip composes independently.
- Both phases always run: `1 <= num_iters <= len(train_dataset)` and `group_size >= 1` are asserted. No skip path.
- No DAPO-style grouping filters or validator-profile configs.

See `config/echo_trainer.yaml` and `docs/logging_readme.md` for knobs and metrics.
