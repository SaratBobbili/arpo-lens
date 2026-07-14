---
name: Prompt5 HL LL Scope
overview: Multi-session ECHO cleanup — prompt-5 format, uniform scoring, simplify phase algos, remove DAPO/profiles. One subtask per chat; agent writes progress back into this plan before ending.
todos:
  - id: S1-format-validator
    content: "S1: Uniform validate_format + compute_score for prompt-5 (ARPO -1 early exits); drop HL/LL split validators and profile routing in deep_research_echo"
    status: completed
  - id: S2-drop-profiles-eval
    content: "S2: Delete leftover validator-profile APIs from ARPO echo twin + evaluation CLIs (evaluate.py / src/evaluator.py); metrics use shared format flag. Do not touch evaluation/xmix/"
    status: completed
  - id: S3-remove-dapo
    content: "S3: Remove DAPO/filter_groups machinery entirely (trainer, core_algos, reward_manager, train.sh, configs, delete dapo yaml)"
    status: completed
  - id: S4-simplify-phase-algos
    content: "S4: Drop reward_strategy + sign_cond_strategy; advantage algorithm + rollout strategy only; sign-cond clip uses advantage sign"
    status: completed
  - id: S5-wire-prompt5
    content: "S5: active_system_prompt via train.sh + training_config; sync system_prompt_5 into ARPO echo_system_prompts.yaml"
    status: completed
  - id: S6-remask-tool
    content: "S6: Rollout masks — replace first_select/select with free-text <tool>; update train.sh mask keys + echo_trainer defaults"
    status: completed
  - id: S7-regroup-configs
    content: "S7: Regroup training_config YAMLs (start echo_3B_ll_hl_grpo.yaml) into contiguous HL/LL blocks with cleaned knobs"
    status: completed
  - id: S8-eval-prompt5
    content: "S8: Eval launchers/defaults to prompt 5. evaluation/xmix/ is out of scope (leave untouched)"
    status: completed
isProject: false
---

# ECHO prompt-5 / HL-LL cleanup plan

## Session system prompt (follow at start of every new chat)

You are continuing the ECHO cleanup tracked in **`projectplan.md`** (workspace root).

**Rules for this session:**

1. Read this entire plan (system prompt, **Architecture map**, active subtask deep brief, locked decisions, Progress log).
2. Work on **exactly one** subtask: the first frontmatter todo with `status: pending` (or the id the user names). Do not start the next.
3. Before editing, open the files listed in that subtask’s deep brief and understand the current call graph — do not implement from the one-line todo alone.
4. Stay inside that subtask’s scope. Blockers / follow-ups go in the Progress log — do not silently expand scope.
5. Prefer the **live ECHO path** [`ECHO/training/`](ECHO/training/) + shared [`ARPO/verl_arpo_entropy/verl/`](ARPO/verl_arpo_entropy/verl/). Touch [`ECHO/training/`](ECHO/training/) only when the subtask says so or import drift would break.
6. Do not commit unless asked. Give `git add` paths when the user accepts changes.
7. Before ending: mark the todo completed (or leave pending + blocker), append a Progress log entry, hand off next pending id.

**Active subtask right now:** _(none — S1–S8 complete)_

---

## Architecture map (read before coding)

ECHO is hierarchical RL on top of verl/ARPO. Two training code trees exist:

| Path | Role |
|------|------|
| [`ECHO/training/`](ECHO/training/) | **Live** trainer used by [`ECHO/training/scripts/train.sh`](ECHO/training/scripts/train.sh) |
| [`ECHO/training/`](ECHO/training/) | Recipe twin (often drifts); keep in sync only when a subtask requires it |
| [`ARPO/verl_arpo_entropy/verl/`](ARPO/verl_arpo_entropy/verl/) | Shared library: rewards, rollout, dataset — **single copy** used by live ECHO |

### Data / prompt flow

1. Launch YAML under [`ECHO/training/training_config/`](ECHO/training/training_config/) → `train.sh` exports keys → Hydra [`echo_trainer.yaml`](ECHO/training/config/echo_trainer.yaml) + [`echo_system_prompts.yaml`](ECHO/training/config/echo_system_prompts.yaml).
2. [`rl_dataset.py`](ARPO/verl_arpo_entropy/verl/utils/dataset/rl_dataset.py) reads `data.active_system_prompt` → loads `system_prompt_{N}` into chat messages.
3. Rollout [`vllm_rollout_echo.py`](ARPO/verl_arpo_entropy/verl/workers/rollout/vllm_rollout/vllm_rollout_echo.py) generates tool-augmented trajectories, builds **phase loss masks** (`high_level_loss_mask` / `low_level_loss_mask`) from XML tags in `_TAG_INFO` / `_compute_hierarchical_masks`.
4. Reward: [`echo_reward_manager.py`](ECHO/training/echo_reward_manager.py) → [`deep_research_echo.compute_score`](ARPO/verl_arpo_entropy/verl/utils/reward_score/deep_research_echo.py) (S1 already rewrote this for prompt-5).
5. Trainer [`echo_ray_trainer.py`](ECHO/training/echo_ray_trainer.py) loops `phase_order`: generate → score → `compute_advantage` → actor update with phase mask. Actor [`echo_dp_actor.py`](ECHO/training/echo_dp_actor.py) applies PG clip / optional sign-cond / entropy reshape using `phase_strategy` meta.

### Prompt-5 trajectory schema (target)

```
<think>…</think>
<tool>rationale</tool><search|python>…</search|python><result>…</result>
…
<tool>no more tools</tool><answer>…\boxed{…}</answer>
```

SFT reference: [`scripts/sft_refactor/trajectory.py`](scripts/sft_refactor/trajectory.py) `verify_text`. Runtime validator after S1: `deep_research_echo.validate_format`.

### What already changed (S1–S2)

- Scoring is **phase-agnostic**: one `validate_format` + `format_valid`; no `high_level_valid` / `low_level_valid`; no validator profiles.
- Eval ([`evaluation/evaluate.py`](evaluation/evaluate.py), [`evaluation/src/evaluator.py`](evaluation/src/evaluator.py)) reports `echo_format_pass_rate` only; launchers no longer pass `ECHO_VALIDATOR_PROFILE`.
- Live + ARPO trainers log `reward/format_valid_rate`.
- **`evaluation/xmix/` is out of scope** — leave untouched (intentionally broken vs new API).

### Cleanup complete (S1–S8)

Eval math/qa launchers default to prompt 5 via live [`ECHO/training/config/echo_system_prompts.yaml`](ECHO/training/config/echo_system_prompts.yaml). **`evaluation/xmix/`** left untouched.

---

## Subtasks (deep briefs)

### S1 — Uniform format validator — COMPLETED

See Progress log. Touchstone file: [`deep_research_echo.py`](ARPO/verl_arpo_entropy/verl/utils/reward_score/deep_research_echo.py).

### S2 — Drop profiles + eval wiring — COMPLETED

See Progress log. Profiles/eval CLI cleaned; xmix skipped.

---

### S3 — Remove DAPO entirely — COMPLETED

See Progress log.

---

### S4 — Simplify phase algorithmic surface — COMPLETED

See Progress log.

---

### S5 — Wire `system_prompt_5` — COMPLETED

See Progress log.

---

### S6 — Remask free-text `<tool>` — COMPLETED

See Progress log. Touchstone: [`vllm_rollout_echo.py`](ARPO/verl_arpo_entropy/verl/workers/rollout/vllm_rollout/vllm_rollout_echo.py).

---

### S7 — Regroup launch YAMLs — COMPLETED

See Progress log. All 10 live `training_config/*.yaml` use shared → `# --- high_level ---` → `# --- low_level ---`.

---

### S8 — Eval defaults → prompt 5 — COMPLETED

See Progress log. Touchstone: math/qa `evaluation/run_*.sh` + `ECHO_ACTIVE_SYSTEM_PROMPT=5`.

---

## Locked decisions (do not reopen unless user says so)

- HL vs LL differ **only** by phase loss mask; scoring/format shared.
- Schema = system_prompt_5 (think / tool / search|python / result / final tool / answer+boxed).
- Format gate = ARPO-style −1 early exits then F1 (+ multi-tool bonus).
- Delete validator profiles; delete DAPO/filter_groups entirely.
- No reward_strategy / sign_cond_strategy; per phase: `advantage_algorithm` (`grpo|entropy|aepo`) + rollout strategy (default temperature). No separate `algorithm` estimator field. Sign-cond clip uses assigned advantage sign.
- Contiguous HL then LL blocks in launch YAMLs; explicit `active_system_prompt` via train.sh (target 5).
- **`evaluation/xmix/` out of scope** — leave untouched.

---

## Progress log

_Agents append here after each session._

### Template

```
### YYYY-MM-DD — <subtask-id> — completed|blocked
- Changes: …
- Follow-ups for later subtasks: …
- Next: <subtask-id>
```

### 2026-07-13 — S1-format-validator — completed
- Changes: Rewrote [`ARPO/verl_arpo_entropy/verl/utils/reward_score/deep_research_echo.py`](ARPO/verl_arpo_entropy/verl/utils/reward_score/deep_research_echo.py):
  - Single `validate_format` for prompt-5 (`think→tool→search|python|answer`, tool calls followed by `result`, answer after tool + `\boxed{}`), aligned with `scripts/sft_refactor/trajectory.verify_text`.
  - `compute_score` is phase-agnostic: −1 on format / missing answer / boxed parse fail; else F1 (+0.1 search+python bonus). Emits `format_valid` + `no_tool_calls` (no `high_level_valid` / `low_level_valid`).
  - Removed HL/LL validators, `VALIDATOR_PROFILE*`, `resolve_validator_profile`, `mask_categories_for_profile`, select-era helpers.
  - `__main__` smoke: good / no-tool / bad structure / multi-tool bonus / legacy select rejection — all passed.
- Follow-ups for later subtasks:
  - **S2**: live eval still imported old APIs; wire to shared `validate_format` / `format_valid`. Skip `evaluation/xmix/`.
- Next: S2-drop-profiles-eval

### 2026-07-13 — plan-amendment — locked
- Decision: carve **`evaluation/xmix/`** out of cleanup (leave untouched).
- Amendments: S2 no longer lists xmix CLIs; S8 renamed `S8-eval-prompt5`; locked decision updated.
- Next: S2-drop-profiles-eval

### 2026-07-13 — S2-drop-profiles-eval — completed
- Changes:
  - Live eval: [`evaluation/evaluate.py`](evaluation/evaluate.py) + [`evaluation/src/evaluator.py`](evaluation/src/evaluator.py) drop `--validator_profile` / `--mask_categories`; import shared `validate_format`; report only `echo_format_valid` / `echo_format_pass_rate`.
  - Eval launchers: strip `ECHO_VALIDATOR_PROFILE` / `VALIDATOR_PROFILE` from shells. **Left `evaluation/xmix/` untouched.**
  - ARPO twin: remove profile resolve/forward from recipe `echo_ray_trainer.py` / `echo_reward_manager.py`.
  - Metrics: format_valid → `reward/format_valid_rate`; dropped HL/LL valid-rate metrics.
- Follow-ups:
  - **S3**: remove DAPO machinery (this plan’s deep brief).
  - Docs drift: `logging_readme.md` may still mention HL/LL valid rates — optional later.
- Next: S3-remove-dapo

### 2026-07-13 — plan-amendment — deepen subtask briefs
- Replaced shallow subtask table with Architecture map + per-subtask deep briefs (files, call graph, done criteria).
- Confirmed S1/S2 completed; active = S3.
- Next: S3-remove-dapo

### 2026-07-13 — S3-remove-dapo — completed
- Changes:
  - Live: removed `_collect_phase_batch_dapo`, dapo train-loop branch, `_phase_algorithm`; algorithm validate is `grpo` only ([`echo_ray_trainer.py`](ECHO/training/echo_ray_trainer.py)).
  - Deleted `filter_informative_groups` / `_ECHO_FILTER_METRICS` ([`echo_core_algos.py`](ECHO/training/echo_core_algos.py)).
  - Dropped `DAPORewardManager` import/branch ([`echo_reward_manager.py`](ECHO/training/echo_reward_manager.py)).
  - Removed `filter_groups` from Hydra defaults + `train.sh` keys/overrides; stripped filter keys from launch YAMLs; deleted `echo_3B_ll_hl_dapo.yaml`.
  - Mirrored same removals under [`ECHO/training/`](ECHO/training/) (incl. both dapo yaml copies).
- Left historical `"ll_grpo_hl_dapo"` name matching in analysis report scripts (non-functional).
- Follow-ups: **S4** simplify reward_strategy / sign_cond_strategy.
- Next: S4-simplify-phase-algos

### 2026-07-14 — S4-simplify-phase-algos — completed
- Changes (live [`ECHO/training/`](ECHO/training/) only):
  - Single per-phase knob `advantage_algorithm` (`grpo|entropy|aepo`); dropped separate `algorithm` estimator field and old `scorer` name (`scorer` → `grpo`).
  - Launch keys: `*_reward_strategy` → `*_advantage_algorithm`; removed `*_algorithm` / `*_sign_cond_strategy` from `train.sh` + Hydra + all `training_config/*.yaml`.
  - Actor: `clip_sign_advantages` = same PG advantages when `use_sign_cond_clip`; no parallel sign channel ([`echo_dp_actor.py`](ECHO/training/echo_dp_actor.py)).
  - Trainer meta_info: `advantage_algorithm` replaces `phase_strategy` / `sign_cond_strategy` ([`echo_ray_trainer.py`](ECHO/training/echo_ray_trainer.py)).
  - Left `phase_rollouts.*.strategy` (`default|aepo`) unchanged.
- Follow-ups:
  - **S5**: wire `active_system_prompt=5`.
  - ARPO recipe twin still has old `reward_strategy` / `sign_cond_strategy` surface — sync later if needed.
  - `scripts/old/` + analysis report still read legacy `strategy` key (non-blocking).
- Next: S5-wire-prompt5

### 2026-07-14 — S5-wire-prompt5 — completed
- Changes:
  - [`train.sh`](ECHO/training/scripts/train.sh): `active_system_prompt` in `VALID_LAUNCH_KEYS`; Hydra override `data.active_system_prompt="${ACTIVE_SYSTEM_PROMPT:-5}"`.
  - Defaults: `data.active_system_prompt: 5` in live + ARPO [`echo_trainer.yaml`](ECHO/training/config/echo_trainer.yaml).
  - All 10 live [`training_config/*.yaml`](ECHO/training/training_config/) set `active_system_prompt: 5`.
  - Synced `system_prompt_5` into ARPO [`echo_system_prompts.yaml`](ECHO/training/config/echo_system_prompts.yaml) (identical to ECHO copy).
- Verified: launch resolve → `data.active_system_prompt=5`; both prompt YAMLs define `_5`.
- Follow-ups: **S6** remask free-text `<tool>`; drop `mask_first_select` / `mask_select`.
- Next: S6-remask-tool

### 2026-07-14 — S6-remask-tool — completed
- Changes:
  - [`vllm_rollout_echo.py`](ARPO/verl_arpo_entropy/verl/workers/rollout/vllm_rollout/vllm_rollout_echo.py): `_TAG_INFO` uses `<tool>`; drop select-era `first_select`/`select` counting; emit `tool_loss_mask` only (removed `select_loss_mask` / `first_select_loss_mask` / `first_select_post_idx`).
  - Live launch: `mask_tool` replaces `mask_first_select`/`mask_select` in [`train.sh`](ECHO/training/scripts/train.sh), [`echo_trainer.yaml`](ECHO/training/config/echo_trainer.yaml), all 10 `training_config/*.yaml` (migrated `mask_tool` from former `mask_select` value).
  - ARPO twin: same Hydra/train.sh/training_config mask keys; `select_loss_mask` → `tool_loss_mask` in recipe [`echo_ray_trainer.py`](ECHO/training/echo_ray_trainer.py).
- Verified: char-token smoke on prompt-5 trajectory — `<tool>` content follows `mask_tool` into HL/LL; results excluded; launch key check on `echo_3B_ll_hl_grpo.yaml` passes.
- Follow-ups: **S7** regroup launch YAMLs into contiguous HL/LL blocks; left `scripts/old/` + analysis report select naming untouched.
- Next: S7-regroup-configs

### 2026-07-14 — S7-regroup-configs — completed
- Changes: Regrouped all 10 live [`ECHO/training/training_config/*.yaml`](ECHO/training/training_config/) into contiguous layout:
  - `# shared` (project/data/rollout infra, phase_order, masks, clip ratios, norm_adv)
  - `# --- high_level ---` (budget, updates, advantage_algorithm, sign_cond clip, hl_kl/entropy/aepo_clip)
  - `# --- low_level ---` (same family + `ll_aepo_*` only when `low_level_rollout_strategy=aepo`)
  - Values preserved; no obsolete keys (`filter_groups` / `reward_strategy` / `sign_cond_strategy` / `mask_first_select`).
- Verified: all 10 pass `train.sh` `VALID_LAUNCH_KEYS` check.
- Follow-ups: **S8** eval launchers/defaults → prompt 5; leave `evaluation/xmix/` untouched. ARPO recipe `training_config/` not synced (live-only this subtask).
- Next: S8-eval-prompt5

### 2026-07-14 — S8-eval-prompt5 — completed
- Changes: 15 math/qa eval launchers under [`evaluation/`](evaluation/):
  - `ECHO_SYSTEM_PROMPT_YAML` → live [`ECHO/training/config/echo_system_prompts.yaml`](ECHO/training/config/echo_system_prompts.yaml)
  - `ECHO_ACTIVE_SYSTEM_PROMPT=5` (was 1 or 4)
  - Comments: `<select>/<tool>` → prompt-5 schema; format-token notes use `<tool>`
  - Left [`evaluation/xmix/`](evaluation/xmix/) untouched (still defaults to 1)
- Verified: YAML loads `system_prompt_5` with `<tool>`, no `<select>`; all 15 launchers export `=5`
- Follow-ups (optional, out of this plan): `tree_hca_eval/run_math_echo.sh` + `main.sh` still default to prompt 1
- Next: _(plan complete)_
