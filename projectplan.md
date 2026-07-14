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
    status: pending
  - id: S4-simplify-phase-algos
    content: "S4: Drop reward_strategy + sign_cond_strategy; advantage algorithm + rollout strategy only; sign-cond clip uses advantage sign"
    status: pending
  - id: S5-wire-prompt5
    content: "S5: active_system_prompt via train.sh + training_config; sync system_prompt_5 into ARPO echo_system_prompts.yaml"
    status: pending
  - id: S6-remask-tool
    content: "S6: Rollout masks — replace first_select/select with free-text <tool>; update train.sh mask keys + echo_trainer defaults"
    status: pending
  - id: S7-regroup-configs
    content: "S7: Regroup training_config YAMLs (start echo_3B_ll_hl_grpo.yaml) into contiguous HL/LL blocks with cleaned knobs"
    status: pending
  - id: S8-eval-prompt5
    content: "S8: Eval launchers/defaults to prompt 5. evaluation/xmix/ is out of scope (leave untouched)"
    status: pending
isProject: false
---

# ECHO prompt-5 / HL-LL cleanup plan

## Session system prompt (paste / follow at start of every new chat)

You are continuing the ECHO cleanup tracked in this plan file:
`projectplan.md` (workspace root)

**Rules for this session:**

1. Read this entire plan first (system prompt, active subtask, locked decisions, progress log).
2. Work on **exactly one** subtask: the first with status `pending` in frontmatter todos (or the id the user names). Do not start the next.
3. Stay inside that subtask’s scope. If you discover blockers or needed follow-ups, note them in the Progress log — do not silently expand scope.
4. Prefer editing the **live ECHO path** under [`ECHO/training/`](ECHO/training/) and shared [`ARPO/verl_arpo_entropy/verl/`](ARPO/verl_arpo_entropy/verl/). Touch the ARPO recipe echo twin only when the subtask says so or drift would break imports.
5. Do not commit unless the user asks. Give `git add` paths when the user accepts changes.
6. When the subtask is done (or blocked), **update this plan file before ending**:
   - Set that todo `status: completed` (or leave `pending` and explain blocker).
   - Append a Progress log entry: what changed (paths), what was deferred, any amendments for later subtasks.
7. End the chat with a one-paragraph handoff: subtask id, done / blocked, and the next pending subtask id.

**Active subtask right now:** `S3-remove-dapo`

---

## Subtasks (one per chat session)

| ID | Name | Done when |
|----|------|-----------|
| **S1** | Uniform format validator | Single `validate_format` for prompt-5; `compute_score` uses ARPO-style −1 early exits for all phases; no HL/LL format split; unit/self-check or `__main__` smoke still passes |
| **S2** | Drop profiles + eval wiring | No `VALIDATOR_PROFILE*` / `resolve_validator_profile` / `--validator_profile` in ARPO twin + live eval; shared `format_valid`. Skip `evaluation/xmix/` |
| **S3** | Remove DAPO | No `dapo` / `filter_groups` / `filter_informative_groups` / `_collect_phase_batch_dapo`; configs and `train.sh` cleaned; `echo_3B_ll_hl_dapo.yaml` deleted |
| **S4** | Simplify phase algos | No reward_strategy / sign_cond_strategy; per phase: advantage algorithm + rollout strategy; sign-cond clip uses assigned advantage sign |
| **S5** | Wire prompt 5 | `active_system_prompt` in `train.sh` + configs; ARPO prompt YAML has `system_prompt_5` |
| **S6** | Remask `<tool>` | Rollout `_TAG_INFO` / hierarchical masks use free-text `tool`; drop `first_select`/`select` mask keys from launch surface |
| **S7** | Regroup configs | Launch YAMLs: shared keys then contiguous HL block then LL block; start with `echo_3B_ll_hl_grpo.yaml` |
| **S8** | Eval prompt 5 | Eval launchers/defaults use prompt 5; **do not** port or delete xmix |

Depends-on order: S1 → S2; S3 independent after S1 or parallel only if no file conflicts; S4 after S3 preferred (same trainer/config files); S5–S7 after S4; S6 can follow S5; S8 last.

---

## Locked decisions (do not reopen unless user says so)

### Design

- HL vs LL differ **only** by phase loss mask. Scoring/format are shared.
- Schema = [`system_prompt_5`](ECHO/training/config/echo_system_prompts.yaml): think → tool → search\|python → result → … → final tool → answer + `\boxed{}`.
- Format gate mirrors ARPO [`deep_research.compute_score`](ARPO/verl_arpo_entropy/verl/utils/reward_score/deep_research.py): −1 on bad `validate_format`, missing answer extract, or boxed parse fail; else F1 (+ existing multi-tool bonus).
- Delete validator profiles (c1–c4) entirely.
- Delete DAPO / filter_groups machinery entirely (not just disable).
- No `*_reward_strategy` / `sign_cond_strategy`. Per phase: **advantage algorithm** + **rollout strategy** (default temperature / `default`). Sign-cond clip (if enabled) uses sign of the token’s assigned advantage.
- Launch configs: group phase knobs in contiguous HL then LL blocks.
- Explicit `active_system_prompt` via `train.sh` (target `5`).
- **`evaluation/xmix/` is out of scope** for this cleanup. It is a dead select-era cross-checkpoint mix experiment (not on the training/eval critical path). Leave the folder untouched; do not port the splicer to prompt-5 and do not delete it in these subtasks.

### Key files

- Format/score: [`ARPO/.../deep_research_echo.py`](ARPO/verl_arpo_entropy/verl/utils/reward_score/deep_research_echo.py) (reuse helpers from [`deep_research.py`](ARPO/verl_arpo_entropy/verl/utils/reward_score/deep_research.py); SFT invariants in [`scripts/sft_refactor/trajectory.py`](scripts/sft_refactor/trajectory.py) `verify_text`)
- Trainer/actor: [`ECHO/training/echo_ray_trainer.py`](ECHO/training/echo_ray_trainer.py), [`echo_dp_actor.py`](ECHO/training/echo_dp_actor.py), [`echo_core_algos.py`](ECHO/training/echo_core_algos.py), [`echo_reward_manager.py`](ECHO/training/echo_reward_manager.py)
- Configs: [`ECHO/training/config/echo_trainer.yaml`](ECHO/training/config/echo_trainer.yaml), [`ECHO/training/scripts/train.sh`](ECHO/training/scripts/train.sh), [`ECHO/training/training_config/`](ECHO/training/training_config/)
- Masks: [`vllm_rollout_echo.py`](ARPO/verl_arpo_entropy/verl/workers/rollout/vllm_rollout/vllm_rollout_echo.py)

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
  - **S2**: live eval (`evaluate.py` / `src/evaluator.py`) still imports `validate_format_echo` / `mask_categories_for_profile`; ARPO recipe twin still calls `resolve_validator_profile`. Wire to shared `validate_format` / `format_valid` and drop `--validator_profile`. Trainer metrics that read `high_level_valid`/`low_level_valid` → `format_valid`. **Skip `evaluation/xmix/`** (may already be broken after S1; intentional).
- Next: S2-drop-profiles-eval

### 2026-07-13 — plan-amendment — locked
- Decision: carve **`evaluation/xmix/`** out of cleanup (user chose leave-untouched).
- Role of xmix (for handoff): one-off cross-checkpoint mixed-prefix harness — cand1/cand2 greedy rollouts → select-era HL/LL splice (`splicer.py`) → prefix continuation → judge. Not used by standard `run_*` eval jobs; `xmix_runs/` empty.
- Amendments: S2 no longer lists xmix CLIs; S8 renamed `S8-eval-prompt5` (prompt-5 eval only); locked decision + S1 follow-up updated.
- Next: S2-drop-profiles-eval

### 2026-07-13 — S2-drop-profiles-eval — completed
- Changes:
  - Live eval: [`evaluation/evaluate.py`](evaluation/evaluate.py) + [`evaluation/src/evaluator.py`](evaluation/src/evaluator.py) drop `--validator_profile` / `--mask_categories`; import shared `validate_format`; report only `echo_format_valid` / `echo_format_pass_rate`.
  - Eval launchers: strip `ECHO_VALIDATOR_PROFILE` / `VALIDATOR_PROFILE` from `echo_evaluate_passk_math_4qa.sh`, `run_layout.sh`, and 15 `run_*` / `high*` shells. **Left `evaluation/xmix/` untouched.**
  - ARPO twin: remove `resolve_validator_profile` + `meta_info["validator_profile"]` from [`recipe/echo/echo_ray_trainer.py`](ARPO/verl_arpo_entropy/recipe/echo/echo_ray_trainer.py); stop forwarding profile in [`echo_reward_manager.py`](ARPO/verl_arpo_entropy/recipe/echo/echo_reward_manager.py).
  - Metrics: ECHO live + ARPO twin trainers + both `plot_training_log.py` now use `format_valid` → `reward/format_valid_rate` (JSONL `format_valid_rate.jsonl`); dropped HL/LL valid-rate metrics.
- Follow-ups for later subtasks:
  - **S3**: remove DAPO / `filter_groups` / `_collect_phase_batch_dapo` / dapo yaml (`evaluation/xmix` still broken vs S1 API — intentional).
  - Docs drift: `ECHO/training/docs/logging_readme.md` (+ ARPO twin twin) still describe HL/LL valid rates — optional cleanup later.
- Next: S3-remove-dapo


