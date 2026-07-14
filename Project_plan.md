---
name: ECHO single-tree cleanup
overview: Multi-session ECHO cleanup. Goal — one live training tree (ECHO/training/) + shared ARPO/verl runtime; delete the diverged recipe/echo twin; purge stale live leftovers; then prune inactive prompts to system_prompt_1 and inventory dead shared verl. One subtask per chat; agent writes progress back into this plan before ending.
todos:
  - id: S1-remove-recipe-echo-twin
    content: "S1: Delete ARPO/.../recipe/echo/ entirely; retarget eval/tree_hca/xmix + ECHO/training path refs to ECHO/training/; do not touch ARPO/.../verl/"
    status: completed
  - id: S2-purge-stale-live
    content: "S2: Inventory and remove stale code/vars/docs/config keys in live ECHO/training (and related analysis) that belong to pre-advantage_algorithm / pre-prompt-5 APIs; leave shared verl untouched unless a later subtask says so"
    status: completed
  - id: S3-prune-prompts-identify-verl
    content: "S3: Prune system_prompt_1–4; rename system_prompt_5→system_prompt_1; retarget live defaults 5→1; inventory (do not edit) dead shared ECHO code under ARPO/.../verl/"
    status: completed
---

# ECHO single-tree cleanup plan

## Session system prompt (follow at start of every new chat)

You are continuing the ECHO cleanup tracked in **`Project_plan.md`** (workspace root).

**Rules for this session:**

1. Read this entire plan (system prompt, **Architecture map**, active subtask deep brief, locked decisions, Progress log).
2. Work on **exactly one** subtask: the first frontmatter todo with `status: pending` (or the id the user names). Do not start the next.
3. Before editing, open the files listed in that subtask’s deep brief and understand the current call graph — do not implement from the one-line todo alone.
4. Stay inside that subtask’s scope. Blockers / follow-ups go in the Progress log — do not silently expand scope.
5. Prefer the **live ECHO path** [`ECHO/training/`](ECHO/training/) + shared [`ARPO/verl_arpo_entropy/verl/`](ARPO/verl_arpo_entropy/verl/). The diverged twin [`ARPO/verl_arpo_entropy/recipe/echo/`](ARPO/verl_arpo_entropy/recipe/echo/) is being **removed** (S1); do not revive it.
6. **Do not edit shared runtime** under `ARPO/verl_arpo_entropy/verl/` unless a later subtask explicitly says so.
7. Do not commit unless asked. Give `git add` paths when the user accepts changes.
8. Before ending: mark the todo completed (or leave pending + blocker), append a Progress log entry, hand off next pending id.

**Active subtask right now:** _(none — S3 completed; await user-named S4+)_

**Later subtasks:** Further S4+ only when the user names them.

---

## Goal

Finish consolidating ECHO onto a **single, current** training package:

| Keep | Remove |
|------|--------|
| [`ECHO/training/`](ECHO/training/) — live recipe | [`ARPO/verl_arpo_entropy/recipe/echo/`](ARPO/verl_arpo_entropy/recipe/echo/) — diverged twin (S1) |
| [`ARPO/verl_arpo_entropy/verl/`](ARPO/verl_arpo_entropy/verl/) — shared rollout / scorer / tools / FSDP | Stale live leftovers from retired APIs (S2) |

Prior work already shipped prompt-5 format, phase-agnostic scoring, no DAPO, `advantage_algorithm` API, remasked `<tool>`, regrouped launch YAMLs, and math/qa eval prompt-5 defaults. Twin still exists and still uses the old `reward_strategy` surface — S1 deletes it. S2 then purges anything left in the live tree that only made sense under those retired surfaces.

---

## Architecture map (read before coding)

ECHO is hierarchical RL on top of verl/ARPO.

| Path | Role |
|------|------|
| [`ECHO/training/`](ECHO/training/) | **Live** trainer used by [`ECHO/training/scripts/train.sh`](ECHO/training/scripts/train.sh) |
| [`ARPO/verl_arpo_entropy/verl/`](ARPO/verl_arpo_entropy/verl/) | Shared library: rewards, rollout, dataset — **single copy** used by live ECHO |
| [`ARPO/verl_arpo_entropy/recipe/echo/`](ARPO/verl_arpo_entropy/recipe/echo/) | **Diverged twin** (old `strategy` / entropy-as-reward API) — **delete in S1** |

### Data / prompt flow (live)

1. Launch YAML under [`ECHO/training/training_config/`](ECHO/training/training_config/) → `train.sh` → Hydra [`echo_trainer.yaml`](ECHO/training/config/echo_trainer.yaml) + [`echo_system_prompts.yaml`](ECHO/training/config/echo_system_prompts.yaml).
2. [`rl_dataset.py`](ARPO/verl_arpo_entropy/verl/utils/dataset/rl_dataset.py) reads `data.active_system_prompt` → `system_prompt_{N}`.
3. Rollout [`vllm_rollout_echo.py`](ARPO/verl_arpo_entropy/verl/workers/rollout/vllm_rollout/vllm_rollout_echo.py) builds phase loss masks from prompt-5 tags.
4. Reward: [`echo_reward_manager.py`](ECHO/training/echo_reward_manager.py) → [`deep_research_echo.compute_score`](ARPO/verl_arpo_entropy/verl/utils/reward_score/deep_research_echo.py).
5. Trainer [`echo_ray_trainer.py`](ECHO/training/echo_ray_trainer.py) loops `phase_order`: generate → score → GRPO advantage on phase mask → actor update. Actor [`echo_dp_actor.py`](ECHO/training/echo_dp_actor.py) applies clip / optional entropy–AEPO advantage reshape via `advantage_algorithm`.

### Twin API mismatch (why delete)

| | Live `ECHO/training/` | Twin `recipe/echo/` |
|--|--|--|
| Entropy | Advantage: `advantage_algorithm ∈ {grpo,entropy,aepo}` | Reward: `strategy ∈ {scorer,entropy,entropy-hybrid}` |
| Entry | `python -m training.main_echo` | `python -m recipe.echo.main_echo` |

### Shared runtime (never touch in S1; S2 default = same)

- [`vllm_rollout_echo.py`](ARPO/verl_arpo_entropy/verl/workers/rollout/vllm_rollout/vllm_rollout_echo.py)
- [`deep_research_echo.py`](ARPO/verl_arpo_entropy/verl/utils/reward_score/deep_research_echo.py)
- [`fsdp_workers.py`](ARPO/verl_arpo_entropy/verl/workers/fsdp_workers.py) `sync_echo` branch
- Agent tools under `verl/workers/agent/tools/` (incl. `search_tool_echo.py`)

### Today's live API (staleness baseline for S2)

Anything that still assumes the following is a stale candidate unless proven live:

- `reward_strategy` / `sign_cond_strategy` / `scorer|entropy|entropy-hybrid` as **phase reward channels**
- `non_border_loss_mask` / `select_loss_mask` / `first_select` / `<select>` schema
- DAPO / `filter_groups` / validator profiles
- Docs or analysis that document those retired surfaces as current behavior

Still **live** (not stale by default): `advantage_algorithm`, `phase_rollouts.*.strategy`, `tool_loss_mask`, prompt-5 tags, scorer **reward** via `deep_research_echo` (always on), experiment names that merely contain historical strings.

---

## Subtasks (deep briefs)

### S1 — Remove `recipe/echo` twin — COMPLETED

**Goal:** Delete the diverged twin so live ECHO is the only recipe package. Retarget every external path that still points at `recipe/echo`. Leave shared `verl/` untouched.

**Why current code looks this way:** Twin was kept in sync through earlier cleanup for drift avoidance; live later moved to `advantage_algorithm` while the twin stayed on `reward_strategy`. User now wants the twin gone entirely.

**Read first:**
- External refs: `evaluation/run_*.sh` (`TRAINING_RECIPE_PATH`), [`evaluation/xmix/run_xmix.sh`](evaluation/xmix/run_xmix.sh), [`tree_hca_eval/main.sh`](tree_hca_eval/main.sh), [`tree_hca_eval/run_math_echo.sh`](tree_hca_eval/run_math_echo.sh)
- Stale `recipe.echo` / `recipe/echo` strings under [`ECHO/training/`](ECHO/training/) (analysis docstrings, `run_monotonic_trend_sweep.sh`, `merge_search_cache_postrun.py`, `echo_ray_trainer.py` comment, `docs/logging_readme.md`)
- [`ECHO/training/scripts/old/`](ECHO/training/scripts/old/) for which launchers already have mirrors
- Do **not** edit anything under `ARPO/verl_arpo_entropy/verl/`

**Do:**
1. Copy any eval-referenced launchers that exist **only** in the twin into `ECHO/training/scripts/old/` via `cp` (at least `ECHO_2.5_7B_Reasoning_1node_v1_maxent.sh`; check all `TRAINING_RECIPE_PATH` targets). Provenance-only — need not remain runnable.
2. `git rm -r ARPO/verl_arpo_entropy/recipe/echo/` (entire tree). User-approved delete.
3. Retarget paths:
   - `TRAINING_RECIPE_PATH` → `ECHO/training/scripts/old/<same>.sh`
   - `ECHO_SYSTEM_PROMPT_YAML` still on twin → [`ECHO/training/config/echo_system_prompts.yaml`](ECHO/training/config/echo_system_prompts.yaml) (`tree_hca_eval`, `evaluation/xmix` default only — **path retarget only**, no xmix API rewrite)
4. Fix stale `recipe.echo` / `recipe/echo` module/path comments under `ECHO/training/` → `training.*` / `ECHO/training/...`.

**Do not:**
- Edit `ARPO/verl_arpo_entropy/verl/`
- Refactor live trainer / actor / core_algos beyond comment/path string fixes
- Delete other `recipe/*` packages (`dapo`, `prime`, …)
- Broad doc rewrite beyond paths broken by the delete (S2 owns staleness purge)

**Done when:**
- `test ! -e ARPO/verl_arpo_entropy/recipe/echo`
- `git diff --stat -- ARPO/verl_arpo_entropy/verl/` is empty
- `rg -n 'recipe/echo|recipe\.echo' -g '!Project_plan.md'` has no live path refs
- Retargeted eval/tree_hca/xmix paths resolve to existing `ECHO/training/` files

**Verify:**
```bash
test ! -e ARPO/verl_arpo_entropy/recipe/echo
git diff --stat -- ARPO/verl_arpo_entropy/verl/
rg -n 'recipe/echo|recipe\.echo' -g '!Project_plan.md'
```

**Depends on:** live `ECHO/training/` API is already source of truth (`advantage_algorithm`, prompt-5).

---

### S2 — Purge stale live leftovers — COMPLETED

**Goal:** Against today's live API (`advantage_algorithm` + prompt-5 + always-on scorer reward), inventory then remove dead code, unused variables/fields, obsolete config keys, and docs that still describe retired surfaces as current.

**Why current code looks this way:** Prior campaigns removed DAPO / profiles / reward_strategy from the live trainer path, but docs, analysis scripts, comments, and possibly unused branches/keys still speak the old language (`entropy-hybrid`, `non_border_loss_mask`, `first_select`, etc.).

**Read first (after S1):**
- Live hot path: [`echo_ray_trainer.py`](ECHO/training/echo_ray_trainer.py), [`echo_dp_actor.py`](ECHO/training/echo_dp_actor.py), [`echo_core_algos.py`](ECHO/training/echo_core_algos.py), [`echo_reward_manager.py`](ECHO/training/echo_reward_manager.py), [`echo_fsdp_workers.py`](ECHO/training/echo_fsdp_workers.py), [`main_echo.py`](ECHO/training/main_echo.py)
- Launch surface: [`scripts/train.sh`](ECHO/training/scripts/train.sh), [`config/echo_trainer.yaml`](ECHO/training/config/echo_trainer.yaml), [`training_config/*.yaml`](ECHO/training/training_config/)
- Likely stale surfaces: [`docs/logging_readme.md`](ECHO/training/docs/logging_readme.md), [`docs/Notes.md`](ECHO/training/docs/Notes.md), [`analysis/generate_echo3binst_report.py`](ECHO/training/analysis/generate_echo3binst_report.py), other `analysis/*` that parse legacy keys
- Seed search patterns: `reward_strategy`, `sign_cond_strategy`, `entropy-hybrid`, `non_border`, `select_loss`, `first_select`, `filter_groups`, `maxentropy`, `phase_strategy`, `mask_first_select`, `validator_profile`

**Do:**
1. **Inventory** under `ECHO/training/` (plus any non-xmix analysis/docs that claim to describe current ECHO training): list each stale item with file, symbol/key, and why it is dead vs today's call graph. Append the inventory to this plan’s Progress log (or a short subsection under this brief) before deleting.
2. **Remove** confirmed-dead items that affect the live path or present a false picture of current behavior:
   - Dead Python branches / unused helpers / unused config fields in live trainer/actor/algos/reward/workers
   - Obsolete launch keys still accepted or documented as live
   - Docs / analysis that document retired reward-strategy / select-era / DAPO surfaces as current — rewrite to today's API or delete the obsolete section/file if entirely wrong
3. Keep a short “kept on purpose” note in the Progress log for anything that looks stale but is intentional (e.g. historical experiment-name strings, provenance-only `scripts/old/`).

**Do not:**
- Start before S1 is completed (twin gone; path retargets done).
- Edit `ARPO/verl_arpo_entropy/verl/` in this subtask (shared runtime stays; escalate to a new user-named subtask if rollout/scorer itself has dead code).
- Mass-delete [`scripts/old/`](ECHO/training/scripts/old/) or ECHO root tool-sandbox junk (`*.png` / `*.db`) unless the inventory explicitly flags them and the user already approved that class of delete in S2 scope — default = leave archives; fix only if they are referenced as current.
- Broad algorithmic refactors, AEPO redesign, or “while we’re here” cleanups outside confirmed staleness.
- Touch `evaluation/xmix/` beyond whatever S1 already path-retargeted.

**Done when:**
- Inventory of removed + kept-on-purpose items is in the Progress log.
- Live hot path and launch/docs no longer present retired APIs as current (spot-check seed search patterns above under `ECHO/training/` excluding `scripts/old/` and Progress-log historical notes).
- `train.sh` + at least one `training_config/*.yaml` still pass launch-key validation mentally / via dry key check.
- `git diff --stat -- ARPO/verl_arpo_entropy/verl/` still empty.

**Verify:**
```bash
rg -n 'reward_strategy|sign_cond_strategy|entropy-hybrid|non_border_loss_mask|select_loss_mask|first_select|filter_groups|phase_strategy|mask_first_select|validator_profile' \
  ECHO/training --glob '!scripts/old/**' --glob '!**/Project_plan.md'
git diff --stat -- ARPO/verl_arpo_entropy/verl/
```

**Depends on:** S1 completed.

---

### S3 — Prune prompts + identify dead shared verl — COMPLETED

**Goal:** Make the sole live Echo prompt `system_prompt_1` (today’s prompt-5 text). Drop inactive `<select>`-era prompts 1–4. Retarget every live train/eval default that still says `5`. Separately, inventory dead shared ECHO leftovers under `ARPO/verl_arpo_entropy/verl/` (list in Progress log; **no verl edits**).

**Why current code looks this way:** S2 kept prompts 1–4 on purpose as inactive candidates; live API is only prompt-5. Canonical index should be `1`. Shared `verl/` was frozen in S1/S2; S3 only identifies candidates for a later delete subtask.

**Read first:**
- [`ECHO/training/config/echo_system_prompts.yaml`](ECHO/training/config/echo_system_prompts.yaml) — keep `system_prompt_5` body; delete keys `_1`…`_4`; rename key `_5` → `_1`.
- Defaults: [`echo_trainer.yaml`](ECHO/training/config/echo_trainer.yaml), [`train.sh`](ECHO/training/scripts/train.sh) (`ACTIVE_SYSTEM_PROMPT:-5`), all [`training_config/*.yaml`](ECHO/training/training_config/) (`active_system_prompt: 5`).
- Eval/tree_hca: `evaluation/run_*.sh` / related HF jobs with `ECHO_ACTIVE_SYSTEM_PROMPT="5"` and “must include system_prompt_5” comments; [`scripts/sft_refactor/constants.py`](scripts/sft_refactor/constants.py) (`ACTIVE_SYSTEM_PROMPT = "system_prompt_5"`); docs [`Notes.md`](ECHO/training/docs/Notes.md).
- Already-`1` loaders ([`tree_hca_eval/main.sh`](tree_hca_eval/main.sh), [`run_math_echo.sh`](tree_hca_eval/run_math_echo.sh), [`evaluation/xmix/run_xmix.sh`](evaluation/xmix/run_xmix.sh)): after rename they load the live schema — leave defaults at `1`; only fix comments if they still imply select-era N.
- Shared verl inventory seeds: [`vllm_rollout_echo.py`](ARPO/verl_arpo_entropy/verl/workers/rollout/vllm_rollout/vllm_rollout_echo.py), [`deep_research_echo.py`](ARPO/verl_arpo_entropy/verl/utils/reward_score/deep_research_echo.py), [`search_tool_echo.py`](ARPO/verl_arpo_entropy/verl/workers/agent/tools/search_tool_echo.py) (live YAMLs use `BingSearchTool` only; `OpenAISearchTool` appears in `scripts/old/`), `fsdp_workers.py` `sync_echo` branch, callers of those symbols.

**Do:**
1. Rewrite `echo_system_prompts.yaml` to a single `system_prompt_1` = current `system_prompt_5` text; header comment says N=1 is the only live schema.
2. Flip live defaults `5` → `1`: `echo_trainer.yaml`, `train.sh`, all live `training_config/*.yaml`, eval launchers that hardcode `ECHO_ACTIVE_SYSTEM_PROMPT=5`, `scripts/sft_refactor/constants.py`, live docs/comments that say prompt-5 as current.
3. Update this plan’s locked “Schema = …” line if still needed so it matches `system_prompt_1` (same tags).
4. **Inventory** shared verl: for each candidate, note file/symbol, why dead vs live call graph, keep-vs-delete recommendation. Append table to Progress log. **Do not** `git rm` or edit under `ARPO/verl_arpo_entropy/verl/`.

**Do not:**
- Edit `ARPO/verl_arpo_entropy/verl/` (inventory only).
- Mass-edit `ECHO/training/scripts/old/` (provenance; may still mention prompt 5 / `search_tool_echo`).
- Rewrite `evaluation/xmix/` beyond leaving `ECHO_ACTIVE_SYSTEM_PROMPT` at 1.
- Invent new prompt text; only prune + rename.

**Done when:**
- YAML has only `system_prompt_1` with former prompt-5 body (`rg system_prompt_[2-5]` clean under live config).
- Live train/eval defaults resolve to `1` (`train.sh` default, training_config, eval `ECHO_ACTIVE_SYSTEM_PROMPT`).
- Progress log has shared-verl dead-code inventory table.
- `git diff --stat -- ARPO/verl_arpo_entropy/verl/` empty.

**Verify:**
```bash
rg -n 'system_prompt_[2-5]|active_system_prompt:\s*5|ACTIVE_SYSTEM_PROMPT:-5|ECHO_ACTIVE_SYSTEM_PROMPT=\"5\"' \
  ECHO/training evaluation tree_hca_eval scripts/sft_refactor \
  --glob '!ECHO/training/scripts/old/**' --glob '!**/projectplan_OLD.md'
python -c "import yaml; p=yaml.safe_load(open('ECHO/training/config/echo_system_prompts.yaml')); assert set(k for k in p if k.startswith('system_prompt_'))=={'system_prompt_1'}"
git diff --stat -- ARPO/verl_arpo_entropy/verl/
```

**Depends on:** S1 + S2 completed.

---

## Locked decisions (do not reopen unless user says so)

**Already shipped (still in force):**
- HL vs LL differ **only** by phase loss mask; scoring/format shared.
- Schema = system_prompt_1 (content of former system_prompt_5: think / tool / search|python / result / final tool / answer+boxed). S3 flips the key index; body unchanged.
- Format gate = ARPO-style −1 early exits then F1 (+ multi-tool bonus).
- No DAPO/filter_groups; no validator profiles.
- Per phase: `advantage_algorithm` (`grpo|entropy|aepo`) + `phase_rollouts.*.strategy` (`default|aepo`). Sign-cond clip uses assigned advantage sign.
- Contiguous HL/LL blocks in launch YAMLs; `active_system_prompt` via train.sh (target **1** after S3).

**This campaign:**
- Canonical recipe = [`ECHO/training/`](ECHO/training/) only.
- Completely remove [`ARPO/verl_arpo_entropy/recipe/echo/`](ARPO/verl_arpo_entropy/recipe/echo/) (S1).
- Shared runtime under `ARPO/verl_arpo_entropy/verl/` stays; **S1–S3 must not edit it**. S3 may **inventory** dead shared ECHO leftovers only; deletes require a later user-named subtask.
- `evaluation/xmix/` remains out of scope for API/behavior cleanup; **path-only** retarget of deleted twin refs is allowed in S1; S3 leaves xmix/tree_hca defaults at `1` (correct after rename).
- S2 may delete confirmed-stale live code/docs; `scripts/old/` and ECHO sandbox junk stay unless inventory + user scope say otherwise.
- S3: prune inactive prompts 1–4; rename prompt-5 → prompt-1; retarget live `5` → `1`; inventory-only for shared verl.
- Further subtasks only when the user adds them to this plan.

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

### 2026-07-14 — orientation — plan
- Mapped live vs twin vs shared runtime; twin API (`reward_strategy`) ≠ live (`advantage_algorithm`).
- Created root [`Project_plan.md`](Project_plan.md).
- Next: await cleanup directive

### 2026-07-14 — plan-amendment — S1 remove twin
- User: completely remove diverged twin; **do not touch shared runtime**.
- Restructured plan to full schema (goal, numbered S-subtasks, session rules, architecture map, locked decisions, Progress log).
- S1 deep brief written.
- Survey notes: most eval `TRAINING_RECIPE_PATH` shells already under `ECHO/training/scripts/old/`; `*_maxent.sh` missing — copy before `git rm`. Twin vs live `echo_system_prompts.yaml` differ — retarget to live only, no merge this subtask.
- Next: `S1-remove-recipe-echo-twin`

### 2026-07-14 — plan-amendment — schema + rule
- Removed cross-references to archived prior-plan filenames from this file.
- Enhanced Plan session protocol User Rule text (paste-over existing; no new `.mdc` file) so fresh Goals always use full plan schema.
- Next: `S1-remove-recipe-echo-twin`

### 2026-07-14 — plan-amendment — S2 stale purge
- User: add subtask to identify/remove stale code, variables, and anything irrelevant to today's codebase.
- Added `S2-purge-stale-live` with deep brief (inventory then remove; baseline = today's `advantage_algorithm` / prompt-5 API; default leave `scripts/old/` + shared `verl/`).
- Seed stale hits already visible: `docs/logging_readme.md`, `docs/Notes.md`, `analysis/generate_echo3binst_report.py` still describe reward_strategy / entropy-hybrid / non_border / first_select.
- Next: execute `S1-remove-recipe-echo-twin` when user says so; then S2.

### 2026-07-14 — S1-remove-recipe-echo-twin — completed
- Changes:
  - Copied twin-only maxent launchers into `ECHO/training/scripts/old/` (`ECHO_2.5_7B_…_maxent.sh`, `ECHO_2.5_3B_…_maxent.sh`, `ECHO_3.1_8B_…_maxent.sh`).
  - `git rm -r ARPO/verl_arpo_entropy/recipe/echo/` + wiped untracked leftover debris (plots/reports/pycache/untracked py). Twin path gone.
  - Retargeted `TRAINING_RECIPE_PATH` → `ECHO/training/scripts/old/<same>.sh` and twin `ECHO_SYSTEM_PROMPT_YAML` → `ECHO/training/config/echo_system_prompts.yaml` in eval `run_*.sh`, `tree_hca_eval/{main,run_math_echo}.sh`, `evaluation/xmix/run_xmix.sh`, `usage_notes.md`.
  - Fixed live path/module comments under `ECHO/training/` (trainer comment, merge_search_cache docstring, analysis plot usage lines, `run_monotonic_trend_sweep.sh` invocation → file path, `docs/logging_readme.md`).
  - `commands.txt` → live `ECHO/training/` merge/snapshot paths; dropped twin entries from `.gitignore` / `.cursorignore`.
  - Shared `ARPO/verl_arpo_entropy/verl/` untouched (`git diff --stat` empty).
- Kept on purpose (not live refs): `recipe.echo.main_echo` inside provenance `scripts/old/*maxent.sh`; frozen historical strings under `evaluation/outputs/**/run_config.yaml`.
- Follow-ups for later subtasks: S2 inventory/purge of stale live leftovers (`reward_strategy`, `entropy-hybrid`, `non_border`, `first_select`, etc.) under `ECHO/training/` excluding `scripts/old/`.
- Next: `S2-purge-stale-live`

### 2026-07-14 — S2-purge-stale-live — completed

**Inventory (then removed / rewritten):**

| Item | Why dead vs today's call graph |
|------|--------------------------------|
| `docs/Notes.md` (entire prior writeup) | Documented `low_level_reward_strategy` / `entropy-hybrid` / `ll_no_tool_penalty` as current. Live path has always-on scorer reward + `advantage_algorithm` only. |
| `docs/logging_readme.md` rows/sections for `entropy_scalar_*`, `entropy_reduced_mean`, `non_border_loss_mask`, `high_level_valid`/`low_level_valid`, `<select>` structure, "valid across scorer/entropy/entropy-hybrid" | Trainer `_build_scorer_metrics` logs score/f1/format_pass/format_valid/no_tool only; scorer is `deep_research_echo` (prompt-5 + `format_valid`). |
| `analysis/plot_monotonic_trend.py` + `scripts/run_monotonic_trend_sweep.sh` defaults | Defaulted y/x to `entropy_scalar_mean` / `entropy_loss` (retired reward channel / ARPO-named key). Live: `effective_reward_mean` / `entropy_old_policy`. |
| `analysis/plot_training_log.py` reward comments | Still described LL reward as "pre-gate entropy". |
| `analysis/generate_echo3binst_report.py` `_config_signature` | Read retired `strategy` / `algorithm` / `no_tool_penalty` / `mask_categories.first_select`. Live: `advantage_algorithm` + prompt-5 mask keys. |
| `config/echo_trainer.yaml` `gen_batch_size` comment | Referenced DAPO dynamic sampling (locked out). |
| `echo_ray_trainer.py` dump-skip comment | Said "scorer phase" for missing entropy_reg. |

**Live hot path checked, no dead branches found:** `echo_ray_trainer.py`, `echo_dp_actor.py`, `echo_core_algos.py`, `echo_reward_manager.py`, `echo_fsdp_workers.py`, `main_echo.py`, `scripts/train.sh`, `training_config/*.yaml` already on `advantage_algorithm` / prompt-5 surface.

**Kept on purpose:**
- `ECHO/training/scripts/old/**` — provenance archives (still use reward_strategy / first_select).
- `echo_system_prompts.yaml` system_prompt_1–4 — inactive `<select>`-era candidates; default remains prompt 5.
- Experiment-name strings in launch YAMLs that only historically mention hybrid/scorer.
- `generate_echo3binst_report.py`: optional legacy `entropy_scalar_*` metric fallbacks + run-name group buckets for old checkpoint folders; signature now prefers live keys.
- `echo_core_algos.py` unused advantage estimators (RLOO/OPO/…) — shared PPO surface, not retired ECHO reward API; out of "confirmed staleness" scope.

- Changes: Rewrote Notes + logging cheatsheet; retargeted analysis defaults to live metrics; fixed report config signature; cleared DAPO/scorer-phase comments. Shared `verl/` untouched. Launch-key check on `echo_3B_ll_hl.yaml` OK. Seed rg clean under `ECHO/training` excluding `scripts/old/`.
- Follow-ups for later subtasks: (none pending unless user names S3 — e.g. shared-verl dead code, or prune inactive system_prompt_1–4).
- Next: _(none)_

### 2026-07-14 — plan-amendment — S3 prune prompts + identify verl
- User: add S3 to prune echo prompts 1–4, rename system_prompt_5 → system_prompt_1, update training scripts accordingly, and identify dead shared verl code.
- Added `S3-prune-prompts-identify-verl` (pending) with deep brief: single live `system_prompt_1` (= today’s prompt-5 body); flip live defaults `5`→`1`; tree_hca/xmix stay at `1`; shared `verl/` inventory only (no edits).
- Locked decisions updated: schema key target = `system_prompt_1`; S1–S3 must not edit `verl/`.
- Next: `S3-prune-prompts-identify-verl`

### 2026-07-14 — S3-prune-prompts-identify-verl — completed

**Prompt prune / retarget:**
- Rewrote [`echo_system_prompts.yaml`](ECHO/training/config/echo_system_prompts.yaml) to sole `system_prompt_1` (= former prompt-5 body); dropped inactive `<select>`-era `_1`…`_4`.
- Flipped defaults `5`→`1`: `echo_trainer.yaml`, `train.sh` (`ACTIVE_SYSTEM_PROMPT:-1`), all 10 `training_config/*.yaml`, 15 eval HF launchers (`ECHO_ACTIVE_SYSTEM_PROMPT="1"` + comments), `scripts/sft_refactor/constants.py` + README, `docs/Notes.md`, `docs/logging_readme.md` wording, `tree_hca_eval/run_math_echo.sh` comment.
- Left xmix/tree_hca defaults already at `1`. Did not touch `scripts/old/` or `ARPO/.../verl/`.
- Verify: YAML assert unique `system_prompt_1`; seed rg clean; `git diff --stat -- ARPO/.../verl/` empty.

**Shared verl dead-code inventory (no edits this subtask):**

| File / symbol | Why dead vs live call graph | Rec |
|---------------|----------------------------|-----|
| [`search_tool_echo.py`](ARPO/verl_arpo_entropy/verl/workers/agent/tools/search_tool_echo.py) (entire module: `BingSearchTool` + `OpenAISearchTool`) | Live YAMLs load `verl.workers.agent.tools.search_tool.BingSearchTool` / `BingSearchToolRAG` only. Zero live importers of `search_tool_echo`; only provenance hit is `ECHO/training/scripts/old/ECHO_2.5_7B_Reasoning_1node.sh` → `OpenAISearchTool`. | **delete** (later subtask) |
| [`vllm_rollout_echo.py`](ARPO/verl_arpo_entropy/verl/workers/rollout/vllm_rollout/vllm_rollout_echo.py) | Live via `rollout.mode=sync_echo`; prompt-1 tags / `tool_loss_mask` / `mask_categories`. No retired `first_select`/`non_border` keys left. | **keep** |
| [`deep_research_echo.py`](ARPO/verl_arpo_entropy/verl/utils/reward_score/deep_research_echo.py) | Live train reward + eval `validate_format` / xmix `get_ordered_blocks`. `__main__` legacy-`<select>` rejection smoke = intentional. | **keep** |
| `fsdp_workers.py` `sync_echo` → `vLLMRolloutECHO` | Live trainer entry for ECHO rollout. | **keep** |
| `mask_categories_for_profile` / `resolve_validator_profile` | Already removed from `deep_research_echo` in prior campaigns. [`evaluation/xmix/score_with_compute_score.py`](evaluation/xmix/score_with_compute_score.py) still imports them → **stale caller** (xmix out of S3 API scope). | symbols gone; fix/drop xmix import later if user names it |

- Changes: Prompt prune + live `5`→`1` retargets; verl inventory table only.
- Follow-ups for later subtasks (user-named): delete `search_tool_echo.py`; optionally repair xmix import of removed profile helpers.
- Next: _(none)_
