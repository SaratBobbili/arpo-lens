---
name: ECHO single-tree cleanup
overview: Multi-session ECHO cleanup. Goal — one live training tree (ECHO/training/) + shared ARPO/verl runtime; delete the diverged recipe/echo twin; then purge stale live leftovers irrelevant to today's API. One subtask per chat; agent writes progress back into this plan before ending.
todos:
  - id: S1-remove-recipe-echo-twin
    content: "S1: Delete ARPO/.../recipe/echo/ entirely; retarget eval/tree_hca/xmix + ECHO/training path refs to ECHO/training/; do not touch ARPO/.../verl/"
    status: completed
  - id: S2-purge-stale-live
    content: "S2: Inventory and remove stale code/vars/docs/config keys in live ECHO/training (and related analysis) that belong to pre-advantage_algorithm / pre-prompt-5 APIs; leave shared verl untouched unless a later subtask says so"
    status: pending
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

**Active subtask right now:** `S2-purge-stale-live`

**Later subtasks:** Further S3+ only when the user names them.

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

### S2 — Purge stale live leftovers — PENDING

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

## Locked decisions (do not reopen unless user says so)

**Already shipped (still in force):**
- HL vs LL differ **only** by phase loss mask; scoring/format shared.
- Schema = system_prompt_5 (think / tool / search|python / result / final tool / answer+boxed).
- Format gate = ARPO-style −1 early exits then F1 (+ multi-tool bonus).
- No DAPO/filter_groups; no validator profiles.
- Per phase: `advantage_algorithm` (`grpo|entropy|aepo`) + `phase_rollouts.*.strategy` (`default|aepo`). Sign-cond clip uses assigned advantage sign.
- Contiguous HL/LL blocks in launch YAMLs; `active_system_prompt` via train.sh (target 5).

**This campaign:**
- Canonical recipe = [`ECHO/training/`](ECHO/training/) only.
- Completely remove [`ARPO/verl_arpo_entropy/recipe/echo/`](ARPO/verl_arpo_entropy/recipe/echo/) (S1).
- Shared runtime under `ARPO/verl_arpo_entropy/verl/` stays; **S1 and S2 must not edit it** (unless a later user-named subtask says so).
- `evaluation/xmix/` remains out of scope for API/behavior cleanup; **path-only** retarget of deleted twin refs is allowed in S1.
- S2 may delete confirmed-stale live code/docs; `scripts/old/` and ECHO sandbox junk stay unless inventory + user scope say otherwise.
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
