# SFT Trajectory Refactor

Transforms [`dongguanting/ARPO-SFT-54K`](https://huggingface.co/datasets/dongguanting/ARPO-SFT-54K)
into the ECHO v3 schema where general reasoning (`<think>`) and per-action tool-selection
rationale (`<tool>`) are cleanly separated, and every action is justified:

```
<think> reasoning </think>
<tool> why this call </tool>
<search|python> ... </search|python>
<result> ... </result>
...
<tool> why no more tools </tool>
<answer> ... \boxed{...} </answer>
```

## Files

| File | Role |
|------|------|
| `constants.py`  | `NEW_SYSTEM_PROMPT`, `ECHO_SYSTEM_PROMPT`, `GPT_SYSTEM`, `GPT_REVIEW_SYSTEM`, tag lists/aliases |
| `trajectory.py` | Segment parsing, think-merging, action extraction, reassembly, `verify_text` |
| `refactor.py`   | CLI: `inspect` / `split` / `verify` / `prune` / `export` |

## Pipeline

```
inspect  →  split  →  verify  →  prune  →  export
```

1. **inspect** — full-dataset pattern scan → `pattern_taxonomy.json`, `conversion_rules.json`,
   `inspect_index.jsonl`, `samples/`. Classifies each row as `drop` (malformed) or `gpt_split`.
2. **split** — two GPT passes per non-drop row: a breakdown pass (`GPT_SYSTEM`) that enriches
   thinks + writes a rationale per action, then a coherence pass (`GPT_REVIEW_SYSTEM`) that audits
   the assembled trajectory. Answers / queries / code / results are copied verbatim.
3. **verify** — checks the structural invariants; prints OK/bad counts + reason histogram.
4. **prune** — drops rows failing `verify`, writes the clean JSONL/parquet.
5. **export** — swaps the GPT-transform `system` prompt for `ECHO_SYSTEM_PROMPT` (ECHO's
   `system_prompt_1` reworded for the new `<think>/<tool>/<search>/<python>/<result>/<answer>`
   structure; mirrored as `system_prompt_4` in `echo_system_prompts.yaml`). `conversations` are
   left untouched, giving a SFT-ready ShareGPT JSONL/parquet.

## Usage

```bash
conda activate arpo
```

### Pilot (heuristic, no API)

```bash
python scripts/sft_refactor/refactor.py inspect \
  --output-dir scripts/sft_refactor/output/inspect --limit 200

python scripts/sft_refactor/refactor.py split \
  --inspect-dir scripts/sft_refactor/output/inspect \
  --output scripts/sft_refactor/output/echo_sft_v3_pilot.parquet \
  --jsonl scripts/sft_refactor/output/echo_sft_v3_pilot.jsonl \
  --limit 20 --heuristic-only --fresh

python scripts/sft_refactor/refactor.py verify \
  scripts/sft_refactor/output/echo_sft_v3_pilot.jsonl
```

`--heuristic-only` skips the API (uses a last-sentence fallback rationale, no coherence pass) —
for testing the plumbing only.

### Full run (GPT)

```bash
export OPENAI_API_KEY="sk-..."

python scripts/sft_refactor/refactor.py inspect \
  --output-dir scripts/sft_refactor/output/inspect

python scripts/sft_refactor/refactor.py split \
  --inspect-dir scripts/sft_refactor/output/inspect \
  --output scripts/sft_refactor/output/echo_sft_v3.parquet \
  --jsonl scripts/sft_refactor/output/echo_sft_v3.jsonl \
  --concurrency 50 --fresh

python scripts/sft_refactor/refactor.py verify \
  scripts/sft_refactor/output/echo_sft_v3.jsonl

python scripts/sft_refactor/refactor.py prune \
  scripts/sft_refactor/output/echo_sft_v3.jsonl \
  -o scripts/sft_refactor/output/echo_sft_v3_clean.jsonl

python scripts/sft_refactor/refactor.py export \
  scripts/sft_refactor/output/echo_sft_v3_clean.jsonl \
  -o scripts/sft_refactor/output/echo_sft_v3_final.jsonl
```

`echo_sft_v3_final.jsonl` is SFT-ready (same ShareGPT schema as `echo_sft_thinkfirst.jsonl`);
point `LLaMA-Factory/arpo_train_sft/dataset_info/dataset_info.json` at it to train.

## Key flags (`split`)

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | `gpt-4o-mini` | OpenAI model |
| `--concurrency` | `50` | max in-flight requests |
| `--limit N` | all | process first N index rows |
| `--fresh` | off | ignore/remove existing checkpoint |
| `--keep-ckpt` | off | keep `<output>.ckpt.jsonl` after finishing |
| `--heuristic-only` | off | no API; disables coherence pass |
| `--no-coherence-review` | (on) | skip the second GPT audit pass |

Runs are checkpointed to `<output>.ckpt.jsonl`; re-running without `--fresh` resumes.
