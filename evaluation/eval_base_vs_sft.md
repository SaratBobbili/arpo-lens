# Base vs Tool-SFT: Qwen2.5-7B-Instruct math evaluation

Tracking doc for the base-vs-SFT comparison on the math benchmark suite
(`DATASET_GROUP=math_all`, judge = Qwen2.5-72B-Instruct, `MAX_TOKENS=4096`).
SFT checkpoint = LLaMA-Factory tool-star SFT of Qwen2.5-7B-Instruct
(`ECHO/sft/checkpoints/Qwen2.5-7B-Instruct`, checkpoint-10185).

## Runs

| id | model | prompt | temp | run folder (`outputs/hf_math_4qa/`) |
|---|---|---|---|---|
| R1 | base (HF hub) | base | 0.6 | `base/Qwen__Qwen2.5-7B-Instruct/20260721_165540_3339647` |
| R2 | base (HF hub) | base | 0.0 | `base/Qwen__Qwen2.5-7B-Instruct/20260721_172545_57738` |
| R3 | SFT ckpt | echo (tools) | 0.6 | `sft/checkpoints/20260721_155449_1033708` |
| R4 | SFT ckpt | base | 0.6 | pending |
| R5 | SFT ckpt | base | 0.0 | pending |

## Results — llm_equal (math_equal in parens), %

| Dataset | R1 base T=0.6 | R2 base T=0 | R3 sft echo T=0.6 | R4 sft base T=0.6 | R5 sft base T=0 | Qwen2.5 report (greedy) |
|---|---|---|---|---|---|---|
| GSM8K | 90.1 (89.2) | 89.7 (89.4) | 87.4 (87.2) | | | 91.6 |
| MATH (5000) | 72.6 (69.8) | 73.4 (70.7) | 65.2 (62.9) | | | 75.5 |
| MATH500 | 71.0 (67.8) | 73.2 (71.2) | 64.4 (62.4) | | | — |
| AIME24 | 10.0 | 10.0 | 16.7 | | | — |
| AIME25 | 10.0 | 3.3 | 6.7 | | | — |

AIME rows are n=30, single turn: noise, do not interpret.

## MATH category breakdown — llm_equal, %

| Category (n) | R1 base T=0.6 | R2 base T=0 | R3 sft echo T=0.6 | Δ (R3 − R2) |
|---|---|---|---|---|
| arithmetic (2846) | 77.8 | 78.4 | 68.5 | −9.9 |
| probability_counting (268) | 73.1 | 75.7 | 71.6 | −4.1 |
| geometry (792) | 71.1 | 70.6 | 59.1 | −11.5 |
| diagram (443) | 43.1 | 47.0 | 34.8 | −12.2 |
| number_theory (587) | 70.0 | 70.9 | 75.0 | +4.1 |
| rate_word (64) | 84.4 | 82.8 | 85.9 | +3.1 |

Tool usage (R3 only): 1.16 python calls/sample, 5.5% python exception rate.
R1/R2 make zero tool calls (base prompt advertises none).

## Analysis

### 1. The harness is sound; its absolute offset is ~2 points

Base at greedy (R2) lands at 89.7 GSM8K / 73.4 MATH vs the reported
91.6 / 75.5. Temperature explains almost nothing (R1 vs R2 differ by
<1 point on the large sets), so the residual ~2 points is the fixed
harness offset: our `<think>/<answer>` tag prompt instead of Qwen's plain
"reason step by step, put answer in \boxed{}" prompt, plus extraction
differences. This offset is stable across runs, so within-harness
comparisons are trustworthy even though absolute numbers read ~2 low.

### 2. The SFT+echo pathway costs ~8 points on MATH — real, not eval noise

R3 vs R2: −2.3 GSM8K, −8.2 MATH, −8.8 MATH500 (llm_equal). Four times the
harness offset, same judge, same datasets. Something in the SFT+tool
pathway is genuinely losing accuracy.

### 3. Category pattern points at the tool loop, not lost math knowledge

The drop is not uniform. SFT+echo *gains* on number_theory (+4.1) and
rate_word (+3.1) — categories where offloading discrete computation to
python is a clean win. It collapses on geometry (−11.5), diagram (−12.2),
and arithmetic (−9.9) — categories needing long uninterrupted symbolic
derivations, where the model instead emits a python call mid-reasoning
(1.16 calls/sample on average), and 5.5% of those calls throw exceptions
that inject error text into the context. If the SFT had destroyed math
ability wholesale, number_theory would not have improved.

Working hypothesis: the tool-star SFT taught a "reach for python early"
reflex. Where python is the right tool the model got better; where the
problem needs sustained latent reasoning (geometry/diagram) the tool call
truncates or derails the chain of thought, and failed executions compound
it.

### 4. What R4/R5 will decide

R4/R5 run the *same SFT weights* through the plain base prompt (no tools).

- If R4/R5 recover to ~72-73 MATH (base level): the weights retain CoT
  math ability, and the ~8-point R3 gap is entirely the echo tool
  pathway (format pressure + execution failures). Fix directions: tool
  reliability, when-to-call policy, tool budget.
- If R4/R5 stay near 65 MATH: the SFT itself eroded pure CoT ability
  (catastrophic-forgetting flavor, since the tool-star data distribution
  is far from MATH-style long derivations). Fix directions: mix general
  math CoT data into SFT, fewer epochs, lower LR.
- Watch `average_python_calls` and `num_valid_answer` in R4: if the SFT
  model emits `<python>` tags despite the base prompt (format drift),
  R4 underestimates its true CoT ability and that itself is evidence the
  SFT over-committed to the tool format.

## Progress log

- 2026-07-21: R1, R2, R3 complete and tabulated. R4, R5 launched by user; update tables when metrics land.
