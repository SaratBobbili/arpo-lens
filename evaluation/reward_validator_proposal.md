# Tool-selection reward / validator proposal (NOT implemented)

Proposal only. No training reward code is changed in this pass. Validate every
item below against the corrected diagnostics (`diagnose.py`) before touching
`ARPO/verl_arpo_entropy/verl/utils/reward_score/deep_research_echo.py`, and gate
each change behind config so training behavior only changes when explicitly
enabled.

## Evidence from the corrected eval (math500, echo 3B)

- `diagram` is the worst category: `math_equal = 0.356` vs `arithmetic = 0.720`.
- Every initial `<select>` plan is effectively `("no-tool", "python")` and python
  is called on ~99.7% of examples; python execution errors on ~7.4% of math500.
- Python confidently returns a wrong number when the diagram/geometry setup was
  mis-translated, so over-tooling actively hurts the visual-reasoning buckets.

## Problems in the current reward/validator

1. `_check_first_select` accepts any non-empty subset of `{search, python, no-tool}`
   with no shaping, so `("no-tool", "python")` is never discouraged.
2. `compute_score` gives a `+0.1` bonus only for using search+python together and
   otherwise has no tool-choice term, so there is no signal against reflexive
   python use.
3. Nothing penalizes python on diagram/geometry items where the model has not
   first translated the figure into equations.

## Proposed changes (config-gated)

1. Plan-quality penalty: when the initial planning `<select>` lists both `no-tool`
   and `python` (a non-committal plan), apply a small negative shaping term.
   Config: `reward.plan_redundant_tool_penalty` (default 0.0 = off).
2. Python-usefulness gating: only reward python when its `<result>` is consumed by
   the final answer (numeric/expression overlap between result and boxed answer);
   penalize python calls that error or whose result is unused.
   Config: `reward.python_unused_penalty`, `reward.python_error_penalty`.
3. Diagram discouragement: for `[asy]`/geometry questions (reuse
   `diagnostics.classify_question`), require an equation-translation `<think>`
   before the first `<python>`; otherwise apply a penalty.
   Config: `reward.diagram_requires_translation` (default off).

## Validation protocol before enabling any of the above

- Re-run `diagnose.py` on a held-out set and confirm the reward change would move
  category-level `math_equal` (especially `diagram`/`geometry`) without collapsing
  `echo_format_pass_rate` or the overall tool-usage distribution.
- Keep all penalties additive and small; roll out one flag at a time.

## Dead code to retire once eval no longer needs it

`VALIDATOR_PROFILE_SIGNATURES`, `resolve_validator_profile`,
`mask_categories_for_profile` and the "c1..c5" comments in
`deep_research_echo.py` only enumerate high/low-only profiles and cannot express
the `both`/`none` phase overlap the rollout now supports. The evaluator has been
migrated to pass the real `mask_categories` dict directly; these helpers can be
removed after the trainer's `resolve_validator_profile` call site is likewise
migrated.
