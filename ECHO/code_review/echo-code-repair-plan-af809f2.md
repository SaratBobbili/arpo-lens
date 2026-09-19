# ECHO response-gradient repair plan

Baseline: `ECHO-Code`, branch `echo`, commit
`af809f271fd9f4236adde0893234e4a7cdb01443`.

Target: implement the practical sampled response algorithm in the current local
v10 paper, with an explicit distinction between the GRPO implementation and the
RLOO analytical proxy. This is a plan; no training source changes have been made.

The detailed defects are recorded in
[the code/pseudocode comparison](echo-code-vs-pseudocode-af809f2.md).
The specification is [Algorithm 1](/Users/sshakkot/L-WORK/L-Research/2026-ECHO-Project/2026-ECHO/2026-ICLR-ECHO-v10/sections/algorithm.tex:28)
and its [implementation guide](/Users/sshakkot/L-WORK/L-Research/2026-ECHO-Project/2026-ECHO/2026-ICLR-ECHO-v10/implementation-guidance.md:204).

## Notation and interfaces

This plan uses Python-style pseudocode and code identifiers. The snippets specify
the intended calculation; they are not drop-in implementations. Proposed helpers
and fields below do not yet exist unless explicitly identified as current code.

| Plan name | Meaning and connection to current code |
| --- | --- |
| `task_advantages` | Proposed unmasked scalar task advantages, one per trajectory. The current leader `advantages` tensor has already been multiplied by the high-level mask. |
| `high_level_loss_mask`, `low_level_loss_mask` | Existing masks used separately for the reasoning and tool losses. |
| `policy_token_mask` | Proposed mask covering all stochastic policy-generated tokens for the complete group score. |
| `self._follower_records` | Existing list, to be expanded with complete query groups and their generating states. |
| `record.step_size`, `record.query_groups` | Proposed fields holding the inner learning rate and complete statistical groups. |
| `g_fol` | Terminal tool task direction computed by `_leader_follower_direction`; that method currently leaves its negative in `p.grad`. |
| `g_dir`, `g_resp` | Direct and response ascent directions; `_response_gradient` is the current response integration point. |
| `incoming_adjoint` | Sensitivity propagated backward from the terminal tool direction through the remaining inner updates. |
| `coordinate="leader"` or `coordinate="follower"` | Logical leader or follower derivative. The current shared-weight construction permits both to act on the same physical parameters. |

All pseudocode directions use the ascent convention. PyTorch optimizers normally
minimize a loss, so the actor must write the negative total ascent direction into
`p.grad`. The reference snippets use tensors; the production backend must implement
the same operations over parameter-aligned buffers and distributed reductions.

## Recommended design decisions

1. Preserve the new snapshot/reset/restore structure and the permitted logical
   representation `working_weights = leader_base_weights + follower_delta`, where
   the leader base includes the fixed core and `follower_delta` starts at zero
   each round. These are descriptive names for the existing snapshot/restore
   behavior, not new parameter partitions. Separate adapters are
   optional scaling work, not a prerequisite for fixing correctness. Because
   routing overlaps, implement the full mixed derivative and complete trajectory
   score; do not apply a role-isolation shortcut.
2. Make the first reference configuration use stateless gradient ascent for the
   inner update, exactly as displayed in the paper. Use the displayed outer
   gradient/projection rule in that configuration too. Disable implicit optimizer
   weight decay and gradient clipping in the reference; regularization belongs in
   the specified objective. Any later clipping, momentum, or Adam extension must
   differentiate the actual transition, including its state and relevant branches.
3. Implement the complete response calculation first on a small unsharded model
   with ordinary differentiable PyTorch operations. Validate K=1 and K=2 before
   K=8. A nonzero response norm is a diagnostic, not a correctness test.
4. Provide an explicitly named numerical response backend for the existing FSDP1
   stack only after comparing it against that reference. PyTorch documents that
   [FSDP1 does not support double backward](https://docs.pytorch.org/docs/2.14/fsdp.html).
   Enabling `create_graph=True` in the current actor is therefore not a sufficient
   production implementation.
5. Keep direct-only training available as a baseline. Record the response backend,
   optimizer, estimator, and approximation settings in every run and checkpoint.

## Change 1: preserve scalar advantages and repair the terminal seed

Primary files: `echo_core_algos.py`, `echo_dp_actor.py`, `echo_ray_trainer.py`.

- Return and store one unmasked, group-relative scalar advantage per trajectory;
  use a proposed `task_advantages` field for the leader batch. Do not try to
  reconstruct it by dividing the existing masked token tensor. In
  `_leader_follower_direction`, broadcast `task_advantages` to the response-token
  shape and let `compute_policy_loss` apply `low_level_loss_mask` once. Build the
  direct loss independently with `high_level_loss_mask`.
- For the leader evaluation batch, form the reasoning direction and terminal tool
  direction from the same scalar task advantages, applying each role mask only
  when evaluating its own loss.
- Keep follower validity-return advantages separate from leader task-return
  advantages. Failed/filtered trajectories need consistent scalar values and masks.
- Separate statistical normalization from token masking: for paper mode, match
  the specified within-query standard deviation, per-trajectory role-token mean
  with a denominator clamped to at least one, group mean, and query mean.
- Retain an all-stochastic-policy-token mask independently of the two loss masks.
  It must exclude prompts, padding, external tool results, and deterministic
  environment insertions while retaining sampled policy tokens needed for the score.

Acceptance:

- Exercise the actual advantage-builder and actor interfaces on a two-rollout
  example with disjoint masks. The tool direction must match the analytical
  nonzero value; the current code gives zero.
- Include a zero-advantage group and a trajectory with no tool tokens. They must
  contribute zero without NaNs.
- Verify that a fixed batch's direct direction is unchanged when only the
  terminal tool-direction plumbing is repaired, with normalization held fixed.

This patch restores the input to the response calculation; it does not establish
that the remaining response estimator is correct.

## Change 2: make adaptation records reproduce the generating state

Primary files: `echo_dp_actor.py`, `echo_fsdp_workers.py`, `echo_ray_trainer.py`;
add a typed `InnerStepRecord` in a dedicated response-state module if useful.

Each record must preserve:

- outer-round and inner-step IDs;
- query-group IDs, trajectory member IDs, and expected group sizes;
- complete tokens, attention/position data, both role masks, and policy-token mask;
- raw returns and unmasked scalar advantages;
- generating-policy snapshot identifier, behavior log probabilities and effective
  sampling settings;
- pre-update logical/realized weights and any optimizer state, learning rate,
  RNG/model state required to reproduce that update;
- objective coefficients and normalization settings.

For the reference implementation, save each pre-update state directly. For the
large model, use CPU snapshots or checkpointed deterministic reconstruction of
updates on the saved trajectories. The reverse pass must not regenerate tool
calls or substitute final-policy weights for intermediate states.

Preserve complete query groups as statistical units. Microbatches may split work,
but a group's contraction and complete group score must be assembled before
multiplication. Keep groups on one data-parallel owner initially; support split
groups only with explicit reductions. Remove deterministic stride subsampling
from paper mode; introduce any stochastic subsampling later with a stated rule.

Acceptance:

- Reloading a record reproduces its pre-update log probabilities and follower
  direction within the chosen numerical tolerance.
- Changing microbatch packing leaves the group estimator unchanged.
- No group members are dropped or duplicated by sequence-length balancing.
- Replay restores the live training state on completion or failure.

## Change 3: implement the full response reference

Primary files: replace the mathematical core of `echo_response.py`; keep actor
integration separate from the pure reference functions. Add a small-model test
module that can run without Ray, vLLM, or FSDP.

For each saved inner step, implement the proposed `response_block_vjp` helper.
It returns a derivative-vector product for the requested logical coordinate:

```python
def response_block_vjp(record, incoming_adjoint, coordinate):
    group_products = []
    for query_group in record.query_groups:
        # Ascent direction from saved scalar follower advantages and tool tokens.
        group_direction = follower_direction(query_group)
        contraction = tensor_dot(group_direction, incoming_adjoint.detach())

        # Differentiate the contraction while holding sampled data fixed.
        fixed_data_product = grad_wrt(contraction, coordinate)

        # Full group log probability includes reasoning AND tool policy tokens.
        group_score = grad_wrt(full_group_log_prob(query_group), coordinate)
        sampling_product = group_score * contraction.detach()
        group_products.append(fixed_data_product + sampling_product)

    return mean(group_products) + regularizer_vjp(
        record, incoming_adjoint, coordinate
    )
```

The helper names in this snippet are proposed reference interfaces. `grad_wrt`
must preserve the graph needed for the fixed-data product. `tensor_dot` must
return a differentiable scalar tensor: the existing `echo_response.dot` returns
a Python float through `.item()` and cannot serve that purpose. Group averaging
uses query counts; `regularizer_vjp` supplies each regularizer contribution once.

Build `follower_direction` from the fixed-data log-probability surrogate, with
saved samples and scalar advantages held constant. The existing
`follower_surrogate(log_prob, advantages, tool_mask, loss_agg_mode)` is a useful
integration point, but its sign, group reduction, and normalization must match
the reference. Do not differentiate a detached-behavior PPO ratio twice:
matching its first derivative at ratio one does not make its second derivative
the required fixed-data product.

Replay every block at its recorded pre-update weights under the fixed leader.
Because the current routing fully overlaps, `full_group_log_prob` must include
all stochastic reasoning and tool tokens. Replace the reasoning-only behavior
of `reasoning_score`; do not normalize the response by reasoning-token count.

Replace `_response_gradient` with the following reverse sweep. The mathematical
operations shown on tensors must become buffer operations in the sharded actor:

```python
adjoint = g_fol.clone()
g_resp = zeros_like(g_fol)

for step in reversed(range(len(self._follower_records))):
    record = self._follower_records[step]
    restore_inner_state(record)
    incoming_adjoint = adjoint

    leader_product = response_block_vjp(
        record, incoming_adjoint, coordinate="leader"
    )
    g_resp = g_resp + record.step_size * leader_product

    if step > 0:
        follower_product = response_block_vjp(
            record, incoming_adjoint, coordinate="follower"
        )
        adjoint = incoming_adjoint + record.step_size * follower_product

g_total = g_dir + g_resp
restore_leader_weights()
write_parameter_grads(-g_total)
leader_optimizer.step()
```

`restore_inner_state`, `restore_leader_weights`, and `write_parameter_grads` are
proposed interfaces to the worker's state and gradient handling. Both products
for a step use the same unchanged `incoming_adjoint`. Compute `g_dir` and `g_fol`
at the terminal adapted state before replay, then preserve those buffers across
restores. Apply the total gradient once after discarding the temporary follower.
Maintain an independent forward-sensitivity implementation for small models.

Acceptance:

- Enumerate all query-group outcomes in a small categorical model. Compare the
  expected local corrected block with a finite difference of the expected
  follower direction. This must detect omission of the group-score term.
- Compare forward sensitivity and reverse adjoints at K=1,2,8 for identical local
  blocks; use examples where both `leader_product` and `follower_product` are nonzero.
- On a deterministic expected-update toy, compare the composed gradient with
  directional finite differences.
- Verify signs, group normalization, and a genuine zero-response control case.

Do not demand that a finite-sample GRPO composition be an unbiased derivative of
the expected post-adaptation reward: the paper itself distinguishes its composed
oracle bias from the local block identities.

## Change 4: implement a supported distributed response backend

Primary files: `echo_response.py`, `echo_fsdp_workers.py`, `echo_dp_actor.py`,
configuration and launchers.

The exact small-model reference is the acceptance oracle. For scaling, either
use a derivative worker whose model kernels and distributed parameter handling
pass higher-derivative tests, or retain FSDP1 with an explicitly numerical backend.
Do not assume that changing an FSDP version or enabling original parameters
automatically establishes higher-derivative support.

The practical FSDP1 route is a central-difference evaluation of the fixed-data
block action:

```python
plus_gradient = fixed_log_surrogate_gradient(
    record,
    follower_offset=epsilon * incoming_adjoint,
    coordinate=coordinate,
)
minus_gradient = fixed_log_surrogate_gradient(
    record,
    follower_offset=-epsilon * incoming_adjoint,
    coordinate=coordinate,
)
fixed_data_product = (plus_gradient - minus_gradient) / (2 * epsilon)
```

The proposed `fixed_log_surrogate_gradient` returns the ascent gradient for
`coordinate`, with the follower perturbed relative to the recorded pre-step
state. It uses the same saved trajectories and scalar advantages in both calls.
It computes gradients of the log-probability surrogate, not a clipped PPO ratio.
Add the complete group-score contribution separately. Under the current shared
identity routing, some products coincide and can be reused after validation.

Per-group scalar differences can also estimate the contraction between
`group_direction` and `incoming_adjoint`, avoiding a
separate backward pass for every query. This introduces an additional numerical
approximation and must be checked against exact group contractions.

Requirements:

- restore the exact pre-step state for both perturbations and the unperturbed
  score calculation;
- hold tokens, advantages, behavior denominators, and RNG fixed;
- choose perturbation precision and scale explicitly: small perturbations can
  disappear in low-precision weights;
- account for FSDP gradient averaging and sharding exactly once, use global
  query/group counts, and keep statistical group identities across ranks;
- record epsilon, precision, and backend in checkpoints and reported results.

Acceptance: an epsilon sweep against the exact reference; stable directional
error over an appropriate range; one-rank versus multiple-rank agreement on the
same global batch; and microbatch/repartition invariance. If the numerical backend
cannot meet its declared error tolerance, use the compatible derivative worker
or a smaller trainable model. It must not silently revert to the current heuristic.

Adam/AdamW is deferred. Reintroducing it requires an augmented-state reverse map
for first and second moments, bias correction, weight decay, and any clipping or
skipped-update branch. The current frozen diagonal multiplier must be removed
from the paper implementation.

## Change 5: align validation, configuration, and checkpoints

Primary files: `echo_ray_trainer.py`, `echo_fsdp_workers.py`, `echo_trainer.yaml`,
`training_config/*.yaml`, `scripts/train.sh`, and reward/rollout interfaces as needed.

- Evaluate the adapted pair under the current leader: clone its follower base,
  adapt on training-distribution adaptation queries, then score on held-out task
  queries. Do not use held-out answers for follower adaptation.
- Separate leader-only diagnostics from adapted-pair metrics. Select the best
  adapted model using the latter, and evaluate the final adapted artifact.
- Validation must preserve the training weights, optimizer/scheduler state,
  dataloader position and RNG state. It must not consume training steps or advance
  the training LR schedule through the generic training iteration helper.
- Save resumable checkpoints with leader optimizer state and full configuration;
  distinguish them from inference-only adapted exports. Test resumed and uninterrupted
  training on a deterministic small run.
- Provide a strict reference profile: one update per fresh batch, complete groups,
  consistent per-query normalization, explicit objective coefficients, SGD,
  no unsupported hidden clipping/decay, full replay and response coefficient one.
- Use independent query draws in the strict sampling profile; retain the shuffled
  stream as an explicitly documented practical sampling variant.
- Treat entropy-only advantages, AEPO branching, partial replay, Adam, and other
  variants as separately identified configurations whose derivatives and sampling
  assumptions require their own validation.

## Change 6: stage the experiments and control cost

Run in this order:

1. CPU/small-model mathematical tests and mask regression.
2. Single-GPU K=1 and K=2 integration runs.
3. Multi-GPU agreement checks, then K=8 smoke runs.
4. Matched-rollout-budget comparison of direct-only, corrected one-step, and
   corrected multi-step training.

Record task performance, tool validity, direct/seed/response norms, per-step
adjoint norms, zero-advantage-group fraction, sampled tokens/tool calls, generation
time, derivative-replay time, communication time, and peak memory. A benchmark
win is not required to establish derivative correctness, and a nonzero response
norm does not establish convergence.

Use saved trajectories for every derivative pass. No generation is needed during
the reverse sweep; memory can be traded for model recomputation through saved-state
checkpointing. Small K is an initial testing choice, not a substitute for implementing
the multi-step recursion. Any adapter restriction changes the trainable policy class
and must be stated in experiments.

## Delivery order and completion criteria

Deliver separate reviewable changes in the order above. Changes 1 and 2 are
prerequisites for the reference; the exact reference is the prerequisite for
accepting a production response backend. Evaluation work can proceed in parallel
once its state-restoration interface is defined.

The repair is complete when the mask regression, expected local derivative checks,
multi-step forward/reverse checks, saved-state replay checks, group/rank invariance,
adapted-pair evaluation, and checkpoint-resume checks all pass, and the run metadata
identifies every remaining numerical approximation. The result should be described
as an implementation of the paper's practical response estimator; it does not by
itself establish the strong assumptions or convergence guarantees for a neural LLM.
