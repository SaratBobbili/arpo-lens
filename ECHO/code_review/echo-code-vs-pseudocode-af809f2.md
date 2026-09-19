# ECHO code versus paper pseudocode: af809f2

Comparison of the successful fast-forward pull from `45ec6fb` to
`af809f271fd9f4236adde0893234e4a7cdb01443` on branch `echo`. The new head is dated
2026-09-18 21:33:13 CDT. Six new commits add the response path, round boundaries,
follower reward, diagnostics, and launch-profile changes. The code working tree is clean.

Paper target: the current local `2026-ECHO/2026-ICLR-ECHO-v10` sources, especially
[Algorithm 1](/Users/sshakkot/L-WORK/L-Research/2026-ECHO-Project/2026-ECHO/2026-ICLR-ECHO-v10/sections/algorithm.tex:28)
and the [score-corrected derivative blocks](/Users/sshakkot/L-WORK/L-Research/2026-ECHO-Project/2026-ECHO/2026-ICLR-ECHO-v10/appendix/bilevel-derivations.tex:220).

## Assessment

The response-gradient machinery is now present and enabled in the current launch
profiles. The old conclusion that the implementation contains no response path is
superseded. However, the new path is not a faithful implementation of the paper's
$K$-step response gradient. A masking error suppresses the terminal tool-gradient
seed, and several further approximations remain even after that error is fixed.

## What now matches

- Each outer round snapshots the leader weights, clears follower optimizer state,
  and clears adaptation records. It runs $K$ fresh follower iterations and one fresh
  leader iteration. See [round initialization](/Users/sshakkot/L-WORK/L-Research/2026-ECHO-Project/ECHO-Code/ECHO/training/echo_ray_trainer.py:1017).
- The worker interprets the shared weights as $w_{\mathrm{core}} + x + y$, with overlapping
  routing maps and $y_{\mathrm{init}} = 0$. It restores the saved leader weights before
  applying the leader update. This is a permissible logical-state construction;
  shared physical weights alone are not a mismatch. See [snapshot/restore](/Users/sshakkot/L-WORK/L-Research/2026-ECHO-Project/ECHO-Code/ECHO/training/echo_fsdp_workers.py:319).
- Response mode enforces one optimizer step per fresh collected batch, through one
  PPO epoch and a full-batch optimizer minibatch. Microbatches accumulate gradients
  without stepping between them. The earlier within-batch multi-step staleness is
  therefore removed in this mode. See [validation](/Users/sshakkot/L-WORK/L-Research/2026-ECHO-Project/ECHO-Code/ECHO/training/echo_ray_trainer.py:390).
- The follower now receives a distinct tool-validity reward when response mode is
  enabled; the leader receives the task reward. See [reward selection](/Users/sshakkot/L-WORK/L-Research/2026-ECHO-Project/ECHO-Code/ECHO/training/echo_reward_manager.py:64).
- A separate final $K$-step adaptation is saved to `final_response/actor`. See
  [final response](/Users/sshakkot/L-WORK/L-Research/2026-ECHO-Project/ECHO-Code/ECHO/training/echo_ray_trainer.py:1059).

## 1. Blocking defect: the terminal tool direction uses reasoning-masked advantages

The leader batch's `response_mask` is set to its high-level token mask at
[trainer lines 714–715](/Users/sshakkot/L-WORK/L-Research/2026-ECHO-Project/ECHO-Code/ECHO/training/echo_ray_trainer.py:714).
GRPO constructs token advantages as the scalar trajectory advantage multiplied by
that mask at [core line 168](/Users/sshakkot/L-WORK/L-Research/2026-ECHO-Project/ECHO-Code/ECHO/training/echo_core_algos.py:168).
The new terminal follower-direction calculation then uses those same advantages
with the low-level mask at [actor lines 279–288](/Users/sshakkot/L-WORK/L-Research/2026-ECHO-Project/ECHO-Code/ECHO/training/echo_dp_actor.py:279).

Consequently the effective coefficient is $A_{\mathrm{scalar}}\,m_H\,m_L$, where $m_H$ and $m_L$ are the high- and low-level token masks.
Every tool-only token is eliminated. For disjoint masks, $g_{\mathrm{fol}}=0$, so the
response coefficients and response gradient are zero. If tokenization produces
boundary tokens marked by both masks, only that overlap contributes; it does not
recover the intended tool-token task gradient. The checked-in categories assign
reasoning/answer to high and tool/search/python to low, although retained tag
tokens can produce boundary overlap.

A two-rollout numerical reproduction with rewards `[1, 0]`, reasoning/tool masks
`[1, 0]` and `[0, 1]`, and Bernoulli tool scores `+0.5` and `-0.5` gives:

- intended tool direction using the scalar task advantages: `0.25`;
- direction using the code's composed masks: `0.0`.

Required repair: retain the scalar task advantage or an unmasked broadcast copy,
then apply the reasoning and tool masks separately to form the two leader-phase
directions. This should have a regression test through the actual advantage and
actor interfaces.

## 2. No $K$-step adjoint propagation

The paper propagates

$$
v_s = \left(I+\eta_L H_{yy,s}\right)^{\top}v_{s+1}
$$

and accumulates the mixed-block products using the corresponding adjoints. The implementation
constructs one $v$ from the terminal follower direction and a final Adam diagonal,
then uses it unchanged for every record. See
[actor lines 331–363](/Users/sshakkot/L-WORK/L-Research/2026-ECHO-Project/ECHO-Code/ECHO/training/echo_dp_actor.py:331).
The [new helper's own description](/Users/sshakkot/L-WORK/L-Research/2026-ECHO-Project/ECHO-Code/ECHO/training/echo_response.py:15)
acknowledges the approximation $v_s\approx g_{\mathrm{fol}}$. Current profiles still use $K=8$.
The asserted order of the approximation error in the comment has not been
established for this implementation.

## 3. Role-isolation simplification applied to overlapping shared coordinates

The helper drops the fixed-rollout mixed derivative on the premise of role
isolation and uses only reasoning-token scores. But the actual logical mapping
is $P_H=P_L=I$: changing $x$ changes tool-token likelihoods as well. The paper
explicitly says that, for shared weights, both the fixed-rollout derivative and
sampling-law term can contribute. Its group score covers all stochastic
reasoning and tool tokens.

Thus both the omitted fixed mixed block and the reasoning-only score need to be
revisited for the chosen shared-coordinate realization. Alternatively, implement
genuinely isolated policy coordinates and validate their conditional likelihoods.

## 4. Old adaptation data are evaluated at the final policy

The saved [record fields](/Users/sshakkot/L-WORK/L-Research/2026-ECHO-Project/ECHO-Code/ECHO/training/echo_dp_actor.py:29)
contain tokens, masks, and advantages, but not the corresponding inner parameter
or optimizer states. Both response passes call the live model at the end of
adaptation, without restoring each generating inner state. See
[replay](/Users/sshakkot/L-WORK/L-Research/2026-ECHO-Project/ECHO-Code/ECHO/training/echo_dp_actor.py:345).

The paper instead evaluates each local block at its own $(x_t,y_s)$. Merely
reusing saved trajectories is appropriate for derivative evaluation; evaluating
all of them under the final policy is an additional approximation. The replay
loss has no behavior-policy importance correction either. This issue remains
even at $K=1$, where it evaluates the adaptation record at $y_1$ instead of $y_0$.

## 5. Frozen Adam scaling is not differentiation through Adam

The implementation scales the terminal direction by
$\mathrm{lr}/(\sqrt{\widehat{v}}+\epsilon)$ from the final follower optimizer state. It does not
differentiate first- or second-moment transitions, and uses the same final scale
for every inner step. See [preconditioner](/Users/sshakkot/L-WORK/L-Research/2026-ECHO-Project/ECHO-Code/ECHO/training/echo_response.py:97).

Here $\widehat{v}$ is Adam's bias-corrected second moment, distinct from the
reverse-pass adjoint $v_s$.

For example, the first scalar Adam ascent update from zero moments, for $g>0$,
is

$$
\Delta(g)=\frac{\mathrm{lr}\,g}{g+\epsilon}.
$$

Its derivative with respect to $g$ is

$$
\frac{\mathrm{d}\Delta}{\mathrm{d}g}
=\frac{\mathrm{lr}\,\epsilon}{(g+\epsilon)^2}
\ne\frac{\mathrm{lr}}{g+\epsilon}.
$$

A numerical finite-difference check confirms
the difference. The diagonal multiplier is an approximation, not the optimizer
state differentiation required by the paper.

## 6. Query-group estimator and normalization differ

The paper averages complete query-group contributions and uses each complete
group's trajectory score. The new replay does not retain query IDs and partitions
records by token-length microbatches. GRPO advantages couple members of a query
group, so arbitrary splitting does not inherit independence of the cross terms.
The code's claim that any microbatch partition is unbiased is not justified.

The final correction is additionally divided by the number of replayed reasoning
tokens, rather than the paper's query-count normalization. See
[actor lines 369–394](/Users/sshakkot/L-WORK/L-Research/2026-ECHO-Project/ECHO-Code/ECHO/training/echo_dp_actor.py:369).
This changes its scale relative to the direct gradient and makes it depend on
trajectory lengths. It requires either an explicitly revised estimator or a
correction to match the paper.

## Other experiment/protocol differences

- Validation and best-checkpoint selection occur after the adapted tool has been
  discarded and the leader updated. They therefore evaluate the bare leader,
  rather than the newly adapted pair. The final adapted artifact is saved but
  not evaluated by `_run_final_response`.
- The current 7B profile still disables within-group standard-deviation
  normalization and entropy/KL regularization. These settings need to be
  reported accurately. An entropy-advantage profile also exists and should not
  be conflated with the validity-return GRPO profile.
- Query batches are consecutive chunks of a shuffled finite dataset, rather
  than independent with-replacement queries as written in the pseudocode.
- The follower LR scheduler is not reset, but the shipped constant/no-warmup
  schedule makes this harmless for the default step-size protocol. It matters
  for configurable warmup/cosine alternatives.

## Verification performed and limits

- Successfully pulled the six commits and confirmed the clean working tree.
- Traced the current launcher, trainer, worker, actor, reward, and replay paths.
- Parsed all eight changed Python files successfully with Python's AST parser.
- Ran small numerical checks reproducing the mask annihilation and the Adam
  derivative/preconditioner distinction.
- Inspected `.s6_dryrun_check.py`: it tests configuration, validity rewards,
  batching assertions, and assembly of a chosen $\sum_b c_b S_b$ expression. It
  does not test that this expression equals the paper's response derivative,
  nor the actual masked terminal-gradient pipeline.
- Did not run the GPU training stack or the full dry-run test: the local
  interpreters lack PyTorch, Hydra, Ray, and TensorDict. No packages were installed.
- No implementation changes were made during this comparison.

The paper separately analyzes a RLOO proxy under stated assumptions. Neither the
presence of this new correction nor a nonzero response-norm diagnostic establishes
the paper's convergence claims for practical GRPO. Correctness should first be
checked against enumerated expected local derivatives and a small reference
implementation of the complete response recursion.
