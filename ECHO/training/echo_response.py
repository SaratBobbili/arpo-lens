# Copyright 2026 ECHO contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Response term for ECHO Algorithm 1: ``g_resp = Z_K^T g_fol`` (Eqs. 15-16).

The leader's response-aware gradient is ``g_GRPO_K,t = g_dir + g_resp``. Today's
alternating GRPO uses ``g_dir`` alone, which is the *frozen-response* gradient whose
displacement from the Stackelberg equilibrium Proposition 1 bounds.

Under the role-isolated coordinates of Eq. (55) -- the parameterization the paper's own
Section 6 analysis uses -- the fixed-rollout block of Eq. (49) vanishes and ``H_yx`` is
exactly the trajectory-score term. Taking ``v_s ~ g_fol`` in the reverse sweep (dropping
``O(K eta_L L_L)``) and folding the follower's AdamW preconditioner in for ``eta_L``
(App. C.3 requires the follower optimizer state be differentiated through) leaves::

    g_resp = coef * sum_s sum_b  c_{s,b} * S_{x,s,b}

    c_{s,b}   = <g_grp_{L,s,b}, P * g_fol>                      scalar per block
    S_{x,s,b} = sum_j sum_{l in I_H(tau_j)} grad_x log pi_H(z_l)  reasoning-token score

Read plainly: reinforce the reasoning tokens of the follower-phase rollouts, with reward
= how much the tool update they induced helped the final task.

Blocks are micro-batches. ``E[S] = 0`` (Eq. 57), so the signal lives in the covariance of
``S_b`` with ``c_b``; under any partition the cross terms obey
``E[c_b S_b'] = E[c_b] E[S_b'] = 0``, so micro-batch granularity is unbiased and coarser
blocks only cost variance.

Every quantity here is a gradient of the *same* parameter vector under a different loss
mask, so nothing needs parameter partitioning or double backward. This matters: the actor
is FSDP1 with ``use_orig_params=False``, where ``torch.autograd.grad`` does not work at
all (the reduce-scatter lives in non-differentiable post-backward hooks). Gradients are
therefore always read off ``p.grad`` after ``.backward()``, and every inner product is a
local partial that must be all-reduced -- each rank owns a disjoint shard.
"""

import torch
import torch.distributed as dist

from .echo_core_algos import agg_loss


def trainable_params(module) -> list:
    """The parameter list every buffer in this module is aligned against.

    FlatParameter ordering is stable across ranks and across calls, so snapshots taken at
    different times are element-aligned with no extra bookkeeping.
    """
    return [p for p in module.parameters() if p.requires_grad]


def grads(params) -> list:
    """Current ``p.grad`` shards, substituting zeros where a parameter got no gradient."""
    return [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]


def clone_grads(params) -> list:
    """Detached copy of the current gradient shards."""
    return [g.detach().clone() for g in grads(params)]


def dot(a_list, b_list, group=None) -> float:
    """Global inner product of two shard-aligned buffers.

    Each rank holds a disjoint slice of the flattened parameter vector, so the global dot
    product is the sum of the per-rank partial dots.
    """
    local = torch.zeros((), dtype=torch.float32, device=a_list[0].device)
    for a, b in zip(a_list, b_list):
        local += torch.sum(a.to(torch.float32) * b.to(torch.float32))
    if dist.is_initialized():
        dist.all_reduce(local, op=dist.ReduceOp.SUM, group=group)
    return local.item()


def norm(buf_list, group=None) -> float:
    """Global L2 norm of a shard-aligned buffer."""
    return dot(buf_list, buf_list, group) ** 0.5


def copy_grads_into_(buf_list, params) -> list:
    """In-place ``buf <- p.grad``, reusing an existing buffer.

    Preferred over :func:`clone_grads` once a buffer exists. On a 7B actor each full
    gradient buffer is ~3.8 GiB per rank, and allocating a second one mid-step fragments
    the caching allocator badly enough that a later ``empty_cache()`` cannot hand the
    memory back to vLLM's ``wake_up``.
    """
    for buf, g in zip(buf_list, grads(params)):
        buf.copy_(g)
    return buf_list


def scale_by_adam_precond_(buf_list, optimizer, params) -> list:
    """In-place ``buf *= lr / (sqrt(v_hat) + eps)``, one parameter at a time.

    Same map as :func:`adam_precond`, but it never materialises the full preconditioner,
    which would be a second full-size buffer alongside the one being scaled, and it keeps
    to a single temporary per parameter rather than one per arithmetic step.
    """
    for buf, factor in zip(buf_list, _adam_precond_iter(optimizer, params)):
        if isinstance(factor, float):
            buf.mul_(factor)
        else:
            buf.mul_(factor[0]).div_(factor[1])
            del factor
    return buf_list


def _adam_precond_iter(optimizer, params):
    state_by_param = optimizer.state
    seen = 0
    for group in optimizer.param_groups:
        lr = float(group["lr"])
        eps = float(group.get("eps", 1e-8))
        beta2 = float(group.get("betas", (0.9, 0.999))[1])
        for p in group["params"]:
            if not p.requires_grad:
                continue
            seen += 1
            state = state_by_param.get(p, {})
            exp_avg_sq = state.get("exp_avg_sq")
            if exp_avg_sq is None:
                # Before the follower's first step the state is empty and the
                # preconditioner is just lr, which is the correct limit.
                yield lr
                continue
            step = state.get("step", 0)
            step = float(step.item()) if torch.is_tensor(step) else float(step)
            bias_correction2 = 1.0 - beta2**step if step > 0 else 1.0
            # One temporary per parameter, built in place: sqrt(v_hat) + eps. The caller
            # applies it as buf *= lr; buf /= denom, so no second full-size tensor is ever
            # live at the same time.
            #
            # copy=True is load-bearing: exp_avg_sq is already fp32 here, so a plain
            # .to(torch.float32) would alias the optimizer's own second-moment state and
            # the in-place ops below would silently destroy it.
            denom = exp_avg_sq.to(torch.float32, copy=True)
            denom.div_(max(bias_correction2, 1e-12)).sqrt_().add_(eps)
            yield (lr, denom)
    assert seen == len(params), (
        f"preconditioner/parameter mismatch: {seen} vs {len(params)}. The follower "
        "optimizer must be built over exactly the trainable actor parameters."
    )


def adam_precond(optimizer, params) -> list:
    """Follower AdamW preconditioner ``lr / (sqrt(v_hat) + eps)``.

    Eq. (13) writes the inner update as plain ascent ``y + eta_L g``, but C.3 is explicit
    that "momentum or Adam augments the follower state, is reset with y_init, and is
    differentiated through by the same reverse sweep". Replacing the scalar ``eta_L`` by
    this per-coordinate map is what carries that through, and it folds the inner step size
    in so ``phases.response.coef`` stays a pure tuning multiplier.

    Before the follower has taken any step the state is empty; the preconditioner is then
    just ``lr``, which is the correct limit.
    """
    state_by_param = optimizer.state
    precond = []
    for group in optimizer.param_groups:
        lr = float(group["lr"])
        eps = float(group.get("eps", 1e-8))
        beta2 = float(group.get("betas", (0.9, 0.999))[1])
        for p in group["params"]:
            if not p.requires_grad:
                continue
            state = state_by_param.get(p, {})
            exp_avg_sq = state.get("exp_avg_sq")
            if exp_avg_sq is None:
                precond.append(torch.full_like(p, lr, dtype=torch.float32))
                continue
            step = state.get("step", 0)
            step = float(step.item()) if torch.is_tensor(step) else float(step)
            bias_correction2 = 1.0 - beta2**step if step > 0 else 1.0
            v_hat = exp_avg_sq.to(torch.float32) / max(bias_correction2, 1e-12)
            precond.append(lr / (v_hat.sqrt() + eps))

    assert len(precond) == len(params), (
        f"preconditioner/parameter mismatch: {len(precond)} vs {len(params)}. The follower "
        "optimizer must be built over exactly the trainable actor parameters."
    )
    return precond


def scale_(buf_list, other_list) -> list:
    """In-place elementwise ``buf *= other``."""
    for buf, other in zip(buf_list, other_list):
        buf.mul_(other)
    return buf_list


def follower_surrogate(log_prob, advantages, tool_mask, loss_agg_mode: str):
    """Eq. (47): the follower phase's group gradient contribution, as a loss to backward.

    ``g_grp_{L,s,b} = (1/G) sum_j A_j / |I_L(tau_j)| sum_{l in I_L} grad_y log pi_L(z_l)``

    Deliberately *not* the clipped ratio surrogate. C.3: "the displayed protocol takes one
    update on each fresh follower-phase group. Its behavior snapshot generated that group,
    so the token ratio equals one and clipping is locally inactive at the displayed
    gradient." The plain score form is the correct object here, not a simplification.

    ``loss_agg_mode`` is the follower update's own aggregation, so the replayed gradient
    points the same way as the update it stands for.
    """
    return -agg_loss(loss_mat=log_prob * advantages, loss_mask=tool_mask, loss_agg_mode=loss_agg_mode)


def reasoning_score(log_prob, reasoning_mask):
    """``S_x,b``: the plain sum of reasoning-token log-probs over a block (Eq. 48).

    A sum, not a mean -- Eq. (48) defines the group score as a sum over every stochastic
    token in the query group, and the relative weighting of differently sized blocks is
    part of the estimator. Callers divide by a single global token count so the magnitude
    stays sane without disturbing those relative weights.
    """
    return (log_prob * reasoning_mask).sum()
