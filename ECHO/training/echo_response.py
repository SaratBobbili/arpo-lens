# Copyright 2026 ECHO contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Response term for ECHO Algorithm 1: ``g_resp`` in ``g_leader = g_dir + g_resp``.

Without it the leader update is ``g_dir`` alone -- the frozen-response gradient whose
displacement from the Stackelberg equilibrium Proposition 1 bounds.

v10 Eq. (35) splits the block this rests on into two additive halves::

    H_yc,s = (1/B_L) sum_b [ D^fix_c g_grp,b  +  stopgrad(g_grp,b) * S_c,b^T ]
                             \\___ fixed-data ___/   \\______ sampling ______/

**This module implements the sampling half only**, contracted against ``g_fol``::

    g_resp = coef * sum_s sum_b  c_{s,b} * S_{s,b}

    c_{s,b} = <g_grp_{L,s,b}, g_fol>                    scalar per block
    S_{s,b} = sum_j sum_{l in I_policy} grad log pi(z_l)  score over ALL sampled tokens

Read plainly: reinforce the sampled tokens of the follower-phase rollouts, with reward =
how much the tool update they induced helped the final task. That is the mechanism the
method is about -- reasoning changes the data the tool learner trains on.

Standing approximations, none of them hidden:

* the fixed-data half ``D^fix_c g_grp`` is absent; it needs a Hessian-vector product,
  which FSDP1 cannot provide (``use_orig_params=False``, and the reduce-scatter lives in
  non-differentiable post-backward hooks, so ``torch.autograd.grad`` does not work here);
* the reverse sweep uses ``v_s = g_fol`` for every ``s`` instead of
  ``v_s = v_{s+1} + eta_L H_s^T v_{s+1}``, so cross-step coupling between inner updates
  is dropped;
* blocks are micro-batches rather than whole query groups, which GRPO's within-group
  centering couples, so the partition is not unbiased as once claimed here;
* ``eta_L`` is a plain scalar inside ``coef``, not the optimizer's derivative.

Every quantity is a gradient of the *same* parameter vector under a different loss mask,
so nothing here needs parameter partitioning or double backward. Gradients are read off
``p.grad`` after ``.backward()``, and every inner product is a per-rank partial that must
be all-reduced -- each rank owns a disjoint shard.
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

    NOT WIRED IN, and deliberately so. This is *not* differentiation through the
    optimizer: the derivative of Adam's first update from zero moments is
    ``lr*eps/(g+eps)^2``, which differs from ``lr/(g+eps)`` in form, not just scale (by
    ~5e6 at g=0.05, lr=1e-3, eps=1e-8). Using it only imposed a wrong-shaped
    per-coordinate weighting while looking principled, so ``eta_L`` is now a plain scalar
    folded into ``phases.response.coef``.

    Kept, with its test, because a correct treatment of C.3's "differentiate through the
    optimizer" would start from this state and needs an augmented-state reverse map for
    the moments, bias correction and weight decay.

    Before the follower has taken any step the state is empty; the value is then ``lr``.
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


def reasoning_score(log_prob, policy_mask):
    """``S_x,b``: the plain sum of sampled-token log-probs over a block (Eq. 34).

    A sum, not a mean -- the group score is a sum over every stochastic token in the
    query group, and the relative weighting of differently sized blocks is part of the
    estimator. Callers divide by a single global token count so the magnitude stays sane
    without disturbing those relative weights.

    ``policy_mask`` covers BOTH roles' sampled tokens. Under the shared-weight routing
    used here (P_H = P_L = I) a change in the leader coordinates moves tool-token
    likelihoods too, so a reasoning-only score drops part of the dependence.
    """
    return (log_prob * policy_mask).sum()
