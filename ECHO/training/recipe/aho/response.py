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

AHO estimator (``phases.response.estimator: aho``), GRPO variant
---------------------------------------------------------------

arXiv:2607.28849 (Saxena, Gaur, Aggarwal), Theorem 3.3 / Eq. (10). Same round structure
as Algorithm 1; only the response term changes. For an entropy-regularised follower
with temperature ``tau`` the optimum is Boltzmann, ``pi_L*(a|s) = exp((Q - V)/tau)``, so::

    grad_x log pi_L*(a|s) = (grad_x Q(s,a) - grad_x V(s)) / tau =: (W(s,a) - U(s)) / tau
    g_resp                = (1/tau) E[ A_H * sum_{h in I_L} (W - U)(s_h, a_h) ]

Hessian-free: no adjoint, no ``grad^fix``, no per-step weight snapshots, no replay.

In ECHO the follower's reward is x-independent (role-isolated coordinates, Eq. 38) and x
enters through the TRANSITION KERNEL: the reasoning segment between two tool tokens is
``P_x(s'|s,a)``. Differentiating the soft Bellman equation with the envelope identity
``grad_x V(s) = E_{a~pi}[grad_x Q(s,a)]`` (AHO Lemma C.5, Eq. 58) and unrolling gives::

    W(s_h,a_h) = sum_{k>=1} gamma^k E[ V(s_{h+k}) * score(segment before tool token h+k) ]
    score(seg) = sum_{m in seg} grad_x log pi_H,x(m | c)

Summing W alone over h counts every reasoning segment once per tool token before it
(``sum_h W(s_h,a_h)`` regroups to ``j(m) * ...`` with ``j(m)`` = #tool tokens before m,
hundreds here) -- that is the ``U`` term's job to cancel, and dropping it is what blew the
term up to 10^3-10^4x the direct gradient on the 2026-09-25 runs. Keep ``U = grad_x V``
and use the deterministic LLM transition (the sampled segment is the only randomness
between two follower states): the one-sample ``grad_x Q(s_h,a_h) = gamma [V(s_{h+1})
score(seg_h) + grad_x V(s_{h+1})]`` makes ``sum_h (W - U)`` telescope::

    sum_h (W - U) = sum_h gamma^{h} V(s_{h+1}) score(seg_h)  +  gamma^H grad_x V(s_{H+1})  -  grad_x V(s_1)

(the ``(gamma-1) sum grad_x V(s_h)`` remainder unrolls into the ``gamma^h`` factors).
``grad_x V(s_{H+1}) = grad_x R_L = 0`` (R_L is x-independent given the final context) and
``grad_x V(s_1)`` is one vector per query, killed by ``E[A_H | q] = 0``. Each segment is
counted ONCE.

GRPO substitution: ``V(s_{h+1}) -> A_L^j``, the trajectory's group-relative outcome
advantage on R_L (what a follower GRPO step would score this group with), exactly as
GRPO substitutes it for its own value function. The whole term is then one surrogate on
the leader batch::

    omega_m = c_m * A_L^j        j(m) = #tool tokens before m,  c_m = gamma^{j(m)} [j(m) >= 1]  (= indicator at gamma = 1)
    L_resp  = -(coef / tau) * sum_j sum_{m in I_H(j)} A_H^j * omega_m * log pi_x(m)   / N_tool^{mb}

``N_tool^{mb}`` is the micro-batch's tool-token count under ``token-mean`` (per-trajectory
``1/max(1,|I_L(j)|)`` then ``/b`` under ``seq-mean-*``): the aggregation the code's
``g_fol`` uses, with ``grad_y log pi_L(a_h)`` replaced by ``(W-U)/tau``. The m-sum is a
score (a SUM, like :func:`reasoning_score`), never a mean. Read plainly: reinforce the
leader batch's reasoning tokens after the first tool call with weight ``coef * A_H * A_L / tau``.

Conventions and approximations, all named:

* Terminal transition. R_L is the task return scored under the follower's schema rows
  (tool/search/python closed and placed correctly, per ``mask_categories``); a missing
  or unboxed answer is the leader's failure and does not gate it. The reasoning after
  the last tool token is still a transition like any other and carries the same ``A_L``. Reasoning BEFORE the first
  tool token is part of the follower's initial state and gets weight 0 (``g_dir`` credits
  it); no-tool trajectories get 0 throughout.
* ``U(s_h)`` is kept, through the telescoping above, not estimated: the one-sample
  ``grad_x Q`` and the exact ``grad_x V`` cancel between consecutive h whatever estimator
  stands in for ``grad_x V``. What remains is ``grad_x V(s_1)``, one vector per query,
  which the group centring of ``A_H`` removes.
* ``V -> A_L`` drops the per-step entropy return ``-tau log pi_L`` and any state
  dependence of V; ``1/tau`` enters only through the explicit factor.
* The identity is for the EXACT response ``xi(x)`` (ECHO Sec. 3.1), not the finite-K
  ``xi_K``; it is evaluated with ``pi_L(y_K)`` standing in for ``pi_L*``, so its accuracy
  is governed by ``||y_K - xi(x)||`` (AHO Lemma 4.9).
* ``tau`` is the follower's entropy temperature: ``phases.low_level.entropy.reg_coeff``
  (when enabled) plus the follower KL coefficient (KL to a fixed reference is entropy plus
  x-independent shaping). ``phases.response.aho.tau`` overrides it.
* Shared weights and tag tokens outside both masks are dropped exactly as ``g_dir``
  drops them (see :func:`reasoning_score`).
* Zero-std R_L groups (R_L is close to the format gate) give ``A_L = 0`` and no response
  term for that group; the trainer logs the fraction.
"""

import numpy as np
import torch
import torch.distributed as dist

from ...core.core_algos import compute_grpo_outcome_advantage


def trainable_params(module) -> list:
    """The parameter list every buffer in this module is aligned against.

    FlatParameter ordering is stable across ranks and across calls, so snapshots taken at
    different times are element-aligned with no extra bookkeeping.
    """
    return [p for p in module.parameters() if p.requires_grad]


def direct_decomposition(resp_list, total_list, group=None) -> tuple:
    """``(||r||^2, ||t - r||^2, <r, t - r>)`` for shard-aligned ``r`` (response) and ``t``
    (total = direct + response) gradients, each formed elementwise before reduction so no
    large-number cancellation enters. Returns three Python floats."""
    acc = torch.zeros(3, dtype=torch.float32, device=resp_list[0].device)
    for r, t in zip(resp_list, total_list):
        r32 = r.to(torch.float32)
        d32 = t.to(torch.float32) - r32
        acc[0] += torch.sum(r32 * r32)
        acc[1] += torch.sum(d32 * d32)
        acc[2] += torch.sum(r32 * d32)
    if dist.is_initialized():
        dist.all_reduce(acc, op=dist.ReduceOp.SUM, group=group)
    return float(acc[0].item()), float(acc[1].item()), float(acc[2].item())


def grads(params) -> list:
    """Current ``p.grad`` shards, substituting zeros where a parameter got no gradient."""
    return [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]


def clone_grads(params) -> list:
    """Detached copy of the current gradient shards."""
    return [g.detach().clone() for g in grads(params)]




# --- AHO response term, GRPO variant (phases.response.estimator: aho) -----------------
#
# See the module docstring. Three pure functions: A_L for the leader batch, the per-token
# weights omega, and the surrogate whose gradient is -g_resp. All fp32; nothing here
# touches parameters, so the CPU check exercises them exactly as the actor calls them.


def follower_group_advantage(follower_scores, uid, norm_adv_by_std: bool) -> torch.Tensor:
    """``A_L`` for the leader batch: R_L scored as a follower GRPO step would score it. (B, 1)

    Routed through :func:`compute_grpo_outcome_advantage` so epsilon, the singleton rule
    (an excised sample with a fresh uid gets exactly 0) and ``norm_adv_by_std_in_grpo``
    match the follower's own update. R_L is placed at one position with an all-ones
    mask; only the unmasked scalar is returned.
    """
    scores = torch.as_tensor(np.asarray(follower_scores, dtype=np.float32)).reshape(-1)
    token_level = scores.unsqueeze(-1).clone()
    _, _, scalar = compute_grpo_outcome_advantage(
        token_level_rewards=token_level,
        response_mask=torch.ones_like(token_level),
        index=np.asarray(uid),
        norm_adv_by_std_in_grpo=bool(norm_adv_by_std),
    )
    return scalar.to(torch.float32)


def aho_token_weights(high_mask, low_mask, follower_adv, gamma: float = 1.0) -> torch.Tensor:
    """``omega_m = c_m * A_L`` on reasoning tokens, 0 elsewhere. (B, T) fp32, UNNORMALISED.

    ``j(m)`` counts the tool tokens strictly before m. ``c_m = gamma^{j(m)}`` for
    ``j(m) >= 1`` and 0 for ``j(m) = 0``: reasoning before the first tool token and every
    token of a no-tool trajectory get 0, every later reasoning token is counted ONCE (the
    telescoped ``sum_h (W - U)``, see the module docstring), and the segment after the last
    tool token is a terminal transition like any other. ``gamma`` discounts per tool TOKEN,
    the follower MDP's step; at ``gamma = 1`` the weight is the plain indicator.
    """
    high = high_mask.to(torch.float32)
    low = low_mask.to(torch.float32)
    n_before = torch.cumsum(low, dim=-1) - low
    after_first = (n_before >= 1.0).to(torch.float32)
    if float(gamma) == 1.0:
        c = after_first
    else:
        c = torch.pow(torch.full_like(n_before, float(gamma)), n_before) * after_first
    adv = follower_adv.to(torch.float32).reshape(-1, 1)
    return c * adv * high


def aho_response_surrogate(log_prob, scalar_adv, weights, high_mask, low_mask,
                           loss_agg_mode: str, coef: float, tau: float):
    """``L_resp`` for one micro-batch; ``grad_x L_resp = -g_resp``. Returns (loss, diagnostics).

    The per-trajectory score ``S_j = sum_{m in I_H(j)} A_H^j * omega_m * log pi_x(m)`` is a
    SUM over reasoning tokens. The h-sum it stands for is aggregated exactly as the code's
    ``g_fol`` aggregates tool tokens (``_leader_follower_direction`` -> ``agg_loss`` over
    the tool mask), so ``response_to_direct_ratio`` compares like with like:

        token-mean            sum_j S_j / N_tool(micro-batch)
        seq-mean-token-mean   mean_j S_j / max(1, |I_L(j)|)
        seq-mean-token-sum    mean_j S_j
        seq-mean-token-sum-norm  sum_j S_j / T
    """
    # fp32 at least: bf16 log-probs are upcast, a float64 reference stays float64.
    dtype = torch.promote_types(log_prob.dtype, torch.float32)
    high = high_mask.to(dtype)
    low = low_mask.to(dtype)
    adv = scalar_adv.to(dtype).reshape(-1, 1)
    w = weights.to(dtype)
    per_token = adv * w * log_prob.to(dtype) * high
    seq_scores = per_token.sum(dim=-1)
    n_tool = low.sum(dim=-1)
    if loss_agg_mode == "token-mean":
        agg = seq_scores.sum() / torch.clamp(n_tool.sum(), min=1.0)
    elif loss_agg_mode == "seq-mean-token-mean":
        agg = (seq_scores / torch.clamp(n_tool, min=1.0)).mean()
    elif loss_agg_mode == "seq-mean-token-sum":
        agg = seq_scores.mean()
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        agg = seq_scores.sum() / low.shape[-1]
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")
    assert tau > 0.0, f"AHO needs tau > 0, got {tau}"
    loss = -(float(coef) / float(tau)) * agg

    with torch.no_grad():
        n_high = torch.clamp(high.sum(), min=1.0)
        abs_w = (w.abs() * high)
        diag = {
            "aho_surrogate": float(loss.detach().item()),
            "aho_omega_absmean": float((abs_w.sum() / n_high).item()),
            "aho_omega_absmax": float(abs_w.max().item()),
            "aho_weighted_frac": float((((w != 0) & (high > 0)).sum().float() / n_high).item()),
        }
    return loss, diag


