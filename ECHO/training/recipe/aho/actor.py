"""AHO recipe actor: Algorithm 1's response term via the AHO surrogate.

Everything here exists for the response term. The shared two-phase GRPO update
lives in PhaseActorBase (core/phase_actor.py); this class fills its hooks.

g_resp is arXiv:2607.28849's Hessian-free Boltzmann sensitivity, GRPO variant: one
surrogate on the leader batch, evaluated at (x_t, y_K) like the direct term. No
follower records, no replay, no adapted-weight snapshots (_aho_response_gradient;
math in response.py's module docstring). It leaves p.grad = -g_resp for the base
loop to accumulate -g_dir on top of. The adjoint / exact estimators live in
training/recipe/echo.
"""
import gc
import logging
import os

import torch
from verl import DataProto
from verl.utils.device import get_torch_device
from verl.utils.seqlen_balancing import rearrange_micro_batches

from ...core.phase_actor import PhaseActorBase
from . import response as echo_response

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))




class AhoActor(PhaseActorBase):
    def __init__(self, config, actor_module, phase_optims, phase_batch_sizes,
                 restore_leader_weights=None):
        super().__init__(config, actor_module, phase_optims, phase_batch_sizes)
        # Injected by the worker, which owns the W_x snapshot. Called between the last
        # backward and the optimizer step, so the gradient formed at (x_t, y_K) is applied
        # to x_t -- Algorithm 1 lines 13-14.
        self._restore_leader_weights = restore_leader_weights
        # Per-update response flags, parsed by _update_begin from meta_info.
        self._resp = {}


    def _leader_micro_batches(self, mini_batch, micro_batch_size_per_gpu, use_dataproto_batches):
        """Same partition the leader's own update uses, so g_fol and g_dir see one batch."""
        if use_dataproto_batches:
            seqs = mini_batch.batch.batch_size[0]
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                _, idx = rearrange_micro_batches(batch=mini_batch.batch, max_token_len=max_token_len)
                return [mini_batch.select_idxs(p) for p in idx], seqs
            return list(mini_batch.chunk(seqs // micro_batch_size_per_gpu)), seqs
        seqs = mini_batch.batch_size[0]
        if self.config.use_dynamic_bsz:
            max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
            micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            return list(micro_batches), seqs
        return list(mini_batch.split(micro_batch_size_per_gpu)), seqs


    def _aho_response_gradient(self, mini_batch, temperature, mini_batch_size,
                               micro_batch_size_per_gpu, use_dataproto_batches,
                               coef: float, tau: float, gamma: float):
        """AHO estimator: leave ``p.grad = -g_resp`` from ONE pass over the leader batch.

        Backwards ``echo_response.aho_response_surrogate`` on the leader batch's own
        micro-batches (same partition as the direct term) with the same loss scaling the
        base loop applies, so the accumulated gradient is the mini-batch's ``-g_resp`` in
        the same units as ``-g_dir``. Every input is a batch tensor: the trainer's
        ``_after_advantage`` put ``follower_scalar_advantages`` (A_L) and
        ``aho_token_weights`` (omega) there; ``scalar_advantages`` (A_H) and both role
        masks were already carried for the adjoint path.

        Kept as a separate pass rather than folded into the direct loss so
        ``_response_diagnostics`` can still read ``||g_resp||`` and its angle to ``g_dir``.
        """
        self.actor_optimizer.zero_grad(set_to_none=False)
        micro_batches, mini_batch_seqs = self._leader_micro_batches(
            mini_batch, micro_batch_size_per_gpu, use_dataproto_batches
        )
        if not self.config.use_dynamic_bsz:
            self.gradient_accumulation = mini_batch_size // micro_batch_size_per_gpu

        diags = {}
        for micro in micro_batches:
            if isinstance(micro, DataProto):
                micro = {**micro.batch.to(get_torch_device().current_device()), **micro.non_tensor_batch}
            else:
                micro = micro.to(get_torch_device().current_device())
            missing = [k for k in ("follower_scalar_advantages", "aho_token_weights", "scalar_advantages")
                       if k not in micro]
            assert not missing, f"AHO leader micro-batch is missing {missing}"

            response_length = micro["responses"].size(1)
            high = micro["high_level_loss_mask"][:, -response_length:]
            low = micro["low_level_loss_mask"][:, -response_length:]
            weights = micro["aho_token_weights"][:, -response_length:]
            _, log_prob = self._forward_micro_batch(
                micro_batch=micro, temperature=temperature, calculate_entropy=False
            )
            loss, diag = echo_response.aho_response_surrogate(
                log_prob=log_prob,
                scalar_adv=micro["scalar_advantages"],
                weights=weights,
                high_mask=high,
                low_mask=low,
                loss_agg_mode=self.config.loss_agg_mode,
                coef=coef,
                tau=tau,
            )
            if self.config.use_dynamic_bsz:
                loss = loss * (micro["responses"].size(0) / mini_batch_seqs)
            else:
                loss = loss / self.gradient_accumulation
            loss.backward()
            for key, value in diag.items():
                diags.setdefault(key, []).append(value)

        params = echo_response.trainable_params(self.actor_module)
        # The one full-size buffer this path allocates; _response_diagnostics reads it and
        # _on_response_grad_consumed releases it before the optimizer step.
        response_grad = echo_response.clone_grads(params)

        metrics = {f"actor/{k}": (sum(v) / max(len(v), 1)) for k, v in diags.items()}
        metrics["actor/aho_omega_absmax"] = max(diags.get("aho_omega_absmax", [0.0]))
        metrics["actor/aho_tau"] = float(tau)
        metrics["actor/response_blocks"] = float(len(micro_batches))
        return metrics, response_grad

    def _response_diagnostics(self, response_grad):
        """Size of the response term against the direct one, without a second buffer.

        p.grad currently holds -(g_dir + g_resp) and ``response_grad`` holds -g_resp, so
        g_dir follows from three inner products. The norm ratio is the empirical content
        of Proposition 1: how far the frozen-response stationary point sits from the
        Stackelberg one.
        """
        params = echo_response.trainable_params(self.actor_module)
        group = getattr(self.actor_module, "process_group", None)
        total = echo_response.grads(params)

        # g_dir = total - g_resp is formed elementwise, per shard, before any reduction.
        # Recovering ||g_dir||^2 as ||total||^2 - 2<resp,total> + ||resp||^2 cancels two
        # fp32 numbers of ~1e7 when the response term is 1e3x the direct one, and on the
        # 2026-09-25 AHO runs that gave direct_norm in {0, 0.707, 1, 1.225} -- sqrt of the
        # rounding floor -- and a ratio of 0.000 for a term that was in fact dominant.
        n_resp_sq, n_dir_sq, resp_dot_dir = echo_response.direct_decomposition(
            response_grad, total, group
        )
        n_resp, n_dir = n_resp_sq**0.5, n_dir_sq**0.5
        cos = resp_dot_dir / (n_resp * n_dir) if n_resp > 0 and n_dir > 0 else 0.0
        return {
            "actor/response_norm": n_resp,
            "actor/direct_norm": n_dir,
            "actor/response_to_direct_ratio": n_resp / n_dir if n_dir > 0 else 0.0,
            "actor/response_direct_cosine": max(-1.0, min(1.0, cos)),
        }

    # --- PhaseActorBase hooks --------------------------------------------------------

    def _update_begin(self, data, phase):
        """ECHO Algorithm 1 with the AHO surrogate: the follower phase is plain GRPO on
        its own rows; the leader phase adds the response term and applies the result to
        x (lines 12-14)."""
        m = data.meta_info
        response_enabled = bool(m.get("response_enabled", False))
        self._resp = {
            "enabled": response_enabled,
            # AHO: tau is the follower's temperature (or the explicit override), gamma the
            # per-tool-token discount; both resolved by the trainer.
            "aho_tau": float(m.get("aho_tau", 0.0)),
            "aho_gamma": float(m.get("aho_gamma", 1.0)),
            "do_response": response_enabled and phase == "high_level",
            # Algorithm 1 line 14 belongs to the ROUND STRUCTURE, not to the response
            # gradient: the adapted follower must be discarded and the leader step applied
            # to x_t whether or not g_resp was computed. Gating this on do_response would
            # leave y_K in the weights, so the next round's "common base" would not be
            # common and the leader step would land on w_core + x_t + y_K.
            "discard_follower": bool(m.get("discard_follower", False)) and phase == "high_level",
        }

    def _extra_select_keys(self, data):
        """The response term reads both phase masks off the same batch: the tool mask for
        g_fol and the Eq.-(47) surrogate, the reasoning mask for the Eq.-(48) score."""
        if not self._resp.get("enabled"):
            return []
        # A_L and omega, prepared by the trainer's _after_advantage on the leader batch.
        keys = ["high_level_loss_mask", "low_level_loss_mask", "scalar_advantages",
                "follower_scalar_advantages", "aho_token_weights"]
        return [k for k in keys if k in data.batch.keys()]


    def _compute_response_gradient(self, dataloader, phase, *, temperature, mini_batch_size,
                                   micro_batch_size_per_gpu, use_dataproto_batches,
                                   use_aepo_clip, use_sign_cond_clip):
        """Stages 1-3 of the leader update. They leave p.grad = -g_resp, which the caller's
        loop then accumulates -g_dir on top of, so the single optimizer step applies
        g_dir + g_resp (Eq. 16). Requires exactly one optimizer step, which the trainer
        asserts when the response is enabled."""
        if not self._resp.get("do_response"):
            return None, {}
        r = self._resp
        metrics = {}
        assert self.config.ppo_epochs == 1 and len(dataloader) == 1, (
            "phases.response.enabled requires one leader optimizer step per round "
            f"(Algorithm 1 line 13): got ppo_epochs={self.config.ppo_epochs}, "
            f"{len(dataloader)} mini-batches. Set ppo_mini_batch_size to the phase's "
            "prompt batch."
        )
        # No g_fol, no records, no sweep: the Boltzmann sensitivity is a surrogate on
        # the leader batch itself, evaluated at (x_t, y_K) like the direct term.
        response_metrics, response_grad = self._aho_response_gradient(
            dataloader[0], temperature, mini_batch_size, micro_batch_size_per_gpu,
            use_dataproto_batches, coef=r["coef"], tau=r["aho_tau"], gamma=r["aho_gamma"],
        )
        metrics.update(response_metrics)
        return response_grad, metrics

    def _should_zero_grad(self, response_grad):
        """Stage 4: accumulate g_dir on top of g_resp rather than clearing it."""
        return response_grad is None

    def _on_response_grad_consumed(self, response_grad, metrics):
        if response_grad is not None:
            metrics.update(self._response_diagnostics(response_grad))
            # Release the scratch buffer before the step, so its segment is free
            # for the empty_cache() below to return to the driver.
            response_grad.clear()
            response_grad = None
        return response_grad

    def _before_optimizer_step(self, phase, metrics):
        """Stage 5: discard y_K before stepping, so the gradient formed at (x_t, y_K)
        lands on x_t (Algorithm 1 lines 13-14). Without the response gradient this is
        exactly first-order MAML: adapt, evaluate the adapted pair, apply the outer
        gradient to the PRE-adaptation parameters."""
        if not self._resp.get("discard_follower"):
            return
        assert self._restore_leader_weights is not None, (
            "phases.response.enabled needs the worker's restore hook; "
            "AhoActor was built without restore_leader_weights."
        )
        self._restore_leader_weights()
        metrics["actor/follower_discarded"] = 1.0


    def _after_update(self, phase, metrics):
        if not self._resp.get("do_response"):
            return
        # The response path allocates a full-size gradient buffer and a stream of
        # large per-parameter temporaries. verl's sharding manager calls
        # empty_cache() before vLLM's wake_up(), but that only returns fully free
        # segments -- so release ours here, while the allocator can still coalesce,
        # rather than leaving vLLM to fail at create_and_map.
        gc.collect()
        get_torch_device().empty_cache()
        device = get_torch_device()
        metrics["actor/response_mem_reserved_gb"] = device.memory_reserved() / (1024**3)
        metrics["actor/response_mem_allocated_gb"] = device.memory_allocated() / (1024**3)
