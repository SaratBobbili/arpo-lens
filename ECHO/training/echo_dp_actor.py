"""HYPERGRADIENT actor: ECHO Algorithm 1 on top of ALTERNATING-GRPO.

Everything here exists for the response term. The shared two-phase GRPO update
lives in DataParallelPhaseActor (alt_dp_actor.py); this class fills its hooks.
"""
import gc
import logging
import os
from collections import deque

import torch
from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_torch_device
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches

from .alt_dp_actor import DataParallelPhaseActor
from .echo_core_algos import compute_policy_loss
from . import echo_response

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Tensors a stashed follower record needs to replay its Eq. (47) gradient and its Eq. (48)
# reasoning score. Kept on CPU: ~100 MB per record per rank, ~800 MB at K=8.
_FOLLOWER_RECORD_KEYS = (
    "input_ids",
    "attention_mask",
    "position_ids",
    "responses",
    "old_log_probs",
    "advantages",
    "scalar_advantages",
    "high_level_loss_mask",
    "low_level_loss_mask",
)


class DataParallelECHOActor(DataParallelPhaseActor):
    def __init__(self, config, actor_module, phase_optims, phase_batch_sizes,
                 restore_leader_weights=None, snapshot_adapted_weights=None,
                 restore_adapted_weights=None):
        super().__init__(config, actor_module, phase_optims, phase_batch_sizes)
        # Injected by the worker, which owns the W_x snapshot. Called between the last
        # backward and the optimizer step, so the gradient formed at (x_t, y_K) is applied
        # to x_t -- Algorithm 1 lines 13-14.
        self._restore_leader_weights = restore_leader_weights
        # The exact path also needs to step back to the adapted point after evaluating
        # the response term at w_0, so the direct gradient is still taken at (x_t, y_1).
        self._snapshot_adapted_weights = snapshot_adapted_weights
        self._restore_adapted_weights = restore_adapted_weights
        self._follower_records = deque()
        # Per follower step s, aligned with _follower_records:
        #   _follower_step_weights[s]  w_s, the weights step s was taken FROM. Each of the
        #     K per-step contributions is evaluated there, which is what makes K>1 a sum of
        #     honest K=1 problems rather than K copies of one evaluated at the wrong point.
        #   _follower_grad_norms[s]    ||ghat_s||, so D_s knows whether clipping fired.
        # On the CPU: one 7B fp32 shard is ~3.5 GiB per rank, so K=8 costs ~28 GiB of host
        # RAM against the ~230 GiB per rank the node has.
        self._follower_step_weights = []
        self._follower_grad_norms = []
        # Per-update response flags, parsed by _update_begin from meta_info.
        self._resp = {}

    @GPUMemoryLogger(role="dp actor", logger=logger)
    # --- follower-record stash (Algorithm 1 line 8) -----------------------------------

    def clear_follower_records(self):
        self._follower_records.clear()
        self._follower_step_weights.clear()
        self._follower_grad_norms.clear()

    def _push_follower_step_weights(self):
        """Keep w_s, this rank's shard, before the follower step mutates it."""
        params = echo_response.trainable_params(self.actor_module)
        self._follower_step_weights.append(
            [p.data.detach().to("cpu", copy=True) for p in params]
        )

    def _restore_follower_step_weights(self, idx: int):
        params = echo_response.trainable_params(self.actor_module)
        saved = self._follower_step_weights[idx]
        assert len(params) == len(saved)
        for p, w in zip(params, saved):
            # copy_ in place: FSDP1's flat_param._local_shard aliases this storage.
            p.data.copy_(w)

    def _stash_follower_record(self, data: DataProto, uids=None):
        """Keep this rank's shard of one follower step's rollouts for the reverse sweep.

        Line 8: "retain the fixed-rollout record and complete query-group scores for the
        score-corrected reverse sweep". Stashed in the worker rather than on the driver so
        no single process holds the full B x G batch.

        ``uids`` carries the query-group identity Eq. (49) indexes b over. It is hashed to
        int64 rather than kept as an object array so the record stays all-tensor; only
        equality of adjacent entries is ever read. Taken from the unselected DataProto:
        routing "uid" through select() would flip use_dataproto_batches and change the
        follower's own batching path as a side effect.
        """
        record = {}
        for key in _FOLLOWER_RECORD_KEYS:
            if key in data.batch.keys():
                record[key] = data.batch[key].detach().to("cpu", copy=True)
        missing = [k for k in _FOLLOWER_RECORD_KEYS if k not in record]
        assert not missing, (
            f"follower record missing {missing}; the response term needs both phase masks "
            "on the rollout (ensure rollout.mode=sync_echo)."
        )
        n = record["input_ids"].shape[0]
        if uids is not None:
            assert len(uids) == n, f"{len(uids)} uids for {n} stashed sequences"
            record["group_id"] = torch.tensor(
                [hash(str(u)) & ((1 << 62) - 1) for u in uids], dtype=torch.int64
            )
        else:
            # No uid on the batch: every sequence is its own group. Correct but weak --
            # the within-group cross terms Eq. (49) pairs are simply absent.
            record["group_id"] = torch.arange(n, dtype=torch.int64)
        self._follower_records.append(record)

    def _replay_blocks(self, record):
        """Micro-batch one stashed record onto the GPU.

        Called once per record in each of the two replay passes rather than cached across
        them: caching every record's blocks would hold K full batches on the GPU at once.
        rearrange_micro_batches is deterministic for a given batch, and it all-reduces the
        block count with MAX across the DP group (same_micro_num_in_dp defaults True), so
        the two passes see the same partition and no rank can desync inside a block.
        """
        from tensordict import TensorDict

        device = get_torch_device().current_device()
        batch = TensorDict(
            {k: v.to(device, non_blocking=True) for k, v in record.items()},
            batch_size=record["input_ids"].shape[0],
        )
        if self.config.use_dynamic_bsz:
            max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
            micro_batches, _ = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
            return list(micro_batches)
        _, micro_bsz = self.phase_batch_sizes["low_level"]
        return list(batch.split(micro_bsz if micro_bsz else batch.batch_size[0]))

    def _replay_group_blocks(self, record):
        """Micro-batch one stashed record WITHOUT straddling a query group.

        Returns ``[[block, ...], ...]``, one inner list per group in group order. The
        caller accumulates p.grad across a group's blocks and reads ghat_grp,s,b off it at
        the group boundary -- the b of Eq. (49), which a token-budget partition destroys.

        A group does not fit in one micro-batch here (16 rollouts at ~1.4k tokens against
        an 11264-token budget), so the gradient has to be accumulated over its blocks
        rather than taken from a single backward.
        """
        from tensordict import TensorDict

        device = get_torch_device().current_device()
        group = getattr(self.actor_module, "process_group", None)
        seq_lens = record["attention_mask"].sum(dim=1).tolist()
        spans = echo_response.group_spans(record["group_id"])
        max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
        index_lists = echo_response.group_block_split(seq_lens, spans, max_token_len, group)

        tensor_keys = [k for k in record if k != "group_id"]
        out = []
        for chunks in index_lists:
            blocks = []
            for idx in chunks:
                sel = torch.tensor(idx, dtype=torch.long)
                blocks.append(
                    TensorDict(
                        {k: record[k][sel].to(device, non_blocking=True) for k in tensor_keys},
                        batch_size=len(idx),
                    )
                )
            out.append(blocks)
        return out

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

    def _leader_follower_direction(
        self, mini_batch, temperature, mini_batch_size, micro_batch_size_per_gpu,
        use_dataproto_batches, use_aepo_clip, use_sign_cond_clip,
    ):
        """Stage 1: leave ``p.grad = -g_fol`` (Eq. 14, via Eq. 46's third surrogate).

        ``U_tool_H,t = GRPO(R_H, pi_L; Q_H, G_H)`` -- the leader batch's *task* return
        applied to the *tool* token mask. Identical to the direct surrogate except for
        which mask it scores, since the batch already carries both.

        No KL or entropy term: u_H is the bare task return here (lambda_H = 0), and the
        entropy half of u_L is flag-gated off by default.
        """
        self.actor_optimizer.zero_grad(set_to_none=False)
        micro_batches, mini_batch_seqs = self._leader_micro_batches(
            mini_batch, micro_batch_size_per_gpu, use_dataproto_batches
        )
        if not self.config.use_dynamic_bsz:
            self.gradient_accumulation = mini_batch_size // micro_batch_size_per_gpu

        pg_losses = []
        for micro in micro_batches:
            if isinstance(micro, DataProto):
                micro = {**micro.batch.to(get_torch_device().current_device()), **micro.non_tensor_batch}
            else:
                micro = micro.to(get_torch_device().current_device())

            response_length = micro["responses"].size(1)
            tool_mask = micro["low_level_loss_mask"][:, -response_length:]
            _, log_prob = self._forward_micro_batch(
                micro_batch=micro, temperature=temperature, calculate_entropy=False
            )
            # (b, 1), broadcast against the (b, T) ratio. NOT micro["advantages"], which
            # has the high-level mask already folded in -- multiplying that by the tool
            # mask annihilates every tool token and makes g_fol identically zero.
            advantages = micro["scalar_advantages"]
            pg_loss, _, _, _ = compute_policy_loss(
                old_log_prob=micro["old_log_probs"],
                log_prob=log_prob,
                advantages=advantages,
                response_mask=tool_mask,
                cliprange=self.config.clip_ratio,
                cliprange_low=self.config.clip_ratio_low if self.config.clip_ratio_low is not None else self.config.clip_ratio,
                cliprange_high=self.config.clip_ratio_high if self.config.clip_ratio_high is not None else self.config.clip_ratio,
                clip_ratio_c=self.config.get("clip_ratio_c", 3.0),
                use_aepo_clip=use_aepo_clip,
                use_sign_cond_clip=use_sign_cond_clip,
                clip_sign_advantages=advantages if use_sign_cond_clip else None,
                cliprange_low_pos=self.config.get("clip_ratio_low_pos", 0.2),
                cliprange_high_pos=self.config.get("clip_ratio_high_pos", 0.2),
                cliprange_low_neg=self.config.get("clip_ratio_low_neg", 0.2),
                cliprange_high_neg=self.config.get("clip_ratio_high_neg", 0.2),
                loss_agg_mode=self.config.loss_agg_mode,
            )
            if self.config.use_dynamic_bsz:
                loss = pg_loss * (micro["responses"].size(0) / mini_batch_seqs)
            else:
                loss = pg_loss / self.gradient_accumulation
            loss.backward()
            pg_losses.append(pg_loss.detach().item())

        return {"actor/follower_direction_pg_loss": sum(pg_losses) / max(len(pg_losses), 1)}

    def _response_gradient(self, temperature, coef: float, replay_fraction: float):
        """Leave ``p.grad = -g_resp`` and return diagnostics.

        Assumes ``p.grad = -g_fol`` on entry (stage 1 ran). Then:

          2. per stashed block, backward the Eq.-(47) tool surrogate and read
             ``c = <g_L, P * g_fol>`` -- a local dot plus one all_reduce of one scalar;
          3. re-walk the blocks accumulating ``-coef * c * sum log pi_H``, whose gradient
             is ``-g_resp``.

        Two passes rather than one because ``c`` for a block is only known after that
        block's backward, and FSDP1 cannot hold a graph across the reduce-scatter.
        """
        params = echo_response.trainable_params(self.actor_module)
        group = getattr(self.actor_module, "process_group", None)

        records = list(self._follower_records)
        if not records:
            return {"actor/response_blocks": 0.0}, None

        # v = g_fol. p.grad holds -g_fol and stage 2 reads -g_L, so the two sign flips
        # cancel and v is deliberately carried unnegated.
        #
        # No optimizer preconditioner: eta_L folds into phases.response.coef as a plain
        # tuning scalar. The previous lr/(sqrt(v_hat)+eps) factor was NOT differentiation
        # through Adam (that is lr*eps/(g+eps)^2 for the first step), so it only added a
        # wrong-shaped per-coordinate weighting while looking principled.
        #
        # This is the ONLY full-size buffer the response path allocates: it is reused to
        # hold -g_resp for the diagnostics once stage 2 is done with it. A second buffer
        # here fragments the caching allocator enough that the empty_cache() verl runs
        # before vLLM's wake_up cannot return the memory, which OOMs the next generation.
        v = echo_response.clone_grads(params)

        # Deterministic stride, never RNG: every rank must keep the same blocks or the
        # all_reduce inside a block desyncs. Subsampling is unbiased -- the estimator is a
        # sum over blocks, so dropping uniformly and rescaling preserves the expectation.
        stride = 1 if replay_fraction >= 1.0 else max(1, int(round(1.0 / max(replay_fraction, 1e-6))))
        loss_agg_mode = self.config.loss_agg_mode

        # Stage 2: c per kept block, plus the reasoning-token count stage 3 normalises by.
        coeffs, kept_per_record, reasoning_tokens = [], [], 0.0
        for record in records:
            blocks = self._replay_blocks(record)
            kept = list(range(0, len(blocks), stride))
            kept_per_record.append((len(blocks), kept))
            for b_idx in kept:
                block = blocks[b_idx]
                response_length = block["responses"].size(1)
                self.actor_optimizer.zero_grad(set_to_none=False)
                _, log_prob = self._forward_micro_batch(
                    micro_batch=dict(block), temperature=temperature, calculate_entropy=False
                )
                loss = echo_response.follower_surrogate(
                    log_prob=log_prob,
                    advantages=block["advantages"],
                    tool_mask=block["low_level_loss_mask"][:, -response_length:],
                    loss_agg_mode=loss_agg_mode,
                )
                loss.backward()
                coeffs.append(echo_response.dot(echo_response.grads(params), v, group))
                # Must match the mask the score uses below, or the normalizer rescales
                # the term by a ratio that drifts with the reasoning/tool token mix.
                reasoning_tokens += float(
                    torch.clamp(
                        block["high_level_loss_mask"][:, -response_length:]
                        + block["low_level_loss_mask"][:, -response_length:],
                        max=1.0,
                    ).sum()
                )
            del blocks

        # v stays allocated: it is reused below to carry -g_resp rather than allocating a
        # second full-size buffer.
        total_blocks = sum(n for n, _ in kept_per_record)
        rescale = total_blocks / max(len(coeffs), 1)

        # Stage 3: accumulate -coef * c * S_x. S_x is a plain sum over reasoning tokens
        # (Eq. 48); one global token count keeps the magnitude sane without disturbing the
        # relative weight of differently sized blocks.
        normalizer = max(reasoning_tokens, 1.0)
        self.actor_optimizer.zero_grad(set_to_none=False)
        cursor = 0
        for record, (n_blocks, kept) in zip(records, kept_per_record):
            blocks = self._replay_blocks(record)
            assert len(blocks) == n_blocks, (
                f"replay partition changed between passes ({len(blocks)} vs {n_blocks}); "
                "the two sweeps must see identical blocks."
            )
            for b_idx in kept:
                block = blocks[b_idx]
                response_length = block["responses"].size(1)
                _, log_prob = self._forward_micro_batch(
                    micro_batch=dict(block), temperature=temperature, calculate_entropy=False
                )
                policy_mask = torch.clamp(
                    block["high_level_loss_mask"][:, -response_length:]
                    + block["low_level_loss_mask"][:, -response_length:],
                    max=1.0,
                )
                score = echo_response.reasoning_score(log_prob, policy_mask)
                scale = coef * rescale * coeffs[cursor] / normalizer
                (-scale * score).backward()
                cursor += 1
            del blocks

        # Reuse v's allocation to carry -g_resp out for the diagnostics.
        echo_response.copy_grads_into_(v, params)

        c_tensor = torch.tensor(coeffs, dtype=torch.float32)
        return {
            "actor/response_blocks": float(len(coeffs)),
            "actor/response_c_mean": float(c_tensor.mean()),
            "actor/response_c_std": float(c_tensor.std()) if len(coeffs) > 1 else 0.0,
        }, v

    def _follower_loss_on_block(self, block, temperature, use_aepo_clip, use_sign_cond_clip, weight):
        """The loss the follower step actually backwarded, replayed on one block.

        NOT ``echo_response.follower_surrogate``. At ratio = 1 the two agree in VALUE and
        in GRADIENT, which is all C.3 claims, but not in curvature: differentiating
        ``-A*r`` twice leaves ``-A grad^2 log pi - A grad log pi grad log pi^T`` where the
        plain score form leaves only the first. The fixed-data half is a Hessian, so it
        has to be the Hessian of the map the follower really applied.
        """
        response_length = block["responses"].size(1)
        _, log_prob = self._forward_micro_batch(
            micro_batch=dict(block), temperature=temperature, calculate_entropy=False
        )
        advantages = block["advantages"]
        pg_loss, _, _, _ = compute_policy_loss(
            old_log_prob=block["old_log_probs"],
            log_prob=log_prob,
            advantages=advantages,
            response_mask=block["low_level_loss_mask"][:, -response_length:],
            cliprange=self.config.clip_ratio,
            cliprange_low=self.config.clip_ratio_low if self.config.clip_ratio_low is not None else self.config.clip_ratio,
            cliprange_high=self.config.clip_ratio_high if self.config.clip_ratio_high is not None else self.config.clip_ratio,
            clip_ratio_c=self.config.get("clip_ratio_c", 3.0),
            use_aepo_clip=use_aepo_clip,
            use_sign_cond_clip=use_sign_cond_clip,
            clip_sign_advantages=advantages if use_sign_cond_clip else None,
            cliprange_low_pos=self.config.get("clip_ratio_low_pos", 0.2),
            cliprange_high_pos=self.config.get("clip_ratio_high_pos", 0.2),
            cliprange_low_neg=self.config.get("clip_ratio_low_neg", 0.2),
            cliprange_high_neg=self.config.get("clip_ratio_high_neg", 0.2),
            loss_agg_mode=self.config.loss_agg_mode,
        )
        return pg_loss * weight

    def _block_weights(self, blocks, total_seqs):
        """Per-block weights reproducing the follower update's own loss aggregation."""
        if self.config.use_dynamic_bsz:
            return [b["responses"].size(0) / max(total_seqs, 1) for b in blocks]
        return [1.0 / max(len(blocks), 1)] * len(blocks)

    def _hessian_vector_product(self, record, temperature, v_buf, eta_scale, restore_fn,
                                use_aepo_clip, use_sign_cond_clip, fd_rel):
        """Accumulate one step's fixed-data half ``grad^2 L_L . u`` onto ``p.grad``.

        Central differences on the GRADIENT, which is the one second-order object FSDP1
        can give us::

            grad^2 L . u  ~  [ grad L(w + e u) - grad L(w - e u) ] / (2e)

        no ``create_graph``, no ``autograd.grad``, just two ordinary backward passes at
        displaced weights. Error is O(e^2 ||u||^3).

        ``u = D_s^T g_fol = eta_scale * g_fol`` and ``v_buf`` holds ``-g_fol``, so the
        displacement applied to the weights is ``-eta_scale * e * v_buf``. Writing
        ``a = fd_rel * ||w|| / ||v_buf||`` for that displacement's coefficient, eta_scale
        cancels out of the step SIZE -- a difference quotient only needs a direction --
        and survives only in the divisor ``1/(2e) = eta_scale / (2a)``.

        ``restore_fn`` puts the weights back from a snapshot rather than adding the
        perturbation back, so no arithmetic drift accumulates in the master weights. It is
        the caller's business which snapshot: at step s that is w_s, not w_0.
        """
        params = echo_response.trainable_params(self.actor_module)
        group = getattr(self.actor_module, "process_group", None)

        # a is the coefficient applied to v_buf, i.e. eps * ||u|| / ||v_buf||.
        a = echo_response.perturb_eps(params, v_buf, group, rel=fd_rel)
        if a <= 0.0 or eta_scale <= 0.0:
            return {"actor/hvp_skipped": 1.0}
        fd_scale = eta_scale / (2.0 * a)

        total_seqs = record["input_ids"].shape[0]
        # Logged, not asserted. The threshold this used to enforce (>0.5) is unreachable:
        # params[0] is the root flat parameter, [embed_tokens | norm | lm_head], and an
        # embedding row is only touched when its token appears in the batch (~6-17% of a
        # 152k vocab), so on the ranks whose shard lands inside embed_tokens the fraction
        # is capped below the threshold at ANY fd_rel. It is a noise-quality number, not a
        # correctness gate -- phases.response.curvature is the gate.
        moved_frac = self._bf16_moved_fraction(params[0].data, v_buf[0], a)

        for sign in (+1.0, -1.0):
            # u points along -v_buf, so the +u point is reached with a NEGATIVE alpha.
            for p_, d_ in zip(params, v_buf):
                p_.data.add_(d_, alpha=-sign * a)
            blocks = self._replay_blocks(record)
            weights = self._block_weights(blocks, total_seqs)
            for block, weight in zip(blocks, weights):
                loss = self._follower_loss_on_block(
                    block, temperature, use_aepo_clip, use_sign_cond_clip, weight
                )
                (loss * (sign * fd_scale)).backward()
            del blocks
            restore_fn()

        return {
            "actor/hvp_eps_rel": float(fd_rel),
            "actor/hvp_step_coef": float(a),
            "actor/hvp_bf16_moved_frac": moved_frac,
            "actor/hvp_skipped": 0.0,
        }

    @staticmethod
    def _bf16_moved_fraction(weight, direction, a: float) -> float:
        """Fraction of elements whose bf16 image actually changes under the perturbation.

        The parameters are fp32 -- FSDP1's MixedPrecision keeps the flat-parameter storage
        in the original dtype -- so ``p.data.add_`` always lands. The forward does not see
        that copy: it all-gathers a bf16 CAST of it. bf16 carries 8 mantissa bits, so a
        relative displacement under ~3.9e-3 rounds straight back to the unperturbed value
        and the central difference returns an exact, silent zero.

        Checking the fp32 store would therefore pass while the quantity being measured is
        zero, which is the trap this whole diagnostic exists for. Check the cast instead,
        on one parameter, and let the caller log it every leader step.
        """
        with torch.no_grad():
            base = weight.to(torch.bfloat16)
            shifted = (weight + direction * a).to(torch.bfloat16)
            return float((shifted != base).to(torch.float32).mean().item())

    def _exact_response_gradient(self, temperature, v_buf, replay_fraction,
                                 use_aepo_clip, use_sign_cond_clip, fd_rel, grad_clip,
                                 follower_lr, curvature, group_aligned):
        """Leave ``p.grad = -g_resp``, accumulated over the round's K follower steps.

        Implements the score-corrected reverse sweep of App. C.4 directly. For
        ``A_s = I + eta_L Hhat_yy,s`` and ``B_s = eta_L Hhat_yx,s``, Eq. (54) is::

            v_K = ghat_fol ,   v_s = A_s^T v_{s+1} ,   s = K-1 ... 1
            Zhat_K^T ghat_fol = sum_r B_r^T v_{r+1}

        and Eq. (51) evaluates either block against a vector without forming it::

            (Hhat_yc,s)^T v = (1/B_L) sum_b [ grad^fix_c <ghat_grp,s,b, v>
                                              + S_grp,c,s,b stopgrad<ghat_grp,s,b, v> ]

        The first term is the only one FSDP1 cannot do; ``curvature`` gates it. The second
        is a scalar dot and one ordinary backward -- the machinery already here.

        Two things make this cheap. The routing maps are shared (P_H = P_L = I), so
        ``S_x = S_y``: Eq. (48)'s score is the trajectory log-likelihood and its ``c`` index
        picks the parameter block, not the token set. Hence the two blocks coincide,

            (Hhat_yy,s)^T v = (Hhat_yx,s)^T v = (1/B_L) sum_b S_s,b <ghat_grp,s,b, v> =: u_s

        so ONE accumulation serves both the adjoint update and the leader contribution::

            v_s = v_{s+1} + eta_s u_s          g_resp = sum_s eta_s u_s

        i.e. the adjoint is ghat_fol plus the response term accumulated so far, and the
        reverse sweep costs no extra pass, buffer or backward. ``v_s`` is therefore carried
        in ``v_buf`` in place, and the running ``-g_resp`` on the HOST -- p.grad is needed
        as scratch to read each group's gradient, and a second device-side full-size buffer
        fragments the caching allocator enough to OOM vLLM's next wake_up.

        What is still dropped, and it is one thing applied uniformly: ``grad^fix``, at both
        c = x and c = y. Under the role-isolated coordinates of Eq. (55) ``grad^fix_x``
        vanishes identically and B_s is exact; ``grad^fix_y`` does not, so A_s keeps an
        O(eta_L) error and g_resp an O(eta_L^2) one. Under the shared weights actually used
        here C.4 says both can contribute, so neither is exact -- call this the
        score-corrected estimator, not the exact hypergradient.

        Steps run in REVERSE (s = K-1 ... 0): step s needs v_{s+1}, which is only complete
        once every later step has contributed. Within a step the two sweeps stay, because
        c_{s,b} is known only after group b's backward and FSDP1 cannot hold a graph across
        the reduce-scatter.
        """
        params = echo_response.trainable_params(self.actor_module)
        group = getattr(self.actor_module, "process_group", None)

        records = list(self._follower_records)
        assert records, "the response term needs at least one stashed follower record."
        assert len(self._follower_step_weights) == len(records), (
            f"{len(self._follower_step_weights)} weight snapshots for {len(records)} "
            "follower records; every step must snapshot the weights it starts from."
        )
        assert len(self._follower_grad_norms) == len(records), (
            f"{len(self._follower_grad_norms)} grad norms for {len(records)} records."
        )

        # Subsampling drops whole GROUPS, never blocks: Eq. (48)'s score is over the
        # complete query group, so a partial group is a different estimator, not a
        # cheaper one.
        stride = 1 if replay_fraction >= 1.0 else max(1, int(round(1.0 / max(replay_fraction, 1e-6))))
        loss_agg_mode = self.config.loss_agg_mode

        etas, clip_flags = [], []
        for grad_norm in self._follower_grad_norms:
            eta_s, clipped = echo_response.follower_step_jacobian(
                lr=follower_lr, grad_norm=grad_norm, grad_clip=grad_clip
            )
            etas.append(eta_s)
            clip_flags.append(clipped)

        # The running -g_resp, on the host. Zeros rather than clone_grads: p.grad is clear
        # here and this must not alias it.
        g_acc = [torch.zeros_like(p.data, device="cpu") for p in params]
        hvp_metrics = {}
        all_c, all_groups, reasoning_total = [], 0, 0.0

        for s in range(len(records) - 1, -1, -1):
            record = records[s]
            self._restore_follower_step_weights(s)
            if group_aligned:
                groups = self._replay_group_blocks(record)
            else:
                # One block per unit, the partition the pre-grouping path used. Its block
                # count is agreed across the DP group by rearrange_micro_batches
                # (same_micro_num_in_dp), so the forward count and the per-unit dot()
                # all-reduce stay in lockstep. Eq. (49)'s per-query pairing is not
                # recovered here -- see _replay_group_blocks for what is.
                groups = [[b] for b in self._replay_blocks(record)]
            kept = list(range(0, len(groups), stride))
            n_groups = len(groups)       # an int, so `del groups` still frees the blocks
            rescale = n_groups / max(len(kept), 1)

            # --- sweep 1: c_b = eta_s <ghat_grp,s,b , v_{s+1}>, one scalar per GROUP ----
            # p.grad holds -ghat_grp once the group's blocks have all backwarded, and
            # v_buf holds -v_{s+1}, so the two sign flips cancel.
            coeffs, reasoning_tokens = [], 0.0
            for g_idx in kept:
                self.actor_optimizer.zero_grad(set_to_none=False)
                for block in groups[g_idx]:
                    response_length = block["responses"].size(1)
                    _, log_prob = self._forward_micro_batch(
                        micro_batch=dict(block), temperature=temperature, calculate_entropy=False
                    )
                    loss = echo_response.follower_surrogate(
                        log_prob=log_prob,
                        advantages=block["advantages"],
                        tool_mask=block["low_level_loss_mask"][:, -response_length:],
                        loss_agg_mode=loss_agg_mode,
                    )
                    loss.backward()                      # accumulates across the group
                    reasoning_tokens += float(
                        torch.clamp(
                            block["high_level_loss_mask"][:, -response_length:]
                            + block["low_level_loss_mask"][:, -response_length:],
                            max=1.0,
                        ).sum()
                    )
                coeffs.append(
                    etas[s] * echo_response.dot(echo_response.grads(params), v_buf, group)
                )

            # Baseline, per step rather than pooled across steps: with the sweep reversed,
            # the earlier steps' coefficients do not exist yet. E[grad log p] = 0 keeps the
            # estimator unbiased under any constant, so this only shifts variance.
            c_mean = float(sum(coeffs) / len(coeffs)) if coeffs else 0.0
            normalizer = max(reasoning_tokens, 1.0)

            if curvature:
                # Between the sweeps, so it still sees v_buf = v_{s+1}: Eq. (54) contracts
                # step s's B_s against v_{s+1}, not against the v_s built below. Banked
                # into g_acc and deliberately NOT propagated into the adjoint -- this is
                # the grad^fix half, the one piece the bf16 finite difference cannot
                # resolve, and feeding it backwards would spread that error over every
                # earlier step. A_s keeps only the score half of Hhat_yy either way.
                self.actor_optimizer.zero_grad(set_to_none=False)
                restore_fn = (lambda idx=s: self._restore_follower_step_weights(idx))
                hvp_metrics = self._hessian_vector_product(
                    record, temperature, v_buf, etas[s], restore_fn,
                    use_aepo_clip, use_sign_cond_clip, fd_rel,
                )
                for acc, p in zip(g_acc, params):
                    if p.grad is not None:
                        acc.add_(p.grad.detach().to("cpu"))

            # --- sweep 2: u_s = sum_b (c_b - c_mean) S_b, into p.grad ------------------
            self.actor_optimizer.zero_grad(set_to_none=False)
            for cursor, g_idx in enumerate(kept):
                scale = rescale * (coeffs[cursor] - c_mean) / normalizer
                for block in groups[g_idx]:
                    response_length = block["responses"].size(1)
                    _, log_prob = self._forward_micro_batch(
                        micro_batch=dict(block), temperature=temperature, calculate_entropy=False
                    )
                    policy_mask = torch.clamp(
                        block["high_level_loss_mask"][:, -response_length:]
                        + block["low_level_loss_mask"][:, -response_length:],
                        max=1.0,
                    )
                    score = echo_response.reasoning_score(log_prob, policy_mask)
                    (-scale * score).backward()
            del groups

            # p.grad now holds this step's -u_s. It lands in BOTH the running response
            # term and the adjoint, which is the S_x = S_y collapse above.
            for acc, p in zip(g_acc, params):
                delta = p.grad if p.grad is not None else torch.zeros_like(p.data)
                acc.add_(delta.detach().to("cpu", non_blocking=False))

            # The adjoint does NOT inherit the token normalizer. g_resp's overall scale is
            # a free choice here -- `coef` is ignored on the exact path, so dividing by a
            # global token count is just magnitude management. The recursion's scale is
            # not free: Eq. (52) fixes A_s = I + eta_L Hhat_yy,s, and Eq. (49) fixes
            # Hhat = (1/B_L) sum_b ghat_b S_b^T with S the SUM-score of Eq. (48). p.grad
            # carries rescale/normalizer instead of 1/n_groups, so undo the difference.
            # Left alone, the adjoint update would arrive ~1e5 times too small and the
            # reverse sweep would be decorative.
            adjoint_alpha = normalizer / max(n_groups, 1)
            for v, p in zip(v_buf, params):
                if p.grad is not None:
                    v.add_(p.grad.detach(), alpha=adjoint_alpha)

            all_c.extend(coeffs)
            all_groups += len(kept)
            reasoning_total += reasoning_tokens

        # Hand the accumulated term back as p.grad; the leader's own gradient lands on top.
        self.actor_optimizer.zero_grad(set_to_none=False)
        for p, acc in zip(params, g_acc):
            if p.grad is None:
                p.grad = torch.zeros_like(p.data)
            p.grad.copy_(acc.to(p.grad.device))
        del g_acc

        echo_response.copy_grads_into_(v_buf, params)

        c_tensor = torch.tensor(all_c, dtype=torch.float64)
        metrics = dict(hvp_metrics)
        metrics.update({
            "actor/response_curvature": 1.0 if curvature else 0.0,
            "actor/response_group_aligned": 1.0 if group_aligned else 0.0,
            "actor/response_steps_K": float(len(records)),
            "actor/response_groups": float(all_groups),
            "actor/response_c_mean": float(c_tensor.mean()) if all_c else 0.0,
            "actor/response_c_std": float(c_tensor.std()) if len(all_c) > 1 else 0.0,
            "actor/response_c_absmean": float(c_tensor.abs().mean()) if all_c else 0.0,
            "actor/response_reasoning_tokens": reasoning_total,
            "actor/follower_eta_mean": float(sum(etas) / max(len(etas), 1)),
            "actor/follower_clip_active": 1.0 if any(clip_flags) else 0.0,
        })
        return metrics, v_buf

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

        n_resp_sq = echo_response.dot(response_grad, response_grad, group)
        n_tot_sq = echo_response.dot(total, total, group)
        cross = echo_response.dot(response_grad, total, group)

        n_dir_sq = max(n_tot_sq - 2.0 * cross + n_resp_sq, 0.0)
        n_resp, n_dir = n_resp_sq**0.5, n_dir_sq**0.5
        cos = (cross - n_resp_sq) / (n_resp * n_dir) if n_resp > 0 and n_dir > 0 else 0.0
        return {
            "actor/response_norm": n_resp,
            "actor/direct_norm": n_dir,
            "actor/response_to_direct_ratio": n_resp / n_dir if n_dir > 0 else 0.0,
            "actor/response_direct_cosine": max(-1.0, min(1.0, cos)),
        }

    # --- DataParallelPhaseActor hooks --------------------------------------------

    def _update_begin(self, data, phase):
        """ECHO Algorithm 1. The follower phase stashes its rollouts (line 8); the leader
        phase adds the response term and applies the result to x (lines 12-14)."""
        m = data.meta_info
        response_enabled = bool(m.get("response_enabled", False))
        self._resp = {
            "enabled": response_enabled,
            "coef": float(m.get("response_coef", 1.0)),
            "replay_fraction": float(m.get("response_replay_fraction", 1.0)),
            # The exact K=1 path. Both halves of Eq. (35), the optimizer Jacobian taken
            # properly, and no free scalar: response_coef is not read on this path.
            "exact": bool(m.get("response_exact", False)),
            "fd_rel": float(m.get("response_fd_rel", 2e-2)),
            "curvature": bool(m.get("response_curvature", False)),
            "group_aligned": bool(m.get("response_group_aligned", False)),
            "stash_record": response_enabled and phase == "low_level",
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
        return [k for k in ("high_level_loss_mask", "low_level_loss_mask", "scalar_advantages")
                if k in data.batch.keys()]

    def _on_batch_selected(self, selected, data, phase):
        if not self._resp.get("stash_record"):
            return
        self._stash_follower_record(selected, uids=data.non_tensor_batch.get("uid"))
        if self._resp["exact"]:
            # Before the step below moves w_s -> w_{s+1}.
            self._push_follower_step_weights()

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
        # Stage 1 and the direct gradient are BOTH evaluated at the adapted point
        # w_1: v = grad_y U_H(x, y_1) and grad_x U_H(x, y_1) are what Algorithm 1
        # line 12 asks for. Only the response term moves back to w_0.
        metrics.update(
            self._leader_follower_direction(
                dataloader[0], temperature, mini_batch_size, micro_batch_size_per_gpu,
                use_dataproto_batches, use_aepo_clip, use_sign_cond_clip,
            )
        )
        if r["exact"]:
            follower_optim = self.phase_optims["low_level"][0]
            assert isinstance(follower_optim, torch.optim.SGD), (
                "the exact response path needs the follower on SGD so that "
                "D_s = eta_L * I. Under AdamW the first step from reset moments is "
                "sign-like and its exact Jacobian is ~4e-12, which zeroes the whole "
                "term. Set phases.low_level.optim.optimizer=sgd."
            )
            v_buf = echo_response.clone_grads(
                echo_response.trainable_params(self.actor_module)
            )
            # w_K has to come back for the direct gradient once the response term has
            # walked back through w_0 ... w_{K-1}.
            assert self._snapshot_adapted_weights is not None and self._restore_adapted_weights is not None, (
                "phases.response.exact needs the worker's adapted-weight hooks."
            )
            self._snapshot_adapted_weights()
            response_metrics, response_grad = self._exact_response_gradient(
                temperature, v_buf, r["replay_fraction"],
                use_aepo_clip, use_sign_cond_clip, r["fd_rel"],
                grad_clip=self.config.grad_clip,
                follower_lr=float(follower_optim.param_groups[0]["lr"]),
                curvature=r["curvature"],
                group_aligned=r["group_aligned"],
            )
            self._restore_adapted_weights()
        else:
            response_metrics, response_grad = self._response_gradient(
                temperature, r["coef"], r["replay_fraction"]
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
            "DataParallelECHOActor was built without restore_leader_weights."
        )
        self._restore_leader_weights()
        metrics["actor/follower_discarded"] = 1.0

    def _after_optimizer_step(self, phase, grad_norm):
        if not self._resp.get("stash_record"):
            return
        # D_s depends on whether clip_grad_norm_ rescaled step s. Appended, so
        # it stays index-aligned with _follower_records / _follower_step_weights.
        gn = grad_norm.detach()
        self._follower_grad_norms.append(
            float(gn.item()) if torch.isfinite(gn) else 0.0
        )

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
