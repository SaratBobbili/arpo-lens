import gc
import logging
import os
from collections import deque

import torch
import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_torch_device, is_cuda_available, is_npu_available
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor.dp_actor import DataParallelPPOActor

from .echo_core_algos import (
    agg_loss,
    compute_entropy_flow_E,
    compute_entropy_normalized,
    compute_opefo_policy_loss,
    compute_policy_loss,
    kl_penalty,
    resolve_advantage_signal,
)
from . import echo_response

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

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelECHOActor(DataParallelPPOActor):
    def __init__(self, config, actor_module, phase_optims, phase_batch_sizes,
                 restore_leader_weights=None, snapshot_adapted_weights=None,
                 restore_adapted_weights=None):
        super().__init__(config, actor_module, actor_optimizer=None)
        self.phase_optims = phase_optims
        self.phase_batch_sizes = phase_batch_sizes
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

    def _forward_micro_batch_opefo(self, micro_batch, temperature):
        """Forward like parent non-fused path; entropy-flow E on rmpad logits then pad (no full (B,T,V))."""
        assert not self.use_fused_kernels, "OPEFO requires use_fused_kernels=false"

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )

                logits_rmpad = output.logits.squeeze(0)
                logits_rmpad.div_(temperature)

                log_probs = logprobs_from_logits(
                    logits=logits_rmpad,
                    labels=input_ids_rmpad_rolled,
                    inplace_backward=False,
                )
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)

                with torch.no_grad():
                    flow_E_rmpad = compute_entropy_flow_E(logits_rmpad.detach())

                if self.use_ulysses_sp:
                    log_probs = gather_outpus_and_unpad(
                        log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )
                    entropy_rmpad = gather_outpus_and_unpad(
                        entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )
                    flow_E_rmpad = gather_outpus_and_unpad(
                        flow_E_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )

                entropy = pad_input(
                    hidden_states=entropy_rmpad.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                ).squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                ).squeeze(-1)[:, -response_length - 1 : -1]
                flow_E = pad_input(
                    hidden_states=flow_E_rmpad.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                ).squeeze(-1)[:, -response_length - 1 : -1]
            else:
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]
                log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                entropy = verl_F.entropy_from_logits(logits)
                with torch.no_grad():
                    flow_E = compute_entropy_flow_E(logits.detach())

            return entropy, log_probs, flow_E

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

    # --- response term (Eqs. 15-16) ---------------------------------------------------

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
                                 follower_lr, curvature):
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
            groups = self._replay_group_blocks(record)
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

    def update_policy(self, data: DataProto):
        phase = data.meta_info["phase"]
        # _optimizer_step() and zero_grad() below act on the phase's own AdamW.
        self.actor_optimizer, _ = self.phase_optims[phase]
        mini_batch_size, micro_batch_size_per_gpu = self.phase_batch_sizes[phase]
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        multi_turn = data.meta_info.get("multi_turn", False)
        advantage_algorithm = data.meta_info.get("advantage_algorithm", "grpo")
        entropy_normalization = data.meta_info.get("entropy_normalization", "token_pool")
        entropy_alpha = float(data.meta_info.get("entropy_alpha", 0.2))
        entropy_coeff_override = data.meta_info.get("entropy_coeff_override", None)
        entropy_loss_mask_key = data.meta_info.get("entropy_loss_mask_key", None)
        entropy_loss_normalizer = data.meta_info.get("entropy_loss_normalizer", None)
        kl_loss_coef_override = data.meta_info.get("kl_loss_coef_override", None)
        use_aepo_clip = bool(data.meta_info.get("use_aepo_clip_override", False))
        use_sign_cond_clip = bool(data.meta_info.get("use_sign_cond_clip_override", False))
        opefo_enabled = bool(data.meta_info.get("opefo_enabled", False))
        needs_entropy_norm = advantage_algorithm in ("entropy", "aepo")

        # ECHO Algorithm 1. The follower phase stashes its rollouts (line 8); the leader
        # phase adds the response term and applies the result to x (lines 12-14).
        response_enabled = bool(data.meta_info.get("response_enabled", False))
        response_coef = float(data.meta_info.get("response_coef", 1.0))
        replay_fraction = float(data.meta_info.get("response_replay_fraction", 1.0))
        # The exact K=1 path. Both halves of Eq. (35), the optimizer Jacobian taken
        # properly, and no free scalar: response_coef is not read on this path.
        response_exact = bool(data.meta_info.get("response_exact", False))
        response_fd_rel = float(data.meta_info.get("response_fd_rel", 2e-2))
        response_curvature = bool(data.meta_info.get("response_curvature", False))
        stash_record = response_enabled and phase == "low_level"
        do_response = response_enabled and phase == "high_level"
        # Algorithm 1 line 14 belongs to the ROUND STRUCTURE, not to the response
        # gradient: the adapted follower must be discarded and the leader step applied to
        # x_t whether or not g_resp was computed. Gating this on do_response would leave
        # y_K in the weights, so the next round's "common base" would not be common and
        # the leader step would land on w_core + x_t + y_K.
        do_discard_follower = bool(data.meta_info.get("discard_follower", False)) and phase == "high_level"

        if opefo_enabled:
            assert not self.use_fused_kernels, "OPEFO requires use_fused_kernels=false"

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn or "loss_mask" in data.batch.keys():
            select_keys.append("loss_mask")
        if response_enabled:
            # The response term reads both phase masks off the same batch: the tool mask
            # for g_fol and the Eq.-(47) surrogate, the reasoning mask for the Eq.-(48)
            # score. select_keys gates what reaches the micro-batches, so both must ride.
            for key in ("high_level_loss_mask", "low_level_loss_mask", "scalar_advantages"):
                if key in data.batch.keys() and key not in select_keys:
                    select_keys.append(key)
        if entropy_loss_mask_key is not None and entropy_loss_mask_key not in select_keys:
            select_keys.append(entropy_loss_mask_key)
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        # uid needed for group-level entropy normalization
        non_tensor_select_keys = []
        if needs_entropy_norm and entropy_normalization == "group":
            non_tensor_select_keys.append("uid")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")

        use_dataproto_batches = has_multi_modal_inputs or bool(non_tensor_select_keys)
        selected = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        batch = selected.batch
        if use_dataproto_batches:
            num_mini_batches = selected.batch.batch_size[0] // mini_batch_size
            dataloader = selected.chunk(num_mini_batches)
        else:
            dataloader = batch.split(mini_batch_size)

        metrics = {}

        if stash_record:
            self._stash_follower_record(selected, uids=data.non_tensor_batch.get("uid"))
            if response_exact:
                # Before the step below moves w_s -> w_{s+1}.
                self._push_follower_step_weights()

        # Stages 1-3 of the leader update. They leave p.grad = -g_resp, which the normal
        # loop below then accumulates -g_dir on top of, so the single optimizer step
        # applies g_dir + g_resp (Eq. 16). Requires exactly one optimizer step, which the
        # trainer asserts when the response is enabled.
        response_grad = None
        if do_response:
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
            if response_exact:
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
                    temperature, v_buf, replay_fraction,
                    use_aepo_clip, use_sign_cond_clip, response_fd_rel,
                    grad_clip=self.config.grad_clip,
                    follower_lr=float(follower_optim.param_groups[0]["lr"]),
                    curvature=response_curvature,
                )
                self._restore_adapted_weights()
            else:
                response_metrics, response_grad = self._response_gradient(
                    temperature, response_coef, replay_fraction
                )
            metrics.update(response_metrics)

        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                mini_batch = data
                if use_dataproto_batches:
                    mini_batch_seqs = mini_batch.batch.batch_size[0]
                    if self.config.use_dynamic_bsz:
                        max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                        _, micro_bsz_idx = rearrange_micro_batches(batch=mini_batch.batch, max_token_len=max_token_len)
                        micro_batches = [mini_batch.select_idxs(partition) for partition in micro_bsz_idx]
                    else:
                        self.gradient_accumulation = mini_batch_size // micro_batch_size_per_gpu
                        num_micro_batches = mini_batch_seqs // micro_batch_size_per_gpu
                        micro_batches = mini_batch.chunk(num_micro_batches)
                else:
                    mini_batch_seqs = mini_batch.batch_size[0]
                    if self.config.use_dynamic_bsz:
                        max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                        micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                    else:
                        self.gradient_accumulation = mini_batch_size // micro_batch_size_per_gpu
                        micro_batches = mini_batch.split(micro_batch_size_per_gpu)

                # Stage 4: accumulate g_dir on top of g_resp rather than clearing it.
                if response_grad is None:
                    self.actor_optimizer.zero_grad()

                for micro_idx, data in enumerate(micro_batches):
                    if isinstance(data, DataProto):
                        uid = data.non_tensor_batch.get("uid", None)
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        uid = None
                        data = data.to(get_torch_device().current_device())

                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn or "loss_mask" in data.keys():
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]
                    grpo_advantages = advantages.clone()

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    clip_ratio_low_pos = self.config.get("clip_ratio_low_pos", 0.2)
                    clip_ratio_high_pos = self.config.get("clip_ratio_high_pos", 0.2)
                    clip_ratio_low_neg = self.config.get("clip_ratio_low_neg", 0.2)
                    clip_ratio_high_neg = self.config.get("clip_ratio_high_neg", 0.2)
                    entropy_coeff = entropy_coeff_override if entropy_coeff_override is not None else self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if opefo_enabled:
                        entropy, log_prob, flow_E = self._forward_micro_batch_opefo(
                            micro_batch=data, temperature=temperature
                        )
                    else:
                        entropy, log_prob = self._forward_micro_batch(
                            micro_batch=data, temperature=temperature, calculate_entropy=True
                        )

                    if needs_entropy_norm:
                        entropy_norm = compute_entropy_normalized(
                            entropy=entropy,
                            response_mask=response_mask,
                            normalization=entropy_normalization,
                            index=uid,
                        )
                        advantages = resolve_advantage_signal(
                            advantage_algorithm, grpo_advantages, entropy_norm, entropy_alpha
                        )

                    if opefo_enabled:
                        delta_H = -advantages * flow_E
                        delta_H = delta_H * response_mask.to(dtype=delta_H.dtype)
                        pg_loss, opefo_diag = compute_opefo_policy_loss(
                            log_prob=log_prob,
                            advantages=advantages,
                            delta_H=delta_H,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        negative_approx_kl = log_prob - old_log_prob
                        ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
                        zero = torch.zeros((), device=pg_loss.device, dtype=pg_loss.dtype)
                        pg_clipfrac = zero
                        pg_clipfrac_lower = zero
                        append_to_dict(
                            metrics,
                            {
                                "actor/opefo_lambda": opefo_diag["opefo_lambda"].item(),
                                "actor/opefo_delta_H_net": opefo_diag["opefo_delta_H_net"].item(),
                                "actor/opefo_pos_mag": opefo_diag["opefo_pos_mag"].item(),
                                "actor/opefo_neg_mag": opefo_diag["opefo_neg_mag"].item(),
                                "actor/opefo_frac_pos": opefo_diag["opefo_frac_pos"].item(),
                                "actor/opefo_frac_neg": opefo_diag["opefo_frac_neg"].item(),
                                "actor/opefo_pg_loss": pg_loss.detach().item(),
                            },
                        )
                    else:
                        clip_sign_advantages = advantages if use_sign_cond_clip else None
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            use_aepo_clip=use_aepo_clip,
                            use_sign_cond_clip=use_sign_cond_clip,
                            clip_sign_advantages=clip_sign_advantages,
                            cliprange_low_pos=clip_ratio_low_pos,
                            cliprange_high_pos=clip_ratio_high_pos,
                            cliprange_low_neg=clip_ratio_low_neg,
                            cliprange_high_neg=clip_ratio_high_neg,
                            loss_agg_mode=loss_agg_mode,
                        )

                    if entropy_coeff != 0:
                        if entropy_loss_mask_key is not None:
                            entropy_loss_mask = data[entropy_loss_mask_key][:, -response_length:]
                        else:
                            entropy_loss_mask = response_mask
                        entropy_for_reg = entropy if entropy_loss_normalizer is None else entropy / entropy_loss_normalizer
                        entropy_loss = agg_loss(loss_mat=entropy_for_reg, loss_mask=entropy_loss_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                        append_to_dict(metrics, {
                            "actor/entropy_reg_loss": entropy_loss.detach().item(),
                            "actor/entropy_reg_coef": float(entropy_coeff),
                        })
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        kl_loss_coef = kl_loss_coef_override if kl_loss_coef_override is not None else self.config.kl_loss_coef
                        ref_log_prob = data["ref_log_prob"]
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = float(kl_loss_coef)

                    if self.config.use_dynamic_bsz:
                        loss = policy_loss * (responses.size(0) / mini_batch_seqs)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    append_to_dict(metrics, {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    })

                if response_grad is not None:
                    metrics.update(self._response_diagnostics(response_grad))
                    # Release the scratch buffer before the step, so its segment is free
                    # for the empty_cache() below to return to the driver.
                    response_grad.clear()
                    response_grad = None

                # Stage 5: discard y_K before stepping, so the gradient formed at
                # (x_t, y_K) lands on x_t (Algorithm 1 lines 13-14). Without the response
                # gradient this is exactly first-order MAML: adapt, evaluate the adapted
                # pair, apply the outer gradient to the PRE-adaptation parameters.
                if do_discard_follower:
                    assert self._restore_leader_weights is not None, (
                        "phases.response.enabled needs the worker's restore hook; "
                        "DataParallelECHOActor was built without restore_leader_weights."
                    )
                    self._restore_leader_weights()
                    metrics["actor/follower_discarded"] = 1.0

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})
                if stash_record:
                    # D_s depends on whether clip_grad_norm_ rescaled step s. Appended, so
                    # it stays index-aligned with _follower_records / _follower_step_weights.
                    gn = grad_norm.detach()
                    self._follower_grad_norms.append(
                        float(gn.item()) if torch.isfinite(gn) else 0.0
                    )
        self.actor_optimizer.zero_grad()

        if do_response:
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

        return metrics

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
