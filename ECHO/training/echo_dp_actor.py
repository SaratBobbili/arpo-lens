import logging
import os

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

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelECHOActor(DataParallelPPOActor):
    def __init__(self, config, actor_module, phase_optims, phase_batch_sizes):
        super().__init__(config, actor_module, actor_optimizer=None)
        self.phase_optims = phase_optims
        self.phase_batch_sizes = phase_batch_sizes

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

        if opefo_enabled:
            assert not self.use_fused_kernels, "OPEFO requires use_fused_kernels=false"

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn or "loss_mask" in data.batch.keys():
            select_keys.append("loss_mask")
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

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})
        self.actor_optimizer.zero_grad()
        return metrics
