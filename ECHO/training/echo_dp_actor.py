import logging
import os

from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_torch_device
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.workers.actor.dp_actor import DataParallelPPOActor

from .echo_core_algos import agg_loss, compute_entropy_normalized, compute_policy_loss, kl_penalty, resolve_advantage_signal

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelECHOActor(DataParallelPPOActor):
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
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
        needs_entropy_norm = advantage_algorithm in ("entropy", "aepo")

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
            num_mini_batches = selected.batch.batch_size[0] // self.config.ppo_mini_batch_size
            dataloader = selected.chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                mini_batch = data
                if use_dataproto_batches:
                    if self.config.use_dynamic_bsz:
                        max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                        _, micro_bsz_idx = rearrange_micro_batches(batch=mini_batch.batch, max_token_len=max_token_len)
                        micro_batches = [mini_batch.select_idxs(partition) for partition in micro_bsz_idx]
                    else:
                        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                        num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                        micro_batches = mini_batch.chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

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

                    # Always compute fresh entropy from the current policy.
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=True)

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
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
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
