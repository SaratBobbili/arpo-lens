import itertools
from contextlib import nullcontext

import torch

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.device import get_torch_device
from verl.workers.fsdp_workers import ActorRolloutRefWorker, logger


class EchoActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        is_lora = data.meta_info.pop("is_lora", False)
        calculate_entropy = data.meta_info.pop("calculate_entropy", True)
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        data = data.to(get_torch_device().current_device())
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            with adapter_ctx:
                log_probs, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=calculate_entropy)

            # dp_actor.compute_log_prob applies revert_indices to log_probs under
            # use_dynamic_bsz but NOT to entropys. ECHO uses per-token entropys as
            # sample-indexed rewards, so we recover the same indices here and revert
            # entropys locally. Only the rearrange step is duplicated, not the
            # micro-batch forward driver.
            if calculate_entropy and self.config.rollout.log_prob_use_dynamic_bsz:
                select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
                rearrange_batch = data.select(batch_keys=select_keys).batch
                max_token_len = self.config.rollout.log_prob_max_token_len_per_gpu * self.actor.ulysses_sequence_parallel_size
                _, indices = rearrange_micro_batches(batch=rearrange_batch, max_token_len=max_token_len)
                indices = list(itertools.chain.from_iterable(indices))
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long, device=entropys.device)
                entropys = entropys[revert_indices]

            tensor_payload = {"old_log_probs": log_probs}
            if calculate_entropy:
                tensor_payload["entropys"] = entropys
            output = DataProto.from_dict(
                tensors=tensor_payload,
                meta_info={"temperature": self.config.rollout.temperature},
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output
