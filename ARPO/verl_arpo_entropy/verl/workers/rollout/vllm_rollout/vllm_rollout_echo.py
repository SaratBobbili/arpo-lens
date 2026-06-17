# Copyright 2026 ECHO contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import concurrent.futures
import importlib
import logging
import os
import time
from copy import deepcopy
from typing import Counter, Dict, List

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.rollout.tools.base_tool import BaseTool
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout, _pre_process_inputs, _repeat_interleave

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Maps each tag string to (block_type, "open"|"close").
_TAG_INFO = {
    "<select>": ("select", "open"),
    "</select>": ("select", "close"),
    "<think>": ("think", "open"),
    "</think>": ("think", "close"),
    "<answer>": ("answer", "open"),
    "</answer>": ("answer", "close"),
    "<search>": ("search", "open"),
    "</search>": ("search", "close"),
    "<python>": ("python", "open"),
    "</python>": ("python", "close"),
    "<result>": ("result", "open"),
    "</result>": ("result", "close"),
}
# Close tags first so `</tag><tag>` boundaries resolve correctly.
_TAG_MATCH_ORDER = tuple(t for t in _TAG_INFO if t.startswith("</")) + tuple(t for t in _TAG_INFO if not t.startswith("</"))

_VALID_MASK_LEVELS = {"high", "low", "both", "none"}
_DEFAULT_MASK_CATEGORIES = {
    "first_select": "high",
    "select": "high",
    "think": "high",
    "answer": "high",
    "search": "low",
    "python": "low",
}


def _load_tool_from_config(tool_config: DictConfig) -> BaseTool:
    """Dynamically loads a tool from its configuration."""
    module_path, class_name = tool_config.class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        tool_class = getattr(module, class_name)
        tool_params = OmegaConf.to_container(tool_config.get("params", {}), resolve=True)
        tool_instance = tool_class(**tool_params)
        return tool_instance
    except ImportError as e:
        logger.error(f"Failed to import module {module_path}: {e}")
        raise
    except AttributeError as e:
        logger.error(f"Failed to find class {class_name} in module {module_path}: {e}")
        raise
    except TypeError as e:
        logger.error(f"Failed to instantiate {class_name} with provided parameters: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading tool from {tool_config.class_path}: {e}")
        raise


class vLLMRolloutECHO(vLLMRollout):
    """
    Tool-enabled rollout used by ECHO.

    This implementation keeps the behavior of `vLLMRolloutWithTools` but removes
    adaptive branching and entropy-based branch control.
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        super().__init__(model_path, config, tokenizer, model_hf_config, **kwargs)
        self.tokenizer = tokenizer

        tools_config = self.config.get("tools", OmegaConf.create({}))
        self.tool_call_limit = tools_config.get("call_limit", 5)
        self.max_tool_workers = tools_config.get("max_workers", 64)
        self.tool_timeout = tools_config.get("timeout", 120)
        self.tool_retry_count = tools_config.get("retry_count", 3)
        self.tool_verbose_logging = tools_config.get("verbose_logging", False)
        self.skip_training_on_tool_failure = bool(tools_config.get("skip_training_on_tool_failure", False))
        self.exclude_tag_tokens_from_phase_masks = bool(
            self.config.get("exclude_tag_tokens_from_phase_masks", True)
        )

        mask_cat_cfg = OmegaConf.to_container(self.config.get("mask_categories", OmegaConf.create({})), resolve=True)
        self.mask_categories = {k: mask_cat_cfg.get(k, v) for k, v in _DEFAULT_MASK_CATEGORIES.items()}
        for cat, level in self.mask_categories.items():
            assert level in _VALID_MASK_LEVELS, f"mask_categories.{cat}={level!r}, must be one of {_VALID_MASK_LEVELS}"
        logger.info(f"ECHO mask_categories: {self.mask_categories}")

        self.tools: Dict[str, BaseTool] = {}
        if "tool_instances" in tools_config:
            for tool_name, tool_config in tools_config.tool_instances.items():
                logger.info(f"Loading tool '{tool_name}' from {tool_config.class_path}")
                try:
                    tool_instance = _load_tool_from_config(tool_config)
                    self.tools[tool_instance.trigger_tag] = tool_instance
                except Exception as e:
                    logger.error(f"Could not initialize tool '{tool_name}'. Please check your configuration. Error: {e}")
                    if tools_config.get("fail_on_error", False):
                        raise

        self.stop_sequences = [f"</{tag}>" for tag in self.tools.keys()]
        self.logprobs = 0

        if not self.tools:
            logger.warning("vLLMRolloutECHO initialized, but no tools were configured.")

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_tool_workers)

    def __del__(self):
        self.executor.shutdown(wait=False)

    def _compute_hierarchical_masks(self, output_ids: List[int], result_mask: List[int]) -> tuple[List[int], List[int], List[int], List[int], List[int], List[int], int]:
        token_texts = [
            self.tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            for token_id in output_ids
        ]
        response_text = "".join(token_texts)
        text_length = len(response_text)

        # Walk character-by-character, classifying each into a category.
        char_categories: List[str | None] = [None] * text_length
        # Per-character flag set on every char that belongs to an open or close
        # tag string (e.g. "<select>", "</python>"). Token-level collapse below
        # converts this to `non_border_mask` consumed by the entropy strategies
        # in the trainer to exclude tag boundary tokens from the entropy reduction.
        char_border: List[int] = [0] * text_length
        current_block: str | None = None
        select_count = 0
        char_idx = 0
        first_select_end_char: int | None = None

        while char_idx < text_length:
            matched_tag = None
            for tag in _TAG_MATCH_ORDER:
                if response_text.startswith(tag, char_idx):
                    matched_tag = tag
                    break

            if matched_tag is not None:
                block_type, direction = _TAG_INFO[matched_tag]
                tag_end = min(text_length, char_idx + len(matched_tag))

                # Open tags: enter the new block before labelling tag chars.
                if direction == "open":
                    if block_type == "select":
                        select_count += 1
                        current_block = "first_select" if select_count == 1 else "select"
                    else:
                        current_block = block_type

                for span_idx in range(char_idx, tag_end):
                    char_categories[span_idx] = current_block
                    char_border[span_idx] = 1

                # Close tags: leave the block after labelling tag chars.
                if direction == "close":
                    if block_type == "select" and select_count == 1 and first_select_end_char is None:
                        first_select_end_char = tag_end
                    current_block = None

                char_idx = tag_end
                continue

            char_categories[char_idx] = current_block
            char_idx += 1

        # Convert per-character categories to high/low masks using config; also
        # emit a separate char_select mask that marks every char inside any
        # <select>...</select> block (both first_select and subsequent select).
        # Consumed by the "entropy-hybrid" reward strategy after being AND-ed
        # with the phase mask in the trainer, so whether first_select is part
        # of the computation is implicitly controlled by mask_categories.
        mask_cfg = self.mask_categories
        char_high = [0] * text_length
        char_low = [0] * text_length
        char_select = [0] * text_length
        char_first_select = [0] * text_length
        for i, cat in enumerate(char_categories):
            if cat is None or cat == "result":
                continue
            if cat == "first_select":
                char_first_select[i] = 1
            if cat in ("first_select", "select"):
                char_select[i] = 1
            level = mask_cfg.get(cat, "none")
            if level == "high":
                char_high[i] = 1
            elif level == "low":
                char_low[i] = 1
            elif level == "both":
                char_high[i] = 1
                char_low[i] = 1

        # Collapse character masks to token masks, intersecting with result_mask.
        high_level_mask: List[int] = []
        low_level_mask: List[int] = []
        select_mask: List[int] = []
        first_select_mask: List[int] = []
        cursor = 0
        for token_text, keep_token in zip(token_texts, result_mask):
            next_cursor = cursor + len(token_text)
            token_high = int(any(char_high[cursor:next_cursor])) if next_cursor > cursor else 0
            token_low = int(any(char_low[cursor:next_cursor])) if next_cursor > cursor else 0
            token_select = int(any(char_select[cursor:next_cursor])) if next_cursor > cursor else 0
            token_first_select = int(any(char_first_select[cursor:next_cursor])) if next_cursor > cursor else 0
            token_border = int(any(char_border[cursor:next_cursor])) if next_cursor > cursor else 0
            phase_active_token = int(
                bool(keep_token)
                and (
                    (not token_border)
                    if self.exclude_tag_tokens_from_phase_masks
                    else True
                )
            )
            high_level_mask.append(int(phase_active_token and token_high))
            low_level_mask.append(int(phase_active_token and token_low))
            select_mask.append(int(phase_active_token and token_select))
            first_select_mask.append(int(phase_active_token and token_first_select))
            cursor = next_cursor

        first_select_post_idx = -1
        if first_select_end_char is not None:
            cursor = 0
            for token_i, keep_token in enumerate(result_mask):
                token_len = len(token_texts[token_i])
                if keep_token and token_len > 0 and cursor >= first_select_end_char:
                    first_select_post_idx = token_i
                    break
                cursor += token_len

        return high_level_mask, low_level_mask, select_mask, first_select_mask, first_select_post_idx

    def _extract_content(self, text: str, tag: str) -> str:
        """Extracts content from within the last <tag>...</tag> block."""
        try:
            start_tag = f"<{tag}>"
            end_tag = f"</{tag}>"
            end_pos = text.rindex(end_tag)
            start_pos = text.rindex(start_tag, 0, end_pos)
            return text[start_pos + len(start_tag) : end_pos].strip()
        except ValueError:
            logger.warning(f"Could not extract content for tag '{tag}' from text: {text}")
            return ""

    def _execute_tool_with_retry(self, tool, content):
        retry_count = 0
        start_time = time.time()

        while retry_count < self.tool_retry_count:
            try:
                result_text = tool.execute(content)
                if result_text:
                    execution_time = time.time() - start_time
                    return {
                        "success": True,
                        "retry_count": retry_count,
                        "execution_time": execution_time,
                        "result": result_text,
                    }
                logger.warning(f"Tool({tool.trigger_tag}) returned empty output. Retrying {retry_count + 1}/{self.tool_retry_count}")
                retry_count += 1
            except Exception as e:
                logger.error(f"Tool({tool.trigger_tag}) execution failed. Retrying {retry_count + 1}/{self.tool_retry_count}: {e}")
                retry_count += 1

        execution_time = time.time() - start_time
        logger.warning(f"Tool({tool.trigger_tag}) execution failed after {self.tool_retry_count} retries. Appending EOS.")
        return {
            "success": False,
            "retry_count": retry_count,
            "execution_time": execution_time,
            "result": "",
        }

    @GPUMemoryLogger(role="vllm rollout echo", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        if vllm_version in ("0.5.4", "0.6.3") and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        input_ids = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = self.tokenizer.eos_token_id
        batch_size = input_ids.size(0)

        tool_metrics = {
            "tools/total_calls": 0,
            "tools/successful_calls": 0,
            "tools/failed_calls": 0,
            "tools/total_execution_time": 0.0,
            "tools/avg_execution_time": 0.0,
            "tools/max_execution_time": 0.0,
            "tools/max_retries": 0,
            "tools/total_retries": 0,
            "tools/call_limit_reached_count": 0,
        }

        calls_per_tool = Counter()
        success_per_tool = Counter()
        total_time_per_tool = Counter()

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        rollout_n_override = prompts.meta_info.get("rollout_n_override", None)

        if not do_sample:
            kwargs.update(
                {
                    "best_of": 1,
                    "top_p": 1.0,
                    "top_k": -1,
                    "min_p": 0.0,
                    "temperature": 0,
                    "n": 1,
                }
            )
        elif is_validate:
            kwargs.update(
                {
                    "top_k": self.config.val_kwargs.top_k,
                    "top_p": self.config.val_kwargs.top_p,
                    "temperature": self.config.val_kwargs.temperature,
                    "n": 1,
                }
            )
        elif rollout_n_override is not None:
            kwargs["n"] = int(rollout_n_override)

        kwargs["allowed_token_ids"] = list(self.tokenizer.get_vocab().values())

        with self.update_sampling_params(**kwargs):
            num_samples = self.sampling_params.n
            prompt_token_ids_list = [_pre_process_inputs(self.pad_token_id, prompt) for prompt in input_ids]

            curr_inputs = []
            init_inputs = []
            result_masks = []
            call_counters = []
            active_indices = []

            for ids in prompt_token_ids_list:
                for _ in range(num_samples):
                    curr_inputs.append(ids.copy())
                    init_inputs.append(ids.copy())
                    result_masks.append([])
                    call_counters.append(0)
                    active_indices.append(len(curr_inputs) - 1)

            sample_to_indices = {
                i: [i * num_samples + j for j in range(num_samples)] for i in range(batch_size)
            }
            rollout_tool_failed = [False] * len(curr_inputs) if self.skip_training_on_tool_failure else None

            max_len = self.config.response_length

            while active_indices:
                active_prompts = [curr_inputs[i] for i in active_indices]

                with self.update_sampling_params(
                    n=1,
                    stop=self.stop_sequences,
                    max_tokens=max(1, max((max_len - (len(curr_inputs[i]) - len(init_inputs[i])) for i in active_indices))),
                    detokenize=True,
                    logprobs=self.logprobs,
                ):
                    outputs = self.inference_engine.generate(
                        prompt_token_ids=active_prompts,
                        sampling_params=self.sampling_params,
                        use_tqdm=False,
                    )

                tool_requests: Dict[str, List[Dict]] = {tag: [] for tag in self.tools}
                next_active_indices = []

                for i, out_idx in enumerate(active_indices):
                    output = outputs[i]
                    generated_tokens = output.outputs[0].token_ids

                    curr_inputs[out_idx].extend(generated_tokens)
                    result_masks[out_idx].extend([1] * len(generated_tokens))

                    finish_reason = output.outputs[0].finish_reason
                    stop_reason = output.outputs[0].stop_reason
                    is_tool_call = finish_reason == "stop" and stop_reason in self.stop_sequences

                    if is_tool_call:
                        tag = stop_reason.strip("</>")
                        if call_counters[out_idx] < self.tool_call_limit:
                            call_counters[out_idx] += 1
                            full_text = self.tokenizer.decode(curr_inputs[out_idx])
                            content = self._extract_content(full_text, tag)
                            if content:
                                tool_requests[tag].append({"index": out_idx, "content": content})
                                next_active_indices.append(out_idx)
                                tool_metrics["tools/total_calls"] += 1
                                calls_per_tool[tag] += 1
                        else:
                            logger.warning(f"Tool call limit reached for sample {out_idx}. Appending EOS.")
                            curr_inputs[out_idx].append(eos_token_id)
                            result_masks[out_idx].append(1)
                            tool_metrics["tools/call_limit_reached_count"] += 1

                    elif finish_reason == "length":
                        if len(curr_inputs[out_idx]) - len(init_inputs[out_idx]) < max_len:
                            next_active_indices.append(out_idx)

                if any(tool_requests.values()):
                    futures = {}
                    for tag, requests in tool_requests.items():
                        if not requests:
                            continue
                        tool = self.tools[tag]
                        for req in requests:
                            future = self.executor.submit(self._execute_tool_with_retry, tool, req["content"])
                            futures[future] = {"index": req["index"], "tag": tag}

                    for future in concurrent.futures.as_completed(futures):
                        fut_info = futures[future]
                        idx = fut_info["index"]
                        tag = fut_info["tag"]

                        try:
                            result = future.result(timeout=self.tool_timeout)
                            success = result["success"]
                            retry_count = result["retry_count"]
                            execution_time = result["execution_time"]
                            result_text = result["result"]

                            if success:
                                tool_metrics["tools/successful_calls"] += 1
                                success_per_tool[tag] += 1
                            else:
                                tool_metrics["tools/failed_calls"] += 1
                                if rollout_tool_failed is not None:
                                    rollout_tool_failed[idx] = True
                                result_text = f"Tool({tag}) returned empty output."

                            tool_metrics["tools/total_execution_time"] += execution_time
                            tool_metrics["tools/max_execution_time"] = max(
                                tool_metrics["tools/max_execution_time"], execution_time
                            )
                            tool_metrics["tools/total_retries"] += retry_count
                            tool_metrics["tools/max_retries"] = max(tool_metrics["tools/max_retries"], retry_count)
                            total_time_per_tool[tag] += execution_time

                            if not result_text:
                                result_text = f"Tool({tag}) returned empty output."

                        except Exception as e:
                            logger.error(f"Tool({tag}) execution failed for sample {idx}: {e}")
                            result_text = f"Error: Tool({tag}) execution failed with message: {e}"
                            tool_metrics["tools/failed_calls"] += 1
                            if rollout_tool_failed is not None:
                                rollout_tool_failed[idx] = True

                        formatted_result = f" <result>\n{result_text}\n</result>"
                        result_tokens = self.tokenizer.encode(formatted_result)
                        curr_inputs[idx].extend(result_tokens)
                        result_masks[idx].extend([0] * len(result_tokens))

                final_active_indices = []
                for idx in next_active_indices:
                    response_len = len(curr_inputs[idx]) - len(init_inputs[idx])
                    if response_len < max_len:
                        final_active_indices.append(idx)

                # No adaptive branching: keep only current active rollouts.
                active_indices = final_active_indices

            for idx in range(len(curr_inputs)):
                response_len = len(curr_inputs[idx]) - len(init_inputs[idx])
                if response_len > max_len:
                    offset = len(init_inputs[idx])
                    curr_inputs[idx] = curr_inputs[idx][: offset + max_len]
                    result_masks[idx] = result_masks[idx][:max_len]

            output_sequences = []
            output_result_masks = []
            output_high_level_masks = []
            output_low_level_masks = []
            output_select_masks = []
            output_first_select_masks = []
            output_first_select_post_idxs = []
            for i in range(batch_size):
                sample_indices = sample_to_indices.get(i, [])
                for idx in sample_indices:
                    output_ids = curr_inputs[idx][len(prompt_token_ids_list[i]) :]
                    output_mask = result_masks[idx]
                    (
                        high_level_mask,
                        low_level_mask,
                        select_mask,
                        first_select_mask,
                        first_select_post_idx,
                    ) = self._compute_hierarchical_masks(
                        output_ids=output_ids, result_mask=output_mask
                    )
                    output_sequences.append(output_ids)
                    output_result_masks.append(output_mask)
                    output_high_level_masks.append(high_level_mask)
                    output_low_level_masks.append(low_level_mask)
                    output_select_masks.append(select_mask)
                    output_first_select_masks.append(first_select_mask)
                    output_first_select_post_idxs.append(first_select_post_idx)

            padded_response_list = []
            padded_result_mask_list = []
            padded_high_level_mask_list = []
            padded_low_level_mask_list = []
            padded_select_mask_list = []
            padded_first_select_mask_list = []
            for output_ids, result_mask, high_level_mask, low_level_mask, select_mask, first_select_mask in zip(
                output_sequences,
                output_result_masks,
                output_high_level_masks,
                output_low_level_masks,
                output_select_masks,
                output_first_select_masks,
            ):
                assert len(output_ids) == len(result_mask), f"output_ids: {len(output_ids)}, result_mask: {len(result_mask)}"
                assert len(output_ids) == len(high_level_mask), f"output_ids: {len(output_ids)}, high_level_mask: {len(high_level_mask)}"
                assert len(output_ids) == len(low_level_mask), f"output_ids: {len(output_ids)}, low_level_mask: {len(low_level_mask)}"
                assert len(output_ids) == len(select_mask), f"output_ids: {len(output_ids)}, select_mask: {len(select_mask)}"
                assert len(output_ids) == len(first_select_mask), f"output_ids: {len(output_ids)}, first_select_mask: {len(first_select_mask)}"

                response = torch.tensor(output_ids)
                response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)

                result_mask_tensor = torch.tensor(result_mask)
                result_mask_tensor = pad_sequence_to_length(result_mask_tensor, self.config.response_length, 0)
                high_level_mask_tensor = torch.tensor(high_level_mask)
                high_level_mask_tensor = pad_sequence_to_length(high_level_mask_tensor, self.config.response_length, 0)
                low_level_mask_tensor = torch.tensor(low_level_mask)
                low_level_mask_tensor = pad_sequence_to_length(low_level_mask_tensor, self.config.response_length, 0)
                select_mask_tensor = torch.tensor(select_mask)
                select_mask_tensor = pad_sequence_to_length(select_mask_tensor, self.config.response_length, 0)
                first_select_mask_tensor = torch.tensor(first_select_mask)
                first_select_mask_tensor = pad_sequence_to_length(first_select_mask_tensor, self.config.response_length, 0)

                padded_response_list.append(response)
                padded_result_mask_list.append(result_mask_tensor)
                padded_high_level_mask_list.append(high_level_mask_tensor)
                padded_low_level_mask_list.append(low_level_mask_tensor)
                padded_select_mask_list.append(select_mask_tensor)
                padded_first_select_mask_list.append(first_select_mask_tensor)

            response = torch.stack(padded_response_list, dim=0).to(input_ids.device)
            loss_mask = torch.stack(padded_result_mask_list, dim=0).to(input_ids.device)
            high_level_loss_mask = torch.stack(padded_high_level_mask_list, dim=0).to(input_ids.device)
            low_level_loss_mask = torch.stack(padded_low_level_mask_list, dim=0).to(input_ids.device)
            select_loss_mask = torch.stack(padded_select_mask_list, dim=0).to(input_ids.device)
            first_select_loss_mask = torch.stack(padded_first_select_mask_list, dim=0).to(input_ids.device)
            first_select_post_idx = torch.tensor(
                output_first_select_post_idxs, dtype=torch.long, device=input_ids.device
            )

            non_tensor_batch = deepcopy(prompts.non_tensor_batch)
            if num_samples > 1 and do_sample:
                input_ids = _repeat_interleave(input_ids, num_samples)
                attention_mask = _repeat_interleave(attention_mask, num_samples)
                position_ids = _repeat_interleave(position_ids, num_samples)
                if non_tensor_batch:
                    for key, value in non_tensor_batch.items():
                        if isinstance(value, np.ndarray):
                            non_tensor_batch[key] = np.repeat(value, num_samples, axis=0)
                        elif isinstance(value, list):
                            non_tensor_batch[key] = [item for item in value for _ in range(num_samples)]

            if rollout_tool_failed is not None:
                non_tensor_batch["tool_rollout_failed"] = np.array(
                    [rollout_tool_failed[idx] for i in range(batch_size) for idx in sample_to_indices[i]],
                    dtype=np.bool_,
                )

            final_batch_size = input_ids.size(0)
            seq = torch.cat([input_ids, response], dim=-1)

            response_length = response.size(1)
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device).unsqueeze(0).expand(
                final_batch_size, -1
            )

            if position_ids.dim() == 3:
                delta_position_id = delta_position_id.view(final_batch_size, 1, -1).expand(
                    final_batch_size, position_ids.size(1), -1
                )
                response_position_ids = position_ids[..., -1:].expand(-1, position_ids.size(1), -1) + delta_position_id
            else:
                response_position_ids = position_ids[..., -1:] + delta_position_id

            final_position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

            response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
            final_attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
            loss_mask = loss_mask * response_attention_mask
            high_level_loss_mask = high_level_loss_mask * response_attention_mask
            low_level_loss_mask = low_level_loss_mask * response_attention_mask
            select_loss_mask = select_loss_mask * response_attention_mask
            first_select_loss_mask = first_select_loss_mask * response_attention_mask

            if tool_metrics["tools/total_calls"] > 0:
                tool_metrics["tools/avg_execution_time"] = tool_metrics["tools/total_execution_time"] / tool_metrics["tools/total_calls"]

            tool_specific_metrics = {}
            for tag in self.tools.keys():
                calls = calls_per_tool[tag]
                if calls > 0:
                    tool_specific_metrics[f"tools/{tag}/calls"] = calls
                    tool_specific_metrics[f"tools/{tag}/avg_time"] = total_time_per_tool[tag] / calls
                    tool_specific_metrics[f"tools/{tag}/success_rate"] = success_per_tool[tag] / calls
                else:
                    tool_specific_metrics[f"tools/{tag}/calls"] = 0
                    tool_specific_metrics[f"tools/{tag}/avg_time"] = 0
                    tool_specific_metrics[f"tools/{tag}/success_rate"] = 0

            batch = TensorDict(
                {
                    "prompts": input_ids,
                    "responses": response,
                    "input_ids": seq,
                    "attention_mask": final_attention_mask,
                    "loss_mask": loss_mask,
                    "high_level_loss_mask": high_level_loss_mask,
                    "low_level_loss_mask": low_level_loss_mask,
                    "select_loss_mask": select_loss_mask,
                    "first_select_loss_mask": first_select_loss_mask,
                    "first_select_post_idx": first_select_post_idx,
                    "position_ids": final_position_ids,
                },
                batch_size=final_batch_size,
            )

        if vllm_version in ("0.5.4", "0.6.3") and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        all_metrics = {**tool_metrics, **tool_specific_metrics}
        meta_info = deepcopy(prompts.meta_info) if prompts.meta_info else {}
        meta_info["metrics"] = all_metrics

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)
