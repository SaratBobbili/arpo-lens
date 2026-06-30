import sys
import os
sys.path.append(os.getcwd())
import json
import asyncio
import datetime
import numpy as np
import re
from typing import List, Dict, Any, Union, Optional, Tuple
from collections import defaultdict
from tqdm.asyncio import tqdm as async_tqdm

from .metrics import (
    evaluate_math_prediction,
    evaluate_qa_prediction,
    evaluate_grpo_mix_prediction,
)
from .utils import extract_answer
from .llm_evaluator_sds import LLMEvaluator
# from .llm_evaluator import LLMEvaluator


def _load_echo_format_validator():
    """Lazy import of ECHO format validator from the verl training package.
    Resolved relative to this file so the evaluator runs without PYTHONPATH setup."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    verl_path = os.path.join(repo_root, "ARPO", "verl_arpo_entropy")
    if verl_path not in sys.path:
        sys.path.insert(0, verl_path)
    from verl.utils.reward_score.deep_research_echo import validate_format_echo, mask_categories_for_profile
    return validate_format_echo, mask_categories_for_profile


class Evaluator:
    def __init__(
        self,
        task_type: str,
        output_path: str,
        use_llm: bool = False,
        api_base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: str = "EMPTY",
        concurrent_limit: int = 50,
        sigma: float = 0.1,
        prompt_type: Optional[str] = None,
        validator_profile: str = "c1",
    ):
        """
        Initialize evaluator.

        Args:
            task_type: Task type ('math', 'qa', 'grpo_mix')
            output_path: Path to the model output JSON file
            use_llm: Whether to use LLM for evaluation
            api_base_url: Base URL for LLM API
            model_name: Name of the LLM model
            api_key: API key
            concurrent_limit: Concurrency limit
            sigma: Smoothing factor
            prompt_type: Prompt schema used during inference. When set to 'echo',
                run the trainer's format validator and report HL/LL pass rates.
            validator_profile: ECHO validator profile id (c1..c5) matching the
                trainer's mask_categories signature; routes which checks gate HL vs LL.
        """
        self.task_type = task_type
        self.output_path = output_path
        self.use_llm = use_llm
        self.concurrent_limit = concurrent_limit
        self.sigma = sigma
        self.prompt_type = prompt_type
        self.validator_profile = validator_profile
        if prompt_type == "echo":
            _validate_format_echo, _mask_categories_for_profile = _load_echo_format_validator()
            self._echo_validator = _validate_format_echo
            self.mask_categories = _mask_categories_for_profile(validator_profile)
        else:
            self._echo_validator = None
            self.mask_categories = None

        # Output paths
        base_path, ext = os.path.splitext(output_path)
        self.output_metrics_path = f"{base_path}_metrics.json"
        self.output_metrics_overall_path = f"{base_path}_metrics_overall.json"

        # Initialize LLM evaluator if needed
        self.llm_evaluator = None
        if self.use_llm:
            self.llm_evaluator = LLMEvaluator(
                api_base_url=api_base_url,
                model_name=model_name,
                api_key=api_key,
                concurrent_limit=concurrent_limit
            )

    async def evaluate_sample(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single sample.

        Args:
            item: Sample data

        Returns:
            Evaluation metrics
        """
        question = item.get('input', '')
        answer = item.get('answer', '')
        prediction = (item.get('prediction') or '').strip()
        output = item.get('output', '')

        if not prediction and output:
            ex = extract_answer(output)
            if ex and str(ex).strip():
                prediction = str(ex).strip()
        if not prediction and output:
            prediction = '\n'.join(output.replace(
                "\n\n", "\n").strip().split('\n')[-5:])
        elif not prediction:
            prediction = ''
        
        if not prediction:
            # Return zero metrics if prediction is empty
            metrics = {
                "is_valid_answer": False,
                "em": 0,
                "acc": 0,
                "f1": 0,
                "math_equal": 0,
                "llm_equal": 0,
                "python_calls": 0,
                "search_calls": 0,
                "output_length": 0
            }
            # Set tool usage stats
            python_calls = metrics["python_calls"]
            search_calls = metrics["search_calls"]
            metrics["tools_used"] = ("both" if python_calls and search_calls else
                             "python" if python_calls else
                             "search" if search_calls else "none")
            metrics["tool_counts"] = python_calls + search_calls
            if self._echo_validator is not None:
                metrics.update({"echo_format_valid": 0, "echo_high_level_valid": 0,
                                "echo_low_level_valid": 0, "echo_format_reason": "empty prediction"})
            return metrics

        # Initialize metrics
        metrics = {
            "is_valid_answer": prediction != ''
        }

        # Tool usage statistics
        python_calls = count_valid_tags(output, "python")
        search_calls = count_valid_tags(output, "search")

        metrics.update({
            "python_calls": python_calls,
            "search_calls": search_calls,
            "tools_used": ("both" if python_calls and search_calls else
                           "python" if python_calls else
                           "search" if search_calls else "none"),
            "tool_counts": python_calls + search_calls
        })

        # Output length
        metrics["output_length"] = len(remove_result_tags(output))

        # ECHO format validation (only when running an ECHO checkpoint).
        # Runs the trainer's validator on the raw rollout text; reported as a
        # standalone metric and never gates the F1/EM/LLM-judge score.
        if self._echo_validator is not None:
            ok, reason, hl_ok, ll_ok = self._echo_validator(output, self.mask_categories)
            metrics["echo_format_valid"] = int(ok)
            metrics["echo_high_level_valid"] = int(hl_ok)
            metrics["echo_low_level_valid"] = int(ll_ok)
            metrics["echo_format_reason"] = reason

        # Evaluate based on task type
        if self.task_type == 'math':
            math_metrics = evaluate_math_prediction(prediction, answer)
            metrics.update(math_metrics)
        elif self.task_type == 'qa':
            qa_metrics = evaluate_qa_prediction(prediction, answer)
            metrics.update(qa_metrics)
        elif self.task_type == 'grpo_mix':
            mix_metrics = evaluate_grpo_mix_prediction(prediction, answer)
            metrics.update(mix_metrics)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        # LLM evaluation
        if self.use_llm and self.llm_evaluator:
            semaphore = asyncio.Semaphore(self.concurrent_limit)
            is_correct, llm_reason_answer = await self.llm_evaluator.evaluate(
                question=question,  
                labeled_answer=answer,
                pred_answer=prediction,
                semaphore=semaphore
            )
            metrics["llm_equal"] = int(is_correct)
            metrics["llm_response"] = llm_reason_answer

        return metrics

    async def evaluate_batch(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of samples.

        Args:
            data: List of sample data

        Returns:
            Updated sample list with evaluation metrics
        """
        semaphore = asyncio.Semaphore(self.concurrent_limit)
        
        async def _evaluate_with_semaphore(item):
            async with semaphore:
                metrics = await self.evaluate_sample(item)
                item_copy = item.copy()
                item_copy['metrics'] = metrics
                return item_copy
        
        tasks = [_evaluate_with_semaphore(item) for item in data]
        results = await async_tqdm.gather(*tasks, desc="Evaluating samples")
        return results

    async def run(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run the evaluation process.

        Args:
            data: List of sample data

        Returns:
            Overall evaluation metrics
        """
        print(f"Starting evaluation of {len(data)} samples, task type: {self.task_type}")
        updated_data = await self.evaluate_batch(data)
        self.overall_metrics = self.calculate_overall_metrics(updated_data)
        self.overall_metrics['datetime'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save_results(updated_data)
        return self.overall_metrics

    def calculate_overall_metrics(self, data: List[Dict[str, Any]]) -> Dict[str, Union[float, str]]:
        """
        Calculate overall evaluation metrics.

        Args:
            data: Sample list with individual metrics

        Returns:
            Dictionary of overall metrics
        """
        num_valid_answer = sum(item['metrics']['is_valid_answer'] for item in data)
        avg_em = [item['metrics'].get('em', 0) for item in data if 'em' in item['metrics']]
        avg_acc = [item['metrics'].get('acc', 0) for item in data if 'acc' in item['metrics']]
        avg_f1 = [item['metrics'].get('f1', 0) for item in data if 'f1' in item['metrics']]
        avg_math = [item['metrics'].get('math_equal', 0) for item in data if 'math_equal' in item['metrics']]
        avg_llm = [item['metrics'].get('llm_equal', 0) for item in data if 'llm_equal' in item['metrics']]
        avg_tool_counts = [item['metrics'].get('tool_counts', 0) for item in data]
        avg_python_calls = [item['metrics'].get('python_calls', 0) for item in data]
        avg_search_calls = [item['metrics'].get('search_calls', 0) for item in data]
        
        tool_usage_rate = sum(1 for count in avg_tool_counts if count > 0) / len(data) if data else 0
        avg_tool_count = np.mean(avg_tool_counts) if avg_tool_counts else 0

        if self.task_type == 'math':
            accuracy = np.mean(avg_math) if avg_math else 0.0
        elif self.task_type in ('qa', 'grpo_mix'):
            accuracy = np.mean(avg_f1) if avg_f1 else 0.0
        else:
            accuracy = 0.0

        if self.use_llm and avg_llm:
            final_accuracy = np.mean(avg_llm)
        else:
            final_accuracy = accuracy

        final_tool_productivity = final_accuracy * avg_tool_count / (avg_tool_count + self.sigma) * 100.0

        overall_metrics = {
            'em': np.mean(avg_em) if avg_em else 0.0,
            'acc': np.mean(avg_acc) if avg_acc else 0.0,
            'f1': np.mean(avg_f1) if avg_f1 else 0.0,
            'math_equal': np.mean(avg_math) if avg_math else 0.0,
            'num_valid_answer': f'{num_valid_answer} of {len(data)}',
            'tool_productivity': final_tool_productivity,
            'average_datas_used_tool_number': tool_usage_rate,
            'tool_call': avg_tool_count,
            'average_python_calls': np.mean(avg_python_calls) if avg_python_calls else 0.0,
            'average_search_calls': np.mean(avg_search_calls) if avg_search_calls else 0.0,
            'llm_equal': np.mean(avg_llm) if avg_llm else 0.0,
            'm1m2': final_tool_productivity,
        }

        # ECHO format pass rates: report only when the validator ran (i.e.
        # prompt_type=='echo'); aggregated over all samples that have the keys.
        if self._echo_validator is not None:
            avg_fmt = [item['metrics'].get('echo_format_valid', 0) for item in data]
            avg_hl = [item['metrics'].get('echo_high_level_valid', 0) for item in data]
            avg_ll = [item['metrics'].get('echo_low_level_valid', 0) for item in data]
            overall_metrics['echo_format_pass_rate'] = float(np.mean(avg_fmt)) if avg_fmt else 0.0
            overall_metrics['echo_high_level_pass_rate'] = float(np.mean(avg_hl)) if avg_hl else 0.0
            overall_metrics['echo_low_level_pass_rate'] = float(np.mean(avg_ll)) if avg_ll else 0.0
        return overall_metrics

    def save_results(self, data: List[Dict[str, Any]]):
        """
        Save evaluation results.

        Args:
            data: Updated list of sample data
        """
        with open(self.output_metrics_path, mode='w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        with open(self.output_metrics_overall_path, mode='w', encoding='utf-8') as f:
            json.dump(self.overall_metrics, f, indent=4, ensure_ascii=False)

        print("Evaluation complete. Results saved.")
        print(f"Detailed metrics: {self.output_metrics_path}")
        print(f"Overall metrics: {self.output_metrics_overall_path}")

        print(f"EM: {self.overall_metrics.get('em', 0):.4f}")
        print(f"F1: {self.overall_metrics.get('f1', 0):.4f}")
        print(f"Valid answers: {self.overall_metrics.get('num_valid_answer', '0')}")
        
        if self.task_type == 'math':
            print(f"Math Equal: {self.overall_metrics.get('math_equal', 0):.4f}")
        
        if 'llm_equal' in self.overall_metrics:
            print(f"LLM Equal: {self.overall_metrics['llm_equal']:.4f}")
        
        print(f"Avg tool calls: {self.overall_metrics.get('tool_call', 0):.2f}")
        print(f"Python calls: {self.overall_metrics.get('average_python_calls', 0):.2f}")
        print(f"Search calls: {self.overall_metrics.get('average_search_calls', 0):.2f}")
        print(f"Tool usage rate: {self.overall_metrics.get('average_datas_used_tool_number', 0):.2f}")
        print(f"Tool productivity (M1*M2): {self.overall_metrics.get('tool_productivity', 0):.2f}")

        if 'echo_format_pass_rate' in self.overall_metrics:
            print(f"ECHO format pass rate: {self.overall_metrics['echo_format_pass_rate']:.4f}"
                  f" (HL={self.overall_metrics['echo_high_level_pass_rate']:.4f},"
                  f" LL={self.overall_metrics['echo_low_level_pass_rate']:.4f})")


def count_valid_tags(text: str, tag: str) -> int:
    """Count valid paired tags."""
    count = 0
    current_pos = 0

    while True:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"

        start_pos = text.find(start_tag, current_pos)
        if start_pos == -1:
            break

        end_pos = text.find(end_tag, start_pos + len(start_tag))
        if end_pos == -1:
            break

        count += 1
        current_pos = end_pos + len(end_tag)

    return count


def remove_result_tags(text: str) -> str:
    """Remove content inside result tags."""
    if not text:
        return ""
    cleaned_text = re.sub(r'<r>.*?</r>', '', text, flags=re.DOTALL)
    cleaned_text = re.sub(r'<result>.*?</result>', '', cleaned_text, flags=re.DOTALL)
    return cleaned_text.strip()

# Compatibility alias for older versions
def remove_r_tags(text: str) -> str:
    """Compatibility function."""
    return remove_result_tags(text)
