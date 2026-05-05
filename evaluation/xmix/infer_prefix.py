"""Prefix-aware variant of ``evaluation/infer.py``.

Same CLI as ``infer.py`` (we re-import ``parse_arguments`` to share every flag
and argparse default), but each row in the dataset's ``test.jsonl`` may carry
an extra ``prefix`` field. When present, the prefix is appended to the
assistant side of the chat-template context once, right after
``process_input()``, before the inference loop's first ``call_llm()``. The
existing tool dispatch / continuation loop in ``SampleProcessorCompletion``
runs unchanged from there, so the model continues from the prefix as if it
had just emitted those tokens itself.

Per-tool budget counters (``python_rounds``/``search_rounds``) are pre-charged
by the number of closed tool tags in the prefix so the existing combined-limit
gate in ``SampleProcessorCompletion.run`` honors the same per-rollout budget
the trainer used.
"""

import asyncio
import json
import os
import sys

# `evaluation/` is added to sys.path when ``cd evaluation && python xmix/infer_prefix.py``
# is run; mirror infer.py's setup so ``src.*`` and ``infer`` import cleanly.
sys.path.append(os.getcwd())

from infer import parse_arguments  # noqa: E402  reuse the shared argparse spec
from src.inference_engine import AsyncInferenceCompletionSDS  # noqa: E402
from src.sample_processor import SampleProcessorCompletion  # noqa: E402


class SampleProcessorCompletionPrefix(SampleProcessorCompletion):
    """``SampleProcessorCompletion`` that pre-pends a fixed assistant prefix.

    The whole behavior change lives in ``process_input``: after the parent
    builds ``self.in_context`` from ``[system, user]`` via the chat template,
    we append the prefix as if it were already-emitted assistant output via
    ``log_output('assistant', ...)``. That single call updates ``in_context``,
    ``sample_stat['output']``, and ``sample_stat['logs']`` consistently with
    how mid-rollout chunks are accumulated by the inference loop.
    """

    def __init__(self, *args, prefix: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix = prefix or ""

    def process_input(self):
        super().process_input()
        if not self.prefix:
            return
        # Append assistant prefix once; the existing call_llm/loop reads
        # self.in_context and continues from here.
        self.log_output("assistant", self.prefix)
        # Pre-charge per-tool budgets for tool calls already present in the
        # prefix so the combined-limit gate in run() applies to total calls
        # (prefix + continuation). Counts only closed tags (open tags without
        # a close indicate a malformed prefix; the splicer drops those).
        self.python_rounds += self.prefix.count("</python>")
        self.search_rounds += self.prefix.count("</search>")


class AsyncInferencePrefix(AsyncInferenceCompletionSDS):
    """``AsyncInferenceCompletionSDS`` that injects a per-question prefix.

    Prefix lookup is by question text since ``DataLoader`` only returns
    questions/answers; we read the same ``test.jsonl`` separately to build a
    ``question -> prefix`` map per dataset.
    """

    def __init__(self, args):
        super().__init__(args)
        self._prefix_by_question: dict[str, str] = {}

    def _load_prefix_map(self, dataset_name: str) -> dict[str, str]:
        path = os.path.join(self.args.dataset_path, dataset_name, "test.jsonl")
        m: dict[str, str] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                m[row["question"]] = row.get("prefix", "")
        return m

    def get_processor(self, sample_stat, session_id):
        return SampleProcessorCompletionPrefix(
            self.prompt_manager,
            self.tool_executor,
            self.vllm_pool,
            self.tokenizer,
            self.args,
            sample_stat,
            session_id,
            prefix=sample_stat.get("prefix", ""),
        )

    async def process_sample(self, question, golden_answer, session_id=None):
        # Re-create parent's sample_stat but stamp the prefix looked up by
        # question; everything else is identical to AsyncInference.process_sample.
        sample_stat = {
            "instruction": self.prompt_manager.get_system_prompt(),
            "input": question,
            "output": "",
            "prediction": "",
            "answer": golden_answer,
            "logs": [],
            "search_query_history": set(),
            "prefix": self._prefix_by_question.get(question, ""),
        }
        current_task = asyncio.current_task()
        if current_task:
            setattr(current_task, "_current_result", sample_stat)
        processor = self.get_processor(sample_stat, session_id)
        await processor.run()
        processor.log_timing()
        return processor.sample_stat

    async def run(self):
        # Single-shot prefix-map load up front (typically one dataset). Later
        # datasets overwrite same-key entries; harmless since runs target a
        # single dataset folder per launch.
        self._prefix_by_question = {}
        for dataloader in self.data_loaders:
            self._prefix_by_question.update(self._load_prefix_map(dataloader.dataset_name))
        await super().run()


async def _amain():
    args = parse_arguments()
    print(vars(args))
    inference = AsyncInferencePrefix(args)
    await inference.run()
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(_amain())
