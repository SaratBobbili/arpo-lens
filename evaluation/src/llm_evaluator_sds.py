import sys
import os
sys.path.append(os.getcwd())
import re
import asyncio
import time
from typing import List, Tuple, Dict, Any, Optional

from openai import OpenAI, AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm

from .math_equivalence import is_equiv


EVALUATION_PROMPT = """Given a Question and its Golden Answer, verify whether the Predicted Answer is correct. The prediction is correct if it fully aligns with the meaning and key information of the Golden Answer. Respond with "Correct" if the prediction is correct and "Incorrect" otherwise.
Golden Answer may have multiple options, and matching any one of them is considered correct.

Question: {question}
Golden Answer: {labeled_answer}
Predicted Answer: {pred_answer}
"""


def parse_judge_verdict(response_text: str) -> bool:
    """Deterministically map judge text to a correct/incorrect verdict.

    The verdict is derived from the judge text alone (never OR-ed with a
    string matcher) so llm_equal always agrees with the stored llm_response:
    any 'incorrect'/'not correct'/'wrong' phrasing yields False, and 'correct'
    (checked only after ruling out the negatives) yields True.
    """
    text = response_text or ""
    m = re.search(r"<judgment>(.*?)</judgment>", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1)
    text = text.strip().lower()
    if not text:
        return False
    if "incorrect" in text or "not correct" in text or "wrong" in text:
        return False
    return "correct" in text


class LLMEvaluator:
    """Class for evaluating answer equivalence using an LLM"""

    def __init__(
        self,
        api_base_url: str = None,
        model_name: str = None,
        api_key: str = "empty",
        concurrent_limit: int = 50,
        retry_limit: int = 3
    ):
        """
        Initialize the LLM evaluator.

        Args:
            api_base_url: Base URL of the API, defaults to local service
            model_name: Model name
            api_key: API key
            concurrent_limit: Concurrency limit
            retry_limit: Retry limit
        """
        if api_base_url is None:
            api_base_url = "http://localhost:8000/v1"
        if model_name is None:
            model_name = "<your_model_name>"

        self.api_base_url = api_base_url
        self.model_name = model_name
        self.api_key = api_key
        self.concurrent_limit = concurrent_limit
        self.retry_limit = retry_limit

        # Create OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base_url
        )

    async def evaluate(
        self,
        question: str,
        labeled_answer: str,
        pred_answer: str,
        semaphore: asyncio.Semaphore
    ) -> Tuple[bool, str]:
        """
        Evaluate a single answer pair.

        Args:
            question: Question text
            labeled_answer: Ground truth answer
            pred_answer: Predicted answer
            semaphore: Semaphore to control concurrency

        Returns:
            (is_correct, evaluation_reason)
        """
        global EVALUATION_PROMPT
        prompt = EVALUATION_PROMPT.format(
            question=question,
            labeled_answer=labeled_answer,
            pred_answer=pred_answer
        )

        for attempt in range(self.retry_limit):
            try:
                async with semaphore:
                    chat_response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}]
                    )

                    reason_answer = chat_response.choices[0].message.content.strip()

                    # Verdict is a deterministic function of the judge text only.
                    is_correct = parse_judge_verdict(reason_answer)

                    return is_correct, reason_answer
            except Exception as e:
                if attempt == self.retry_limit - 1:
                    print(f"Error in LLM evaluation: {e}")
                    print(
                        f"-------------------pred_answer: {pred_answer}----------------------")
                    print(
                        f"-------------------labeled_answer: {labeled_answer}----------------------")
                    return is_equiv(pred_answer, labeled_answer), "Error"
                await asyncio.sleep(1 * (attempt + 1))

        return is_equiv(pred_answer, labeled_answer), "Error"
