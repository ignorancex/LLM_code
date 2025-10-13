from ..prompt_model import PromptModel

import re
import numpy as np

from .am_model import BaseAMModel

SYSTEM_PROMPT = "You are a helpful AI assistant skilled in evaluating answers to questions given a context."
QUERY_TEMPLATE = """You are an evaluator tasked with assessing the quality of an answer based on a given context and question. Assign a score between 1 and 5 using the following criteria:

Scoring Guidelines:
1 (Poor): The answer does not address the question, provides incorrect information, or is unrelated to the context.
2 (Below Average): The answer demonstrates limited understanding, contains major errors, or omits critical elements, making it inadequate.
3 (Average): The answer is somewhat correct but lacks clarity, completeness, or sufficient relevance to fully address the question.
4 (Good): The answer is clear, mostly correct, and addresses the key points of the question effectively, though minor improvements are possible.
5 (Excellent): The answer is entirely correct, comprehensive, and demonstrates a deep understanding of the question and context.

Instructions:
Read the context, question, and answer carefully.
Use the scoring guidelines to assign a score from 1 to 5.
Ensure your score reflects the distinctions between the levels.
Please output only the score.

###
Context:
{context}
###
Question:
{question}
###
Answer:
{answer}
###
Score:"""


class PromptAM(PromptModel, BaseAMModel):
    def __init__(
        self,
        normalize_scores=False,
        num_runs=20,
        provider="vllm",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        max_tokens=1024,
        max_model_len=None,
        temperature=0.0,
        top_p=1.0,
        tensor_parallel_size=1,
        base_url="https://cmu.litellm.ai",
        parent_prompt_model=None,
        api_key=None,
        gpu_memory_utilization=0.9,
        logprobs=None,
    ):
        super().__init__(
            provider=provider,
            model_name=model_name,
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            temperature=temperature,
            top_p=top_p,
            tensor_parallel_size=tensor_parallel_size,
            base_url=base_url,
            parent_prompt_model=parent_prompt_model,
            api_key=api_key,
            gpu_memory_utilization=gpu_memory_utilization,
            logprobs=logprobs,
        )
        self.maximum_val = 5.0
        self.minimum_val = 0.0
        self.normalize_scores = normalize_scores
        self.num_runs = num_runs

    def prepare_chat(
        self,
        context: str,
        question: str,
        answer: str,
    ) -> list[dict[str, str]]:
        query_prompt = QUERY_TEMPLATE.format(
            context=context,
            question=question,
            answer=answer,
        )
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query_prompt},
        ]
        return chat

    def parse_scores(self, scores_raw):
        scores = []
        for x in scores_raw:
            match = re.search(r"\d+", x.split("\n\n")[0])
            if match:
                scores.append(float(match.group()))
        return scores

    def evaluate_answers(
        self,
        predicted_answers: list[str],
        questions: list[str] | None = None,
        reference_answers: list[str] | None = None,
        contexts: list[str] | None = None,
    ) -> list[float]:
        assert questions is not None, "Must provide questions"
        assert contexts is not None, "Must provide contexts"

        all_chats = []
        for context, question, answer in zip(contexts, questions, predicted_answers):
            all_chats.append(self.prepare_chat(context, question, answer))

        if self.normalize_scores:
            repeated_chats = []
            for x in all_chats:
                repeated_chats.extend([x] * self.num_runs)
            scores_raw = self.generate(repeated_chats, unique_only=False)
            scores = self.parse_scores(scores_raw)
            scores = (
                np.array(scores).reshape((len(all_chats), self.num_runs)).mean(axis=1)
            )
        else:
            scores_raw = self.generate(all_chats)
            scores = self.parse_scores(scores_raw)

        assert len(scores) == len(all_chats)

        return scores
