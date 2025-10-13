from ..prompt_model import PromptModel

from .qa_model import BaseQAModel
from .qa_templates import template_dict, system_prompt_dict


class PromptQA(PromptModel, BaseQAModel):
    def __init__(
        self,
        provider="vllm",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        max_tokens=1024,
        max_model_len=None,
        temperature=0.0,
        top_p=1.0,
        tensor_parallel_size=1,
        base_url="https://cmu.litellm.ai",
        api_key=None,
        template="standard",
        system_template="standard",
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
            api_key=api_key,
        )
        self.template = template_dict[template]
        self.system_prompt = system_prompt_dict[system_template]

    def prepare_chat(self, passage: str, question: str) -> list[dict[str, str]]:
        query_prompt = self.template.format(passage=passage, question=question)
        chat = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query_prompt},
        ]
        return chat

    def extract_answers(
        self,
        passages: list[str],
        questions: list[list[str]],
        answer_file: str = "answers.json",
    ) -> list[list[str]]:

        all_chats = []
        for passage, passage_qs in zip(passages, questions):
            passage_chats = [
                self.prepare_chat(passage, question) for question in passage_qs
            ]
            all_chats.extend(passage_chats)

        all_answers = self.generate(all_chats)

        # group answers by passage
        answers = []
        idx = 0
        for passage_qs in questions:
            answers.append(all_answers[idx : idx + len(passage_qs)])
            idx += len(passage_qs)

        # HACK: dump answers to file
        with open(answer_file, "w") as f:
            import json

            json.dump(answers, f)

        return answers
