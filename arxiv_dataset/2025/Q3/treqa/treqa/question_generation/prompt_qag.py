from ..prompt_model import PromptModel

from .qg_model import BaseQAGModel
from .output_parser import OutputParser
from .qag_templates import (
    system_prompt_dict,
    template_dict,
    disc_template_dict,
    eval_template_dict,
    template_dict_both,
)


def number_list(alternatives):
    return "\n".join(
        f"{i+1}. {alternative}" for i, alternative in enumerate(alternatives)
    )


class PromptQAG(PromptModel, BaseQAGModel):
    def __init__(
        self,
        provider="vllm",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        max_tokens=1024,
        max_model_len=None,
        temperature=0.0,
        top_p=1.0,
        min_p=0.0,
        tensor_parallel_size=1,
        base_url="https://cmu.litellm.ai",
        api_key=None,
        template="eng-cands-0shot",
        answer_overlap_threshold=0.0,
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
        self.template = template
        self.answer_overlap_threshold = answer_overlap_threshold
        self.system_prompt = system_prompt_dict[system_template]
        self.out_parser = OutputParser(model_name)

    def prepare_chat(
        self,
        passage: str,
        num_questions: int | str,
        alternatives: list[str] | None = None,
        alt_passage: str | None = None,
    ) -> list[dict[str, str]]:
        if alternatives is None:
            if alt_passage is not None:
                query_prompt = template_dict_both[self.template].format(
                    src_passage=passage,
                    num_questions=num_questions,
                    ref_passage=alt_passage,
                )
            else:
                query_prompt = template_dict[self.template].format(
                    passage=passage, num_questions=num_questions
                )
        else:
            alternatives_str = "\n-\n".join(alternatives)
            if alt_passage is not None:
                query_prompt = eval_template_dict[self.template].format(
                    src_passage=passage,
                    num_questions=num_questions,
                    alternatives=alternatives_str,
                    ref_passage=alt_passage,
                )
            else:

                query_prompt = disc_template_dict[self.template].format(
                    passage=passage,
                    num_questions=num_questions,
                    alternatives=alternatives_str,
                )

        chat = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query_prompt},
        ]
        return chat

    def generate_qa_pairs(
        self,
        passages: list[str],
        num_questions: int | None = None,
        candidates: list[list[str]] | None = None,
        alt_passages: list[str] | None = None,
    ) -> list[list[tuple[str, str]]]:

        num_questions_def = f" {num_questions}" if num_questions else ""

        chats = [
            self.prepare_chat(
                target,
                num_questions=num_questions_def,
                alternatives=candidates[idx] if candidates is not None else None,
                alt_passage=alt_passages[idx] if alt_passages is not None else None,
            )
            for idx, target in enumerate(passages)
        ]

        outputs = self.generate(chats)

        # Process the generated questions
        qa_pairs = []
        for output, target in zip(outputs, passages):
            qas, _ = self.out_parser.parse(
                output, target, self.answer_overlap_threshold
            )
            qa_pairs.append(qas)

        return qa_pairs
