from ..prompt_model import PromptModel

from .qg_model import BaseQGModel
from .output_parser import OutputParser

SYSTEM_PROMPT = "You are a helpful AI assistant skilled in generating questions when given a passage and a list of keyphrases as answers."

QUERY_TEMPLATE = """You are provided with a passage and a list of keyphrases separated by a newline. These keyphrases are extracted from the passage and should be used as answers. Your task is to generate diverse question-answer pairs based on the passage, ensuring that each answer is about one of the provided keyphrases.

Format your response as follows:
Q: <question>
A: <answer containing exact keyphrase from the list>

Passage:
{passage}

Keyphrases:
{keyphrases}

Question-Answer Pairs:
"""

ENG_QUERY_TEMPLATE = """You are provided with a passage and a list of keyphrases separated by a newline. These keyphrases are extracted from the passage. Your task is to generate diverse question-answer pairs that are based on the passage and are strictly in English. Ensure that each answer about one of the provided keyphrases.

Format your response as follows:
Q: <question>
A: <answer containing exact keyphrase from the list>

Passage:
{passage}

Keyphrases:
{keyphrases}

Question-Answer Pairs:
"""

template_dict = {
    "standard": QUERY_TEMPLATE,
    "eng-standard": ENG_QUERY_TEMPLATE,
}


class PromptQG(PromptModel, BaseQGModel):
    def __init__(
        self,
        provider="vllm",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        max_tokens=1024,
        max_model_len=None,
        temperature=0.0,
        top_p=0.9,
        tensor_parallel_size=1,
        base_url="https://cmu.litellm.ai",
        api_key=None,
        template="standard",
        answer_overlap_threshold=60,
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
        self.answer_overlap_threshold = answer_overlap_threshold
        self.out_parser = OutputParser(model_name)

    def prepare_chat(self, passage: str, answers: str) -> list[dict[str, str]]:
        query_prompt = self.template.format(passage=passage, keyphrases=answers)
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query_prompt},
        ]
        return chat

    def generate_questions(
        self,
        passages: list[str],
        answers: list[list[str]],
        log_skipped_qa_pairs: bool = True,
    ) -> list[list[tuple[str, str]]]:
        # NOTE: for now this is ignored, as we are exploring
        # allowing the model to generate as many questions as it wants
        chats = [
            self.prepare_chat(target, "\n".join(answer))
            for target, answer in zip(passages, answers)
        ]

        # Generate questions
        outputs = self.generate(chats)

        qa_pairs = []
        for output, target in zip(outputs, passages):
            qas, _ = self.out_parser.parse(
                output, target, self.answer_overlap_threshold
            )
            qa_pairs.append(qas)

        return qa_pairs
