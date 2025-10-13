from ..prompt_model import PromptModel

from .ae_model import BaseAEModel

SYSTEM_PROMPT_STANDARD = """You are an expert in keyphrase extraction from text. Your goal is to identify keyphrases in a given text that are crucial to its meaning. Focus on phrases that, if changed, would significantly alter the sentence's meaning or tone. Output only the phrases, one per line."""
SYSTEM_PROMPT_TEACHER = """You are an expert teacher who is tasked with designing reading comprehension questions from a text."""

QUERY_TEMPLATE_STD = """Identify 1-{num_answers} short (1-3 words) keyphrases from the following text that are crucial to its meaning. Focus on phrases that, if changed, would significantly alter the sentence's meaning or tone. Output only the phrases, one per line:

Text: {text}

Key phrases:"""

QUERY_TEMPLATE_RQ = """Analyze the following passage to identify 1-{num_answers} key facts, concepts, and primary relationships presented. Focus on pieces of text that highlight core ideas, essential information, and connections between ideas that are central to understanding the passage as a whole. Extract concise segments that capture the main points and that can serve as a basis for comprehension questions. Output only the phrases, one per line:

Text: {text}

Key phrases:"""

ENG_QUERY_TEMPLATE_STD = """Identify 1-{num_answers} short (1-3 words) keyphrases from the following text that are crucial to its meaning. Focus on phrases that, if changed, would significantly alter the sentence's meaning or tone. Output only the phrases in English, one per line:

Text: {text}

Key phrases:"""

ENG_QUERY_TEMPLATE_RQ = """Analyze the following passage to identify 1-{num_answers} key facts, concepts, and primary relationships presented. Focus on pieces of text that highlight core ideas, essential information, and connections between ideas that are central to understanding the passage as a whole. Extract concise segments that capture the main points and that can serve as a basis for comprehension questions. Output only the phrases in English, one per line:

Text: {text}

Key phrases:"""

template_dict = {
    "standard": QUERY_TEMPLATE_STD,
    "answer-rq": QUERY_TEMPLATE_RQ,
    "eng-standard": ENG_QUERY_TEMPLATE_STD,
    "eng-answer-rq": ENG_QUERY_TEMPLATE_RQ,
}

system_prompt_dict = {
    "standard": SYSTEM_PROMPT_STANDARD,
    "teacher": SYSTEM_PROMPT_TEACHER,
}

NUM_ANSWERS_DEFAULT = 5


class PromptAE(PromptModel, BaseAEModel):
    def __init__(
        self,
        provider="vllm",
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        max_tokens=1024,
        max_model_len=None,
        temperature=0.0,
        top_p=0.9,
        tensor_parallel_size=1,
        base_url="https://cmu.litellm.ai",
        api_key=None,
        template="standard",
        parent_prompt_model=None,
        gpu_memory_utilization=0.7,
        system_template="standard",
        extract_from="target",
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
            parent_prompt_model=parent_prompt_model,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.template = template_dict[template]
        self.system_prompt = system_prompt_dict[system_template]
        if extract_from not in ["target", "source"]:
            raise ValueError(
                "Please specify from where the keyphrases should be extracted: 'source' or 'target'."
            )
        self.extract_from = extract_from

    def prepare_chat(self, text: str, num_answers: int) -> list[dict[str, str]]:
        query_prompt = self.template.format(text=text, num_answers=num_answers)
        chat = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query_prompt},
        ]
        return chat

    def parse_generated_keyphrases(self, generated_text, original_text):
        # Split the text into lines and remove empty lines
        lines = [line.strip() for line in generated_text.split("\n") if line.strip()]

        # Remove any leading numbers or dashes
        keyphrases = [line.lstrip("0123456789.- ") for line in lines]

        # Remove empty strings and keyphrases not in original text
        keyphrases = [
            phrase for phrase in keyphrases if phrase and phrase in original_text
        ]

        # Remove duplicates while preserving order
        seen = set()
        unique_keyphrases = [
            phrase for phrase in keyphrases if not (phrase in seen or seen.add(phrase))
        ]

        return unique_keyphrases

    def extract_answers(
        self,
        passages: list[str],
        num_answers: int | None = None,
    ) -> list[list[str]]:
        if not num_answers:
            num_answers = NUM_ANSWERS_DEFAULT

        chats = [self.prepare_chat(src, num_answers) for src in passages]

        # Generate questions
        outputs = self.generate(chats)

        answers = []
        for output, target in zip(outputs, passages):
            keyphrase = self.parse_generated_keyphrases(output, target)
            answers.append(keyphrase)

        return answers
