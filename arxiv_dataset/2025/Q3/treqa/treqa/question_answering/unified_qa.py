from tqdm import tqdm

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

from .qa_model import BaseQAModel


class UnifiedQA(BaseQAModel):
    def __init__(
        self,
        model_name="allenai/unifiedqa-v2-t5-3b-1363200",
        batch_size=2,
        device="cuda",
    ):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.batch_size = batch_size
        self.device = device

    def prepare_input(self, passage: str, question: str) -> str:
        return f"{question} \\n {passage}".lower()

    def extract_answers(
        self, passages: list[str], questions: list[list[str]]
    ) -> list[list[str]]:
        all_inputs = []
        for passage, passage_questions in zip(passages, questions):
            all_inputs.extend(
                [self.prepare_input(passage, q) for q in passage_questions]
            )

        all_answers = []
        for i in tqdm(
            range(0, len(all_inputs), self.batch_size),
            total=len(all_inputs) // self.batch_size,
        ):
            batch_inputs = all_inputs[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch_inputs, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                )

            batch_answers = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            all_answers.extend(batch_answers)

        answers = []
        idx = 0
        for passage_qs in questions:
            answers.append(all_answers[idx : idx + len(passage_qs)])
            idx += len(passage_qs)

        return answers
