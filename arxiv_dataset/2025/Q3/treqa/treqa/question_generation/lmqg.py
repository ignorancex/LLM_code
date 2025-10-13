from collections import defaultdict

from lmqg import TransformersQG
from tqdm import tqdm
from transformers.utils.logging import disable_progress_bar

from .qg_model import BaseQAGModel


class LMQG(BaseQAGModel):
    def __init__(
        self,
        model_name: str = "lmqg/t5-large-squad-qag",
        batch_size: int = 16,
        trim_val: int = 512,
        model_ae: str | None = None,
    ):
        """Task-specific question generation models."""
        if model_ae is not None:
            self.model = TransformersQG(model_name, model_ae=model_ae)
        else:
            self.model = TransformersQG(model_name)
        self.batch_size = batch_size
        self.trim_val = trim_val

    def generate_qa_pairs(
        self,
        passages: list[str],
        num_questions: int | None = None,
        candidates: list[list[str]] | None = None,
        ref_passages: list[str] | None = None,
    ) -> list[list[tuple[str, str]]]:
        # HACK: for now, trim targets to 2048 since, longer than this causes an error
        # due to the max lenght of the t5 models. report how many were trimmed
        num_trimmed = 0
        for target in passages:
            if len(target) > self.trim_val:
                num_trimmed += 1
        print(f"Trimmed {num_trimmed} targets to {self.trim_val} characters")
        targets = [target[: self.trim_val] for target in passages]

        # Create a mapping of unique targets to their indices
        # this is avoid duplicate computation when generating questions
        # TODO: is this the right abstraction layer to do this?
        target_to_indices = defaultdict(list)
        for i, target in enumerate(targets):
            target_to_indices[target].append(i)

        unique_targets = list(target_to_indices.keys())

        # Batch the targets and use tqdm for progress bar
        # TODO: this still doesn't work, there is some progress bar that is being shown
        disable_progress_bar()
        unique_qa_pairs = []
        for i in tqdm(
            range(0, len(unique_targets), self.batch_size), desc="Generating QA pairs"
        ):
            batch_targets = unique_targets[i : i + self.batch_size]
            unique_qa_pairs.extend(
                self.model.generate_qa(
                    batch_targets,
                    num_questions=num_questions,
                    batch_size=self.batch_size,
                )
            )

        # map the questions back to the original indices & remove
        cleaned_qa_pairs: list[list[tuple[str, str]]] = [
            [] for _ in range(len(targets))
        ]
        for unique_tgt, qa_pairs in zip(unique_targets, unique_qa_pairs):
            for original_index in target_to_indices[unique_tgt]:
                cleaned_qa_pairs[original_index] = list(set(qa_pairs))

        return cleaned_qa_pairs
