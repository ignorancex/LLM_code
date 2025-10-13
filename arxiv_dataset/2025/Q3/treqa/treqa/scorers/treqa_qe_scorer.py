import numpy as np

from treqa.question_answering import BaseQAModel
from treqa.question_answering.qa_templates import template_dict
from treqa.answer_matching import BaseAMModel

from .doc_scorer import DocScorer, flatten


class TREQAQEScorer(DocScorer):
    def __init__(
        self,
        qa_model: BaseQAModel,
        answer_comparator: BaseAMModel,
        utility_type="default",
        fallback="error",
        gen_ref_answers=True,
        ref_template="eng-standard",
    ):
        self.qa_model = qa_model
        self.answer_comparator = answer_comparator
        self.fallback = fallback
        self.utility_type = utility_type
        self.gen_ref_answers = gen_ref_answers
        self.ref_template = ref_template

    def get_scores(
        self,
        translations: list[str],
        sources: list[str] | None = None,
        references: list[str] | None = None,
        qa_pairs: list[list[dict]] | None = None,
        return_detailed_evaluation: bool = False,
    ):
        assert qa_pairs is not None, "Must provide QA pairs"
        assert sources is not None, "Must provide sources"

        # Compute QA-based scores
        questions = [
            [qa_pair["question"] for qa_pair in doc_qa_pairs]
            for doc_qa_pairs in qa_pairs
        ]
        # (answers from source)

        predicted_answers = self.qa_model.extract_answers(
            translations, questions, answer_file="pred_answers.json"
        )

        if self.gen_ref_answers:
            if self.ref_template is not None:
                self.qa_model.template = template_dict[self.ref_template]
            reference_answers = self.qa_model.extract_answers(
                sources, questions, answer_file="src_answers.json"
            )
        else:
            reference_answers = [
                [qa_pair["answer"] for qa_pair in doc_qa_pairs]
                for doc_qa_pairs in qa_pairs
            ]

        all_scores = self.answer_comparator.evaluate_answers(
            flatten(predicted_answers),
            flatten(questions),
            flatten(reference_answers),
            flatten([[sources[i]] * len(qa_pairs[i]) for i in range(len(sources))]),
        )

        assert len(all_scores) == len(flatten(predicted_answers))

        # group scores by passage
        qa_scores = []
        per_q_scores = []
        idx = 0
        for ind, passage_qs in enumerate(qa_pairs):
            if len(passage_qs) == 0:  # fallback for no questions
                print(f"Using fallback for index: {ind}")
                if self.fallback == "no_error":
                    qa_scores.append(self.answer_comparator.maximum_val)
                    continue
                elif self.fallback == "error":
                    qa_scores.append(self.answer_comparator.minimum_val)
                    continue
                else:
                    raise ValueError(f"Unknown fallback method: {self.fallback}")

            qs_scores = all_scores[idx : idx + len(passage_qs)]
            qa_scores.append(np.mean(qs_scores))
            per_q_scores.append(qs_scores)
            idx += len(passage_qs)

        assert idx == len(all_scores)

        if return_detailed_evaluation:
            return qa_scores, predicted_answers, reference_answers, per_q_scores
        else:
            return qa_scores
