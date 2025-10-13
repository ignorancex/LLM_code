import numpy as np
from difflib import SequenceMatcher
from collections import Counter
import random

from treqa.question_answering import BaseQAModel
from treqa.answer_matching import BaseAMModel
from treqa.question_answering.qa_templates import template_dict

from .doc_scorer import DocScorer, flatten


def select_questions(
    questions_list, N, K, num_questions=None, select_strategy="frequent"
):
    if len(questions_list) != N * K:
        raise ValueError("The size of the list must be N x K.")

    # Split the flattened list into chunks of size K corresponding to a unique source
    cands_chunks = [questions_list[i * K : (i + 1) * K] for i in range(N)]

    result = []
    for cand_chunk in cands_chunks:

        all_questions = [q for questions in cand_chunk for q in questions]
        freq_counter = Counter(all_questions)

        # discriminative questions
        if num_questions:
            if select_strategy == "frequent":
                most_common_values = [
                    item for item, _ in freq_counter.most_common(num_questions)
                ]
                result.extend([most_common_values] * K)
            elif select_strategy == "similarity":
                # Compute pairwise similarity scores
                similarity_matrix = np.zeros((len(all_questions), len(all_questions)))
                for i, q1 in enumerate(all_questions):
                    for j, q2 in enumerate(all_questions):
                        if i != j:
                            # Return a measure of the sequences’ lexical similarity as a float in the range [0, 1].
                            similarity_matrix[i, j] = SequenceMatcher(
                                None, q1, q2
                            ).ratio()

                # Sum similarity scores to find the most "central" questions
                scores = similarity_matrix.sum(axis=1)
                most_similar_indices = np.argsort(-scores)[:num_questions]
                result.extend([[all_questions[i] for i in most_similar_indices]] * K)
            elif select_strategy == "random":
                result.extend(
                    [
                        random.sample(
                            all_questions, min(num_questions, len(all_questions))
                        )
                    ]
                    * K
                )
            else:
                raise ValueError(f"Unknown selection strategy: {select_strategy}")
        else:
            result.extend([list(set(all_questions))] * K)

    return result


class TREQAScorer(DocScorer):
    def __init__(
        self,
        qa_model: BaseQAModel,
        answer_comparator: BaseAMModel,
        fallback="no_error",
        utility_type="default",
        gen_ref_answers=True,
        ref_template="standard",
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
        assert references is not None, "References are required for TREQAScorer"

        # Compute QA-based scores
        questions = [
            [qa_pair["question"] for qa_pair in doc_qa_pairs]
            for doc_qa_pairs in qa_pairs
        ]

        predicted_answers = self.qa_model.extract_answers(
            translations, questions, answer_file="pred_answers.json"
        )

        if self.gen_ref_answers:
            if self.ref_template is not None:
                self.qa_model.template = template_dict[self.ref_template]
            reference_answers = self.qa_model.extract_answers(
                references, questions, answer_file="ref_answers.json"
            )
        else:
            reference_answers = [
                [qa_pair["answer"] for qa_pair in doc_qa_pairs]
                for doc_qa_pairs in qa_pairs
            ]

        # free qa space  -> should be done for quip only
        # if isinstance(self.answer_comparator, QuipAM):
        #     self.qa_model.cleanup_model()

        all_scores = self.answer_comparator.evaluate_answers(
            flatten(predicted_answers),
            flatten(questions),
            flatten(reference_answers),
            flatten(
                [[references[i]] * len(qa_pairs[i]) for i in range(len(references))]
            ),
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
