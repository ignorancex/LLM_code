import sacrebleu

from .am_model import BaseAMModel


class ChrfAM(BaseAMModel):

    def __init__(self):
        self.maximum_val = 100.0
        self.minimum_val = 0.0

    def evaluate_answers(
        self,
        predicted_answers: list[str],
        questions: list[str] | None = None,
        reference_answers: list[str] | None = None,
        contexts: list[str] | None = None,
    ) -> list[float]:
        assert reference_answers is not None, "must provide reference answers"
        scores = []
        for predicted_answer, reference_answer in zip(
            predicted_answers, reference_answers
        ):
            scores.append(
                sacrebleu.sentence_chrf(predicted_answer, [reference_answer]).score
            )
        return scores
