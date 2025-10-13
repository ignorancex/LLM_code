from bert_score import score

from .am_model import BaseAMModel


class BertScoreAM(BaseAMModel):

    def __init__(self, lang="en"):
        self.maximum_val = 100.0
        self.minimum_val = 0.0
        self.lang = lang

    def evaluate_answers(
        self,
        predicted_answers: list[str],
        questions: list[str] | None = None,
        reference_answers: list[str] | None = None,
        contexts: list[str] | None = None,
        batch_size: int = 128,
    ) -> list[float]:
        assert reference_answers is not None, "must provide reference answers"

        _, _, F1 = score(
            predicted_answers,
            reference_answers,
            lang=self.lang,
            verbose=True,
            batch_size=batch_size,
        )

        return F1.numpy()
