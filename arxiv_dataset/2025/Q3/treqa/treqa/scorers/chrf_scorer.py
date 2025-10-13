import sacrebleu

from .doc_scorer import DocScorer


class ChrfScorer(DocScorer):
    def __init__(self):
        self.maximum_val = 100.0
        self.minimum_val = 0.0

    def get_scores(
        self,
        translations: list[str],
        sources: list[str] | None = None,
        references: list[str] | None = None,
    ):
        assert references is not None, "References are required for CHRF"
        return [
            sacrebleu.sentence_chrf(x, [y]).score
            for x, y in zip(translations, references)
        ]
