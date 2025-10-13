from .doc_scorer import DocScorer


class MixScorer(DocScorer):
    def __init__(self, scorer_list: list[DocScorer], weights: list[float]):
        assert len(scorer_list) == len(
            weights
        ), "Scorer list and weights must be the same length"
        self.scorer_list = scorer_list
        self.weights = weights

    def get_scores(
        self,
        translations: list[str],
        sources: list[str] | None = None,
        references: list[str] | None = None,
        **kwargs
    ):
        agg_scores = [0.0] * len(translations)
        for scorer, weight in zip(self.scorer_list, self.weights):
            scores = scorer.get_scores(translations, sources, references, **kwargs)
            agg_scores = [a + b * weight for a, b in zip(agg_scores, scores)]
        return agg_scores
