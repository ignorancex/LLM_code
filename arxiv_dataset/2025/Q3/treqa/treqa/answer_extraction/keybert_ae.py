from keybert import KeyBERT

from .ae_model import BaseAEModel


class KeyBertAE(BaseAEModel):
    def __init__(
        self,
        keyphrase_ngram_range: tuple = (2, 5),
        use_maxsum: bool = False,
        use_mmr: bool = True,
        diversity: float = 0.7,
        model: str = "all-MiniLM-L6-v2",
        nr_candidates: int = 20,
    ):
        """Task-specific question generation models."""
        self.kw_model = KeyBERT(model=model)
        self.keyphrase_ngram_range = keyphrase_ngram_range
        self.use_maxsum = use_maxsum
        self.use_mmr = use_mmr
        self.diversity = diversity
        self.nr_candidates = nr_candidates

    def extract_answers(
        self,
        passages: list[str],
        num_answers: int | None = None,
    ) -> list[list[str]]:

        keywords = []
        for text in passages:
            extracted_kws = self.kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=self.keyphrase_ngram_range,
                use_mmr=self.use_mmr,
                use_maxsum=self.use_maxsum,
                diversity=self.diversity,
                top_n=num_answers,
                nr_candidates=self.nr_candidates,
            )
            keywords.append([x[0] for x in extracted_kws])
        return keywords
