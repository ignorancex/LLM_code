from tqdm import tqdm
from treqa.answer_extraction.ae_model import BaseAEModel

from .doc_scorer import DocScorer


def jaccard_similarity(hyp_list, ref_list):
    if len(hyp_list) == 0 or len(ref_list) == 0:
        return 0
    set1, set2 = set(hyp_list), set(ref_list)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if len(union) > 0 else 0


def compare_keyphrases(
    predicted_keyphrases: str, reference_keyphrases: str, comparator: str = "jaccard"
) -> float:
    """Evaluates a list of keyphrases extracted from hypothesis and reference using a specified comparator."""
    if comparator == "jaccard":
        return jaccard_similarity(predicted_keyphrases, reference_keyphrases)
    else:
        raise ValueError(f"Unknown comparator: {comparator}")


class KeyPhraseScorer(DocScorer):
    def __init__(
        self,
        ae_model: BaseAEModel,
        comparator: str = "jaccard",
        num_answers: int | None = None,
    ):
        self.ae_model = ae_model
        self.comparator = comparator
        self.num_answers = num_answers

    def get_scores(
        self,
        translations: list[str],
        sources: list[str] | None = None,
        references: list[str] | None = None,
    ):
        reference_keyphases = self.ae_model.extract_answers(
            references, self.num_answers
        )
        translation_keyphases = self.ae_model.extract_answers(
            translations, self.num_answers
        )

        scores = []
        for ref_kps, hyp_kps in tqdm(zip(reference_keyphases, translation_keyphases)):
            scores.append(compare_keyphrases(hyp_kps, ref_kps, self.comparator))
        return scores
