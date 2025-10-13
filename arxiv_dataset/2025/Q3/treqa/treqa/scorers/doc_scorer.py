from abc import ABC, abstractmethod


def flatten(xss):
    return [x for xs in xss for x in xs]


class DocScorer(ABC):
    @abstractmethod
    def get_scores(
        self,
        translations: list[str],
        sources: list[str] | None = None,
        references: list[str] | None = None,
    ):
        """
        Args:
            sources: a list of source sentences
            translations: a list of model generated translations, one per source
            references: a list of reference translations
        Returns: score
        """
        raise NotImplementedError
