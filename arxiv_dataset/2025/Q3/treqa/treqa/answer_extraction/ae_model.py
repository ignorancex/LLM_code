from abc import ABC, abstractmethod


class BaseAEModel(ABC):
    @abstractmethod
    def extract_answers(
        self,
        passages: list[str],
        num_answers: int | None = None,
    ) -> list[list[str]]:
        """
        Extracts answers from the given sources and targets.

        Args:
            texts (List[str]): List of texts to extract text from.
            num_answers (int | None): Number of answers to generate per target.

        Returns:
            List[List[str]]: List of lists of answer spans.
        """
        raise NotImplementedError("Subclass must implement abstract method")
