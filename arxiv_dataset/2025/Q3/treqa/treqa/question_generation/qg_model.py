from abc import ABC, abstractmethod


class BaseQAGModel(ABC):
    @abstractmethod
    def generate_qa_pairs(
        self,
        passages: list[str],
        num_questions: int | None = None,
        candidates: list[list[str]] | None = None,
        ref_passages: list[str] | None = None,
    ) -> list[list[tuple[str, str]]]:
        """
        Generate question-answer pairs for the given sources and targets.

        Args:
            passages (List[str]): List of texts.
            num_questions (int | None): Number of questions to generate per passage.
            candidates (List[List[str]] | None): List of alternative candidate translations for each passage.
            ref_passages (List[str]): List of alternative original texts in a different language.

        Returns:
            List[List[Tuple[str, str]]]: List of lists of (question, answer) tuples.
        """
        raise NotImplementedError("Subclass must implement abstract method")


class BaseQGModel(ABC):
    @abstractmethod
    def generate_questions(
        self,
        passages: list[str],
        answers: list[list[str]],
    ) -> list[list[tuple[str, str]]]:
        """
        Generate question-answer pairs for the given sources and targets.
        TODO: add candidate-based generation

        Args:
            passages (List[str]): List of texts.
            answers (List[List[str]]): Potential list of answers

        Returns:
            List[List[Tuple[str, str]]]: List of lists of (question, answer) tuples.
        """
        raise NotImplementedError("Subclass must implement abstract method")
