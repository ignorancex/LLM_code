from abc import ABC, abstractmethod


class BaseAMModel(ABC):
    @abstractmethod
    def evaluate_answers(
        self,
        predicted_answers: list[str],
        questions: list[str] | None = None,
        reference_answers: list[str] | None = None,
        contexts: list[str] | None = None,
    ) -> list[float]:
        raise NotImplementedError("Subclass must implement abstract method")
