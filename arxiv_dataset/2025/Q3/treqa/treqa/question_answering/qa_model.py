from abc import ABC, abstractmethod


class BaseQAModel(ABC):
    @abstractmethod
    def extract_answers(
        self, passages: list[str], questions: list[list[str]]
    ) -> list[list[str]]:
        raise NotImplementedError("Subclass must implement abstract method")
